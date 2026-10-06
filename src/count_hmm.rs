//! A count HMM's negative log-likelihood and its gradient in one streamed pass (issue #1255).
//!
//! `EmissionHmmObjective`'s gradient for a count family was the compiled E step
//! (`ragged_posterior_probabilities`) and a torch backward pass through the emission
//! log-density: 66--91% of a call at 291,142 positions was the torch half. By Fisher's
//! identity the score is the posterior expectation of the complete-data score, and every
//! count family's per-state score depends on the observation only through its count (and,
//! with an exposure, the exposure), so the posterior is folded into a histogram over the
//! counts as the E step streams, and the emission gradient is a sum over the histogram's
//! occupied entries afterwards: no `(n, K)` posterior is stored and no backward pass runs.
//!
//! Each density and each score is `count_mixture`'s, the `lgamma` and `digamma` differences
//! as prefix sums over integers (issue #1136), so neither cancels as a shape grows:
//!
//! - Poisson: `y ln mu - mu - ln y!`; `d/dmu = y / mu - 1`;
//! - negative binomial at rate `m` (`m = e mu` with an exposure `e`, issue #631):
//!   `R_r[y] - ln y! - r ln(1 + m / r) + y (ln m - ln(1 + m / r))`;
//!   `d/dr = D_r[y] - ln(1 + m / r) + (m - y) / (r + m)`, `d/dmu = y / mu - e (y + r) / (r + m)`.
//!   A zero exposure marks the count unobserved: it scores 0 under every state and adds
//!   nothing to the gradient (issue #933);
//! - beta-binomial over `n` trials (a constant per state, a count per position, or the observed
//!   total of the joint pair): `count_mixture`'s, `d/da = D_a[z] - D_tau[n]`,
//!   `d/db = D_b[n - z] - D_tau[n]`. A trial count per position (issue #1265) is an integer,
//!   so it folds into histograms over `z`, `n - z` and `n` as the joint pair's observed total
//!   does; a zero trial count marks the successes unobserved (issue #933).
//!
//! The independent pair takes one covariate per channel (issue #1265): an exposure on the
//! total, accumulated per position since it is continuous, and a trial count on the successes,
//! folded into histograms. A zero exposure leaves the success channel scored.
//!
//! The streaming, the block cut and the merge order are `hmm_stream::stream_counts`', so the
//! result is the same at every thread count.

use rayon::prelude::*;

use crate::count_mixture::{Gradient, Natural, Rising, Slot};
use crate::energy::{pinned_simplex, Energy};
use crate::hmm_stream::{segment_lengths, stream_counts, Statistics};
use crate::special::ln_gamma;

/// A count HMM over segments end to end, in `theta = (K - 1 free initial, K (K - 1) free
/// transition, then each slot's K free values at its offset)`, as `opt.hmm.EmissionHmmObjective`
/// states it on a count start, behind [`Energy`].
///
/// A total channel (`totals`) is Poisson where `poisson` is set and a negative binomial
/// otherwise, with an optional exposure per observation; a success channel (`successes`) is a
/// beta-binomial over `trial_counts`, one per position, where given, over `trials`, one per
/// state, or over the totals where both are empty (the joint pair). A point at which a natural parameter is not finite and positive, or at
/// which an observation has zero density under every state, has value and gradient nan: a
/// chain rejects it, and the objective takes the E step and backward pass there.
pub struct CountHmm {
    k: usize,
    lengths: Vec<usize>,
    totals: Option<Vec<u32>>,
    successes: Option<Vec<u32>>,
    trials: Vec<f64>,
    trial_counts: Option<Vec<u32>>,
    joint: bool,
    exposure: Option<Vec<f64>>,
    log_exposure: Vec<f64>,
    poisson: bool,
    slots: Vec<(Slot, usize)>,
    largest_total: usize,
    largest_success: usize,
    largest_rest: usize,
    largest_trials: usize,
    log_factorial: Vec<f64>,
}

impl CountHmm {
    /// # Errors
    /// Fewer than two states, segments that do not cover the observations, channels of
    /// different lengths, slots that do not match the channels, a trial count that is not a
    /// non-negative integer, trial counts per position beside trials per state or without a
    /// success channel, an exposure without a negative-binomial total channel, beside the joint
    /// pair, or one that is negative or not finite, or `dimension` other than `K^2 - 1 + K` per
    /// slot.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        k: usize,
        lengths: Vec<usize>,
        totals: Option<Vec<u32>>,
        successes: Option<Vec<u32>>,
        trials: Vec<f64>,
        trial_counts: Option<Vec<u32>>,
        exposure: Option<Vec<f64>>,
        poisson: bool,
        slots: Vec<(Slot, usize)>,
        dimension: usize,
    ) -> Result<Self, String> {
        let has = |slot| slots.iter().any(|(s, _)| *s == slot);
        let total_slots = has(Slot::Mean) && (poisson != has(Slot::Dispersion));
        let success_slots =
            (has(Slot::Alpha) && has(Slot::Beta)) || (has(Slot::Rate) && has(Slot::Concentration));
        if k < 2
            || lengths.is_empty()
            || dimension != k * k - 1 + k * slots.len()
            || slots
                .iter()
                .any(|(_, offset)| *offset < k * k - 1 || offset + k > dimension)
            || totals.is_some() != total_slots
            || successes.is_some() != success_slots
        {
            return Err(format!(
                "a count HMM takes k >= 2 states on at least one segment, slots matching its \
                 channels and d = k^2 - 1 + k per slot, got k = {k}, {} slots and d = {dimension}",
                slots.len()
            ));
        }
        let n = totals
            .as_ref()
            .or(successes.as_ref())
            .map_or(0, std::vec::Vec::len);
        if successes.as_ref().is_some_and(|s| s.len() != n)
            || exposure.as_ref().is_some_and(|e| e.len() != n)
            || trial_counts.as_ref().is_some_and(|t| t.len() != n)
        {
            return Err("every channel and each covariate hold one value per position".to_string());
        }
        segment_lengths(n, &lengths)?;
        if trial_counts.is_some() && (successes.is_none() || !trials.is_empty()) {
            return Err(
                "trial counts per position replace trials per state on a success channel"
                    .to_string(),
            );
        }
        let joint = successes.is_some() && trials.is_empty() && trial_counts.is_none();
        // A zero trial count marks the successes unobserved: scored at zero successes, they
        // score log 1 under every state and add nothing to the gradient (issue #933).
        let successes = match (successes, &trial_counts) {
            (Some(mut z), Some(t)) => {
                for (z, &t) in z.iter_mut().zip(t) {
                    if t == 0 {
                        *z = 0;
                    }
                }
                Some(z)
            }
            (z, _) => z,
        };
        if successes.is_some() {
            if joint && totals.is_none() {
                return Err("the joint pair's trials are the totals, which are absent".to_string());
            }
            if !trials.is_empty()
                && (trials.len() != k
                    || trials
                        .iter()
                        .any(|&t| !(t >= 0.0 && t.fract() == 0.0 && t < 4.0e9)))
            {
                return Err("trials holds one non-negative integer per state".to_string());
            }
        }
        if let Some(e) = &exposure {
            if totals.is_none() || poisson || joint {
                return Err(
                    "an exposure scales a negative binomial's mean, outside the joint pair"
                        .to_string(),
                );
            }
            if e.iter().any(|&v| !(v >= 0.0 && v.is_finite())) {
                return Err("every exposure is finite and non-negative".to_string());
            }
        }
        let largest = |c: &Option<Vec<u32>>| {
            c.as_ref()
                .map_or(0, |c| c.iter().copied().max().unwrap_or(0) as usize)
        };
        let largest_total = largest(&totals);
        let largest_success = largest(&successes);
        let largest_trials = if let Some(t) = &trial_counts {
            t.iter().copied().max().unwrap_or(0) as usize
        } else if trials.is_empty() {
            largest_total
        } else {
            trials.iter().fold(0.0_f64, |a, &b| a.max(b)) as usize
        };
        // The largest `n - z` any table is read at: over the observations where `n` is one per
        // position (the joint pair's total or a trial count), `max n` otherwise, since `z` may
        // be 0 under any state.
        let rest_of = |n: &[u32], s: &[u32]| {
            n.iter()
                .zip(s)
                .map(|(&n, &z)| n.saturating_sub(z) as usize)
                .max()
                .unwrap_or(0)
        };
        let largest_rest = match (&totals, &successes, &trial_counts) {
            (_, Some(s), Some(t)) => rest_of(t, s),
            (Some(t), Some(s), None) if joint => rest_of(t, s),
            _ => largest_trials,
        };
        let extent = largest_total
            .max(largest_success)
            .max(largest_trials)
            .max(largest_rest);
        let log_factorial = (0..=extent).map(|m| ln_gamma(m as f64 + 1.0)).collect();
        let log_exposure = exposure
            .as_ref()
            .map_or_else(Vec::new, |e| e.iter().map(|v| v.ln()).collect());
        Ok(Self {
            k,
            lengths,
            totals,
            successes,
            trials,
            trial_counts,
            joint,
            exposure,
            log_exposure,
            poisson,
            slots,
            largest_total,
            largest_success,
            largest_rest,
            largest_trials,
            log_factorial,
        })
    }

    fn n_positions(&self) -> usize {
        self.totals
            .as_ref()
            .or(self.successes.as_ref())
            .map_or(0, Vec::len)
    }
}

/// Per-call tables of the total channel.
struct TotalTables<'a> {
    dispersion: &'a [f64],
    mean: &'a [f64],
    /// `R_r` and `D_r`, absent for the Poisson.
    rising: Option<Rising>,
    /// `ln(1 + mu / r)` per state, or per position and state under an exposure.
    log_share: Vec<f64>,
    /// `r ln(1 + mu / r)` per state; `mu` for the Poisson.
    pull: Vec<f64>,
    log_mean: Vec<f64>,
}

/// Per-call tables of the success channel.
struct SuccessTables {
    alpha: Rising,
    beta: Rising,
    total: Rising,
    log_rate: Vec<f64>,
    log_rest: Vec<f64>,
}

/// Each state's posterior weight on each count of each channel, and the per-position sums an
/// exposure's score needs.
struct Tallies<'a> {
    hmm: &'a CountHmm,
    tables: Option<&'a TotalTables<'a>>,
    /// `(largest_total + 1) * K`: the weight on each total.
    total: Vec<f64>,
    /// `(largest_success + 1) * K`: the weight on each success count.
    success: Vec<f64>,
    /// `(largest_rest + 1) * K`: the weight on each `n - z`, where `n` is one per position.
    rest: Vec<f64>,
    /// `(largest_trials + 1) * K`: the weight on each trial count, under trial counts per
    /// position alone (the joint pair's are `total`).
    depth: Vec<f64>,
    /// Under an exposure, per state: `sum gamma (-ln(1 + m / r) + (m - y) / (r + m))`.
    dispersion: Vec<f64>,
    /// Under an exposure, per state: `sum gamma (y / mu - e (y + r) / (r + m))`.
    mean: Vec<f64>,
}

impl Statistics for Tallies<'_> {
    #[inline]
    fn add(&mut self, index: usize, posterior: &[f64]) {
        let k = self.hmm.k;
        'total: {
            let Some(totals) = &self.hmm.totals else {
                break 'total;
            };
            let y = totals[index] as usize;
            if let (Some(exposure), Some(t)) = (&self.hmm.exposure, self.tables) {
                let e = exposure[index];
                if e == 0.0 {
                    break 'total;
                }
                let yf = y as f64;
                let share = &t.log_share[index * k..][..k];
                for (state, &w) in posterior.iter().enumerate() {
                    let (r, mu) = (t.dispersion[state], t.mean[state]);
                    let m = e * mu;
                    let reach = r + m;
                    self.dispersion[state] += w * (-share[state] + (m - yf) / reach);
                    self.mean[state] += w * (yf / mu - e * (yf + r) / reach);
                }
            }
            for (h, &w) in self.total[y * k..][..k].iter_mut().zip(posterior) {
                *h += w;
            }
            if let (Some(successes), true) = (&self.hmm.successes, self.hmm.joint) {
                let rest = y - successes[index].min(totals[index]) as usize;
                for (h, &w) in self.rest[rest * k..][..k].iter_mut().zip(posterior) {
                    *h += w;
                }
            }
        }
        if let Some(successes) = &self.hmm.successes {
            let z = successes[index] as usize;
            for (h, &w) in self.success[z * k..][..k].iter_mut().zip(posterior) {
                *h += w;
            }
            if let Some(trials) = &self.hmm.trial_counts {
                let n = trials[index] as usize;
                for (h, &w) in self.rest[n.saturating_sub(z) * k..][..k]
                    .iter_mut()
                    .zip(posterior)
                {
                    *h += w;
                }
                for (h, &w) in self.depth[n * k..][..k].iter_mut().zip(posterior) {
                    *h += w;
                }
            }
        }
    }

    fn merge(&mut self, other: Self) {
        for (mine, theirs) in [
            (&mut self.total, &other.total),
            (&mut self.success, &other.success),
            (&mut self.rest, &other.rest),
            (&mut self.depth, &other.depth),
            (&mut self.dispersion, &other.dispersion),
            (&mut self.mean, &other.mean),
        ] {
            for (a, b) in mine.iter_mut().zip(theirs) {
                *a += b;
            }
        }
    }
}

impl CountHmm {
    /// The `K` log-densities of position `index`.
    #[inline]
    fn log_density(
        &self,
        index: usize,
        totals: Option<&TotalTables<'_>>,
        successes: Option<&SuccessTables>,
        out: &mut [f64],
    ) {
        let k = self.k;
        out.fill(0.0);
        let mut n_observed = None;
        if let (Some(counts), Some(t)) = (&self.totals, totals) {
            let y = counts[index] as usize;
            let yf = y as f64;
            n_observed = Some(y);
            match (&self.exposure, &t.rising) {
                (_, None) => {
                    for (state, o) in out.iter_mut().enumerate() {
                        *o = yf * t.log_mean[state] - t.pull[state] - self.log_factorial[y];
                    }
                }
                (None, Some(rising)) => {
                    for (state, o) in out.iter_mut().enumerate() {
                        *o = rising.log[y * k + state] - self.log_factorial[y] - t.pull[state]
                            + yf * (t.log_mean[state] - t.log_share[state]);
                    }
                }
                (Some(exposure), Some(rising)) => {
                    // A zero exposure leaves the total unobserved: it scores 0 (issue #933).
                    if exposure[index] != 0.0 {
                        let share = &t.log_share[index * k..][..k];
                        let log_e = self.log_exposure[index];
                        for (state, o) in out.iter_mut().enumerate() {
                            let r = t.dispersion[state];
                            *o = rising.log[y * k + state]
                                - self.log_factorial[y]
                                - r * share[state]
                                + yf * (log_e + t.log_mean[state] - share[state]);
                        }
                    }
                }
            }
        }
        if let (Some(counts), Some(s)) = (&self.successes, successes) {
            let z = counts[index] as usize;
            let per_position = match &self.trial_counts {
                Some(t) => Some(t[index] as usize),
                None if self.joint => Some(n_observed.unwrap_or(0)),
                None => None,
            };
            for (state, o) in out.iter_mut().enumerate() {
                let n_trials = per_position.unwrap_or_else(|| self.trials[state] as usize);
                if z > n_trials {
                    *o = f64::NEG_INFINITY;
                    continue;
                }
                let rest = n_trials - z;
                *o += (self.log_factorial[n_trials]
                    - self.log_factorial[z]
                    - self.log_factorial[rest])
                    + z as f64 * s.log_rate[state]
                    + rest as f64 * s.log_rest[state]
                    + s.alpha.log[z * k + state]
                    + s.beta.log[rest * k + state]
                    - s.total.log[n_trials * k + state];
            }
        }
    }

    /// What position `index`'s density depends on, or `usize::MAX` where a covariate makes
    /// every position its own: a run of one count is scored once.
    #[inline]
    fn key(&self, index: usize) -> usize {
        if self.exposure.is_some() || self.trial_counts.is_some() {
            return usize::MAX;
        }
        match (&self.totals, &self.successes) {
            (Some(t), Some(s)) => {
                t[index] as usize * (self.largest_success + 1) + s[index] as usize
            }
            (Some(t), None) => t[index] as usize,
            (None, Some(s)) => s[index] as usize,
            (None, None) => usize::MAX,
        }
    }

    /// The emission gradient in the natural parameters, from the tallies.
    fn emission_gradient(
        &self,
        tallies: &Tallies<'_>,
        totals: Option<&TotalTables<'_>>,
        successes: Option<&SuccessTables>,
    ) -> Gradient {
        let k = self.k;
        let block = |on: bool| if on { vec![0.0; k] } else { Vec::new() };
        let mut gradient = Gradient {
            log_weight: Vec::new(),
            dispersion: block(totals.is_some()),
            mean: block(totals.is_some()),
            alpha: block(successes.is_some()),
            beta: block(successes.is_some()),
        };
        if let Some(t) = totals {
            if self.exposure.is_some() {
                gradient.dispersion.copy_from_slice(&tallies.dispersion);
                gradient.mean.copy_from_slice(&tallies.mean);
            }
            for y in 0..=self.largest_total {
                let yf = y as f64;
                for state in 0..k {
                    let w = tallies.total[y * k + state];
                    if w == 0.0 {
                        continue;
                    }
                    let mu = t.mean[state];
                    match (&t.rising, self.exposure.is_some()) {
                        (None, _) => gradient.mean[state] += w * (yf / mu - 1.0),
                        (Some(rising), true) => {
                            gradient.dispersion[state] += w * rising.reciprocal[y * k + state];
                        }
                        (Some(rising), false) => {
                            let r = t.dispersion[state];
                            gradient.dispersion[state] += w
                                * (rising.reciprocal[y * k + state] - t.log_share[state]
                                    + (mu - yf) / (r + mu));
                            gradient.mean[state] += w * (yf / mu - (yf + r) / (r + mu));
                        }
                    }
                }
            }
        }
        if let Some(s) = successes {
            if self.joint || self.trial_counts.is_some() {
                // `n` is one per position, the joint pair's total or a trial count, so each
                // term is a sum over its own count.
                let (depth, largest_depth) = if self.joint {
                    (&tallies.total, self.largest_total)
                } else {
                    (&tallies.depth, self.largest_trials)
                };
                for z in 0..=self.largest_success {
                    for state in 0..k {
                        let w = tallies.success[z * k + state];
                        if w != 0.0 {
                            gradient.alpha[state] += w * s.alpha.reciprocal[z * k + state];
                        }
                    }
                }
                for q in 0..=self.largest_rest {
                    for state in 0..k {
                        let w = tallies.rest[q * k + state];
                        if w != 0.0 {
                            gradient.beta[state] += w * s.beta.reciprocal[q * k + state];
                        }
                    }
                }
                for n in 0..=largest_depth {
                    for state in 0..k {
                        let w = depth[n * k + state];
                        if w != 0.0 {
                            let shared = w * s.total.reciprocal[n * k + state];
                            gradient.alpha[state] -= shared;
                            gradient.beta[state] -= shared;
                        }
                    }
                }
            } else {
                for z in 0..=self.largest_success {
                    for state in 0..k {
                        let w = tallies.success[z * k + state];
                        let n_trials = self.trials[state] as usize;
                        if w == 0.0 || z > n_trials {
                            continue;
                        }
                        let shared = s.total.reciprocal[n_trials * k + state];
                        gradient.alpha[state] += w * (s.alpha.reciprocal[z * k + state] - shared);
                        gradient.beta[state] +=
                            w * (s.beta.reciprocal[(n_trials - z) * k + state] - shared);
                    }
                }
            }
        }
        gradient
    }
}

impl Energy for CountHmm {
    /// The negative log-likelihood and the negative score: the chain's terms as
    /// `hmm_stream::GaussianHmm` forms them, the emission's from the tallies.
    fn value_and_gradient(&self, theta: &[f64], out: &mut [f64]) -> f64 {
        let k = self.k;
        let log_initial = pinned_simplex(&theta[..k - 1]);
        let log_transition: Vec<f64> = theta[k - 1..k * k - 1]
            .chunks_exact(k - 1)
            .flat_map(pinned_simplex)
            .collect();
        let natural = Natural::at(theta, &self.slots, k);
        let fail = |out: &mut [f64]| {
            out.iter_mut().for_each(|o| *o = f64::NAN);
            f64::NAN
        };
        if !natural.admissible() {
            return fail(out);
        }
        let n = self.n_positions();
        let empty: &[f64] = &[];
        let total_tables = self.totals.as_ref().map(|_| {
            let mean = natural.mean.as_deref().unwrap_or(empty);
            let dispersion = natural.dispersion.as_deref().unwrap_or(empty);
            if self.poisson {
                return TotalTables {
                    dispersion,
                    mean,
                    rising: None,
                    log_share: Vec::new(),
                    pull: mean.to_vec(),
                    log_mean: mean.iter().map(|mu| mu.ln()).collect(),
                };
            }
            let log_share = match &self.exposure {
                None => dispersion
                    .iter()
                    .zip(mean)
                    .map(|(&r, &mu)| (mu / r).ln_1p())
                    .collect(),
                Some(exposure) => {
                    let mut share = vec![0.0; n * k];
                    share
                        .par_chunks_mut(k * 1024)
                        .enumerate()
                        .for_each(|(chunk, rows)| {
                            for (i, row) in rows.chunks_exact_mut(k).enumerate() {
                                let e = exposure[chunk * 1024 + i];
                                for (state, value) in row.iter_mut().enumerate() {
                                    *value = (e * mean[state] / dispersion[state]).ln_1p();
                                }
                            }
                        });
                    share
                }
            };
            TotalTables {
                dispersion,
                mean,
                rising: Some(Rising::new(dispersion, self.largest_total)),
                log_share,
                pull: dispersion
                    .iter()
                    .zip(mean)
                    .map(|(&r, &mu)| r * (mu / r).ln_1p())
                    .collect(),
                log_mean: mean.iter().map(|mu| mu.ln()).collect(),
            }
        });
        let success_tables = self.successes.as_ref().map(|_| {
            let alpha = natural.alpha.as_deref().unwrap_or(empty);
            let beta = natural.beta.as_deref().unwrap_or(empty);
            let tau: Vec<f64> = alpha.iter().zip(beta).map(|(a, b)| a + b).collect();
            let ((alpha_rising, beta_rising), total) = rayon::join(
                || {
                    rayon::join(
                        || Rising::new(alpha, self.largest_success),
                        || Rising::new(beta, self.largest_rest),
                    )
                },
                || Rising::new(&tau, self.largest_trials),
            );
            SuccessTables {
                alpha: alpha_rising,
                beta: beta_rising,
                total,
                log_rate: alpha.iter().zip(&tau).map(|(a, t)| (a / t).ln()).collect(),
                log_rest: beta.iter().zip(&tau).map(|(b, t)| (b / t).ln()).collect(),
            }
        });
        let (totals, successes) = (total_tables.as_ref(), success_tables.as_ref());
        let per_position = self.trial_counts.is_some();
        let tally_size = |on: bool, extent: usize| if on { (extent + 1) * k } else { 0 };
        let sizes = (
            tally_size(self.totals.is_some(), self.largest_total),
            tally_size(self.successes.is_some(), self.largest_success),
            tally_size(self.joint || per_position, self.largest_rest),
            if self.exposure.is_some() { k } else { 0 },
            tally_size(per_position, self.largest_trials),
        );
        let streamed = stream_counts(
            n,
            &self.lengths,
            &log_initial,
            &log_transition,
            |index, row| self.log_density(index, totals, successes, row),
            |index| self.key(index),
            || Tallies {
                hmm: self,
                tables: totals,
                total: vec![0.0; sizes.0],
                success: vec![0.0; sizes.1],
                rest: vec![0.0; sizes.2],
                depth: vec![0.0; sizes.4],
                dispersion: vec![0.0; sizes.3],
                mean: vec![0.0; sizes.3],
            },
            8 * (sizes.0 + sizes.1 + sizes.2 + 2 * sizes.3 + sizes.4),
        );
        let Ok((counts, tallies)) = streamed else {
            return fail(out);
        };
        if !counts.log_likelihood.is_finite() {
            return fail(out);
        }
        let n_sequences = self.lengths.len() as f64;
        for (slot, (first, log_p)) in out
            .iter_mut()
            .zip(counts.first.iter().zip(&log_initial).skip(1))
        {
            *slot = -(first - n_sequences * log_p.exp());
        }
        let mut index = k - 1;
        for i in 0..k {
            let row: f64 = counts.pairs[i * k..(i + 1) * k].iter().sum();
            for j in 1..k {
                out[index] = -(counts.pairs[i * k + j] - row * log_transition[i * k + j].exp());
                index += 1;
            }
        }
        let gradient = self.emission_gradient(&tallies, totals, successes);
        natural.pull_back(&gradient, &self.slots, k, out);
        -counts.log_likelihood
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A two-state Poisson HMM on one segment, against the forward recursion and central
    /// differences written out here.
    #[test]
    fn a_poisson_hmm_is_its_forward_recursion_and_its_slope() {
        let counts = vec![0_u32, 3, 7, 2, 9, 1];
        let hmm = CountHmm::new(
            2,
            vec![6],
            Some(counts.clone()),
            None,
            Vec::new(),
            None,
            None,
            true,
            vec![(Slot::Mean, 3)],
            5,
        )
        .unwrap();
        let reference = |theta: &[f64]| -> f64 {
            let initial = pinned_simplex(&theta[..1]);
            let rows: Vec<Vec<f64>> = theta[1..3].iter().map(|v| pinned_simplex(&[*v])).collect();
            let rate = [theta[3].exp(), theta[4].exp()];
            let density = |y: u32, s: usize| {
                let yf = f64::from(y);
                (yf * rate[s].ln() - rate[s] - ln_gamma(yf + 1.0)).exp()
            };
            let mut alpha: Vec<f64> = (0..2)
                .map(|s| initial[s].exp() * density(counts[0], s))
                .collect();
            for &y in &counts[1..] {
                alpha = (0..2)
                    .map(|j| {
                        (0..2).map(|i| alpha[i] * rows[i][j].exp()).sum::<f64>() * density(y, j)
                    })
                    .collect();
            }
            -alpha.iter().sum::<f64>().ln()
        };
        let theta = [0.3, -0.4, 0.2, 0.5, 1.7];
        let mut gradient = [0.0; 5];
        let value = hmm.value_and_gradient(&theta, &mut gradient);
        assert!((value - reference(&theta)).abs() < 1e-12 * value.abs());
        for i in 0..5 {
            let (mut up, mut down) = (theta, theta);
            up[i] += 1e-6;
            down[i] -= 1e-6;
            let slope = (reference(&up) - reference(&down)) / 2e-6;
            assert!((gradient[i] - slope).abs() < 1e-6, "coordinate {i}");
        }
    }

    /// A trial count per position that is one constant is the trial count per state at that
    /// constant (issue #1265): the same value, and the gradient to rounding, since the
    /// per-position counts fold into a histogram over `n` and sum in another order.
    #[test]
    fn a_constant_count_trial_per_position_is_the_trial_per_state() {
        let successes = vec![0_u32, 3, 7, 2, 9, 10];
        let slots = vec![(Slot::Alpha, 3), (Slot::Beta, 5)];
        let made = |trials: Vec<f64>, trial_counts: Option<Vec<u32>>| {
            CountHmm::new(
                2,
                vec![2, 4],
                None,
                Some(successes.clone()),
                trials,
                trial_counts,
                None,
                false,
                slots.clone(),
                7,
            )
            .unwrap()
        };
        let per_state = made(vec![10.0, 10.0], None);
        let per_position = made(Vec::new(), Some(vec![10; 6]));
        let theta = [0.3, -0.4, 0.2, 0.5, 1.7, 0.9, -0.2];
        let (mut want, mut got) = ([0.0; 7], [0.0; 7]);
        let want_value = per_state.value_and_gradient(&theta, &mut want);
        let value = per_position.value_and_gradient(&theta, &mut got);
        assert_eq!(value, want_value);
        let scale = want.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() <= 1e-13 * scale);
        }
        let refused = CountHmm::new(
            2,
            vec![6],
            None,
            Some(successes.clone()),
            vec![10.0, 10.0],
            Some(vec![10; 6]),
            None,
            false,
            slots,
            7,
        );
        assert!(refused.is_err());
    }

    #[test]
    fn slots_that_do_not_match_the_channels_are_refused() {
        let made = CountHmm::new(
            2,
            vec![2],
            Some(vec![1, 2]),
            None,
            Vec::new(),
            None,
            None,
            false,
            vec![(Slot::Mean, 3)],
            5,
        );
        assert!(made.is_err());
    }
}
