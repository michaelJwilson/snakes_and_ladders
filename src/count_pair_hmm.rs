//! The unphased count-pair HMM as one stateful object: data, emission, buffers and fitted
//! parameters held across calls, so a Baum-Welch fit crosses the FFI boundary once (issue #1412).
//!
//! **The model** is `sal.emissions.CountPairEmission`'s independent form under per-position
//! covariates: a negative-binomial total `y_t ~ NB(r_k, e_t mu_k)` over an exposure `e_t`, and
//! beta-binomial successes `z_t ~ BB(n_t, a_k, b_k)` over a trial count `n_t`, on `K <= 255`
//! hidden states of a first-order chain over ragged segments. A zero exposure marks the total
//! unobserved and a zero trial count the successes, as the family marks them (issue #933).
//!
//! **One iteration** is `sal.opt.hmm.baum_welch_family`'s: score every position under every
//! state, the ragged E step ([`crate::ragged::ragged_posteriors_into`]), then the M step --- the
//! initial distribution and the transition from the expected counts, each selectable, and the
//! emission by one of two solvers:
//!
//! - [`Solver::Newton`], per block as the family's own `reestimate`: each mean in closed form,
//!   each dispersion by [`count_mstep::solve_exposed_dispersion`] warm-started from the last
//!   iteration (issue #1410), each `(a, b)` by [`count_mstep::solve_beta_binomial`]. Under
//!   `tied` one dispersion solves the score summed over the states by the same safeguarded
//!   Newton, and one concentration by [`count_mstep::solve_beta_binomial_tied`]: what
//!   `sal.qa.hmm_fit_semantics.tied_m_step` runs in Python, 98.8% of its iteration.
//! - [`Solver::Lbfgs`], a joint quasi-Newton over every emission parameter in
//!   `(log mu, logit p, log r, log tau)` on the closed-form gradient of the expected
//!   complete-data log-likelihood (the partials of issue #1353): the joint optimum a caller's
//!   BFGS M step targets. Under a varying exposure the closed-form mean is not the joint
//!   maximum, so the two solvers' optima differ there.
//!
//! **Sums over positions fold into tails.** The `lgamma` differences the expected log-likelihood
//! carries are `sum_t w_t (lgamma(u_t + x) - lgamma(x)) = sum_j T_j ln(x + j)` with
//! `T_j = sum_{u_t > j} w_t`, and their derivatives `sum_j T_j / (x + j)`, so one M step
//! builds each state's tails once and every solver step reads them (`count_mstep`'s identity).
//!
//! The implementations are plain Rust with no PyO3 types except the binding at the end, so
//! `cargo test` and `benches/` link them.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::count_mstep::{
    self, solve_beta_binomial, solve_beta_binomial_tied, solve_bracketed_newton,
    solve_exposed_dispersion, BetaBinomialProblem, Bisection, Exposed,
};
use crate::ragged::{log_sum, ragged_posteriors_into, SwitchKind};
use crate::ragged_viterbi::ragged_viterbi_into;
use crate::special::ln_gamma;

/// The posterior mass below which a state has no data to re-estimate from, and keeps its
/// parameters: `sal.emissions.base.COLLAPSED_MASS`.
pub const COLLAPSED_MASS: f64 = 1e-8;
/// The least dispersion an M step returns: `sal.emissions.mstep.DISPERSION_FLOOR`.
pub const DISPERSION_FLOOR: f64 = f64::EPSILON;
/// The dispersion bracket's lower end under its upper: `_DISPERSION_BRACKET_RATIO`.
const DISPERSION_BRACKET_RATIO: f64 = 1e-9;
/// The concentration bracket's lower end under its bound: `_CONCENTRATION_BRACKET_RATIO`.
const CONCENTRATION_BRACKET_RATIO: f64 = 1e-9;
/// The dispersion solve's tolerance in `log r`, as `solve_dispersion`'s default.
const DISPERSION_TOLERANCE: f64 = 1e-12;
/// The beta-binomial solve's settings, as `solve_beta_binomial_m_step` passes them.
const BETA_BINOMIAL: Bisection = Bisection {
    tolerance: 1e-10,
    max_iterations: 60,
    max_bisections: 60,
    margin: 1e-12,
};
/// The most states a model takes: a state index fits a `u8`.
pub const MAX_STATES: usize = 255;

/// The emission M step's solver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Solver {
    /// Closed-form means, then a Newton or bisection solve per block, as the family's own.
    Newton,
    /// One L-BFGS over every emission parameter on the closed-form gradient.
    Lbfgs,
}

impl Solver {
    /// # Errors
    /// A name other than `newton` or `lbfgs`.
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "newton" => Ok(Self::Newton),
            "lbfgs" => Ok(Self::Lbfgs),
            other => Err(format!("solver is 'newton' or 'lbfgs', got {other:?}")),
        }
    }
}

/// How a fit left its loop: `sal.opt.termination.Stop`'s values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stop {
    /// The relative log-likelihood change, or the parameter change, met its tolerance.
    Converged,
    /// The iteration cap ran out.
    Budget,
    /// An emission M step did not settle; the fit ends on the parameters before it.
    Degenerate,
}

impl Stop {
    /// `sal.opt.termination.Stop`'s value for this branch.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Converged => "converged",
            Self::Budget => "budget",
            Self::Degenerate => "degenerate",
        }
    }
}

/// The observations, held for the model's lifetime.
#[derive(Clone, Debug)]
pub struct Data {
    /// Totals `y_t`.
    pub counts: Vec<u32>,
    /// Exposures `e_t`, zero where the total is unobserved.
    pub exposures: Vec<f64>,
    /// Successes `z_t`.
    pub successes: Vec<u32>,
    /// Trial counts `n_t`, zero where the successes are unobserved.
    pub trials: Vec<u32>,
    /// Segment lengths, summing to the positions.
    pub lengths: Vec<usize>,
    /// `ln y!` per position.
    log_factorial_count: Vec<f64>,
    /// `ln C(n, z)` per position.
    log_choose: Vec<f64>,
    /// `ln e_t`, zero where unobserved.
    log_exposure: Vec<f64>,
    /// Whether the observed exposures differ, which keeps the term profiling drops.
    varying: bool,
    /// Whether every observed exposure is the same value, the oracle's constant test.
    constant_trials: Option<f64>,
}

impl Data {
    /// # Errors
    /// Channels of different lengths, segments that do not cover them, a negative or
    /// non-finite exposure, or successes above their trials.
    pub fn new(
        counts: Vec<u32>,
        exposures: Vec<f64>,
        successes: Vec<u32>,
        trials: Vec<u32>,
        lengths: Vec<usize>,
    ) -> Result<Self, String> {
        let n = counts.len();
        if exposures.len() != n || successes.len() != n || trials.len() != n {
            return Err(
                "counts, exposures, successes and trials hold one value per position".into(),
            );
        }
        if lengths.is_empty() || lengths.contains(&0) || lengths.iter().sum::<usize>() != n {
            return Err(format!(
                "lengths must be positive and sum to the {n} positions"
            ));
        }
        if exposures.iter().any(|&e| !(e.is_finite() && e >= 0.0)) {
            return Err("every exposure must be finite and non-negative".into());
        }
        if successes.iter().zip(&trials).any(|(z, t)| z > t) {
            return Err("a success count exceeds its trial count".into());
        }
        let log_factorial_count = counts
            .iter()
            .map(|&y| ln_gamma(f64::from(y) + 1.0))
            .collect();
        let log_choose = successes
            .iter()
            .zip(&trials)
            .map(|(&z, &t)| {
                ln_gamma(f64::from(t) + 1.0)
                    - ln_gamma(f64::from(z) + 1.0)
                    - ln_gamma(f64::from(t - z) + 1.0)
            })
            .collect();
        let log_exposure = exposures
            .iter()
            .map(|&e| if e > 0.0 { e.ln() } else { 0.0 })
            .collect();
        let observed: Vec<f64> = exposures.iter().copied().filter(|&e| e > 0.0).collect();
        let varying = observed.iter().any(|&e| e != observed[0]);
        let trial_observed: Vec<u32> = trials.iter().copied().filter(|&t| t > 0).collect();
        let constant_trials = match trial_observed.first() {
            Some(&first) if trial_observed.iter().all(|&t| t == first) => Some(f64::from(first)),
            _ => None,
        };
        Ok(Self {
            counts,
            exposures,
            successes,
            trials,
            lengths,
            log_factorial_count,
            log_choose,
            log_exposure,
            varying,
            constant_trials,
        })
    }

    /// Positions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.counts.len()
    }

    /// Whether there are no positions; never, once built.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }
}

/// The parameters a fit moves.
#[derive(Clone, Debug, PartialEq)]
pub struct Params {
    /// NB dispersion `r` per state.
    pub dispersion: Vec<f64>,
    /// NB mean per unit exposure `mu` per state.
    pub mean: Vec<f64>,
    /// BB `a` per state.
    pub alpha: Vec<f64>,
    /// BB `b` per state.
    pub beta: Vec<f64>,
    /// `(K,)` log initial distribution.
    pub log_initial: Vec<f64>,
    /// `(K, K)` row-major log transition.
    pub log_transition: Vec<f64>,
}

impl Params {
    /// States.
    #[must_use]
    pub fn n_states(&self) -> usize {
        self.mean.len()
    }

    /// The largest relative change of any emission parameter from `other`.
    #[must_use]
    pub fn relative_change(&self, other: &Self) -> f64 {
        [
            (&self.dispersion, &other.dispersion),
            (&self.mean, &other.mean),
            (&self.alpha, &other.alpha),
            (&self.beta, &other.beta),
        ]
        .iter()
        .flat_map(|(a, b)| a.iter().zip(b.iter()))
        .map(|(a, b)| (a - b).abs() / a.abs().max(f64::MIN_POSITIVE))
        .fold(0.0, f64::max)
    }
}

/// What a model may fit, and how.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// One dispersion and one concentration across states.
    pub tied: bool,
    /// Whether the M step re-estimates the initial distribution.
    pub fit_initial: bool,
    /// Whether the M step re-estimates the transition.
    pub fit_transition: bool,
}

/// What one emission M step reported.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Report {
    /// Whether every block's solve settled.
    pub converged: bool,
    /// Whether any parameter stopped at the edge of what the data identify.
    pub at_boundary: bool,
    /// The most inner iterations any block took.
    pub iterations: u32,
    /// The largest residual any block reported.
    pub residual: f64,
    /// States held at their parameters for want of data.
    pub frozen: Vec<usize>,
    /// States whose dispersion reached the floor.
    pub degenerate: Vec<usize>,
}

/// One fit's outcome; the parameters are the model's.
#[derive(Clone, Debug, PartialEq)]
pub struct Fitted {
    /// The log-likelihood of the last completed E step.
    pub log_likelihood: f64,
    /// EM iterations run.
    pub iterations: u32,
    /// How the loop ended.
    pub stop: Stop,
    /// Whether any M step stopped a parameter at its identifiable bound.
    pub at_boundary: bool,
    /// States any M step held, ascending.
    pub frozen: Vec<usize>,
    /// The M step that ended a degenerate fit.
    pub unsettled: Option<Report>,
}

/// Each state's tails and sums, built once per M step from the posterior.
struct Moments {
    k: usize,
    /// `(K, max y)` tails of the observed totals' weights.
    count_tails: Vec<f64>,
    count_width: usize,
    /// `(K, max z)`, `(K, max n - z)`, `(K, max n)` tails of the observed successes' weights.
    success_tails: Vec<f64>,
    success_width: usize,
    failure_tails: Vec<f64>,
    failure_width: usize,
    depth_tails: Vec<f64>,
    depth_width: usize,
    /// Per state: weight on observed totals, `sum w y`, `sum w e`.
    total_weight: Vec<f64>,
    count_sum: Vec<f64>,
    exposure_sum: Vec<f64>,
    /// Per state: weight on observed successes, and `sum w n`.
    success_weight: Vec<f64>,
    depth_sum: Vec<f64>,
    /// `(K, n_observed_totals)` weight columns and the positions they index.
    columns: Vec<f64>,
    observed: Vec<usize>,
    counts: Vec<f64>,
    exposures: Vec<f64>,
}

/// `T[k, j] = sum_{u_t > j} w_tk` for each state, row-major `(K, max u)`.
fn tails_of(
    k: usize,
    weights: &[f64],
    values: impl Iterator<Item = (usize, usize)>,
    width: usize,
) -> Vec<f64> {
    let mut dense = vec![0.0; k * (width + 1)];
    for (t, u) in values {
        let row = &weights[t * k..(t + 1) * k];
        for (state, &w) in row.iter().enumerate() {
            dense[state * (width + 1) + u] += w;
        }
    }
    let mut tails = vec![0.0; k * width];
    for state in 0..k {
        let source = &dense[state * (width + 1)..(state + 1) * (width + 1)];
        let mut running = 0.0;
        for j in (0..width).rev() {
            running += source[j + 1];
            tails[state * width + j] = running;
        }
    }
    tails
}

impl Moments {
    /// `weights` are `(n, K)` probabilities.
    fn new(data: &Data, weights: &[f64], k: usize) -> Self {
        let observed: Vec<usize> = (0..data.len())
            .filter(|&t| data.exposures[t] > 0.0)
            .collect();
        let scored: Vec<usize> = (0..data.len()).filter(|&t| data.trials[t] > 0).collect();
        let count_width = observed
            .iter()
            .map(|&t| data.counts[t] as usize)
            .max()
            .unwrap_or(0);
        let success_width = scored
            .iter()
            .map(|&t| data.successes[t] as usize)
            .max()
            .unwrap_or(0);
        let failure_width = scored
            .iter()
            .map(|&t| (data.trials[t] - data.successes[t]) as usize)
            .max()
            .unwrap_or(0);
        let depth_width = scored
            .iter()
            .map(|&t| data.trials[t] as usize)
            .max()
            .unwrap_or(0);
        let count_tails = tails_of(
            k,
            weights,
            observed.iter().map(|&t| (t, data.counts[t] as usize)),
            count_width,
        );
        let success_tails = tails_of(
            k,
            weights,
            scored.iter().map(|&t| (t, data.successes[t] as usize)),
            success_width,
        );
        let failure_tails = tails_of(
            k,
            weights,
            scored
                .iter()
                .map(|&t| (t, (data.trials[t] - data.successes[t]) as usize)),
            failure_width,
        );
        let depth_tails = tails_of(
            k,
            weights,
            scored.iter().map(|&t| (t, data.trials[t] as usize)),
            depth_width,
        );
        let mut total_weight = vec![0.0; k];
        let mut count_sum = vec![0.0; k];
        let mut exposure_sum = vec![0.0; k];
        let mut columns = vec![0.0; k * observed.len()];
        for (i, &t) in observed.iter().enumerate() {
            let (y, e) = (f64::from(data.counts[t]), data.exposures[t]);
            for state in 0..k {
                let w = weights[t * k + state];
                total_weight[state] += w;
                count_sum[state] += w * y;
                exposure_sum[state] += w * e;
                columns[state * observed.len() + i] = w;
            }
        }
        let mut success_weight = vec![0.0; k];
        let mut depth_sum = vec![0.0; k];
        for &t in &scored {
            let n = f64::from(data.trials[t]);
            for state in 0..k {
                let w = weights[t * k + state];
                success_weight[state] += w;
                depth_sum[state] += w * n;
            }
        }
        let counts = observed
            .iter()
            .map(|&t| f64::from(data.counts[t]))
            .collect();
        let exposures = observed.iter().map(|&t| data.exposures[t]).collect();
        Self {
            k,
            count_tails,
            count_width,
            success_tails,
            success_width,
            failure_tails,
            failure_width,
            depth_tails,
            depth_width,
            total_weight,
            count_sum,
            exposure_sum,
            success_weight,
            depth_sum,
            columns,
            observed,
            counts,
            exposures,
        }
    }

    fn count_tail(&self, state: usize) -> &[f64] {
        &self.count_tails[state * self.count_width..(state + 1) * self.count_width]
    }

    fn column(&self, state: usize) -> &[f64] {
        let n = self.observed.len();
        &self.columns[state * n..(state + 1) * n]
    }

    fn beta_binomial(&self, state: usize) -> (&[f64], &[f64], &[f64]) {
        (
            &self.success_tails[state * self.success_width..(state + 1) * self.success_width],
            &self.failure_tails[state * self.failure_width..(state + 1) * self.failure_width],
            &self.depth_tails[state * self.depth_width..(state + 1) * self.depth_width],
        )
    }

    /// The trial count the concentration bound is read at: the constant one, or the mean
    /// under `weight` (`effective_trials`).
    fn effective_trials(&self, data: &Data, depth: f64, weight: f64) -> f64 {
        data.constant_trials.unwrap_or(depth / weight)
    }
}

/// `r ln(1 + m / r)` and `ln(1 + m / r)` without the `0 * inf` at large `r`.
#[inline]
fn log_share(r: f64, rate: f64) -> f64 {
    (rate / r).ln_1p()
}

/// Every position's log-density under every state, `(n, K)` row-major, into `out`.
///
/// The total: `lgamma(y + r) - lgamma(r) - y ln r - ln y! - (r + y) ln(1 + m / r) + y ln m` at
/// `m = e mu`, zero where the exposure is. The successes: `ln C(n, z) + lgamma(z + a) -
/// lgamma(a) + lgamma(n - z + b) - lgamma(b) - lgamma(n + tau) + lgamma(tau)`, zero where `n`
/// is. Each `lgamma` difference is taken directly: at the fixtures' `r <= 1e6` its rounding is
/// below 3e-9 absolute a term against a log-likelihood of 1e4 to 1e5.
pub fn log_density_into(data: &Data, params: &Params, out: &mut [f64]) {
    let k = params.n_states();
    let ln_gamma_r: Vec<f64> = params.dispersion.iter().map(|&r| ln_gamma(r)).collect();
    let ln_r: Vec<f64> = params.dispersion.iter().map(|r| r.ln()).collect();
    let ln_mean: Vec<f64> = params.mean.iter().map(|m| m.ln()).collect();
    let ln_gamma_a: Vec<f64> = params.alpha.iter().map(|&a| ln_gamma(a)).collect();
    let ln_gamma_b: Vec<f64> = params.beta.iter().map(|&b| ln_gamma(b)).collect();
    let tau: Vec<f64> = params
        .alpha
        .iter()
        .zip(&params.beta)
        .map(|(a, b)| a + b)
        .collect();
    let ln_gamma_tau: Vec<f64> = tau.iter().map(|&t| ln_gamma(t)).collect();
    let tied_r = params.dispersion.iter().all(|&r| r == params.dispersion[0]);
    out.par_chunks_mut(k * 256)
        .enumerate()
        .for_each(|(tile, rows)| {
            for (i, row) in rows.chunks_mut(k).enumerate() {
                let t = tile * 256 + i;
                let (y, e) = (f64::from(data.counts[t]), data.exposures[t]);
                if e > 0.0 {
                    let shared = if tied_r {
                        ln_gamma(y + params.dispersion[0]) - ln_gamma_r[0]
                    } else {
                        0.0
                    };
                    for state in 0..k {
                        let r = params.dispersion[state];
                        let rate = e * params.mean[state];
                        let rise = if tied_r {
                            shared
                        } else {
                            ln_gamma(y + r) - ln_gamma_r[state]
                        };
                        row[state] = rise
                            - y * ln_r[state]
                            - data.log_factorial_count[t]
                            - (r + y) * log_share(r, rate)
                            + y * (data.log_exposure[t] + ln_mean[state]);
                    }
                } else {
                    row.fill(0.0);
                }
                let n = data.trials[t];
                if n > 0 {
                    let z = f64::from(data.successes[t]);
                    let rest = f64::from(n) - z;
                    for state in 0..k {
                        let (a, b) = (params.alpha[state], params.beta[state]);
                        row[state] += data.log_choose[t] + ln_gamma(z + a) - ln_gamma_a[state]
                            + ln_gamma(rest + b)
                            - ln_gamma_b[state]
                            - ln_gamma(f64::from(n) + tau[state])
                            + ln_gamma_tau[state];
                    }
                }
            }
        });
}

/// The negative expected complete-data log-likelihood of the emission at `theta`, per unit
/// weight, and its gradient in `theta = (log mu, logit p, log r, log tau)` (the last two one
/// entry each under `tied`), less the terms constant in the parameters.
fn emission_objective(moments: &Moments, theta: &[f64], tied: bool, gradient: &mut [f64]) -> f64 {
    let k = moments.k;
    let shared = if tied { 1 } else { k };
    let index_r = |state: usize| 2 * k + if tied { 0 } else { state };
    let index_tau = |state: usize| 2 * k + shared + if tied { 0 } else { state };
    gradient.fill(0.0);
    let scale: f64 = moments
        .total_weight
        .iter()
        .chain(&moments.success_weight)
        .sum::<f64>()
        .max(f64::MIN_POSITIVE);
    let mut value = 0.0;
    for state in 0..k {
        let mu = theta[state].exp();
        let p = 1.0 / (1.0 + (-theta[k + state]).exp());
        let r = theta[index_r(state)].exp();
        let tau = theta[index_tau(state)].exp();
        // The total: sum_j T_j ln(1 + j / r) + sum_t w [-(r + y) ln(1 + m / r) + y ln mu].
        let mut nb = 0.0;
        let mut d_r = 0.0;
        for (j, &tail) in moments.count_tail(state).iter().enumerate() {
            let jf = j as f64;
            nb += tail * (jf / r).ln_1p();
            d_r += tail * (1.0 / (r + jf) - 1.0 / r);
        }
        let mut d_mu = 0.0;
        let column = moments.column(state);
        for (i, &w) in column.iter().enumerate() {
            if w == 0.0 {
                continue;
            }
            let (y, e) = (moments.counts[i], moments.exposures[i]);
            let rate = e * mu;
            let share = log_share(r, rate);
            nb -= w * (r + y) * share;
            d_r += w * (-share + (rate - y) / (r + rate) + y / r);
            d_mu -= w * e * (y + r) / (r + rate);
        }
        nb += moments.count_sum[state] * mu.ln();
        d_mu += moments.count_sum[state] / mu;
        // The successes: tails of ln(a + j), ln(b + j), ln(tau + j); the ln a, ln b, ln tau
        // halves of each rising factorial are carried by the sums of z, n - z and n.
        let (a, b) = (p * tau, (1.0 - p) * tau);
        let (success, failure, depth) = moments.beta_binomial(state);
        let mut bb = 0.0;
        let (mut d_a, mut d_b, mut d_tau) = (0.0, 0.0, 0.0);
        for (j, &tail) in success.iter().enumerate() {
            bb += tail * (a + j as f64).ln();
            d_a += tail / (a + j as f64);
        }
        for (j, &tail) in failure.iter().enumerate() {
            bb += tail * (b + j as f64).ln();
            d_b += tail / (b + j as f64);
        }
        for (j, &tail) in depth.iter().enumerate() {
            bb -= tail * (tau + j as f64).ln();
            d_tau += tail / (tau + j as f64);
        }
        let g_a = d_a - d_tau;
        let g_b = d_b - d_tau;
        value += nb + bb;
        gradient[state] -= d_mu * mu;
        gradient[k + state] -= tau * (g_a - g_b) * p * (1.0 - p);
        gradient[index_r(state)] -= d_r * r;
        gradient[index_tau(state)] -= (p * g_a + (1.0 - p) * g_b) * tau;
    }
    for g in gradient.iter_mut() {
        *g /= scale;
    }
    -value / scale
}

/// How an L-BFGS solve ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LbfgsStop {
    /// The gradient's largest entry fell under the tolerance.
    Gradient,
    /// An accepted step lowered the value by at most the value tolerance, relative.
    Value,
    /// The backtracking line search found no decrease.
    LineSearch,
    /// The iteration cap ran out.
    Budget,
}

/// L-BFGS with `memory` pairs and a backtracking Armijo search, minimizing `objective` from
/// `x` in place: Nocedal & Wright's two-loop recursion (Algorithm 7.4), the first step scaled
/// to unit length, a pair dropped where its curvature `s . y` is not positive. It stops where
/// the gradient's largest entry is at most `gradient_tolerance`, or a step lowers the value by
/// at most `value_tolerance` relative to it (scipy's `ftol`), or the search finds no decrease.
///
/// # Returns
/// The final value, the iterations taken and how it stopped.
pub fn lbfgs(
    mut objective: impl FnMut(&[f64], &mut [f64]) -> f64,
    x: &mut [f64],
    memory: usize,
    gradient_tolerance: f64,
    value_tolerance: f64,
    max_iterations: u32,
) -> (f64, u32, LbfgsStop) {
    let d = x.len();
    let mut g = vec![0.0; d];
    let mut f = objective(x, &mut g);
    let mut pairs: std::collections::VecDeque<(Vec<f64>, Vec<f64>, f64)> =
        std::collections::VecDeque::with_capacity(memory);
    let mut direction = vec![0.0; d];
    let mut trial = vec![0.0; d];
    let mut g_trial = vec![0.0; d];
    let mut alpha_k = vec![0.0; memory];
    for iteration in 0..max_iterations {
        if g.iter().fold(0.0_f64, |m, v| m.max(v.abs())) <= gradient_tolerance {
            return (f, iteration, LbfgsStop::Gradient);
        }
        // Two-loop recursion: direction = -H g.
        direction.copy_from_slice(&g);
        for (i, (s, y, rho)) in pairs.iter().enumerate().rev() {
            let a = rho * dot(s, &direction);
            alpha_k[i] = a;
            for (q, yi) in direction.iter_mut().zip(y) {
                *q -= a * yi;
            }
        }
        let gamma = pairs
            .back()
            .map_or(1.0 / norm(&g).max(f64::MIN_POSITIVE), |(s, y, _)| {
                dot(s, y) / dot(y, y)
            });
        for q in &mut direction {
            *q *= gamma;
        }
        for (i, (s, y, rho)) in pairs.iter().enumerate() {
            let b = rho * dot(y, &direction);
            for (q, si) in direction.iter_mut().zip(s) {
                *q += (alpha_k[i] - b) * si;
            }
        }
        for q in &mut direction {
            *q = -*q;
        }
        let mut slope = dot(&g, &direction);
        if slope >= 0.0 {
            // Not a descent direction: restart from steepest descent.
            pairs.clear();
            let scale = 1.0 / norm(&g).max(f64::MIN_POSITIVE);
            for (q, gi) in direction.iter_mut().zip(&g) {
                *q = -gi * scale;
            }
            slope = dot(&g, &direction);
        }
        let mut step = 1.0;
        let mut accepted = None;
        for _ in 0..40 {
            for ((t, xi), di) in trial.iter_mut().zip(x.iter()).zip(&direction) {
                *t = xi + step * di;
            }
            let value = objective(&trial, &mut g_trial);
            if value.is_finite() && value <= f + 1e-4 * step * slope {
                accepted = Some(value);
                break;
            }
            step *= 0.5;
        }
        let Some(value) = accepted else {
            return (f, iteration, LbfgsStop::LineSearch);
        };
        let s: Vec<f64> = trial.iter().zip(x.iter()).map(|(a, b)| a - b).collect();
        let y: Vec<f64> = g_trial.iter().zip(&g).map(|(a, b)| a - b).collect();
        let curvature = dot(&s, &y);
        if curvature > 1e-16 * dot(&y, &y).sqrt() * norm(&s) {
            if pairs.len() == memory {
                pairs.pop_front();
            }
            pairs.push_back((s, y, 1.0 / curvature));
        }
        x.copy_from_slice(&trial);
        g.copy_from_slice(&g_trial);
        let decrease = f - value;
        f = value;
        if decrease <= value_tolerance * f.abs().max(1.0) {
            return (f, iteration + 1, LbfgsStop::Value);
        }
    }
    (f, max_iterations, LbfgsStop::Budget)
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[inline]
fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// The unphased count-pair HMM: data, parameters and buffers.
pub struct Model {
    pub data: Data,
    pub params: Params,
    pub options: Options,
    density: Vec<f64>,
    gamma: Vec<f64>,
    weights: Vec<f64>,
    counts: Vec<f64>,
    evidence: Vec<f64>,
}

impl Model {
    /// # Errors
    /// Parameters of the wrong length or outside their domains, more than [`MAX_STATES`]
    /// states, or a tied model whose dispersions or concentrations differ.
    pub fn new(data: Data, params: Params, options: Options) -> Result<Self, String> {
        let k = params.n_states();
        if !(2..=MAX_STATES).contains(&k) {
            return Err(format!("a model takes 2 to {MAX_STATES} states, got {k}"));
        }
        if params.dispersion.len() != k
            || params.alpha.len() != k
            || params.beta.len() != k
            || params.log_initial.len() != k
            || params.log_transition.len() != k * k
        {
            return Err("every emission parameter holds K values, the transition K x K".into());
        }
        if params
            .dispersion
            .iter()
            .chain(&params.mean)
            .chain(&params.alpha)
            .chain(&params.beta)
            .any(|&v| !(v.is_finite() && v > 0.0))
        {
            return Err(
                "every dispersion, mean, alpha and beta must be finite and positive".into(),
            );
        }
        if options.tied {
            let tau: Vec<f64> = params
                .alpha
                .iter()
                .zip(&params.beta)
                .map(|(a, b)| a + b)
                .collect();
            let close = |v: &[f64]| v.iter().all(|&x| (x - v[0]).abs() <= 1e-12 * v[0]);
            if !close(&params.dispersion) || !close(&tau) {
                return Err(
                    "a tied model holds one dispersion and one concentration across states".into(),
                );
            }
        }
        let n = data.len();
        Ok(Self {
            density: vec![0.0; n * k],
            gamma: vec![0.0; n * k],
            weights: vec![0.0; n * k],
            counts: vec![0.0; k * k],
            evidence: vec![0.0; data.lengths.len()],
            data,
            params,
            options,
        })
    }

    /// States.
    #[must_use]
    pub fn n_states(&self) -> usize {
        self.params.n_states()
    }

    /// Score and run the E step at the held parameters; the log-likelihood.
    ///
    /// # Errors
    /// The ragged kernel's refusal.
    pub fn e_step(&mut self) -> Result<f64, String> {
        log_density_into(&self.data, &self.params, &mut self.density);
        ragged_posteriors_into(
            &self.density,
            self.n_states(),
            &self.data.lengths,
            &self.params.log_initial,
            &self.params.log_transition,
            &mut self.gamma,
            &mut self.counts,
            &mut self.evidence,
            &[],
            SwitchKind::StayOrMove,
        )?;
        Ok(self.evidence.iter().sum())
    }

    /// The log posterior `(n, K)`, log transition counts `(K, K)` and evidences of the last E step.
    #[must_use]
    pub fn posterior_buffers(&self) -> (&[f64], &[f64], &[f64]) {
        (&self.gamma, &self.counts, &self.evidence)
    }

    /// The emission M step on `(n, K)` posterior probabilities, into the held parameters.
    #[must_use]
    pub fn emission_m_step(&mut self, weights: &[f64], solver: Solver) -> Report {
        let k = self.n_states();
        let moments = Moments::new(&self.data, weights, k);
        match solver {
            Solver::Newton => self.newton(&moments),
            Solver::Lbfgs => self.joint(&moments),
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn newton(&mut self, m: &Moments) -> Report {
        let k = self.n_states();
        let data = &self.data;
        let p = &mut self.params;
        let mut report = Report {
            converged: true,
            ..Report::default()
        };
        // --- the total: closed-form mean, then the dispersion ----------------------------
        let live: Vec<bool> = (0..k)
            .map(|s| m.total_weight[s] >= COLLAPSED_MASS && m.count_sum[s] > 0.0)
            .collect();
        for state in 0..k {
            if live[state] {
                p.mean[state] = m.count_sum[state] / m.exposure_sum[state];
            } else {
                report.frozen.push(state);
            }
        }
        let bounds = |state: usize| {
            let rate = p.mean[state] * m.exposure_sum[state] / m.total_weight[state];
            let rate = if data.varying {
                rate
            } else {
                m.exposures.first().map_or(rate, |&e| e * p.mean[state])
            };
            let upper = rate * (m.total_weight[state] / 2.0).sqrt();
            (
                (upper * DISPERSION_BRACKET_RATIO).max(DISPERSION_FLOOR),
                upper,
            )
        };
        if self.options.tied {
            let states: Vec<usize> = (0..k).filter(|&s| live[s]).collect();
            if !states.is_empty() {
                let pooled: Vec<f64> = (0..m.count_width)
                    .map(|j| states.iter().map(|&s| m.count_tail(s)[j]).sum())
                    .collect();
                let total: f64 = states.iter().map(|&s| m.total_weight[s]).sum();
                let rate_sum: f64 = states.iter().map(|&s| p.mean[s] * m.exposure_sum[s]).sum();
                let upper = rate_sum / total * (total / 2.0).sqrt();
                let lower = (upper * DISPERSION_BRACKET_RATIO).max(DISPERSION_FLOOR);
                let means: Vec<f64> = states.iter().map(|&s| p.mean[s]).collect();
                let columns: Vec<&[f64]> = states.iter().map(|&s| m.column(s)).collect();
                let varying = data.varying;
                let score = |r: f64| {
                    let mut sum = count_mstep::rising(&pooled, r);
                    for (column, &mean) in columns.iter().zip(&means) {
                        for (i, &w) in column.iter().enumerate() {
                            let rate = m.exposures[i] * mean;
                            sum += w * (r / (r + rate)).ln();
                            if varying {
                                sum += w * (rate - m.counts[i]) / (r + rate);
                            }
                        }
                    }
                    sum
                };
                let score_slope = |r: f64| {
                    let (mut value, mut slope_r, mut slope_log) = (0.0, 0.0, 0.0);
                    for (j, &tail) in pooled.iter().enumerate() {
                        let inverse = 1.0 / (r + j as f64);
                        value += tail * inverse;
                        slope_r -= tail * inverse * inverse;
                    }
                    for (column, &mean) in columns.iter().zip(&means) {
                        for (i, &w) in column.iter().enumerate() {
                            let rate = m.exposures[i] * mean;
                            let inverse = 1.0 / (r + rate);
                            value += w * (r * inverse).ln();
                            slope_log += w * rate * inverse;
                            if varying {
                                let term = w * (rate - m.counts[i]) * inverse;
                                value += term;
                                slope_r -= term * inverse;
                            }
                        }
                    }
                    (value, r * slope_r + slope_log)
                };
                let solved = solve_bracketed_newton(
                    score,
                    score_slope,
                    total,
                    lower,
                    upper,
                    DISPERSION_TOLERANCE,
                    p.dispersion[states[0]],
                );
                p.dispersion.fill(solved.value);
                report.at_boundary |= solved.at_boundary;
                report.iterations = report.iterations.max(solved.iterations);
                report.residual = report.residual.max(solved.residual);
            }
        } else {
            let solved: Vec<(usize, count_mstep::Dispersion)> = (0..k)
                .filter(|&s| live[s])
                .collect::<Vec<_>>()
                .par_iter()
                .map(|&state| {
                    let (lower, upper) = bounds(state);
                    (
                        state,
                        solve_exposed_dispersion(
                            Exposed {
                                tails: m.count_tail(state),
                                weights: m.column(state),
                                exposures: &m.exposures,
                                counts: &m.counts,
                                mean: p.mean[state],
                                total: m.total_weight[state],
                                varying: data.varying,
                            },
                            lower,
                            upper,
                            DISPERSION_TOLERANCE,
                            p.dispersion[state],
                        ),
                    )
                })
                .collect();
            for (state, one) in solved {
                p.dispersion[state] = one.value;
                report.at_boundary |= one.at_boundary;
                report.iterations = report.iterations.max(one.iterations);
                report.residual = report.residual.max(one.residual);
            }
        }
        for state in 0..k {
            if live[state] && p.dispersion[state] <= DISPERSION_FLOOR {
                report.degenerate.push(state);
                report.converged = false;
            }
        }
        // --- the successes: rate and concentration ---------------------------------------
        let effective: Vec<f64> = (0..k)
            .map(|s| m.effective_trials(data, m.depth_sum[s], m.success_weight[s]))
            .collect();
        let bb_live: Vec<bool> = (0..k)
            .map(|s| m.success_weight[s] >= COLLAPSED_MASS && effective[s] >= 2.0)
            .collect();
        for state in 0..k {
            if !bb_live[state] && !report.frozen.contains(&state) {
                report.frozen.push(state);
            }
        }
        let problem = |state: usize, bound: f64| {
            let (success, failure, depth) = m.beta_binomial(state);
            let tau = p.alpha[state] + p.beta[state];
            BetaBinomialProblem {
                success,
                failure,
                depth,
                total: m.success_weight[state],
                rate: p.alpha[state] / tau,
                concentration: tau.min(bound),
                bound,
                log_low: bound.ln() + CONCENTRATION_BRACKET_RATIO.ln(),
                log_high: bound.ln(),
            }
        };
        if self.options.tied {
            let states: Vec<usize> = (0..k).filter(|&s| bb_live[s]).collect();
            if !states.is_empty() {
                let weight: f64 = states.iter().map(|&s| m.success_weight[s]).sum();
                let depth: f64 = states.iter().map(|&s| m.depth_sum[s]).sum();
                let trials = m.effective_trials(data, depth, weight);
                let bound = (trials - 1.0) * (weight / 2.0).sqrt();
                let problems: Vec<BetaBinomialProblem<'_>> =
                    states.iter().map(|&s| problem(s, bound)).collect();
                let rates: Vec<f64> = problems.iter().map(|q| q.rate).collect();
                let start = (p.alpha[states[0]] + p.beta[states[0]]).min(bound);
                let solved = solve_beta_binomial_tied(
                    &problems,
                    &rates,
                    start,
                    bound,
                    CONCENTRATION_BRACKET_RATIO,
                    BETA_BINOMIAL,
                );
                for (&state, &rate) in states.iter().zip(&solved.rates) {
                    p.alpha[state] = rate * solved.concentration;
                    p.beta[state] = (1.0 - rate) * solved.concentration;
                }
                for state in (0..k).filter(|&s| !bb_live[s]) {
                    let rate = p.alpha[state] / (p.alpha[state] + p.beta[state]);
                    p.alpha[state] = rate * solved.concentration;
                    p.beta[state] = (1.0 - rate) * solved.concentration;
                }
                report.at_boundary |= solved.at_boundary;
                report.converged &= solved.converged;
                report.iterations = report.iterations.max(solved.iterations);
                report.residual = report.residual.max(solved.residual);
            }
        } else {
            let solved: Vec<(usize, count_mstep::BetaBinomial)> = (0..k)
                .filter(|&s| bb_live[s])
                .collect::<Vec<_>>()
                .par_iter()
                .map(|&state| {
                    let bound = (effective[state] - 1.0) * (m.success_weight[state] / 2.0).sqrt();
                    (
                        state,
                        solve_beta_binomial(problem(state, bound), BETA_BINOMIAL),
                    )
                })
                .collect();
            for (state, one) in solved {
                p.alpha[state] = one.alpha;
                p.beta[state] = one.beta;
                report.at_boundary |= one.at_boundary;
                report.converged &= one.converged;
                report.iterations = report.iterations.max(one.iterations);
                report.residual = report.residual.max(one.residual);
            }
        }
        report.frozen.sort_unstable();
        report
    }

    fn joint(&mut self, m: &Moments) -> Report {
        let k = self.n_states();
        let tied = self.options.tied;
        let shared = if tied { 1 } else { k };
        let p = &mut self.params;
        let mut theta = vec![0.0; 2 * k + 2 * shared];
        for state in 0..k {
            let tau = p.alpha[state] + p.beta[state];
            let rate = p.alpha[state] / tau;
            theta[state] = p.mean[state].ln();
            theta[k + state] = (rate / (1.0 - rate)).ln();
        }
        for i in 0..shared {
            theta[2 * k + i] = p.dispersion[i].ln();
            theta[2 * k + shared + i] = (p.alpha[i] + p.beta[i]).ln();
        }
        let (_, iterations, stop) = lbfgs(
            |x, g| emission_objective(m, x, tied, g),
            &mut theta,
            10,
            1e-10,
            1e-14,
            500,
        );
        for state in 0..k {
            let index = if tied { 0 } else { state };
            let rate = 1.0 / (1.0 + (-theta[k + state]).exp());
            let tau = theta[2 * k + shared + index].exp();
            p.mean[state] = theta[state].exp();
            p.dispersion[state] = theta[2 * k + index].exp();
            p.alpha[state] = rate * tau;
            p.beta[state] = (1.0 - rate) * tau;
        }
        let mut gradient = vec![0.0; theta.len()];
        emission_objective(m, &theta, tied, &mut gradient);
        let finite = p
            .dispersion
            .iter()
            .chain(&p.mean)
            .chain(&p.alpha)
            .chain(&p.beta)
            .all(|&v| v.is_finite() && v > 0.0);
        Report {
            converged: finite && stop != LbfgsStop::Budget,
            at_boundary: false,
            iterations,
            residual: gradient.iter().fold(0.0_f64, |a, b| a.max(b.abs())),
            frozen: Vec::new(),
            degenerate: if finite { Vec::new() } else { (0..k).collect() },
        }
    }

    /// The initial distribution and transition from the last E step, as the options select.
    fn chain_m_step(&mut self) {
        let k = self.n_states();
        if self.options.fit_initial {
            let segments = self.data.lengths.len() as f64;
            let mut first = 0;
            let mut column = vec![0.0; self.data.lengths.len()];
            for state in 0..k {
                first = 0;
                for (s, &length) in self.data.lengths.iter().enumerate() {
                    column[s] = self.gamma[first * k + state];
                    first += length;
                }
                self.params.log_initial[state] = log_sum(&column) - segments.ln();
            }
            debug_assert_eq!(first, self.data.len());
        }
        if self.options.fit_transition {
            for row in 0..k {
                let counts = &self.counts[row * k..(row + 1) * k];
                let norm = log_sum(counts);
                for (to, &c) in counts.iter().enumerate() {
                    self.params.log_transition[row * k + to] = c - norm;
                }
            }
        }
    }

    /// Baum-Welch to the stopping rule, `sal.opt.em.em_loop`'s alternation: each iteration
    /// scores the held parameters (its log-likelihood), then re-estimates them. It stops when the
    /// log-likelihood moves by at most `tolerance` relative to its magnitude, when `parameter_tolerance`
    /// is positive and no emission parameter moved by more than it relative to its value, or when
    /// `max_iterations` run out. An emission M step that does not settle ends the fit on the
    /// parameters before it, as `em_loop` ends one with `Stop.DEGENERATE`.
    ///
    /// # Errors
    /// A non-finite log-likelihood, naming the iteration, or the ragged kernel's refusal.
    pub fn fit(
        &mut self,
        max_iterations: u32,
        tolerance: f64,
        parameter_tolerance: f64,
        solver: Solver,
    ) -> Result<Fitted, String> {
        let mut previous = f64::NEG_INFINITY;
        let mut log_likelihood = previous;
        let mut iterations = 0;
        let mut at_boundary = false;
        let mut frozen: Vec<usize> = Vec::new();
        let mut stop = Stop::Budget;
        let mut unsettled = None;
        while iterations < max_iterations {
            iterations += 1;
            let before = self.params.clone();
            let value = self.e_step()?;
            for (w, g) in self.weights.iter_mut().zip(&self.gamma) {
                *w = g.exp();
            }
            let weights = std::mem::take(&mut self.weights);
            let report = self.emission_m_step(&weights, solver);
            self.weights = weights;
            if !report.converged {
                self.params = before;
                unsettled = Some(report);
                stop = Stop::Degenerate;
                break;
            }
            self.chain_m_step();
            log_likelihood = value;
            if !log_likelihood.is_finite() {
                return Err(format!(
                    "EM log-likelihood is {log_likelihood} at iteration {iterations}; a \
                     non-finite value is refused rather than iterated on"
                ));
            }
            at_boundary |= report.at_boundary;
            for state in report.frozen {
                if !frozen.contains(&state) {
                    frozen.push(state);
                }
            }
            if (log_likelihood - previous).abs() <= tolerance * log_likelihood.abs()
                || (parameter_tolerance > 0.0
                    && self.params.relative_change(&before) <= parameter_tolerance)
            {
                stop = Stop::Converged;
                break;
            }
            previous = log_likelihood;
        }
        frozen.sort_unstable();
        Ok(Fitted {
            log_likelihood,
            iterations,
            stop,
            at_boundary,
            frozen,
            unsettled,
        })
    }

    /// The most probable path of every segment at the held parameters, and each one's joint
    /// log-probability.
    ///
    /// # Errors
    /// The ragged kernel's refusal.
    pub fn viterbi(&mut self) -> Result<(Vec<i64>, Vec<f64>), String> {
        log_density_into(&self.data, &self.params, &mut self.density);
        let mut path = vec![0_i64; self.data.len()];
        let mut joint = vec![0.0; self.data.lengths.len()];
        ragged_viterbi_into(
            &self.density,
            self.n_states(),
            &self.data.lengths,
            &self.params.log_initial,
            &self.params.log_transition,
            &[],
            SwitchKind::StayOrMove,
            &mut path,
            &mut joint,
        )?;
        Ok((path, joint))
    }
}

// --- the binding ------------------------------------------------------------------------

fn slice<'a, T: numpy::Element>(
    array: &'a PyReadonlyArray1<'_, T>,
    name: &str,
) -> PyResult<&'a [T]> {
    array
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be C-contiguous")))
}

/// `sal.opt.hmm.CountPairHmm`'s compiled half: one object per model, its parameters read and
/// written through it, each method one crossing with the GIL released.
#[pyclass(name = "CountPairHmm", module = "sal.oxisal")]
pub struct PyCountPairHmm {
    model: Model,
}

/// The parameters as `(dispersion, mean, alpha, beta, log_initial, log_transition)`.
type Arrays<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
);

#[pymethods]
impl PyCountPairHmm {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (counts, exposures, successes, trials, lengths, dispersion, mean, alpha, beta, log_initial, log_transition, *, tied, fit_initial, fit_transition))]
    fn new(
        counts: PyReadonlyArray1<'_, u32>,
        exposures: PyReadonlyArray1<'_, f64>,
        successes: PyReadonlyArray1<'_, u32>,
        trials: PyReadonlyArray1<'_, u32>,
        lengths: PyReadonlyArray1<'_, i64>,
        dispersion: PyReadonlyArray1<'_, f64>,
        mean: PyReadonlyArray1<'_, f64>,
        alpha: PyReadonlyArray1<'_, f64>,
        beta: PyReadonlyArray1<'_, f64>,
        log_initial: PyReadonlyArray1<'_, f64>,
        log_transition: PyReadonlyArray2<'_, f64>,
        tied: bool,
        fit_initial: bool,
        fit_transition: bool,
    ) -> PyResult<Self> {
        let widths = slice(&lengths, "lengths")?
            .iter()
            .map(|&l| {
                usize::try_from(l)
                    .map_err(|_| PyValueError::new_err("lengths must be non-negative"))
            })
            .collect::<PyResult<Vec<usize>>>()?;
        let data = Data::new(
            slice(&counts, "counts")?.to_vec(),
            slice(&exposures, "exposures")?.to_vec(),
            slice(&successes, "successes")?.to_vec(),
            slice(&trials, "trials")?.to_vec(),
            widths,
        )
        .map_err(PyValueError::new_err)?;
        let params = Params {
            dispersion: slice(&dispersion, "dispersion")?.to_vec(),
            mean: slice(&mean, "mean")?.to_vec(),
            alpha: slice(&alpha, "alpha")?.to_vec(),
            beta: slice(&beta, "beta")?.to_vec(),
            log_initial: slice(&log_initial, "log_initial")?.to_vec(),
            log_transition: log_transition
                .as_slice()
                .map_err(|_| PyValueError::new_err("log_transition must be C-contiguous"))?
                .to_vec(),
        };
        let model = Model::new(
            data,
            params,
            Options {
                tied,
                fit_initial,
                fit_transition,
            },
        )
        .map_err(PyValueError::new_err)?;
        Ok(Self { model })
    }

    /// States.
    #[getter]
    fn n_states(&self) -> usize {
        self.model.n_states()
    }

    /// The held parameters, copied out.
    fn parameters<'py>(&self, py: Python<'py>) -> PyResult<Arrays<'py>> {
        let p = &self.model.params;
        let k = p.n_states();
        let transition = numpy::ndarray::Array2::from_shape_vec((k, k), p.log_transition.clone())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((
            p.dispersion.clone().into_pyarray(py),
            p.mean.clone().into_pyarray(py),
            p.alpha.clone().into_pyarray(py),
            p.beta.clone().into_pyarray(py),
            p.log_initial.clone().into_pyarray(py),
            transition.into_pyarray(py),
        ))
    }

    /// The whole fit; `(log_likelihood, iterations, stop, at_boundary, frozen, unsettled)`,
    /// `unsettled` `(iterations, residual, degenerate states)` or `None`.
    #[allow(clippy::type_complexity)]
    fn fit(
        &mut self,
        py: Python<'_>,
        max_iterations: u32,
        tolerance: f64,
        parameter_tolerance: f64,
        solver: &str,
    ) -> PyResult<(
        f64,
        u32,
        &'static str,
        bool,
        Vec<usize>,
        Option<(u32, f64, Vec<usize>)>,
    )> {
        let solver = Solver::parse(solver).map_err(PyValueError::new_err)?;
        let model = &mut self.model;
        let fitted = py
            .detach(|| model.fit(max_iterations, tolerance, parameter_tolerance, solver))
            .map_err(PyValueError::new_err)?;
        Ok((
            fitted.log_likelihood,
            fitted.iterations,
            fitted.stop.name(),
            fitted.at_boundary,
            fitted.frozen,
            fitted
                .unsettled
                .map(|r| (r.iterations, r.residual, r.degenerate)),
        ))
    }

    /// The E step at the held parameters: `(log_posterior (n, K), log_counts (K, K), log_evidence)`.
    #[allow(clippy::type_complexity)]
    fn posteriors<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray1<f64>>,
    )> {
        let model = &mut self.model;
        py.detach(|| model.e_step())
            .map_err(PyValueError::new_err)?;
        let k = model.n_states();
        let (gamma, counts, evidence) = model.posterior_buffers();
        let shape = |rows: usize, values: &[f64]| {
            numpy::ndarray::Array2::from_shape_vec((rows, k), values.to_vec())
                .map_err(|e| PyValueError::new_err(e.to_string()))
        };
        Ok((
            shape(gamma.len() / k, gamma)?.into_pyarray(py),
            shape(k, counts)?.into_pyarray(py),
            evidence.to_vec().into_pyarray(py),
        ))
    }

    /// The log-likelihood at the held parameters.
    fn log_likelihood(&mut self, py: Python<'_>) -> PyResult<f64> {
        let model = &mut self.model;
        py.detach(|| model.e_step()).map_err(PyValueError::new_err)
    }

    /// The emission M step on `(n, K)` posterior probabilities, into the held parameters:
    /// `(converged, at_boundary, iterations, residual, frozen, degenerate)`.
    #[allow(clippy::type_complexity)]
    fn m_step(
        &mut self,
        py: Python<'_>,
        posterior: PyReadonlyArray2<'_, f64>,
        solver: &str,
    ) -> PyResult<(bool, bool, u32, f64, Vec<usize>, Vec<usize>)> {
        let solver = Solver::parse(solver).map_err(PyValueError::new_err)?;
        let weights = posterior
            .as_slice()
            .map_err(|_| PyValueError::new_err("posterior must be C-contiguous"))?;
        let model = &mut self.model;
        if weights.len() != model.data.len() * model.n_states() {
            return Err(PyValueError::new_err("posterior must be (n, K)"));
        }
        let report = py.detach(|| model.emission_m_step(weights, solver));
        Ok((
            report.converged,
            report.at_boundary,
            report.iterations,
            report.residual,
            report.frozen,
            report.degenerate,
        ))
    }

    /// The most probable path and each segment's joint log-probability at the held parameters.
    #[allow(clippy::type_complexity)]
    fn viterbi<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>)> {
        let model = &mut self.model;
        let (path, joint) = py
            .detach(|| model.viterbi())
            .map_err(PyValueError::new_err)?;
        Ok((path.into_pyarray(py), joint.into_pyarray(py)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A quadratic's minimum, from a start away from it.
    #[test]
    fn lbfgs_finds_a_quadratic_minimum() {
        let mut x = vec![3.0, -2.0];
        let (value, _, stop) = lbfgs(
            |x, g| {
                g[0] = 2.0 * (x[0] - 1.0);
                g[1] = 20.0 * (x[1] + 0.5);
                (x[0] - 1.0).powi(2) + 10.0 * (x[1] + 0.5).powi(2)
            },
            &mut x,
            5,
            1e-12,
            0.0,
            100,
        );
        assert_eq!(stop, LbfgsStop::Gradient);
        assert!((x[0] - 1.0).abs() < 1e-10 && (x[1] + 0.5).abs() < 1e-10);
        assert!(value < 1e-20);
    }

    /// The objective's gradient against central differences on a small model.
    #[test]
    fn the_emission_gradient_is_the_objectives() {
        let data = Data::new(
            vec![3, 10, 0, 7, 25, 4],
            vec![1.0, 2.0, 0.5, 1.5, 3.0, 1.0],
            vec![1, 4, 0, 2, 9, 3],
            vec![5, 8, 0, 6, 12, 3],
            vec![6],
        )
        .unwrap();
        let weights = [0.3, 0.7, 0.9, 0.1, 0.5, 0.5, 0.2, 0.8, 0.6, 0.4, 0.1, 0.9];
        let moments = Moments::new(&data, &weights, 2);
        for tied in [false, true] {
            let theta: Vec<f64> = if tied {
                vec![1.0, 1.5, -0.2, 0.4, 0.7, 2.0]
            } else {
                vec![1.0, 1.5, -0.2, 0.4, 0.7, 0.2, 2.0, 1.5]
            };
            let mut gradient = vec![0.0; theta.len()];
            emission_objective(&moments, &theta, tied, &mut gradient);
            for i in 0..theta.len() {
                let h = 1e-6;
                let mut up = theta.clone();
                let mut down = theta.clone();
                up[i] += h;
                down[i] -= h;
                let mut scratch = vec![0.0; theta.len()];
                let numeric = (emission_objective(&moments, &up, tied, &mut scratch)
                    - emission_objective(&moments, &down, tied, &mut scratch))
                    / (2.0 * h);
                assert!(
                    (numeric - gradient[i]).abs() < 1e-7,
                    "tied {tied} entry {i}: {numeric} against {}",
                    gradient[i]
                );
            }
        }
    }
}
