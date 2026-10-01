//! A count mixture's log-likelihood and its gradient in one pass (issue #1136), exposed to
//! Python as `sal.oxisal.count_mixture_value_and_gradient`.
//!
//! A Hamiltonian chain on `EmissionMixtureObjective` spent its time in torch's per-operation
//! dispatch and autograd's backward pass over `N x K` (7.4 s of 19 s at the stress mixture),
//! not in arithmetic. Here the value and the gradient in the natural parameters come out of
//! one crossing; the map from `theta` to them stays in torch, `K`-sized.
//!
//! **Every special function is a sum over integers.** With `R_x[m] = sum_{j < m} ln(1 + j/x)`
//! and `D_x[m] = sum_{j < m} 1 / (x + j)`,
//! `lgamma(x + m) - lgamma(x) = m ln x + R_x[m]` and `digamma(x + m) - digamma(x) = D_x[m]`,
//! exactly, for integer `m`. Both are prefix sums, tabulated once per call per state up to the
//! largest count: neither cancels as the shape grows, which is the large-shape defect this
//! issue fixes in torch by a differenced series, and neither needs a special function beyond
//! `ln_gamma` for the parameter-free `ln m!`. The prefix sums are compensated (Neumaier), so a
//! table of a few thousand entries keeps its rounding near one ulp of its largest entry.
//!
//! The densities, with `u = mu / r` and `tau = a + b`:
//!
//! - negative binomial: `R_r[y] - ln y! - mu ln(1 + u) / u + y (ln mu - ln(1 + u))`, the
//!   Poisson as `r -> inf`; `d/dr = D_r[y] - ln(1 + u) + (mu - y) / (r + mu)`,
//!   `d/dmu = y / mu - (y + r) / (r + mu)`;
//! - beta-binomial: `ln C(n, z) + z ln(a / tau) + (n - z) ln(b / tau) + R_a[z] + R_b[n - z] -
//!   R_tau[n]`; `d/da = D_a[z] - D_tau[n]`, `d/db = D_b[n - z] - D_tau[n]`. Its trial count is
//!   the observed total (the joint pair) or a constant per state.
//!
//! **Deterministic in parallel.** Observations are cut into fixed tiles, each tile's sums are
//! formed in order, and the tiles' partial sums are added in tile order, so the result is the
//! same at every thread count.

use numpy::{PyReadonlyArray1, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::special::ln_gamma;

/// Observations per tile of the parallel pass: 3,000 observations make twelve, enough for every
/// core of the reference host.
const TILE: usize = 256;

/// `sum` of `values` in order, Neumaier-compensated, as a running prefix: `out[m]` is the sum
/// of the first `m`.
fn compensated_prefix(len: usize, term: impl Fn(usize) -> f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(len + 1);
    let (mut sum, mut carry) = (0.0_f64, 0.0_f64);
    out.push(0.0);
    for j in 0..len {
        let value = term(j);
        let next = sum + value;
        carry += if sum.abs() >= value.abs() {
            (sum - next) + value
        } else {
            (value - next) + sum
        };
        sum = next;
        out.push(sum + carry);
    }
    out
}

/// `R_x` and `D_x` up to `extent` (inclusive), count-major over `K` states: `[m * K + k]`.
struct Rising {
    log: Vec<f64>,
    reciprocal: Vec<f64>,
}

impl Rising {
    fn new(shapes: &[f64], extent: usize) -> Self {
        let k = shapes.len();
        // Each state's two prefix sums are independent of every other's, so the states run in
        // parallel; the transpose to count-major is a copy.
        let columns: Vec<(Vec<f64>, Vec<f64>)> = shapes
            .par_iter()
            .map(|&x| {
                (
                    compensated_prefix(extent, |j| (j as f64 / x).ln_1p()),
                    compensated_prefix(extent, |j| 1.0 / (x + j as f64)),
                )
            })
            .collect();
        let mut log = vec![0.0; (extent + 1) * k];
        let mut reciprocal = vec![0.0; (extent + 1) * k];
        for (state, (r, d)) in columns.iter().enumerate() {
            for m in 0..=extent {
                log[m * k + state] = r[m];
                reciprocal[m * k + state] = d[m];
            }
        }
        Self { log, reciprocal }
    }
}

/// The negative-binomial channel: its counts and parameters.
pub struct Totals<'a> {
    /// `N` counts.
    pub counts: &'a [u32],
    /// `K` dispersions `r`, finite and positive.
    pub dispersion: &'a [f64],
    /// `K` means `mu`, positive.
    pub mean: &'a [f64],
}

/// The beta-binomial channel: its successes, parameters and trial count.
pub struct Successes<'a> {
    /// `N` successes.
    pub counts: &'a [u32],
    /// `K` values of `a`.
    pub alpha: &'a [f64],
    /// `K` values of `b`.
    pub beta: &'a [f64],
    /// `K` trial counts, or empty for the joint pair, whose trials are the totals.
    pub trials: &'a [f64],
}

/// The gradient of the log-likelihood in each block, `K` entries each; a channel's blocks are
/// empty where it is absent.
#[derive(Clone, Default)]
pub struct Gradient {
    pub log_weight: Vec<f64>,
    pub dispersion: Vec<f64>,
    pub mean: Vec<f64>,
    pub alpha: Vec<f64>,
    pub beta: Vec<f64>,
}

impl Gradient {
    fn zeros(k: usize, totals: bool, successes: bool) -> Self {
        let block = |on: bool| if on { vec![0.0; k] } else { Vec::new() };
        Self {
            log_weight: vec![0.0; k],
            dispersion: block(totals),
            mean: block(totals),
            alpha: block(successes),
            beta: block(successes),
        }
    }

    fn add(&mut self, other: &Self) {
        for (mine, theirs) in [
            (&mut self.log_weight, &other.log_weight),
            (&mut self.dispersion, &other.dispersion),
            (&mut self.mean, &other.mean),
            (&mut self.alpha, &other.alpha),
            (&mut self.beta, &other.beta),
        ] {
            for (a, b) in mine.iter_mut().zip(theirs) {
                *a += b;
            }
        }
    }
}

/// Per-call tables of the negative-binomial channel.
struct TotalTables<'a> {
    channel: &'a Totals<'a>,
    rising: Rising,
    log_factorial: &'a [f64],
    /// `ln(1 + mu / r)` per state.
    log_share: Vec<f64>,
    /// `mu ln(1 + u) / u`, `u = mu / r`, per state: `r ln(1 + mu / r)` without the `0 * inf`.
    pull: Vec<f64>,
    log_mean: Vec<f64>,
}

/// Per-call tables of the beta-binomial channel.
struct SuccessTables<'a> {
    channel: &'a Successes<'a>,
    alpha: Rising,
    beta: Rising,
    total: Rising,
    log_factorial: &'a [f64],
    log_rate: Vec<f64>,
    log_rest: Vec<f64>,
}

/// Validate, then return the log-likelihood and its gradient.
///
/// # Errors
/// A message naming the first violated precondition.
pub fn value_and_gradient(
    log_weight: &[f64],
    totals: Option<&Totals<'_>>,
    successes: Option<&Successes<'_>>,
) -> Result<(f64, Gradient), String> {
    let k = log_weight.len();
    if k == 0 {
        return Err("a mixture has at least one component".to_string());
    }
    let n = totals
        .map(|t| t.counts.len())
        .or_else(|| successes.map(|s| s.counts.len()))
        .ok_or("a count mixture scores at least one channel")?;
    if let Some(t) = totals {
        if t.counts.len() != n || t.dispersion.len() != k || t.mean.len() != k {
            return Err("the totals' counts or parameters have the wrong length".to_string());
        }
        if t.dispersion
            .iter()
            .chain(t.mean)
            .any(|&v| !(v > 0.0 && v.is_finite()))
        {
            return Err("every dispersion and mean must be finite and positive".to_string());
        }
    }
    if let Some(s) = successes {
        if s.counts.len() != n || s.alpha.len() != k || s.beta.len() != k {
            return Err("the successes' counts or parameters have the wrong length".to_string());
        }
        if s.trials.is_empty() && totals.is_none() {
            return Err("the joint pair's trials are the totals, which are absent".to_string());
        }
        if !s.trials.is_empty() && s.trials.len() != k {
            return Err("trials holds one count per state, or none for the joint pair".to_string());
        }
        if s.alpha
            .iter()
            .chain(s.beta)
            .any(|&v| !(v > 0.0 && v.is_finite()))
        {
            return Err("every alpha and beta must be finite and positive".to_string());
        }
    }

    // The largest integer any table is read at, and `ln m!` up to it.
    let largest_total = totals.map_or(0, |t| t.counts.iter().copied().max().unwrap_or(0) as usize);
    let largest_trials = successes.map_or(0, |s| {
        if s.trials.is_empty() {
            largest_total
        } else {
            s.trials.iter().fold(0.0_f64, |a, &b| a.max(b)) as usize
        }
    });
    let largest_success =
        successes.map_or(0, |s| s.counts.iter().copied().max().unwrap_or(0) as usize);
    let extent = largest_total.max(largest_trials).max(largest_success);
    let log_factorial: Vec<f64> = (0..=extent).map(|m| ln_gamma(m as f64 + 1.0)).collect();

    let total_tables = totals.map(|t| TotalTables {
        channel: t,
        rising: Rising::new(t.dispersion, largest_total),
        log_factorial: &log_factorial,
        log_share: t
            .dispersion
            .iter()
            .zip(t.mean)
            .map(|(&r, &mu)| (mu / r).ln_1p())
            .collect(),
        pull: t
            .dispersion
            .iter()
            .zip(t.mean)
            .map(|(&r, &mu)| r * (mu / r).ln_1p())
            .collect(),
        log_mean: t.mean.iter().map(|mu| mu.ln()).collect(),
    });
    let largest_rest = successes.map_or(0, |s| {
        (0..n)
            .map(|i| {
                let trials = if s.trials.is_empty() {
                    totals.map_or(0, |t| t.counts[i] as usize)
                } else {
                    s.trials.iter().fold(0.0_f64, |a, &b| a.max(b)) as usize
                };
                trials.saturating_sub(s.counts[i] as usize)
            })
            .max()
            .unwrap_or(0)
    });
    let success_tables = successes.map(|s| {
        let tau: Vec<f64> = s.alpha.iter().zip(s.beta).map(|(a, b)| a + b).collect();
        let ((alpha, beta), total) = rayon::join(
            || {
                rayon::join(
                    || Rising::new(s.alpha, largest_success),
                    || Rising::new(s.beta, largest_rest),
                )
            },
            || Rising::new(&tau, largest_trials),
        );
        SuccessTables {
            channel: s,
            alpha,
            beta,
            total,
            log_factorial: &log_factorial,
            log_rate: s
                .alpha
                .iter()
                .zip(&tau)
                .map(|(a, t)| (a / t).ln())
                .collect(),
            log_rest: s.beta.iter().zip(&tau).map(|(b, t)| (b / t).ln()).collect(),
        }
    });

    let tiles: Vec<(f64, Gradient)> = (0..n.div_ceil(TILE))
        .into_par_iter()
        .map(|tile| {
            let mut gradient = Gradient::zeros(k, totals.is_some(), successes.is_some());
            let mut log_likelihood = 0.0;
            let mut score = vec![0.0; k];
            let mut d_first = vec![0.0; k];
            let mut d_second = vec![0.0; k];
            let mut d_third = vec![0.0; k];
            let mut d_fourth = vec![0.0; k];
            for i in tile * TILE..((tile + 1) * TILE).min(n) {
                score.copy_from_slice(log_weight);
                if let Some(t) = &total_tables {
                    let y = t.channel.counts[i] as usize;
                    let yf = y as f64;
                    for state in 0..k {
                        let (r, mu) = (t.channel.dispersion[state], t.channel.mean[state]);
                        score[state] +=
                            t.rising.log[y * k + state] - t.log_factorial[y] - t.pull[state]
                                + yf * (t.log_mean[state] - t.log_share[state]);
                        d_first[state] = t.rising.reciprocal[y * k + state] - t.log_share[state]
                            + (mu - yf) / (r + mu);
                        d_second[state] = yf / mu - (yf + r) / (r + mu);
                    }
                }
                if let Some(s) = &success_tables {
                    let z = s.channel.counts[i] as usize;
                    for state in 0..k {
                        let n_trials = if s.channel.trials.is_empty() {
                            total_tables
                                .as_ref()
                                .map_or(0, |t| t.channel.counts[i] as usize)
                        } else {
                            s.channel.trials[state] as usize
                        };
                        if z > n_trials {
                            score[state] = f64::NEG_INFINITY;
                            d_third[state] = 0.0;
                            d_fourth[state] = 0.0;
                            continue;
                        }
                        let rest = n_trials - z;
                        score[state] += (s.log_factorial[n_trials]
                            - s.log_factorial[z]
                            - s.log_factorial[rest])
                            + z as f64 * s.log_rate[state]
                            + rest as f64 * s.log_rest[state]
                            + s.alpha.log[z * k + state]
                            + s.beta.log[rest * k + state]
                            - s.total.log[n_trials * k + state];
                        let shared = s.total.reciprocal[n_trials * k + state];
                        d_third[state] = s.alpha.reciprocal[z * k + state] - shared;
                        d_fourth[state] = s.beta.reciprocal[rest * k + state] - shared;
                    }
                }
                let top = score.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                if top == f64::NEG_INFINITY {
                    log_likelihood = f64::NEG_INFINITY;
                    continue;
                }
                let mut sum = 0.0;
                for value in score.iter_mut() {
                    *value = (*value - top).exp();
                    sum += *value;
                }
                log_likelihood += top + sum.ln();
                for state in 0..k {
                    let rho = score[state] / sum;
                    gradient.log_weight[state] += rho;
                    if totals.is_some() {
                        gradient.dispersion[state] += rho * d_first[state];
                        gradient.mean[state] += rho * d_second[state];
                    }
                    if successes.is_some() {
                        gradient.alpha[state] += rho * d_third[state];
                        gradient.beta[state] += rho * d_fourth[state];
                    }
                }
            }
            (log_likelihood, gradient)
        })
        .collect();

    let mut gradient = Gradient::zeros(k, totals.is_some(), successes.is_some());
    let mut log_likelihood = 0.0;
    for (value, part) in &tiles {
        log_likelihood += value;
        gradient.add(part);
    }
    Ok((log_likelihood, gradient))
}

fn slice<'a>(array: &'a PyReadonlyArray1<'_, f64>, name: &str) -> PyResult<&'a [f64]> {
    array
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be C-contiguous")))
}

fn counts<'a>(array: &'a PyReadonlyArray1<'_, u32>, name: &str) -> PyResult<&'a [u32]> {
    array
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be C-contiguous")))
}

fn write(out: &mut Option<PyReadwriteArray1<'_, f64>>, values: &[f64], name: &str) -> PyResult<()> {
    if let Some(array) = out {
        let target = array
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err(format!("{name} must be C-contiguous")))?;
        if target.len() != values.len() {
            return Err(PyValueError::new_err(format!(
                "{name} has {} entries, expected {}",
                target.len(),
                values.len()
            )));
        }
        target.copy_from_slice(values);
    }
    Ok(())
}

/// [`value_and_gradient`] as a Python binding.
///
/// The total channel is `totals`, `dispersion` and `mean`, given together; the success channel
/// `successes`, `alpha` and `beta`, with `trials` one per state or `None` for the joint pair.
/// The gradient blocks are written into the arrays passed for them; the GIL is released for
/// the pass.
///
/// # Returns
/// The log-likelihood.
///
/// # Errors
/// `ValueError` naming the first violated precondition.
#[pyfunction]
#[pyo3(signature = (log_weight, grad_log_weight, totals=None, dispersion=None, mean=None, grad_dispersion=None, grad_mean=None, successes=None, alpha=None, beta=None, trials=None, grad_alpha=None, grad_beta=None))]
#[allow(clippy::too_many_arguments)]
pub fn count_mixture_value_and_gradient(
    py: Python<'_>,
    log_weight: PyReadonlyArray1<'_, f64>,
    grad_log_weight: PyReadwriteArray1<'_, f64>,
    totals: Option<PyReadonlyArray1<'_, u32>>,
    dispersion: Option<PyReadonlyArray1<'_, f64>>,
    mean: Option<PyReadonlyArray1<'_, f64>>,
    grad_dispersion: Option<PyReadwriteArray1<'_, f64>>,
    grad_mean: Option<PyReadwriteArray1<'_, f64>>,
    successes: Option<PyReadonlyArray1<'_, u32>>,
    alpha: Option<PyReadonlyArray1<'_, f64>>,
    beta: Option<PyReadonlyArray1<'_, f64>>,
    trials: Option<PyReadonlyArray1<'_, f64>>,
    grad_alpha: Option<PyReadwriteArray1<'_, f64>>,
    grad_beta: Option<PyReadwriteArray1<'_, f64>>,
) -> PyResult<f64> {
    let log_weight_slice = slice(&log_weight, "log_weight")?;
    let total_channel = match (&totals, &dispersion, &mean) {
        (Some(c), Some(r), Some(m)) => Some(Totals {
            counts: counts(c, "totals")?,
            dispersion: slice(r, "dispersion")?,
            mean: slice(m, "mean")?,
        }),
        (None, None, None) => None,
        _ => {
            return Err(PyValueError::new_err(
                "the total channel takes totals, dispersion and mean together",
            ))
        }
    };
    let success_channel = match (&successes, &alpha, &beta) {
        (Some(c), Some(a), Some(b)) => Some(Successes {
            counts: counts(c, "successes")?,
            alpha: slice(a, "alpha")?,
            beta: slice(b, "beta")?,
            trials: match &trials {
                Some(t) => slice(t, "trials")?,
                None => &[],
            },
        }),
        (None, None, None) => None,
        _ => {
            return Err(PyValueError::new_err(
                "the success channel takes successes, alpha and beta together",
            ))
        }
    };
    let (value, gradient) = py
        .detach(|| {
            value_and_gradient(
                log_weight_slice,
                total_channel.as_ref(),
                success_channel.as_ref(),
            )
        })
        .map_err(PyValueError::new_err)?;
    let mut grad_log_weight = Some(grad_log_weight);
    write(
        &mut grad_log_weight,
        &gradient.log_weight,
        "grad_log_weight",
    )?;
    let (mut gd, mut gm, mut ga, mut gb) = (grad_dispersion, grad_mean, grad_alpha, grad_beta);
    write(&mut gd, &gradient.dispersion, "grad_dispersion")?;
    write(&mut gm, &gradient.mean, "grad_mean")?;
    write(&mut ga, &gradient.alpha, "grad_alpha")?;
    write(&mut gb, &gradient.beta, "grad_beta")?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_rising_tables_are_the_sums_they_name() {
        let tables = Rising::new(&[2.5], 4);
        let exact: f64 = (0..4).map(|j| (1.0 + j as f64 / 2.5).ln()).sum();
        assert!((tables.log[4] - exact).abs() < 1e-15);
        let digamma: f64 = (0..4).map(|j| 1.0 / (2.5 + j as f64)).sum();
        assert!((tables.reciprocal[4] - digamma).abs() < 1e-15);
    }

    #[test]
    fn a_single_component_has_all_the_responsibility() {
        let counts = [3_u32, 0, 7];
        let totals = Totals {
            counts: &counts,
            dispersion: &[4.0],
            mean: &[5.0],
        };
        let (_, gradient) = value_and_gradient(&[0.0], Some(&totals), None).unwrap();
        assert!((gradient.log_weight[0] - 3.0).abs() < 1e-12);
    }
}
