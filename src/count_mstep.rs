//! The count families' M-step solves, ported from
//! `python/snakes_and_ladders/emissions.py` (the batched `torch` solves, the
//! oracle) to Rust, exposed to Python via PyO3 as
//! `snakes_and_ladders.oxi_snakes_and_ladders.negative_binomial_dispersions`
//! and `...beta_binomial_parameters` (issue #922).
//!
//! **The digamma differences are sums of reciprocals.** Every score term the
//! two families' M steps evaluate is `sum_u w_u (digamma(u + x) - digamma(x))`
//! over integer counts `u`, and for an integer `u` that difference is
//! `sum_{j < u} 1 / (x + j)`. Exchanging the sums gives
//! `sum_j T_j / (x + j)` with `T_j = sum_{u > j} w_u`, the tail of the
//! weights: one pass over `0 .. max u` with a reciprocal per step and no
//! special function. The tails are built by the caller from the same
//! histograms the oracle reads, so this crate needs no special-function
//! dependency and the oracle's `digamma` is replaced by an identity, not an
//! approximation. The two sum in different orders, so the kernel is pinned
//! to the oracle at a tolerance rather than bitwise.
//!
//! **Within one component the loop is the oracle's:** the same brackets, the
//! same bisection steps, the same stops. The components are solved in
//! sequence and not over `rayon`: one M step is milliseconds, and its caller
//! is already parallel one level up --- the starts notebooks run four worker
//! processes, each of which would otherwise start a pool the size of the host.
//!
//! The implementations are plain Rust with no PyO3 types so `cargo test` and
//! `benches/` can link them, per `src/pruning.rs`'s module docs.

use numpy::{PyReadonlyArray1, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// `sum_j tails[j] / (x + j)`: the weighted digamma difference at `x`.
#[inline]
pub fn rising(tails: &[f64], x: f64) -> f64 {
    let mut sum = 0.0;
    for (j, &tail) in tails.iter().enumerate() {
        sum += tail / (x + j as f64);
    }
    sum
}

/// One negative-binomial state's dispersion solve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dispersion {
    /// The solved `r`, or the bracket end it stopped at.
    pub value: f64,
    /// Whether the score kept its sign across the bracket.
    pub at_boundary: bool,
    /// Bisection steps taken.
    pub iterations: u32,
    /// The score at `value`, divided by the state's weight.
    pub residual: f64,
}

/// The profiled score `sum_j T_j / (r + j) + W log(r / (r + mu))`.
#[inline]
fn dispersion_score(tails: &[f64], total: f64, mean: f64, r: f64) -> f64 {
    rising(tails, r) + total * (r / (r + mean)).ln()
}

/// Bisection on `log r` inside `[lower, upper]`, as `_solve_dispersion`.
pub fn solve_dispersion(
    tails: &[f64],
    total: f64,
    mean: f64,
    lower: f64,
    upper: f64,
    tolerance: f64,
) -> Dispersion {
    if dispersion_score(tails, total, mean, upper) > 0.0 {
        return Dispersion {
            value: upper,
            at_boundary: true,
            iterations: 0,
            residual: 0.0,
        };
    }
    if dispersion_score(tails, total, mean, lower) < 0.0 {
        return Dispersion {
            value: lower,
            at_boundary: true,
            iterations: 0,
            residual: 0.0,
        };
    }
    let (mut low, mut high) = (lower.ln(), upper.ln());
    let mut iterations = 0;
    while high - low > tolerance {
        let middle = 0.5 * (low + high);
        if dispersion_score(tails, total, mean, middle.exp()) > 0.0 {
            low = middle;
        } else {
            high = middle;
        }
        iterations += 1;
    }
    let value = (0.5 * (low + high)).exp();
    Dispersion {
        value,
        at_boundary: false,
        iterations,
        residual: dispersion_score(tails, total, mean, value).abs() / total,
    }
}

/// One state's data under a per-observation exposure (issue #933).
#[derive(Clone, Copy, Debug)]
pub struct Exposed<'a> {
    /// Tails of the state's count weights: the digamma half, as without one.
    pub tails: &'a [f64],
    /// The state's weight per observation.
    pub weights: &'a [f64],
    /// The exposure per observation.
    pub exposures: &'a [f64],
    /// The count per observation.
    pub counts: &'a [f64],
    /// The state's mean per unit exposure.
    pub mean: f64,
    /// The state's summed weight.
    pub total: f64,
    /// Whether the exposure varies, which keeps the term profiling drops.
    pub varying: bool,
}

/// The score in `r` at rates `e_t mu`: the tails' digamma half, then
/// `sum_t w_t log(r / (r + e_t mu))` and, where the exposure varies,
/// `sum_t w_t (e_t mu - y_t) / (r + e_t mu)` --- `_weighted_dispersion_score`
/// term for term.
fn exposed_score(problem: Exposed<'_>, r: f64) -> f64 {
    let mut logs = 0.0;
    let mut drift = 0.0;
    for t in 0..problem.weights.len() {
        let rate = problem.exposures[t] * problem.mean;
        let weight = problem.weights[t];
        logs += weight * (r / (r + rate)).ln();
        if problem.varying {
            drift += weight * (rate - problem.counts[t]) / (r + rate);
        }
    }
    rising(problem.tails, r) + logs + drift
}

/// Bisection on `log r` inside `[lower, upper]` at per-observation rates,
/// as `_solve_dispersion` under an exposure.
pub fn solve_exposed_dispersion(
    problem: Exposed<'_>,
    lower: f64,
    upper: f64,
    tolerance: f64,
) -> Dispersion {
    let score = |r: f64| exposed_score(problem, r);
    if score(upper) > 0.0 {
        return Dispersion {
            value: upper,
            at_boundary: true,
            iterations: 0,
            residual: 0.0,
        };
    }
    if score(lower) < 0.0 {
        return Dispersion {
            value: lower,
            at_boundary: true,
            iterations: 0,
            residual: 0.0,
        };
    }
    let (mut low, mut high) = (lower.ln(), upper.ln());
    let mut iterations = 0;
    while high - low > tolerance {
        let middle = 0.5 * (low + high);
        if score(middle.exp()) > 0.0 {
            low = middle;
        } else {
            high = middle;
        }
        iterations += 1;
    }
    let value = (0.5 * (low + high)).exp();
    Dispersion {
        value,
        at_boundary: false,
        iterations,
        residual: score(value).abs() / problem.total,
    }
}

/// One beta-binomial state's `(a, b)` solve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BetaBinomial {
    /// `a`.
    pub alpha: f64,
    /// `b`.
    pub beta: f64,
    /// Whether the concentration stopped at its identifiable bound.
    pub at_boundary: bool,
    /// Whether the last outer step moved less than the tolerance.
    pub converged: bool,
    /// Outer iterations taken.
    pub iterations: u32,
    /// The last outer step's largest relative move.
    pub residual: f64,
}

/// One state's tails and bracket for [`solve_beta_binomial`].
#[derive(Clone, Copy, Debug)]
pub struct BetaBinomialProblem<'a> {
    /// Tails of the success counts' weights.
    pub success: &'a [f64],
    /// Tails of the failure counts' weights.
    pub failure: &'a [f64],
    /// Tails of the trial counts' weights.
    pub depth: &'a [f64],
    /// The state's summed weight.
    pub total: f64,
    /// The starting mean rate `a / (a + b)`.
    pub rate: f64,
    /// The starting concentration, already capped at the bound.
    pub concentration: f64,
    /// The identifiable concentration bound.
    pub bound: f64,
    /// `log` of the concentration bracket's lower end.
    pub log_low: f64,
    /// `log` of the bound.
    pub log_high: f64,
}

/// The bisection settings the oracle uses.
#[derive(Clone, Copy, Debug)]
pub struct Bisection {
    /// Stop when a bracket or an outer move is at most this.
    pub tolerance: f64,
    /// Outer alternations at most.
    pub max_iterations: u32,
    /// Bisection steps per inner solve at most.
    pub max_bisections: u32,
    /// The rate bracket's distance from 0 and 1.
    pub margin: f64,
}

fn bisect(mut low: f64, mut high: f64, settings: Bisection, score: impl Fn(f64) -> f64) -> f64 {
    for _ in 0..settings.max_bisections {
        if high - low <= settings.tolerance {
            break;
        }
        let middle = 0.5 * (low + high);
        if score(middle) > 0.0 {
            low = middle;
        } else {
            high = middle;
        }
    }
    0.5 * (low + high)
}

/// The alternating bisection of `_solve_beta_binomial_batched`, one state.
pub fn solve_beta_binomial(problem: BetaBinomialProblem<'_>, settings: Bisection) -> BetaBinomial {
    let p = problem;
    let rate_score = |rate: f64, held: f64| {
        rising(p.success, rate * held) - rising(p.failure, (1.0 - rate) * held)
    };
    let concentration_score = |held: f64, total: f64| {
        held * rising(p.success, held * total)
            + (1.0 - held) * rising(p.failure, (1.0 - held) * total)
            - rising(p.depth, total)
    };
    let (mut rate, mut concentration) = (p.rate, p.concentration);
    let mut at_boundary = false;
    let mut residual = f64::INFINITY;
    let mut iterations = 0;
    for _ in 0..settings.max_iterations {
        iterations += 1;
        let (previous_rate, previous_concentration) = (rate, concentration);
        let held = concentration;
        rate = bisect(settings.margin, 1.0 - settings.margin, settings, |r| {
            rate_score(r, held)
        });
        let pinned = concentration_score(rate, p.bound) > 0.0;
        concentration = if pinned {
            p.bound
        } else {
            bisect(p.log_low, p.log_high, settings, |log_total| {
                concentration_score(rate, log_total.exp())
            })
            .exp()
        };
        at_boundary = pinned;
        residual = (rate - previous_rate)
            .abs()
            .max((concentration - previous_concentration).abs() / concentration);
        if residual <= settings.tolerance {
            break;
        }
    }
    BetaBinomial {
        alpha: rate * concentration,
        beta: (1.0 - rate) * concentration,
        at_boundary,
        converged: residual <= settings.tolerance,
        iterations,
        residual,
    }
}

fn borrowed<'a, T: numpy::Element>(
    array: &'a PyReadonlyArray1<'_, T>,
    name: &str,
) -> PyResult<&'a [T]> {
    array.as_slice().map_err(|_| {
        PyValueError::new_err(format!("{name} must be C-contiguous to cross the boundary"))
    })
}

fn written<'a, T: numpy::Element>(
    array: &'a mut PyReadwriteArray1<'_, T>,
    name: &str,
) -> PyResult<&'a mut [T]> {
    array
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err(format!("{name} must be C-contiguous")))
}

/// Every state's dispersion, written into the caller's arrays.
///
/// `tails` is `(K, J)` row-major; `total`, `mean`, `lower` and `upper` are
/// `(K,)`. Results: the dispersion, `1` where the solve stopped at a bracket
/// end, the bisection steps and the residual, each `(K,)`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn negative_binomial_dispersions(
    py: Python<'_>,
    tails: PyReadonlyArray1<'_, f64>,
    total: PyReadonlyArray1<'_, f64>,
    mean: PyReadonlyArray1<'_, f64>,
    lower: PyReadonlyArray1<'_, f64>,
    upper: PyReadonlyArray1<'_, f64>,
    tolerance: f64,
    mut value: PyReadwriteArray1<'_, f64>,
    mut at_boundary: PyReadwriteArray1<'_, u8>,
    mut iterations: PyReadwriteArray1<'_, u32>,
    mut residual: PyReadwriteArray1<'_, f64>,
) -> PyResult<()> {
    let tails = borrowed(&tails, "tails")?;
    let total = borrowed(&total, "total")?;
    let mean = borrowed(&mean, "mean")?;
    let lower = borrowed(&lower, "lower")?;
    let upper = borrowed(&upper, "upper")?;
    let k = total.len();
    if k == 0 || tails.len() % k != 0 {
        return Err(PyValueError::new_err(
            "tails must be (K, J) with K = len(total)",
        ));
    }
    let width = tails.len() / k;
    let value = written(&mut value, "value")?;
    let at_boundary = written(&mut at_boundary, "at_boundary")?;
    let iterations = written(&mut iterations, "iterations")?;
    let residual = written(&mut residual, "residual")?;
    py.detach(|| {
        let solved: Vec<Dispersion> = (0..k)
            .map(|state| {
                solve_dispersion(
                    &tails[state * width..(state + 1) * width],
                    total[state],
                    mean[state],
                    lower[state],
                    upper[state],
                    tolerance,
                )
            })
            .collect();
        for (state, one) in solved.iter().enumerate() {
            value[state] = one.value;
            at_boundary[state] = u8::from(one.at_boundary);
            iterations[state] = one.iterations;
            residual[state] = one.residual;
        }
    });
    Ok(())
}

/// Every state's dispersion under a per-observation exposure (issue #933).
///
/// `tails` is `(K, J)` and `weights` `(K, n)`, row-major; `exposures` and
/// `counts` are `(n,)`; `total`, `mean`, `lower` and `upper` are `(K,)`.
/// Results as [`negative_binomial_dispersions`]'s.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn negative_binomial_dispersions_exposed(
    py: Python<'_>,
    tails: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    exposures: PyReadonlyArray1<'_, f64>,
    counts: PyReadonlyArray1<'_, f64>,
    total: PyReadonlyArray1<'_, f64>,
    mean: PyReadonlyArray1<'_, f64>,
    lower: PyReadonlyArray1<'_, f64>,
    upper: PyReadonlyArray1<'_, f64>,
    tolerance: f64,
    mut value: PyReadwriteArray1<'_, f64>,
    mut at_boundary: PyReadwriteArray1<'_, u8>,
    mut iterations: PyReadwriteArray1<'_, u32>,
    mut residual: PyReadwriteArray1<'_, f64>,
) -> PyResult<()> {
    let tails = borrowed(&tails, "tails")?;
    let weights = borrowed(&weights, "weights")?;
    let exposures = borrowed(&exposures, "exposures")?;
    let counts = borrowed(&counts, "counts")?;
    let total = borrowed(&total, "total")?;
    let mean = borrowed(&mean, "mean")?;
    let lower = borrowed(&lower, "lower")?;
    let upper = borrowed(&upper, "upper")?;
    let (k, n) = (total.len(), exposures.len());
    if k == 0 || tails.len() % k != 0 || weights.len() != k * n || counts.len() != n {
        return Err(PyValueError::new_err(
            "tails must be (K, J) and weights (K, n), with n exposures and counts",
        ));
    }
    let width = tails.len() / k;
    let varying = exposures.iter().any(|&e| e != exposures[0]);
    let value = written(&mut value, "value")?;
    let at_boundary = written(&mut at_boundary, "at_boundary")?;
    let iterations = written(&mut iterations, "iterations")?;
    let residual = written(&mut residual, "residual")?;
    py.detach(|| {
        for state in 0..k {
            let one = solve_exposed_dispersion(
                Exposed {
                    tails: &tails[state * width..(state + 1) * width],
                    weights: &weights[state * n..(state + 1) * n],
                    exposures,
                    counts,
                    mean: mean[state],
                    total: total[state],
                    varying,
                },
                lower[state],
                upper[state],
                tolerance,
            );
            value[state] = one.value;
            at_boundary[state] = u8::from(one.at_boundary);
            iterations[state] = one.iterations;
            residual[state] = one.residual;
        }
    });
    Ok(())
}

/// Every state's `(a, b)`, written into the caller's arrays.
///
/// The three tails are `(K, J_s)`, `(K, J_f)` and `(K, J_d)` row-major; the
/// rest are `(K,)`. `out` is `(K, 6)`: alpha, beta, at boundary, converged,
/// iterations, residual.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn beta_binomial_parameters(
    py: Python<'_>,
    success: PyReadonlyArray1<'_, f64>,
    failure: PyReadonlyArray1<'_, f64>,
    depth: PyReadonlyArray1<'_, f64>,
    total: PyReadonlyArray1<'_, f64>,
    rate: PyReadonlyArray1<'_, f64>,
    concentration: PyReadonlyArray1<'_, f64>,
    bound: PyReadonlyArray1<'_, f64>,
    log_low: PyReadonlyArray1<'_, f64>,
    tolerance: f64,
    max_iterations: u32,
    max_bisections: u32,
    margin: f64,
    mut out: PyReadwriteArray1<'_, f64>,
) -> PyResult<()> {
    let success = borrowed(&success, "success")?;
    let failure = borrowed(&failure, "failure")?;
    let depth = borrowed(&depth, "depth")?;
    let total = borrowed(&total, "total")?;
    let rate = borrowed(&rate, "rate")?;
    let concentration = borrowed(&concentration, "concentration")?;
    let bound = borrowed(&bound, "bound")?;
    let log_low = borrowed(&log_low, "log_low")?;
    let k = total.len();
    let widths = [success.len(), failure.len(), depth.len()];
    if k == 0 || widths.iter().any(|w| w % k != 0) {
        return Err(PyValueError::new_err("each tails array must be (K, J)"));
    }
    let [ws, wf, wd] = widths.map(|w| w / k);
    let out = written(&mut out, "out")?;
    if out.len() != 6 * k {
        return Err(PyValueError::new_err("out must be (K, 6)"));
    }
    let settings = Bisection {
        tolerance,
        max_iterations,
        max_bisections,
        margin,
    };
    py.detach(|| {
        out.chunks_mut(6).enumerate().for_each(|(state, row)| {
            let one = solve_beta_binomial(
                BetaBinomialProblem {
                    success: &success[state * ws..(state + 1) * ws],
                    failure: &failure[state * wf..(state + 1) * wf],
                    depth: &depth[state * wd..(state + 1) * wd],
                    total: total[state],
                    rate: rate[state],
                    concentration: concentration[state],
                    bound: bound[state],
                    log_low: log_low[state],
                    log_high: bound[state].ln(),
                },
                settings,
            );
            row[0] = one.alpha;
            row[1] = one.beta;
            row[2] = f64::from(u8::from(one.at_boundary));
            row[3] = f64::from(u8::from(one.converged));
            row[4] = f64::from(one.iterations);
            row[5] = one.residual;
        });
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `digamma(u + x) - digamma(x)` by the recurrence, against the sum of
    /// reciprocals written out term by term.
    #[test]
    fn a_point_mass_tail_is_the_reciprocal_sum() {
        // Weight 1 at u = 3: tails are 1 for j < 3.
        let tails = [1.0, 1.0, 1.0];
        let x = 0.7;
        let expected = 1.0 / x + 1.0 / (x + 1.0) + 1.0 / (x + 2.0);
        assert!((rising(&tails, x) - expected).abs() < 1e-15);
    }

    #[test]
    fn a_state_with_no_spread_stops_at_the_upper_bound() {
        // All mass at one count: no overdispersion, so the score is positive
        // everywhere and the solve reports the bound.
        let tails: Vec<f64> = vec![10.0; 4];
        let solved = solve_dispersion(&tails, 10.0, 4.0, 1e-6, 50.0, 1e-12);
        assert!(solved.at_boundary);
        assert_eq!(solved.value, 50.0);
    }
}
