//! One Baum--Welch step for a categorical HMM, streamed sequence by sequence (issue #986).
//!
//! `opt.hmm.baum_welch_family` batches the E step over every sequence at
//! once in log space, so it holds `alpha`, `beta` and `gamma` over every
//! position and `xi` over every pair: 62.6 MB at 10^5 positions of three
//! states, where hmmlearn's fit peaked at 0.37 MB (#987). The M step of a
//! categorical family needs none of that, only expected counts, so this
//! accumulates them as it goes:
//!
//! - **Scaled messages in probability space** (Rabiner 1989, section V.A).
//!   The forward pass normalizes each position's `alpha` to sum to one and
//!   keeps the normalizer `c_t`, so `ln P(x) = sum_t ln c_t` and nothing
//!   underflows at any length; the backward pass scales `beta` by the same
//!   `c_t`, so `alpha_hat_t * beta_hat_t` is the posterior as it stands.
//! - **One sequence's forward messages, one backward vector.** The backward
//!   pass adds each position's posterior and each pair's to the counts and
//!   discards them, so the working set is `length * m` numbers for the
//!   sequence in hand, not `n_positions * m * m` for the batch.
//!
//! The M step is `baum_welch_family`'s: the initial distribution is the mean
//! first posterior, the transition rows normalized expected pair counts, and
//! the emission rows normalized expected symbol counts with the smallest
//! normal `f64` added, as `CategoricalEmission.reestimate` adds
//! `torch.finfo(float64).tiny`. The Python route is the oracle that pins it.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// New log parameters and the log-likelihood at the parameters given.
pub struct Step {
    pub log_initial: Vec<f64>,
    pub log_transition: Vec<f64>,
    pub log_emission: Vec<f64>,
    pub log_likelihood: f64,
}

/// One E step and one M step over `n_sequences` rows of `length` symbols.
///
/// `observations` is row-major `n_sequences * length`; `log_transition` is
/// `m * m` and `log_emission` `m * n_symbols`, both row-major.
pub fn categorical_step(
    observations: &[i64],
    length: usize,
    log_initial: &[f64],
    log_transition: &[f64],
    log_emission: &[f64],
) -> Result<Step, String> {
    let m = log_initial.len();
    if m == 0 || log_transition.len() != m * m || !log_emission.len().is_multiple_of(m) {
        return Err(format!(
            "{m} states need a {m}x{m} transition and an {m}-row emission, got {} and {} entries",
            log_transition.len(),
            log_emission.len()
        ));
    }
    let n_symbols = log_emission.len() / m;
    if length == 0 || !observations.len().is_multiple_of(length) {
        return Err(format!(
            "{} observations are not rows of {length}",
            observations.len()
        ));
    }
    if let Some(&bad) = observations
        .iter()
        .find(|&&o| o < 0 || o as usize >= n_symbols)
    {
        return Err(format!("symbol {bad} outside [0, {n_symbols})"));
    }
    let initial: Vec<f64> = log_initial.iter().map(|v| v.exp()).collect();
    let transition: Vec<f64> = log_transition.iter().map(|v| v.exp()).collect();
    // Column-major emission, so one symbol's `m` probabilities are contiguous.
    let mut emission = vec![0.0; m * n_symbols];
    for state in 0..m {
        for symbol in 0..n_symbols {
            emission[symbol * m + state] = log_emission[state * n_symbols + symbol].exp();
        }
    }

    let mut first = vec![0.0; m];
    let mut pairs = vec![0.0; m * m];
    let mut symbols = vec![0.0; m * n_symbols];
    let mut alpha = vec![0.0; length * m];
    let mut scale = vec![0.0; length];
    let mut beta = vec![1.0; m];
    let mut onward = vec![0.0; m];
    let mut log_likelihood = 0.0;

    for row in observations.chunks_exact(length) {
        // Forward, each position normalized to sum to one.
        let b0 = &emission[row[0] as usize * m..][..m];
        let mut total = 0.0;
        for state in 0..m {
            alpha[state] = initial[state] * b0[state];
            total += alpha[state];
        }
        scale[0] = total;
        for value in &mut alpha[..m] {
            *value /= total;
        }
        for t in 1..length {
            let b = &emission[row[t] as usize * m..][..m];
            let (done, rest) = alpha.split_at_mut(t * m);
            let previous = &done[(t - 1) * m..];
            let current = &mut rest[..m];
            let mut total = 0.0;
            for next in 0..m {
                let mut reach = 0.0;
                for state in 0..m {
                    reach += previous[state] * transition[state * m + next];
                }
                current[next] = reach * b[next];
                total += current[next];
            }
            scale[t] = total;
            for value in current.iter_mut() {
                *value /= total;
            }
        }
        log_likelihood += scale.iter().map(|c| c.ln()).sum::<f64>();

        // Backward, adding each posterior and pair posterior as it forms.
        beta.fill(1.0);
        for t in (0..length).rev() {
            let a = &alpha[t * m..][..m];
            let symbol = row[t] as usize;
            for state in 0..m {
                symbols[state * n_symbols + symbol] += a[state] * beta[state];
            }
            if t == 0 {
                for state in 0..m {
                    first[state] += a[state] * beta[state];
                }
                break;
            }
            // Pairs (t - 1, t), then beta at t - 1.
            let b = &emission[symbol * m..][..m];
            let c = scale[t];
            for next in 0..m {
                onward[next] = b[next] * beta[next] / c;
            }
            let previous = &alpha[(t - 1) * m..][..m];
            for state in 0..m {
                let mut back = 0.0;
                for next in 0..m {
                    let through = transition[state * m + next] * onward[next];
                    pairs[state * m + next] += previous[state] * through;
                    back += through;
                }
                beta[state] = back;
            }
        }
    }

    let n_sequences = (observations.len() / length) as f64;
    let log_initial = first.iter().map(|&g| g.ln() - n_sequences.ln()).collect();
    let normalized = |counts: &[f64], width: usize, floor: f64| -> Vec<f64> {
        counts
            .chunks_exact(width)
            .flat_map(|row| {
                let logs: Vec<f64> = row.iter().map(|&c| (c + floor).ln()).collect();
                let high = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let total = if high == f64::NEG_INFINITY {
                    f64::NEG_INFINITY
                } else {
                    high + logs.iter().map(|&l| (l - high).exp()).sum::<f64>().ln()
                };
                logs.into_iter().map(move |l| l - total)
            })
            .collect()
    };
    Ok(Step {
        log_initial,
        log_transition: normalized(&pairs, m, 0.0),
        log_emission: normalized(&symbols, n_symbols, f64::MIN_POSITIVE),
        log_likelihood,
    })
}

/// One categorical Baum--Welch step; see the module docs.
///
/// Returns `(log_initial, log_transition, log_emission, log_likelihood)`,
/// the matrices flattened row-major and the log-likelihood at the
/// parameters given.
#[pyfunction]
#[pyo3(signature = (observations, log_initial, log_transition, log_emission))]
#[allow(clippy::type_complexity)]
pub fn categorical_em_step<'py>(
    py: Python<'py>,
    observations: PyReadonlyArray2<'py, i64>,
    log_initial: PyReadonlyArray1<'py, f64>,
    log_transition: PyReadonlyArray1<'py, f64>,
    log_emission: PyReadonlyArray1<'py, f64>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    f64,
)> {
    let length = observations.shape()[1];
    let observations = observations.as_slice()?;
    let (log_initial, log_transition, log_emission) = (
        log_initial.as_slice()?,
        log_transition.as_slice()?,
        log_emission.as_slice()?,
    );
    let step = py
        .detach(|| {
            categorical_step(
                observations,
                length,
                log_initial,
                log_transition,
                log_emission,
            )
        })
        .map_err(PyValueError::new_err)?;
    Ok((
        PyArray1::from_vec(py, step.log_initial),
        PyArray1::from_vec(py, step.log_transition),
        PyArray1::from_vec(py, step.log_emission),
        step.log_likelihood,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force posteriors by enumerating every path of a short chain.
    #[test]
    fn the_step_is_the_enumerated_expectation() {
        let (m, k, length) = (2usize, 3usize, 4usize);
        let observations = [0i64, 2, 1, 1, 2, 2, 0, 1];
        let initial = [0.6f64, 0.4];
        let transition = [0.7f64, 0.3, 0.2, 0.8];
        let emission = [0.5f64, 0.3, 0.2, 0.1, 0.4, 0.5];
        let ln = |v: &[f64]| v.iter().map(|x| x.ln()).collect::<Vec<_>>();
        let step = categorical_step(
            &observations,
            length,
            &ln(&initial),
            &ln(&transition),
            &ln(&emission),
        )
        .unwrap();
        let mut ll = 0.0;
        let mut first = [0.0; 2];
        let mut pairs = [0.0; 4];
        let mut symbols = [0.0; 6];
        for row in observations.chunks_exact(length) {
            let mut evidence = 0.0;
            let mut weighted = Vec::new();
            for path in 0..m.pow(length as u32) {
                let states: Vec<usize> = (0..length).map(|t| (path >> t) & 1).collect();
                let mut p = initial[states[0]] * emission[states[0] * k + row[0] as usize];
                for t in 1..length {
                    p *= transition[states[t - 1] * m + states[t]]
                        * emission[states[t] * k + row[t] as usize];
                }
                evidence += p;
                weighted.push((states, p));
            }
            ll += evidence.ln();
            for (states, p) in weighted {
                let w = p / evidence;
                first[states[0]] += w;
                for t in 0..length {
                    symbols[states[t] * k + row[t] as usize] += w;
                    if t > 0 {
                        pairs[states[t - 1] * m + states[t]] += w;
                    }
                }
            }
        }
        assert!((step.log_likelihood - ll).abs() < 1e-12);
        for s in 0..m {
            assert!((step.log_initial[s] - (first[s] / 2.0).ln()).abs() < 1e-12);
            let row: f64 = pairs[s * m..][..m].iter().sum();
            for n in 0..m {
                assert!(
                    (step.log_transition[s * m + n] - (pairs[s * m + n] / row).ln()).abs() < 1e-12
                );
            }
            let row: f64 = symbols[s * k..][..k].iter().sum();
            for o in 0..k {
                assert!(
                    (step.log_emission[s * k + o] - (symbols[s * k + o] / row).ln()).abs() < 1e-12
                );
            }
        }
    }

    #[test]
    fn a_symbol_outside_the_alphabet_is_refused() {
        let ln2 = 0.5f64.ln();
        assert!(categorical_step(&[0, 3], 2, &[ln2, ln2], &[ln2; 4], &[ln2; 4]).is_err());
    }
}
