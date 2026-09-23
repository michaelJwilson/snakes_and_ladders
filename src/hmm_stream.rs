//! One Baum--Welch step for a categorical, Gaussian or count HMM, streamed sequence by sequence (issues #986, #997).
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
use rayon::prelude::*;

use crate::special::{ln_gamma, ln_gamma_approx, STIRLING_FROM};

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

/// What an E step adds up besides the chain counts: moments or a histogram.
///
/// One is made per block of sequences and the blocks' are merged in block
/// order, so the sums are the same whatever the number of threads.
pub trait Statistics: Send {
    fn add(&mut self, index: usize, posterior: &[f64]);
    fn merge(&mut self, other: Self);
}

/// Expected chain counts from one streamed E step.
struct Counts {
    first: Vec<f64>,
    pairs: Vec<f64>,
    log_likelihood: f64,
}

impl Counts {
    fn merge(&mut self, other: Counts) {
        for (a, b) in self.first.iter_mut().zip(other.first) {
            *a += b;
        }
        for (a, b) in self.pairs.iter_mut().zip(other.pairs) {
            *a += b;
        }
        self.log_likelihood += other.log_likelihood;
    }
}

/// The most blocks of sequences an E step is split into: fixed, so the
/// partition and with it every sum depend on the data alone.
const BLOCKS: usize = 16;

/// The bytes the blocks' statistics may take together before fewer are used.
const BLOCK_BYTES: usize = 4 << 20;

/// One streamed E step over any per-position density, over sequence blocks.
///
/// `log_density(index, out)` writes the `m` log-densities of the observation
/// at flat `index`; each block's `S` receives every posterior. Each
/// position's densities are shifted by their maximum before they are
/// exponentiated and the shift is added back to the log-likelihood, so an
/// observation far from every state underflows nothing. `key(index)` names
/// what the density depends on --- the cell of a count, `usize::MAX` for "no
/// two alike" --- and a position whose key is the previous one's copies its
/// densities rather than scoring again: counts repeat along a sticky chain.
/// The sequences are
/// independent given the parameters, so blocks run on `rayon` and are
/// merged in order; `statistics_bytes` is one block's `S`, which sets how
/// many blocks the memory allows.
#[allow(clippy::too_many_arguments)]
fn stream_counts<S: Statistics>(
    n_positions: usize,
    length: usize,
    log_initial: &[f64],
    log_transition: &[f64],
    log_density: impl Fn(usize, &mut [f64]) + Sync,
    key: impl Fn(usize) -> usize + Sync,
    fresh: impl Fn() -> S + Sync,
    statistics_bytes: usize,
) -> Result<(Counts, S), String> {
    let m = log_initial.len();
    if m == 0 || log_transition.len() != m * m {
        return Err(format!(
            "{m} states need a {m}x{m} transition, got {} entries",
            log_transition.len()
        ));
    }
    if length == 0 || !n_positions.is_multiple_of(length) {
        return Err(format!(
            "{n_positions} observations are not rows of {length}"
        ));
    }
    let n_sequences = n_positions / length;
    let blocks = BLOCKS
        .min(n_sequences)
        .min((BLOCK_BYTES / statistics_bytes.max(1)).max(1))
        .max(1);
    let initial: Vec<f64> = log_initial.iter().map(|v| v.exp()).collect();
    let transition: Vec<f64> = log_transition.iter().map(|v| v.exp()).collect();
    let block = |b: usize| -> (Counts, S) {
        let sequences = (b * n_sequences / blocks)..((b + 1) * n_sequences / blocks);
        let mut statistics = fresh();
        let counts = stream_block(
            sequences,
            length,
            &initial,
            &transition,
            &log_density,
            &key,
            &mut statistics,
        );
        (counts, statistics)
    };
    let mut parts: Vec<(Counts, S)> = (0..blocks).into_par_iter().map(block).collect();
    let (mut counts, mut statistics) = parts.remove(0);
    for (more, extra) in parts {
        counts.merge(more);
        statistics.merge(extra);
    }
    Ok((counts, statistics))
}

/// The forward--backward pass over a range of sequences, in order.
fn stream_block<S: Statistics>(
    sequences: std::ops::Range<usize>,
    length: usize,
    initial: &[f64],
    transition: &[f64],
    log_density: &impl Fn(usize, &mut [f64]),
    key: &impl Fn(usize) -> usize,
    statistics: &mut S,
) -> Counts {
    let m = initial.len();
    let mut first = vec![0.0; m];
    let mut pairs = vec![0.0; m * m];
    let mut alpha = vec![0.0; length * m];
    let mut emitted = vec![0.0; length * m];
    let mut scale = vec![0.0; length];
    let mut beta = vec![1.0; m];
    let mut onward = vec![0.0; m];
    let mut posterior = vec![0.0; m];
    let mut last = vec![0.0; m];
    let mut last_key = usize::MAX;
    let mut log_likelihood = 0.0;

    for sequence in sequences {
        let start = sequence * length;
        // Densities, shifted per position, and the shifts into the evidence.
        for t in 0..length {
            let b = &mut emitted[t * m..][..m];
            let here = key(start + t);
            if here != usize::MAX && here == last_key {
                b.copy_from_slice(&last);
            } else {
                log_density(start + t, b);
                last.copy_from_slice(b);
                last_key = here;
            }
            let high = b.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            for value in b.iter_mut() {
                *value = (*value - high).exp();
            }
            log_likelihood += high;
        }
        // Forward, each position normalized to sum to one.
        let mut total = 0.0;
        for state in 0..m {
            alpha[state] = initial[state] * emitted[state];
            total += alpha[state];
        }
        scale[0] = total;
        for value in &mut alpha[..m] {
            *value /= total;
        }
        for t in 1..length {
            let b = &emitted[t * m..][..m];
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

        // Backward, handing each posterior over and adding each pair's.
        beta.fill(1.0);
        for t in (0..length).rev() {
            let a = &alpha[t * m..][..m];
            for state in 0..m {
                posterior[state] = a[state] * beta[state];
            }
            statistics.add(start + t, &posterior);
            if t == 0 {
                for state in 0..m {
                    first[state] += posterior[state];
                }
                break;
            }
            let b = &emitted[t * m..][..m];
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
    Counts {
        first,
        pairs,
        log_likelihood,
    }
}

/// The initial and transition M step from expected counts: the mean first
/// posterior and the normalized pair counts, in logs.
fn chain_m_step(counts: &Counts, n_sequences: usize) -> (Vec<f64>, Vec<f64>) {
    let m = counts.first.len();
    let n = (n_sequences as f64).ln();
    let log_initial = counts.first.iter().map(|&g| g.ln() - n).collect();
    let log_transition = counts
        .pairs
        .chunks_exact(m)
        .flat_map(|row| {
            let total: f64 = row.iter().sum();
            row.iter().map(move |&c| (c / total).ln())
        })
        .collect();
    (log_initial, log_transition)
}

/// One Gaussian step: chain parameters, the family's, and the
/// log-likelihood at the parameters given.
pub struct FamilyStep {
    pub log_initial: Vec<f64>,
    pub log_transition: Vec<f64>,
    /// The means.
    pub mean: Vec<f64>,
    /// The variances.
    pub variance: Vec<f64>,
    pub log_likelihood: f64,
}

/// Posterior moments about each state's old mean: `S0`, `S1`, `S2` per state.
struct Moments<'a> {
    observations: &'a [f64],
    mean: &'a [f64],
    sums: Vec<f64>,
}

impl Statistics for Moments<'_> {
    #[inline]
    fn add(&mut self, index: usize, posterior: &[f64]) {
        let x = self.observations[index];
        for (state, &w) in posterior.iter().enumerate() {
            let d = x - self.mean[state];
            self.sums[3 * state] += w;
            self.sums[3 * state + 1] += w * d;
            self.sums[3 * state + 2] += w * d * d;
        }
    }

    fn merge(&mut self, other: Self) {
        for (a, b) in self.sums.iter_mut().zip(other.sums) {
            *a += b;
        }
    }
}

/// One streamed step of a one-channel Gaussian HMM.
///
/// The variance is the posterior-weighted second moment about the new mean,
/// accumulated about the old one so the one pass cancels nothing at the
/// scale of the data: with `d = x - mu_old`, `mu = mu_old + S1 / S0` and
/// `var = S2 / S0 - (S1 / S0)^2`. The floor is the caller's to apply.
pub fn gaussian_step(
    observations: &[f64],
    length: usize,
    log_initial: &[f64],
    log_transition: &[f64],
    mean: &[f64],
    scale: &[f64],
) -> Result<FamilyStep, String> {
    let m = log_initial.len();
    if mean.len() != m || scale.len() != m {
        return Err(format!(
            "{m} states need {m} means and scales, got {} and {}",
            mean.len(),
            scale.len()
        ));
    }
    let offset: Vec<f64> = scale
        .iter()
        .map(|s| -0.5 * (2.0 * std::f64::consts::PI).ln() - s.ln())
        .collect();
    let (counts, moments) = stream_counts(
        observations.len(),
        length,
        log_initial,
        log_transition,
        |index, out| {
            let x = observations[index];
            for state in 0..m {
                let z = (x - mean[state]) / scale[state];
                out[state] = offset[state] - 0.5 * z * z;
            }
        },
        |_| usize::MAX,
        || Moments {
            observations,
            mean,
            sums: vec![0.0; 3 * m],
        },
        24 * m,
    )?;
    let (log_initial, log_transition) = chain_m_step(&counts, observations.len() / length);
    let (mut new_mean, mut variance) = (vec![0.0; m], vec![0.0; m]);
    for state in 0..m {
        let s = &moments.sums[3 * state..][..3];
        let shift = s[1] / s[0];
        new_mean[state] = mean[state] + shift;
        variance[state] = s[2] / s[0] - shift * shift;
    }
    Ok(FamilyStep {
        log_initial,
        log_transition,
        mean: new_mean,
        variance,
        log_likelihood: counts.log_likelihood,
    })
}

/// A Gaussian HMM's expected statistics at the parameters given: the
/// first-state posteriors summed over sequences (`m`), the expected pair
/// counts (`m * m`), per state `S0 = sum gamma`, `S1 = sum gamma (x - mu)`,
/// `S2 = sum gamma (x - mu)^2` (`3 * m`), and the log-likelihood.
///
/// By Fisher's identity the gradient of the log-likelihood is the posterior
/// expectation of the complete-data score, and these are every term of it
/// (issue #997): `GaussianHmmObjective.gradient` assembles it from them.
#[allow(clippy::type_complexity)]
pub fn gaussian_statistics(
    observations: &[f64],
    length: usize,
    log_initial: &[f64],
    log_transition: &[f64],
    mean: &[f64],
    scale: &[f64],
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, f64), String> {
    let m = log_initial.len();
    if mean.len() != m || scale.len() != m {
        return Err(format!(
            "{m} states need {m} means and scales, got {} and {}",
            mean.len(),
            scale.len()
        ));
    }
    let offset: Vec<f64> = scale
        .iter()
        .map(|s| -0.5 * (2.0 * std::f64::consts::PI).ln() - s.ln())
        .collect();
    let (counts, moments) = stream_counts(
        observations.len(),
        length,
        log_initial,
        log_transition,
        |index, out| {
            let x = observations[index];
            for state in 0..m {
                let z = (x - mean[state]) / scale[state];
                out[state] = offset[state] - 0.5 * z * z;
            }
        },
        |_| usize::MAX,
        || Moments {
            observations,
            mean,
            sums: vec![0.0; 3 * m],
        },
        24 * m,
    )?;
    Ok((
        counts.first,
        counts.pairs,
        moments.sums,
        counts.log_likelihood,
    ))
}

/// A Gaussian HMM's expected statistics; see `gaussian_statistics`.
///
/// Returns `(first, pairs, moments, log_likelihood)`, `pairs` row-major and
/// `moments` `(S0, S1, S2)` per state.
#[pyfunction]
#[pyo3(signature = (observations, log_initial, log_transition, mean, scale))]
#[allow(clippy::type_complexity)]
pub fn gaussian_hmm_statistics<'py>(
    py: Python<'py>,
    observations: PyReadonlyArray2<'py, f64>,
    log_initial: PyReadonlyArray1<'py, f64>,
    log_transition: PyReadonlyArray1<'py, f64>,
    mean: PyReadonlyArray1<'py, f64>,
    scale: PyReadonlyArray1<'py, f64>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    f64,
)> {
    let length = observations.shape()[1];
    let (observations, log_initial, log_transition, mean, scale) = (
        observations.as_slice()?,
        log_initial.as_slice()?,
        log_transition.as_slice()?,
        mean.as_slice()?,
        scale.as_slice()?,
    );
    let (first, pairs, moments, log_likelihood) = py
        .detach(|| {
            gaussian_statistics(
                observations,
                length,
                log_initial,
                log_transition,
                mean,
                scale,
            )
        })
        .map_err(PyValueError::new_err)?;
    Ok((
        PyArray1::from_vec(py, first),
        PyArray1::from_vec(py, pairs),
        PyArray1::from_vec(py, moments),
        log_likelihood,
    ))
}

/// Where one observation's row of the histogram is.
///
/// A count `y` with covariate `c` is the cell `y * stride + c` (`c = 0`,
/// `stride = 1` without a covariate), and `rows[cell]` is that cell's row:
/// only the cells the data occupies have one.
pub struct Cells<'a> {
    pub covariate: Option<&'a [i64]>,
    pub stride: usize,
    pub rows: &'a [u32],
}

impl Cells<'_> {
    #[inline]
    fn cell(&self, observations: &[i64], index: usize) -> usize {
        let c = self.covariate.map_or(0, |c| c[index] as usize);
        observations[index] as usize * self.stride + c
    }

    #[inline]
    fn row(&self, observations: &[i64], index: usize) -> usize {
        self.rows[self.cell(observations, index)] as usize
    }
}

/// The cells the data occupies, ascending, and how many observations each
/// holds: the rows a count step reads, and which of them repeat most.
pub fn occupied_cells(
    observations: &[i64],
    covariate: Option<&[i64]>,
    stride: usize,
) -> Result<(Vec<i64>, Vec<i64>), String> {
    if covariate.is_some_and(|c| c.len() != observations.len()) {
        return Err("the covariate is not one per observation".to_string());
    }
    if let Some(&v) = observations.iter().find(|v| **v < 0) {
        return Err(format!("count {v} is negative"));
    }
    if let Some(&v) = covariate.and_then(|c| c.iter().find(|v| **v < 0 || **v as usize >= stride)) {
        return Err(format!("covariate {v} outside [0, {stride})"));
    }
    let top = observations.iter().copied().max().unwrap_or(0) as usize;
    let mut seen = vec![0i64; (top + 1) * stride];
    let cells = Cells {
        covariate,
        stride,
        rows: &[],
    };
    for index in 0..observations.len() {
        seen[cells.cell(observations, index)] += 1;
    }
    Ok(seen
        .iter()
        .enumerate()
        .filter(|(_, &n)| n > 0)
        .map(|(cell, &n)| (cell as i64, n))
        .unzip())
}

/// Each state's posterior weight on each occupied cell, `n_rows * m`.
struct Histogram<'a> {
    observations: &'a [i64],
    cells: &'a Cells<'a>,
    m: usize,
    values: Vec<f64>,
}

impl Statistics for Histogram<'_> {
    #[inline]
    fn add(&mut self, index: usize, posterior: &[f64]) {
        let row = self.cells.row(self.observations, index);
        for (h, &w) in self.values[row * self.m..][..self.m]
            .iter_mut()
            .zip(posterior)
        {
            *h += w;
        }
    }

    fn merge(&mut self, other: Self) {
        for (a, b) in self.values.iter_mut().zip(other.values) {
            *a += b;
        }
    }
}

/// A count family's parameters, each per state, for scoring without a table.
///
/// Each log-density is the family's `log_density` term for term, with
/// `ln_gamma` for `torch.lgamma`. A covariate is the exposure a negative
/// binomial's rate is scaled by, or the trial count a binomial or
/// beta-binomial count is out of. `constants` holds, per state, every term
/// that does not depend on the observation, so each is evaluated once per
/// step rather than once per observation; under a covariate the terms that
/// depend on it are evaluated per observation.
pub struct CountFamily<'a> {
    kind: Kind<'a>,
    constants: Vec<f64>,
    /// Whether `ln_gamma_approx` scores the observations, and from where.
    approx: bool,
    stirling_from: f64,
}

/// `ln Gamma`, the Lanczos sum or with the Stirling series from 10.
#[inline]
fn lgamma<const APPROX: bool>(x: f64, from: f64) -> f64 {
    if APPROX {
        ln_gamma_approx(x, from)
    } else {
        ln_gamma(x)
    }
}

enum Kind<'a> {
    Poisson {
        rate: &'a [f64],
    },
    Binomial {
        trials: &'a [f64],
        probability: &'a [f64],
    },
    NegativeBinomial {
        dispersion: &'a [f64],
        mean: &'a [f64],
    },
    BetaBinomial {
        trials: &'a [f64],
        alpha: &'a [f64],
        beta: &'a [f64],
    },
}

/// Constants per state, `CONSTANTS` of them, in a fixed order per family.
const CONSTANTS: usize = 3;

impl<'a> CountFamily<'a> {
    fn new(kind: Kind<'a>, approx: bool, stirling_from: f64) -> Self {
        let m = match &kind {
            Kind::Poisson { rate } => rate.len(),
            Kind::Binomial { trials, .. } | Kind::BetaBinomial { trials, .. } => trials.len(),
            Kind::NegativeBinomial { mean, .. } => mean.len(),
        };
        let mut constants = vec![0.0; CONSTANTS * m];
        for (state, row) in constants.chunks_exact_mut(CONSTANTS).enumerate() {
            match &kind {
                // ln(rate).
                Kind::Poisson { rate } => row[0] = rate[state].ln(),
                // ln(n!), ln(p), ln(1 - p).
                Kind::Binomial {
                    trials,
                    probability,
                } => {
                    let p = probability[state];
                    row[0] = ln_gamma(trials[state] + 1.0);
                    row[1] = p.ln();
                    row[2] = (-p).ln_1p();
                }
                // ln Gamma(r), r ln(r / (r + mu)), ln(mu / (r + mu)).
                Kind::NegativeBinomial { dispersion, mean } => {
                    let (r, mu) = (dispersion[state], mean[state]);
                    let total = r + mu;
                    row[0] = ln_gamma(r);
                    row[1] = r * (r / total).ln();
                    row[2] = (mu / total).ln();
                }
                // ln(n!) - ln Gamma(n + a + b), ln Gamma(a + b) - ln Gamma(a) - ln Gamma(b).
                Kind::BetaBinomial {
                    trials,
                    alpha,
                    beta,
                } => {
                    let (n, a, b) = (trials[state], alpha[state], beta[state]);
                    row[0] = ln_gamma(n + 1.0) - ln_gamma(n + a + b);
                    row[1] = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b);
                }
            }
        }
        CountFamily {
            kind,
            constants,
            approx,
            stirling_from,
        }
    }

    fn n_states(&self) -> usize {
        self.constants.len() / CONSTANTS
    }

    /// A row per occupied cell, `cells.len() * m`, filled where `tabled`:
    /// the table `Scoring` reads.
    pub fn table(
        &self,
        cells: &[i64],
        tabled: &[bool],
        stride: usize,
        covariate: bool,
    ) -> Vec<f64> {
        let m = self.n_states();
        let mut table = vec![0.0; cells.len() * m];
        for ((row, &cell), _) in table
            .chunks_exact_mut(m)
            .zip(cells)
            .zip(tabled)
            .filter(|(_, &keep)| keep)
        {
            let cell = cell as usize;
            let c = covariate.then_some((cell % stride) as f64);
            self.log_density((cell / stride) as f64, c, row);
        }
        table
    }

    /// The `m` log-densities of the count `y`, given the covariate `c` if any.
    #[inline]
    pub fn log_density(&self, y: f64, c: Option<f64>, out: &mut [f64]) {
        if self.approx {
            self.score::<true>(y, c, out);
        } else {
            self.score::<false>(y, c, out);
        }
    }

    #[inline]
    fn score<const APPROX: bool>(&self, y: f64, c: Option<f64>, out: &mut [f64]) {
        let from = self.stirling_from;
        let ln_gamma = |x: f64| lgamma::<APPROX>(x, from);
        let log_factorial = ln_gamma(y + 1.0);
        let rows = self.constants.chunks_exact(CONSTANTS);
        match &self.kind {
            Kind::Poisson { rate } => {
                for ((o, &r), k) in out.iter_mut().zip(rate.iter()).zip(rows) {
                    *o = y * k[0] - r - log_factorial;
                }
            }
            Kind::Binomial { trials, .. } => {
                for (state, (o, k)) in out.iter_mut().zip(rows).enumerate() {
                    let n = c.unwrap_or(trials[state]);
                    let choose = c.map_or(k[0], |n| ln_gamma(n + 1.0));
                    *o = choose - log_factorial - ln_gamma(n - y + 1.0) + y * k[1] + (n - y) * k[2];
                }
            }
            Kind::NegativeBinomial { dispersion, mean } => {
                for (state, (o, k)) in out.iter_mut().zip(rows).enumerate() {
                    let r = dispersion[state];
                    let rising = ln_gamma(y + r) - k[0] - log_factorial;
                    *o = match c {
                        None => rising + k[1] + y * k[2],
                        Some(e) => {
                            let rate = e * mean[state];
                            let total = r + rate;
                            rising + r * (r / total).ln() + y * (rate / total).ln()
                        }
                    };
                }
            }
            Kind::BetaBinomial {
                trials,
                alpha,
                beta,
            } => {
                for (state, (o, k)) in out.iter_mut().zip(rows).enumerate() {
                    let n = c.unwrap_or(trials[state]);
                    if y > n {
                        *o = f64::NEG_INFINITY;
                        continue;
                    }
                    let (a, b) = (alpha[state], beta[state]);
                    let outer = c.map_or(k[0], |n| ln_gamma(n + 1.0) - ln_gamma(n + a + b));
                    *o = outer - log_factorial - ln_gamma(n - y + 1.0)
                        + ln_gamma(y + a)
                        + ln_gamma(n - y + b)
                        + k[1];
                }
            }
        }
    }
}

/// How a count step scores an observation: from `table` where its cell's
/// row is tabled, by `family` otherwise.
pub struct Scoring<'a> {
    /// `n_rows * m`; only the tabled rows are read.
    pub table: &'a [f64],
    /// Per row, whether `table` holds it.
    pub tabled: &'a [bool],
    pub family: &'a CountFamily<'a>,
}

/// One streamed step of an HMM over integer counts.
///
/// The E step hands back `histogram`, `n_rows * m`, each state's posterior
/// weight on each occupied cell: the data a count family's M step reads,
/// since each of its sums over observations is a sum over the occupied cells
/// weighted by these. A tabled cell is scored once per step, which pays
/// where its count repeats; any other observation is scored where it
/// stands, which pays where counts do not.
pub fn count_step(
    observations: &[i64],
    length: usize,
    log_initial: &[f64],
    log_transition: &[f64],
    cells: &Cells<'_>,
    n_rows: usize,
    scoring: &Scoring<'_>,
) -> Result<TableStep, String> {
    let m = log_initial.len();
    if scoring.family.n_states() != m {
        return Err(format!(
            "{m} states and a family of {}",
            scoring.family.n_states()
        ));
    }
    if scoring.table.len() != n_rows * m || scoring.tabled.len() != n_rows {
        return Err(format!(
            "a table of {} entries and {} flags for {n_rows} rows of {m} states",
            scoring.table.len(),
            scoring.tabled.len()
        ));
    }
    let out_of_table = (0..observations.len()).find(|&index| {
        let cell = cells.cell(observations, index);
        cell >= cells.rows.len() || cells.rows[cell] as usize >= n_rows
    });
    if let Some(index) = out_of_table {
        return Err(format!("observation {index} has no row in the histogram"));
    }
    let (counts, histogram) = stream_counts(
        observations.len(),
        length,
        log_initial,
        log_transition,
        |index, out| {
            let row = cells.row(observations, index);
            if scoring.tabled[row] {
                out.copy_from_slice(&scoring.table[row * m..][..m]);
            } else {
                let c = cells.covariate.map(|c| c[index] as f64);
                scoring
                    .family
                    .log_density(observations[index] as f64, c, out);
            }
        },
        |index| cells.cell(observations, index),
        || Histogram {
            observations,
            cells,
            m,
            values: vec![0.0; n_rows * m],
        },
        8 * n_rows * m,
    )?;
    let (log_initial, log_transition) = chain_m_step(&counts, observations.len() / length);
    Ok(TableStep {
        log_initial,
        log_transition,
        histogram: histogram.values,
        log_likelihood: counts.log_likelihood,
    })
}

/// A count step's chain parameters, histogram and log-likelihood.
pub struct TableStep {
    pub log_initial: Vec<f64>,
    pub log_transition: Vec<f64>,
    pub histogram: Vec<f64>,
    pub log_likelihood: f64,
}

type FamilyOut<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    f64,
);

/// One Gaussian Baum--Welch step; see `gaussian_step`.
///
/// Returns `(log_initial, log_transition, mean, variance, log_likelihood)`.
#[pyfunction]
#[pyo3(signature = (observations, log_initial, log_transition, mean, scale))]
pub fn gaussian_em_step<'py>(
    py: Python<'py>,
    observations: PyReadonlyArray2<'py, f64>,
    log_initial: PyReadonlyArray1<'py, f64>,
    log_transition: PyReadonlyArray1<'py, f64>,
    mean: PyReadonlyArray1<'py, f64>,
    scale: PyReadonlyArray1<'py, f64>,
) -> PyResult<FamilyOut<'py>> {
    let length = observations.shape()[1];
    let (observations, log_initial, log_transition, mean, scale) = (
        observations.as_slice()?,
        log_initial.as_slice()?,
        log_transition.as_slice()?,
        mean.as_slice()?,
        scale.as_slice()?,
    );
    let step = py
        .detach(|| {
            gaussian_step(
                observations,
                length,
                log_initial,
                log_transition,
                mean,
                scale,
            )
        })
        .map_err(PyValueError::new_err)?;
    Ok((
        PyArray1::from_vec(py, step.log_initial),
        PyArray1::from_vec(py, step.log_transition),
        PyArray1::from_vec(py, step.mean),
        PyArray1::from_vec(py, step.variance),
        step.log_likelihood,
    ))
}

/// The occupied cells of a count HMM's data and their multiplicities; see
/// `occupied_cells`.
#[pyfunction]
#[pyo3(signature = (observations, covariate, stride))]
#[allow(clippy::type_complexity)]
pub fn count_cells<'py>(
    py: Python<'py>,
    observations: PyReadonlyArray2<'py, i64>,
    covariate: Option<PyReadonlyArray2<'py, i64>>,
    stride: usize,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)> {
    let observations = observations.as_slice()?;
    let covariate = covariate.as_ref().map(|c| c.as_slice()).transpose()?;
    let (cells, multiplicity) = py
        .detach(|| occupied_cells(observations, covariate, stride))
        .map_err(PyValueError::new_err)?;
    Ok((
        PyArray1::from_vec(py, cells),
        PyArray1::from_vec(py, multiplicity),
    ))
}

/// A `CountFamily` from its name and its parameters stacked per state.
pub fn count_family<'a>(
    name: &str,
    parameters: &'a [f64],
    m: usize,
    approx: bool,
    stirling_from: f64,
) -> Result<CountFamily<'a>, String> {
    let row = |k: usize| &parameters[k * m..][..m];
    let wanted = match name {
        "poisson" => 1,
        "binomial" | "negative_binomial" => 2,
        "beta_binomial" => 3,
        _ => return Err(format!("no direct density for {name:?}")),
    };
    if parameters.len() != wanted * m {
        return Err(format!(
            "{name} takes {wanted} rows of {m} parameters, got {}",
            parameters.len()
        ));
    }
    Ok(CountFamily::new(
        match name {
            "poisson" => Kind::Poisson { rate: row(0) },
            "binomial" => Kind::Binomial {
                trials: row(0),
                probability: row(1),
            },
            "negative_binomial" => Kind::NegativeBinomial {
                dispersion: row(0),
                mean: row(1),
            },
            _ => Kind::BetaBinomial {
                trials: row(0),
                alpha: row(1),
                beta: row(2),
            },
        },
        approx,
        stirling_from,
    ))
}

/// One count-family Baum--Welch E step; see `count_step`.
///
/// `cells` are the occupied cells, ascending, and `rows` their inverse.
/// `family` and its `parameters` stacked per state (`rate`; `trials,
/// probability`; `dispersion, mean`; `trials, alpha, beta`) are scored here,
/// term for term the family's `log_density`: once per cell into a table for
/// the rows `tabled` flags, at every observation for the rest, with
/// `ln_gamma_approx` from `stirling_from` where `approx` is set. Returns `(log_initial,
/// log_transition, histogram, log_likelihood)`, the histogram flattened
/// `cells.len() * m`.
#[pyfunction]
#[pyo3(signature = (observations, covariate, stride, cells, rows, log_initial, log_transition, family, parameters, tabled, approx=false, stirling_from=STIRLING_FROM))]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn count_em_step<'py>(
    py: Python<'py>,
    observations: PyReadonlyArray2<'py, i64>,
    covariate: Option<PyReadonlyArray2<'py, i64>>,
    stride: usize,
    cells: PyReadonlyArray1<'py, i64>,
    rows: PyReadonlyArray1<'py, u32>,
    log_initial: PyReadonlyArray1<'py, f64>,
    log_transition: PyReadonlyArray1<'py, f64>,
    family: &str,
    parameters: PyReadonlyArray1<'py, f64>,
    tabled: PyReadonlyArray1<'py, bool>,
    approx: bool,
    stirling_from: f64,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    f64,
)> {
    let length = observations.shape()[1];
    let (observations, occupied, rows, log_initial, log_transition, parameters) = (
        observations.as_slice()?,
        cells.as_slice()?,
        rows.as_slice()?,
        log_initial.as_slice()?,
        log_transition.as_slice()?,
        parameters.as_slice()?,
    );
    let tabled = tabled.as_slice()?;
    let covariate = covariate.as_ref().map(|c| c.as_slice()).transpose()?;
    let family = count_family(family, parameters, log_initial.len(), approx, stirling_from)
        .map_err(PyValueError::new_err)?;
    let cells = Cells {
        covariate,
        stride,
        rows,
    };
    let step = py
        .detach(|| {
            let table = family.table(occupied, tabled, stride, covariate.is_some());
            let scoring = Scoring {
                table: &table,
                tabled,
                family: &family,
            };
            count_step(
                observations,
                length,
                log_initial,
                log_transition,
                &cells,
                occupied.len(),
                &scoring,
            )
        })
        .map_err(PyValueError::new_err)?;
    Ok((
        PyArray1::from_vec(py, step.log_initial),
        PyArray1::from_vec(py, step.log_transition),
        PyArray1::from_vec(py, step.histogram),
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

    /// The Gaussian step's log-likelihood and means by path enumeration.
    #[test]
    fn the_gaussian_step_is_the_enumerated_expectation() {
        let (m, length) = (2usize, 4usize);
        let observations = [0.3f64, -1.2, 2.5, 0.8, 1.9, -0.4, 0.1, 3.0];
        let (initial, transition) = ([0.6f64, 0.4], [0.7f64, 0.3, 0.2, 0.8]);
        let (mean, scale) = ([0.0f64, 2.0], [1.0f64, 0.7]);
        let ln = |v: &[f64]| v.iter().map(|x| x.ln()).collect::<Vec<_>>();
        let step = gaussian_step(
            &observations,
            length,
            &ln(&initial),
            &ln(&transition),
            &mean,
            &scale,
        )
        .unwrap();
        let density = |x: f64, s: usize| {
            let z = (x - mean[s]) / scale[s];
            (-0.5 * z * z).exp() / (scale[s] * (2.0 * std::f64::consts::PI).sqrt())
        };
        let (mut ll, mut s0, mut s1, mut s2) = (0.0, [0.0; 2], [0.0; 2], [0.0; 2]);
        for row in observations.chunks_exact(length) {
            let mut weighted = Vec::new();
            let mut evidence = 0.0;
            for path in 0..m.pow(length as u32) {
                let states: Vec<usize> = (0..length).map(|t| (path >> t) & 1).collect();
                let mut p = initial[states[0]] * density(row[0], states[0]);
                for t in 1..length {
                    p *= transition[states[t - 1] * m + states[t]] * density(row[t], states[t]);
                }
                evidence += p;
                weighted.push((states, p));
            }
            ll += evidence.ln();
            for (states, p) in weighted {
                for t in 0..length {
                    let w = p / evidence;
                    s0[states[t]] += w;
                    s1[states[t]] += w * row[t];
                    s2[states[t]] += w * row[t] * row[t];
                }
            }
        }
        assert!((step.log_likelihood - ll).abs() < 1e-12);
        for s in 0..m {
            let mu = s1[s] / s0[s];
            assert!((step.mean[s] - mu).abs() < 1e-12);
            assert!((step.variance[s] - (s2[s] / s0[s] - mu * mu)).abs() < 1e-12);
        }
    }

    /// A table is a categorical emission read by column: the two steps agree.
    #[test]
    fn the_table_step_is_the_categorical_step() {
        let length = 4usize;
        let observations = [0i64, 2, 1, 1, 2, 2, 0, 1];
        let ln = |v: &[f64]| v.iter().map(|x| x.ln()).collect::<Vec<_>>();
        let (initial, transition) = (ln(&[0.6, 0.4]), ln(&[0.7, 0.3, 0.2, 0.8]));
        let emission = ln(&[0.5, 0.3, 0.2, 0.1, 0.4, 0.5]);
        let table: Vec<f64> = (0..3)
            .flat_map(|u| (0..2).map(move |s| (s, u)))
            .map(|(s, u)| emission[s * 3 + u])
            .collect();
        let rows = [0u32, 1, 2];
        let cells = Cells {
            covariate: None,
            stride: 1,
            rows: &rows,
        };
        assert_eq!(
            occupied_cells(&observations, None, 1).unwrap(),
            (vec![0, 1, 2], vec![2, 3, 3])
        );
        let family_parameters = [30.0, 30.0];
        let family = CountFamily::new(
            Kind::Poisson {
                rate: &family_parameters,
            },
            false,
            STIRLING_FROM,
        );
        let by_table = count_step(
            &observations,
            length,
            &initial,
            &transition,
            &cells,
            3,
            &Scoring {
                table: &table,
                tabled: &[true; 3],
                family: &family,
            },
        )
        .unwrap();
        let by_symbol =
            categorical_step(&observations, length, &initial, &transition, &emission).unwrap();
        assert!((by_table.log_likelihood - by_symbol.log_likelihood).abs() < 1e-12);
        for (a, b) in by_table
            .log_transition
            .iter()
            .zip(&by_symbol.log_transition)
        {
            assert!((a - b).abs() < 1e-12);
        }
        for s in 0..2 {
            let row: f64 = (0..3).map(|u| by_table.histogram[u * 2 + s]).sum();
            for u in 0..3 {
                let fitted = (by_table.histogram[u * 2 + s] / row).ln();
                assert!((fitted - by_symbol.log_emission[s * 3 + u]).abs() < 1e-12);
            }
        }
    }
}
