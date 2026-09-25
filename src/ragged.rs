//! Forward-backward over segments of unequal length, laid end to end.
//!
//! Issue #666. The Python path pads every segment to the longest and masks,
//! which is what lets it batch one step per position of the longest. This one
//! does not pad at all: it walks the segments in place, so it does the work the
//! problem has rather than the work the longest segment implies. Where the
//! lengths are uneven that is the whole difference between the two, and the
//! benchmark reports it rather than this comment claiming it.
//!
//! The recursions restart at each boundary --- `alpha` at the initial
//! distribution, `beta` at one --- and the pair spanning a boundary is not
//! counted as a transition, because the model never took it.
//!
//! Plain Rust with no PyO3 types in `ragged_posteriors_into`, so `cargo test`
//! can link it, per `src/pruning.rs`'s module docs.

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1, PyReadwriteArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// `ln(exp(a) + exp(b))`, stable and with the `-inf` case exact.
#[inline]
fn log_add(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let (high, low) = if a > b { (a, b) } else { (b, a) };
    high + (low - high).exp().ln_1p()
}

/// `ln(sum(exp(values)))` over a slice, by the same reduction as `log_add`.
#[inline]
fn log_sum(values: &[f64]) -> f64 {
    values
        .iter()
        .fold(f64::NEG_INFINITY, |total, &one| log_add(total, one))
}

/// Posterior marginals, transition counts and evidence, segment by segment.
///
/// # Parameters
/// - `log_density`: row-major `total * n_states`, the segments end to end.
/// - `lengths`: one per segment, each at least 2, summing to `total`.
/// - `log_initial`: `n_states`, the distribution each segment restarts at.
/// - `log_transition`: row-major `n_states * n_states`.
/// - `gamma`: written, `total * n_states`, each row a log posterior.
/// - `counts`: written, `n_states * n_states`, log expected transitions summed
///   over every segment --- the boundary pairs are not among them.
/// - `evidence`: written, one log evidence per segment.
///
/// # Returns
/// `Ok(())`, or `Err` naming the first violated precondition.
#[allow(clippy::too_many_arguments)]
pub fn ragged_posteriors_into(
    log_density: &[f64],
    n_states: usize,
    lengths: &[usize],
    log_initial: &[f64],
    log_transition: &[f64],
    gamma: &mut [f64],
    counts: &mut [f64],
    evidence: &mut [f64],
) -> Result<(), String> {
    if n_states == 0 {
        return Err("n_states must be positive".to_string());
    }
    if log_initial.len() != n_states {
        return Err(format!(
            "log_initial has {} entries for {} states",
            log_initial.len(),
            n_states
        ));
    }
    if log_transition.len() != n_states * n_states {
        return Err(format!(
            "log_transition has {} entries for {} states",
            log_transition.len(),
            n_states
        ));
    }
    if let Some(index) = lengths.iter().position(|&one| one < 2) {
        return Err(format!(
            "segment {} has length {}; a segment carries at least 2 positions, \
             since one position is an initial distribution and no transition",
            index, lengths[index]
        ));
    }
    let total: usize = lengths.iter().sum();
    if log_density.len() != total * n_states {
        return Err(format!(
            "log_density has {} entries for {} positions over {} states",
            log_density.len(),
            total,
            n_states
        ));
    }
    if gamma.len() != log_density.len() || evidence.len() != lengths.len() {
        return Err("gamma and evidence must match the segments they describe".to_string());
    }

    counts.fill(f64::NEG_INFINITY);
    let mut alpha = vec![0.0_f64; n_states];
    let mut previous = vec![0.0_f64; n_states];
    let mut forward = Vec::<f64>::new();
    let mut beta = vec![0.0_f64; n_states];
    let mut ahead = vec![0.0_f64; n_states];

    let mut start = 0_usize;
    for (segment, &length) in lengths.iter().enumerate() {
        let base = start * n_states;
        forward.clear();
        forward.resize(length * n_states, 0.0);

        // Forward: the chain restarts here, at the prior and not at a kernel.
        for state in 0..n_states {
            alpha[state] = log_initial[state] + log_density[base + state];
            forward[state] = alpha[state];
        }
        for step in 1..length {
            previous.copy_from_slice(&alpha);
            for state in 0..n_states {
                let mut carried = f64::NEG_INFINITY;
                for from in 0..n_states {
                    carried = log_add(
                        carried,
                        previous[from] + log_transition[from * n_states + state],
                    );
                }
                alpha[state] = carried + log_density[base + step * n_states + state];
                forward[step * n_states + state] = alpha[state];
            }
        }
        let total_evidence = log_sum(&alpha);
        evidence[segment] = total_evidence;

        // Backward, and the pair counts as it goes. `beta` is one at the last
        // position: the chain ends, it does not continue into the next segment.
        beta.fill(0.0);
        for state in 0..n_states {
            gamma[base + (length - 1) * n_states + state] =
                forward[(length - 1) * n_states + state] - total_evidence;
        }
        for step in (0..length - 1).rev() {
            let next = base + (step + 1) * n_states;
            for state in 0..n_states {
                ahead[state] = log_density[next + state] + beta[state];
            }
            for from in 0..n_states {
                let row = from * n_states;
                let mut carried = f64::NEG_INFINITY;
                for to in 0..n_states {
                    let pair =
                        forward[step * n_states + from] + log_transition[row + to] + ahead[to]
                            - total_evidence;
                    counts[row + to] = log_add(counts[row + to], pair);
                    carried = log_add(carried, log_transition[row + to] + ahead[to]);
                }
                beta[from] = carried;
                gamma[base + step * n_states + from] =
                    forward[step * n_states + from] + carried - total_evidence;
            }
        }
        start += length;
    }
    Ok(())
}

/// PyO3 wrapper over [`ragged_posteriors_into`]; converts `Err` to `ValueError`.
///
/// The recursion touches no Python object, so it runs with the GIL released
/// and a thread pool runs segments' batches at once (#604, #1059).
#[pyfunction]
#[pyo3(name = "ragged_posteriors")]
#[allow(clippy::too_many_arguments)]
pub fn ragged_posteriors(
    py: Python<'_>,
    log_density: PyReadonlyArray2<f64>,
    lengths: PyReadonlyArray1<i64>,
    log_initial: PyReadonlyArray1<f64>,
    log_transition: PyReadonlyArray2<f64>,
    mut gamma: PyReadwriteArray2<f64>,
    mut counts: PyReadwriteArray2<f64>,
    mut evidence: PyReadwriteArray1<f64>,
) -> PyResult<()> {
    let density = log_density.as_slice()?;
    let n_states = log_initial.len()?;
    let widths: Vec<usize> = lengths
        .as_slice()?
        .iter()
        .map(|&one| usize::try_from(one).unwrap_or(0))
        .collect();
    let (initial, transition) = (log_initial.as_slice()?, log_transition.as_slice()?);
    let (gamma, counts, evidence) = (
        gamma.as_slice_mut()?,
        counts.as_slice_mut()?,
        evidence.as_slice_mut()?,
    );
    py.detach(|| {
        ragged_posteriors_into(
            density, n_states, &widths, initial, transition, gamma, counts, evidence,
        )
    })
    .map_err(PyValueError::new_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two segments of two positions, against the value worked by hand.
    #[test]
    fn a_two_segment_batch_sums_its_evidences() {
        let density = vec![0.0; 8];
        let lengths = [2_usize, 2];
        let initial = vec![(0.5_f64).ln(), (0.5_f64).ln()];
        let transition = vec![(0.5_f64).ln(); 4];
        let mut gamma = vec![0.0; 8];
        let mut counts = vec![0.0; 4];
        let mut evidence = vec![0.0; 2];
        ragged_posteriors_into(
            &density,
            2,
            &lengths,
            &initial,
            &transition,
            &mut gamma,
            &mut counts,
            &mut evidence,
        )
        .unwrap();
        // Every density is one and every choice even, so each segment's
        // evidence is one and each posterior a half.
        for one in &evidence {
            assert!((one - 0.0).abs() < 1e-12, "evidence {one}");
        }
        for one in &gamma {
            assert!((one - (0.5_f64).ln()).abs() < 1e-12, "gamma {one}");
        }
    }

    #[test]
    fn a_one_position_segment_is_refused() {
        let mut gamma = vec![0.0; 3];
        let mut counts = vec![0.0; 1];
        let mut evidence = vec![0.0; 2];
        let error = ragged_posteriors_into(
            &[0.0; 3],
            1,
            &[2, 1],
            &[0.0],
            &[0.0],
            &mut gamma,
            &mut counts,
            &mut evidence,
        )
        .unwrap_err();
        assert!(error.contains("at least 2 positions"), "{error}");
    }
}
