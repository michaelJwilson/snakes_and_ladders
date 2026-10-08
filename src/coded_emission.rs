//! Coded count observations (issue #1340): `encode` and the coded
//! log-emission, exposed to Python as `sal.oxisal.coded_encode` and
//! `sal.oxisal.coded_log_emission`.
//!
//! **Encode** sorts the observed keys, deduplicates them and reads each
//! observation's code by binary search: `inverse[i]` is the code of
//! observation `i`, `-1` where its covariate is zero (unobserved), and
//! `weight[u]` the observations carrying code `u`. A key is a count, or a
//! pair's total in the high 32 bits and successes in the low 32, so the codes
//! run in ascending count order.
//!
//! **The log-emission is `src/dense_emission.rs`'s kernel, not a copy of it.**
//! Its channels take `rows`, the inverse, and tables with one row per code;
//! the covariate term is completed per observation by `coupled.rs`'s
//! [`crate::coupled::ExposureTerm`] and [`crate::coupled::TrialTerm`], in the
//! order the dense route completes it, so a coded score is the dense score.
//!
//! Plain Rust with no PyO3 types in [`encode`], so `cargo test` links it.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::coupled::{borrowed, exposure_term, trial_term};
use crate::dense_emission::{log_emission_into, ExposureOrder, SuccessChannel, TotalChannel};

/// The codes of `keys`: the sorted distinct observed keys, the inverse, and the weights.
pub struct Encoded {
    /// `U` distinct keys, ascending.
    pub keys: Vec<u64>,
    /// `N` codes, `-1` where the observation is unobserved.
    pub inverse: Vec<i32>,
    /// `U` observation counts, one per code.
    pub weight: Vec<i64>,
}

/// Code `keys`, skipping each observation whose `observed` entry is false.
///
/// # Errors
/// If `observed` is not `N` long, or more than `i32::MAX` codes would be needed.
pub fn encode(keys: &[u64], observed: Option<&[bool]>) -> Result<Encoded, String> {
    if let Some(mask) = observed {
        if mask.len() != keys.len() {
            return Err(format!(
                "the observed mask has {} entries, expected N = {}",
                mask.len(),
                keys.len()
            ));
        }
    }
    let seen = |i: usize| observed.is_none_or(|mask| mask[i]);
    let mut distinct: Vec<u64> = (0..keys.len())
        .filter(|&i| seen(i))
        .map(|i| keys[i])
        .collect();
    distinct.par_sort_unstable();
    distinct.dedup();
    if i32::try_from(distinct.len()).is_err() {
        return Err("more distinct counts than an int32 inverse codes".to_string());
    }
    let inverse: Vec<i32> = (0..keys.len())
        .into_par_iter()
        .map(|i| {
            if seen(i) {
                // Present by construction; the cast is checked above.
                distinct.binary_search(&keys[i]).map_or(-1, |u| u as i32)
            } else {
                -1
            }
        })
        .collect();
    let mut weight = vec![0_i64; distinct.len()];
    for &u in &inverse {
        if let Ok(u) = usize::try_from(u) {
            weight[u] += 1;
        }
    }
    Ok(Encoded {
        keys: distinct,
        inverse,
        weight,
    })
}

/// [`encode`] as a Python binding: `(keys, inverse, weight)`.
///
/// `observed`, where given, is `N` bytes, nonzero where the observation is
/// observed. The GIL is released for the whole pass.
///
/// # Errors
/// `ValueError` naming the first violated precondition.
#[pyfunction]
#[pyo3(signature = (keys, observed=None))]
#[allow(clippy::type_complexity)]
pub fn coded_encode<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'_, u64>,
    observed: Option<PyReadonlyArray1<'_, bool>>,
) -> PyResult<(
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i64>>,
)> {
    let keys = borrowed(&keys, "keys")?;
    let mask = match &observed {
        None => None,
        Some(mask) => Some(borrowed(mask, "observed")?),
    };
    let encoded = py
        .detach(|| encode(keys, mask))
        .map_err(PyValueError::new_err)?;
    Ok((
        encoded.keys.into_pyarray(py),
        encoded.inverse.into_pyarray(py),
        encoded.weight.into_pyarray(py),
    ))
}

/// The coded log-emission, `out[k * N + i]`, through [`log_emission_into`].
///
/// `inverse` is `N` codes; `totals` and `total_table` are the `U` distinct
/// totals and their table, one row per code, and `exposure`, `dispersion`
/// and `mean` its exposure term; `successes`, `success_table` and the trial
/// arguments the second channel, as `dense_log_emission` takes them. Every
/// array crosses once, contiguous and borrowed; the GIL is released.
///
/// # Errors
/// `ValueError` naming the first violated precondition.
#[pyfunction]
#[pyo3(signature = (n_states, inverse, out, totals=None, total_table=None, exposure=None, dispersion=None, mean=None, successes=None, success_table=None, trials=None, failure_table=None, trial_table=None, log_factorial=None, log_rate=None))]
#[allow(clippy::too_many_arguments)]
pub fn coded_log_emission(
    py: Python<'_>,
    n_states: usize,
    inverse: PyReadonlyArray1<'_, i32>,
    mut out: PyReadwriteArray1<'_, f64>,
    totals: Option<PyReadonlyArray1<'_, u32>>,
    total_table: Option<PyReadonlyArray1<'_, f64>>,
    exposure: Option<PyReadonlyArray1<'_, f64>>,
    dispersion: Option<PyReadonlyArray1<'_, f64>>,
    mean: Option<PyReadonlyArray1<'_, f64>>,
    successes: Option<PyReadonlyArray1<'_, u32>>,
    success_table: Option<PyReadonlyArray1<'_, f64>>,
    trials: Option<PyReadonlyArray1<'_, u32>>,
    failure_table: Option<PyReadonlyArray1<'_, f64>>,
    trial_table: Option<PyReadonlyArray1<'_, f64>>,
    log_factorial: Option<PyReadonlyArray1<'_, f64>>,
    log_rate: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<()> {
    let rows = borrowed(&inverse, "inverse")?;
    let total = match (&totals, &total_table) {
        (None, None) => None,
        (Some(counts), Some(table)) => Some(TotalChannel {
            counts: borrowed(counts, "totals")?,
            rows: Some(rows),
            table: borrowed(table, "total_table")?,
            exposure: exposure_term(&exposure, &dispersion, &mean)?,
            order: ExposureOrder::Family,
        }),
        _ => {
            return Err(PyValueError::new_err(
                "the first channel takes totals and total_table together",
            ))
        }
    };
    let second = match (&successes, &success_table) {
        (None, None) => None,
        (Some(counts), Some(table)) => Some(SuccessChannel {
            counts: borrowed(counts, "successes")?,
            rows: Some(rows),
            table: borrowed(table, "success_table")?,
            trials: trial_term(
                &trials,
                &failure_table,
                &trial_table,
                &log_factorial,
                &log_rate,
            )?,
        }),
        _ => {
            return Err(PyValueError::new_err(
                "the second channel takes successes and success_table together",
            ))
        }
    };
    let out = out
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("out must be C-contiguous"))?;
    let n = rows.len();
    py.detach(|| log_emission_into(n_states, n, total.as_ref(), second.as_ref(), out))
        .map_err(PyValueError::new_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_codes_ascending_and_skips_the_unobserved() {
        let keys = [5, 2, 5, 9, 2, 7];
        let observed = [true, true, true, true, true, false];
        let encoded = encode(&keys, Some(&observed)).unwrap();
        assert_eq!(encoded.keys, [2, 5, 9]);
        assert_eq!(encoded.inverse, [1, 0, 1, 2, 0, -1]);
        assert_eq!(encoded.weight, [2, 2, 1]);
    }

    #[test]
    fn a_coded_gather_is_the_dense_read_bitwise() {
        // Two states, counts 0..3, dense against coded on the distinct counts.
        let table = [0.0, 10.0, 1.0, 11.0, 2.0, 12.0];
        let counts = [2_u32, 0, 1, 2];
        let dense = TotalChannel {
            counts: &counts,
            rows: None,
            table: &table,
            exposure: None,
            order: ExposureOrder::Family,
        };
        let mut expected = vec![0.0; 8];
        log_emission_into(2, 4, Some(&dense), None, &mut expected).unwrap();
        let keys: Vec<u64> = counts.iter().map(|&c| u64::from(c)).collect();
        let encoded = encode(&keys, None).unwrap();
        let distinct: Vec<u32> = encoded.keys.iter().map(|&k| k as u32).collect();
        let coded_table: Vec<f64> = distinct
            .iter()
            .flat_map(|&c| table[c as usize * 2..][..2].to_vec())
            .collect();
        let coded = TotalChannel {
            counts: &distinct,
            rows: Some(&encoded.inverse),
            table: &coded_table,
            exposure: None,
            order: ExposureOrder::Family,
        };
        let mut out = vec![0.0; 8];
        log_emission_into(2, 4, Some(&coded), None, &mut out).unwrap();
        assert_eq!(
            out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn an_unobserved_code_scores_zero_and_a_stray_row_is_refused() {
        let table = [3.0, 4.0];
        let counts = [7_u32];
        let rows = [0, -1];
        let channel = TotalChannel {
            counts: &counts,
            rows: Some(&rows),
            table: &table,
            exposure: None,
            order: ExposureOrder::Family,
        };
        let mut out = vec![1.0; 4];
        log_emission_into(2, 2, Some(&channel), None, &mut out).unwrap();
        assert_eq!(out, [3.0, 0.0, 4.0, 0.0]);
        let stray = [1, 0];
        let channel = TotalChannel {
            rows: Some(&stray),
            ..channel
        };
        let error = log_emission_into(2, 2, Some(&channel), None, &mut out).unwrap_err();
        assert!(error.contains("past the 1 codes"), "{error}");
    }
}
