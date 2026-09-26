//! The dense count log-emission, state-major, from the families' own tables
//! (issue #1132), exposed to Python as `sal.oxisal.dense_log_emission`.
//!
//! `src/coupled.rs` completes the negative binomial's exposure term and the
//! beta-binomial's trial term per observation, but only inside the sums its
//! E step and field form, so a caller whose inference is not that E step ---
//! an M step differentiating the emission by finite differences, say --- had
//! no way to read the scores out. This writes them: `out[k * N + i]` is
//! `log p(observation i | state k)`, the first channel's score plus the
//! second's, in the order the pair families add them.
//!
//! **The per-observation arithmetic is `coupled.rs`'s, not a copy of it.**
//! Each score is [`ExposureTerm::score_into`] (or its family-order sibling)
//! and [`TrialTerm::score_into`] with one class, so the dense result and the
//! coupled kernel's terms are one implementation.
//!
//! **State-major output, observation-major walk.** A caller holds the result
//! as `(K, T, N)`, so each state's scores are one contiguous row; the walk
//! goes observation by observation, since the counts, the covariates and a
//! table row at one count are read once for every state. Observations are cut
//! into tiles handed to the thread pool, and a tile owns the same columns of
//! every state's row, so no two threads write one cache line but at the seams.
//!
//! Plain Rust with no PyO3 types in [`log_emission_into`], so `cargo test`
//! can link it, per `src/pruning.rs`'s module docs.

use numpy::{PyReadonlyArray1, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::coupled::{borrowed, exposure_term, trial_term, CoupledShape, ExposureTerm, TrialTerm};

/// Observations per tile: 4,096 columns of every state's row.
const TILE: usize = 4096;

/// The order the negative binomial's exposure term is completed in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExposureOrder {
    /// `A + r ln(r / t) + y ln(mu c / t)`, the family's own: two `ln` a score.
    Family,
    /// `B + y ln c - (y + r) ln t`, the coupled kernel's: one `ln` a score.
    Tabulated,
}

/// The first channel: a negative binomial by count, with or without an exposure.
pub struct TotalChannel<'a> {
    /// `N` counts.
    pub counts: &'a [u32],
    /// `extent * K`, row `y` the count `y`: the density itself without an
    /// exposure, `A` with one in [`ExposureOrder::Family`], `B` in
    /// [`ExposureOrder::Tabulated`].
    pub table: &'a [f64],
    /// The exposure per observation and the family's `r` and `mu`, where it
    /// carries one.
    pub exposure: Option<ExposureTerm<'a>>,
    /// The order the exposure term is completed in; unread without one.
    pub order: ExposureOrder,
}

/// The second channel: a beta-binomial by successes, with or without a trial count.
pub struct SuccessChannel<'a> {
    /// `N` successes.
    pub counts: &'a [u32],
    /// `extent * K`, row `z` the successes `z`: the density itself without a
    /// trial count, `U[z] = lgamma(z + a)` with one.
    pub table: &'a [f64],
    /// The trial count per observation and the other tables, where it carries one.
    pub trials: Option<TrialTerm<'a>>,
}

/// One class of `n_states` over `n` observations, the shape `coupled.rs`'s checks take.
fn shape(n: usize, n_states: usize) -> CoupledShape {
    CoupledShape {
        n_positions: 1,
        n_nodes: n,
        n_classes: 1,
        n_states,
    }
}

/// Refuse a table that is not a whole number of rows, or does not reach a count.
fn check_table(name: &str, table: &[f64], counts: &[u32], n_states: usize) -> Result<(), String> {
    if !table.len().is_multiple_of(n_states) {
        return Err(format!(
            "the {name} table has {} entries, not a multiple of K = {n_states}",
            table.len()
        ));
    }
    let extent = table.len() / n_states;
    match counts.iter().max() {
        Some(&largest) if largest as usize >= extent => Err(format!(
            "a {name} count of {largest} is past the table's extent {extent}"
        )),
        _ => Ok(()),
    }
}

/// Every precondition, before any score is written.
fn validate(
    n_states: usize,
    n: usize,
    total: Option<&TotalChannel<'_>>,
    successes: Option<&SuccessChannel<'_>>,
    out: &[f64],
) -> Result<(), String> {
    if n_states == 0 {
        return Err("a family has at least one state".to_string());
    }
    if total.is_none() && successes.is_none() {
        return Err("a dense emission scores at least one channel".to_string());
    }
    if out.len() != n_states * n {
        return Err(format!(
            "out has {} entries, expected K * N = {}",
            out.len(),
            n_states * n
        ));
    }
    let shape = shape(n, n_states);
    if let Some(channel) = total {
        if channel.counts.len() != n {
            return Err(format!(
                "the totals have {} entries, expected N = {n}",
                channel.counts.len()
            ));
        }
        check_table("total", channel.table, channel.counts, n_states)?;
        if let Some(term) = &channel.exposure {
            term.validate(&shape)?;
        }
    }
    if let Some(channel) = successes {
        if channel.counts.len() != n {
            return Err(format!(
                "the successes have {} entries, expected N = {n}",
                channel.counts.len()
            ));
        }
        check_table("success", channel.table, channel.counts, n_states)?;
        if let Some(term) = &channel.trials {
            term.validate(&shape)?;
        }
    }
    Ok(())
}

/// The first channel's scores at observation `i`, into `out` (`K` entries).
#[inline]
fn total_scores(channel: &TotalChannel<'_>, i: usize, out: &mut [f64]) {
    let n_states = out.len();
    let count = channel.counts[i];
    let row = &channel.table[count as usize * n_states..][..n_states];
    match (&channel.exposure, channel.order) {
        (None, _) => out.copy_from_slice(row),
        (Some(term), ExposureOrder::Tabulated) => {
            term.score_into(row, count, term.exposure[i], 0, out);
        }
        (Some(term), ExposureOrder::Family) => {
            term.score_family_into(row, count, term.exposure[i], 0, out);
        }
    }
}

/// The second channel's scores at observation `i`, into `out` (`K` entries).
#[inline]
fn success_scores(channel: &SuccessChannel<'_>, i: usize, out: &mut [f64]) {
    let n_states = out.len();
    let count = channel.counts[i];
    let row = &channel.table[count as usize * n_states..][..n_states];
    match &channel.trials {
        None => out.copy_from_slice(row),
        Some(term) => term.score_into(row, count, term.trials[i], n_states, 0, out),
    }
}

/// Write every observation's score under every state, `out[k * N + i]`.
///
/// With both channels a score is the first's plus the second's, the one
/// addition `log_density` of either pair family makes.
///
/// # Errors
/// A message naming the first violated precondition; nothing is written then.
pub fn log_emission_into(
    n_states: usize,
    n: usize,
    total: Option<&TotalChannel<'_>>,
    successes: Option<&SuccessChannel<'_>>,
    out: &mut [f64],
) -> Result<(), String> {
    validate(n_states, n, total, successes, out)?;
    if n == 0 {
        return Ok(());
    }
    // Each state's row cut at the same tile boundaries, then regrouped by
    // tile: tile `j` owns columns `j * TILE ..` of every row, disjointly.
    let mut rows: Vec<_> = out.chunks_mut(n).map(|row| row.chunks_mut(TILE)).collect();
    let tiles: Vec<Vec<&mut [f64]>> = (0..n.div_ceil(TILE))
        .map(|_| rows.iter_mut().filter_map(Iterator::next).collect())
        .collect();
    tiles.into_par_iter().enumerate().for_each(|(j, mut tile)| {
        let mut first = vec![0.0; n_states];
        let mut second = vec![0.0; n_states];
        for column in 0..tile[0].len() {
            let i = j * TILE + column;
            match (total, successes) {
                (Some(t), Some(s)) => {
                    total_scores(t, i, &mut first);
                    success_scores(s, i, &mut second);
                    for (k, row) in tile.iter_mut().enumerate() {
                        row[column] = first[k] + second[k];
                    }
                }
                (Some(t), None) => {
                    total_scores(t, i, &mut first);
                    for (k, row) in tile.iter_mut().enumerate() {
                        row[column] = first[k];
                    }
                }
                (None, Some(s)) => {
                    success_scores(s, i, &mut second);
                    for (k, row) in tile.iter_mut().enumerate() {
                        row[column] = second[k];
                    }
                }
                (None, None) => unreachable!("validated: at least one channel"),
            }
        }
    });
    Ok(())
}

/// [`log_emission_into`] as a Python binding.
///
/// `totals` and `total_table` name the first channel, and `exposure`,
/// `dispersion` and `mean`, given together, its exposure; `family_order`
/// says which order that term is completed in. `successes` and
/// `success_table` name the second, and `trials`, `failure_table`,
/// `trial_table`, `log_factorial` and `log_beta`, given together, its trial
/// count, as `class_posteriors` takes them. Every array crosses once,
/// contiguous and borrowed; the GIL is released for the whole pass.
///
/// # Returns
/// `None`; the scores are written into `out`, `K * N`, state-major.
///
/// # Errors
/// `ValueError` naming the first violated precondition.
#[pyfunction]
#[pyo3(signature = (n_states, family_order, out, totals=None, total_table=None, exposure=None, dispersion=None, mean=None, successes=None, success_table=None, trials=None, failure_table=None, trial_table=None, log_factorial=None, log_beta=None))]
#[allow(clippy::too_many_arguments)]
pub fn dense_log_emission(
    py: Python<'_>,
    n_states: usize,
    family_order: bool,
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
    log_beta: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<()> {
    let total = match (&totals, &total_table) {
        (None, None) => None,
        (Some(counts), Some(table)) => Some(TotalChannel {
            counts: borrowed(counts, "totals")?,
            table: borrowed(table, "total_table")?,
            exposure: exposure_term(&exposure, &dispersion, &mean)?,
            order: if family_order {
                ExposureOrder::Family
            } else {
                ExposureOrder::Tabulated
            },
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
            table: borrowed(table, "success_table")?,
            trials: trial_term(
                &trials,
                &failure_table,
                &trial_table,
                &log_factorial,
                &log_beta,
            )?,
        }),
        _ => {
            return Err(PyValueError::new_err(
                "the second channel takes successes and success_table together",
            ))
        }
    };
    if total.is_none() && exposure.is_some() {
        return Err(PyValueError::new_err("an exposure needs the first channel"));
    }
    if second.is_none() && trials.is_some() {
        return Err(PyValueError::new_err(
            "a trial count needs the second channel",
        ));
    }
    let n = total
        .as_ref()
        .map(|channel| channel.counts.len())
        .or_else(|| second.as_ref().map(|channel| channel.counts.len()))
        .unwrap_or(0);
    let out = out
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("out must be C-contiguous"))?;
    py.detach(|| log_emission_into(n_states, n, total.as_ref(), second.as_ref(), out))
        .map_err(PyValueError::new_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_table_without_a_covariate_is_read_state_major() {
        // Two states, counts 0..3: the score is the table's row, transposed.
        let table = [0.0, 10.0, 1.0, 11.0, 2.0, 12.0];
        let counts = [2, 0, 1, 2];
        let channel = TotalChannel {
            counts: &counts,
            table: &table,
            exposure: None,
            order: ExposureOrder::Family,
        };
        let mut out = vec![0.0; 8];
        log_emission_into(2, 4, Some(&channel), None, &mut out).unwrap();
        assert_eq!(out, [2.0, 0.0, 1.0, 2.0, 12.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn a_count_past_the_table_is_refused() {
        let table = [0.0, 0.0];
        let counts = [1];
        let channel = SuccessChannel {
            counts: &counts,
            table: &table,
            trials: None,
        };
        let mut out = vec![0.0; 2];
        let error = log_emission_into(2, 1, None, Some(&channel), &mut out).unwrap_err();
        assert!(error.contains("past the table's extent"), "{error}");
    }
}
