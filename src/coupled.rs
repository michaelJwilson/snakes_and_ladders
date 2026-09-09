//! The E step of the coupled spatio-sequential model, ported from
//! `python/snakes_and_ladders/likelihood/spatio_sequential.py` (the NumPy
//! oracle) to Rust, exposed to Python via PyO3 as
//! `snakes_and_ladders.oxi_snakes_and_ladders.class_posteriors` and
//! `...external_field`.
//!
//! **What the port is for.** At the declared 5,041-vertex instance
//! (`tests/regression/fixtures/spatio_sequential_counts/stress.yaml`) the
//! field costs `S x V x M x K` emission log-densities per sweep --- 1.0e9 at
//! bin factor 10 --- and each of those is three `lgamma` calls in the NumPy
//! path. `cProfile`'s self time there is dominated by `torch.lgamma`; the
//! numbers are in `STATUS.md`.
//!
//! **Table reads replace the special functions.** The counts are integers in
//! a range of a few thousand, so the emission log-density under state `k` of
//! class `m` is a *function of an integer* and can be tabulated once per call:
//! `total_table[y, m, k]` and `success_table[z, m, k]`, built by the NumPy
//! families themselves, so the kernel does two loads and an add where the
//! oracle does three `lgamma` calls. Tabulating in Python rather than here
//! also means the tables are the oracle's own arithmetic, and this crate
//! needs no special-function dependency.
//!
//! **The tables are indexed count-major.** `[y][m][k]` and not `[m][k][y]`:
//! the field's inner loop wants every class and state *at one count*, which
//! count-major makes `M x K` contiguous doubles --- 800 bytes at the declared
//! size, a handful of cache lines --- where state-major would scatter them
//! across `M x K` lines one stride apart.
//!
//! **The observations are walked position-major.** They cross the boundary in
//! the `(S, n_nodes)` layout `snakes_and_ladders.sim` already holds them in,
//! and the loops run position outer, vertex inner, so both count arrays are
//! read in stride order. The live accumulator is then one `(M, K)` block per
//! position, which stays in L1 at every declared size; a vertex-major walk
//! would read the counts with a stride of `n_nodes` instead.
//!
//! **Forward--backward is scaled, and the oracle's is in the log domain.**
//! The class density is a sum over a class's members, so at 500 members it
//! runs to thousands in magnitude and `exp` of it underflows: each position's
//! row maximum is subtracted before exponentiating and added back into the
//! evidence. That is a different arithmetic from the oracle's `logsumexp`,
//! which is why the two are pinned at a relative tolerance rather than
//! bitwise (`likelihood/CLAUDE.md`).
//!
//! The implementations are plain Rust with no PyO3 types so `cargo test` and
//! `benches/` can link them, per `src/pruning.rs`'s module docs.

use numpy::{PyReadonlyArray1, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// The shape of one coupled E step.
///
/// Carried as one value rather than four arguments because every function
/// here takes all four and a caller that transposes two of them silently
/// computes something else.
#[derive(Clone, Copy, Debug)]
pub struct CoupledShape {
    /// `S`, sequential positions.
    pub n_positions: usize,
    /// `V`, lattice vertices.
    pub n_nodes: usize,
    /// `M`, class labels.
    pub n_classes: usize,
    /// `K`, hidden states per class.
    pub n_states: usize,
}

impl CoupledShape {
    /// `M * K`, the width of one position's block of the emission tables.
    #[inline]
    fn block(&self) -> usize {
        self.n_classes * self.n_states
    }
}

/// The emission log-density of every class and state, tabulated by count.
///
/// `total[(y * M + m) * K + k]` is `log p(y | class m, state k)` in the first
/// channel and `success[(z * M + m) * K + k]` the same in the second. A count
/// past a table's extent is a caller error and is refused, never clamped: the
/// table is built from the counts it will be indexed by.
pub struct EmissionTables<'a> {
    /// The first channel's table, `n_totals * M * K`.
    pub total: &'a [f64],
    /// The second channel's table, `n_successes * M * K`.
    pub success: &'a [f64],
}

impl EmissionTables<'_> {
    /// Check that both tables are whole numbers of blocks and cover the counts.
    fn validate(
        &self,
        shape: &CoupledShape,
        totals: &[u16],
        successes: &[u16],
    ) -> Result<(), String> {
        let block = shape.block();
        if block == 0 {
            return Err("a coupled instance has at least one class and one state".to_string());
        }
        for (name, table, counts) in [
            ("total", self.total, totals),
            ("success", self.success, successes),
        ] {
            if !table.len().is_multiple_of(block) {
                return Err(format!(
                    "the {name} table has {} entries, not a multiple of M * K = {block}",
                    table.len()
                ));
            }
            let extent = table.len() / block;
            match counts.iter().max() {
                Some(&largest) if usize::from(largest) >= extent => {
                    return Err(format!(
                        "a {name} count of {largest} is past the table's extent {extent}"
                    ));
                }
                _ => {}
            }
        }
        Ok(())
    }
}

/// Accumulate every class's summed member scores, `(M, S, K)` row-major.
///
/// The `class_log_density` of the oracle: given the labels the classes
/// decouple, and a vertex contributes to its own class alone.
fn class_log_density(
    shape: &CoupledShape,
    tables: &EmissionTables<'_>,
    totals: &[u16],
    successes: &[u16],
    labels: &[i64],
    density: &mut [f64],
) {
    let (n_positions, n_nodes, n_states) = (shape.n_positions, shape.n_nodes, shape.n_states);
    let block = shape.block();
    density.fill(0.0);
    for s in 0..n_positions {
        let row = s * n_nodes;
        for v in 0..n_nodes {
            let m = labels[v] as usize;
            let total = &tables.total[usize::from(totals[row + v]) * block + m * n_states..];
            let success = &tables.success[usize::from(successes[row + v]) * block + m * n_states..];
            let into = &mut density[(m * n_positions + s) * n_states..][..n_states];
            for k in 0..n_states {
                into[k] += total[k] + success[k];
            }
        }
    }
}

/// Scaled forward--backward on one chain, writing the posterior and the pairwise.
///
/// Returns the chain's log evidence. `density` is `(T, K)` row-major.
fn forward_backward(
    density: &[f64],
    log_initial: &[f64],
    log_transition: &[f64],
    n_states: usize,
    posterior: &mut [f64],
    pairwise: &mut [f64],
) -> f64 {
    let length = density.len() / n_states;
    // The transition and initial probabilities, exponentiated once rather
    // than once per position: the chain is `S` long and the matrix is not.
    let transition: Vec<f64> = log_transition.iter().map(|value| value.exp()).collect();
    // Each position's emission weights with its row maximum divided out. The
    // density is a sum over a class's members, so at hundreds of members it
    // runs to thousands in magnitude and `exp` of it underflows; the maximum
    // is added back into the evidence, where it belongs.
    let mut weight = vec![0.0f64; density.len()];
    let mut evidence = 0.0f64;
    for t in 0..length {
        let row = &density[t * n_states..][..n_states];
        let largest = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        evidence += largest;
        for (k, value) in weight[t * n_states..][..n_states].iter_mut().enumerate() {
            *value = (row[k] - largest).exp();
        }
    }

    let mut alpha = vec![0.0f64; density.len()];
    let mut scale = vec![0.0f64; length];
    for (k, value) in alpha[..n_states].iter_mut().enumerate() {
        *value = log_initial[k].exp() * weight[k];
    }
    let first: f64 = alpha[..n_states].iter().sum();
    scale[0] = first;
    evidence += first.ln();
    for value in alpha[..n_states].iter_mut() {
        *value /= first;
    }
    for t in 1..length {
        let (previous, current) = alpha.split_at_mut(t * n_states);
        let previous = &previous[(t - 1) * n_states..];
        for (j, cell) in current[..n_states].iter_mut().enumerate() {
            let mut total = 0.0;
            for (i, before) in previous.iter().enumerate() {
                total += before * transition[i * n_states + j];
            }
            *cell = total * weight[t * n_states + j];
        }
        let sum: f64 = current[..n_states].iter().sum();
        scale[t] = sum;
        evidence += sum.ln();
        for value in current[..n_states].iter_mut() {
            *value /= sum;
        }
    }

    let mut beta = vec![1.0f64; density.len()];
    for t in (0..length.saturating_sub(1)).rev() {
        for i in 0..n_states {
            let mut total = 0.0;
            for j in 0..n_states {
                total += transition[i * n_states + j]
                    * weight[(t + 1) * n_states + j]
                    * beta[(t + 1) * n_states + j];
            }
            beta[t * n_states + i] = total / scale[t + 1];
        }
    }

    for (index, cell) in posterior.iter_mut().enumerate() {
        *cell = alpha[index] * beta[index];
    }
    for t in 1..length {
        for i in 0..n_states {
            for j in 0..n_states {
                pairwise[((t - 1) * n_states + i) * n_states + j] = alpha[(t - 1) * n_states + i]
                    * transition[i * n_states + j]
                    * weight[t * n_states + j]
                    * beta[t * n_states + j]
                    / scale[t];
            }
        }
    }
    evidence
}

/// The coupled E step: every class's state posterior, pairwise and evidence.
///
/// # Parameters
/// - `shape`: `S`, `V`, `M`, `K`.
/// - `tables`: the count-indexed emission log-densities.
/// - `totals`, `successes`: `S * V` counts, position-major.
/// - `labels`: one class per vertex, entries in `[0, M)`.
/// - `log_initial`: `M * K`.
/// - `log_transition`: `M * K * K`, rows the source state.
/// - `posterior`: written, `M * S * K`.
/// - `pairwise`: written, `M * (S - 1) * K * K`.
/// - `log_evidence`: written, `M`.
///
/// # Returns
/// `Ok(())`, or `Err` naming the first violated precondition.
#[allow(clippy::too_many_arguments)]
pub fn class_posteriors_into(
    shape: CoupledShape,
    tables: &EmissionTables<'_>,
    totals: &[u16],
    successes: &[u16],
    labels: &[i64],
    log_initial: &[f64],
    log_transition: &[f64],
    posterior: &mut [f64],
    pairwise: &mut [f64],
    log_evidence: &mut [f64],
) -> Result<(), String> {
    check_inputs(&shape, tables, totals, successes, labels)?;
    let (n_positions, n_classes, n_states) = (shape.n_positions, shape.n_classes, shape.n_states);
    let expected = n_classes * n_positions * n_states;
    if posterior.len() != expected {
        return Err(format!(
            "posterior has {} entries, expected M * S * K = {expected}",
            posterior.len()
        ));
    }
    let pairs = n_classes * n_positions.saturating_sub(1) * n_states * n_states;
    if pairwise.len() != pairs {
        return Err(format!(
            "pairwise has {} entries, expected M * (S - 1) * K * K = {pairs}",
            pairwise.len()
        ));
    }
    if log_evidence.len() != n_classes {
        return Err(format!(
            "log_evidence has {} entries, expected M = {n_classes}",
            log_evidence.len()
        ));
    }

    let mut density = vec![0.0f64; expected];
    class_log_density(&shape, tables, totals, successes, labels, &mut density);
    let per_class = n_positions * n_states;
    let per_class_pairs = n_positions.saturating_sub(1) * n_states * n_states;
    for m in 0..n_classes {
        log_evidence[m] = forward_backward(
            &density[m * per_class..][..per_class],
            &log_initial[m * n_states..][..n_states],
            &log_transition[m * n_states * n_states..][..n_states * n_states],
            n_states,
            &mut posterior[m * per_class..][..per_class],
            &mut pairwise[m * per_class_pairs..][..per_class_pairs],
        );
    }
    Ok(())
}

/// The external field `H[v, m]`: minus the posterior-expected emission score.
///
/// # Parameters
/// - `weights`: the state posterior transposed to `(S, M, K)` row-major, so
///   one position's `M x K` weights are contiguous beside the tables' rows.
/// - `field`: written, `V * M`.
///
/// # Returns
/// `Ok(())`, or `Err` naming the first violated precondition.
pub fn external_field_into(
    shape: CoupledShape,
    tables: &EmissionTables<'_>,
    totals: &[u16],
    successes: &[u16],
    weights: &[f64],
    field: &mut [f64],
) -> Result<(), String> {
    let block = shape.block();
    if block > 0 {
        tables.validate(&shape, totals, successes)?;
    }
    let (n_positions, n_nodes, n_states) = (shape.n_positions, shape.n_nodes, shape.n_states);
    if totals.len() != n_positions * n_nodes || successes.len() != totals.len() {
        return Err(format!(
            "the counts have {} and {} entries, expected S * V = {}",
            totals.len(),
            successes.len(),
            n_positions * n_nodes
        ));
    }
    if weights.len() != n_positions * block {
        return Err(format!(
            "weights has {} entries, expected S * M * K = {}",
            weights.len(),
            n_positions * block
        ));
    }
    if field.len() != n_nodes * shape.n_classes {
        return Err(format!(
            "field has {} entries, expected V * M = {}",
            field.len(),
            n_nodes * shape.n_classes
        ));
    }

    field.fill(0.0);
    for s in 0..n_positions {
        let row = s * n_nodes;
        let weight = &weights[s * block..][..block];
        for v in 0..n_nodes {
            let total = &tables.total[usize::from(totals[row + v]) * block..][..block];
            let success = &tables.success[usize::from(successes[row + v]) * block..][..block];
            let into = &mut field[v * shape.n_classes..][..shape.n_classes];
            for (m, cell) in into.iter_mut().enumerate() {
                let mut accumulated = 0.0;
                for k in 0..n_states {
                    let index = m * n_states + k;
                    accumulated += (total[index] + success[index]) * weight[index];
                }
                *cell -= accumulated;
            }
        }
    }
    Ok(())
}

/// The preconditions both kernels share.
fn check_inputs(
    shape: &CoupledShape,
    tables: &EmissionTables<'_>,
    totals: &[u16],
    successes: &[u16],
    labels: &[i64],
) -> Result<(), String> {
    if shape.n_positions == 0 || shape.n_nodes == 0 {
        return Err("a coupled instance has at least one position and one vertex".to_string());
    }
    if totals.len() != shape.n_positions * shape.n_nodes || successes.len() != totals.len() {
        return Err(format!(
            "the counts have {} and {} entries, expected S * V = {}",
            totals.len(),
            successes.len(),
            shape.n_positions * shape.n_nodes
        ));
    }
    if labels.len() != shape.n_nodes {
        return Err(format!(
            "labels has {} entries, expected V = {}",
            labels.len(),
            shape.n_nodes
        ));
    }
    if labels
        .iter()
        .any(|&label| label < 0 || label as usize >= shape.n_classes)
    {
        return Err(format!("every label must lie in [0, {})", shape.n_classes));
    }
    tables.validate(shape, totals, successes)
}

/// Read a borrowed 1-D array as a contiguous slice, or say why it is not one.
fn borrowed<'a, T: numpy::Element>(
    array: &'a PyReadonlyArray1<'_, T>,
    name: &str,
) -> PyResult<&'a [T]> {
    array.as_slice().map_err(|_| {
        PyValueError::new_err(format!("{name} must be C-contiguous to cross the boundary"))
    })
}

/// `class_posteriors_into` as a Python binding.
///
/// Every array crosses the boundary once, contiguous and borrowed rather than
/// copied; the three results are written into the caller's own arrays for the
/// reason `src/sampling.rs` gives, that a second pass over `M x S x K` is the
/// last copy left at this size.
///
/// # Returns
/// `None`; the results are written into `posterior`, `pairwise` and
/// `log_evidence`.
///
/// # Errors
/// `ValueError` naming the first violated precondition, including a count
/// past the extent of the table that is indexed by it.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn class_posteriors(
    totals: PyReadonlyArray1<'_, u16>,
    successes: PyReadonlyArray1<'_, u16>,
    labels: PyReadonlyArray1<'_, i64>,
    total_table: PyReadonlyArray1<'_, f64>,
    success_table: PyReadonlyArray1<'_, f64>,
    log_initial: PyReadonlyArray1<'_, f64>,
    log_transition: PyReadonlyArray1<'_, f64>,
    n_positions: usize,
    n_nodes: usize,
    n_classes: usize,
    n_states: usize,
    mut posterior: PyReadwriteArray1<'_, f64>,
    mut pairwise: PyReadwriteArray1<'_, f64>,
    mut log_evidence: PyReadwriteArray1<'_, f64>,
) -> PyResult<()> {
    let tables = EmissionTables {
        total: borrowed(&total_table, "total_table")?,
        success: borrowed(&success_table, "success_table")?,
    };
    class_posteriors_into(
        CoupledShape {
            n_positions,
            n_nodes,
            n_classes,
            n_states,
        },
        &tables,
        borrowed(&totals, "totals")?,
        borrowed(&successes, "successes")?,
        borrowed(&labels, "labels")?,
        borrowed(&log_initial, "log_initial")?,
        borrowed(&log_transition, "log_transition")?,
        posterior
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("posterior must be C-contiguous"))?,
        pairwise
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("pairwise must be C-contiguous"))?,
        log_evidence
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("log_evidence must be C-contiguous"))?,
    )
    .map_err(PyValueError::new_err)
}

/// `external_field_into` as a Python binding.
///
/// # Returns
/// `None`; the result is written into `field`.
///
/// # Errors
/// `ValueError` naming the first violated precondition.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn external_field(
    totals: PyReadonlyArray1<'_, u16>,
    successes: PyReadonlyArray1<'_, u16>,
    total_table: PyReadonlyArray1<'_, f64>,
    success_table: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    n_positions: usize,
    n_nodes: usize,
    n_classes: usize,
    n_states: usize,
    mut field: PyReadwriteArray1<'_, f64>,
) -> PyResult<()> {
    let tables = EmissionTables {
        total: borrowed(&total_table, "total_table")?,
        success: borrowed(&success_table, "success_table")?,
    };
    external_field_into(
        CoupledShape {
            n_positions,
            n_nodes,
            n_classes,
            n_states,
        },
        &tables,
        borrowed(&totals, "totals")?,
        borrowed(&successes, "successes")?,
        borrowed(&weights, "weights")?,
        field
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("field must be C-contiguous"))?,
    )
    .map_err(PyValueError::new_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A two-position, two-vertex, one-class, two-state instance whose
    /// forward--backward can be written out by hand.
    fn tiny() -> (CoupledShape, Vec<f64>, Vec<f64>) {
        let shape = CoupledShape {
            n_positions: 2,
            n_nodes: 2,
            n_classes: 1,
            n_states: 2,
        };
        // Counts 0 and 1; one class, two states.
        let total = vec![-1.0, -2.0, -3.0, -0.5];
        let success = vec![0.0, 0.0, 0.0, 0.0];
        (shape, total, success)
    }

    #[test]
    fn posteriors_sum_to_one_at_every_position() {
        let (shape, total, success) = tiny();
        let tables = EmissionTables {
            total: &total,
            success: &success,
        };
        let totals = [0u16, 1, 1, 0];
        let successes = [0u16, 0, 0, 0];
        let labels = [0i64, 0];
        let log_initial = [(0.5f64).ln(), (0.5f64).ln()];
        let log_transition = [(0.7f64).ln(), (0.3f64).ln(), (0.4f64).ln(), (0.6f64).ln()];
        let mut posterior = vec![0.0; 4];
        let mut pairwise = vec![0.0; 4];
        let mut evidence = vec![0.0; 1];

        class_posteriors_into(
            shape,
            &tables,
            &totals,
            &successes,
            &labels,
            &log_initial,
            &log_transition,
            &mut posterior,
            &mut pairwise,
            &mut evidence,
        )
        .unwrap();

        assert!((posterior[0] + posterior[1] - 1.0).abs() < 1e-12);
        assert!((posterior[2] + posterior[3] - 1.0).abs() < 1e-12);
        assert!((pairwise.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(evidence[0] < 0.0);
    }

    #[test]
    fn a_count_past_the_table_is_refused() {
        let (shape, total, success) = tiny();
        let tables = EmissionTables {
            total: &total,
            success: &success,
        };
        let totals = [0u16, 2, 1, 0];
        let successes = [0u16, 0, 0, 0];
        let labels = [0i64, 0];
        let mut posterior = vec![0.0; 4];
        let mut pairwise = vec![0.0; 4];
        let mut evidence = vec![0.0; 1];

        let refused = class_posteriors_into(
            shape,
            &tables,
            &totals,
            &successes,
            &labels,
            &[0.0, 0.0],
            &[0.0, 0.0, 0.0, 0.0],
            &mut posterior,
            &mut pairwise,
            &mut evidence,
        );

        assert!(refused.unwrap_err().contains("past the table's extent"));
    }

    #[test]
    fn the_field_is_minus_the_expected_score() {
        let (shape, total, success) = tiny();
        let tables = EmissionTables {
            total: &total,
            success: &success,
        };
        let totals = [0u16, 1, 1, 0];
        let successes = [0u16, 0, 0, 0];
        // One class, two states, two positions: put all the weight on state 0.
        let weights = [1.0, 0.0, 1.0, 0.0];
        let mut field = vec![0.0; 2];

        external_field_into(shape, &tables, &totals, &successes, &weights, &mut field).unwrap();

        // Vertex 0 sees counts 0 then 1: -(t[0][0] + t[1][0]) = -(-1 + -3).
        assert!((field[0] - 4.0).abs() < 1e-12);
        // Vertex 1 sees counts 1 then 0.
        assert!((field[1] - 4.0).abs() < 1e-12);
    }
}
