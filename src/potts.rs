//! Single-site heat-bath sweeps for the Potts model, ported from
//! `python/snakes_and_ladders/search/potts_mcmc.py::_single_site_sweep` (the
//! oracle) and exposed to Python as
//! `snakes_and_ladders.oxi_snakes_and_ladders.single_site_sweeps`.
//!
//! **Why this one is a Rust port.** Issue #232 profiled the sweep as the one
//! place a Python-level loop dominates: 100 sweeps on a 32x32 periodic lattice
//! (1024 nodes, 2048 edges) take 1.05 s against 0.064 s at 8x8, linear in
//! nodes with the constant set by interpreter overhead rather than arithmetic.
//! Each site update is five NumPy calls to move one spin. Root `CLAUDE.md`
//! reserves the Rust backend for exactly this: control flow over an adjacency
//! structure, with no array arithmetic for NumPy to vectorize.
//!
//! **This does not replace the oracle, and the reason is arithmetic.** The
//! Python sweep calls `np.exp` and `np.searchsorted`. Rust's `f64::exp` agrees
//! with NumPy's SIMD implementation to within a unit in the last place, not
//! bit-exactly, and `searchsorted` is a threshold: one draw landing across a
//! boundary that moved by 1 ulp picks a different state, and from that step
//! the two chains are unrelated rather than approximately equal. Replacing the
//! Python path would move every autocorrelation figure `STATUS.md` pins, every
//! committed notebook output that reads a chain, and the goodness-of-fit
//! fixtures. So this lands beside the oracle, in the shape `maxflow`,
//! `pruning` and `sampling` already establish, and which caller uses which is
//! a later judgement with its own evidence (issue #246).
//!
//! **The uniforms are drawn in Python and passed in.** This module holds no
//! generator, for the reason `sampling.rs` states: `snakes_and_ladders.sim`'s
//! reproducibility contract is that a seeded generator determines the result,
//! and a second stream inside Rust would break it silently.
//!
//! The adjacency arrives flattened in compressed-row form -- `offsets` marking
//! each node's slice of `neighbours` and `couplings` -- rather than as nested
//! lists, because #232 measured a nested binding boxing one Python object per
//! element and that cost growing with the problem while the kernel's advantage
//! does not.

use numpy::{PyReadonlyArray1, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Run `n_sweeps` heat-bath sweeps over `state`, in place.
///
/// Each sweep visits every site in index order and redraws it from its exact
/// conditional given its current neighbours,
/// `p(s_i = k | rest) proportional to exp(h_k + sum_j J_ij delta(k, s_j))`,
/// consuming one uniform per site per sweep in the same order the oracle does.
///
/// # Parameters
/// - `state`: current configuration, length `n_nodes`, updated in place.
/// - `field`: external field `h`, length `n_states`.
/// - `offsets`: length `n_nodes + 1`; node `i`'s neighbours are
///   `neighbours[offsets[i]..offsets[i + 1]]`, with `couplings` in step.
/// - `neighbours`, `couplings`: the flattened adjacency, each of length
///   `offsets[n_nodes]`. Every edge appears twice, once from each end.
/// - `draws`: `n_sweeps * n_nodes` uniforms in `[0, 1)`, drawn by the caller.
///
/// # Errors
/// Returns `Err` describing the first violated precondition: a ragged
/// adjacency, a draw count that does not match the sweeps requested, a state
/// or neighbour index outside its array, or an empty alphabet.
///
/// `pub` so `cargo test` can exercise it without linking Python, per
/// `src/pruning.rs`'s module docs.
pub fn single_site_sweeps_impl(
    state: &mut [i64],
    field: &[f64],
    offsets: &[usize],
    neighbours: &[i64],
    couplings: &[f64],
    draws: &[f64],
    n_sweeps: usize,
) -> Result<(), String> {
    let n_nodes = state.len();
    let n_states = field.len();
    if n_states == 0 {
        return Err("field is empty, so there are no states to draw from".to_string());
    }
    if offsets.len() != n_nodes + 1 {
        return Err(format!(
            "offsets has length {}, expected {} (one per node, plus the end)",
            offsets.len(),
            n_nodes + 1
        ));
    }
    if neighbours.len() != couplings.len() {
        return Err(format!(
            "neighbours has length {} and couplings {}; they index in step",
            neighbours.len(),
            couplings.len()
        ));
    }
    if offsets[n_nodes] != neighbours.len() {
        return Err(format!(
            "offsets ends at {} but the adjacency has {} entries",
            offsets[n_nodes],
            neighbours.len()
        ));
    }
    if draws.len() != n_sweeps * n_nodes {
        return Err(format!(
            "draws has length {}, expected {n_sweeps} * {n_nodes}",
            draws.len()
        ));
    }
    for (node, &value) in state.iter().enumerate() {
        if value < 0 || value as usize >= n_states {
            return Err(format!(
                "state at node {node} is {value}, expected [0, {n_states})"
            ));
        }
    }

    // One scratch buffer for the whole run: the conditional is rebuilt per
    // site, and allocating it per site is the cost the port exists to remove.
    let mut local = vec![0.0f64; n_states];

    for sweep in 0..n_sweeps {
        for node in 0..n_nodes {
            local.copy_from_slice(field);
            for position in offsets[node]..offsets[node + 1] {
                let neighbour = neighbours[position];
                if neighbour < 0 || neighbour as usize >= n_nodes {
                    return Err(format!(
                        "adjacency entry {position} names node {neighbour}, \
                         expected [0, {n_nodes})"
                    ));
                }
                local[state[neighbour as usize] as usize] += couplings[position];
            }

            // Shift by the maximum before exponentiating, as the oracle does:
            // the field and the accumulated couplings are unbounded above, and
            // exp of the raw sum overflows well inside the couplings this
            // repository samples at.
            let shift = local.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut total = 0.0f64;
            for value in local.iter_mut() {
                *value = (*value - shift).exp();
                total += *value;
                // The cumulative sum in place, so the search below reads the
                // same array rather than a second allocation.
                *value = total;
            }

            let target = draws[sweep * n_nodes + node] * total;
            // `searchsorted`'s left side: the first index whose cumulative
            // weight is strictly greater, clamped to the last state so a draw
            // in the rounding sliver above `total` still lands in support.
            let mut chosen = n_states - 1;
            for (index, &cumulative) in local.iter().enumerate() {
                if target < cumulative {
                    chosen = index;
                    break;
                }
            }
            state[node] = chosen as i64;
        }
    }
    Ok(())
}

/// PyO3 boundary for [`single_site_sweeps_impl`], `Err` mapped to a Python
/// `ValueError`. See the free function's docs for the algorithm and shapes.
///
/// Arrays are borrowed rather than copied, per issue #232: `as_slice` succeeds
/// only for a C-contiguous array, so a borrow with the wrong stride is
/// impossible rather than merely unlikely, and the wrapper normalizes with
/// `ascontiguousarray` before calling.
#[pyfunction]
#[pyo3(signature = (state, field, offsets, neighbours, couplings, draws, n_sweeps))]
pub fn single_site_sweeps(
    mut state: PyReadwriteArray1<'_, i64>,
    field: PyReadonlyArray1<'_, f64>,
    offsets: PyReadonlyArray1<'_, i64>,
    neighbours: PyReadonlyArray1<'_, i64>,
    couplings: PyReadonlyArray1<'_, f64>,
    draws: PyReadonlyArray1<'_, f64>,
    n_sweeps: usize,
) -> PyResult<()> {
    let offsets: Vec<usize> = offsets
        .as_slice()?
        .iter()
        .map(|&value| {
            usize::try_from(value).map_err(|_| PyValueError::new_err("offsets must be >= 0"))
        })
        .collect::<PyResult<_>>()?;
    single_site_sweeps_impl(
        state.as_slice_mut()?,
        field.as_slice()?,
        &offsets,
        neighbours.as_slice()?,
        couplings.as_slice()?,
        draws.as_slice()?,
        n_sweeps,
    )
    .map_err(PyValueError::new_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two isolated sites, no coupling, a field favouring state 1 by `ln 3`.
    /// The conditional is then `(1/4, 3/4)` exactly, so a draw below 0.25
    /// takes state 0 and one above takes state 1 -- hand-computable, and
    /// independent of this implementation.
    #[test]
    fn a_field_only_conditional_matches_the_hand_computed_split() {
        let field = vec![0.0, 3.0f64.ln()];
        let offsets = vec![0usize, 0, 0];
        let mut state = vec![0i64, 0];

        let mut low = state.clone();
        single_site_sweeps_impl(&mut low, &field, &offsets, &[], &[], &[0.1, 0.1], 1).unwrap();
        assert_eq!(low, vec![0, 0]);

        single_site_sweeps_impl(&mut state, &field, &offsets, &[], &[], &[0.9, 0.9], 1).unwrap();
        assert_eq!(state, vec![1, 1]);
    }

    /// A coupling large enough to dominate a zero field drives a neighbour to
    /// agree whatever the draw: the conditional puts all but `exp(-20)` of its
    /// mass on the neighbour's state.
    #[test]
    fn a_dominant_coupling_makes_a_neighbour_agree() {
        let field = vec![0.0, 0.0];
        let offsets = vec![0usize, 1, 2];
        let neighbours = vec![1i64, 0];
        let couplings = vec![20.0, 20.0];
        let mut state = vec![0i64, 1];

        single_site_sweeps_impl(
            &mut state,
            &field,
            &offsets,
            &neighbours,
            &couplings,
            &[0.5, 0.5],
            1,
        )
        .unwrap();

        assert_eq!(state[0], state[1], "a dominant coupling did not align them");
    }

    #[test]
    fn a_ragged_adjacency_is_refused() {
        let error = single_site_sweeps_impl(
            &mut [0i64],
            &[0.0, 0.0],
            &[0usize, 1],
            &[0i64],
            &[],
            &[0.5],
            1,
        )
        .unwrap_err();
        assert!(error.contains("index in step"), "{error}");
    }

    #[test]
    fn a_draw_count_that_does_not_match_the_sweeps_is_refused() {
        let error =
            single_site_sweeps_impl(&mut [0i64], &[0.0, 0.0], &[0usize, 0], &[], &[], &[0.5], 2)
                .unwrap_err();
        assert!(error.contains("expected 2 * 1"), "{error}");
    }

    #[test]
    fn a_state_outside_the_alphabet_is_refused() {
        let error =
            single_site_sweeps_impl(&mut [5i64], &[0.0, 0.0], &[0usize, 0], &[], &[], &[0.5], 1)
                .unwrap_err();
        assert!(error.contains("expected [0, 2)"), "{error}");
    }
}
