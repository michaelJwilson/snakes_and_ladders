//! Felsenstein pruning, ported from `python/phylo/likelihood/pruning.py`
//! (the NumPy oracle) to Rust, exposed to Python via PyO3 as
//! `phylo.oxiphylo.pruning_log_likelihood`.
//!
//! Implements eq. (pruning) and eq. (root) of `docs/tex/main.tex` (Sec.
//! "Pruning") exactly: message passing `partial[s, i] = sum_j P_ij(t) *
//! child_partial[s, j]` over `(site, state)` arrays, post-order over the
//! topology, with the same per-node rescaling behavior as the NumPy oracle
//! (log of the scale factor accumulated separately; a site whose partial
//! likelihood vanishes entirely is left at zero rather than divided, so
//! `ln(0) = -inf` propagates instead of being masked by a spurious
//! `log_scale` contribution -- see `pruning.py`'s docstring).
//!
//! The Python wrapper (`phylo.likelihood.pruning_rust`) flattens a
//! `phylo.sim.tree.Node` topology into the arrays this module expects,
//! mirroring `pruning_torch.py`'s convention of keeping branch lengths as a
//! flat array in a defined order rather than read off `Node.branch_length`
//! inside the accelerated call, even though Rust has no autograd graph to
//! protect.
//!
//! The recursion itself (`pruning_log_likelihood_impl`) is plain Rust with
//! no PyO3 types, returning `Result<f64, String>`; `pruning_log_likelihood`
//! is a thin `#[pyfunction]` wrapper converting `Err` to a Python
//! `ValueError`. Keeping the implementation PyO3-free is what lets
//! `cargo test` exercise it directly: the crate's `extension-module`
//! feature (required so `maturin`-built `.so`s link against whatever
//! `libpython` loads them, not one pinned at compile time) leaves Python
//! runtime symbols unresolved for a standalone `cargo test` binary, so any
//! path that touches `PyResult`/`PyErr` fails to link outside of a real
//! Python process.

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Closed-form k-state Jukes-Cantor transition probabilities P(t), eq. (jc)
/// of `docs/tex/main.tex`, ported from `phylo.sim.jc.jc_transition_probabilities`.
///
/// Returns a row-major `k * k` matrix flattened into a `Vec<f64>`; entry
/// `i * k + j` is Pr(state j at the branch's end | state i at its start).
fn jc_transition_probabilities(t: f64, k: usize) -> Vec<f64> {
    let kf = k as f64;
    let decay = (-kf * t / (kf - 1.0)).exp();
    let off_diagonal = (1.0 - decay) / kf;
    let diagonal = 1.0 / kf + (kf - 1.0) / kf * decay;

    let mut p = vec![off_diagonal; k * k];
    for i in 0..k {
        p[i * k + i] = diagonal;
    }
    p
}

/// An alignment as it crosses the FFI boundary: one row per leaf, plus the
/// index saying which row each node reads.
///
/// The three fields travel together and are meaningless apart, so they are
/// one argument rather than three. `states` is row-major `n_rows * n_sites`
/// with entries in `[0, k)`; a flat slice rather than a `Vec<Vec<i64>>` so
/// the caller's NumPy buffer can be borrowed instead of copied element by
/// element -- at `n = 200, L = 11 000` the nested form boxes 2.2 million
/// Python integers on the way in, which is the whole of the gap issue #232
/// measured. `n_sites` is passed rather than inferred, so a leaf whose row
/// is genuinely empty is a length mismatch rather than an unnoticed
/// zero-site alignment. `row` has one entry per node -- the row of `states`
/// holding that node's observations, or `-1` for an internal node -- and the
/// indirection is what lets `states` carry one row per *leaf* rather than
/// one per node, which at 200 taxa is half the array.
#[derive(Clone, Copy)]
pub struct LeafObservations<'a> {
    /// Row-major `n_rows * n_sites` observed states, one row per leaf.
    pub states: &'a [i64],
    /// Row width of `states`.
    pub n_sites: usize,
    /// Per node, its row of `states`, or `-1` if the node is internal.
    pub row: &'a [i64],
}

/// Total log-likelihood of an alignment under the k-state Jukes-Cantor model,
/// computed by Felsenstein pruning -- the Rust port of
/// `phylo.likelihood.pruning.log_likelihood`. Plain Rust (no PyO3 types) so
/// `cargo test` can call it directly; see the module docs for why.
///
/// # Parameters
/// - `branch_length`: length `n_nodes`, the branch length above node `i`
///   (the root's entry is present but unused, matching the NumPy oracle's
///   root having no incoming edge).
/// - `children`: length `n_nodes`; `children[i]` holds the indices of node
///   `i`'s children. Every child index must be `< i` -- callers (the Python
///   wrapper) must supply nodes in post-order, children before parents, so
///   a single forward pass over `0..n_nodes` suffices with no recursion.
///   The tree's root is `n_nodes - 1`.
/// - `observations`: the alignment, as [`LeafObservations`].
/// - `k`: number of states.
/// - `pi`: root state distribution, length `k`.
/// - `rescale`: whether to rescale partial likelihoods per node, log of the
///   scale factor accumulated separately (docs/tex/main.tex, Sec.
///   "Pruning"). Disabling underflows for realistic (site, taxa) sizes.
///
/// # Errors
/// Returns `Err` with a message if array lengths are inconsistent, `k < 2`,
/// `pi` does not have length `k`, a branch length is negative, a leaf state
/// is outside `[0, k)`, or no leaf provides `n_sites`.
///
/// `pub` (not just `pub(crate)`) so `benches/oxiphylo_bench.rs` can call it
/// directly, staying PyO3-free for the same link-time reason unit tests do
/// (see the module docs).
pub fn pruning_log_likelihood_impl(
    branch_length: &[f64],
    children: &[Vec<usize>],
    observations: LeafObservations<'_>,
    k: usize,
    pi: &[f64],
    rescale: bool,
) -> Result<f64, String> {
    let LeafObservations {
        states: leaf_states,
        n_sites,
        row: leaf_row,
    } = observations;
    let n_nodes = children.len();
    if branch_length.len() != n_nodes {
        return Err(format!(
            "branch_length has length {}, expected {n_nodes} (one per node)",
            branch_length.len()
        ));
    }
    if leaf_row.len() != n_nodes {
        return Err(format!(
            "leaf_row has length {}, expected {n_nodes} (one per node)",
            leaf_row.len()
        ));
    }
    if n_sites == 0 {
        return Err("n_sites is 0, expected at least one site".to_string());
    }
    if !leaf_states.len().is_multiple_of(n_sites) {
        return Err(format!(
            "leaf_states has length {}, not a multiple of n_sites {n_sites}",
            leaf_states.len()
        ));
    }
    if k < 2 {
        return Err(format!("k must be >= 2, got {k}"));
    }
    if pi.len() != k {
        return Err(format!("pi has length {}, expected {k}", pi.len()));
    }
    if n_nodes == 0 {
        return Err("tree has no nodes".to_string());
    }

    let n_rows = leaf_states.len() / n_sites;

    let mut partials: Vec<Vec<f64>> = Vec::with_capacity(n_nodes);
    let mut log_scale = vec![0.0f64; n_sites];

    for idx in 0..n_nodes {
        let is_leaf = children[idx].is_empty();
        let partial = if is_leaf {
            let row = leaf_row[idx];
            if row < 0 || row as usize >= n_rows {
                return Err(format!(
                    "leaf at node {idx} has leaf_row {row}, expected [0, {n_rows})"
                ));
            }
            let start = row as usize * n_sites;
            let states = &leaf_states[start..start + n_sites];
            let mut partial = vec![0.0f64; n_sites * k];
            for (s, &state) in states.iter().enumerate() {
                if state < 0 || state as usize >= k {
                    return Err(format!(
                        "leaf at node {idx}, site {s} has state {state}, expected [0, {k})"
                    ));
                }
                partial[s * k + state as usize] = 1.0;
            }
            partial
        } else {
            let mut partial = vec![1.0f64; n_sites * k];
            for &child_idx in &children[idx] {
                if child_idx >= idx {
                    return Err(format!(
                        "node {idx} has child index {child_idx}, expected < {idx} \
                         (nodes must be in post-order, children before parents)"
                    ));
                }
                let t = branch_length[child_idx];
                if t < 0.0 {
                    return Err(format!(
                        "branch_length at node {child_idx} is {t}, expected >= 0"
                    ));
                }
                let transition = jc_transition_probabilities(t, k);
                let child_partial = &partials[child_idx];
                // message[s, i] = sum_j P_ij(t) * L_child(s, j) -- eq. (pruning).
                for s in 0..n_sites {
                    let child_row = &child_partial[s * k..s * k + k];
                    for i in 0..k {
                        let mut acc = 0.0f64;
                        let transition_row = &transition[i * k..i * k + k];
                        for j in 0..k {
                            acc += transition_row[j] * child_row[j];
                        }
                        partial[s * k + i] *= acc;
                    }
                }
            }

            // A child's partial is read exactly once, by this parent, and the
            // post-order guarantees no later node reads it. Releasing it here
            // holds one partial per *open* node rather than one per node: at
            // `n = 200, L = 11 000, k = 4` that is the difference between 140
            // MB live and a few MB.
            for &child_idx in &children[idx] {
                partials[child_idx] = Vec::new();
            }

            if rescale {
                for s in 0..n_sites {
                    let row = &mut partial[s * k..s * k + k];
                    let scale = row.iter().copied().fold(0.0f64, f64::max);
                    // A site with scale == 0 has zero likelihood under the
                    // model; leave it at 0 rather than dividing, so
                    // log(0) = -inf propagates correctly instead of being
                    // masked by a spurious log_scale contribution -- see
                    // pruning.py's docstring for the identical rationale.
                    if scale > 0.0 {
                        for v in row.iter_mut() {
                            *v /= scale;
                        }
                        log_scale[s] += scale.ln();
                    }
                }
            }
            partial
        };
        partials.push(partial);
    }

    let root_partial = &partials[n_nodes - 1];
    let mut total_log_likelihood = 0.0f64;
    for s in 0..n_sites {
        let mut site_likelihood = 0.0f64;
        for i in 0..k {
            site_likelihood += root_partial[s * k + i] * pi[i];
        }
        total_log_likelihood += site_likelihood.ln() + log_scale[s];
    }

    Ok(total_log_likelihood)
}

/// PyO3 boundary for [`pruning_log_likelihood_impl`], `Err` mapped to a
/// Python `ValueError`. See the free function's docs for the algorithm and
/// argument shapes.
///
/// **Arrays are borrowed, not copied.** The previous binding took
/// `Vec<Vec<i64>>`, so PyO3 built a Python integer per observed state on the
/// way in. That cost grows with `n * L` while the kernel's advantage does
/// not, which is why the caller-visible speedup decayed from 1.8x at
/// `n = 10, L = 1 000` to 1.0x at `n = 200, L = 11 000` (issue #232).
/// `rust-numpy` hands over the buffer itself.
#[pyfunction]
#[pyo3(signature = (branch_length, children, leaf_states, leaf_row, k, pi, rescale))]
pub fn pruning_log_likelihood(
    branch_length: PyReadonlyArray1<'_, f64>,
    children: Vec<Vec<usize>>,
    leaf_states: PyReadonlyArray2<'_, i64>,
    leaf_row: Vec<i64>,
    k: usize,
    pi: PyReadonlyArray1<'_, f64>,
    rescale: bool,
) -> PyResult<f64> {
    // `as_slice` succeeds only for a C-contiguous array, so a borrow with the
    // wrong stride is impossible rather than merely unlikely -- the same
    // contract `sampling::sample_rows` states, and the wrapper normalizes with
    // `ascontiguousarray` before calling, which is free when the array already
    // is one.
    let n_sites = leaf_states.shape()[1];
    let leaf_states = leaf_states.as_slice()?;
    let branch_length = branch_length.as_slice()?;
    let pi = pi.as_slice()?;
    let observations = LeafObservations {
        states: leaf_states,
        n_sites,
        row: &leaf_row,
    };
    pruning_log_likelihood_impl(branch_length, &children, observations, k, pi, rescale)
        .map_err(PyValueError::new_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jc_transition_probabilities_rows_sum_to_one() {
        for &t in &[0.0, 0.1, 0.5, 2.0] {
            let p = jc_transition_probabilities(t, 4);
            for i in 0..4 {
                let row_sum: f64 = p[i * 4..i * 4 + 4].iter().sum();
                assert!((row_sum - 1.0).abs() < 1e-12, "row {i} sums to {row_sum}");
            }
        }
    }

    #[test]
    fn test_jc_transition_probabilities_at_zero_is_identity() {
        let p = jc_transition_probabilities(0.0, 4);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((p[i * 4 + j] - expected).abs() < 1e-12);
            }
        }
    }

    /// Two-leaf, one-site tree, hand-computed against eq. (pruning)/(root):
    /// root -> {A (t=0.1, state 0), B (t=0.2, state 1)}, k=2, pi uniform.
    /// L_root(i) = P(t_A)[i, 0] * P(t_B)[i, 1]; site likelihood = sum_i
    /// pi[i] * L_root(i).
    #[test]
    fn test_two_leaf_tree_matches_hand_computation() {
        let k = 2usize;
        let t_a = 0.1f64;
        let t_b = 0.2f64;
        // node 0 = leaf A, node 1 = leaf B, node 2 = root.
        let branch_length = vec![t_a, t_b, 0.0];
        let children = vec![vec![], vec![], vec![0usize, 1usize]];
        let leaf_states = vec![0i64, 1i64];
        let leaf_row = vec![0i64, 1, -1];
        let pi = vec![0.5, 0.5];

        let observations = LeafObservations {
            states: &leaf_states,
            n_sites: 1,
            row: &leaf_row,
        };
        let ll = pruning_log_likelihood_impl(&branch_length, &children, observations, k, &pi, true)
            .unwrap();

        let p_a = jc_transition_probabilities(t_a, k);
        let p_b = jc_transition_probabilities(t_b, k);
        let mut expected_site_likelihood = 0.0f64;
        for i in 0..k {
            let l_root_i = p_a[i * k] * p_b[i * k + 1];
            expected_site_likelihood += pi[i] * l_root_i;
        }
        let expected = expected_site_likelihood.ln();

        assert!(
            (ll - expected).abs() < 1e-12,
            "got {ll}, expected {expected}"
        );
    }

    #[test]
    fn test_rejects_pi_with_wrong_length() {
        let branch_length = vec![0.1, 0.0];
        let children = vec![vec![], vec![0usize]];
        let leaf_states = vec![0i64];
        let observations = LeafObservations {
            states: &leaf_states,
            n_sites: 1,
            row: &[0i64, -1],
        };
        let err = pruning_log_likelihood_impl(
            &branch_length,
            &children,
            observations,
            4,
            &[0.5, 0.5],
            true,
        )
        .unwrap_err();
        assert!(err.contains("pi has length"));
    }

    #[test]
    fn test_rejects_negative_branch_length() {
        let branch_length = vec![-0.1, 0.2, 0.0];
        let children = vec![vec![], vec![], vec![0usize, 1usize]];
        let leaf_states = vec![0i64, 1i64];
        let observations = LeafObservations {
            states: &leaf_states,
            n_sites: 1,
            row: &[0i64, 1, -1],
        };
        let err = pruning_log_likelihood_impl(
            &branch_length,
            &children,
            observations,
            2,
            &[0.5, 0.5],
            true,
        )
        .unwrap_err();
        assert!(err.contains("expected >= 0"));
    }
}
