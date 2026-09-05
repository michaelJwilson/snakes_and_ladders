//! Felsenstein pruning, ported from `python/snakes_and_ladders/likelihood/pruning.py`
//! (the NumPy oracle) to Rust, exposed to Python via PyO3 as
//! `snakes_and_ladders.oxi_snakes_and_ladders.pruning_log_likelihood`.
//!
//! Implements eq. (pruning) of `docs/tex/textbook.tex` (Sec. "Problem
//! Statement: Phylogenetic Inference", "The algorithm: pruning") exactly: message passing `partial[s, i] = sum_j P_ij(t) *
//! child_partial[s, j]` over `(site, state)` arrays, post-order over the
//! topology, with the same per-node rescaling behavior as the NumPy oracle
//! (log of the scale factor accumulated separately; a site whose partial
//! likelihood vanishes entirely is left at zero rather than divided, so
//! `ln(0) = -inf` propagates instead of being masked by a spurious
//! `log_scale` contribution -- see `pruning.py`'s docstring).
//!
//! The Python wrapper (`snakes_and_ladders.likelihood.pruning_rust`) flattens a
//! `snakes_and_ladders.sim.tree.Node` topology into the arrays this module expects,
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

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Closed-form k-state Jukes-Cantor transition probabilities P(t), the model
/// of `docs/tex/textbook.tex` Sec. "Problem Statement: Phylogenetic
/// Inference", ported from `snakes_and_ladders.sim.jc.jc_transition_probabilities`.
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

/// Total log-likelihood of an alignment under the k-state Jukes-Cantor model,
/// computed by Felsenstein pruning -- the Rust port of
/// `snakes_and_ladders.likelihood.pruning.log_likelihood`. Plain Rust (no PyO3 types) so
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
/// - `leaf_states`: length `n_nodes`; for a leaf, the observed state per
///   site (length `n_sites`, entries in `[0, k)`); for an internal node, an
///   empty vector (unused). `n_sites` is inferred from the first non-empty
///   entry.
/// - `k`: number of states.
/// - `pi`: root state distribution, length `k`.
/// - `rescale`: whether to rescale partial likelihoods per node, log of the
///   scale factor accumulated separately (docs/tex/textbook.tex, "The
///   algorithm: pruning"). Disabling underflows for realistic (site, taxa) sizes.
///
/// # Errors
/// Returns `Err` with a message if array lengths are inconsistent, `k < 2`,
/// `pi` does not have length `k`, a branch length is negative, a leaf state
/// is outside `[0, k)`, or no leaf provides `n_sites`.
///
/// `pub` (not just `pub(crate)`) so `benches/oxi_snakes_and_ladders_bench.rs` can call it
/// directly, staying PyO3-free for the same link-time reason unit tests do
/// (see the module docs).
pub fn pruning_log_likelihood_impl(
    branch_length: &[f64],
    children: &[Vec<usize>],
    leaf_states: &[Vec<i64>],
    k: usize,
    pi: &[f64],
    rescale: bool,
) -> Result<f64, String> {
    let n_nodes = children.len();
    if branch_length.len() != n_nodes {
        return Err(format!(
            "branch_length has length {}, expected {n_nodes} (one per node)",
            branch_length.len()
        ));
    }
    if leaf_states.len() != n_nodes {
        return Err(format!(
            "leaf_states has length {}, expected {n_nodes} (one per node)",
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

    let n_sites = leaf_states
        .iter()
        .find(|states| !states.is_empty())
        .map(|states| states.len())
        .ok_or_else(|| "no leaf provided any observed states".to_string())?;

    let mut partials: Vec<Vec<f64>> = Vec::with_capacity(n_nodes);
    let mut log_scale = vec![0.0f64; n_sites];

    for idx in 0..n_nodes {
        let is_leaf = children[idx].is_empty();
        let partial = if is_leaf {
            let states = &leaf_states[idx];
            if states.len() != n_sites {
                return Err(format!(
                    "leaf at node {idx} has {} observed states, expected {n_sites}",
                    states.len()
                ));
            }
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

/// PyO3 boundary for [`pruning_log_likelihood_impl`]: same arguments and
/// return value, `Err` mapped to a Python `ValueError`. See the free
/// function's docs for the algorithm and argument shapes.
#[pyfunction]
#[pyo3(signature = (branch_length, children, leaf_states, k, pi, rescale))]
pub fn pruning_log_likelihood(
    branch_length: Vec<f64>,
    children: Vec<Vec<usize>>,
    leaf_states: Vec<Vec<i64>>,
    k: usize,
    pi: Vec<f64>,
    rescale: bool,
) -> PyResult<f64> {
    pruning_log_likelihood_impl(&branch_length, &children, &leaf_states, k, &pi, rescale)
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
        let leaf_states = vec![vec![0i64], vec![1i64], vec![]];
        let pi = vec![0.5, 0.5];

        let ll = pruning_log_likelihood_impl(&branch_length, &children, &leaf_states, k, &pi, true)
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
        let leaf_states = vec![vec![0i64], vec![]];
        let err = pruning_log_likelihood_impl(
            &branch_length,
            &children,
            &leaf_states,
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
        let leaf_states = vec![vec![0i64], vec![1i64], vec![]];
        let err = pruning_log_likelihood_impl(
            &branch_length,
            &children,
            &leaf_states,
            2,
            &[0.5, 0.5],
            true,
        )
        .unwrap_err();
        assert!(err.contains("expected >= 0"));
    }
}
