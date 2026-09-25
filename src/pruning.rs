//! Felsenstein pruning, ported from `python/sal/likelihood/pruning/__init__.py`
//! (the NumPy oracle) to Rust, exposed to Python via PyO3 as
//! `sal.oxisal.pruning_log_likelihood`.
//!
//! Implements `eq:pruning` and `eq:root` of `docs/tex/textbook.tex` exactly: message passing `partial[i, s] = sum_j P_ij(t) *
//! child_partial[j, s]`, post-order over the topology, with the same
//! per-node rescaling behavior as the NumPy oracle (log of the scale factor
//! accumulated separately; a site whose partial likelihood vanishes entirely
//! is left at zero rather than divided, so `ln(0) = -inf` propagates instead
//! of being masked by a spurious `log_scale` contribution -- see
//! `pruning.py`'s docstring).
//!
//! **A partial is `(state, site)`, sites contiguous** -- states innermost is
//! what the port started from, and it made the inner loop `k` elements long
//! with a strided read of the child. Sites innermost makes every loop here a
//! pass over one contiguous row of length `n_sites`, with no early exit and
//! no data-dependent reduction, which is the shape the compiler vectorizes
//! (root `CLAUDE.md`, Vectorization and SIMD). The layout is internal: the
//! arrays crossing the FFI boundary are unchanged. The two orders were
//! measured against each other rather than argued about, and the numbers are
//! in `STATUS.md`; the rule that they be confirmed by benchmark and never by
//! asserting a vector width is why no width appears anywhere in this file.
//!
//! The Python wrapper (`sal.likelihood.pruning.rust`) flattens a
//! `sal.sim.tree.Node` topology into the arrays this module expects,
//! mirroring `pruning/torch.py`'s convention of keeping branch lengths as a
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
use rayon::prelude::*;

/// Sites per tile in the message pass.
///
/// Chosen by measurement, not by arithmetic on a cache size: the sweep is in
/// `STATUS.md`. What sets the shape is that one tile's `k` parent rows and
/// `k` child rows are live at once, so the `k * k` passes over them read L1
/// rather than memory.
const TILE: usize = 128;

/// Closed-form k-state Jukes-Cantor transition probabilities P(t), `eq:jc`
/// of `docs/tex/textbook.tex`, ported from `sal.sim.jc.jc_transition_probabilities`.
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
/// `sal.likelihood.pruning.log_likelihood`. Plain Rust (no PyO3 types) so
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
///   scale factor accumulated separately (`docs/tex/textbook.tex`,
///   `eq:pruning`). Disabling underflows for realistic (site, taxa) sizes.
///
/// # Errors
/// Returns `Err` with a message if array lengths are inconsistent, `k < 2`,
/// `pi` does not have length `k`, a branch length is negative, a leaf state
/// is outside `[0, k)`, or no leaf provides `n_sites`.
///
/// `pub` (not just `pub(crate)`) so `benches/oxisal_bench.rs` can call it
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

    // Bounds-check every leaf row before the windows start: a window runs
    // inside a `par_chunks_mut` closure, which cannot `?` out of this
    // function, and one rule for both arms beats two.
    for (idx, &row) in leaf_row.iter().enumerate() {
        if children[idx].is_empty() && (row < 0 || row as usize >= n_rows) {
            return Err(format!(
                "leaf at node {idx} has leaf_row {row}, expected [0, {n_rows})"
            ));
        }
    }

    // **Sites are the axis, and the sum over them is not.** A site's
    // likelihood depends on no other site, so a window of sites can run the
    // whole post-order traversal alone. What cannot be split is the last
    // line: `total += ln(L_s) + scale_s` left to right over every site. So a
    // window writes its own `ln(L_s) + scale_s` into `per_site[s]` and never
    // adds anything up, and one sequential pass below sums `per_site` in
    // index order -- the serial path's sum, to the bit, for the price of one
    // `f64` per site and a pass over it (issue #627).
    //
    // A `fold`/`reduce` over an indexed parallel iterator would have been the
    // obvious shape and would still reassociate: combining partials in index
    // order is `(a+b)+(c+d)`, not `((a+b)+c)+d`.
    let mut per_site = vec![0.0f64; n_sites];
    let window = if n_sites >= PARALLEL_ABOVE {
        n_sites.div_ceil(rayon::current_num_threads().max(1))
    } else {
        n_sites
    };
    let outcome: Result<(), String> = per_site
        .par_chunks_mut(window)
        .enumerate()
        .map(|(index, into)| {
            traverse_window(
                branch_length,
                children,
                leaf_states,
                n_sites,
                leaf_row,
                k,
                pi,
                rescale,
                index * window,
                into,
            )
        })
        .collect();
    outcome?;

    let mut total_log_likelihood = 0.0f64;
    for &value in per_site.iter() {
        total_log_likelihood += value;
    }
    Ok(total_log_likelihood)
}

/// Sites below this run the traversal once, on one thread.
///
/// Measured, not chosen: `STATUS.md` carries what the pool costs on each side
/// of this line, against issue #627's 117.64 ms at 8 taxa and 200,000 sites.
const PARALLEL_ABOVE: usize = 20_000;

/// The post-order traversal over one window of sites.
///
/// `into` receives `ln(L_s) + log_scale_s` for each site of the window, in
/// site order, and nothing is summed here: the caller owns the reduction, so
/// that the arithmetic does not depend on how many threads ran.
#[allow(clippy::too_many_arguments)]
fn traverse_window(
    branch_length: &[f64],
    children: &[Vec<usize>],
    leaf_states: &[i64],
    n_sites: usize,
    leaf_row: &[i64],
    k: usize,
    pi: &[f64],
    rescale: bool,
    site_start: usize,
    into: &mut [f64],
) -> Result<(), String> {
    let n_nodes = children.len();
    let n_window = into.len();

    let mut partials: Vec<Vec<f64>> = Vec::with_capacity(n_nodes);
    let mut log_scale = vec![0.0f64; n_window];
    // Two scratch rows reused for the whole traversal rather than allocated
    // per node: one holds the message a child sends for one state over one
    // tile of sites, one the per-site scale factor.
    let mut message = vec![0.0f64; TILE.min(n_window)];
    let mut scale = vec![0.0f64; n_window];

    for idx in 0..n_nodes {
        let is_leaf = children[idx].is_empty();
        let partial = if is_leaf {
            // The row is bounds-checked by the caller; the window reads its
            // own slice of it, `n_sites` being the full stride.
            let row = leaf_row[idx] as usize;
            let start = row * n_sites + site_start;
            let states = &leaf_states[start..start + n_window];
            let mut partial = vec![0.0f64; k * n_window];
            for (s, &state) in states.iter().enumerate() {
                if state < 0 || state as usize >= k {
                    return Err(format!(
                        "leaf at node {idx}, site {} has state {state}, expected [0, {k})",
                        site_start + s
                    ));
                }
                partial[state as usize * n_window + s] = 1.0;
            }
            partial
        } else {
            // Validated before the arithmetic starts, so the loops below carry
            // no bounds test that could exit them early.
            for &child_idx in &children[idx] {
                if child_idx >= idx {
                    return Err(format!(
                        "node {idx} has child index {child_idx}, expected < {idx} \
                         (nodes must be in post-order, children before parents)"
                    ));
                }
                if branch_length[child_idx] < 0.0 {
                    return Err(format!(
                        "branch_length at node {child_idx} is {}, expected >= 0",
                        branch_length[child_idx]
                    ));
                }
            }

            // One transition matrix per child, before the site loop: it does
            // not depend on the site, and `exp` inside the loop would be paid
            // once per tile.
            let transitions: Vec<Vec<f64>> = children[idx]
                .iter()
                .map(|&child_idx| jc_transition_probabilities(branch_length[child_idx], k))
                .collect();

            let mut partial = vec![0.0f64; k * n_window];
            // Sites in tiles, so a child's `k` rows and the parent's `k` rows
            // for one tile are live together in L1 while the `k * k` passes
            // over them run. Untiled, the sites-contiguous layout reads each
            // child row `k` times from memory rather than once, and at
            // 200,000 sites that traffic ate the whole of what vectorizing
            // the loop bought -- measured, and the numbers are in `STATUS.md`.
            let mut start = 0usize;
            while start < n_window {
                let width = TILE.min(n_window - start);
                for (position, transition) in transitions.iter().enumerate() {
                    let child_partial = &partials[children[idx][position]];
                    // message[i, s] = sum_j P_ij(t) * L_child(j, s) --
                    // eq:pruning, as k * k passes over one contiguous run of
                    // sites each. The products are formed in the same order
                    // the (site, state) layout formed them, so the sum is the
                    // same sum.
                    for i in 0..k {
                        let sent = &mut message[..width];
                        for j in 0..k {
                            let probability = transition[i * k + j];
                            let offset = j * n_window + start;
                            let child_row = &child_partial[offset..offset + width];
                            if j == 0 {
                                for (value, &child) in sent.iter_mut().zip(child_row) {
                                    *value = probability * child;
                                }
                            } else {
                                for (value, &child) in sent.iter_mut().zip(child_row) {
                                    *value += probability * child;
                                }
                            }
                        }
                        let offset = i * n_window + start;
                        let row = &mut partial[offset..offset + width];
                        if position == 0 {
                            row.copy_from_slice(sent);
                        } else {
                            for (value, &value_sent) in row.iter_mut().zip(sent.iter()) {
                                *value *= value_sent;
                            }
                        }
                    }
                }
                start += width;
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
                scale.copy_from_slice(&partial[0..n_window]);
                for i in 1..k {
                    let row = &partial[i * n_window..(i + 1) * n_window];
                    for (largest, &value) in scale.iter_mut().zip(row) {
                        *largest = largest.max(value);
                    }
                }
                // A site with scale == 0 has zero likelihood under the model;
                // leave its row alone rather than dividing, so log(0) = -inf
                // propagates instead of being masked by a spurious log_scale
                // contribution -- see pruning.py's docstring for the identical
                // rationale. `scale` is overwritten with the reciprocal so the
                // k rows below are one multiply each rather than one divide;
                // that reassociates a division and is why this kernel is
                // pinned to the NumPy oracle within a relative tolerance
                // rather than bitwise (`likelihood/CLAUDE.md`).
                for (accumulated, factor) in log_scale.iter_mut().zip(scale.iter_mut()) {
                    if *factor > 0.0 {
                        *accumulated += factor.ln();
                        *factor = 1.0 / *factor;
                    } else {
                        *factor = 1.0;
                    }
                }
                for i in 0..k {
                    let row = &mut partial[i * n_window..(i + 1) * n_window];
                    for (value, &reciprocal) in row.iter_mut().zip(scale.iter()) {
                        *value *= reciprocal;
                    }
                }
            }
            partial
        };
        partials.push(partial);
    }

    // eq:root, reusing `scale` as the per-site accumulator: one contiguous
    // pass per state rather than a k-long reduction per site.
    let root_partial = &partials[n_nodes - 1];
    for (site, &value) in scale.iter_mut().zip(&root_partial[0..n_window]) {
        *site = value * pi[0];
    }
    for i in 1..k {
        let row = &root_partial[i * n_window..(i + 1) * n_window];
        let weight = pi[i];
        for (site, &value) in scale.iter_mut().zip(row) {
            *site += value * weight;
        }
    }
    // Per site, not summed: the caller adds these up in index order, so the
    // total does not depend on how the sites were divided.
    for ((slot, &site_likelihood), &scaled) in
        into.iter_mut().zip(scale.iter()).zip(log_scale.iter())
    {
        *slot = site_likelihood.ln() + scaled;
    }

    Ok(())
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
#[allow(clippy::too_many_arguments)]
pub fn pruning_log_likelihood(
    py: Python<'_>,
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
    py.detach(|| {
        pruning_log_likelihood_impl(branch_length, &children, observations, k, pi, rescale)
    })
    .map_err(PyValueError::new_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A balanced tree with `n_leaves` leaves and random states, for the
    /// agreement tests below.
    fn fixture(
        n_leaves: usize,
        n_sites: usize,
        k: usize,
    ) -> (Vec<f64>, Vec<Vec<usize>>, Vec<i64>, Vec<i64>) {
        let n_nodes = 2 * n_leaves - 1;
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); n_leaves];
        for i in 0..n_leaves - 1 {
            children.push(vec![2 * i, 2 * i + 1]);
        }
        // A fixed multiplicative generator, so the fixture is the same on
        // every host and no dependency is added for it.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let branch_length: Vec<f64> = (0..n_nodes)
            .map(|_| 0.05 + (next() % 1000) as f64 / 10_000.0)
            .collect();
        let leaf_states: Vec<i64> = (0..n_leaves * n_sites)
            .map(|_| (next() % k as u64) as i64)
            .collect();
        let leaf_row: Vec<i64> = (0..n_nodes)
            .map(|i| if i < n_leaves { i as i64 } else { -1 })
            .collect();
        (branch_length, children, leaf_states, leaf_row)
    }

    /// Splitting the sites must not move a bit of the answer.
    ///
    /// The whole point of writing `ln(L_s) + scale_s` per site and summing
    /// sequentially: the result cannot depend on how many threads ran, so a
    /// run on a 4-core host and a run on a 64-core one agree exactly. A
    /// `fold`/`reduce` would not have this property, which is why the
    /// benchmark carries both and only this one is called (issue #627).
    #[test]
    fn test_the_total_does_not_depend_on_how_the_sites_were_divided() {
        let (k, n_sites) = (4usize, 50_000usize);
        let pi = vec![0.25; k];
        let (branch_length, children, leaf_states, leaf_row) = fixture(4, n_sites, k);
        let observations = || LeafObservations {
            states: &leaf_states,
            n_sites,
            row: &leaf_row,
        };
        let whole = traverse_and_sum(&branch_length, &children, observations(), k, &pi, n_sites);
        for window in [n_sites, n_sites / 2 + 1, n_sites / 3 + 1, 1024] {
            let split = traverse_and_sum(&branch_length, &children, observations(), k, &pi, window);
            assert_eq!(
                whole.to_bits(),
                split.to_bits(),
                "window {window} gave {split}, not {whole}"
            );
        }
    }

    /// The traversal at a chosen window width, summed the caller's way.
    fn traverse_and_sum(
        branch_length: &[f64],
        children: &[Vec<usize>],
        observations: LeafObservations<'_>,
        k: usize,
        pi: &[f64],
        window: usize,
    ) -> f64 {
        let n_sites = observations.n_sites;
        let mut per_site = vec![0.0f64; n_sites];
        for (index, into) in per_site.chunks_mut(window).enumerate() {
            traverse_window(
                branch_length,
                children,
                observations.states,
                n_sites,
                observations.row,
                k,
                pi,
                true,
                index * window,
                into,
            )
            .unwrap();
        }
        let mut total = 0.0f64;
        for &value in per_site.iter() {
            total += value;
        }
        total
    }

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

    /// Two-leaf, one-site tree, hand-computed against `eq:pruning` and `eq:root`:
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
