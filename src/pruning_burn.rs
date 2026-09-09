//! Felsenstein pruning under `burn`'s reverse-mode autodiff, returning the
//! log-likelihood and its gradient in the branch lengths.
//!
//! Route A of issue #449. The recursion is `src/pruning.rs`'s -- `eq:pruning`
//! and `eq:root` of `docs/tex/textbook.tex` -- rebuilt out of
//! `burn_tensor::Tensor` operations so `burn-autodiff` can tape it, over
//! `Autodiff<NdArray<f64>>`. The element type is the decision this route was
//! adopted on: `NdArray` is generic over its float element, so the tape is
//! `f64` end to end and the gradient it returns agrees with central
//! differences to 4.4e-10 relative on the four-taxon case below, which is the
//! difference quotient's own truncation rather than a narrowing to `f32`.
//!
//! **A partial is `(site, state)` here, states contiguous** -- the transpose
//! of `src/pruning.rs`'s layout. The message pass is one `matmul` against a
//! `k * k` transition rather than the tiled hand-written loop, so the layout
//! is the one `burn`'s `matmul` wants and not the one the hand-written loop
//! was measured into.
//!
//! **No general rate matrix.** `burn` exposes no matrix exponential, so this
//! route computes the closed-form Jukes-Cantor transition only; the general
//! `Q` path of `snakes_and_ladders.likelihood.pruning_torch` has no
//! counterpart here. `python/snakes_and_ladders/sandbox/pruning_burn.py`
//! refuses a rate matrix rather than ignoring one.
//!
//! Like `src/pruning.rs`, the implementation takes no PyO3 types so
//! `cargo test` and `benches/` can call it directly.
//!
//! **Behind the `sandbox` Cargo feature, and off by default.** The route lost
//! its measurement (`docs/experiments/007-pruning-gradient-routes.md`), so it
//! is conserved to keep that comparison re-runnable rather than adopted, and
//! neither the default build nor the shipped wheel links `burn`.
//! `infra/release.sh` compiles and tests it, which is what a `cfg`-gated route
//! needs to stay honest: one nothing builds stops compiling the first time a
//! neighbouring API moves, and nobody learns for months.

use crate::pruning::LeafObservations;
use burn_autodiff::Autodiff;
use burn_ndarray::{NdArray, NdArrayDevice};
use burn_tensor::{Tensor, TensorData};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// The taped `f64` CPU backend. `NdArray`'s float element parameter is what
/// makes this `f64` rather than the framework default.
type Backend = Autodiff<NdArray<f64>>;

/// Log-likelihood and its gradient in `branch_length`, by reverse-mode
/// autodiff over the pruning recursion.
///
/// # Parameters
/// - `branch_length`: length `n_nodes`, the branch above node `i`; the
///   root's entry is unused and its gradient is returned as `0`.
/// - `children`: length `n_nodes`, post-order (every child index `< i`), root
///   last -- the layout `crate::pruning` documents.
/// - `observations`: the alignment, as [`LeafObservations`].
/// - `k`: number of states.
/// - `pi`: root state distribution, length `k`.
/// - `weight`: one weight per site, or `None` for one occurrence each.
/// - `rescale`: whether to rescale partials per node, as the oracle does.
///
/// # Errors
/// Returns `Err` with a message where an array length is inconsistent, `k` is
/// below 2, a leaf state is outside `[0, k)`, or the tape yields no gradient.
pub fn pruning_gradient_impl(
    branch_length: &[f64],
    children: &[Vec<usize>],
    observations: LeafObservations<'_>,
    k: usize,
    pi: &[f64],
    weight: Option<&[f64]>,
    rescale: bool,
) -> Result<(f64, Vec<f64>), String> {
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
    if k < 2 {
        return Err(format!("k must be >= 2, got {k}"));
    }
    if pi.len() != k {
        return Err(format!("pi has length {}, expected {k}", pi.len()));
    }
    if n_nodes == 0 {
        return Err("tree has no nodes".to_string());
    }
    if let Some(w) = weight {
        if w.len() != n_sites {
            return Err(format!(
                "weight has length {}, expected {n_sites} (one per site)",
                w.len()
            ));
        }
    }

    let device = NdArrayDevice::default();
    let kf = k as f64;
    let branch = Tensor::<Backend, 1>::from_data(
        TensorData::new(branch_length.to_vec(), [n_nodes]),
        &device,
    )
    .require_grad();

    // P(t) = base + shift * exp(-k t / (k - 1)), `eq:jc` split into the two
    // constants the branch length does not enter, so each branch costs one
    // `exp` and one affine combination on the tape rather than a rebuild.
    let mut shift_data = vec![-1.0 / kf; k * k];
    for i in 0..k {
        shift_data[i * k + i] = 1.0 - 1.0 / kf;
    }
    let base =
        Tensor::<Backend, 3>::from_data(TensorData::new(vec![1.0 / kf; k * k], [1, k, k]), &device);
    let shift = Tensor::<Backend, 3>::from_data(TensorData::new(shift_data, [1, k, k]), &device);
    // Every branch's transition at once, as `pruning_torch` does it: one
    // `exp` over the branch vector rather than one per child, so the tape
    // carries the transition build once instead of once per edge.
    let decay = (branch.clone() * (-kf / (kf - 1.0)))
        .exp()
        .reshape([n_nodes, 1, 1]);
    let transitions = base + shift * decay;

    let n_rows = leaf_states.len() / n_sites;
    let mut partials: Vec<Option<Tensor<Backend, 2>>> = vec![None; n_nodes];
    let mut log_scale = Tensor::<Backend, 2>::zeros([n_sites, 1], &device);

    for idx in 0..n_nodes {
        if children[idx].is_empty() {
            let row = leaf_row[idx];
            if row < 0 || row as usize >= n_rows {
                return Err(format!(
                    "leaf at node {idx} has leaf_row {row}, expected [0, {n_rows})"
                ));
            }
            let start = row as usize * n_sites;
            let mut data = vec![0.0f64; n_sites * k];
            for (s, &state) in leaf_states[start..start + n_sites].iter().enumerate() {
                if state < 0 || state as usize >= k {
                    return Err(format!(
                        "leaf at node {idx}, site {s} has state {state}, expected [0, {k})"
                    ));
                }
                data[s * k + state as usize] = 1.0;
            }
            partials[idx] = Some(Tensor::<Backend, 2>::from_data(
                TensorData::new(data, [n_sites, k]),
                &device,
            ));
            continue;
        }

        let mut partial = Tensor::<Backend, 2>::ones([n_sites, k], &device);
        for &child in &children[idx] {
            if child >= idx {
                return Err(format!(
                    "node {idx} has child index {child}, expected < {idx} \
                     (nodes must be in post-order, children before parents)"
                ));
            }
            if branch_length[child] < 0.0 {
                return Err(format!(
                    "branch_length at node {child} is {}, expected >= 0",
                    branch_length[child]
                ));
            }
            // One range per dimension is what `burn`'s `slice` takes and
            // `transitions` is one-dimensional here; the lint reads the array
            // as a `Vec` literal it could widen.
            #[allow(clippy::single_range_in_vec_init)]
            let transition = transitions
                .clone()
                .slice([child..child + 1])
                .reshape([k, k]);
            let child_partial = partials[child]
                .as_ref()
                .ok_or_else(|| format!("node {idx} reads released child {child}"))?
                .clone();
            // message[s, i] = sum_j P_ij(t) L_child[s, j] -- eq:pruning.
            partial = partial * child_partial.matmul(transition.transpose());
        }
        // The forward values are released; the tape keeps what the backward
        // pass needs, which is exactly the cost this route is measured for.
        for &child in &children[idx] {
            partials[child] = None;
        }

        if rescale {
            let scale = partial.clone().max_dim(1);
            // A site whose partial vanishes is left undivided so ln(0) = -inf
            // propagates, as the NumPy oracle does.
            let safe = scale.clone().mask_fill(scale.lower_equal_elem(0.0), 1.0);
            partial = partial / safe.clone();
            log_scale = log_scale + safe.log();
        }
        partials[idx] = Some(partial);
    }

    let root = partials[n_nodes - 1]
        .take()
        .ok_or_else(|| "root partial was released".to_string())?;
    let pi_column = Tensor::<Backend, 2>::from_data(TensorData::new(pi.to_vec(), [k, 1]), &device);
    let site = root.matmul(pi_column).log() + log_scale; // eq:root
    let total = match weight {
        None => site.sum(),
        Some(w) => {
            let column =
                Tensor::<Backend, 2>::from_data(TensorData::new(w.to_vec(), [n_sites, 1]), &device);
            (site * column).sum()
        }
    };

    let value: Vec<f64> = total
        .to_data()
        .to_vec()
        .map_err(|error| format!("reading the log-likelihood: {error:?}"))?;
    let grads = total.backward();
    let gradient = branch
        .grad(&grads)
        .ok_or_else(|| "the tape produced no gradient for branch_length".to_string())?;
    let gradient: Vec<f64> = gradient
        .to_data()
        .to_vec()
        .map_err(|error| format!("reading the gradient: {error:?}"))?;
    Ok((value[0], gradient))
}

/// PyO3 boundary for [`pruning_gradient_impl`]; returns
/// `(log_likelihood, gradient)` with the gradient one entry per node, in the
/// caller's post-order. Arrays are borrowed, not copied, as
/// `crate::pruning::pruning_log_likelihood` documents.
// Eight at the boundary: `crate::pruning::pruning_log_likelihood`'s seven and
// the per-site `weight`. The grouping clippy asks for exists on the Rust side
// (`LeafObservations`); PyO3 takes its arguments flat.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (branch_length, children, leaf_states, leaf_row, k, pi, weight, rescale))]
pub fn pruning_gradient(
    branch_length: PyReadonlyArray1<'_, f64>,
    children: Vec<Vec<usize>>,
    leaf_states: PyReadonlyArray2<'_, i64>,
    leaf_row: Vec<i64>,
    k: usize,
    pi: PyReadonlyArray1<'_, f64>,
    weight: Option<PyReadonlyArray1<'_, f64>>,
    rescale: bool,
) -> PyResult<(f64, Vec<f64>)> {
    let n_sites = leaf_states.shape()[1];
    let leaf_states = leaf_states.as_slice()?;
    let branch_length = branch_length.as_slice()?;
    let pi = pi.as_slice()?;
    let weight = match weight.as_ref() {
        None => None,
        Some(array) => Some(array.as_slice()?),
    };
    let observations = LeafObservations {
        states: leaf_states,
        n_sites,
        row: &leaf_row,
    };
    pruning_gradient_impl(
        branch_length,
        &children,
        observations,
        k,
        pi,
        weight,
        rescale,
    )
    .map_err(PyValueError::new_err)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pruning::pruning_log_likelihood_impl;
    use std::sync::Mutex;

    /// Taken by every test that builds a tape. `burn`'s `NodeID` generator is
    /// one process-wide counter, so `tape_nodes` below can only read it as a
    /// difference while nothing else is allocating from it -- two of these
    /// tests running at once put a sibling's nodes in the count.
    static TAPE: Mutex<()> = Mutex::new(());

    /// The branch lengths, the child lists, the leaf states, the leaf rows
    /// and the site count `fixture` returns.
    type Fixture = (Vec<f64>, Vec<Vec<usize>>, Vec<i64>, Vec<i64>, usize);

    /// Four leaves under two cherries: (0,1) -> 4, (2,3) -> 5, (4,5) -> 6.
    fn fixture() -> Fixture {
        let branch_length = vec![0.1, 0.25, 0.3, 0.15, 0.05, 0.2, 0.0];
        let children = vec![
            vec![],
            vec![],
            vec![],
            vec![],
            vec![0, 1],
            vec![2, 3],
            vec![4, 5],
        ];
        let leaf_states = vec![
            0, 1, 2, 3, 0, 1, //
            1, 1, 2, 0, 3, 2, //
            2, 3, 0, 1, 1, 0, //
            3, 0, 1, 2, 2, 3,
        ];
        let leaf_row = vec![0, 1, 2, 3, -1, -1, -1];
        (branch_length, children, leaf_states, leaf_row, 6)
    }

    fn value_at(branch_length: &[f64]) -> f64 {
        let (_, children, leaf_states, leaf_row, n_sites) = fixture();
        pruning_gradient_impl(
            branch_length,
            &children,
            LeafObservations {
                states: &leaf_states,
                n_sites,
                row: &leaf_row,
            },
            4,
            &[0.25; 4],
            None,
            true,
        )
        .unwrap()
        .0
    }

    #[test]
    fn test_value_matches_the_hand_written_kernel() {
        let _taped = TAPE.lock().unwrap_or_else(|error| error.into_inner());
        let (branch_length, children, leaf_states, leaf_row, n_sites) = fixture();
        let observations = LeafObservations {
            states: &leaf_states,
            n_sites,
            row: &leaf_row,
        };
        let oracle = pruning_log_likelihood_impl(
            &branch_length,
            &children,
            observations,
            4,
            &[0.25; 4],
            true,
        )
        .unwrap();
        let taped = value_at(&branch_length);
        assert!(
            ((taped - oracle) / oracle).abs() < 1e-12,
            "burn {taped} against the hand-written kernel {oracle}"
        );
    }

    #[test]
    fn test_gradient_matches_central_differences() {
        let _taped = TAPE.lock().unwrap_or_else(|error| error.into_inner());
        let (branch_length, children, leaf_states, leaf_row, n_sites) = fixture();
        let (_, gradient) = pruning_gradient_impl(
            &branch_length,
            &children,
            LeafObservations {
                states: &leaf_states,
                n_sites,
                row: &leaf_row,
            },
            4,
            &[0.25; 4],
            None,
            true,
        )
        .unwrap();
        let step = 1e-6;
        for index in 0..branch_length.len() - 1 {
            let mut up = branch_length.clone();
            let mut down = branch_length.clone();
            up[index] += step;
            down[index] -= step;
            let difference = (value_at(&up) - value_at(&down)) / (2.0 * step);
            assert!(
                ((gradient[index] - difference) / difference).abs() < 1e-8,
                "branch {index}: taped {} against central difference {difference}",
                gradient[index]
            );
        }
    }

    #[test]
    fn test_root_branch_has_no_gradient() {
        let _taped = TAPE.lock().unwrap_or_else(|error| error.into_inner());
        let (branch_length, children, leaf_states, leaf_row, n_sites) = fixture();
        let (_, gradient) = pruning_gradient_impl(
            &branch_length,
            &children,
            LeafObservations {
                states: &leaf_states,
                n_sites,
                row: &leaf_row,
            },
            4,
            &[0.25; 4],
            None,
            true,
        )
        .unwrap();
        assert_eq!(gradient[children.len() - 1], 0.0);
    }

    #[test]
    fn test_weights_reproduce_a_repeated_alignment() {
        let _taped = TAPE.lock().unwrap_or_else(|error| error.into_inner());
        let (branch_length, children, leaf_states, leaf_row, n_sites) = fixture();
        let weight = vec![2.0f64; n_sites];
        let observations = LeafObservations {
            states: &leaf_states,
            n_sites,
            row: &leaf_row,
        };
        let (weighted, weighted_gradient) = pruning_gradient_impl(
            &branch_length,
            &children,
            observations,
            4,
            &[0.25; 4],
            Some(&weight),
            true,
        )
        .unwrap();
        let (plain, plain_gradient) = pruning_gradient_impl(
            &branch_length,
            &children,
            LeafObservations {
                states: &leaf_states,
                n_sites,
                row: &leaf_row,
            },
            4,
            &[0.25; 4],
            None,
            true,
        )
        .unwrap();
        assert!((weighted - 2.0 * plain).abs() < 1e-9);
        for (weighted_entry, plain_entry) in weighted_gradient.iter().zip(&plain_gradient) {
            assert!((weighted_entry - 2.0 * plain_entry).abs() < 1e-9);
        }
    }

    /// Nodes `burn` puts on its tape for one gradient, counted by the
    /// difference between two fresh `NodeID`s: the generator is a process-wide
    /// counter, so the ids allocated between two calls are exactly the tape's
    /// (the run is single-threaded, as every measurement in this repository
    /// is).
    fn tape_nodes(n_leaves: usize) -> usize {
        let n_nodes = 2 * n_leaves - 1;
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); n_leaves];
        let mut frontier: Vec<usize> = (0..n_leaves).collect();
        let mut next = n_leaves;
        while frontier.len() > 1 {
            let mut parents = Vec::new();
            for pair in frontier.chunks(2) {
                children.push(pair.to_vec());
                parents.push(next);
                next += 1;
            }
            frontier = parents;
        }
        let n_sites = 8usize;
        let leaf_states: Vec<i64> = (0..n_leaves * n_sites).map(|i| (i % 4) as i64).collect();
        let mut leaf_row = vec![-1i64; n_nodes];
        for (leaf, entry) in leaf_row.iter_mut().enumerate().take(n_leaves) {
            *entry = leaf as i64;
        }
        let branch_length = vec![0.1f64; n_nodes];

        let before = burn_autodiff::NodeID::new().value;
        pruning_gradient_impl(
            &branch_length,
            &children,
            LeafObservations {
                states: &leaf_states,
                n_sites,
                row: &leaf_row,
            },
            4,
            &[0.25; 4],
            None,
            true,
        )
        .unwrap();
        let after = burn_autodiff::NodeID::new().value;
        (after - before - 1) as usize
    }

    /// The prediction issue #449 states in advance: a `burn` tape is built the
    /// way PyTorch's is, one node per operation per tree node, so its size
    /// tracks the tree rather than collapsing. Asserted as growth rather than
    /// as a constant, because the per-node op count is an implementation
    /// detail of this file and the claim is about the shape.
    #[test]
    fn test_tape_size_tracks_the_tree() {
        let _taped = TAPE.lock().unwrap_or_else(|error| error.into_inner());
        let four = tape_nodes(4);
        let eight = tape_nodes(8);
        let sixteen = tape_nodes(16);
        println!("burn tape nodes: 4 taxa {four}, 8 taxa {eight}, 16 taxa {sixteen}");
        assert!(eight > four, "8 taxa {eight} against 4 taxa {four}");
        assert!(sixteen > eight, "16 taxa {sixteen} against 8 taxa {eight}");
        // Growth is linear in the node count, not sub-linear: doubling the
        // taxa roughly doubles the tape.
        let ratio = (sixteen - eight) as f64 / (eight - four) as f64;
        assert!(
            (1.8..2.2).contains(&ratio),
            "tape growth ratio {ratio}, expected near 2"
        );
    }

    #[test]
    fn test_rejects_a_state_outside_the_alphabet() {
        let _taped = TAPE.lock().unwrap_or_else(|error| error.into_inner());
        let (branch_length, children, mut leaf_states, leaf_row, n_sites) = fixture();
        leaf_states[0] = 9;
        let error = pruning_gradient_impl(
            &branch_length,
            &children,
            LeafObservations {
                states: &leaf_states,
                n_sites,
                row: &leaf_row,
            },
            4,
            &[0.25; 4],
            None,
            true,
        )
        .unwrap_err();
        assert!(error.contains("expected [0, 4)"), "{error}");
    }
}
