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
//! **It reproduces the oracle state for state, and a guard is what buys
//! that.** The Python sweep calls `np.exp` and `np.searchsorted`. Rust's
//! `f64::exp` agrees with NumPy's SIMD implementation to within a unit in
//! the last place, not bit-exactly, and `searchsorted` is a threshold: one
//! draw landing across a boundary that moved by 1 ulp picks a different
//! state, and from that step the two chains are unrelated rather than
//! approximately equal. So the kernel decides a site only where the draw
//! clears every cumulative boundary by more than the two exponentials can
//! move it, and returns the position of the first site it declines so the
//! NumPy path decides that one and the caller resumes. That is a bound, not
//! an assumption about rounding, and it is the construction
//! `search/kernels.py::gibbs_sweep_sites` established for the Gibbs sweep
//! (issues #561, #599). The oracle stays, per root `CLAUDE.md`, and pins
//! this one.
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

use numpy::{PyReadonlyArray1, PyReadonlyArrayDyn, PyReadwriteArray1, PyUntypedArrayMethods};
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
/// - `field`: external field `h`, one row per site, flattened row-major to
///   `n_nodes * n_states`. A field shared by every site reaches here widened,
///   because `sim.potts.site_field` widens at the entry point and the oracle
///   has one code path rather than two (issue #551).
/// - `n_states`: the alphabet size, passed rather than derived, since `field`
///   is now flat and its length alone cannot give it.
/// - `beta`: inverse temperature, applied to the accumulated local field
///   *after* the couplings are summed into it. That is where the Python sweep
///   applies it, and the order is load-bearing: `(h + sum J) * beta` and
///   `beta * h + sum (beta * J)` agree in real arithmetic and not in floating
///   point, so pre-scaling the arguments costs bitwise agreement at every
///   temperature but 1.0 (issue #571).
/// - `offsets`: length `n_nodes + 1`; node `i`'s neighbours are
///   `neighbours[offsets[i]..offsets[i + 1]]`, with `couplings` in step.
/// - `neighbours`, `couplings`: the flattened adjacency, each of length
///   `offsets[n_nodes]`. Every edge appears twice, once from each end.
/// - `draws`: `n_sweeps * n_nodes` uniforms in `[0, 1)`, drawn by the caller.
/// - `guard`: how far from a cumulative boundary a draw must land for this
///   kernel to decide the site itself, in units of the last place per state.
///   NumPy's `exp` and `libm`'s differ by at most one such unit, the
///   cumulative sum carries that across at most `n_states` additions, and
///   scaling by the last entry carries it once more: four units per state
///   bounds it. The caller passes the width it derives, and a test raises it
///   until every site is handed back.
/// - `first`: where to start, as a flat `sweep * n_nodes + node` position, so
///   a run the caller resumed after deciding one site itself picks up at the
///   next.
///
/// # Returns
/// `n_sweeps * n_nodes` if every remaining site was decided, otherwise the
/// flat position of the first site it declined to decide -- which it leaves
/// unwritten, for the caller's NumPy path to decide.
///
/// # Errors
/// Returns `Err` describing the first violated precondition: a ragged
/// adjacency, a draw count that does not match the sweeps requested, a state
/// or neighbour index outside its array, an empty alphabet, a negative guard,
/// or a start beyond the run.
///
/// `pub` so `cargo test` can exercise it without linking Python, per
/// `src/pruning.rs`'s module docs.
#[allow(clippy::too_many_arguments)]
pub fn single_site_sweeps_impl(
    state: &mut [i64],
    field: &[f64],
    n_states: usize,
    offsets: &[usize],
    neighbours: &[i64],
    couplings: &[f64],
    draws: &[f64],
    n_sweeps: usize,
    beta: f64,
    guard: f64,
    first: usize,
) -> Result<usize, String> {
    let n_nodes = state.len();
    if n_states == 0 {
        return Err("field is empty, so there are no states to draw from".to_string());
    }
    if field.len() != n_nodes * n_states {
        return Err(format!(
            "field has {} entries, expected {n_nodes} * {n_states} = {} \
             (one row per site)",
            field.len(),
            n_nodes * n_states
        ));
    }
    if !beta.is_finite() {
        return Err(format!("beta must be finite, got {beta}"));
    }
    if guard < 0.0 || guard.is_nan() {
        return Err(format!("guard must be >= 0, got {guard}"));
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
    if first > n_sweeps * n_nodes {
        return Err(format!(
            "first is {first}, past the {} sites of this run",
            n_sweeps * n_nodes
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

    // `draws` is `n_sweeps * n_nodes` long and validated above, so enumerating
    // it is the flat sweep-and-node position this returns.
    for (position, &draw) in draws.iter().enumerate().skip(first) {
        let node = position % n_nodes;
        local.copy_from_slice(&field[node * n_states..(node + 1) * n_states]);
        for entry in offsets[node]..offsets[node + 1] {
            let neighbour = neighbours[entry];
            if neighbour < 0 || neighbour as usize >= n_nodes {
                return Err(format!(
                    "adjacency entry {entry} names node {neighbour}, \
                     expected [0, {n_nodes})"
                ));
            }
            local[state[neighbour as usize] as usize] += couplings[entry];
        }

        // `beta` here and not on the arguments, because the oracle
        // scales the accumulated local field rather than its parts.
        for value in local.iter_mut() {
            *value *= beta;
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

        let target = draw * total;
        // 2**-52, the largest relative gap between neighbouring float64s.
        let slack = guard * n_states as f64 * 2.220446049250313e-16 * total;
        // `searchsorted`'s left side: the first index whose cumulative weight
        // is at least the target. A boundary this draw sits within `slack` of
        // is one the two exponentials could have moved across it, so the site
        // goes back to the caller undecided rather than being guessed at.
        let mut chosen: Option<usize> = None;
        for (index, &cumulative) in local.iter().enumerate() {
            let gap = cumulative - target;
            if !(gap > slack || -gap > slack) {
                return Ok(position);
            }
            if chosen.is_none() && gap >= 0.0 {
                chosen = Some(index);
            }
        }
        // No boundary reached the draw: `total` rounded below `draw * total`,
        // which `searchsorted` answers by running off the end. The oracle's
        // clamp is its own, so this one goes back too.
        let Some(chosen) = chosen else {
            return Ok(position);
        };
        state[node] = chosen as i64;
    }
    Ok(n_sweeps * n_nodes)
}

/// PyO3 boundary for [`single_site_sweeps_impl`], `Err` mapped to a Python
/// `ValueError`. See the free function's docs for the algorithm, the shapes,
/// and the flat position it returns.
///
/// `guard` and `first` carry no defaults, and that is root `CLAUDE.md`'s rule
/// against a silent behaviour change: a caller left on the old signature would
/// run a partial sweep, ignore the position it came back at, and not be told.
///
/// Arrays are borrowed rather than copied, per issue #232: `as_slice` succeeds
/// only for a C-contiguous array, so a borrow with the wrong stride is
/// impossible rather than merely unlikely, and the wrapper normalizes with
/// `ascontiguousarray` before calling.
#[pyfunction]
#[pyo3(signature = (state, field, offsets, neighbours, couplings, draws, n_sweeps, beta, guard, first))]
#[allow(clippy::too_many_arguments)]
pub fn single_site_sweeps(
    py: Python<'_>,
    mut state: PyReadwriteArray1<'_, i64>,
    field: PyReadonlyArrayDyn<'_, f64>,
    offsets: PyReadonlyArray1<'_, i64>,
    neighbours: PyReadonlyArray1<'_, i64>,
    couplings: PyReadonlyArray1<'_, f64>,
    draws: PyReadonlyArray1<'_, f64>,
    n_sweeps: usize,
    beta: f64,
    guard: f64,
    first: usize,
) -> PyResult<usize> {
    // PyO3 reports a dimensionality mismatch as "'ndarray' object is not an
    // instance of 'ndarray'", which names neither shape. Read the dimensions
    // here so the caller is told what was wanted and what arrived (#571).
    // Taken as a dynamic array rather than a 2-D one so a wrong dimensionality
    // is reported here. PyO3 rejects it before the body otherwise, as
    // "'ndarray' object is not an instance of 'ndarray'", which names neither
    // the shape wanted nor the shape given (#571).
    let [n_rows, n_states] = *field.shape() else {
        return Err(PyValueError::new_err(format!(
            "field must be 2-D, (n_nodes, n_states), one row per site; got \
             shape {:?}",
            field.shape()
        )));
    };
    if n_rows != state.len() {
        return Err(PyValueError::new_err(format!(
            "field has {n_rows} rows and state has {} sites; the field \
             carries one row per site",
            state.len()
        )));
    }
    let offsets: Vec<usize> = offsets
        .as_slice()?
        .iter()
        .map(|&value| {
            usize::try_from(value).map_err(|_| PyValueError::new_err("offsets must be >= 0"))
        })
        .collect::<PyResult<_>>()?;
    let state = state.as_slice_mut()?;
    let field = field.as_slice()?;
    let neighbours = neighbours.as_slice()?;
    let couplings = couplings.as_slice()?;
    let draws = draws.as_slice()?;
    py.detach(|| {
        single_site_sweeps_impl(
            state, field, n_states, &offsets, neighbours, couplings, draws, n_sweeps, beta, guard,
            first,
        )
    })
    .map_err(PyValueError::new_err)
}

/// The largest relative gap between neighbouring `f64`s, `2**-52`. The width
/// the caller's guard is counted in, as in [`single_site_sweeps_impl`].
const ULP: f64 = 2.220446049250313e-16;

/// Union-find root, with path compression. The oracle's `find_root`.
#[inline]
fn find(parent: &mut [usize], node: usize) -> usize {
    let mut root = node;
    while parent[root] != root {
        root = parent[root];
    }
    let mut walk = node;
    while parent[walk] != root {
        let next = parent[walk];
        parent[walk] = root;
        walk = next;
    }
    root
}

/// Merge two components, keeping the first edge end's root. The oracle's
/// `union_roots`, whose rule decides *which* node labels the component and so
/// which root the recolouring order is taken in.
#[inline]
fn union(parent: &mut [usize], first: usize, second: usize) {
    let first_root = find(parent, first);
    let second_root = find(parent, second);
    if first_root != second_root {
        parent[second_root] = first_root;
    }
}

/// Every node's union-find root once the bonds `first[b] -- second[b]` are
/// merged in order, by the rule `find_root` and `union_roots` apply, so the
/// roots are the Python loop's bitwise (issue #986).
pub fn bond_roots_impl(n_nodes: usize, first: &[i64], second: &[i64]) -> Result<Vec<i64>, String> {
    if first.len() != second.len() {
        return Err(format!(
            "first has {} bond ends and second {}",
            first.len(),
            second.len()
        ));
    }
    let mut parent: Vec<usize> = (0..n_nodes).collect();
    for (&a, &b) in first.iter().zip(second) {
        if a < 0 || b < 0 || a as usize >= n_nodes || b as usize >= n_nodes {
            return Err(format!(
                "bond ({a}, {b}) names a node outside [0, {n_nodes})"
            ));
        }
        union(&mut parent, a as usize, b as usize);
    }
    Ok((0..n_nodes)
        .map(|node| find(&mut parent, node) as i64)
        .collect())
}

/// The union-find roots of a bond set; see [`bond_roots_impl`].
#[pyfunction]
#[pyo3(signature = (n_nodes, first, second))]
pub fn bond_roots<'py>(
    py: Python<'py>,
    n_nodes: usize,
    first: PyReadonlyArray1<'py, i64>,
    second: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, numpy::PyArray1<i64>>> {
    let (first, second) = (first.as_slice()?, second.as_slice()?);
    let roots = py
        .detach(|| bond_roots_impl(n_nodes, first, second))
        .map_err(PyValueError::new_err)?;
    Ok(numpy::PyArray1::from_vec(py, roots))
}

/// One Swendsen-Wang bond-and-recolour pass over `state`, in place.
///
/// Ported from `search.potts_mcmc.swendsen_wang_sweep` and its `_recolour`,
/// which stay as the oracle. The pass activates a bond on each like-coloured
/// edge whose draw clears the bond probability, joins the active bonds into
/// clusters, and offers each cluster one colour, accepted on the field
/// difference alone --- the Fortuin-Kasteleyn construction contributes
/// nothing to the ratio, which `potts_mcmc`'s module docstring derives.
///
/// # Parameters
/// - `state`: the configuration, length `n_nodes`, updated in place.
/// - `field`: the field **already scaled by `beta`**, one row per site,
///   flattened row-major to `n_nodes * n_states`. Scaled by the caller rather
///   than here so every term this sums is the oracle's own `beta * rows`
///   entry, bit for bit, and `beta` reaches no arithmetic in this file.
/// - `n_states`: the alphabet size, which flat `field` cannot give alone.
/// - `edges`: the edge list flattened to `2 * n_edges`, each edge's two ends
///   adjacent, in the graph's edge order. That order is the contract
///   `PottsGraph.edge_index` states: the bond draws index it.
/// - `bond_probability`: `1 - exp(-beta J)` per edge, computed by the caller
///   for the same reason the field is scaled there --- `exp` is a threshold
///   this kernel would otherwise have to guard, and NumPy has already
///   evaluated it for the oracle.
/// - `bond_draws`: one uniform per edge, `[0, 1)`.
/// - `colour_draws`: one proposed colour per *cluster*, indexed by the
///   cluster's rank in increasing root order. Length `n_nodes`, the upper
///   bound on the cluster count, since the caller cannot know the count
///   before the bond pass and crossing twice to learn it is the boundary
///   rule root `CLAUDE.md` states.
/// - `accept_draws`: one uniform per cluster, indexed the same way, read only
///   where the field difference is negative.
/// - `labels`: `n_nodes` out: the component root per node, as the oracle's
///   `find_root` reports it. Read rather than written when `first > 0`, so a
///   resumed pass recolours the clusters it already built instead of
///   rebuilding them from a state it has half-changed.
/// - `guard`: how far from a decision a quantity must sit for this kernel to
///   decide it, in units of the last place. Two decisions are thresholds:
///   the sign of the field difference and the uniform against `exp` of it.
///   This kernel sums a cluster's field terms left to right and NumPy sums
///   them pairwise, and two orders of `m` terms differ by at most
///   `2 m * ULP * sum |term|`; `guard = 16` is eight times that bound, and
///   the same width, on the same derivation, as the sweep above.
/// - `first`: the cluster rank to resume at. `0` builds the clusters; above
///   it the caller has decided the clusters below it and `labels` is an
///   input.
///
/// # Returns
/// `(n_clusters, stop)`: how many clusters the pass built, and either
/// `n_clusters` --- every one decided --- or the rank of the first cluster it
/// declined to decide, which it leaves as it found it for the caller's NumPy
/// path to decide.
///
/// # Errors
/// Returns `Err` describing the first violated precondition: a draw or label
/// array whose length does not match, a node or colour index outside its
/// array, a guard below zero, a resume past the clusters, or a `labels` on a
/// resume that is not a labelling by roots.
#[allow(clippy::too_many_arguments)]
pub fn swendsen_wang_sweep_impl(
    state: &mut [i64],
    field: &[f64],
    n_states: usize,
    edges: &[i64],
    bond_probability: &[f64],
    bond_draws: &[f64],
    colour_draws: &[i64],
    accept_draws: &[f64],
    labels: &mut [i64],
    guard: f64,
    first: usize,
) -> Result<(usize, usize), String> {
    let n_nodes = state.len();
    if n_states == 0 {
        return Err("field is empty, so there are no colours to propose".to_string());
    }
    if field.len() != n_nodes * n_states {
        return Err(format!(
            "field has {} entries, expected {n_nodes} * {n_states} = {} (one row per site)",
            field.len(),
            n_nodes * n_states
        ));
    }
    if !edges.len().is_multiple_of(2) {
        return Err(format!(
            "edges has {} entries, which is not two per edge",
            edges.len()
        ));
    }
    let n_edges = edges.len() / 2;
    if bond_probability.len() != n_edges || bond_draws.len() != n_edges {
        return Err(format!(
            "bond_probability has {} entries and bond_draws {}, expected {n_edges} each",
            bond_probability.len(),
            bond_draws.len()
        ));
    }
    if colour_draws.len() != n_nodes || accept_draws.len() != n_nodes {
        return Err(format!(
            "colour_draws has {} entries and accept_draws {}, expected {n_nodes} each: \
             one per cluster, and a pass builds at most one cluster per site",
            colour_draws.len(),
            accept_draws.len()
        ));
    }
    if labels.len() != n_nodes {
        return Err(format!(
            "labels has {} entries, expected {n_nodes} (one root per site)",
            labels.len()
        ));
    }
    if guard < 0.0 || guard.is_nan() {
        return Err(format!("guard must be >= 0, got {guard}"));
    }
    for (node, &value) in state.iter().enumerate() {
        if value < 0 || value as usize >= n_states {
            return Err(format!(
                "state at node {node} is {value}, expected [0, {n_states})"
            ));
        }
    }

    if first == 0 {
        let mut parent: Vec<usize> = (0..n_nodes).collect();
        for edge in 0..n_edges {
            let left = edges[2 * edge];
            let right = edges[2 * edge + 1];
            if left < 0 || right < 0 || left as usize >= n_nodes || right as usize >= n_nodes {
                return Err(format!(
                    "edge {edge} joins {left} and {right}, expected [0, {n_nodes})"
                ));
            }
            let (left, right) = (left as usize, right as usize);
            // The oracle's two conditions in its order: like colours first,
            // then the draw against the probability the caller evaluated.
            if state[left] == state[right] && bond_draws[edge] < bond_probability[edge] {
                union(&mut parent, left, right);
            }
        }
        for (node, label) in labels.iter_mut().enumerate() {
            *label = find(&mut parent, node) as i64;
        }
    } else {
        for (node, &label) in labels.iter().enumerate() {
            if label < 0 || label as usize >= n_nodes {
                return Err(format!(
                    "labels at node {node} is {label}, expected [0, {n_nodes})"
                ));
            }
            if labels[label as usize] != label {
                return Err(format!(
                    "labels at node {node} names {label}, which is not its own root: \
                     a resumed pass reads the clusters the first call wrote"
                ));
            }
        }
    }

    // The clusters in increasing root order, as one pass over the nodes: a
    // root labels itself, so scanning upward meets the roots sorted, which is
    // the order `np.unique` puts the oracle's recolourings in. Members land
    // in increasing node order inside each cluster, which is
    // `np.flatnonzero`'s.
    let mut rank = vec![usize::MAX; n_nodes];
    let mut n_clusters = 0usize;
    for (node, &label) in labels.iter().enumerate() {
        if label as usize == node {
            rank[node] = n_clusters;
            n_clusters += 1;
        }
    }
    let mut bounds = vec![0usize; n_clusters + 1];
    for &label in labels.iter() {
        bounds[rank[label as usize] + 1] += 1;
    }
    for index in 0..n_clusters {
        bounds[index + 1] += bounds[index];
    }
    let mut cursor = bounds.clone();
    let mut members = vec![0usize; n_nodes];
    for (node, &label) in labels.iter().enumerate() {
        let cluster = rank[label as usize];
        members[cursor[cluster]] = node;
        cursor[cluster] += 1;
    }

    if first > n_clusters {
        return Err(format!(
            "first is {first}, past the {n_clusters} clusters this pass built"
        ));
    }

    for cluster in first..n_clusters {
        let own = &members[bounds[cluster]..bounds[cluster + 1]];
        let current = state[own[0]] as usize;
        let proposed = colour_draws[cluster];
        if proposed < 0 || proposed as usize >= n_states {
            return Err(format!(
                "colour_draws at cluster {cluster} is {proposed}, expected [0, {n_states})"
            ));
        }
        let proposed = proposed as usize;
        if proposed == current {
            continue;
        }

        // The field term of the acceptance, summed over the cluster's own
        // members: the bond construction cancels and the field does not.
        // `magnitude` is what the guard below is counted against, accumulated
        // here rather than in a second walk of the same rows.
        let mut offered = 0.0f64;
        let mut held = 0.0f64;
        let mut magnitude = 0.0f64;
        for &node in own {
            let to = field[node * n_states + proposed];
            let from = field[node * n_states + current];
            offered += to;
            held += from;
            magnitude += to.abs() + from.abs();
        }
        let difference = offered - held;
        let slack = guard * own.len() as f64 * ULP * magnitude;
        // `slack` is zero only where every term is, and then both orders sum
        // to exactly zero and the oracle's `>= 0` decides it here too: a
        // guard that handed back on a zero field would hand back every
        // cluster of the instance the benchmarks run.
        if slack > 0.0 && difference.abs() <= slack {
            return Ok((n_clusters, cluster));
        }
        let accepted = if difference >= 0.0 {
            true
        } else {
            // `exp` is the second threshold: NumPy's and `libm`'s differ in
            // the last place, and the argument carries `slack` of its own,
            // which `exp` scales by its own value.
            let threshold = difference.exp();
            let draw = accept_draws[cluster];
            if (draw - threshold).abs() <= threshold * (slack + guard * ULP) {
                return Ok((n_clusters, cluster));
            }
            draw < threshold
        };
        if accepted {
            for &node in own {
                state[node] = proposed as i64;
            }
        }
    }
    Ok((n_clusters, n_clusters))
}

/// PyO3 boundary for [`swendsen_wang_sweep_impl`], `Err` mapped to a Python
/// `ValueError`. See the free function's docs for the pass, the shapes and
/// the pair it returns.
///
/// Arrays are borrowed rather than copied: `as_slice` succeeds only for a
/// C-contiguous array, and the adapter normalizes with `ascontiguousarray`
/// before calling, so a strided caller is refused here rather than read
/// wrongly.
#[pyfunction]
#[pyo3(signature = (state, field, edges, bond_probability, bond_draws, colour_draws, accept_draws, labels, guard, first))]
#[allow(clippy::too_many_arguments)]
pub fn swendsen_wang_sweep(
    py: Python<'_>,
    mut state: PyReadwriteArray1<'_, i64>,
    field: PyReadonlyArrayDyn<'_, f64>,
    edges: PyReadonlyArray1<'_, i64>,
    bond_probability: PyReadonlyArray1<'_, f64>,
    bond_draws: PyReadonlyArray1<'_, f64>,
    colour_draws: PyReadonlyArray1<'_, i64>,
    accept_draws: PyReadonlyArray1<'_, f64>,
    mut labels: PyReadwriteArray1<'_, i64>,
    guard: f64,
    first: usize,
) -> PyResult<(usize, usize)> {
    // Read the dimensions here, for the reason `single_site_sweeps` states:
    // PyO3 reports a wrong dimensionality as "'ndarray' object is not an
    // instance of 'ndarray'", which names neither shape (#571).
    let [n_rows, n_states] = *field.shape() else {
        return Err(PyValueError::new_err(format!(
            "field must be 2-D, (n_nodes, n_states), one row per site; got shape {:?}",
            field.shape()
        )));
    };
    if n_rows != state.len() {
        return Err(PyValueError::new_err(format!(
            "field has {n_rows} rows and state has {} sites; the field carries one row per site",
            state.len()
        )));
    }
    let state = state.as_slice_mut()?;
    let labels = labels.as_slice_mut()?;
    let field = field.as_slice()?;
    let edges = edges.as_slice()?;
    let bond_probability = bond_probability.as_slice()?;
    let bond_draws = bond_draws.as_slice()?;
    let colour_draws = colour_draws.as_slice()?;
    let accept_draws = accept_draws.as_slice()?;
    py.detach(|| {
        swendsen_wang_sweep_impl(
            state,
            field,
            n_states,
            edges,
            bond_probability,
            bond_draws,
            colour_draws,
            accept_draws,
            labels,
            guard,
            first,
        )
    })
    .map_err(PyValueError::new_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `search.potts_mcmc._GUARD`, the width the callers pass.
    const GUARD: f64 = 16.0;

    /// Two isolated sites, no coupling, a field favouring state 1 by `ln 3`.
    /// The conditional is then `(1/4, 3/4)` exactly, so a draw below 0.25
    /// takes state 0 and one above takes state 1 -- hand-computable, and
    /// independent of this implementation.
    #[test]
    fn a_field_only_conditional_matches_the_hand_computed_split() {
        // Two sites, so the field is two rows: the kernel indexes per site.
        let field = vec![0.0, 3.0f64.ln(), 0.0, 3.0f64.ln()];
        let offsets = vec![0usize, 0, 0];
        let mut state = vec![0i64, 0];

        let mut low = state.clone();
        let stop = single_site_sweeps_impl(
            &mut low,
            &field,
            2,
            &offsets,
            &[],
            &[],
            &[0.1, 0.1],
            1,
            1.0,
            GUARD,
            0,
        )
        .unwrap();
        assert_eq!((stop, low), (2, vec![0, 0]));

        let stop = single_site_sweeps_impl(
            &mut state,
            &field,
            2,
            &offsets,
            &[],
            &[],
            &[0.9, 0.9],
            1,
            1.0,
            GUARD,
            0,
        )
        .unwrap();
        assert_eq!((stop, state), (2, vec![1, 1]));
    }

    /// A coupling large enough to dominate a zero field drives a neighbour to
    /// agree whatever the draw: the conditional puts all but `exp(-20)` of its
    /// mass on the neighbour's state.
    #[test]
    fn a_dominant_coupling_makes_a_neighbour_agree() {
        let field = vec![0.0, 0.0, 0.0, 0.0];
        let offsets = vec![0usize, 1, 2];
        let neighbours = vec![1i64, 0];
        let couplings = vec![20.0, 20.0];
        let mut state = vec![0i64, 1];

        single_site_sweeps_impl(
            &mut state,
            &field,
            2,
            &offsets,
            &neighbours,
            &couplings,
            &[0.5, 0.5],
            1,
            1.0,
            GUARD,
            0,
        )
        .unwrap();

        assert_eq!(state[0], state[1], "a dominant coupling did not align them");
    }

    #[test]
    fn a_ragged_adjacency_is_refused() {
        let error = single_site_sweeps_impl(
            &mut [0i64],
            &[0.0, 0.0],
            2,
            &[0usize, 1],
            &[0i64],
            &[],
            &[0.5],
            1,
            1.0,
            GUARD,
            0,
        )
        .unwrap_err();
        assert!(error.contains("index in step"), "{error}");
    }

    #[test]
    fn a_draw_count_that_does_not_match_the_sweeps_is_refused() {
        let error = single_site_sweeps_impl(
            &mut [0i64],
            &[0.0, 0.0],
            2,
            &[0usize, 0],
            &[],
            &[],
            &[0.5],
            2,
            1.0,
            GUARD,
            0,
        )
        .unwrap_err();
        assert!(error.contains("expected 2 * 1"), "{error}");
    }

    #[test]
    fn a_state_outside_the_alphabet_is_refused() {
        let error = single_site_sweeps_impl(
            &mut [5i64],
            &[0.0, 0.0],
            2,
            &[0usize, 0],
            &[],
            &[],
            &[0.5],
            1,
            1.0,
            GUARD,
            0,
        )
        .unwrap_err();
        assert!(error.contains("expected [0, 2)"), "{error}");
    }

    /// A guard wide enough to cover the whole cumulative sum can decide no
    /// site, so the call returns where it started and writes nothing. This
    /// pins the hand-back path itself, which no realistic draw reaches
    /// (issue #599).
    #[test]
    fn a_guard_wider_than_the_distribution_decides_nothing() {
        let mut state = vec![0i64];
        let stop = single_site_sweeps_impl(
            &mut state,
            &[0.0, 3.0f64.ln()],
            2,
            &[0usize, 0],
            &[],
            &[],
            &[0.9],
            1,
            1.0,
            1e18,
            0,
        )
        .unwrap();
        assert_eq!((stop, state), (0, vec![0]));
    }

    /// `first` resumes: a run started past its only site decides nothing and
    /// reports the run complete.
    #[test]
    fn a_run_resumed_past_its_last_site_decides_nothing() {
        let mut state = vec![0i64];
        let stop = single_site_sweeps_impl(
            &mut state,
            &[0.0, 3.0f64.ln()],
            2,
            &[0usize, 0],
            &[],
            &[],
            &[0.9],
            1,
            1.0,
            GUARD,
            1,
        )
        .unwrap();
        assert_eq!((stop, state), (1, vec![0]));
    }

    #[test]
    fn a_negative_guard_is_refused() {
        let error = single_site_sweeps_impl(
            &mut [0i64],
            &[0.0, 0.0],
            2,
            &[0usize, 0],
            &[],
            &[],
            &[0.5],
            1,
            1.0,
            -1.0,
            0,
        )
        .unwrap_err();
        assert!(error.contains("guard must be >= 0"), "{error}");
    }

    /// Two sites, one edge, every bond active: they are one cluster, and a
    /// field favouring the proposed colour recolours both or neither. The
    /// partition is hand-computable and independent of this implementation.
    #[test]
    fn an_active_bond_makes_one_cluster_that_moves_together() {
        let mut state = vec![0i64, 0];
        let mut labels = vec![0i64, 0];
        // Colour 1 is worth 1.0 a site and colour 0 nothing, so the offered
        // difference is +2 and the accept step takes it without a draw.
        let field = vec![0.0, 1.0, 0.0, 1.0];
        let (n_clusters, stop) = swendsen_wang_sweep_impl(
            &mut state,
            &field,
            2,
            &[0i64, 1],
            &[1.0],
            &[0.5],
            &[1i64, 1],
            &[0.5, 0.5],
            &mut labels,
            GUARD,
            0,
        )
        .unwrap();

        assert_eq!((n_clusters, stop), (1, 1));
        assert_eq!(state, vec![1, 1]);
        assert_eq!(labels, vec![0, 0]);
    }

    /// The same two sites with the bond refused: two clusters, recoloured
    /// independently, and the second cluster reads the *second* colour draw.
    #[test]
    fn a_refused_bond_leaves_two_clusters_each_with_its_own_draw() {
        let mut state = vec![0i64, 0];
        let mut labels = vec![0i64, 0];
        let field = vec![0.0, 1.0, 0.0, 1.0];
        let (n_clusters, stop) = swendsen_wang_sweep_impl(
            &mut state,
            &field,
            2,
            &[0i64, 1],
            // The draw does not clear the probability, so no bond is active.
            &[0.25],
            &[0.5],
            &[1i64, 0],
            &[0.5, 0.5],
            &mut labels,
            GUARD,
            0,
        )
        .unwrap();

        assert_eq!((n_clusters, stop), (2, 2));
        assert_eq!(state, vec![1, 0], "each cluster took its own colour");
        assert_eq!(labels, vec![0, 1]);
    }

    /// A cluster whose proposal loses field is kept only when the uniform
    /// clears `exp(difference)`. Two sites at `exp(-2) = 0.135`: a draw of
    /// 0.1 takes it and one of 0.9 does not, which is the Metropolis step
    /// computed by hand.
    #[test]
    fn a_losing_proposal_is_accepted_only_below_the_exponential() {
        let field = vec![0.0, -1.0, 0.0, -1.0];
        for (draw, expected) in [(0.1, vec![1i64, 1]), (0.9, vec![0i64, 0])] {
            let mut state = vec![0i64, 0];
            let mut labels = vec![0i64, 0];
            let (_, stop) = swendsen_wang_sweep_impl(
                &mut state,
                &field,
                2,
                &[0i64, 1],
                &[1.0],
                &[0.5],
                &[1i64, 1],
                &[draw, draw],
                &mut labels,
                GUARD,
                0,
            )
            .unwrap();

            assert_eq!(stop, 1);
            assert_eq!(state, expected, "draw {draw} decided the wrong way");
        }
    }

    /// A draw sitting on `exp(difference)` is not decided here: the kernel
    /// returns the cluster's rank and leaves it as it found it, which is the
    /// hand-back the sweep above makes per site.
    #[test]
    fn a_draw_on_the_boundary_is_handed_back_undecided() {
        let field = vec![0.0, -1.0, 0.0, -1.0];
        let mut state = vec![0i64, 0];
        let mut labels = vec![0i64, 0];
        let (n_clusters, stop) = swendsen_wang_sweep_impl(
            &mut state,
            &field,
            2,
            &[0i64, 1],
            &[1.0],
            &[0.5],
            &[1i64, 1],
            &[(-2.0f64).exp(), 0.0],
            &mut labels,
            GUARD,
            0,
        )
        .unwrap();

        assert_eq!((n_clusters, stop), (1, 0));
        assert_eq!(state, vec![0, 0], "the handed-back cluster was written");
    }

    /// Resuming reads the labels the first call wrote rather than rebuilding
    /// them: the state it resumes on has already moved, so a second bond pass
    /// would partition a different configuration.
    #[test]
    fn a_resumed_pass_recolours_the_clusters_it_was_given() {
        let field = vec![0.0, 1.0, 0.0, 1.0];
        let mut state = vec![0i64, 1];
        // Two singletons, as a first call would have left them.
        let mut labels = vec![0i64, 1];
        let (n_clusters, stop) = swendsen_wang_sweep_impl(
            &mut state,
            &field,
            2,
            &[0i64, 1],
            &[1.0],
            &[0.5],
            &[0i64, 0],
            &[0.5, 0.9],
            &mut labels,
            GUARD,
            1,
        )
        .unwrap();

        assert_eq!((n_clusters, stop), (2, 2));
        // Cluster 0 was the caller's to decide and is untouched; cluster 1
        // was offered colour 0, which loses 1.0 of field, and 0.9 > exp(-1).
        assert_eq!(state, vec![0, 1]);
        assert_eq!(labels, vec![0, 1]);
    }

    #[test]
    fn a_colour_outside_the_alphabet_is_refused() {
        let error = swendsen_wang_sweep_impl(
            &mut [0i64, 0],
            &[0.0, 1.0, 0.0, 1.0],
            2,
            &[0i64, 1],
            &[1.0],
            &[0.5],
            &[2i64, 0],
            &[0.5, 0.5],
            &mut [0i64, 0],
            GUARD,
            0,
        )
        .unwrap_err();
        assert!(error.contains("expected [0, 2)"), "{error}");
    }

    #[test]
    fn a_resume_on_labels_that_are_not_roots_is_refused() {
        let error = swendsen_wang_sweep_impl(
            &mut [0i64, 0],
            &[0.0, 1.0, 0.0, 1.0],
            2,
            &[0i64, 1],
            &[1.0],
            &[0.5],
            &[1i64, 1],
            &[0.5, 0.5],
            &mut [1i64, 0],
            GUARD,
            1,
        )
        .unwrap_err();
        assert!(error.contains("not its own root"), "{error}");
    }

    #[test]
    fn a_bond_draw_count_that_does_not_match_the_edges_is_refused() {
        let error = swendsen_wang_sweep_impl(
            &mut [0i64, 0],
            &[0.0, 1.0, 0.0, 1.0],
            2,
            &[0i64, 1],
            &[1.0],
            &[0.5, 0.5],
            &[1i64, 1],
            &[0.5, 0.5],
            &mut [0i64, 0],
            GUARD,
            0,
        )
        .unwrap_err();
        assert!(error.contains("expected 1 each"), "{error}");
    }
}
