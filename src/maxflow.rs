//! Boykov--Kolmogorov maximum flow, and the Ising ground-state reduction.
//!
//! The Python implementation in `snakes_and_ladders.search.maxflow` --- Dinic,
//! kept readable --- stays as the oracle this is pinned against, per root
//! `CLAUDE.md`. What Rust buys is what root `CLAUDE.md` reserves it for:
//! control flow and irregular memory access over an adjacency structure, with
//! no array arithmetic a vectorized NumPy version could exploit.
//!
//! **Why this kernel.** Issue #715 built four behind one seam --- Dinic,
//! highest-label push-relabel, Boykov--Kolmogorov and a synchronous parallel
//! push-relabel --- pinned each to the Python Dinic arc for arc, and timed
//! them on square lattices with a random per-node field and as the inner
//! solver of alpha expansion. Boykov--Kolmogorov won both tables: 61 ms
//! against Dinic's 234 on the 256x256 cut and 352 ms against 637 on the
//! 64x64 x 10-label expansion. The other three are conserved in
//! `src/maxflow_declined.rs` behind the `sandbox` feature, so the comparison
//! stays re-runnable.
//!
//! The origin check is cached by timestamp and distance as the authors'
//! implementation does, and the source side of the cut is read by one
//! breadth-first search over the residual graph on termination, which is the
//! minimal minimum cut every maximum flow shares.
//!
//! Capacities are `f64` and the termination test is `> 0.0` rather than a
//! tolerance, matching the reference exactly so the two cannot disagree on
//! which arcs are saturated.
//!
//! As in `pruning.rs`, the kernel is a plain function returning `Result` and
//! the `#[pyfunction]` is a thin wrapper, so `cargo test` exercises the
//! algorithm without touching `PyResult`.

use std::collections::VecDeque;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

const NONE: usize = usize::MAX;

/// A flow network as paired residual arcs.
///
/// Arc `2 * e` and `2 * e + 1` are the two directions of one edge, so an
/// arc's reverse is its index with the low bit flipped. Pushing flow
/// subtracts from one and adds to the other, which keeps the residual graph
/// implicit rather than a second structure to hold in step.
pub struct FlowNetwork {
    pub(crate) n_nodes: usize,
    pub(crate) target: Vec<usize>,
    pub(crate) capacity: Vec<f64>,
    pub(crate) outgoing: Vec<Vec<usize>>,
}

impl FlowNetwork {
    /// An empty network on `n_nodes` nodes.
    pub fn new(n_nodes: usize) -> Self {
        Self {
            n_nodes,
            target: Vec::new(),
            capacity: Vec::new(),
            outgoing: vec![Vec::new(); n_nodes],
        }
    }

    /// Add `source -> sink`, with `reverse` capacity on the back arc.
    pub fn add_edge(
        &mut self,
        source: usize,
        sink: usize,
        capacity: f64,
        reverse: f64,
    ) -> Result<(), String> {
        if capacity < 0.0 || reverse < 0.0 {
            return Err(format!(
                "capacities must be non-negative, got {capacity} and {reverse}"
            ));
        }
        if source >= self.n_nodes || sink >= self.n_nodes {
            return Err(format!(
                "edge ({source}, {sink}) names a node outside [0, {})",
                self.n_nodes
            ));
        }
        self.outgoing[source].push(self.target.len());
        self.target.push(sink);
        self.capacity.push(capacity);
        self.outgoing[sink].push(self.target.len());
        self.target.push(source);
        self.capacity.push(reverse);
        Ok(())
    }

    /// Breadth-first distances in the residual graph; `usize::MAX` if unreached.
    pub(crate) fn levels(&self, source: usize) -> Vec<usize> {
        let mut level = vec![usize::MAX; self.n_nodes];
        level[source] = 0;
        let mut queue = VecDeque::new();
        queue.push_back(source);
        while let Some(node) = queue.pop_front() {
            for &arc in &self.outgoing[node] {
                let neighbour = self.target[arc];
                if self.capacity[arc] > 0.0 && level[neighbour] == usize::MAX {
                    level[neighbour] = level[node] + 1;
                    queue.push_back(neighbour);
                }
            }
        }
        level
    }
}

/// Maximum flow, and the source side of the minimum cut it certifies.
///
/// Returns `(value, source_side)`, where `source_side[i]` is whether node `i`
/// is reachable from the source in the residual graph on termination. That
/// set is the *minimal* minimum cut, which every maximum flow shares, so a
/// configuration read off it does not depend on the kernel that produced
/// the flow --- which is what let issue #715 pin four kernels against one
/// another element for element before keeping this one.
pub fn max_flow_impl(
    network: &mut FlowNetwork,
    source: usize,
    sink: usize,
) -> Result<(f64, Vec<bool>), String> {
    if source == sink {
        return Err(format!("source and sink must differ, both are {source}"));
    }
    if source >= network.n_nodes || sink >= network.n_nodes {
        return Err(format!(
            "terminals ({source}, {sink}) must lie in [0, {})",
            network.n_nodes
        ));
    }
    let total = boykov_kolmogorov(network, source, sink);
    let level = network.levels(source);
    Ok((total, level.iter().map(|&d| d != usize::MAX).collect()))
}

// ---------------------------------------------------------------------------
// Boykov--Kolmogorov.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tree {
    Free,
    Source,
    Sink,
}

struct BoykovKolmogorov<'a> {
    network: &'a mut FlowNetwork,
    source: usize,
    sink: usize,
    tree: Vec<Tree>,
    /// The arc that joins a node to its parent, oriented toward the flow:
    /// `parent -> node` in the source tree and `node -> parent` in the sink
    /// tree, so its capacity is the residual the path may carry.
    parent: Vec<usize>,
    active: VecDeque<usize>,
    orphans: Vec<usize>,
    /// Origin cache: a node stamped with the current time is known to reach
    /// its terminal, at `distance` hops.
    stamp: Vec<usize>,
    distance: Vec<usize>,
    time: usize,
}

impl BoykovKolmogorov<'_> {
    fn parent_node(&self, node: usize) -> usize {
        // The other end of the parent arc, whichever way it is oriented.
        let arc = self.parent[node];
        match self.tree[node] {
            Tree::Source => self.network.target[arc ^ 1],
            Tree::Sink => self.network.target[arc],
            Tree::Free => NONE,
        }
    }

    /// Grow both trees until a crossing arc is found; returns it as the arc
    /// from the source-tree node to the sink-tree node.
    fn grow(&mut self) -> Option<usize> {
        while let Some(node) = self.active.front().copied() {
            if self.tree[node] == Tree::Free {
                self.active.pop_front();
                continue;
            }
            let tree = self.tree[node];
            for position in 0..self.network.outgoing[node].len() {
                let arc = self.network.outgoing[node][position];
                // Residual in the tree's direction of travel.
                let travel = if tree == Tree::Source { arc } else { arc ^ 1 };
                if self.network.capacity[travel] <= 0.0 {
                    continue;
                }
                let neighbour = self.network.target[arc];
                match self.tree[neighbour] {
                    Tree::Free => {
                        self.tree[neighbour] = tree;
                        self.parent[neighbour] = travel;
                        self.stamp[neighbour] = self.stamp[node];
                        self.distance[neighbour] = self.distance[node] + 1;
                        self.active.push_back(neighbour);
                    }
                    other if other != tree => {
                        return Some(if tree == Tree::Source { arc } else { arc ^ 1 });
                    }
                    _ => {}
                }
            }
            self.active.pop_front();
        }
        None
    }

    /// Push the bottleneck along the path through `crossing`; saturated
    /// tree arcs orphan their child.
    fn augment(&mut self, crossing: usize) -> f64 {
        let mut bottleneck = self.network.capacity[crossing];
        let mut node = self.network.target[crossing ^ 1];
        while node != self.source {
            bottleneck = bottleneck.min(self.network.capacity[self.parent[node]]);
            node = self.parent_node(node);
        }
        node = self.network.target[crossing];
        while node != self.sink {
            bottleneck = bottleneck.min(self.network.capacity[self.parent[node]]);
            node = self.parent_node(node);
        }

        let push = |network: &mut FlowNetwork, arc: usize| {
            network.capacity[arc] -= bottleneck;
            network.capacity[arc ^ 1] += bottleneck;
            network.capacity[arc] <= 0.0
        };
        push(self.network, crossing);
        node = self.network.target[crossing ^ 1];
        while node != self.source {
            let arc = self.parent[node];
            let next = self.parent_node(node);
            if push(self.network, arc) {
                self.parent[node] = NONE;
                self.orphans.push(node);
            }
            node = next;
        }
        node = self.network.target[crossing];
        while node != self.sink {
            let arc = self.parent[node];
            let next = self.parent_node(node);
            if push(self.network, arc) {
                self.parent[node] = NONE;
                self.orphans.push(node);
            }
            node = next;
        }
        bottleneck
    }

    /// Whether `node` reaches its terminal through parents, caching the
    /// answer along the way; `Some(distance)` when it does.
    fn origin(&mut self, node: usize) -> Option<usize> {
        let terminal = if self.tree[node] == Tree::Source {
            self.source
        } else {
            self.sink
        };
        let mut walk = node;
        let mut hops = 0usize;
        let found = loop {
            if walk == terminal {
                break Some(0usize);
            }
            if self.stamp[walk] == self.time {
                break Some(self.distance[walk]);
            }
            if self.parent[walk] == NONE {
                break None;
            }
            walk = self.parent_node(walk);
            hops += 1;
        };
        let base = found?;
        // Stamp the path so the next check from any of its nodes is O(1).
        let mut walk = node;
        let mut remaining = hops;
        while remaining > 0 {
            self.stamp[walk] = self.time;
            self.distance[walk] = base + remaining;
            walk = self.parent_node(walk);
            remaining -= 1;
        }
        Some(base + hops)
    }

    fn adopt(&mut self) {
        while let Some(orphan) = self.orphans.pop() {
            let tree = self.tree[orphan];
            let mut best_arc = NONE;
            let mut best_distance = NONE;
            for position in 0..self.network.outgoing[orphan].len() {
                let arc = self.network.outgoing[orphan][position];
                let neighbour = self.network.target[arc];
                if self.tree[neighbour] != tree {
                    continue;
                }
                // A parent must carry residual toward the orphan in the
                // source tree, and away from it in the sink tree.
                let travel = if tree == Tree::Source { arc ^ 1 } else { arc };
                if self.network.capacity[travel] <= 0.0 {
                    continue;
                }
                if let Some(distance) = self.origin(neighbour) {
                    if distance + 1 < best_distance {
                        best_distance = distance + 1;
                        best_arc = travel;
                    }
                }
            }
            if best_arc != NONE {
                self.parent[orphan] = best_arc;
                self.stamp[orphan] = self.time;
                self.distance[orphan] = best_distance;
                continue;
            }
            // No parent: the orphan leaves the tree, its children become
            // orphans, and every same-tree neighbour with residual toward it
            // is reactivated.
            for position in 0..self.network.outgoing[orphan].len() {
                let arc = self.network.outgoing[orphan][position];
                let neighbour = self.network.target[arc];
                if self.tree[neighbour] != tree {
                    continue;
                }
                let travel = if tree == Tree::Source { arc ^ 1 } else { arc };
                if self.network.capacity[travel] > 0.0 {
                    self.active.push_back(neighbour);
                }
                if self.parent[neighbour] != NONE && self.parent_node(neighbour) == orphan {
                    self.parent[neighbour] = NONE;
                    self.orphans.push(neighbour);
                }
            }
            self.tree[orphan] = Tree::Free;
            self.parent[orphan] = NONE;
        }
    }

    fn run(&mut self) -> f64 {
        self.tree[self.source] = Tree::Source;
        self.tree[self.sink] = Tree::Sink;
        self.stamp[self.source] = 1;
        self.stamp[self.sink] = 1;
        self.time = 1;
        self.active.push_back(self.source);
        self.active.push_back(self.sink);
        let mut total = 0.0;
        while let Some(crossing) = self.grow() {
            total += self.augment(crossing);
            self.time += 1;
            self.stamp[self.source] = self.time;
            self.stamp[self.sink] = self.time;
            self.adopt();
        }
        total
    }
}

/// Boykov--Kolmogorov maximum flow; returns the flow value.
fn boykov_kolmogorov(network: &mut FlowNetwork, source: usize, sink: usize) -> f64 {
    let n = network.n_nodes;
    let mut state = BoykovKolmogorov {
        network,
        source,
        sink,
        tree: vec![Tree::Free; n],
        parent: vec![NONE; n],
        active: VecDeque::new(),
        orphans: Vec::new(),
        stamp: vec![0; n],
        distance: vec![0; n],
        time: 0,
    };
    state.run()
}

/// Build the network for a two-state ferromagnetic Ising ground state.
///
/// `field` is `n_nodes * 2` in row-major order, `edges` is `2 * n_edges` as
/// flattened `(i, j)` pairs, and `coupling` is one entry per edge. The
/// construction is the one `snakes_and_ladders.search.maxflow.ising_ground_state`
/// documents; it is duplicated here rather than shared because the two
/// implementations must be independent for one to be the other's oracle.
pub(crate) fn ising_network(
    n_nodes: usize,
    field: &[f64],
    edges: &[usize],
    coupling: &[f64],
) -> Result<FlowNetwork, String> {
    if field.len() != 2 * n_nodes {
        return Err(format!(
            "field must have {} entries for {n_nodes} nodes, got {}",
            2 * n_nodes,
            field.len()
        ));
    }
    if edges.len() != 2 * coupling.len() {
        return Err(format!(
            "edges has {} entries for {} couplings; expected {}",
            edges.len(),
            coupling.len(),
            2 * coupling.len()
        ));
    }
    if let Some(negative) = coupling.iter().find(|&&j| j < 0.0) {
        return Err(format!(
            "every coupling must be non-negative, got {negative}: a negative \
             coupling makes the energy non-submodular and the ground state \
             NP-hard"
        ));
    }

    let (source, sink) = (n_nodes, n_nodes + 1);
    let mut network = FlowNetwork::new(n_nodes + 2);
    for node in 0..n_nodes {
        let cost_zero = -field[2 * node];
        let cost_one = -field[2 * node + 1];
        let offset = cost_zero.min(cost_one);
        network.add_edge(source, node, cost_one - offset, 0.0)?;
        network.add_edge(node, sink, cost_zero - offset, 0.0)?;
    }
    for (position, &weight) in coupling.iter().enumerate() {
        network.add_edge(edges[2 * position], edges[2 * position + 1], weight, weight)?;
    }
    Ok(network)
}

/// The exact ground state of a two-state ferromagnet, as one state per node.
pub fn ising_ground_state_impl(
    n_nodes: usize,
    field: &[f64],
    edges: &[usize],
    coupling: &[f64],
) -> Result<Vec<i64>, String> {
    let mut network = ising_network(n_nodes, field, edges, coupling)?;
    let (_, side) = max_flow_impl(&mut network, n_nodes, n_nodes + 1)?;
    Ok(side[..n_nodes]
        .iter()
        .map(|&reachable| i64::from(!reachable))
        .collect())
}

/// Node indices from an `int64` array, refusing a negative one.
///
/// Indices cross the boundary as `int64` because that is what `numpy` builds
/// by default and what `potts.rs` and `sampling.rs` already take; the kernel
/// indexes with `usize`, so the conversion is one checked pass in Rust rather
/// than one Python integer per entry.
pub(crate) fn node_indices(indices: &[i64]) -> PyResult<Vec<usize>> {
    indices
        .iter()
        .map(|&index| {
            usize::try_from(index).map_err(|_| {
                PyValueError::new_err(format!("node indices must be non-negative, got {index}"))
            })
        })
        .collect()
}

/// Maximum flow on an explicitly given network, and the cut it certifies.
///
/// `arcs` is `2 * n_arcs` flattened `(from, to)` pairs and `capacity` one
/// entry per arc. `reverse` is the back arc's capacity, one entry per arc:
/// `None` leaves every back arc at zero, which is a directed network, and a
/// caller wanting an undirected edge passes the same capacity in both.
///
/// Returns `(value, source_side)`. Until issue #528 this returned the value
/// alone and discarded the side [`max_flow_impl`] had already built, which
/// left `search.alpha_expansion` --- whose expansion step needs the cut and
/// not the flow --- on the Python solver, at 49.0% of its self time. The
/// side costs one `Vec<bool>` per call and no second traversal.
///
/// **Arrays are borrowed, not copied.** The first binding took `Vec<usize>`
/// and `Vec<f64>`, so PyO3 read one Python object per entry on the way in
/// (issue #336). `rust-numpy` hands over the buffer itself, the contract
/// `sampling::sample_rows` and `pruning::pruning_log_likelihood` already
/// state.
#[pyfunction]
#[pyo3(signature = (n_nodes, arcs, capacity, source, sink, reverse=None))]
pub fn max_flow<'py>(
    py: Python<'py>,
    n_nodes: usize,
    arcs: PyReadonlyArray1<'_, i64>,
    capacity: PyReadonlyArray1<'_, f64>,
    source: usize,
    sink: usize,
    reverse: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<(f64, Bound<'py, PyArray1<bool>>)> {
    // `as_slice` succeeds only for a C-contiguous array, the same contract
    // `sampling::sample_rows` states; the wrapper normalizes with
    // `ascontiguousarray`, free when the array already is one.
    let arcs = node_indices(arcs.as_slice()?)?;
    let capacity = capacity.as_slice()?;
    if arcs.len() != 2 * capacity.len() {
        return Err(PyValueError::new_err(format!(
            "arcs has {} entries for {} capacities",
            arcs.len(),
            capacity.len()
        )));
    }
    let back = match reverse.as_ref() {
        Some(array) => Some(array.as_slice()?),
        None => None,
    };
    if let Some(back) = back {
        if back.len() != capacity.len() {
            return Err(PyValueError::new_err(format!(
                "reverse has {} entries for {} capacities",
                back.len(),
                capacity.len()
            )));
        }
    }
    let mut network = FlowNetwork::new(n_nodes);
    for (position, &weight) in capacity.iter().enumerate() {
        network
            .add_edge(
                arcs[2 * position],
                arcs[2 * position + 1],
                weight,
                back.map_or(0.0, |back| back[position]),
            )
            .map_err(PyValueError::new_err)?;
    }
    let (value, side) = py
        .detach(|| max_flow_impl(&mut network, source, sink))
        .map_err(PyValueError::new_err)?;
    Ok((value, PyArray1::from_vec(py, side)))
}

/// The exact ground state of a two-state ferromagnetic Ising model.
///
/// `field` is `2 * n_nodes` `float64` in row-major order, `edges` is
/// `2 * n_edges` `int64` as flattened `(i, j)` pairs, and `coupling` one
/// `float64` per edge. All three are borrowed (see [`max_flow`]), and the
/// result comes back as an `int64` array rather than a list, so nothing at
/// the boundary is built one Python object at a time.
#[pyfunction]
#[pyo3(signature = (n_nodes, field, edges, coupling))]
pub fn ising_ground_state<'py>(
    py: Python<'py>,
    n_nodes: usize,
    field: PyReadonlyArray1<'py, f64>,
    edges: PyReadonlyArray1<'py, i64>,
    coupling: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let edges = node_indices(edges.as_slice()?)?;
    let field = field.as_slice()?;
    let coupling = coupling.as_slice()?;
    let states = py
        .detach(|| ising_ground_state_impl(n_nodes, field, &edges, coupling))
        .map_err(PyValueError::new_err)?;
    Ok(PyArray1::from_vec(py, states))
}

/// The ground states of a batch of fields on one graph, one cut per field.
///
/// `fields` is `batch * 2 * n_nodes`, row-major, and the result
/// `batch * n_nodes`. The cuts are independent, so the batch is the axis
/// rayon takes (root `CLAUDE.md`, parallel over independent tasks): a
/// `par_iter` over the fields, each building and solving its own network,
/// with the GIL released for the whole call. `threads` sizes the pool,
/// `None` for the global one. Measured on the 4-core host (issue #715):
/// eight 256x256 cuts in 2.16 s on one thread and 0.90 s on four.
#[pyfunction]
#[pyo3(signature = (n_nodes, fields, edges, coupling, threads=None))]
pub fn ising_ground_states<'py>(
    py: Python<'py>,
    n_nodes: usize,
    fields: PyReadonlyArray1<'py, f64>,
    edges: PyReadonlyArray1<'py, i64>,
    coupling: PyReadonlyArray1<'py, f64>,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let edges = node_indices(edges.as_slice()?)?;
    let fields = fields.as_slice()?;
    let coupling = coupling.as_slice()?;
    let width = 2 * n_nodes;
    if width == 0 || fields.len() % width != 0 {
        return Err(PyValueError::new_err(format!(
            "fields has {} entries, not a multiple of 2 * {n_nodes}",
            fields.len()
        )));
    }
    let states = py
        .detach(|| {
            on_pool(threads, || {
                fields
                    .par_chunks(width)
                    .map(|field| ising_ground_state_impl(n_nodes, field, &edges, coupling))
                    .collect::<Result<Vec<Vec<i64>>, String>>()
            })
        })
        .map_err(PyValueError::new_err)?;
    Ok(PyArray1::from_vec(py, states.concat()))
}

/// Run `body` on a pool of `threads`, or on the global pool for `None`.
pub(crate) fn on_pool<T: Send>(threads: Option<usize>, body: impl FnOnce() -> T + Send) -> T {
    match threads {
        None => body(),
        Some(count) => rayon::ThreadPoolBuilder::new()
            .num_threads(count.max(1))
            .build()
            .expect("a rayon pool")
            .install(body),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_edge_carries_its_capacity() {
        let mut network = FlowNetwork::new(2);
        network.add_edge(0, 1, 7.5, 0.0).unwrap();
        let (value, _) = max_flow_impl(&mut network, 0, 1).unwrap();
        assert!((value - 7.5).abs() < 1e-12);
    }

    #[test]
    fn a_diamond_needs_the_cross_edge_to_reach_its_maximum() {
        // Two disjoint paths carry 2 each; the 1 -> 2 edge carries a third
        // unit that a greedy first path would have blocked.
        let mut network = FlowNetwork::new(4);
        network.add_edge(0, 1, 3.0, 0.0).unwrap();
        network.add_edge(0, 2, 2.0, 0.0).unwrap();
        network.add_edge(1, 3, 2.0, 0.0).unwrap();
        network.add_edge(2, 3, 3.0, 0.0).unwrap();
        network.add_edge(1, 2, 1.0, 0.0).unwrap();
        let (value, _) = max_flow_impl(&mut network, 0, 3).unwrap();
        assert!((value - 5.0).abs() < 1e-12);
    }

    #[test]
    fn a_disconnected_sink_receives_nothing() {
        let mut network = FlowNetwork::new(3);
        network.add_edge(0, 1, 4.0, 0.0).unwrap();
        let (value, side) = max_flow_impl(&mut network, 0, 2).unwrap();
        assert!(value.abs() < 1e-12);
        assert!(!side[2]);
    }

    #[test]
    fn the_cut_separates_the_terminals() {
        let mut network = FlowNetwork::new(4);
        network.add_edge(0, 1, 3.0, 0.0).unwrap();
        network.add_edge(1, 2, 1.0, 0.0).unwrap();
        network.add_edge(2, 3, 3.0, 0.0).unwrap();
        let (value, side) = max_flow_impl(&mut network, 0, 3).unwrap();
        assert!((value - 1.0).abs() < 1e-12);
        assert!(side[0] && !side[3]);
    }

    #[test]
    fn coincident_terminals_are_refused() {
        let mut network = FlowNetwork::new(2);
        assert!(max_flow_impl(&mut network, 1, 1).is_err());
    }

    #[test]
    fn a_negative_capacity_is_refused() {
        let mut network = FlowNetwork::new(2);
        assert!(network.add_edge(0, 1, -1.0, 0.0).is_err());
    }

    #[test]
    fn a_negative_coupling_is_refused() {
        let field = vec![0.0, 0.0, 0.0, 0.0];
        let edges = vec![0, 1];
        assert!(ising_ground_state_impl(2, &field, &edges, &[-0.5]).is_err());
    }

    #[test]
    fn a_field_of_the_wrong_length_is_refused() {
        assert!(ising_ground_state_impl(2, &[0.0, 0.0], &[0, 1], &[0.5]).is_err());
    }

    #[test]
    fn a_zero_coupling_chain_follows_its_field_site_by_site() {
        // With no coupling every site independently takes its better state,
        // so the answer is known without solving anything.
        let field = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
        let ground = ising_ground_state_impl(3, &field, &[0, 1, 1, 2], &[0.0, 0.0]).unwrap();
        assert_eq!(ground, vec![0, 1, 0]);
    }

    #[test]
    fn a_strong_coupling_overrides_a_weak_field() {
        // Two sites pulled to opposite states by a weak field, bound by a
        // coupling stronger than the disagreement is worth: they must align.
        let field = vec![0.1, 0.0, 0.0, 0.1];
        let ground = ising_ground_state_impl(2, &field, &[0, 1], &[5.0]).unwrap();
        assert_eq!(ground[0], ground[1]);
    }
}
