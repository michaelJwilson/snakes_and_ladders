//! Dinic's maximum-flow algorithm, and the Ising ground-state reduction.
//!
//! The Python implementation in `snakes_and_ladders.search.maxflow` stays as the oracle
//! this is pinned against, per root `CLAUDE.md`. What Rust buys is what root
//! `CLAUDE.md` reserves it for: this is control flow and irregular memory
//! access over an adjacency structure, with no array arithmetic a vectorized
//! NumPy version could exploit. Measured on a 100x100 lattice with a per-node
//! field, the Python reference takes 744 ms.
//!
//! Two differences from the reference, both deliberate.
//!
//! The blocking flow is **iterative**, not recursive. The Python version
//! recurses to the depth of the level graph and needs `setrecursionlimit`
//! raised past a few thousand nodes; a lattice deep enough is a stack
//! overflow rather than a slow answer. An explicit stack has no such bound.
//!
//! Capacities are `f64` and the termination test is `> 0.0` rather than a
//! tolerance, matching the reference exactly so the two cannot disagree on
//! which arcs are saturated.
//!
//! As in `pruning.rs`, the kernel is a plain function returning `Result` and
//! the `#[pyfunction]` is a thin wrapper, so `cargo test` exercises the
//! algorithm without touching `PyResult`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// A flow network as paired residual arcs.
///
/// Arc `2 * e` and `2 * e + 1` are the two directions of one edge, so an
/// arc's reverse is its index with the low bit flipped. Pushing flow
/// subtracts from one and adds to the other, which keeps the residual graph
/// implicit rather than a second structure to hold in step.
pub struct FlowNetwork {
    n_nodes: usize,
    target: Vec<usize>,
    capacity: Vec<f64>,
    outgoing: Vec<Vec<usize>>,
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
    fn levels(&self, source: usize) -> Vec<usize> {
        let mut level = vec![usize::MAX; self.n_nodes];
        level[source] = 0;
        let mut queue = std::collections::VecDeque::new();
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

    /// One level-respecting augmenting path, found with an explicit stack.
    ///
    /// `progress` is what keeps the blocking flow linear: an arc that cannot
    /// carry more in this phase is never revisited, so each is examined once
    /// per level graph.
    fn augment(
        &mut self,
        source: usize,
        sink: usize,
        level: &[usize],
        progress: &mut [usize],
    ) -> f64 {
        let mut path: Vec<usize> = Vec::new();
        let mut node = source;
        loop {
            if node == sink {
                // The bottleneck is the smallest residual capacity on the path.
                let bottleneck = path
                    .iter()
                    .map(|&arc| self.capacity[arc])
                    .fold(f64::INFINITY, f64::min);
                for &arc in &path {
                    self.capacity[arc] -= bottleneck;
                    self.capacity[arc ^ 1] += bottleneck;
                }
                return bottleneck;
            }

            let mut advanced = false;
            while progress[node] < self.outgoing[node].len() {
                let arc = self.outgoing[node][progress[node]];
                let neighbour = self.target[arc];
                if self.capacity[arc] > 0.0
                    && level[neighbour] != usize::MAX
                    && level[neighbour] == level[node] + 1
                {
                    path.push(arc);
                    node = neighbour;
                    advanced = true;
                    break;
                }
                progress[node] += 1;
            }
            if advanced {
                continue;
            }

            // Dead end: retreat, and mark the arc that led here exhausted so
            // this phase never tries it again.
            match path.pop() {
                None => return 0.0,
                Some(arc) => {
                    node = self.target[arc ^ 1];
                    progress[node] += 1;
                }
            }
        }
    }
}

/// Maximum flow, and the source side of the minimum cut it certifies.
///
/// Returns `(value, source_side)`, where `source_side[i]` is whether node `i`
/// is reachable from the source in the residual graph on termination. That
/// set *is* a minimum cut, by the max-flow min-cut theorem, so nothing here
/// searches for one separately.
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

    let mut total = 0.0;
    loop {
        let level = network.levels(source);
        if level[sink] == usize::MAX {
            break;
        }
        let mut progress = vec![0usize; network.n_nodes];
        loop {
            let pushed = network.augment(source, sink, &level, &mut progress);
            if pushed <= 0.0 {
                break;
            }
            total += pushed;
        }
    }

    let level = network.levels(source);
    let side = level.iter().map(|&d| d != usize::MAX).collect();
    Ok((total, side))
}

/// Build the network for a two-state ferromagnetic Ising ground state.
///
/// `field` is `n_nodes * 2` in row-major order, `edges` is `2 * n_edges` as
/// flattened `(i, j)` pairs, and `coupling` is one entry per edge. The
/// construction is the one `snakes_and_ladders.search.maxflow.ising_ground_state`
/// documents; it is duplicated here rather than shared because the two
/// implementations must be independent for one to be the other's oracle.
fn ising_network(
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

/// Maximum flow on an explicitly given network.
///
/// `arcs` is `2 * n_arcs` flattened `(from, to)` pairs and `capacity` one
/// entry per arc; back arcs are added automatically with zero capacity, so a
/// caller wanting an undirected edge passes it twice.
#[pyfunction]
#[pyo3(signature = (n_nodes, arcs, capacity, source, sink))]
pub fn max_flow(
    n_nodes: usize,
    arcs: Vec<usize>,
    capacity: Vec<f64>,
    source: usize,
    sink: usize,
) -> PyResult<f64> {
    if arcs.len() != 2 * capacity.len() {
        return Err(PyValueError::new_err(format!(
            "arcs has {} entries for {} capacities",
            arcs.len(),
            capacity.len()
        )));
    }
    let mut network = FlowNetwork::new(n_nodes);
    for (position, &weight) in capacity.iter().enumerate() {
        network
            .add_edge(arcs[2 * position], arcs[2 * position + 1], weight, 0.0)
            .map_err(PyValueError::new_err)?;
    }
    let (value, _) = max_flow_impl(&mut network, source, sink).map_err(PyValueError::new_err)?;
    Ok(value)
}

/// The exact ground state of a two-state ferromagnetic Ising model.
#[pyfunction]
#[pyo3(signature = (n_nodes, field, edges, coupling))]
pub fn ising_ground_state(
    n_nodes: usize,
    field: Vec<f64>,
    edges: Vec<usize>,
    coupling: Vec<f64>,
) -> PyResult<Vec<i64>> {
    ising_ground_state_impl(n_nodes, &field, &edges, &coupling).map_err(PyValueError::new_err)
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
