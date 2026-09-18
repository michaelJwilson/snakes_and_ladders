//! The three maximum-flow kernels issue #715 declined, conserved behind the
//! `sandbox` feature so the comparison that declined them stays re-runnable.
//!
//! Boykov--Kolmogorov won both of the ticket's tables and lives in
//! `maxflow.rs`; these lost and live here, on `sandbox/CLAUDE.md`'s declined
//! rule. Each takes the same [`FlowNetwork`] of paired residual arcs, returns
//! the flow value, and leaves the residual graph for [`max_flow_with`] to
//! read the source side off, so every kernel certifies its cut the same way
//! the package kernel does and the Python Dinic pins all of them.
//!
//! * [`dinic`]: level graphs and blocking flows, the port of the Python
//!   reference that was the package kernel from issue #220 to #715. 234 ms on
//!   the 256x256 cut against Boykov--Kolmogorov's 61.
//! * [`push_relabel`]: Goldberg--Tarjan (1988) with highest-label selection
//!   (Cheriyan--Maheshwari 1989) over buckets, a global relabel by reverse
//!   breadth-first search from the sink, and the gap heuristic. Both phases
//!   run --- excess is returned to the source --- so what is left is a flow
//!   and not a preflow. 1,680 ms on the 256x256 cut: the best generic worst
//!   case, and the worst constant on a grid.
//! * [`parallel_push_relabel`]: the synchronous variant (Baumstark, Blelloch
//!   and Shun 2015). A round computes every active vertex's pushes in parallel
//!   from the round's opening state and applies them in vertex order, then
//!   relabels in parallel from the state that results; the arithmetic is
//!   therefore a fixed sequence whatever the thread count, and the output is
//!   identical at one thread and at four. 339 ms on the 256x256 cut, and no
//!   faster on four threads than on one: below 65,536 nodes a round is
//!   smaller than its barrier.
//!
//! Capacities are `f64` and saturation is `> 0.0`, as in `maxflow.rs`, so no
//! kernel here can disagree with the reference about which arcs are cut.

use std::collections::VecDeque;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::maxflow::{ising_network, node_indices, on_pool, FlowNetwork};

/// The declined kernels, by the name the Python front spells.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Algorithm {
    /// The package kernel, reachable here so one table holds all four.
    BoykovKolmogorov,
    /// Level graphs and blocking flows.
    Dinic,
    /// Goldberg--Tarjan, highest label first, with global relabel and gap.
    PushRelabel,
    /// Synchronous parallel push-relabel on rayon.
    ParallelPushRelabel,
}

impl Algorithm {
    /// The kernel a name selects.
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "boykov-kolmogorov" => Ok(Self::BoykovKolmogorov),
            "dinic" => Ok(Self::Dinic),
            "push-relabel" => Ok(Self::PushRelabel),
            "parallel-push-relabel" => Ok(Self::ParallelPushRelabel),
            other => Err(format!(
                "unknown max-flow algorithm {other:?}; one of boykov-kolmogorov, dinic, \
                 push-relabel, parallel-push-relabel"
            )),
        }
    }
}

/// Maximum flow by the named kernel, and the minimal minimum cut it certifies.
pub fn max_flow_with(
    network: &mut FlowNetwork,
    source: usize,
    sink: usize,
    algorithm: Algorithm,
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
    let value = match algorithm {
        Algorithm::BoykovKolmogorov => return crate::maxflow::max_flow_impl(network, source, sink),
        Algorithm::Dinic => dinic(network, source, sink),
        Algorithm::PushRelabel => push_relabel(network, source, sink),
        Algorithm::ParallelPushRelabel => parallel_push_relabel(network, source, sink),
    };
    let level = network.levels(source);
    Ok((value, level.iter().map(|&d| d != usize::MAX).collect()))
}

/// Distance to the sink over residual arcs, by reverse breadth-first search.
///
/// `unreached` is the label a node that cannot reach the sink receives; the
/// callers pass `n` (phase one) or `2n` (nothing left to route).
fn sink_distances(network: &FlowNetwork, sink: usize, unreached: usize) -> Vec<usize> {
    let mut label = vec![unreached; network.n_nodes];
    label[sink] = 0;
    let mut queue = VecDeque::with_capacity(network.n_nodes);
    queue.push_back(sink);
    while let Some(node) = queue.pop_front() {
        for &arc in &network.outgoing[node] {
            // `arc` runs node -> neighbour; its reverse runs neighbour -> node,
            // which is the direction flow would travel toward the sink.
            let neighbour = network.target[arc];
            if network.capacity[arc ^ 1] > 0.0 && label[neighbour] == unreached {
                label[neighbour] = label[node] + 1;
                queue.push_back(neighbour);
            }
        }
    }
    label
}

// ---------------------------------------------------------------------------
// Dinic.
// ---------------------------------------------------------------------------

/// One level-respecting augmenting path, found with an explicit stack.
///
/// `progress` is what keeps the blocking flow linear: an arc that cannot
/// carry more in this phase is never revisited, so each is examined once
/// per level graph.
fn augment(
    network: &mut FlowNetwork,
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
                .map(|&arc| network.capacity[arc])
                .fold(f64::INFINITY, f64::min);
            for &arc in &path {
                network.capacity[arc] -= bottleneck;
                network.capacity[arc ^ 1] += bottleneck;
            }
            return bottleneck;
        }

        let mut advanced = false;
        while progress[node] < network.outgoing[node].len() {
            let arc = network.outgoing[node][progress[node]];
            let neighbour = network.target[arc];
            if network.capacity[arc] > 0.0
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
                node = network.target[arc ^ 1];
                progress[node] += 1;
            }
        }
    }
}

/// Dinic's algorithm: repeated level graphs and blocking flows; the value.
pub fn dinic(network: &mut FlowNetwork, source: usize, sink: usize) -> f64 {
    let mut total = 0.0;
    loop {
        let level = network.levels(source);
        if level[sink] == usize::MAX {
            break;
        }
        let mut progress = vec![0usize; network.n_nodes];
        loop {
            let pushed = augment(network, source, sink, &level, &mut progress);
            if pushed <= 0.0 {
                break;
            }
            total += pushed;
        }
    }
    total
}

// ---------------------------------------------------------------------------
// Highest-label push-relabel.
// ---------------------------------------------------------------------------

struct PushRelabel<'a> {
    network: &'a mut FlowNetwork,
    source: usize,
    sink: usize,
    excess: Vec<f64>,
    label: Vec<usize>,
    current: Vec<usize>,
    /// Active nodes by label; a stack per label, the highest served first.
    active: Vec<Vec<usize>>,
    is_active: Vec<bool>,
    highest: usize,
    /// Nodes per label below `n`, for the gap heuristic.
    count: Vec<usize>,
    /// Relabel work since the last global relabel.
    work: usize,
}

impl PushRelabel<'_> {
    fn n(&self) -> usize {
        self.network.n_nodes
    }

    fn activate(&mut self, node: usize) {
        if !self.is_active[node] && node != self.source && node != self.sink {
            self.is_active[node] = true;
            let label = self.label[node];
            self.active[label].push(node);
            if label > self.highest {
                self.highest = label;
            }
        }
    }

    /// Labels from a reverse search from the sink, and the buckets rebuilt.
    fn global_relabel(&mut self) {
        let n = self.n();
        let distance = sink_distances(self.network, self.sink, 2 * n);
        // A node that cannot reach the sink routes its excess back to the
        // source and is labelled `n` plus its residual distance to it.
        let back = sink_distances(self.network, self.source, n);
        for node in 0..n {
            if node == self.source {
                continue;
            }
            self.label[node] = if distance[node] >= 2 * n {
                (n + back[node]).min(2 * n)
            } else {
                distance[node]
            };
        }
        self.label[self.source] = n;
        self.count.iter_mut().for_each(|c| *c = 0);
        for node in 0..n {
            if node != self.source && self.label[node] < n {
                self.count[self.label[node]] += 1;
            }
        }
        for bucket in &mut self.active {
            bucket.clear();
        }
        self.highest = 0;
        let mut pending: Vec<usize> = (0..n)
            .filter(|&node| self.is_active[node] && self.label[node] < 2 * n)
            .collect();
        for node in &pending {
            self.is_active[*node] = false;
        }
        for node in pending.drain(..) {
            self.activate(node);
        }
        self.current.iter_mut().for_each(|c| *c = 0);
        self.work = 0;
    }

    fn discharge(&mut self, node: usize) {
        let n = self.n();
        while self.excess[node] > 0.0 {
            let degree = self.network.outgoing[node].len();
            if self.current[node] >= degree {
                // Relabel: one past the lowest label across residual arcs.
                let old = self.label[node];
                let mut lowest = 2 * n;
                for &arc in &self.network.outgoing[node] {
                    if self.network.capacity[arc] > 0.0 {
                        lowest = lowest.min(self.label[self.network.target[arc]]);
                    }
                }
                let new = (lowest + 1).min(2 * n);
                self.label[node] = new;
                self.current[node] = 0;
                self.work += 12 + degree;
                if old < n {
                    self.count[old] -= 1;
                    if new < n {
                        self.count[new] += 1;
                    }
                    if self.count[old] == 0 {
                        self.gap(old);
                    }
                }
                if self.label[node] >= 2 * n {
                    return;
                }
                continue;
            }
            let arc = self.network.outgoing[node][self.current[node]];
            let neighbour = self.network.target[arc];
            let residual = self.network.capacity[arc];
            if residual > 0.0 && self.label[node] == self.label[neighbour] + 1 {
                let delta = self.excess[node].min(residual);
                self.network.capacity[arc] -= delta;
                self.network.capacity[arc ^ 1] += delta;
                self.excess[node] -= delta;
                self.excess[neighbour] += delta;
                self.activate(neighbour);
                if self.excess[node] > 0.0 {
                    self.current[node] += 1;
                }
            } else {
                self.current[node] += 1;
            }
        }
    }

    /// No node is labelled `gone`, so every node between it and `n` cannot
    /// reach the sink and is relabelled to route back to the source.
    fn gap(&mut self, gone: usize) {
        let n = self.n();
        for node in 0..n {
            let label = self.label[node];
            if node != self.source && label > gone && label < n {
                self.count[label] -= 1;
                self.label[node] = n + 1;
                if self.is_active[node] {
                    // It sits in a bucket it no longer belongs to; the bucket
                    // walk below skips stale entries by label.
                    self.is_active[node] = false;
                    self.activate(node);
                }
            }
        }
    }

    fn run(&mut self) -> f64 {
        let n = self.n();
        self.label[self.source] = n;
        for position in 0..self.network.outgoing[self.source].len() {
            let arc = self.network.outgoing[self.source][position];
            let capacity = self.network.capacity[arc];
            if capacity > 0.0 {
                let neighbour = self.network.target[arc];
                self.network.capacity[arc] = 0.0;
                self.network.capacity[arc ^ 1] += capacity;
                self.excess[neighbour] += capacity;
                self.excess[self.source] -= capacity;
            }
        }
        for node in 0..n {
            if self.excess[node] > 0.0 {
                self.activate(node);
            }
        }
        self.global_relabel();
        let threshold = 6 * n + self.network.target.len();
        loop {
            // Highest label first; buckets may hold stale entries after a
            // relabel, which the label check discards.
            let node = loop {
                if self.highest == 0 && self.active[0].is_empty() {
                    break None;
                }
                match self.active[self.highest].pop() {
                    Some(node) if self.is_active[node] && self.label[node] == self.highest => {
                        self.is_active[node] = false;
                        break Some(node);
                    }
                    Some(_) => continue,
                    None => {
                        if self.highest == 0 {
                            break None;
                        }
                        self.highest -= 1;
                    }
                }
            };
            let Some(node) = node else { break };
            if self.excess[node] <= 0.0 || self.label[node] >= 2 * n {
                continue;
            }
            self.discharge(node);
            if self.excess[node] > 0.0 && self.label[node] < 2 * n {
                self.activate(node);
            }
            if self.work > threshold {
                self.global_relabel();
            }
        }
        self.excess[self.sink]
    }
}

/// Goldberg--Tarjan push-relabel, highest label first; returns the flow value.
pub fn push_relabel(network: &mut FlowNetwork, source: usize, sink: usize) -> f64 {
    let n = network.n_nodes;
    let mut state = PushRelabel {
        network,
        source,
        sink,
        excess: vec![0.0; n],
        label: vec![0; n],
        current: vec![0; n],
        active: vec![Vec::new(); 2 * n + 1],
        is_active: vec![false; n],
        highest: 0,
        count: vec![0; n + 1],
        work: 0,
    };
    state.run()
}

// ---------------------------------------------------------------------------
// Synchronous parallel push-relabel.
// ---------------------------------------------------------------------------

/// One vertex's pushes for a round, computed from the round's opening state.
struct Pushes {
    node: usize,
    arcs: Vec<(usize, f64)>,
}

/// Synchronous parallel push-relabel; returns the flow value.
///
/// Runs on the current rayon pool, so a caller that wants a fixed thread
/// count installs one around the call. The output does not depend on it.
pub fn parallel_push_relabel(network: &mut FlowNetwork, source: usize, sink: usize) -> f64 {
    let n = network.n_nodes;
    let mut excess = vec![0.0f64; n];
    for position in 0..network.outgoing[source].len() {
        let arc = network.outgoing[source][position];
        let capacity = network.capacity[arc];
        if capacity > 0.0 {
            let neighbour = network.target[arc];
            network.capacity[arc] = 0.0;
            network.capacity[arc ^ 1] += capacity;
            excess[neighbour] += capacity;
            excess[source] -= capacity;
        }
    }
    let mut label = sink_distances(network, sink, n);
    label[source] = n;
    // Global relabel on the same work budget as the sequential kernel: the
    // arcs scanned since the last one, against `6n + m`.
    let threshold = 6 * n + network.target.len();
    let mut work = 0usize;

    loop {
        let active: Vec<usize> = (0..n)
            .filter(|&node| {
                node != source && node != sink && excess[node] > 0.0 && label[node] < 2 * n
            })
            .collect();
        if active.is_empty() {
            break;
        }

        // Phase one: pushes from the opening state, in parallel. A vertex
        // reads only its own arcs and the labels, and no two active vertices
        // can be admissible toward each other in one round.
        let net: &FlowNetwork = network;
        let label_ref: &[usize] = &label;
        let excess_ref: &[f64] = &excess;
        let planned: Vec<Pushes> = active
            .par_iter()
            .map(|&node| {
                let mut remaining = excess_ref[node];
                let mut arcs = Vec::new();
                for &arc in &net.outgoing[node] {
                    if remaining <= 0.0 {
                        break;
                    }
                    let residual = net.capacity[arc];
                    if residual > 0.0 && label_ref[node] == label_ref[net.target[arc]] + 1 {
                        let delta = remaining.min(residual);
                        arcs.push((arc, delta));
                        remaining -= delta;
                    }
                }
                Pushes { node, arcs }
            })
            .collect();

        // Apply in vertex order: a fixed sequence of floating-point updates.
        for plan in &planned {
            for &(arc, delta) in &plan.arcs {
                network.capacity[arc] -= delta;
                network.capacity[arc ^ 1] += delta;
                excess[plan.node] -= delta;
                excess[network.target[arc]] += delta;
            }
        }

        // Phase two: relabel, in parallel from the state that resulted.
        work += active
            .iter()
            .map(|&node| network.outgoing[node].len() + 12)
            .sum::<usize>();
        if work > threshold {
            work = 0;
            // Two-sided, as the sequential kernel's: a node that cannot reach
            // the sink is labelled `n` plus its residual distance to the
            // source, so it can push back. Keeping such a node's running
            // label instead pinned every source-side node at `n` on a network
            // with no path to the sink, where every round was a global
            // relabel and none could reach `n + 1`: the expansion's first
            // swap network (81 nodes, flow 0) never terminated.
            let mut fresh = sink_distances(network, sink, 2 * n);
            let back = sink_distances(network, source, n);
            for node in 0..n {
                if fresh[node] >= 2 * n {
                    fresh[node] = (n + back[node]).min(2 * n);
                }
            }
            fresh[source] = n;
            label = fresh;
        } else {
            let net: &FlowNetwork = network;
            let label_ref: &[usize] = &label;
            let excess_ref: &[f64] = &excess;
            let relabelled: Vec<(usize, usize)> = active
                .par_iter()
                .filter(|&&node| excess_ref[node] > 0.0)
                .map(|&node| {
                    let mut lowest = 2 * n;
                    for &arc in &net.outgoing[node] {
                        if net.capacity[arc] > 0.0 {
                            lowest = lowest.min(label_ref[net.target[arc]]);
                        }
                    }
                    (node, (lowest + 1).min(2 * n).max(label_ref[node]))
                })
                .collect();
            for (node, new) in relabelled {
                label[node] = new;
            }
        }
    }
    excess[sink]
}

// ---------------------------------------------------------------------------
// The bindings the sandbox front calls.
// ---------------------------------------------------------------------------

/// Maximum flow on an explicit network by a declined kernel; see `max_flow`.
///
/// `algorithm` names the kernel (see [`Algorithm::parse`]) and `threads` the
/// rayon pool the parallel one runs on, `None` for the global pool; the
/// sequential kernels ignore it.
#[pyfunction]
#[pyo3(signature = (n_nodes, arcs, capacity, source, sink, reverse=None, algorithm="dinic", threads=None))]
#[allow(clippy::too_many_arguments)]
pub fn max_flow_declined<'py>(
    py: Python<'py>,
    n_nodes: usize,
    arcs: PyReadonlyArray1<'_, i64>,
    capacity: PyReadonlyArray1<'_, f64>,
    source: usize,
    sink: usize,
    reverse: Option<PyReadonlyArray1<'_, f64>>,
    algorithm: &str,
    threads: Option<usize>,
) -> PyResult<(f64, Bound<'py, PyArray1<bool>>)> {
    let algorithm = Algorithm::parse(algorithm).map_err(PyValueError::new_err)?;
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
        .detach(|| {
            on_pool(threads, || {
                max_flow_with(&mut network, source, sink, algorithm)
            })
        })
        .map_err(PyValueError::new_err)?;
    Ok((value, PyArray1::from_vec(py, side)))
}

/// The Ising ground state by a declined kernel; see `ising_ground_state`.
#[pyfunction]
#[pyo3(signature = (n_nodes, field, edges, coupling, algorithm="dinic", threads=None))]
pub fn ising_ground_state_declined<'py>(
    py: Python<'py>,
    n_nodes: usize,
    field: PyReadonlyArray1<'py, f64>,
    edges: PyReadonlyArray1<'py, i64>,
    coupling: PyReadonlyArray1<'py, f64>,
    algorithm: &str,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let algorithm = Algorithm::parse(algorithm).map_err(PyValueError::new_err)?;
    let edges = node_indices(edges.as_slice()?)?;
    let field = field.as_slice()?;
    let coupling = coupling.as_slice()?;
    let states = py
        .detach(|| {
            on_pool(threads, || {
                let mut network = ising_network(n_nodes, field, &edges, coupling)?;
                let (_, side) = max_flow_with(&mut network, n_nodes, n_nodes + 1, algorithm)?;
                Ok::<Vec<i64>, String>(
                    side[..n_nodes]
                        .iter()
                        .map(|&reachable| i64::from(!reachable))
                        .collect(),
                )
            })
        })
        .map_err(PyValueError::new_err)?;
    Ok(PyArray1::from_vec(py, states))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::maxflow::max_flow_impl;

    /// A seeded network with back capacities, the shape the Python suite
    /// pins the binding on.
    fn seeded(n_nodes: usize, seed: u64) -> FlowNetwork {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut network = FlowNetwork::new(n_nodes);
        for _ in 0..4 * n_nodes {
            let tail = (next() % n_nodes as u64) as usize;
            let head = (next() % n_nodes as u64) as usize;
            let forward = 0.1 + 2.9 * (next() % 1000) as f64 / 1000.0;
            let back = (next() % 1000) as f64 / 1000.0;
            if tail != head {
                network.add_edge(tail, head, forward, back).unwrap();
            }
        }
        network
    }

    #[test]
    fn every_declined_kernel_reproduces_the_package_kernel_on_seeded_networks() {
        for n_nodes in [6usize, 12, 24, 96] {
            for seed in 1..=8u64 {
                let (expected, side) =
                    max_flow_impl(&mut seeded(n_nodes, seed), 0, n_nodes - 1).unwrap();
                for algorithm in [
                    Algorithm::Dinic,
                    Algorithm::PushRelabel,
                    Algorithm::ParallelPushRelabel,
                ] {
                    let (value, realized) =
                        max_flow_with(&mut seeded(n_nodes, seed), 0, n_nodes - 1, algorithm)
                            .unwrap();
                    assert!(
                        (value - expected).abs() <= 1e-9 * expected.abs().max(1.0),
                        "{algorithm:?} on n={n_nodes} seed={seed}: {value} vs {expected}"
                    );
                    assert_eq!(realized, side, "{algorithm:?} on n={n_nodes} seed={seed}");
                }
            }
        }
    }

    #[test]
    fn a_disconnected_sink_receives_nothing_from_every_kernel() {
        for algorithm in [
            Algorithm::Dinic,
            Algorithm::PushRelabel,
            Algorithm::ParallelPushRelabel,
        ] {
            let mut network = FlowNetwork::new(3);
            network.add_edge(0, 1, 4.0, 0.0).unwrap();
            let (value, side) = max_flow_with(&mut network, 0, 2, algorithm).unwrap();
            assert!(value.abs() < 1e-12);
            assert!(side[0] && side[1] && !side[2]);
        }
    }

    #[test]
    fn the_parallel_kernel_is_schedule_independent() {
        let n_nodes = 96;
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    max_flow_with(
                        &mut seeded(n_nodes, 3),
                        0,
                        n_nodes - 1,
                        Algorithm::ParallelPushRelabel,
                    )
                    .unwrap()
                })
        };
        let reference = run(1);
        for threads in [2usize, 4] {
            let realized = run(threads);
            assert_eq!(realized.0.to_bits(), reference.0.to_bits());
            assert_eq!(realized.1, reference.1);
        }
    }
}
