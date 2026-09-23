//! Boykov--Kolmogorov maximum flow laid out as the authors' implementation is (issue #986).
//!
//! The same algorithm as `maxflow::max_flow_impl`, with the layout of
//! Kolmogorov's `maxflow-v3.04`, which is what issue #986's profile of the
//! PyMaxflow goal pointed at:
//!
//! - **Terminal arcs live on the node.** `tr_cap` is the residual from the
//!   source (positive) or to the sink (negative), so the source and sink are
//!   not nodes of degree `n` whose adjacency every growth and adoption walks,
//!   and every node with a terminal residual starts in its tree, active.
//! - **Compressed adjacency.** Each node's arcs are one contiguous run of an
//!   arc array (`first[i]..first[i + 1]`), with the reverse arc stored as
//!   `sister`, in place of a `Vec<Vec<usize>>` of per-node heap allocations.
//! - **32-bit indices and array-of-structs records.** An arc is its residual,
//!   head and sister in 16 bytes; a node its terminal residual, parent, time
//!   stamp, distance and two flags in 24. The fields a step reads together sit
//!   on one cache line.
//! - **The active queue is a flag and a ring.** A node is queued once however
//!   often it is reactivated, and the node being grown keeps its flag while
//!   its adjacency is rescanned after an augmentation, as the reference does.
//!
//! Capacities stay `f64` and saturation is an exact zero, as in
//! `maxflow.rs`: the bottleneck subtracted from the arc that set it leaves
//! exactly `0.0`. On termination the source tree is exactly the set the
//! source reaches in the residual graph, the minimal minimum cut, so the cut
//! read off it is the one `max_flow_impl` and the Python Dinic return.

use std::collections::VecDeque;

use crate::maxflow::SATURATED;

/// No parent: the node is in neither tree.
const FREE: u32 = u32::MAX;
/// The parent is the node's terminal, through `tr_cap`.
const TERMINAL: u32 = u32::MAX - 1;
/// The parent arc saturated; the node awaits adoption.
const ORPHAN: u32 = u32::MAX - 2;

#[derive(Clone, Copy)]
#[repr(C)]
struct Node {
    /// Residual from the source when positive, to the sink when negative.
    tr_cap: f64,
    /// The arc from this node to its parent, or `FREE`, `TERMINAL`, `ORPHAN`.
    parent: u32,
    /// The time the distance to the terminal was last known exact.
    stamp: u32,
    /// Hops to the terminal as of `stamp`.
    distance: u32,
    is_sink: bool,
    active: bool,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct Arc {
    /// Residual from the arc's tail to its head.
    rcap: f64,
    head: u32,
    sister: u32,
}

/// A flow network with terminal residuals on the nodes, built once.
pub struct Graph {
    nodes: Vec<Node>,
    /// Node `i`'s arcs are `arcs[first[i] as usize..first[i + 1] as usize]`.
    first: Vec<u32>,
    arcs: Vec<Arc>,
    /// Flow already routed source to sink without touching an arc.
    flow: f64,
    /// The largest capacity built, arc or terminal: what the side's floor
    /// scales with.
    largest: f64,
}

impl Graph {
    /// A graph on `n_nodes` from undirected pairs with a capacity each way.
    ///
    /// `ends` is `2 * n_edges` flattened `(i, j)`, read as `int64` where
    /// NumPy wrote them rather than copied to `usize` (issue #986: the copy
    /// was 0.64 of the 2.6 MB a 142x142 cut peaked at); `forward[e]` is the
    /// capacity `i -> j` and `backward[e]` that of `j -> i`. A self-loop
    /// carries nothing and is dropped.
    pub fn from_edges(
        n_nodes: usize,
        ends: &[i64],
        forward: &[f64],
        backward: &[f64],
    ) -> Result<Self, String> {
        if n_nodes >= ORPHAN as usize {
            return Err(format!("{n_nodes} nodes exceed the 32-bit index range"));
        }
        if ends.len() != 2 * forward.len() || forward.len() != backward.len() {
            return Err(format!(
                "{} edge ends for {} forward and {} backward capacities",
                ends.len(),
                forward.len(),
                backward.len()
            ));
        }
        let mut degree = vec![0u32; n_nodes + 1];
        for (edge, pair) in ends.chunks_exact(2).enumerate() {
            if pair[0] < 0 || pair[1] < 0 {
                return Err(format!(
                    "node indices must be non-negative, got {}",
                    pair[0].min(pair[1])
                ));
            }
            let (i, j) = (pair[0] as usize, pair[1] as usize);
            if i >= n_nodes || j >= n_nodes {
                return Err(format!(
                    "edge ({i}, {j}) names a node outside [0, {n_nodes})"
                ));
            }
            if forward[edge] < 0.0 || backward[edge] < 0.0 {
                return Err(format!(
                    "capacities must be non-negative, got {} and {}",
                    forward[edge], backward[edge]
                ));
            }
            if i != j {
                degree[i + 1] += 1;
                degree[j + 1] += 1;
            }
        }
        for node in 0..n_nodes {
            degree[node + 1] += degree[node];
        }
        let first = degree;
        let n_arcs = first[n_nodes] as usize;
        if n_arcs >= ORPHAN as usize {
            return Err(format!("{n_arcs} arcs exceed the 32-bit index range"));
        }
        let mut fill: Vec<u32> = first[..n_nodes].to_vec();
        let mut arcs = vec![
            Arc {
                rcap: 0.0,
                head: 0,
                sister: 0
            };
            n_arcs
        ];
        for (edge, pair) in ends.chunks_exact(2).enumerate() {
            let (i, j) = (pair[0] as usize, pair[1] as usize);
            if i == j {
                continue;
            }
            let (a, b) = (fill[i], fill[j]);
            fill[i] += 1;
            fill[j] += 1;
            arcs[a as usize] = Arc {
                rcap: forward[edge],
                head: j as u32,
                sister: b,
            };
            arcs[b as usize] = Arc {
                rcap: backward[edge],
                head: i as u32,
                sister: a,
            };
        }
        let largest = forward
            .iter()
            .chain(backward)
            .fold(0.0_f64, |a, &c| a.max(c));
        let nodes = vec![
            Node {
                tr_cap: 0.0,
                parent: FREE,
                stamp: 0,
                distance: 0,
                is_sink: false,
                active: false,
            };
            n_nodes
        ];
        Ok(Self {
            nodes,
            first,
            arcs,
            flow: 0.0,
            largest,
        })
    }

    /// Add terminal capacities to `node`: `source` from the source, `sink`
    /// to the sink. The common part is routed at once, as the reference does.
    pub fn add_terminal(&mut self, node: usize, source: f64, sink: f64) -> Result<(), String> {
        if source < 0.0 || sink < 0.0 {
            return Err(format!(
                "capacities must be non-negative, got {source} and {sink}"
            ));
        }
        let n_nodes = self.nodes.len();
        let record = self
            .nodes
            .get_mut(node)
            .ok_or_else(|| format!("node {node} outside [0, {n_nodes})"))?;
        self.largest = self.largest.max(source).max(sink);
        let delta = record.tr_cap;
        let (source, sink) = if delta > 0.0 {
            (source + delta, sink)
        } else {
            (source, sink - delta)
        };
        self.flow += source.min(sink);
        record.tr_cap = source - sink;
        Ok(())
    }

    /// Add `amount` to the flow routed directly source to sink.
    pub fn add_direct(&mut self, amount: f64) {
        self.flow += amount;
    }

    /// Run to the maximum flow; returns its value and, per node, whether the
    /// source reaches it in the residual graph.
    ///
    /// The side is read by a breadth-first pass over residuals above
    /// `maxflow::SATURATED` times the largest capacity built, as
    /// `max_flow_impl` and `search.maxflow` read it (issue #935), rather
    /// than off the source tree, which is the same set at a floor of zero.
    pub fn solve(mut self) -> (f64, Vec<bool>) {
        let flow = Solver::new(&mut self).run();
        let floor = SATURATED * self.largest;
        let mut side: Vec<bool> = self.nodes.iter().map(|n| n.tr_cap > floor).collect();
        let mut queue: Vec<u32> = (0..self.nodes.len() as u32)
            .filter(|&n| side[n as usize])
            .collect();
        while let Some(node) = queue.pop() {
            let range = self.first[node as usize] as usize..self.first[node as usize + 1] as usize;
            for arc in &self.arcs[range] {
                if arc.rcap > floor && !side[arc.head as usize] {
                    side[arc.head as usize] = true;
                    queue.push(arc.head);
                }
            }
        }
        (self.flow + flow, side)
    }
}

struct Solver<'a> {
    nodes: &'a mut [Node],
    first: &'a [u32],
    arcs: &'a mut [Arc],
    queue: VecDeque<u32>,
    orphans: VecDeque<u32>,
    time: u32,
}

impl<'a> Solver<'a> {
    fn new(graph: &'a mut Graph) -> Self {
        Self {
            nodes: &mut graph.nodes,
            first: &graph.first,
            arcs: &mut graph.arcs,
            queue: VecDeque::new(),
            orphans: VecDeque::new(),
            time: 0,
        }
    }

    #[inline]
    fn set_active(&mut self, node: u32) {
        let record = &mut self.nodes[node as usize];
        if !record.active {
            record.active = true;
            self.queue.push_back(node);
        }
    }

    /// The next queued node still in a tree, its flag cleared.
    #[inline]
    fn next_active(&mut self) -> Option<u32> {
        while let Some(node) = self.queue.pop_front() {
            let record = &mut self.nodes[node as usize];
            record.active = false;
            if record.parent != FREE {
                return Some(node);
            }
        }
        None
    }

    #[inline]
    fn arcs_of(&self, node: u32) -> std::ops::Range<usize> {
        self.first[node as usize] as usize..self.first[node as usize + 1] as usize
    }

    fn run(&mut self) -> f64 {
        for index in 0..self.nodes.len() {
            let record = &mut self.nodes[index];
            if record.tr_cap != 0.0 {
                record.is_sink = record.tr_cap < 0.0;
                record.parent = TERMINAL;
                record.stamp = 0;
                record.distance = 1;
                self.set_active(index as u32);
            }
        }
        let mut flow = 0.0;
        let mut current: Option<u32> = None;
        loop {
            // The node grown last is kept, flagged, while it has a parent.
            let node = match current.take() {
                Some(node) if self.nodes[node as usize].parent != FREE => {
                    self.nodes[node as usize].active = false;
                    node
                }
                Some(node) => {
                    self.nodes[node as usize].active = false;
                    match self.next_active() {
                        Some(next) => next,
                        None => break,
                    }
                }
                None => match self.next_active() {
                    Some(next) => next,
                    None => break,
                },
            };
            let crossing = self.grow(node);
            self.time += 1;
            if let Some(arc) = crossing {
                self.nodes[node as usize].active = true;
                current = Some(node);
                flow += self.augment(arc);
                self.adopt();
            }
        }
        flow
    }

    /// Scan `node`'s arcs, growing its tree; the arc from the source tree to
    /// the sink tree if one is found.
    #[inline]
    fn grow(&mut self, node: u32) -> Option<u32> {
        let (is_sink, stamp, distance) = {
            let record = &self.nodes[node as usize];
            (record.is_sink, record.stamp, record.distance)
        };
        for arc in self.arcs_of(node) {
            let Arc { rcap, head, sister } = self.arcs[arc];
            let residual = if is_sink {
                self.arcs[sister as usize].rcap
            } else {
                rcap
            };
            if residual <= 0.0 {
                continue;
            }
            let neighbour = &mut self.nodes[head as usize];
            if neighbour.parent == FREE {
                neighbour.is_sink = is_sink;
                neighbour.parent = sister;
                neighbour.stamp = stamp;
                neighbour.distance = distance + 1;
                self.set_active(head);
            } else if neighbour.is_sink != is_sink {
                return Some(if is_sink { sister } else { arc as u32 });
            }
            // The reference also re-parents a same-tree neighbour onto a
            // shorter path here; measured without it (issue #986) the 142x142
            // cut solves 6% faster and 284x284 7%, and adoption still takes
            // the nearest parent.
        }
        None
    }

    #[inline]
    fn orphan_front(&mut self, node: u32) {
        self.nodes[node as usize].parent = ORPHAN;
        self.orphans.push_front(node);
    }

    /// Push the bottleneck along the path through `middle`, the arc from the
    /// source tree to the sink tree; saturated arcs orphan their child.
    fn augment(&mut self, middle: u32) -> f64 {
        let middle_arc = self.arcs[middle as usize];
        let mut bottleneck = middle_arc.rcap;
        // The source side, from the tail of `middle` up to the source.
        let mut node = self.arcs[middle_arc.sister as usize].head;
        loop {
            let parent = self.nodes[node as usize].parent;
            if parent == TERMINAL {
                break;
            }
            let arc = self.arcs[parent as usize];
            bottleneck = bottleneck.min(self.arcs[arc.sister as usize].rcap);
            node = arc.head;
        }
        bottleneck = bottleneck.min(self.nodes[node as usize].tr_cap);
        let source_root = node;
        // The sink side, from the head of `middle` down to the sink.
        node = middle_arc.head;
        loop {
            let parent = self.nodes[node as usize].parent;
            if parent == TERMINAL {
                break;
            }
            let arc = self.arcs[parent as usize];
            bottleneck = bottleneck.min(arc.rcap);
            node = arc.head;
        }
        bottleneck = bottleneck.min(-self.nodes[node as usize].tr_cap);
        let sink_root = node;

        self.arcs[middle_arc.sister as usize].rcap += bottleneck;
        self.arcs[middle as usize].rcap -= bottleneck;
        node = self.arcs[middle_arc.sister as usize].head;
        while node != source_root {
            let parent = self.nodes[node as usize].parent;
            let arc = self.arcs[parent as usize];
            self.arcs[parent as usize].rcap += bottleneck;
            let back = &mut self.arcs[arc.sister as usize];
            back.rcap -= bottleneck;
            if back.rcap == 0.0 {
                self.orphan_front(node);
            }
            node = arc.head;
        }
        self.nodes[source_root as usize].tr_cap -= bottleneck;
        if self.nodes[source_root as usize].tr_cap == 0.0 {
            self.orphan_front(source_root);
        }
        node = middle_arc.head;
        while node != sink_root {
            let parent = self.nodes[node as usize].parent;
            let arc = self.arcs[parent as usize];
            self.arcs[arc.sister as usize].rcap += bottleneck;
            let forward = &mut self.arcs[parent as usize];
            forward.rcap -= bottleneck;
            if forward.rcap == 0.0 {
                self.orphan_front(node);
            }
            node = arc.head;
        }
        self.nodes[sink_root as usize].tr_cap += bottleneck;
        if self.nodes[sink_root as usize].tr_cap == 0.0 {
            self.orphan_front(sink_root);
        }
        bottleneck
    }

    fn adopt(&mut self) {
        while let Some(orphan) = self.orphans.pop_front() {
            self.process_orphan(orphan);
        }
    }

    /// Find `orphan` a parent in its own tree whose path reaches the
    /// terminal, the nearest; failing that, free it and orphan its children.
    fn process_orphan(&mut self, orphan: u32) {
        let is_sink = self.nodes[orphan as usize].is_sink;
        let time = self.time;
        let mut best_arc = FREE;
        let mut best_distance = u32::MAX;
        for arc in self.arcs_of(orphan) {
            let Arc { rcap, head, sister } = self.arcs[arc];
            // Residual from the candidate parent toward the orphan in the
            // source tree, from the orphan toward it in the sink tree.
            let residual = if is_sink {
                rcap
            } else {
                self.arcs[sister as usize].rcap
            };
            if residual <= 0.0 {
                continue;
            }
            let candidate = &self.nodes[head as usize];
            if candidate.is_sink != is_sink || candidate.parent == FREE {
                continue;
            }
            // Walk to the terminal or to a node stamped this time.
            let mut walk = head;
            let mut hops = 0u32;
            let reached = loop {
                let record = &self.nodes[walk as usize];
                if record.stamp == time {
                    break Some(hops + record.distance);
                }
                match record.parent {
                    TERMINAL => {
                        let record = &mut self.nodes[walk as usize];
                        record.stamp = time;
                        record.distance = 1;
                        break Some(hops + 1);
                    }
                    ORPHAN | FREE => break None,
                    parent => {
                        walk = self.arcs[parent as usize].head;
                        hops += 1;
                    }
                }
            };
            let Some(mut distance) = reached else {
                continue;
            };
            if distance < best_distance {
                best_arc = arc as u32;
                best_distance = distance;
            }
            // Stamp the path so the next check through it is O(1).
            let mut walk = head;
            while self.nodes[walk as usize].stamp != time {
                let record = &mut self.nodes[walk as usize];
                record.stamp = time;
                record.distance = distance;
                distance -= 1;
                walk = self.arcs[record.parent as usize].head;
            }
        }
        if best_arc != FREE {
            let record = &mut self.nodes[orphan as usize];
            record.parent = best_arc;
            record.stamp = time;
            record.distance = best_distance + 1;
            return;
        }
        // No parent: the orphan leaves its tree.
        self.nodes[orphan as usize].parent = FREE;
        for arc in self.arcs_of(orphan) {
            let Arc { rcap, head, sister } = self.arcs[arc];
            let neighbour = self.nodes[head as usize];
            if neighbour.is_sink != is_sink || neighbour.parent == FREE {
                continue;
            }
            let residual = if is_sink {
                rcap
            } else {
                self.arcs[sister as usize].rcap
            };
            if residual > 0.0 {
                self.set_active(head);
            }
            let parent = neighbour.parent;
            if parent != TERMINAL && parent != ORPHAN && self.arcs[parent as usize].head == orphan {
                self.nodes[head as usize].parent = ORPHAN;
                self.orphans.push_back(head);
            }
        }
    }
}

/// The two-state ferromagnet's ground state, as `maxflow::ising_ground_state_impl` states it.
pub fn ising_ground_state(
    n_nodes: usize,
    field: &[f64],
    edges: &[i64],
    coupling: &[f64],
) -> Result<Vec<i64>, String> {
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
    let mut graph = Graph::from_edges(n_nodes, edges, coupling, coupling)?;
    for node in 0..n_nodes {
        let cost_zero = -field[2 * node];
        let cost_one = -field[2 * node + 1];
        let offset = cost_zero.min(cost_one);
        graph.add_terminal(node, cost_one - offset, cost_zero - offset)?;
    }
    let (_, side) = graph.solve();
    Ok(side
        .iter()
        .map(|&reachable| i64::from(!reachable))
        .collect())
}

/// Maximum flow on a network with explicit terminal nodes, as `maxflow::max_flow` takes it.
///
/// Arcs touching `source` or `sink` become terminal residuals; the rest
/// are the graph. Returns the value and the source side, `source` on it and
/// `sink` off it.
pub fn max_flow(
    n_nodes: usize,
    arcs: &[i64],
    capacity: &[f64],
    reverse: Option<&[f64]>,
    source: usize,
    sink: usize,
) -> Result<(f64, Vec<bool>), String> {
    if source == sink {
        return Err(format!("source and sink must differ, both are {source}"));
    }
    if source >= n_nodes || sink >= n_nodes {
        return Err(format!(
            "terminals ({source}, {sink}) must lie in [0, {n_nodes})"
        ));
    }
    let back = |edge: usize| reverse.map_or(0.0, |r| r[edge]);
    let mut from_source = vec![0.0; n_nodes];
    let mut to_sink = vec![0.0; n_nodes];
    let mut direct = 0.0;
    let mut ends = Vec::with_capacity(arcs.len());
    let mut forward = Vec::with_capacity(capacity.len());
    let mut backward = Vec::with_capacity(capacity.len());
    if arcs.len() != 2 * capacity.len() {
        return Err(format!(
            "arcs has {} entries for {} capacities",
            arcs.len(),
            capacity.len()
        ));
    }
    for (edge, pair) in arcs.chunks_exact(2).enumerate() {
        if pair[0] < 0 || pair[1] < 0 {
            return Err(format!(
                "node indices must be non-negative, got {}",
                pair[0].min(pair[1])
            ));
        }
        let (i, j) = (pair[0] as usize, pair[1] as usize);
        let (c, r) = (capacity[edge], back(edge));
        if i >= n_nodes || j >= n_nodes {
            return Err(format!(
                "edge ({i}, {j}) names a node outside [0, {n_nodes})"
            ));
        }
        if c < 0.0 || r < 0.0 {
            return Err(format!("capacities must be non-negative, got {c} and {r}"));
        }
        match (i == source || i == sink, j == source || j == sink) {
            (false, false) => {
                ends.extend_from_slice(&[pair[0], pair[1]]);
                forward.push(c);
                backward.push(r);
            }
            (true, true) => {
                if i == source && j == sink {
                    direct += c;
                } else if i == sink && j == source {
                    direct += r;
                }
            }
            (true, false) => {
                if i == source {
                    from_source[j] += c;
                } else {
                    to_sink[j] += r;
                }
            }
            (false, true) => {
                if j == sink {
                    to_sink[i] += c;
                } else {
                    from_source[i] += r;
                }
            }
        }
    }
    let mut graph = Graph::from_edges(n_nodes, &ends, &forward, &backward)?;
    for node in 0..n_nodes {
        if node != source && node != sink && (from_source[node] > 0.0 || to_sink[node] > 0.0) {
            graph.add_terminal(node, from_source[node], to_sink[node])?;
        }
    }
    graph.add_direct(direct);
    let (value, mut side) = graph.solve();
    side[source] = true;
    side[sink] = false;
    Ok((value, side))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::maxflow::{max_flow_impl, FlowNetwork};

    /// Edge ends, forward and backward capacities, source and sink capacities.
    type Case = (Vec<i64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

    /// A seeded random network: every node has terminal capacities and a
    /// handful of arcs, so both trees and adoption are exercised.
    fn random_case(n_nodes: usize, seed: u64) -> Case {
        let mut state = seed;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as f64 / (1u64 << 31) as f64
        };
        let mut ends = Vec::new();
        let (mut forward, mut backward) = (Vec::new(), Vec::new());
        for i in 0..n_nodes {
            for _ in 0..3 {
                let j = (next() * n_nodes as f64) as usize % n_nodes;
                ends.extend_from_slice(&[i as i64, j as i64]);
                forward.push((next() * 4.0).floor());
                backward.push((next() * 4.0).floor());
            }
        }
        let source: Vec<f64> = (0..n_nodes).map(|_| (next() * 6.0).floor()).collect();
        let sink: Vec<f64> = (0..n_nodes).map(|_| (next() * 6.0).floor()).collect();
        (ends, forward, backward, source, sink)
    }

    #[test]
    fn the_flow_and_cut_are_the_explicit_terminal_kernels() {
        for seed in 0..40 {
            let n = 30 + (seed as usize % 7) * 11;
            let (ends, forward, backward, from_source, to_sink) = random_case(n, seed);
            let mut graph = Graph::from_edges(n, &ends, &forward, &backward).unwrap();
            let mut reference = FlowNetwork::new(n + 2);
            for (edge, pair) in ends.chunks_exact(2).enumerate() {
                if pair[0] != pair[1] {
                    reference
                        .add_edge(
                            pair[0] as usize,
                            pair[1] as usize,
                            forward[edge],
                            backward[edge],
                        )
                        .unwrap();
                }
            }
            for node in 0..n {
                graph
                    .add_terminal(node, from_source[node], to_sink[node])
                    .unwrap();
                reference.add_edge(n, node, from_source[node], 0.0).unwrap();
                reference.add_edge(node, n + 1, to_sink[node], 0.0).unwrap();
            }
            let (value, side) = graph.solve();
            let (expected, expected_side) = max_flow_impl(&mut reference, n, n + 1).unwrap();
            assert_eq!(value, expected, "seed {seed}");
            assert_eq!(side, expected_side[..n].to_vec(), "seed {seed}");
        }
    }

    #[test]
    fn explicit_terminals_fold_into_node_residuals() {
        // 0 -> 1 -> 3 and 0 -> 2 -> 3 with a direct 0 -> 3 arc.
        let arcs = [0, 1, 1, 3, 0, 2, 2, 3, 0, 3];
        let capacity = [3.0, 2.0, 1.0, 5.0, 4.0];
        let (value, side) = max_flow(4, &arcs, &capacity, None, 0, 3).unwrap();
        assert_eq!(value, 2.0 + 1.0 + 4.0);
        assert_eq!(side, vec![true, true, false, false]);
    }

    #[test]
    fn coincident_terminals_are_refused() {
        assert!(max_flow(2, &[0, 1], &[1.0], None, 1, 1).is_err());
    }

    #[test]
    fn a_negative_capacity_is_refused() {
        assert!(Graph::from_edges(2, &[0, 1], &[-1.0], &[0.0]).is_err());
    }
}
