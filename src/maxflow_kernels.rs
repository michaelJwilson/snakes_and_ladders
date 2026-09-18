//! Three maximum-flow kernels beside Dinic, behind `maxflow::Algorithm` (issue #715).
//!
//! Each takes the same [`FlowNetwork`] of paired residual arcs, returns the
//! flow value, and leaves the residual graph in the network for
//! `maxflow::max_flow_with` to read the source side off, so every kernel
//! certifies its cut the same way and the Python Dinic pins all of them.
//!
//! * [`push_relabel`]: Goldberg--Tarjan (1988) with highest-label selection
//!   (Cheriyan--Maheshwari 1989) over buckets, a global relabel by reverse
//!   breadth-first search from the sink, and the gap heuristic. Both phases
//!   run --- excess is returned to the source --- so what is left is a flow
//!   and not a preflow, and its residual graph is a maximum flow's.
//! * [`boykov_kolmogorov`]: two search trees grown from the terminals with
//!   orphan adoption after each augmentation (Boykov--Kolmogorov 2004), the
//!   origin check cached by timestamp and distance as their implementation
//!   does.
//! * [`parallel_push_relabel`]: the synchronous variant (Baumstark, Blelloch
//!   and Shun 2015). A round computes every active vertex's pushes in parallel
//!   from the round's opening state and applies them in vertex order, then
//!   relabels in parallel from the state that results; the arithmetic is
//!   therefore a fixed sequence whatever the thread count, and the output is
//!   identical at one thread and at eight.
//!
//! Capacities are `f64` and saturation is `> 0.0`, as in `maxflow.rs`, so no
//! kernel here can disagree with the reference about which arcs are cut.

use std::collections::VecDeque;

use rayon::prelude::*;

use crate::maxflow::FlowNetwork;

const NONE: usize = usize::MAX;

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
pub fn boykov_kolmogorov(network: &mut FlowNetwork, source: usize, sink: usize) -> f64 {
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
            let mut fresh = sink_distances(network, sink, n);
            fresh[source] = n;
            // Above `n`, keep the running label: those nodes route back to
            // the source and the reverse search does not see them.
            for node in 0..n {
                if fresh[node] >= n && node != source {
                    fresh[node] = label[node].max(n);
                }
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::maxflow::{max_flow_impl, max_flow_with, Algorithm};

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
    fn every_kernel_reproduces_dinic_on_seeded_networks() {
        for n_nodes in [6usize, 12, 24, 96] {
            for seed in 1..=8u64 {
                let (expected, side) =
                    max_flow_impl(&mut seeded(n_nodes, seed), 0, n_nodes - 1).unwrap();
                for algorithm in [
                    Algorithm::PushRelabel,
                    Algorithm::BoykovKolmogorov,
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
            Algorithm::PushRelabel,
            Algorithm::BoykovKolmogorov,
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
        let reference = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
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
            });
        for threads in [2usize, 4] {
            let realized = rayon::ThreadPoolBuilder::new()
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
                });
            assert_eq!(realized.0.to_bits(), reference.0.to_bits());
            assert_eq!(realized.1, reference.1);
        }
    }
}
