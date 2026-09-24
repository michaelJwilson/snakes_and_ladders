//! One lattice's flow network, built once and refilled per cut move (issue #935).
//!
//! `maxflow.rs` builds a network per call: arcs to and from two terminal
//! nodes, and for the expansion an auxiliary node on every edge whose ends
//! disagree, so the layout changes with the labelling and cannot be kept.
//! This keeps the lattice's arcs in compressed rows for the life of a
//! [`LatticeCut`] and each node's terminal capacity as one signed number, the
//! representation of Boykov & Kolmogorov's own implementation, so a move
//! rewrites capacities and allocates nothing.
//!
//! **The expansion needs no auxiliary node here.** Its pairwise term over the
//! binary choice `x` (1: take `alpha`) is `E(0,0) = V(l_a, l_b)`,
//! `E(0,1) = V(l_a, alpha)`, `E(1,0) = V(alpha, l_b)`, `E(1,1) = 0`, and
//! Kolmogorov & Zabih (2004) write any submodular such term as
//! `E(0,0) + (E(1,0) - E(0,0)) x_a + (E(1,1) - E(1,0)) x_b` plus one arc
//! `a -> b` of capacity `E(0,1) + E(1,0) - E(0,0) - E(1,1)`, non-negative
//! for a metric. Two networks encoding the same energy have the same minimal
//! minimum cut, so the labelling is the one `maxflow::expansion_network`'s
//! cut gives, up to rounding in the reordered sums; the Python network stays
//! as the oracle.
//!
//! The side is read at `maxflow::SATURATED` times the largest capacity
//! filled, as `max_flow_impl` reads it.

use std::collections::VecDeque;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::maxflow::{node_indices, SATURATED};

/// A node's parent is its terminal.
const TERMINAL: u32 = u32::MAX - 1;
/// A node without a parent: free, or orphaned mid-adoption.
const NONE: u32 = u32::MAX;

/// The lattice's residual network and the search trees' buffers, reused.
///
/// Arc `a` runs from the row it is stored in to `head[a]`, and `sister[a]` is
/// the reverse arc. `terminal[n]`
/// is node `n`'s residual from the source when positive and to the sink,
/// negated, when negative.
#[pyclass(module = "snakes_and_ladders.oxi_snakes_and_ladders")]
pub struct LatticeCut {
    n_nodes: usize,
    start: Vec<u32>,
    head: Vec<u32>,
    sister: Vec<u32>,
    first: Vec<u32>,
    second: Vec<u32>,
    coupling: Vec<f64>,
    residual: Vec<f64>,
    terminal: Vec<f64>,
    // Search trees (Boykov & Kolmogorov 2004).
    /// The arc from a node to its parent, [`TERMINAL`] or [`NONE`].
    parent: Vec<u32>,
    sink_tree: Vec<bool>,
    active: VecDeque<u32>,
    queued: Vec<bool>,
    orphans: VecDeque<u32>,
    stamp: Vec<u64>,
    distance: Vec<u32>,
    time: u64,
    /// The arc index `first -> second` of each edge, and of its reverse.
    arc_of_edge: Vec<[u32; 2]>,
    /// The capacities the last fill wrote, before any flow is placed.
    capacity: Vec<f64>,
    /// The maximum flow each move key ended on, to start its next cut from.
    kept: Vec<Option<Kept>>,
    /// Each arc's edge, so a swap walks a moving node's own arcs.
    edge_of_arc: Vec<u32>,
    /// A node's index in the swap's compact network, `NONE` if not in it.
    local: Vec<u32>,
}

/// A maximum flow as it stood: per edge the flow `first -> second`.
///
/// The terminals' flows are not held (issue #986): ten labels' edge and
/// terminal flows were 4.8 of the 10.6 MB a 142x142 expansion peaked at,
/// and the terminals' third of that is implied by the edges'. [`LatticeCut::restore`] derives each
/// node's terminal flow from the edge flows it places, so the start it
/// builds conserves flow at every node whatever the rounding; a start is a
/// start, and every maximum flow reached from it leaves the same minimal
/// minimum cut.
struct Kept {
    flow: Vec<f64>,
}

impl LatticeCut {
    /// The compressed rows of the lattice `first[e] -- second[e]`.
    pub fn build(
        n_nodes: usize,
        first: &[usize],
        second: &[usize],
        coupling: &[f64],
    ) -> Result<Self, String> {
        if first.len() != second.len() || first.len() != coupling.len() {
            return Err("first, second and coupling must have one entry per edge".into());
        }
        let n_arcs = 2 * first.len();
        if n_nodes >= TERMINAL as usize || n_arcs >= TERMINAL as usize {
            return Err(format!(
                "{n_nodes} nodes and {n_arcs} arcs exceed u32 indexing"
            ));
        }
        if let Some(&node) = first.iter().chain(second).find(|&&node| node >= n_nodes) {
            return Err(format!("edge names node {node} outside [0, {n_nodes})"));
        }
        if let Some(negative) = coupling.iter().find(|&&j| j < 0.0) {
            return Err(format!(
                "every coupling must be non-negative, got {negative}"
            ));
        }
        let mut start = vec![0_u32; n_nodes + 1];
        for (&a, &b) in first.iter().zip(second) {
            start[a + 1] += 1;
            start[b + 1] += 1;
        }
        for node in 0..n_nodes {
            start[node + 1] += start[node];
        }
        let mut next = start.clone();
        let mut head = vec![0_u32; n_arcs];
        let mut sister = vec![0_u32; n_arcs];
        let mut arc_of_edge = vec![[0_u32; 2]; first.len()];
        let mut edge_of_arc = vec![0_u32; n_arcs];
        for (position, (&a, &b)) in first.iter().zip(second).enumerate() {
            let (out, back) = (next[a], next[b]);
            next[a] += 1;
            next[b] += 1;
            head[out as usize] = b as u32;
            head[back as usize] = a as u32;
            sister[out as usize] = back;
            sister[back as usize] = out;
            arc_of_edge[position] = [out, back];
            edge_of_arc[out as usize] = position as u32;
            edge_of_arc[back as usize] = position as u32;
        }
        Ok(Self {
            n_nodes,
            start,
            head,
            sister,
            first: first.iter().map(|&n| n as u32).collect(),
            second: second.iter().map(|&n| n as u32).collect(),
            coupling: coupling.to_vec(),
            residual: vec![0.0; n_arcs],
            terminal: vec![0.0; n_nodes],
            parent: vec![NONE; n_nodes],
            sink_tree: vec![false; n_nodes],
            active: VecDeque::new(),
            queued: vec![false; n_nodes],
            orphans: VecDeque::new(),
            stamp: vec![0; n_nodes],
            distance: vec![0; n_nodes],
            time: 0,
            arc_of_edge,
            capacity: vec![0.0; n_arcs],
            kept: Vec::new(),
            edge_of_arc,
            local: vec![NONE; n_nodes],
        })
    }

    /// One alpha-beta swap's source side, cut on the moving nodes alone.
    ///
    /// A node at neither label carries no capacity, and an edge with an end
    /// at neither carries none, so the swap's network is the subgraph on the
    /// nodes at `alpha` or `beta` and the edges between them: built here from
    /// the moving nodes' own arcs and cut by `bk::Graph`, where
    /// `fill_swap` and `solve` walked the whole lattice five times per move
    /// (issue #997). The side is read at the same floor, so it is the same
    /// minimal minimum cut.
    pub fn swap_side(
        &mut self,
        values: &[f64],
        n_states: usize,
        labels: &[usize],
        alpha: usize,
        beta: usize,
    ) -> Vec<bool> {
        let moving: Vec<u32> = (0..self.n_nodes as u32)
            .filter(|&n| {
                let label = labels[n as usize];
                label == alpha || label == beta
            })
            .collect();
        for (index, &node) in moving.iter().enumerate() {
            self.local[node as usize] = index as u32;
        }
        let (mut ends, mut capacity) = (Vec::new(), Vec::new());
        for &node in &moving {
            let u = node as usize;
            for arc in self.start[u] as usize..self.start[u + 1] as usize {
                let v = self.head[arc] as usize;
                if v > u && self.local[v] != NONE {
                    ends.push(i64::from(self.local[u]));
                    ends.push(i64::from(self.local[v]));
                    capacity.push(self.coupling[self.edge_of_arc[arc] as usize]);
                }
            }
        }
        let mut graph = crate::bk::Graph::from_edges(moving.len(), &ends, &capacity, &capacity)
            .expect("the moving subgraph's indices are in range");
        for (index, &node) in moving.iter().enumerate() {
            let u = node as usize;
            let net = values[u * n_states + alpha] - values[u * n_states + beta];
            graph
                .add_terminal(index, net.max(0.0), (-net).max(0.0))
                .expect("terminal capacities are non-negative");
        }
        let (_, local_side) = graph.solve();
        let mut side = vec![false; self.n_nodes];
        for (index, &node) in moving.iter().enumerate() {
            side[node as usize] = local_side[index];
            self.local[node as usize] = NONE;
        }
        side
    }

    /// Fill one alpha-expansion move: the source side keeps its label, the
    /// sink side takes `alpha`, and a node already at `alpha` is held on the
    /// sink side by a keep cost of `pinned`.
    pub fn fill_expansion(
        &mut self,
        values: &[f64],
        n_states: usize,
        labels: &[usize],
        alpha: usize,
        pinned: f64,
    ) {
        for node in 0..self.n_nodes {
            let label = labels[node];
            let switch = -values[node * n_states + alpha];
            let keep = if label == alpha {
                pinned
            } else {
                -values[node * n_states + label]
            };
            self.terminal[node] = switch - keep;
        }
        for position in 0..self.first.len() {
            let (a, b, weight) = (
                self.first[position] as usize,
                self.second[position] as usize,
                self.coupling[position],
            );
            let [out, back] = self.arc_of_edge[position];
            let (la, lb) = (labels[a], labels[b]);
            if la == lb {
                let differs = if la == alpha { 0.0 } else { weight };
                self.residual[out as usize] = differs;
                self.residual[back as usize] = differs;
                continue;
            }
            // E(0,0) = weight, E(0,1) = V(l_a, alpha), E(1,0) = V(alpha, l_b).
            let keep_switch = if la == alpha { 0.0 } else { weight };
            let switch_keep = if lb == alpha { 0.0 } else { weight };
            self.terminal[a] += switch_keep - weight;
            self.terminal[b] -= switch_keep;
            self.residual[out as usize] = keep_switch + switch_keep - weight;
            self.residual[back as usize] = 0.0;
        }
    }

    /// Fill one alpha-beta swap: only nodes at `alpha` or `beta` take part,
    /// the source side takes `alpha` and the sink side `beta`; every other
    /// node is left with no capacity and so off the source side.
    pub fn fill_swap(
        &mut self,
        values: &[f64],
        n_states: usize,
        labels: &[usize],
        alpha: usize,
        beta: usize,
    ) {
        for node in 0..self.n_nodes {
            let label = labels[node];
            self.terminal[node] = if label == alpha || label == beta {
                -values[node * n_states + beta] + values[node * n_states + alpha]
            } else {
                0.0
            };
        }
        for position in 0..self.first.len() {
            let (a, b) = (
                labels[self.first[position] as usize],
                labels[self.second[position] as usize],
            );
            let moving = |l: usize| l == alpha || l == beta;
            let capacity = if moving(a) && moving(b) {
                self.coupling[position]
            } else {
                0.0
            };
            let [out, back] = self.arc_of_edge[position];
            self.residual[out as usize] = capacity;
            self.residual[back as usize] = capacity;
        }
    }

    /// Maximum flow over the filled capacities, and the source side of the
    /// minimal minimum cut.
    ///
    /// With a `key`, the flow starts from the one the same key's last cut
    /// ended on rather than from zero (Kohli & Torr 2007; Alahari, Kohli &
    /// Torr 2008 reuse it across expansion cycles for the same label). Every
    /// maximum flow leaves the same minimal minimum cut, so the start changes
    /// the work and not the side, up to rounding in the flow's sums.
    pub fn solve(&mut self, key: Option<usize>) -> Vec<bool> {
        let largest = self
            .residual
            .iter()
            .chain(&self.terminal)
            .fold(0.0_f64, |m, &c| m.max(c.abs()));
        self.capacity.copy_from_slice(&self.residual);
        if let Some(key) = key {
            self.restore(key);
            // A terminal flow summed from the edges' can miss the exact
            // net by rounding; below the side's floor that residue is not a
            // capacity, and left in place it roots a tree of tiny
            // augmentations (issue #986).
            let floor = SATURATED * largest;
            for terminal in &mut self.terminal {
                if terminal.abs() <= floor {
                    *terminal = 0.0;
                }
            }
        }
        self.max_flow();
        let side = self.source_side(SATURATED * largest);
        if let Some(key) = key {
            self.keep(key);
        }
        side
    }

    /// Place `key`'s kept flow on the new capacities. An edge whose flow now
    /// exceeds its capacity is clamped, and the excess is returned through
    /// the two ends' terminals: a terminal's net flow is unbounded in either
    /// sign, so the result is a feasible flow on the new network.
    fn restore(&mut self, key: usize) {
        let Some(kept) = self.kept.get(key).and_then(Option::as_ref) else {
            return;
        };
        for (position, &[out, back]) in self.arc_of_edge.iter().enumerate() {
            let (out, back) = (out as usize, back as usize);
            let placed = kept.flow[position].clamp(-self.residual[back], self.residual[out]);
            // The flow leaving `first` along the edge arrives from its
            // terminal, and the flow reaching `second` leaves through its.
            self.terminal[self.first[position] as usize] -= placed;
            self.terminal[self.second[position] as usize] += placed;
            self.residual[out] -= placed;
            self.residual[back] += placed;
        }
    }

    /// Record the flow the cut ended on under `key`.
    fn keep(&mut self, key: usize) {
        if self.kept.len() <= key {
            self.kept.resize_with(key + 1, || None);
        }
        let kept = self.kept[key].get_or_insert_with(|| Kept {
            flow: vec![0.0; self.first.len()],
        });
        for (position, &[out, _]) in self.arc_of_edge.iter().enumerate() {
            kept.flow[position] = self.capacity[out as usize] - self.residual[out as usize];
        }
    }

    fn activate(&mut self, node: u32) {
        if !self.queued[node as usize] {
            self.queued[node as usize] = true;
            self.active.push_back(node);
        }
    }

    fn max_flow(&mut self) {
        self.active.clear();
        self.orphans.clear();
        self.time = 0;
        for node in 0..self.n_nodes {
            self.stamp[node] = 0;
            self.queued[node] = false;
            let capacity = self.terminal[node];
            if capacity == 0.0 {
                self.parent[node] = NONE;
                continue;
            }
            self.parent[node] = TERMINAL;
            self.sink_tree[node] = capacity < 0.0;
            self.distance[node] = 1;
            self.activate(node as u32);
        }
        while let Some(crossing) = self.grow() {
            self.time += 1;
            self.augment(crossing);
            self.adopt();
        }
    }

    /// Grow both trees from the active nodes until an arc joins them; the
    /// arc returned runs from the source tree into the sink tree.
    fn grow(&mut self) -> Option<u32> {
        while let Some(&node) = self.active.front() {
            let i = node as usize;
            if self.parent[i] == NONE {
                self.active.pop_front();
                self.queued[i] = false;
                continue;
            }
            let sink = self.sink_tree[i];
            for arc in self.start[i]..self.start[i + 1] {
                let arc = arc as usize;
                // Residual in the direction the tree's flow travels.
                let travel = if sink { self.sister[arc] as usize } else { arc };
                if self.residual[travel] <= 0.0 {
                    continue;
                }
                let j = self.head[arc] as usize;
                if self.parent[j] == NONE {
                    self.sink_tree[j] = sink;
                    self.parent[j] = self.sister[arc];
                    self.stamp[j] = self.stamp[i];
                    self.distance[j] = self.distance[i] + 1;
                    self.activate(j as u32);
                } else if self.sink_tree[j] != sink {
                    return Some(travel as u32);
                }
            }
            self.active.pop_front();
            self.queued[i] = false;
        }
        None
    }

    /// Push the bottleneck along the path through `crossing`; an arc or a
    /// terminal it saturates orphans the node below it.
    fn augment(&mut self, crossing: u32) {
        let crossing = crossing as usize;
        let from = self.head[self.sister[crossing] as usize] as usize;
        let to = self.head[crossing] as usize;
        let mut bottleneck = self.residual[crossing];
        let mut node = from;
        while self.parent[node] != TERMINAL {
            let arc = self.parent[node] as usize;
            bottleneck = bottleneck.min(self.residual[self.sister[arc] as usize]);
            node = self.head[arc] as usize;
        }
        bottleneck = bottleneck.min(self.terminal[node]);
        node = to;
        while self.parent[node] != TERMINAL {
            let arc = self.parent[node] as usize;
            bottleneck = bottleneck.min(self.residual[arc]);
            node = self.head[arc] as usize;
        }
        bottleneck = bottleneck.min(-self.terminal[node]);

        self.residual[crossing] -= bottleneck;
        self.residual[self.sister[crossing] as usize] += bottleneck;
        node = from;
        while self.parent[node] != TERMINAL {
            let arc = self.parent[node] as usize;
            let down = self.sister[arc] as usize;
            self.residual[down] -= bottleneck;
            self.residual[arc] += bottleneck;
            let next = self.head[arc] as usize;
            if self.residual[down] <= 0.0 {
                self.parent[node] = NONE;
                self.orphans.push_back(node as u32);
            }
            node = next;
        }
        self.terminal[node] -= bottleneck;
        if self.terminal[node] <= 0.0 {
            self.parent[node] = NONE;
            self.orphans.push_back(node as u32);
        }
        node = to;
        while self.parent[node] != TERMINAL {
            let arc = self.parent[node] as usize;
            self.residual[arc] -= bottleneck;
            self.residual[self.sister[arc] as usize] += bottleneck;
            let next = self.head[arc] as usize;
            if self.residual[arc] <= 0.0 {
                self.parent[node] = NONE;
                self.orphans.push_back(node as u32);
            }
            node = next;
        }
        self.terminal[node] += bottleneck;
        if self.terminal[node] >= 0.0 {
            self.parent[node] = NONE;
            self.orphans.push_back(node as u32);
        }
    }

    /// The distance from `node` to its terminal through parents, or `None`
    /// where the walk meets an orphan; the path is stamped on success.
    fn origin(&mut self, node: usize) -> Option<u32> {
        let mut walk = node;
        let mut hops = 0_u32;
        let base = loop {
            if self.stamp[walk] == self.time {
                break self.distance[walk];
            }
            let arc = self.parent[walk];
            if arc == TERMINAL {
                self.stamp[walk] = self.time;
                self.distance[walk] = 1;
                break 1;
            }
            if arc == NONE {
                return None;
            }
            walk = self.head[arc as usize] as usize;
            hops += 1;
        };
        let mut walk = node;
        let mut remaining = hops;
        while remaining > 0 {
            self.stamp[walk] = self.time;
            self.distance[walk] = base + remaining;
            walk = self.head[self.parent[walk] as usize] as usize;
            remaining -= 1;
        }
        Some(base + hops)
    }

    fn adopt(&mut self) {
        while let Some(orphan) = self.orphans.pop_front() {
            let i = orphan as usize;
            let sink = self.sink_tree[i];
            // A terminal with residual left is a parent at distance one.
            let mut best = NONE;
            let mut best_distance = u32::MAX;
            if (sink && self.terminal[i] < 0.0) || (!sink && self.terminal[i] > 0.0) {
                best = TERMINAL;
                best_distance = 1;
            }
            if best == NONE {
                for arc in self.start[i]..self.start[i + 1] {
                    let arc = arc as usize;
                    let j = self.head[arc] as usize;
                    if self.parent[j] == NONE || self.sink_tree[j] != sink {
                        continue;
                    }
                    // The parent sends flow to the orphan in the source tree
                    // and receives it in the sink tree.
                    let travel = if sink { arc } else { self.sister[arc] as usize };
                    if self.residual[travel] <= 0.0 {
                        continue;
                    }
                    if let Some(distance) = self.origin(j) {
                        if distance + 1 < best_distance {
                            best_distance = distance + 1;
                            best = arc as u32;
                        }
                    }
                }
            }
            if best != NONE {
                self.parent[i] = best;
                self.stamp[i] = self.time;
                self.distance[i] = best_distance;
                continue;
            }
            // No parent: the orphan leaves its tree, its children are
            // orphaned, and a neighbour that could reach it is reactivated.
            for arc in self.start[i]..self.start[i + 1] {
                let arc = arc as usize;
                let j = self.head[arc] as usize;
                if self.parent[j] == NONE || self.sink_tree[j] != sink {
                    continue;
                }
                let travel = if sink { arc } else { self.sister[arc] as usize };
                if self.residual[travel] > 0.0 {
                    self.activate(j as u32);
                }
                let parent = self.parent[j];
                if parent != TERMINAL && self.head[parent as usize] as usize == i {
                    self.parent[j] = NONE;
                    self.orphans.push_back(j as u32);
                }
            }
        }
    }

    /// Nodes reachable from the source through residual above `floor`.
    fn source_side(&self, floor: f64) -> Vec<bool> {
        let mut side = vec![false; self.n_nodes];
        let mut queue = VecDeque::new();
        for (node, &terminal) in self.terminal.iter().enumerate() {
            if terminal > floor {
                side[node] = true;
                queue.push_back(node);
            }
        }
        while let Some(i) = queue.pop_front() {
            for arc in self.start[i]..self.start[i + 1] {
                let arc = arc as usize;
                let j = self.head[arc] as usize;
                if !side[j] && self.residual[arc] > floor {
                    side[j] = true;
                    queue.push_back(j);
                }
            }
        }
        side
    }
}

fn labels_of(
    labels: &[i64],
    n_nodes: usize,
    n_states: usize,
    values: usize,
) -> PyResult<Vec<usize>> {
    let labels = node_indices(labels)?;
    if labels.len() != n_nodes || values != n_nodes * n_states {
        return Err(PyValueError::new_err(
            "labels must have one entry per node and values be (n_nodes, n_states)",
        ));
    }
    if let Some(&label) = labels.iter().find(|&&l| l >= n_states) {
        return Err(PyValueError::new_err(format!(
            "label {label} outside [0, {n_states})"
        )));
    }
    Ok(labels)
}

#[pymethods]
impl LatticeCut {
    /// The lattice `first[e] -- second[e]` at `coupling[e]`, laid out once.
    #[new]
    fn py_new(
        n_nodes: usize,
        first: PyReadonlyArray1<'_, i64>,
        second: PyReadonlyArray1<'_, i64>,
        coupling: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Self> {
        let first = node_indices(first.as_slice()?)?;
        let second = node_indices(second.as_slice()?)?;
        Self::build(n_nodes, &first, &second, coupling.as_slice()?).map_err(PyValueError::new_err)
    }

    /// The source side of one alpha-expansion move's minimal minimum cut:
    /// `True` keeps its label, `False` takes `alpha`.
    fn expansion_source_side<'py>(
        &mut self,
        py: Python<'py>,
        values: PyReadonlyArray1<'_, f64>,
        n_states: usize,
        labels: PyReadonlyArray1<'_, i64>,
        alpha: usize,
        pinned: f64,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let values = values.as_slice()?;
        let labels = labels_of(labels.as_slice()?, self.n_nodes, n_states, values.len())?;
        if alpha >= n_states {
            return Err(PyValueError::new_err(format!(
                "alpha {alpha} outside [0, {n_states})"
            )));
        }
        let side = py.detach(|| {
            self.fill_expansion(values, n_states, &labels, alpha, pinned);
            self.solve(Some(alpha))
        });
        Ok(PyArray1::from_vec(py, side))
    }

    /// The source side of one alpha-beta swap's minimal minimum cut over
    /// every node: `True` takes `alpha` where the node is at `alpha` or
    /// `beta`, and every other node reads `False`.
    fn swap_source_side<'py>(
        &mut self,
        py: Python<'py>,
        values: PyReadonlyArray1<'_, f64>,
        n_states: usize,
        labels: PyReadonlyArray1<'_, i64>,
        alpha: usize,
        beta: usize,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let values = values.as_slice()?;
        let labels = labels_of(labels.as_slice()?, self.n_nodes, n_states, values.len())?;
        if alpha >= n_states || beta >= n_states {
            return Err(PyValueError::new_err(
                "alpha and beta must lie in [0, n_states)",
            ));
        }
        let side = py.detach(|| self.swap_side(values, n_states, &labels, alpha, beta));
        Ok(PyArray1::from_vec(py, side))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::maxflow::{expansion_network, max_flow_impl};

    /// `first`, `second`, `coupling`, the field row-major, and a labelling.
    type Problem = (Vec<usize>, Vec<usize>, Vec<f64>, Vec<f64>, Vec<usize>);

    /// A `side x side` open lattice with couplings in `[0.2, 1.2)` and a
    /// random field and labelling, from a fixed linear congruential stream.
    fn lattice(side: usize, n_states: usize, seed: u64) -> Problem {
        let mut state = seed;
        let mut draw = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1_u64 << 53) as f64
        };
        let (mut first, mut second, mut coupling) = (Vec::new(), Vec::new(), Vec::new());
        for row in 0..side {
            for column in 0..side {
                let node = row * side + column;
                if column + 1 < side {
                    first.push(node);
                    second.push(node + 1);
                    coupling.push(0.2 + draw());
                }
                if row + 1 < side {
                    first.push(node);
                    second.push(node + side);
                    coupling.push(0.2 + draw());
                }
            }
        }
        let n = side * side;
        let values: Vec<f64> = (0..n * n_states).map(|_| 2.0 * draw() - 1.0).collect();
        let labels: Vec<usize> = (0..n)
            .map(|_| (draw() * n_states as f64) as usize)
            .collect();
        (first, second, coupling, values, labels)
    }

    #[test]
    fn arcs_pair_with_their_sisters() {
        let (first, second, coupling, _, _) = lattice(5, 3, 1);
        let cut = LatticeCut::build(25, &first, &second, &coupling).unwrap();
        for (edge, &[out, back]) in cut.arc_of_edge.iter().enumerate() {
            let (out, back) = (out as usize, back as usize);
            assert_eq!(
                (cut.sister[out] as usize, cut.sister[back] as usize),
                (back, out)
            );
            assert_eq!(cut.head[out] as usize, second[edge]);
            assert_eq!(cut.head[back] as usize, first[edge]);
            let row = |arc: usize| {
                (0..25).find(|&n| (cut.start[n] as usize..cut.start[n + 1] as usize).contains(&arc))
            };
            assert_eq!(
                (row(out), row(back)),
                (Some(first[edge]), Some(second[edge]))
            );
        }
    }

    #[test]
    fn the_expansion_cut_is_the_auxiliary_networks() {
        // Two encodings of one energy share the minimal minimum cut.
        for seed in 0..20 {
            let (side, q) = (9, 4);
            let (first, second, coupling, values, labels) = lattice(side, q, seed);
            let n = side * side;
            let pinned =
                1.0 + values.iter().map(|v| v.abs()).sum::<f64>() + coupling.iter().sum::<f64>();
            let mut cut = LatticeCut::build(n, &first, &second, &coupling).unwrap();
            for alpha in 0..q {
                let mut network = expansion_network(
                    &first, &second, &coupling, &values, q, &labels, alpha, pinned,
                );
                let (_, oracle) = max_flow_impl(&mut network, n, n + 1).unwrap();
                cut.fill_expansion(&values, q, &labels, alpha, pinned);
                assert_eq!(
                    cut.solve(None),
                    oracle[..n].to_vec(),
                    "seed {seed}, alpha {alpha}"
                );
            }
        }
    }

    #[test]
    fn the_swap_cut_is_the_subset_networks() {
        for seed in 0..20 {
            let (side, q) = (9, 4);
            let (first, second, coupling, values, labels) = lattice(side, q, seed);
            let n = side * side;
            let mut cut = LatticeCut::build(n, &first, &second, &coupling).unwrap();
            for alpha in 0..q {
                for beta in alpha + 1..q {
                    let moving: Vec<usize> = (0..n)
                        .filter(|&v| labels[v] == alpha || labels[v] == beta)
                        .collect();
                    let mut position = vec![usize::MAX; n];
                    for (index, &v) in moving.iter().enumerate() {
                        position[v] = index;
                    }
                    let m = moving.len();
                    let mut network = crate::maxflow::FlowNetwork::new(m + 2);
                    for (index, &v) in moving.iter().enumerate() {
                        let (take_alpha, take_beta) =
                            (-values[v * q + alpha], -values[v * q + beta]);
                        let offset = take_alpha.min(take_beta);
                        network.add_edge(m, index, take_beta - offset, 0.0).unwrap();
                        network
                            .add_edge(index, m + 1, take_alpha - offset, 0.0)
                            .unwrap();
                    }
                    for e in 0..first.len() {
                        let (a, b) = (position[first[e]], position[second[e]]);
                        if a != usize::MAX && b != usize::MAX {
                            network.add_edge(a, b, coupling[e], coupling[e]).unwrap();
                        }
                    }
                    let (_, oracle) = max_flow_impl(&mut network, m, m + 1).unwrap();
                    cut.fill_swap(&values, q, &labels, alpha, beta);
                    let side_all = cut.solve(None);
                    let got: Vec<bool> = moving.iter().map(|&v| side_all[v]).collect();
                    assert_eq!(got, oracle[..m].to_vec(), "seed {seed}, ({alpha}, {beta})");
                    assert!((0..n).all(|v| position[v] != usize::MAX || !side_all[v]));
                }
            }
        }
    }

    #[test]
    fn a_warm_start_leaves_the_cut_where_a_cold_one_puts_it() {
        // Three expansion cycles with the labelling moving under each: every
        // cut started from its label's last flow is the cut from zero.
        for seed in 0..10 {
            let (side, q) = (12, 5);
            let (first, second, coupling, values, mut labels) = lattice(side, q, 100 + seed);
            let n = side * side;
            let pinned = 1.0 + values.iter().map(|v| v.abs()).sum::<f64>();
            let mut warm = LatticeCut::build(n, &first, &second, &coupling).unwrap();
            let mut cold = LatticeCut::build(n, &first, &second, &coupling).unwrap();
            for _ in 0..3 {
                for alpha in 0..q {
                    warm.fill_expansion(&values, q, &labels, alpha, pinned);
                    let reused = warm.solve(Some(alpha));
                    cold.fill_expansion(&values, q, &labels, alpha, pinned);
                    let fresh = cold.solve(None);
                    assert_eq!(reused, fresh, "seed {seed}, alpha {alpha}");
                    for node in 0..n {
                        if !fresh[node] {
                            labels[node] = alpha;
                        }
                    }
                }
            }
        }
    }
}
