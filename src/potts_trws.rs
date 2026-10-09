//! Sequential tree-reweighted message passing held in a `PottsProblem`
//! (issue #1413): `sal.search.trws.numba.trws_iterations` in Rust, operation
//! for operation.
//!
//! The chain layout is `sal.search.trws.chain_layout`'s, handed over once
//! and held with the messages, so a solve crosses once and allocates
//! nothing it held before. Every sum runs in the kernel's order (a site's
//! incoming messages in row order, the chains in their order, the decode's
//! energy over sites then edges), so the bound trace and the labelling are
//! the numba kernel's bitwise; `tests/regression/search/test_potts_problem.py`
//! pins them.

/// `sal.search.trws.ChainLayout`, narrowed to `u32` indices, and the
/// buffers one run fills.
pub struct Chains {
    /// `n_nodes + 1` row bounds into `slots`.
    pub offsets: Vec<u32>,
    /// Each row's entries, in edge order.
    pub slots: Vec<u32>,
    /// The site each entry names.
    pub neighbours: Vec<u32>,
    /// Each edge's lower end.
    pub low: Vec<u32>,
    /// Each edge's higher end.
    pub high: Vec<u32>,
    /// `J` per edge.
    pub coupling: Vec<f64>,
    /// `1 / n_s` per site.
    pub weight: Vec<f64>,
    /// Chain bounds into `chain_edges`.
    pub chain_offsets: Vec<u32>,
    /// Each chain's first site.
    pub chain_heads: Vec<u32>,
    /// The edges each chain walks.
    pub chain_edges: Vec<u32>,
    messages: Vec<f64>,
    shares: Vec<f64>,
    unary: Vec<f64>,
    labels: Vec<usize>,
    scratch: [Vec<f64>; 4],
}

/// What one run returns: the kept labelling, its decoded energy, the bound
/// trace, and whether it converged.
pub struct TrwsRun {
    /// The lowest-energy decode, the first at a tie.
    pub labelling: Vec<i64>,
    /// Its energy as the decode scores it.
    pub energy: f64,
    /// The chain bound after each iteration.
    pub trace: Vec<f64>,
    /// Whether the bound or the gap settled within the tolerance.
    pub converged: bool,
}

fn narrow(values: &[i64], name: &str) -> Result<Vec<u32>, String> {
    values
        .iter()
        .map(|&v| u32::try_from(v).map_err(|_| format!("{name} holds {v}, outside u32")))
        .collect()
}

/// `out[b] = min_a [values[a] - coupling [a == b]]`: the kernel's `_potts_min`.
#[inline]
fn potts_min(values: &[f64], coupling: f64, out: &mut [f64]) {
    let mut first = 0usize;
    for label in 1..values.len() {
        if values[label] < values[first] {
            first = label;
        }
    }
    let mut second = f64::INFINITY;
    for (label, &value) in values.iter().enumerate() {
        if label != first && value < second {
            second = value;
        }
    }
    for (label, slot) in out.iter_mut().enumerate() {
        let other = if label == first {
            second
        } else {
            values[first]
        };
        let diagonal = values[label] - coupling;
        *slot = if diagonal < other { diagonal } else { other };
    }
}

impl Chains {
    /// From `chain_layout`'s arrays (`ends` flattened row-major) for
    /// `n_nodes` sites of `n_states` labels.
    ///
    /// # Errors
    /// Returns `Err` where a length or an index is inconsistent.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        n_nodes: usize,
        n_states: usize,
        offsets: &[i64],
        slots: &[i64],
        neighbours: &[i64],
        ends: &[i64],
        coupling: &[f64],
        weight: &[f64],
        chain_offsets: &[i64],
        chain_heads: &[i64],
        chain_edges: &[i64],
    ) -> Result<Self, String> {
        let n_edges = coupling.len();
        let n_chains = chain_heads.len();
        let n_entries = 2 * n_edges;
        let offsets = narrow(offsets, "offsets")?;
        let slots = narrow(slots, "slots")?;
        let neighbours = narrow(neighbours, "neighbours")?;
        let ends = narrow(ends, "ends")?;
        let chain_offsets = narrow(chain_offsets, "chain_offsets")?;
        let chain_heads = narrow(chain_heads, "chain_heads")?;
        let chain_edges = narrow(chain_edges, "chain_edges")?;
        let consistent = offsets.len() == n_nodes + 1
            && offsets[0] == 0
            && offsets[n_nodes] as usize == n_entries
            && offsets.windows(2).all(|w| w[0] <= w[1])
            && slots.len() == n_entries
            && slots.iter().all(|&s| (s as usize) < n_entries)
            && neighbours.len() == n_entries
            && neighbours.iter().all(|&s| (s as usize) < n_nodes)
            && ends.len() == n_edges * 2
            && ends
                .chunks(2)
                .all(|e| e[0] < e[1] && (e[1] as usize) < n_nodes)
            && weight.len() == n_nodes
            && chain_offsets.len() == n_chains + 1
            && chain_offsets[n_chains] as usize == chain_edges.len()
            && chain_offsets.windows(2).all(|w| w[0] <= w[1])
            && chain_heads.iter().all(|&s| (s as usize) < n_nodes)
            && chain_edges.iter().all(|&e| (e as usize) < n_edges);
        if !consistent {
            return Err("the chain layout is inconsistent with the problem".to_string());
        }
        Ok(Self {
            offsets,
            slots,
            neighbours,
            low: ends.iter().step_by(2).copied().collect(),
            high: ends.iter().skip(1).step_by(2).copied().collect(),
            coupling: coupling.to_vec(),
            weight: weight.to_vec(),
            chain_offsets,
            chain_heads,
            chain_edges,
            messages: vec![0.0; n_entries * n_states],
            shares: vec![0.0; n_nodes * n_states],
            unary: vec![0.0; n_nodes * n_states],
            labels: vec![0; n_nodes],
            scratch: std::array::from_fn(|_| vec![0.0; n_states]),
        })
    }

    /// The heap bytes the layout and the buffers hold.
    #[must_use]
    pub fn bytes(&self) -> usize {
        4 * (self.offsets.capacity()
            + self.slots.capacity()
            + self.neighbours.capacity()
            + self.low.capacity()
            + self.high.capacity()
            + self.chain_offsets.capacity()
            + self.chain_heads.capacity()
            + self.chain_edges.capacity())
            + 8 * (self.coupling.capacity()
                + self.weight.capacity()
                + self.messages.capacity()
                + self.shares.capacity()
                + self.unary.capacity()
                + self.labels.capacity()
                + self.scratch.iter().map(Vec::capacity).sum::<usize>())
    }

    /// `theta_s + sum_k M_k`, the incoming messages in row order.
    #[inline]
    fn node_share(&self, node: usize, q: usize, out: &mut [f64]) {
        out.copy_from_slice(&self.unary[node * q..(node + 1) * q]);
        for position in self.offsets[node] as usize..self.offsets[node + 1] as usize {
            let entry = self.slots[position] as usize;
            for (label, slot) in out.iter_mut().enumerate() {
                *slot += self.messages[entry * q + label];
            }
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn pass(&mut self, forward: bool, q: usize) {
        let n_nodes = self.weight.len();
        let parity = usize::from(!forward);
        let [mut share, mut lent, mut sent, value] = std::mem::take(&mut self.scratch);
        for step in 0..n_nodes {
            let node = if forward { step } else { n_nodes - 1 - step };
            self.node_share(node, q, &mut share);
            for position in self.offsets[node] as usize..self.offsets[node + 1] as usize {
                let entry = self.slots[position] as usize;
                if entry % 2 != parity {
                    continue;
                }
                for label in 0..q {
                    lent[label] =
                        self.weight[node] * share[label] - self.messages[entry * q + label];
                }
                potts_min(&lent, self.coupling[entry / 2], &mut sent);
                let mut lowest = sent[0];
                for &s in &sent[1..] {
                    if s < lowest {
                        lowest = s;
                    }
                }
                let reverse = entry ^ 1;
                for label in 0..q {
                    self.messages[reverse * q + label] = sent[label] - lowest;
                }
            }
        }
        self.scratch = [share, lent, sent, value];
    }

    #[allow(clippy::needless_range_loop)]
    fn chain_bound(&mut self, q: usize) -> f64 {
        let n_nodes = self.weight.len();
        let mut shares = std::mem::take(&mut self.shares);
        for node in 0..n_nodes {
            self.node_share(node, q, &mut shares[node * q..(node + 1) * q]);
        }
        let [share, mut lent, mut sent, mut value] = std::mem::take(&mut self.scratch);
        let mut total = 0.0f64;
        for chain in 0..self.chain_heads.len() {
            let head = self.chain_heads[chain] as usize;
            for label in 0..q {
                value[label] = self.weight[head] * shares[head * q + label];
            }
            for position in
                self.chain_offsets[chain] as usize..self.chain_offsets[chain + 1] as usize
            {
                let edge = self.chain_edges[position] as usize;
                let high = self.high[edge] as usize;
                for label in 0..q {
                    lent[label] = value[label] - self.messages[2 * edge * q + label];
                }
                potts_min(&lent, self.coupling[edge], &mut sent);
                for label in 0..q {
                    value[label] = sent[label] - self.messages[(2 * edge + 1) * q + label]
                        + self.weight[high] * shares[high * q + label];
                }
            }
            let mut lowest = value[0];
            for &v in &value[1..] {
                if v < lowest {
                    lowest = v;
                }
            }
            total += lowest;
        }
        self.scratch = [share, lent, sent, value];
        self.shares = shares;
        total
    }

    fn decode(&mut self, q: usize) -> f64 {
        let n_nodes = self.weight.len();
        let [mut local, lent, sent, value] = std::mem::take(&mut self.scratch);
        for node in 0..n_nodes {
            local.copy_from_slice(&self.unary[node * q..(node + 1) * q]);
            for position in self.offsets[node] as usize..self.offsets[node + 1] as usize {
                let entry = self.slots[position] as usize;
                let other = self.neighbours[entry] as usize;
                if other < node {
                    local[self.labels[other]] -= self.coupling[entry / 2];
                } else {
                    for (label, slot) in local.iter_mut().enumerate() {
                        *slot += self.messages[entry * q + label];
                    }
                }
            }
            let mut best = 0usize;
            for label in 1..q {
                if local[label] < local[best] {
                    best = label;
                }
            }
            self.labels[node] = best;
        }
        self.scratch = [local, lent, sent, value];
        let mut total = 0.0f64;
        for node in 0..n_nodes {
            total += self.unary[node * q + self.labels[node]];
        }
        for edge in 0..self.coupling.len() {
            if self.labels[self.neighbours[2 * edge + 1] as usize]
                == self.labels[self.neighbours[2 * edge] as usize]
            {
                total -= self.coupling[edge];
            }
        }
        total
    }

    /// TRW-S from zero messages on the field `field` (a log-weight,
    /// `n_nodes * n_states`): `trws_iterations`' loop and its stops.
    pub fn run(
        &mut self,
        field: &[f64],
        q: usize,
        max_iterations: usize,
        tolerance: f64,
    ) -> TrwsRun {
        let n_nodes = self.weight.len();
        for (slot, &value) in self.unary.iter_mut().zip(field) {
            *slot = -value;
        }
        self.messages.fill(0.0);
        let mut labelling = vec![0_i64; n_nodes];
        let mut trace: Vec<f64> = Vec::with_capacity(max_iterations.min(1024));
        let mut best_energy = f64::INFINITY;
        let mut converged = false;
        for iteration in 0..max_iterations {
            self.pass(true, q);
            self.pass(false, q);
            let bound = self.chain_bound(q);
            trace.push(bound);
            let current = self.decode(q);
            if current < best_energy {
                best_energy = current;
                for (slot, &label) in labelling.iter_mut().zip(&self.labels) {
                    *slot = label as i64;
                }
            }
            let scale = 1.0f64.max(bound.abs());
            if best_energy - bound <= tolerance * scale
                || (iteration > 0 && bound - trace[iteration - 1] <= tolerance * scale)
            {
                converged = true;
                break;
            }
        }
        TrwsRun {
            labelling,
            energy: best_energy,
            trace,
            converged,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One edge 0 -- 1 at coupling 1 and two labels: the bound is the
    /// minimum and the decode the ground state.
    #[test]
    fn one_edge_is_solved_exactly() {
        // Edge entries: 0 at site 0 naming 1, 1 at site 1 naming 0.
        let mut chains = Chains::build(
            2,
            2,
            &[0, 1, 2],
            &[0, 1],
            &[1, 0],
            &[0, 1],
            &[1.0],
            &[1.0, 1.0],
            &[0, 1],
            &[0],
            &[0],
        )
        .unwrap();
        // h_0 = (0, 0.5), h_1 = (0.2, 0): (1, 1) scores -0.5 - 1, the minimum.
        let ran = chains.run(&[0.0, 0.5, 0.2, 0.0], 2, 50, 1e-12);
        assert_eq!(ran.labelling, vec![1, 1]);
        assert!((ran.energy + 1.5).abs() < 1e-15);
        assert!((ran.trace.last().unwrap() + 1.5).abs() < 1e-12);
        assert!(ran.converged);
    }
}
