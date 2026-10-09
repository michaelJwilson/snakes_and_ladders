//! One Potts problem held across solves (#1413): the graph, the field and
//! the buffers, exposed as `sal.oxisal.PottsProblem` behind
//! `sal.search.potts_problem.PottsProblem`.
//!
//! **One crossing per solve.** Each solver method copies its start in, runs
//! with the GIL released, and copies the labelling out; the graph is copied
//! once, at construction, and `set_field` copies `n q` floats.
//!
//! **Bitwise the kernels it replaces where the visit order is theirs.** A
//! site's conditional is `h_i + sum_j J_ij [s_j = k]` summed in the
//! adjacency's order, the first maximum taken, as
//! [`crate::potts::conditional`] and `icm_sweeps_impl` do; the whole-label
//! merge is `potts_loop::merge_labels`' arithmetic; the anneal is
//! [`potts_loop_impl`] itself.
//!
//! **Storage chosen on the stream's problems (#1413).** The `potts_labelling`
//! fixtures hold degree 6, `q` of 4 and 5, every field row distinct and no
//! forbidden label. So the adjacency is `u32` (half of `int64`) and labels
//! are `u8` where `q <= 255`, `u32` otherwise; couplings stay per entry,
//! since no problem promises one coupling for every edge. A field row is not
//! deduplicated (no row repeats on the stream) and the field stays `f64`
//! (`f32` is not exact).
//!
//! **The expansion and TRW-S hold their own structures.** The expansion's
//! [`LatticeCut`] over the symmetric edges and TRW-S's chain layout
//! ([`Chains`]) are handed over on the first call and kept, so each later
//! solve crosses once; every run starts from zero flow and zero messages,
//! as the per-call functions do, and is bitwise theirs.
//!
//! **Directed rows** hold a directed adjacency as given: a descent reads a
//! site's own row (the downstream labelling step's conditional), the energy
//! halves every entry, and a clean site is not skipped, since a row need not
//! name every site that names it.
//!
//! **Skipping a clean site is exact.** A site's conditional reads its field
//! row and its neighbours' labels only; a site visited since any of them
//! changed, and not itself moved since by a floor, recomputes the argmax it
//! already holds. The descent keeps that set (`dirty`) and visits only the
//! rest, so a later sweep costs the sites near a change.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardUniform};

use crate::icm::no_survivor;
use crate::lattice_cut::LatticeCut;
use crate::potts_loop::{potts_loop_impl, ran_dict, Plan};
use crate::potts_trws::{Chains, TrwsRun};
use crate::wolff::Lattice;

/// `SweepOrder.INDEX`: `range(n_nodes)` every sweep.
pub const INDEX: u8 = 0;
/// `SweepOrder.RANDOM`: a fresh uniform permutation per sweep.
pub const RANDOM: u8 = 1;
/// `SweepOrder.CHECKERBOARD`: the greedy colouring's classes in turn.
pub const CHECKERBOARD: u8 = 2;
/// `SweepOrder.WORKLIST`: the downstream two-queue descent with its
/// shuffles replaced by index order. An epoch visits its queue ascending; a
/// site that changes queues each site of its row not already queued, for the
/// next epoch. The descent stops after an epoch that changed nothing (and
/// whose floor did not count against the guard), once at most one label is
/// in use, or when the next queue is empty.
pub const WORKLIST: u8 = 3;

/// `FloorPolicy.UNIFORM`: every state below the floor dissolved at once onto
/// uniform draws among the surviving states a site allows.
pub const UNIFORM: u8 = 0;
/// `FloorPolicy.SMALLEST_FIRST_BEST_FIELD`: the smallest state first, each
/// site onto its best-field live state; no draws.
pub const SMALLEST_FIRST_BEST_FIELD: u8 = 1;

/// `FloorAt.SWEEP`: the floor after every sweep, wherever a used label is
/// below it.
pub const AT_SWEEP: u8 = 0;
/// `FloorAt.EPOCH_GUARDED`: the downstream floor. After every sweep or
/// epoch, and only while every label is in use and one is below the floor;
/// a firing with two or more labels at the floor counts against a guard,
/// and the descent stops once the guard passes [`GUARD`].
pub const AT_EPOCH: u8 = 1;
/// The firings `EPOCH_GUARDED` allows before it stops the descent: the
/// downstream `min_spot_guard > 10`.
pub const GUARD: usize = 10;

/// The stop codes the Python class reads as `Stop`.
pub const CONVERGED: u8 = 0;
/// The sweeps ran out.
pub const BUDGET: u8 = 1;
/// A site below the floor allows no state at it.
pub const INFEASIBLE: u8 = 2;

/// A label stored narrow or wide; `i64` reads a caller's labelling.
pub trait Label: Copy + PartialEq + Send + Sync {
    /// The label as an index.
    fn get(self) -> usize;
    /// The index as a label.
    fn put(value: usize) -> Self;
}

impl Label for u8 {
    #[inline]
    fn get(self) -> usize {
        self as usize
    }
    #[inline]
    fn put(value: usize) -> Self {
        value as u8
    }
}

impl Label for u32 {
    #[inline]
    fn get(self) -> usize {
        self as usize
    }
    #[inline]
    fn put(value: usize) -> Self {
        value as u32
    }
}

impl Label for usize {
    #[inline]
    fn get(self) -> usize {
        self
    }
    #[inline]
    fn put(value: usize) -> Self {
        value
    }
}

impl Label for i64 {
    #[inline]
    fn get(self) -> usize {
        self as usize
    }
    #[inline]
    fn put(value: usize) -> Self {
        value as i64
    }
}

/// The problem's read-only half: the field and the adjacency.
pub struct Graph<'a> {
    /// `n_nodes * n_states`, row-major.
    pub field: &'a [f64],
    /// Labels per site.
    pub n_states: usize,
    /// `n_nodes + 1` row bounds.
    pub offsets: &'a [u32],
    /// The far site of each entry.
    pub neighbours: &'a [u32],
    /// Each entry's coupling.
    pub couplings: &'a [f64],
    /// Whether the rows are directed: a row then need not name every site
    /// that names it, so a change marks no neighbour clean or dirty exactly.
    pub directed: bool,
}

impl Graph<'_> {
    /// The first label of greatest conditional at `node`, `local` left
    /// holding the conditional.
    #[inline]
    pub fn best<L: Label>(&self, local: &mut [f64], labels: &[L], node: usize) -> usize {
        let q = self.n_states;
        local.copy_from_slice(&self.field[node * q..(node + 1) * q]);
        for entry in self.offsets[node] as usize..self.offsets[node + 1] as usize {
            local[labels[self.neighbours[entry] as usize].get()] += self.couplings[entry];
        }
        let mut best = 0usize;
        for label in 1..q {
            if local[label] > local[best] {
                best = label;
            }
        }
        best
    }

    /// [`Graph::best`] with the row's couplings summed per label first and
    /// then added to the field, the downstream descent's arithmetic.
    #[inline]
    pub fn best_split<L: Label>(
        &self,
        local: &mut [f64],
        sums: &mut [f64],
        labels: &[L],
        node: usize,
    ) -> usize {
        let q = self.n_states;
        sums.fill(0.0);
        for entry in self.offsets[node] as usize..self.offsets[node + 1] as usize {
            sums[labels[self.neighbours[entry] as usize].get()] += self.couplings[entry];
        }
        let row = &self.field[node * q..(node + 1) * q];
        let mut best = 0usize;
        for label in 0..q {
            local[label] = row[label] + sums[label];
            if label > 0 && local[label] > local[best] {
                best = label;
            }
        }
        best
    }

    /// `E(s) = -sum_i h_i[s_i] - sum_(ij) J_ij [s_i = s_j]`, each edge read
    /// from both ends and halved: `potts_loop`'s `energy_of`, bitwise.
    pub fn energy<L: Label>(&self, labels: &[L]) -> f64 {
        let q = self.n_states;
        let (mut field, mut bonds) = (0.0f64, 0.0f64);
        for (node, &label) in labels.iter().enumerate() {
            field += self.field[node * q + label.get()];
            for entry in self.offsets[node] as usize..self.offsets[node + 1] as usize {
                if labels[self.neighbours[entry] as usize] == label {
                    bonds += self.couplings[entry];
                }
            }
        }
        -field - 0.5 * bonds
    }

    fn mark(&self, dirty: &mut [bool], node: usize) {
        dirty[node] = true;
        for entry in self.offsets[node] as usize..self.offsets[node + 1] as usize {
            dirty[self.neighbours[entry] as usize] = true;
        }
    }
}

/// What one descent is asked for.
#[derive(Clone, Copy, Debug)]
pub struct Descend {
    /// A sweep order code.
    pub order: u8,
    /// The floor; `0` dissolves nothing.
    pub min_sites: usize,
    /// A floor policy code.
    pub policy: u8,
    /// A floor placement code.
    pub floor_at: u8,
    /// Sweeps allowed; a worklist's visits are `max_iterations * n_nodes`.
    pub max_iterations: usize,
    /// Whether a site whose neighbourhood has not changed is skipped.
    pub skip_clean: bool,
}

/// The buffers every solve reuses.
#[derive(Default)]
pub struct Buffers {
    local: Vec<f64>,
    counts: Vec<usize>,
    surviving: Vec<usize>,
    allowed: Vec<usize>,
    stuck: Vec<bool>,
    dirty: Vec<bool>,
    order: Vec<u32>,
    next: Vec<u32>,
    moved: Vec<u32>,
    sums: Vec<f64>,
    field_sums: Vec<f64>,
    bond_sums: Vec<f64>,
    admissible: Vec<usize>,
}

impl Buffers {
    /// For `n_nodes` sites of `n_states` labels.
    #[must_use]
    pub fn new(n_nodes: usize, n_states: usize) -> Self {
        let square = n_states * n_states;
        Self {
            local: vec![0.0; n_states],
            counts: vec![0; n_states],
            surviving: vec![0; n_states],
            allowed: vec![0; n_states],
            stuck: vec![false; n_states],
            dirty: vec![true; n_nodes],
            order: (0..n_nodes as u32).collect(),
            next: Vec::with_capacity(n_nodes),
            moved: Vec::with_capacity(n_nodes),
            sums: vec![0.0; n_states],
            field_sums: vec![0.0; square],
            bond_sums: vec![0.0; square],
            admissible: vec![0; square],
        }
    }

    /// The heap bytes the buffers hold.
    #[must_use]
    pub fn bytes(&self) -> usize {
        8 * (self.local.capacity()
            + self.counts.capacity()
            + self.surviving.capacity()
            + self.allowed.capacity()
            + self.sums.capacity()
            + self.field_sums.capacity()
            + self.bond_sums.capacity()
            + self.admissible.capacity())
            + self.stuck.capacity()
            + self.dirty.capacity()
            + 4 * (self.order.capacity() + self.next.capacity() + self.moved.capacity())
    }
}

/// The greedy colouring's classes, in index order within a class:
/// `icm.colour_order`.
#[must_use]
pub fn colour_order(offsets: &[u32], neighbours: &[u32]) -> Vec<u32> {
    let n_nodes = offsets.len() - 1;
    let mut colour = vec![u32::MAX; n_nodes];
    let mut taken: Vec<usize> = Vec::new();
    for node in 0..n_nodes {
        let (lo, hi) = (offsets[node] as usize, offsets[node + 1] as usize);
        if taken.len() < hi - lo + 1 {
            taken.resize(hi - lo + 1, usize::MAX);
        }
        for &far in &neighbours[lo..hi] {
            let c = colour[far as usize] as usize;
            if c < taken.len() {
                taken[c] = node;
            }
        }
        let mut chosen = 0usize;
        while taken[chosen] == node {
            chosen += 1;
        }
        colour[node] = chosen as u32;
    }
    let mut order: Vec<u32> = (0..n_nodes as u32).collect();
    order.sort_by_key(|&node| colour[node as usize]);
    order
}

/// A uniform index below `n` from 64 random bits (Lemire's multiply, no
/// rejection: a bias of at most `n / 2^64`).
#[inline]
fn below(rng: &mut ChaCha8Rng, n: usize) -> usize {
    ((u128::from(rng.next_u64()) * n as u128) >> 64) as usize
}

/// The uniform floor after a sweep, in place: the counts read once, then
/// each state below the floor in ascending order and its sites in index
/// order, each site taking `allowed[floor(u m)]` among the ascending
/// surviving states its field allows; one uniform per site moved. The
/// downstream floor's order; `icm_sweeps_impl` draws the same law site by
/// site in index order. The sites moved go to
/// `buffers.moved`.
fn floor_uniform<L: Label>(
    graph: &Graph<'_>,
    labels: &mut [L],
    min_sites: usize,
    buffers: &mut Buffers,
    rng: &mut ChaCha8Rng,
    sweep: usize,
) -> Result<(), String> {
    let q = graph.n_states;
    let Buffers {
        counts,
        surviving,
        allowed,
        moved,
        ..
    } = buffers;
    counts.fill(0);
    for &label in labels.iter() {
        counts[label.get()] += 1;
    }
    let (mut m, mut short) = (0usize, false);
    for (label, &count) in counts.iter().enumerate() {
        if count >= min_sites {
            surviving[m] = label;
            m += 1;
        } else if count > 0 {
            short = true;
        }
    }
    if !short {
        return Ok(());
    }
    if m == 0 {
        return Err(no_survivor(sweep, min_sites));
    }
    let n_nodes = labels.len();
    for (node, short) in (0..q)
        .filter(|&state| counts[state] > 0 && counts[state] < min_sites)
        .flat_map(|state| (0..n_nodes).map(move |node| (node, state)))
    {
        if labels[node].get() != short {
            continue;
        }
        let mut n_allowed = 0usize;
        for &label in &surviving[..m] {
            if graph.field[node * q + label] > f64::NEG_INFINITY {
                allowed[n_allowed] = label;
                n_allowed += 1;
            }
        }
        if n_allowed == 0 {
            continue;
        }
        let draw: f64 = StandardUniform.sample(&mut *rng);
        let pick = (draw * n_allowed as f64) as usize;
        labels[node] = L::put(allowed[pick.min(n_allowed - 1)]);
        moved.push(node as u32);
    }
    Ok(())
}

/// The smallest-first, best-field floor in place: `numba.floor_smallest_first`
/// bitwise. The sites moved go to `buffers.moved`.
fn floor_smallest_first<L: Label>(
    graph: &Graph<'_>,
    labels: &mut [L],
    min_sites: usize,
    buffers: &mut Buffers,
    sweep: usize,
) -> Result<(), String> {
    let q = graph.n_states;
    let Buffers {
        counts,
        stuck,
        moved,
        ..
    } = buffers;
    counts.fill(0);
    for &label in labels.iter() {
        counts[label.get()] += 1;
    }
    stuck.fill(false);
    loop {
        let mut smallest = usize::MAX;
        for state in 0..q {
            let count = counts[state];
            if 0 < count
                && count < min_sites
                && !stuck[state]
                && (smallest == usize::MAX || count < counts[smallest])
            {
                smallest = state;
            }
        }
        if smallest == usize::MAX {
            return Ok(());
        }
        if !(0..q).any(|state| state != smallest && counts[state] > 0) {
            return Err(no_survivor(sweep, min_sites));
        }
        for (node, label) in labels.iter_mut().enumerate() {
            if label.get() != smallest {
                continue;
            }
            let row = &graph.field[node * q..(node + 1) * q];
            let mut best = usize::MAX;
            for state in 0..q {
                if state != smallest
                    && counts[state] > 0
                    && row[state] > f64::NEG_INFINITY
                    && (best == usize::MAX || row[state] > row[best])
                {
                    best = state;
                }
            }
            if best != usize::MAX {
                *label = L::put(best);
                counts[smallest] -= 1;
                counts[best] += 1;
                moved.push(node as u32);
            }
        }
        if counts[smallest] > 0 {
            stuck[smallest] = true;
        }
    }
}

/// One floor of `policy`; whether it moved a site, each moved site and its
/// neighbours marked dirty and, on a worklist, queued.
fn apply_floor<L: Label>(
    graph: &Graph<'_>,
    labels: &mut [L],
    ask: &Descend,
    buffers: &mut Buffers,
    rng: &mut ChaCha8Rng,
    sweep: usize,
    queued: bool,
) -> Result<bool, String> {
    buffers.moved.clear();
    if ask.policy == SMALLEST_FIRST_BEST_FIELD {
        floor_smallest_first(graph, labels, ask.min_sites, buffers, sweep)?;
    } else {
        floor_uniform(graph, labels, ask.min_sites, buffers, rng, sweep)?;
    }
    let Buffers {
        moved, dirty, next, ..
    } = buffers;
    for &node in moved.iter() {
        let node = node as usize;
        if queued {
            // The downstream queue rule: the moved site's row, not the site.
            for entry in graph.offsets[node] as usize..graph.offsets[node + 1] as usize {
                let site = graph.neighbours[entry] as usize;
                if !dirty[site] {
                    dirty[site] = true;
                    next.push(site as u32);
                }
            }
        } else {
            graph.mark(dirty, node);
        }
    }
    Ok(!moved.is_empty())
}

/// The `EPOCH_GUARDED` floor: `None` unless every label is in use and one
/// is below `min_sites`; otherwise the floor of `ask.policy` over the labels
/// at the floor (none moves where no label is), and whether two or more
/// labels were at the floor, the firings the guard counts.
fn epoch_floor<L: Label>(
    graph: &Graph<'_>,
    labels: &mut [L],
    ask: &Descend,
    buffers: &mut Buffers,
    rng: &mut ChaCha8Rng,
    sweep: usize,
    queued: bool,
) -> Result<Option<bool>, String> {
    let counts = &mut buffers.counts;
    counts.fill(0);
    for &label in labels.iter() {
        counts[label.get()] += 1;
    }
    let smallest = counts.iter().copied().min().unwrap_or(0);
    if !(0 < smallest && smallest < ask.min_sites) {
        return Ok(None);
    }
    let eligible = counts
        .iter()
        .filter(|&&count| count >= ask.min_sites)
        .count();
    if eligible > 0 {
        apply_floor(graph, labels, ask, buffers, rng, sweep, queued)?;
    }
    Ok(Some(eligible > 1))
}

/// Labels in use.
fn in_use<L: Label>(labels: &[L], counts: &mut [usize]) -> usize {
    counts.fill(0);
    for &label in labels {
        counts[label.get()] += 1;
    }
    counts.iter().filter(|&&count| count > 0).count()
}

/// The descent in place; the sweeps run (a worklist's visits over
/// `n_nodes`, rounded up) and the stop code.
///
/// # Errors
/// Returns `Err` with [`no_survivor`]'s message where a floor leaves no
/// state to dissolve into, or for an unknown code.
pub fn icm_run<L: Label>(
    graph: &Graph<'_>,
    labels: &mut [L],
    ask: &Descend,
    buffers: &mut Buffers,
    colours: &[u32],
    seed: u64,
) -> Result<(usize, u8), String> {
    if ask.order > WORKLIST || ask.policy > SMALLEST_FIRST_BEST_FIELD || ask.floor_at > AT_EPOCH {
        return Err(format!(
            "unknown code: order {}, policy {}, floor_at {}",
            ask.order, ask.policy, ask.floor_at
        ));
    }
    let n_nodes = labels.len();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    buffers.dirty.fill(true);
    let skip_clean = ask.skip_clean && !graph.directed;
    let mut guard = 0usize;
    let sweeps = if ask.order == WORKLIST {
        worklist(graph, labels, ask, buffers, &mut rng)?
    } else {
        let mut sweeps = 0usize;
        let mut local = std::mem::take(&mut buffers.local);
        let mut order = std::mem::take(&mut buffers.order);
        let result = (|| -> Result<usize, String> {
            while sweeps < ask.max_iterations {
                sweeps += 1;
                let visit: &[u32] = match ask.order {
                    RANDOM => {
                        for (position, slot) in order.iter_mut().enumerate() {
                            *slot = position as u32;
                        }
                        for top in (1..n_nodes).rev() {
                            order.swap(top, below(&mut rng, top + 1));
                        }
                        &order
                    }
                    CHECKERBOARD => colours,
                    _ => &[],
                };
                let mut changed = false;
                for position in 0..n_nodes {
                    let node = if visit.is_empty() {
                        position
                    } else {
                        visit[position] as usize
                    };
                    if skip_clean && !buffers.dirty[node] {
                        continue;
                    }
                    buffers.dirty[node] = false;
                    let best = graph.best(&mut local, labels, node);
                    if best != labels[node].get() {
                        labels[node] = L::put(best);
                        changed = true;
                        for entry in graph.offsets[node] as usize..graph.offsets[node + 1] as usize
                        {
                            buffers.dirty[graph.neighbours[entry] as usize] = true;
                        }
                    }
                }
                if ask.min_sites > 0 && ask.floor_at == AT_SWEEP {
                    if apply_floor(graph, labels, ask, buffers, &mut rng, sweeps, false)? {
                        changed = true;
                    }
                } else if ask.min_sites > 0
                    && epoch_floor(graph, labels, ask, buffers, &mut rng, sweeps, false)?
                        == Some(true)
                {
                    guard += 1;
                    changed = true;
                }
                if !changed || guard > GUARD {
                    break;
                }
            }
            Ok(sweeps)
        })();
        buffers.local = local;
        buffers.order = order;
        result?
    };
    if ask.order == WORKLIST || !skip_clean {
        // A floor-moved site is not queued, and a skipped read is not exact
        // on directed rows: read every site.
        buffers.dirty.fill(true);
    }
    Ok((sweeps, status(graph, labels, ask.min_sites, buffers)))
}

/// The worklist descent ([`WORKLIST`]): epochs over an ascending queue,
/// each changed site queuing its row's sites not already queued, the floor
/// after every epoch; at most `max_iterations` epochs. The epochs run.
fn worklist<L: Label>(
    graph: &Graph<'_>,
    labels: &mut [L],
    ask: &Descend,
    buffers: &mut Buffers,
    rng: &mut ChaCha8Rng,
) -> Result<usize, String> {
    let n_nodes = labels.len();
    let mut local = std::mem::take(&mut buffers.local);
    let mut sums = std::mem::take(&mut buffers.sums);
    let mut current = std::mem::take(&mut buffers.order);
    current.clear();
    current.extend(0..n_nodes as u32);
    buffers.next.clear();
    let (mut epochs, mut guard) = (0usize, 0usize);
    let result = (|| -> Result<(), String> {
        while !current.is_empty() && epochs < ask.max_iterations {
            let mut edits = 0usize;
            for &node in &current {
                let node = node as usize;
                buffers.dirty[node] = false;
                let best = graph.best_split(&mut local, &mut sums, labels, node);
                if best != labels[node].get() {
                    edits += 1;
                    labels[node] = L::put(best);
                    for entry in graph.offsets[node] as usize..graph.offsets[node + 1] as usize {
                        let far = graph.neighbours[entry] as usize;
                        if !buffers.dirty[far] {
                            buffers.dirty[far] = true;
                            buffers.next.push(far as u32);
                        }
                    }
                }
            }
            if ask.min_sites > 0 {
                if ask.floor_at == AT_SWEEP {
                    if apply_floor(graph, labels, ask, buffers, rng, epochs + 1, true)? {
                        edits += 1;
                    }
                } else if epoch_floor(graph, labels, ask, buffers, rng, epochs + 1, true)?
                    == Some(true)
                {
                    guard += 1;
                    edits += 1;
                }
            }
            epochs += 1;
            if edits == 0 || in_use(labels, &mut buffers.counts) <= 1 || guard > GUARD {
                return Ok(());
            }
            buffers.next.sort_unstable();
            std::mem::swap(&mut current, &mut buffers.next);
            buffers.next.clear();
        }
        Ok(())
    })();
    buffers.local = local;
    buffers.sums = sums;
    current.clear();
    current.extend(0..n_nodes as u32);
    buffers.order = current;
    result?;
    Ok(epochs)
}

/// `icm.descended`'s stop, read off the labelling: infeasible where a site
/// of a state below the floor allows no state at it, the budget where
/// another state is below it or a site not at its first best label,
/// converged otherwise. A site not dirty is at its best by construction,
/// so only the dirty ones are read.
pub fn status<L: Label>(
    graph: &Graph<'_>,
    labels: &[L],
    min_sites: usize,
    buffers: &mut Buffers,
) -> u8 {
    let q = graph.n_states;
    let mut settled = true;
    for node in 0..labels.len() {
        if buffers.dirty[node] && graph.best(&mut buffers.local, labels, node) != labels[node].get()
        {
            settled = false;
            break;
        }
    }
    if min_sites > 0 {
        let counts = &mut buffers.counts;
        counts.fill(0);
        for &label in labels {
            counts[label.get()] += 1;
        }
        let short = |label: usize| counts[label] > 0 && counts[label] < min_sites;
        if (0..q).any(short) {
            let stuck = labels.iter().enumerate().any(|(node, &label)| {
                short(label.get())
                    && !(0..q).any(|other| {
                        counts[other] >= min_sites
                            && graph.field[node * q + other] > f64::NEG_INFINITY
                    })
            });
            return if stuck { INFEASIBLE } else { BUDGET };
        }
    }
    if settled {
        CONVERGED
    } else {
        BUDGET
    }
}

/// The greedy whole-label merge in place, `potts_loop::merge_labels`'
/// rounds; the rounds run, the last finding no pair. `B[u, v]` sums the
/// couplings of `u`'s rows' entries at `v`. The gain of merging `u` into `v`
/// is `(B[u, v] + B[v, u]) / 2`, every coupling between the two once, or
/// with `halved` `B[u, v] / 2`, `u`'s rows alone halved: the downstream
/// `merge_assignment`'s. With `adjacent` a pair is admissible only where
/// `B[u, v] > 0`, as `merge_assignment` admits it. The full gain over every
/// pair is `merge_labels` bitwise.
pub fn merge_run<L: Label>(
    graph: &Graph<'_>,
    labels: &mut [L],
    halved: bool,
    adjacent: bool,
    buffers: &mut Buffers,
) -> usize {
    let q = graph.n_states;
    let Buffers {
        field_sums,
        bond_sums,
        admissible,
        counts,
        ..
    } = buffers;
    let mut rounds = 0usize;
    loop {
        rounds += 1;
        field_sums.fill(0.0);
        bond_sums.fill(0.0);
        admissible.fill(0);
        counts.fill(0);
        for (node, &label) in labels.iter().enumerate() {
            let u = label.get();
            counts[u] += 1;
            let row = &graph.field[node * q..(node + 1) * q];
            for (k, &value) in row.iter().enumerate() {
                field_sums[u * q + k] += value;
                admissible[u * q + k] += usize::from(value > f64::NEG_INFINITY);
            }
            for entry in graph.offsets[node] as usize..graph.offsets[node + 1] as usize {
                let far = labels[graph.neighbours[entry] as usize].get();
                bond_sums[u * q + far] += graph.couplings[entry];
            }
        }
        let mut chosen: Option<(usize, usize)> = None;
        let mut lowest = 0.0f64;
        for u in 0..q {
            if counts[u] == 0 {
                continue;
            }
            for v in 0..q {
                if v == u || counts[v] == 0 || admissible[u * q + v] != counts[u] {
                    continue;
                }
                if adjacent && bond_sums[u * q + v] <= 0.0 {
                    continue;
                }
                let gain = if halved {
                    bond_sums[u * q + v] / 2.0
                } else {
                    (bond_sums[u * q + v] + bond_sums[v * q + u]) / 2.0
                };
                let delta = -(field_sums[u * q + v] - field_sums[u * q + u]) - gain;
                if delta < lowest {
                    lowest = delta;
                    chosen = Some((u, v));
                }
            }
        }
        let Some((u, v)) = chosen else {
            return rounds;
        };
        for label in labels.iter_mut() {
            if label.get() == u {
                *label = L::put(v);
            }
        }
    }
}

/// The labels, narrow where every label fits a byte.
enum Labels {
    Narrow(Vec<u8>),
    Wide(Vec<u32>),
}

/// Dispatch `$body` over the label width, with `$graph`, `$labels`,
/// `$buffers` and `$colours` bound to the problem's views.
macro_rules! dispatch {
    ($problem:expr, |$graph:ident, $labels:ident, $buffers:ident, $colours:ident| $body:expr) => {{
        let problem = $problem;
        let $buffers = &mut *problem.buffers;
        let $colours: &[u32] = problem.colours;
        let $graph = Graph {
            field: &problem.held.field,
            n_states: problem.held.n_states,
            offsets: &problem.held.offsets,
            neighbours: &problem.held.neighbours,
            couplings: &problem.held.couplings,
            directed: problem.held.directed,
        };
        match &mut *problem.labels {
            Labels::Narrow($labels) => $body,
            Labels::Wide($labels) => $body,
        }
    }};
}

/// The held half of a problem, split from the labels so a solve borrows
/// both.
struct Held {
    n_nodes: usize,
    n_states: usize,
    offsets: Vec<u32>,
    neighbours: Vec<u32>,
    couplings: Vec<f64>,
    field: Vec<f64>,
    directed: bool,
}

/// One Potts problem: the adjacency, the field and the buffers, held
/// across solves. See the module docs.
#[pyclass(module = "sal.oxisal")]
pub struct PottsProblem {
    held: Held,
    labels: Labels,
    buffers: Buffers,
    colours: Vec<u32>,
    /// The expansion's network over the symmetric edges, laid out on the
    /// first `alpha_expansion` and kept.
    cut: Option<LatticeCut>,
    /// TRW-S's chain layout and messages, handed over on the first `trws`.
    chains: Option<Chains>,
}

/// What one `alpha_expansion` reports.
pub struct Expanded {
    /// Complete cycles over the labels.
    pub cycles: usize,
    /// Moves that lowered the energy.
    pub moves: usize,
    /// Whether a cycle lowered nothing.
    pub converged: bool,
    /// The composed ICM's sweeps and stop code, where one ran.
    pub icm: Option<(usize, u8)>,
}

/// The parts [`dispatch`] borrows at once.
struct Parts<'a> {
    held: &'a Held,
    labels: &'a mut Labels,
    buffers: &'a mut Buffers,
    colours: &'a [u32],
}

impl PottsProblem {
    /// Build from compressed rows and a `(n_nodes, n_states)` field; with
    /// `compact`, labels in a byte where `n_states <= 255`. With `directed`
    /// the rows are a directed adjacency: each descent reads a site's own
    /// row, and the energy halves every entry, so it scores the symmetric
    /// coupling `(A_ij + A_ji) / 2`.
    ///
    /// # Errors
    /// Returns `Err` for an adjacency that is not compressed rows over the
    /// sites, or one too large for `u32` indices.
    pub fn build(
        offsets: &[i64],
        neighbours: &[i64],
        couplings: &[f64],
        field: Vec<f64>,
        n_states: usize,
        compact: bool,
        directed: bool,
    ) -> Result<Self, String> {
        if offsets.is_empty() || n_states == 0 {
            return Err("a problem needs one site and one label".to_string());
        }
        let n_nodes = offsets.len() - 1;
        if field.len() != n_nodes * n_states {
            return Err(format!(
                "field has {} entries for {n_nodes} sites of {n_states} labels",
                field.len()
            ));
        }
        if neighbours.len() != couplings.len()
            || offsets[0] != 0
            || offsets[n_nodes] as usize != neighbours.len()
            || offsets.windows(2).any(|w| w[1] < w[0])
        {
            return Err("the adjacency is not compressed rows over the sites".to_string());
        }
        if neighbours.len() >= u32::MAX as usize || n_nodes >= u32::MAX as usize {
            return Err("the adjacency exceeds u32 indices".to_string());
        }
        if neighbours.iter().any(|&j| j < 0 || j as usize >= n_nodes) {
            return Err(format!("a neighbour lies outside [0, {n_nodes})"));
        }
        let narrow = compact && n_states <= 255;
        let colours = colour_order(
            &offsets.iter().map(|&o| o as u32).collect::<Vec<_>>(),
            &neighbours.iter().map(|&j| j as u32).collect::<Vec<_>>(),
        );
        Ok(Self {
            held: Held {
                n_nodes,
                n_states,
                offsets: offsets.iter().map(|&o| o as u32).collect(),
                neighbours: neighbours.iter().map(|&j| j as u32).collect(),
                couplings: couplings.to_vec(),
                field,
                directed,
            },
            labels: if narrow {
                Labels::Narrow(vec![0; n_nodes])
            } else {
                Labels::Wide(vec![0; n_nodes])
            },
            buffers: Buffers::new(n_nodes, n_states),
            colours,
            cut: None,
            chains: None,
        })
    }

    fn parts(&mut self) -> Parts<'_> {
        Parts {
            held: &self.held,
            labels: &mut self.labels,
            buffers: &mut self.buffers,
            colours: &self.colours,
        }
    }

    /// Copy `start` into the held labels.
    ///
    /// # Errors
    /// Returns `Err` for a wrong length or a label out of range.
    pub fn load(&mut self, start: &[i64]) -> Result<(), String> {
        let (n_nodes, n_states) = (self.held.n_nodes, self.held.n_states);
        if start.len() != n_nodes {
            return Err(format!(
                "start has {} sites, expected {n_nodes}",
                start.len()
            ));
        }
        if start.iter().any(|&s| s < 0 || s as usize >= n_states) {
            return Err(format!("a label lies outside [0, {n_states})"));
        }
        match &mut self.labels {
            Labels::Narrow(labels) => {
                for (slot, &s) in labels.iter_mut().zip(start) {
                    *slot = s as u8;
                }
            }
            Labels::Wide(labels) => {
                for (slot, &s) in labels.iter_mut().zip(start) {
                    *slot = s as u32;
                }
            }
        }
        Ok(())
    }

    /// The held labels as `int64`.
    #[must_use]
    pub fn unload(&self) -> Vec<i64> {
        match &self.labels {
            Labels::Narrow(labels) => labels.iter().map(|&s| i64::from(s)).collect(),
            Labels::Wide(labels) => labels.iter().map(|&s| i64::from(s)).collect(),
        }
    }

    /// The descent from the held labels; the sweeps, the stop code and the
    /// energy.
    ///
    /// # Errors
    /// As [`icm_run`].
    pub fn descend(&mut self, ask: &Descend, seed: u64) -> Result<(usize, u8, f64), String> {
        dispatch!(self.parts(), |graph, labels, buffers, colours| {
            let (sweeps, stop) = icm_run(&graph, labels, ask, buffers, colours, seed)?;
            Ok((sweeps, stop, graph.energy(labels)))
        })
    }

    /// The merge from the held labels; the rounds and the energy.
    pub fn merge_held(&mut self, halved: bool, adjacent: bool) -> (usize, f64) {
        dispatch!(self.parts(), |graph, labels, buffers, colours| {
            let _ = colours;
            let rounds = merge_run(&graph, labels, halved, adjacent, buffers);
            (rounds, graph.energy(labels))
        })
    }

    /// The energy of the held labels.
    pub fn held_energy(&mut self) -> f64 {
        dispatch!(self.parts(), |graph, labels, buffers, colours| {
            let _ = (buffers, colours);
            graph.energy(labels)
        })
    }

    /// Each site's first label of greatest field, into the held labels.
    pub fn argmax_held(&mut self) {
        let q = self.held.n_states;
        let field = &self.held.field;
        let best = |node: usize| {
            let row = &field[node * q..(node + 1) * q];
            let mut best = 0usize;
            for label in 1..q {
                if row[label] > row[best] {
                    best = label;
                }
            }
            best
        };
        match &mut self.labels {
            Labels::Narrow(labels) => {
                for (node, slot) in labels.iter_mut().enumerate() {
                    *slot = best(node) as u8;
                }
            }
            Labels::Wide(labels) => {
                for (node, slot) in labels.iter_mut().enumerate() {
                    *slot = best(node) as u32;
                }
            }
        }
    }

    /// The bytes each structure holds: adjacency, couplings, field, labels,
    /// buffers.
    #[must_use]
    pub fn bytes_held(&self) -> [usize; 7] {
        let labels = match &self.labels {
            Labels::Narrow(labels) => labels.capacity(),
            Labels::Wide(labels) => 4 * labels.capacity(),
        };
        [
            4 * (self.held.offsets.capacity() + self.held.neighbours.capacity()),
            8 * self.held.couplings.capacity(),
            8 * self.held.field.capacity(),
            labels,
            self.buffers.bytes() + 4 * self.colours.capacity(),
            self.cut.as_ref().map_or(0, LatticeCut::bytes),
            self.chains.as_ref().map_or(0, Chains::bytes),
        ]
    }

    /// Lay out the expansion's network over the symmetric edges
    /// `first[e] -- second[e]` at `coupling[e]`, kept for every later
    /// `alpha_expansion`.
    ///
    /// # Errors
    /// As [`LatticeCut::build`].
    pub fn set_cut(
        &mut self,
        first: &[usize],
        second: &[usize],
        coupling: &[f64],
    ) -> Result<(), String> {
        self.cut = Some(LatticeCut::build(
            self.held.n_nodes,
            first,
            second,
            coupling,
        )?);
        Ok(())
    }

    /// Whether the expansion's network is held.
    #[must_use]
    pub fn holds_cut(&self) -> bool {
        self.cut.is_some()
    }

    /// Hold TRW-S's chain layout.
    pub fn set_chains(&mut self, chains: Chains) {
        self.chains = Some(chains);
    }

    /// Whether TRW-S's chain layout is held.
    #[must_use]
    pub fn holds_chains(&self) -> bool {
        self.chains.is_some()
    }

    /// The field with each `-inf` replaced as `sim.potts.penalized` does:
    /// the row's least allowed log-weight less `1 + sum_e |J_e|`.
    ///
    /// # Errors
    /// Returns `Err` where a site allows no label.
    fn penalized(&self) -> Result<Option<Vec<f64>>, String> {
        let q = self.held.n_states;
        let field = &self.held.field;
        if field.iter().all(|v| v.is_finite()) {
            return Ok(None);
        }
        let margin = 1.0 + 0.5 * self.held.couplings.iter().map(|j| j.abs()).sum::<f64>();
        let mut values = field.clone();
        for (node, row) in values.chunks_mut(q).enumerate() {
            let least = row
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .fold(f64::INFINITY, f64::min);
            if least == f64::INFINITY {
                return Err(format!("site {node} allows no label"));
            }
            for value in row.iter_mut().filter(|v| !v.is_finite()) {
                *value = least - margin;
            }
        }
        Ok(Some(values))
    }

    /// Alpha-expansion from the held labels on the held network:
    /// `alpha_expansion`'s cycle, move for move. Each label's cut starts from
    /// the flow its last cut in this run ended on, and no flow is carried
    /// between runs, as a network laid out per run does. A move is kept where
    /// its energy change is negative; a cycle counts where the energy fell by
    /// more than `1e-12`. With `then_icm`, index-order ICM of at most
    /// `icm_iterations` sweeps follows in the same call.
    ///
    /// # Errors
    /// Returns `Err` where the network is not held or a site allows no label.
    pub fn expand_held(
        &mut self,
        max_iterations: usize,
        then_icm: bool,
        icm_iterations: usize,
    ) -> Result<Expanded, String> {
        let penalized = self.penalized()?;
        let (n_nodes, q) = (self.held.n_nodes, self.held.n_states);
        let Some(cut) = self.cut.as_mut() else {
            return Err("the expansion's network is not held".to_string());
        };
        cut.forget();
        let values: &[f64] = penalized.as_deref().unwrap_or(&self.held.field);
        let graph = Graph {
            field: values,
            n_states: q,
            offsets: &self.held.offsets,
            neighbours: &self.held.neighbours,
            couplings: &self.held.couplings,
            directed: self.held.directed,
        };
        let mut labels: Vec<usize> = match &self.labels {
            Labels::Narrow(held) => held.iter().map(|&s| s as usize).collect(),
            Labels::Wide(held) => held.iter().map(|&s| s as usize).collect(),
        };
        // `_infinite_capacity`: no cut pays it.
        let pinned = 1.0
            + values.iter().map(|v| v.abs()).sum::<f64>()
            + 0.5 * self.held.couplings.iter().sum::<f64>();
        let mut proposed = labels.clone();
        let mut changed: Vec<u32> = Vec::with_capacity(n_nodes);
        let mut moved = vec![false; n_nodes];
        let mut held = graph.energy(&labels);
        let mut current = held;
        let (mut moves, mut cycles, mut converged) = (0usize, 0usize, false);
        for cycle in 1..=max_iterations {
            cycles = cycle;
            let mut improved = false;
            for alpha in 0..q {
                cut.fill_expansion(values, q, &labels, alpha, pinned);
                let side = cut.solve(Some(alpha));
                changed.clear();
                for node in 0..n_nodes {
                    proposed[node] = if side[node] { labels[node] } else { alpha };
                    moved[node] = proposed[node] != labels[node];
                    if moved[node] {
                        changed.push(node as u32);
                    }
                }
                if !changed.is_empty() {
                    let change = cut.energy_change(values, q, &labels, &proposed, &changed, &moved);
                    let candidate = held + change;
                    if candidate < held {
                        std::mem::swap(&mut labels, &mut proposed);
                        held = candidate;
                    }
                    for &node in &changed {
                        moved[node as usize] = false;
                    }
                }
                if held < current - 1e-12 {
                    current = held;
                    improved = true;
                    moves += 1;
                }
            }
            if !improved {
                converged = true;
                break;
            }
        }
        match &mut self.labels {
            Labels::Narrow(held) => {
                for (slot, &s) in held.iter_mut().zip(&labels) {
                    *slot = s as u8;
                }
            }
            Labels::Wide(held) => {
                for (slot, &s) in held.iter_mut().zip(&labels) {
                    *slot = s as u32;
                }
            }
        }
        let icm = if then_icm {
            let ask = Descend {
                order: INDEX,
                min_sites: 0,
                policy: UNIFORM,
                floor_at: AT_SWEEP,
                max_iterations: icm_iterations,
                skip_clean: true,
            };
            let (sweeps, stop, _) = self.descend(&ask, 0)?;
            Some((sweeps, stop))
        } else {
            None
        };
        Ok(Expanded {
            cycles,
            moves,
            converged,
            icm,
        })
    }

    /// TRW-S on the held field and chain layout.
    ///
    /// # Errors
    /// Returns `Err` where the chain layout is not held.
    pub fn trws_held(&mut self, max_iterations: usize, tolerance: f64) -> Result<TrwsRun, String> {
        let q = self.held.n_states;
        let Some(chains) = self.chains.as_mut() else {
            return Err("the chain layout is not held".to_string());
        };
        Ok(chains.run(&self.held.field, q, max_iterations, tolerance))
    }

    /// The anneal and its polish on the held graph: [`potts_loop_impl`] on
    /// `int64` rows built for the call (the loop's kernels read them), and
    /// the polished best's stop code.
    ///
    /// # Errors
    /// As [`potts_loop_impl`].
    pub fn anneal_held(
        &mut self,
        state: &mut [i64],
        plan: &Plan<'_>,
        seed: u64,
    ) -> Result<(crate::potts_loop::Ran, u8), String> {
        let offsets: Vec<i64> = self.held.offsets.iter().map(|&o| i64::from(o)).collect();
        let neighbours: Vec<i64> = self.held.neighbours.iter().map(|&j| i64::from(j)).collect();
        let lattice = Lattice {
            field: &self.held.field,
            n_states: self.held.n_states,
            offsets: &offsets,
            neighbours: &neighbours,
            couplings: &self.held.couplings,
        };
        let ran = potts_loop_impl(state, &lattice, plan, seed)?;
        let stop = if plan.polish {
            let graph = Graph {
                field: &self.held.field,
                n_states: self.held.n_states,
                offsets: &self.held.offsets,
                neighbours: &self.held.neighbours,
                couplings: &self.held.couplings,
                directed: self.held.directed,
            };
            self.buffers.dirty.fill(true);
            status(
                &graph,
                &ran.best_polished,
                plan.min_sites,
                &mut self.buffers,
            )
        } else {
            BUDGET
        };
        Ok((ran, stop))
    }
}

fn field_rows(field: &PyReadonlyArray2<'_, f64>, n_nodes: usize) -> PyResult<(Vec<f64>, usize)> {
    let [rows, n_states] = *field.shape() else {
        unreachable!()
    };
    if rows != n_nodes || n_states == 0 {
        return Err(PyValueError::new_err(format!(
            "field has shape ({rows}, {n_states}) for {n_nodes} sites"
        )));
    }
    Ok((field.as_array().iter().copied().collect(), n_states))
}

#[pymethods]
impl PottsProblem {
    /// From `PottsGraph.compressed_adjacency()` and a `(n_nodes, n_states)`
    /// field; `compact=False` keeps `u32` labels (the general path, for
    /// measurement).
    #[new]
    #[pyo3(signature = (offsets, neighbours, couplings, field, compact = true, directed = false))]
    fn py_new(
        offsets: PyReadonlyArray1<'_, i64>,
        neighbours: PyReadonlyArray1<'_, i64>,
        couplings: PyReadonlyArray1<'_, f64>,
        field: PyReadonlyArray2<'_, f64>,
        compact: bool,
        directed: bool,
    ) -> PyResult<Self> {
        let offsets = offsets.as_slice()?;
        let (values, n_states) = field_rows(&field, offsets.len().saturating_sub(1))?;
        Self::build(
            offsets,
            neighbours.as_slice()?,
            couplings.as_slice()?,
            values,
            n_states,
            compact,
            directed,
        )
        .map_err(PyValueError::new_err)
    }

    /// Whether the rows are a directed adjacency.
    #[getter]
    fn directed(&self) -> bool {
        self.held.directed
    }

    /// Whether the expansion's network is held.
    #[getter]
    fn cut_held(&self) -> bool {
        self.holds_cut()
    }

    /// Whether TRW-S's chain layout is held.
    #[getter]
    fn chains_held(&self) -> bool {
        self.holds_chains()
    }

    /// Lay out the expansion's network over `first[e] -- second[e]` at
    /// `coupling[e]` (`PottsGraph.endpoints`), once.
    fn hold_cut(
        &mut self,
        first: PyReadonlyArray1<'_, i64>,
        second: PyReadonlyArray1<'_, i64>,
        coupling: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<()> {
        let first = crate::maxflow::node_indices(first.as_slice()?)?;
        let second = crate::maxflow::node_indices(second.as_slice()?)?;
        self.set_cut(&first, &second, coupling.as_slice()?)
            .map_err(PyValueError::new_err)
    }

    /// Alpha-expansion from `start`, then index ICM where `then_icm`: the
    /// labelling, its energy, the cycles, the moves, whether the expansion
    /// converged, and the ICM's sweeps and stop code (`0, 0` without it).
    #[pyo3(signature = (start, max_iterations, then_icm = false, icm_iterations = 200))]
    #[allow(clippy::type_complexity)]
    fn alpha_expansion<'py>(
        &mut self,
        py: Python<'py>,
        start: PyReadonlyArray1<'_, i64>,
        max_iterations: usize,
        then_icm: bool,
        icm_iterations: usize,
    ) -> PyResult<(
        Bound<'py, PyArray1<i64>>,
        f64,
        usize,
        usize,
        bool,
        usize,
        u8,
    )> {
        let start = start.as_slice()?;
        if start.is_empty() {
            // The field's argmax, `alpha_expansion`'s default start.
            self.argmax_held();
        } else {
            self.load(start).map_err(PyValueError::new_err)?;
        }
        let ran = py
            .detach(|| self.expand_held(max_iterations, then_icm, icm_iterations))
            .map_err(PyValueError::new_err)?;
        let value = self.held_energy();
        let (sweeps, stop) = ran.icm.unwrap_or((0, CONVERGED));
        Ok((
            self.unload().into_pyarray(py),
            value,
            ran.cycles,
            ran.moves,
            ran.converged,
            sweeps,
            stop,
        ))
    }

    /// Hold `sal.search.trws.chain_layout`'s arrays, `ends` flattened, once.
    #[allow(clippy::too_many_arguments)]
    fn hold_chains(
        &mut self,
        offsets: PyReadonlyArray1<'_, i64>,
        slots: PyReadonlyArray1<'_, i64>,
        neighbours: PyReadonlyArray1<'_, i64>,
        ends: PyReadonlyArray1<'_, i64>,
        coupling: PyReadonlyArray1<'_, f64>,
        weight: PyReadonlyArray1<'_, f64>,
        chain_offsets: PyReadonlyArray1<'_, i64>,
        chain_heads: PyReadonlyArray1<'_, i64>,
        chain_edges: PyReadonlyArray1<'_, i64>,
    ) -> PyResult<()> {
        let chains = Chains::build(
            self.held.n_nodes,
            self.held.n_states,
            offsets.as_slice()?,
            slots.as_slice()?,
            neighbours.as_slice()?,
            ends.as_slice()?,
            coupling.as_slice()?,
            weight.as_slice()?,
            chain_offsets.as_slice()?,
            chain_heads.as_slice()?,
            chain_edges.as_slice()?,
        )
        .map_err(PyValueError::new_err)?;
        self.set_chains(chains);
        Ok(())
    }

    /// TRW-S from zero messages: the kept labelling, its decoded energy, the
    /// bound trace and whether it converged.
    #[allow(clippy::type_complexity)]
    fn trws<'py>(
        &mut self,
        py: Python<'py>,
        max_iterations: usize,
        tolerance: f64,
    ) -> PyResult<(
        Bound<'py, PyArray1<i64>>,
        f64,
        Bound<'py, PyArray1<f64>>,
        bool,
    )> {
        if max_iterations < 1 || tolerance.is_nan() || tolerance < 0.0 {
            return Err(PyValueError::new_err(format!(
                "max_iterations must be >= 1 and tolerance >= 0, got {max_iterations} and {tolerance}"
            )));
        }
        let ran = py
            .detach(|| self.trws_held(max_iterations, tolerance))
            .map_err(PyValueError::new_err)?;
        Ok((
            ran.labelling.into_pyarray(py),
            ran.energy,
            ran.trace.into_pyarray(py),
            ran.converged,
        ))
    }

    /// Sites.
    #[getter]
    fn n_nodes(&self) -> usize {
        self.held.n_nodes
    }

    /// Labels per site.
    #[getter]
    fn n_states(&self) -> usize {
        self.held.n_states
    }

    /// Whether labels are held in a byte.
    #[getter]
    fn narrow_labels(&self) -> bool {
        matches!(self.labels, Labels::Narrow(_))
    }

    /// Bytes held per structure: adjacency, couplings, field, labels, buffers.
    fn footprint(&self) -> [usize; 7] {
        self.bytes_held()
    }

    /// Replace the field, same shape; the graph is kept.
    fn set_field(&mut self, field: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let (values, n_states) = field_rows(&field, self.held.n_nodes)?;
        if n_states != self.held.n_states {
            return Err(PyValueError::new_err(format!(
                "field has {n_states} labels, the problem {}",
                self.held.n_states
            )));
        }
        self.held.field.copy_from_slice(&values);
        Ok(())
    }

    /// The energy of `labelling`.
    fn energy(&mut self, labelling: PyReadonlyArray1<'_, i64>) -> PyResult<f64> {
        self.load(labelling.as_slice()?)
            .map_err(PyValueError::new_err)?;
        Ok(self.held_energy())
    }

    /// The field's argmax per site and its energy.
    fn argmax<'py>(&mut self, py: Python<'py>) -> (Bound<'py, PyArray1<i64>>, f64) {
        self.argmax_held();
        let value = self.held_energy();
        (self.unload().into_pyarray(py), value)
    }

    /// ICM from `start`: the labelling, its energy, the sweeps and the stop
    /// code.
    #[pyo3(signature = (start, order, min_sites, policy, floor_at, max_iterations, seed, skip_clean = true))]
    #[allow(clippy::too_many_arguments)]
    fn icm<'py>(
        &mut self,
        py: Python<'py>,
        start: PyReadonlyArray1<'_, i64>,
        order: u8,
        min_sites: usize,
        policy: u8,
        floor_at: u8,
        max_iterations: usize,
        seed: u64,
        skip_clean: bool,
    ) -> PyResult<(Bound<'py, PyArray1<i64>>, f64, usize, u8)> {
        self.load(start.as_slice()?)
            .map_err(PyValueError::new_err)?;
        let ask = Descend {
            order,
            min_sites,
            policy,
            floor_at,
            max_iterations,
            skip_clean,
        };
        let (sweeps, stop, value) = py
            .detach(|| self.descend(&ask, seed))
            .map_err(PyValueError::new_err)?;
        Ok((self.unload().into_pyarray(py), value, sweeps, stop))
    }

    /// The whole-label merge of `labelling`: the labelling, its energy and
    /// the rounds.
    fn merge<'py>(
        &mut self,
        py: Python<'py>,
        labelling: PyReadonlyArray1<'_, i64>,
        halved: bool,
        adjacent: bool,
    ) -> PyResult<(Bound<'py, PyArray1<i64>>, f64, usize)> {
        self.load(labelling.as_slice()?)
            .map_err(PyValueError::new_err)?;
        let (rounds, value) = py.detach(|| self.merge_held(halved, adjacent));
        Ok((self.unload().into_pyarray(py), value, rounds))
    }

    /// `potts_loop` on the held graph from `state`, moved in place: its dict,
    /// with `polish_stop` the polished best's stop code.
    #[pyo3(signature = (state, moves, temperatures, budget, n_main, seed, polish, min_sites, merge))]
    #[allow(clippy::too_many_arguments)]
    fn anneal<'py>(
        &mut self,
        py: Python<'py>,
        mut state: numpy::PyReadwriteArray1<'py, i64>,
        moves: Vec<u8>,
        temperatures: PyReadonlyArray1<'py, f64>,
        budget: u64,
        n_main: usize,
        seed: u64,
        polish: bool,
        min_sites: usize,
        merge: bool,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let state = state.as_slice_mut()?;
        if self.held.directed {
            return Err(PyValueError::new_err(
                "the anneal's conditional needs symmetric rows; this problem holds directed ones",
            ));
        }
        if state.len() != self.held.n_nodes {
            return Err(PyValueError::new_err("state has the wrong number of sites"));
        }
        let plan = Plan {
            moves: &moves,
            temperatures: temperatures.as_slice()?,
            budget,
            n_main,
            lead: 0,
            thin: 1,
            record: false,
            track_best: true,
            polish,
            min_sites,
            merge,
        };
        let (ran, stop) = py
            .detach(|| self.anneal_held(state, &plan, seed))
            .map_err(PyValueError::new_err)?;
        let out = ran_dict(py, ran)?;
        out.set_item("polish_stop", stop)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 4-site path 0-1-2-3 at unit coupling.
    fn path(field: Vec<f64>, q: usize) -> PottsProblem {
        let offsets = [0, 1, 3, 5, 6];
        let neighbours = [1, 0, 2, 1, 3, 2];
        PottsProblem::build(&offsets, &neighbours, &[1.0; 6], field, q, true, false).unwrap()
    }

    fn ask(order: u8, skip_clean: bool) -> Descend {
        Descend {
            order,
            min_sites: 0,
            policy: UNIFORM,
            floor_at: AT_SWEEP,
            max_iterations: 50,
            skip_clean,
        }
    }

    #[test]
    fn a_descent_aligns_a_path_with_its_field() {
        // Site 0 prefers label 1 by 3, more than its one bond; site 1 then
        // ties 1 against 1 and keeps the first maximum, label 0.
        let mut field = vec![0.0; 8];
        field[1] = 3.0;
        let mut problem = path(field, 2);
        for order in [INDEX, WORKLIST, CHECKERBOARD] {
            problem.load(&[0, 0, 0, 0]).unwrap();
            let (_, stop, energy) = problem.descend(&ask(order, true), 0).unwrap();
            assert_eq!(problem.unload(), vec![1, 0, 0, 0]);
            assert_eq!(stop, CONVERGED);
            // -3 field, two like bonds.
            assert_eq!(energy, -5.0);
        }
    }

    #[test]
    fn skipping_clean_sites_changes_nothing() {
        let field: Vec<f64> = (0..12).map(|i| f64::from((i * 7) % 5) * 0.3).collect();
        let mut problem = path(field, 3);
        let mut out = Vec::new();
        for skip in [false, true] {
            problem.load(&[2, 0, 1, 2]).unwrap();
            let ran = problem.descend(&ask(INDEX, skip), 0).unwrap();
            out.push((problem.unload(), ran));
        }
        assert_eq!(out[0], out[1]);
    }

    #[test]
    fn the_halved_gain_misses_a_merge_the_full_gain_takes() {
        // Labels 0 | 1 split the path at one unit bond; label 0's sites
        // forbid 1 by 5, label 1's prefer it by 0.3 each: full gain
        // -1 + 0.6 < 0 merges 1 into 0, half gain -0.5 + 0.6 > 0 does not.
        let field = vec![0.0, -5.0, 0.0, -5.0, 0.0, 0.3, 0.0, 0.3];
        let mut problem = path(field, 2);
        problem.load(&[0, 0, 1, 1]).unwrap();
        let (rounds, _) = problem.merge_held(false, false);
        assert_eq!((problem.unload(), rounds), (vec![0, 0, 0, 0], 2));
        problem.load(&[0, 0, 1, 1]).unwrap();
        let (rounds, _) = problem.merge_held(true, false);
        assert_eq!((problem.unload(), rounds), (vec![0, 0, 1, 1], 1));
    }

    #[test]
    fn a_floor_with_no_survivor_is_refused() {
        // Two sites pinned to each label: a 2-2 split no sweep moves.
        let field = vec![10.0, 0.0, 10.0, 0.0, 0.0, 10.0, 0.0, 10.0];
        let mut problem = path(field, 2);
        problem.load(&[0, 0, 1, 1]).unwrap();
        let mut floored = ask(INDEX, true);
        floored.min_sites = 3;
        let refused = problem.descend(&floored, 0).unwrap_err();
        assert!(refused.contains("left no state"));
    }
}
