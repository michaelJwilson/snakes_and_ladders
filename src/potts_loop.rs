//! The Potts anneal and sample loops in Rust: one call per run (#1368),
//! exposed to Python as `sal.oxisal.potts_loop`.
//!
//! **Why the loop and not only the moves.** `sample/loop.py` calls one step
//! per schedule entry, and each step pays a Python iteration, an FFI crossing
//! and a whole-lattice energy, about 1--3 us before the move does any work.
//! A single-site sweep or a Swendsen-Wang pass is O(N) and pays it once per
//! sweep; a Wolff cluster at high temperature is a few sites, so the
//! overhead is the step. Here the schedule, the composition, the `spent`
//! charge, the best-so-far and the recording run in one crossing with the
//! GIL released, and Python reads the result.
//!
//! **The moves are the kernels' own functions**, called as plain Rust:
//! [`single_site_sweeps_impl`], [`swendsen_wang_sweep_impl`] and
//! [`wolff_step`]. The first two take their uniforms as arrays and a guard
//! that hands a near-tie back to the NumPy oracle; this loop owns its stream,
//! so it passes a zero guard and decides an exact tie by redrawing that one
//! uniform. A tie is a draw equal to a boundary, probability about `2^-53`
//! per site, and redrawing it conditions the uniform on missing a point of
//! that mass.
//!
//! **Randomness** is one ChaCha8 seed from the run's generator, as
//! `wolff.rs` takes it, so a seeded generator determines the run. The chain
//! is of the Python loop's law and is not its chain: the referee is the
//! enumerated law and, for `spent` and the schedule index, the Python loop
//! replayed on the charges this one returns.

use numpy::{
    PyArray1, PyReadonlyArray1, PyReadonlyArrayDyn, PyReadwriteArray1, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardUniform, Uniform};

use crate::icm::{icm_sweeps_impl, no_survivor, IcmScratch};
use crate::potts::{conditional, single_site_sweeps_impl, swendsen_wang_sweep_impl};
use crate::wolff::{wolff_step, Lattice, WolffDraw, WolffScratch};

/// A heat-bath (Glauber) sweep over every site in index order.
pub const SINGLE_SITE: u8 = 0;
/// A Swendsen-Wang pass with a uniform label proposal and a field accept.
pub const SWENDSEN_WANG: u8 = 1;
/// A Wolff step with a uniform label proposal and a field accept.
pub const WOLFF: u8 = 2;
/// A Wolff step drawing its cluster's label from the field weight.
pub const WOLFF_HEAT_BATH: u8 = 3;

/// The schedule and the recording rule of one run.
pub struct Plan<'a> {
    /// The moves of one step, in composition order.
    pub moves: &'a [u8],
    /// The temperature per schedule entry.
    pub temperatures: &'a [f64],
    /// Site visits the main phase spends; `0` runs `n_main` steps instead.
    pub budget: u64,
    /// Steps of the main phase without a budget.
    pub n_main: usize,
    /// Steps before the main phase at `temperatures[0]`: neither charged to
    /// the budget nor recorded (a chain's burn-in).
    pub lead: usize,
    /// Record every `thin`-th main step, the `thin`-th the first.
    pub thin: usize,
    /// Whether to keep the recorded states.
    pub record: bool,
    /// Whether to keep the lowest-energy state.
    pub track_best: bool,
    /// Whether to descend the final state, and the best where kept, by ICM
    /// to a fixed point once the schedule ends (`Polish.ICM`, #1363).
    pub polish: bool,
    /// The polish's floor, `iterated_conditional_modes`' `min_sites`.
    pub min_sites: usize,
}

/// What one run returns beside the final state, which it leaves in place.
#[derive(Debug, Default, PartialEq)]
pub struct Ran {
    /// The lowest energy visited, the start included, and its state.
    pub best_energy: f64,
    /// The best state, `n_nodes` long; empty when not tracked.
    pub best: Vec<i64>,
    /// The energy the run ended at.
    pub final_energy: f64,
    /// Site visits of the main phase, and of every step.
    pub spent_main: u64,
    /// Site visits of every step, the lead included.
    pub spent_total: u64,
    /// Main-phase steps run.
    pub n_main: usize,
    /// Steps that changed the labelling, the lead included.
    pub accepted: usize,
    /// Sites over the steps that built a single cluster, their count and the
    /// largest: `PottsChain.mean_cluster_size`'s inputs.
    pub cluster_total: u64,
    /// Steps whose clusters held at least one site.
    pub cluster_count: usize,
    /// The largest step's cluster sites.
    pub largest: usize,
    /// The recorded states, flattened row-major.
    pub records: Vec<i64>,
    /// The schedule index each main step read.
    pub indices: Vec<u32>,
    /// The site visits each main step charged.
    pub charges: Vec<u64>,
    /// The best state descended by the polish; empty unpolished.
    pub best_polished: Vec<i64>,
    /// The polish's sweeps from the final state and from the best.
    pub polish_sweeps: (usize, usize),
}

/// `E(s) = -sum_i h_i[s_i] - sum_(ij) J_ij [s_i == s_j]`, from the
/// compressed adjacency, where each edge appears from both ends.
fn energy_of(state: &[i64], lattice: &Lattice<'_>) -> f64 {
    let n_states = lattice.n_states;
    let mut field = 0.0f64;
    let mut bonds = 0.0f64;
    for (node, &label) in state.iter().enumerate() {
        field += lattice.field[node * n_states + label as usize];
        let (lo, hi) = (
            lattice.offsets[node] as usize,
            lattice.offsets[node + 1] as usize,
        );
        for position in lo..hi {
            if state[lattice.neighbours[position] as usize] == label {
                bonds += lattice.couplings[position];
            }
        }
    }
    -field - 0.5 * bonds
}

/// Run one anneal or chain on `state`, in place; see the module docs.
///
/// # Errors
/// Returns `Err` for an empty move set or schedule, an unknown move code, a
/// non-positive temperature, `thin = 0`, a step charging nothing under a
/// budget, a malformed adjacency, and whatever the kernels refuse.
pub fn potts_loop_impl(
    state: &mut [i64],
    lattice: &Lattice<'_>,
    plan: &Plan<'_>,
    seed: u64,
) -> Result<Ran, String> {
    let n_nodes = state.len();
    let n_states = lattice.n_states;
    let (offsets, neighbours, couplings) = (lattice.offsets, lattice.neighbours, lattice.couplings);
    if plan.moves.is_empty() || plan.temperatures.is_empty() {
        return Err("a run needs a move and a temperature".to_string());
    }
    if let Some(&code) = plan.moves.iter().find(|&&code| code > WOLFF_HEAT_BATH) {
        return Err(format!("move code {code} is not one this loop runs"));
    }
    if let Some(&t) = plan
        .temperatures
        .iter()
        .find(|&&t| !(t > 0.0 && t.is_finite()))
    {
        return Err(format!("a temperature is positive and finite, got {t}"));
    }
    if plan.thin == 0 {
        return Err("thin is at least 1".to_string());
    }
    if n_nodes == 0 || n_states == 0 || lattice.field.len() != n_nodes * n_states {
        return Err(format!(
            "field has {} entries for {n_nodes} sites of {n_states} labels",
            lattice.field.len()
        ));
    }
    if offsets.len() != n_nodes + 1 || neighbours.len() != couplings.len() {
        return Err("the adjacency is not compressed rows over the sites".to_string());
    }
    let offsets_usize: Vec<usize> = offsets
        .iter()
        .map(|&o| usize::try_from(o).map_err(|_| "offsets must be >= 0".to_string()))
        .collect::<Result<_, _>>()?;
    if offsets_usize.windows(2).any(|w| w[1] < w[0]) || offsets_usize[n_nodes] != neighbours.len() {
        return Err("offsets are not non-decreasing bounds of the adjacency".to_string());
    }
    if neighbours.iter().any(|&j| j < 0 || j as usize >= n_nodes) {
        return Err(format!("a neighbour lies outside [0, {n_nodes})"));
    }
    if state.iter().any(|&s| s < 0 || s as usize >= n_states) {
        return Err(format!("a label lies outside [0, {n_states})"));
    }

    // `step_visits`' charges, in its integer arithmetic.
    let incident = neighbours.len() as u64;
    let per_sweep = n_nodes as u64 + incident;
    let per_cluster_site = 1 + incident / n_nodes as u64;

    // The edges once, each from its lower end, for the Swendsen-Wang pass.
    let has_sw = plan.moves.contains(&SWENDSEN_WANG);
    let mut edges: Vec<i64> = Vec::new();
    let mut edge_couplings: Vec<f64> = Vec::new();
    if has_sw {
        for node in 0..n_nodes {
            for position in offsets_usize[node]..offsets_usize[node + 1] {
                let far = neighbours[position];
                if far as usize > node {
                    edges.extend([node as i64, far]);
                    edge_couplings.push(couplings[position]);
                }
            }
        }
    }
    let n_edges = edge_couplings.len();

    // Every buffer once per run.
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let wolff_draw = WolffDraw::new(n_nodes, n_states)?;
    let colours = Uniform::new(0, n_states as i64).map_err(|e| e.to_string())?;
    let mut scratch = WolffScratch::new(n_nodes, n_states);
    let mut members = vec![0i64; n_nodes];
    let mut draws = vec![0.0f64; n_nodes];
    let (mut scaled, mut bond_probability, mut bond_draws) = if has_sw {
        (
            vec![0.0f64; n_nodes * n_states],
            vec![0.0f64; n_edges],
            vec![0.0f64; n_edges],
        )
    } else {
        (Vec::new(), Vec::new(), Vec::new())
    };
    let mut colour_draws = vec![0i64; if has_sw { n_nodes } else { 0 }];
    let mut accept_draws = vec![0.0f64; if has_sw { n_nodes } else { 0 }];
    let mut labels = vec![0i64; if has_sw { n_nodes } else { 0 }];
    let compare = plan.moves.len() > 1 || plan.moves[0] <= SWENDSEN_WANG;
    let mut before = vec![0i64; if compare { n_nodes } else { 0 }];

    let mut energy = energy_of(state, lattice);
    let mut ran = Ran {
        best_energy: energy,
        best: if plan.track_best {
            state.to_vec()
        } else {
            Vec::new()
        },
        ..Ran::default()
    };
    let n_temperatures = plan.temperatures.len();
    let mut step: usize = 0;
    loop {
        let main = step >= plan.lead;
        let index = step.saturating_sub(plan.lead);
        if main {
            let more = if plan.budget > 0 {
                ran.spent_main < plan.budget
            } else {
                index < plan.n_main
            };
            if !more {
                break;
            }
        }
        let at = if !main {
            0
        } else if plan.budget > 0 {
            let fraction =
                u128::from(ran.spent_main) * n_temperatures as u128 / u128::from(plan.budget);
            (n_temperatures - 1).min(fraction as usize)
        } else {
            (n_temperatures - 1).min(index)
        };
        let beta = 1.0 / plan.temperatures[at];
        if compare {
            before.copy_from_slice(state);
        }
        let (mut visits, mut sites, mut changed) = (0u64, 0usize, false);
        for &code in plan.moves {
            match code {
                SINGLE_SITE => {
                    for draw in draws.iter_mut() {
                        *draw = StandardUniform.sample(&mut rng);
                    }
                    let mut first = 0usize;
                    loop {
                        let stop = single_site_sweeps_impl(
                            state,
                            lattice.field,
                            n_states,
                            &offsets_usize,
                            neighbours,
                            couplings,
                            &draws,
                            1,
                            beta,
                            0.0,
                            first,
                        )?;
                        if stop == n_nodes {
                            break;
                        }
                        // An exact tie with a boundary: redraw that uniform.
                        draws[stop] = StandardUniform.sample(&mut rng);
                        first = stop;
                    }
                    energy = energy_of(state, lattice);
                    visits += per_sweep;
                }
                SWENDSEN_WANG => {
                    for (s, &h) in scaled.iter_mut().zip(lattice.field) {
                        *s = beta * h;
                    }
                    for (p, &j) in bond_probability.iter_mut().zip(&edge_couplings) {
                        *p = -(-beta * j).exp_m1();
                    }
                    for draw in bond_draws.iter_mut() {
                        *draw = StandardUniform.sample(&mut rng);
                    }
                    for colour in colour_draws.iter_mut() {
                        *colour = colours.sample(&mut rng);
                    }
                    for draw in accept_draws.iter_mut() {
                        *draw = StandardUniform.sample(&mut rng);
                    }
                    let mut first = 0usize;
                    loop {
                        let (n_clusters, stop) = swendsen_wang_sweep_impl(
                            state,
                            &scaled,
                            n_states,
                            &edges,
                            &bond_probability,
                            &bond_draws,
                            &colour_draws,
                            &accept_draws,
                            &mut labels,
                            0.0,
                            first,
                        )?;
                        if stop == n_clusters {
                            break;
                        }
                        accept_draws[stop] = StandardUniform.sample(&mut rng);
                        first = stop;
                    }
                    energy = energy_of(state, lattice);
                    visits += per_sweep;
                }
                _ => {
                    let (len, _, accept, from) = wolff_step(
                        state,
                        lattice,
                        beta,
                        code == WOLFF_HEAT_BATH,
                        -1,
                        -1,
                        &mut rng,
                        &wolff_draw,
                        &mut scratch,
                        &mut members,
                    )?;
                    if accept {
                        changed = true;
                        let to = state[members[0] as usize];
                        energy = if energy.is_finite() {
                            energy + recoloured(state, lattice, &members[..len], &scratch, from, to)
                        } else {
                            energy_of(state, lattice)
                        };
                    }
                    sites += len;
                    visits += len as u64 * per_cluster_site;
                }
            }
        }
        if plan.budget > 0 && visits == 0 {
            return Err(
                "a step charged nothing, so a budget in its unit is never spent".to_string(),
            );
        }
        if compare {
            changed = before != state;
        }
        ran.accepted += usize::from(changed);
        if sites > 0 {
            ran.cluster_total += sites as u64;
            ran.cluster_count += 1;
            ran.largest = ran.largest.max(sites);
        }
        ran.spent_total += visits;
        if plan.track_best && energy < ran.best_energy {
            ran.best_energy = energy;
            ran.best.copy_from_slice(state);
        }
        if main {
            ran.spent_main += visits;
            ran.indices.push(at as u32);
            ran.charges.push(visits);
            if plan.record && (index + 1).is_multiple_of(plan.thin) {
                ran.records.extend_from_slice(state);
            }
            ran.n_main += 1;
        }
        step += 1;
    }
    ran.final_energy = energy;
    if plan.polish {
        let mut polish = Polisher::new(n_nodes, n_states, plan.min_sites);
        ran.polish_sweeps.0 = polish.descend(
            state,
            lattice.field,
            &offsets_usize,
            neighbours,
            couplings,
            &mut rng,
        )?;
        if plan.track_best {
            ran.best_polished = ran.best.clone();
            ran.polish_sweeps.1 = polish.descend(
                &mut ran.best_polished,
                lattice.field,
                &offsets_usize,
                neighbours,
                couplings,
                &mut rng,
            )?;
        }
    }
    Ok(ran)
}

/// The polish's ICM descent, in index order to a fixed point (#1363), on the
/// run's stream for the floor's uniforms.
struct Polisher {
    scratch: IcmScratch,
    draws: Vec<f64>,
    chunk: usize,
    min_sites: usize,
}

impl Polisher {
    fn new(n_nodes: usize, n_states: usize, min_sites: usize) -> Self {
        // Chunks of `n_nodes` sweeps unfloored, as `chains._descend` runs
        // them; a floor draws `chunk * n_nodes` uniforms a chunk, so 16.
        let chunk = if min_sites > 0 { 16 } else { n_nodes.max(1) };
        Self {
            scratch: IcmScratch::new(n_states),
            draws: vec![0.0; if min_sites > 0 { chunk * n_nodes } else { 0 }],
            chunk,
            min_sites,
        }
    }

    /// Descend `state` until a sweep changes nothing, or a chunk ends on a
    /// labelling no sweep can change (settled, or a floor only a forbidden
    /// label could meet); the sweeps run.
    fn descend(
        &mut self,
        state: &mut [i64],
        field: &[f64],
        offsets: &[usize],
        neighbours: &[i64],
        couplings: &[f64],
        rng: &mut ChaCha8Rng,
    ) -> Result<usize, String> {
        let mut swept = 0usize;
        loop {
            for draw in self.draws.iter_mut() {
                *draw = StandardUniform.sample(&mut *rng);
            }
            let sweeps = icm_sweeps_impl(
                state,
                field,
                offsets,
                neighbours,
                couplings,
                &[],
                &self.draws,
                self.chunk,
                true,
                self.min_sites,
                &mut self.scratch,
            )
            .map_err(|message| {
                // The sweep the message names counts every chunk's.
                if message.contains("left no state") {
                    let sweep = swept + message_sweep(&message);
                    no_survivor(sweep, self.min_sites)
                } else {
                    message
                }
            })?;
            swept += sweeps;
            if sweeps < self.chunk
                || self.unmovable(state, field, offsets, neighbours, couplings)?
            {
                return Ok(swept);
            }
        }
    }

    /// Whether a sweep would leave `state` as it is, or a site below the
    /// floor allows no surviving label: `icm._descended`'s two stops.
    fn unmovable(
        &mut self,
        state: &[i64],
        field: &[f64],
        offsets: &[usize],
        neighbours: &[i64],
        couplings: &[f64],
    ) -> Result<bool, String> {
        let n_states = self.scratch.local.len();
        let mut local = vec![0.0f64; n_states];
        let mut settled = true;
        for node in 0..state.len() {
            conditional(
                &mut local, field, state, offsets, neighbours, couplings, node,
            )?;
            let mut best = 0usize;
            for label in 1..n_states {
                if local[label] > local[best] {
                    best = label;
                }
            }
            if best as i64 != state[node] {
                settled = false;
                break;
            }
        }
        if self.min_sites == 0 {
            return Ok(settled);
        }
        let mut counts = vec![0usize; n_states];
        for &label in state {
            counts[label as usize] += 1;
        }
        let below = |label: usize| counts[label] > 0 && counts[label] < self.min_sites;
        if !(0..n_states).any(below) {
            return Ok(settled);
        }
        Ok(state.iter().enumerate().any(|(node, &label)| {
            below(label as usize)
                && !(0..n_states).any(|other| {
                    counts[other] >= self.min_sites
                        && field[node * n_states + other] > f64::NEG_INFINITY
                })
        }))
    }
}

/// The sweep [`no_survivor`]'s message names.
fn message_sweep(message: &str) -> usize {
    message
        .split_whitespace()
        .nth(1)
        .and_then(|word| word.parse().ok())
        .unwrap_or(0)
}

/// The energy change of recolouring `cluster` from `from` to `to`, read after
/// the step: the field over the members and the bonds leaving the cluster,
/// the bonds inside it unchanged. O(cluster), where a full energy is O(N).
fn recoloured(
    state: &[i64],
    lattice: &Lattice<'_>,
    cluster: &[i64],
    scratch: &WolffScratch,
    from: i64,
    to: i64,
) -> f64 {
    let n_states = lattice.n_states;
    let mut change = 0.0f64;
    for &member in cluster {
        let member = member as usize;
        let row = member * n_states;
        change -= lattice.field[row + to as usize] - lattice.field[row + from as usize];
        let (lo, hi) = (
            lattice.offsets[member] as usize,
            lattice.offsets[member + 1] as usize,
        );
        for position in lo..hi {
            let far = lattice.neighbours[position] as usize;
            if scratch.stamp[far] == scratch.epoch {
                continue;
            }
            let label = state[far];
            let coupling = lattice.couplings[position];
            change -=
                coupling * (f64::from(u8::from(label == to)) - f64::from(u8::from(label == from)));
        }
    }
    change
}

/// PyO3 boundary for [`potts_loop_impl`]: the final state in place, and a
/// dict of the rest, with the GIL released for the run.
#[pyfunction]
#[pyo3(signature = (state, field, offsets, neighbours, couplings, moves, temperatures, budget, n_main, lead, thin, record, track_best, seed, polish = false, min_sites = 0))]
#[allow(clippy::too_many_arguments)]
pub fn potts_loop<'py>(
    py: Python<'py>,
    mut state: PyReadwriteArray1<'py, i64>,
    field: PyReadonlyArrayDyn<'py, f64>,
    offsets: PyReadonlyArray1<'py, i64>,
    neighbours: PyReadonlyArray1<'py, i64>,
    couplings: PyReadonlyArray1<'py, f64>,
    moves: Vec<u8>,
    temperatures: PyReadonlyArray1<'py, f64>,
    budget: u64,
    n_main: usize,
    lead: usize,
    thin: usize,
    record: bool,
    track_best: bool,
    seed: u64,
    polish: bool,
    min_sites: usize,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let [n_rows, n_states] = *field.shape() else {
        return Err(PyValueError::new_err(format!(
            "field must be 2-D, (n_nodes, n_states), one row per site; got shape {:?}",
            field.shape()
        )));
    };
    if n_rows != state.len() {
        return Err(PyValueError::new_err(format!(
            "field has {n_rows} rows and state has {} sites",
            state.len()
        )));
    }
    let state = state.as_slice_mut()?;
    let lattice = Lattice {
        field: field.as_slice()?,
        n_states,
        offsets: offsets.as_slice()?,
        neighbours: neighbours.as_slice()?,
        couplings: couplings.as_slice()?,
    };
    let plan = Plan {
        moves: &moves,
        temperatures: temperatures.as_slice()?,
        budget,
        n_main,
        lead,
        thin,
        record,
        track_best,
        polish,
        min_sites,
    };
    let ran = py
        .detach(|| potts_loop_impl(state, &lattice, &plan, seed))
        .map_err(PyValueError::new_err)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("best", PyArray1::from_vec(py, ran.best))?;
    out.set_item("best_energy", ran.best_energy)?;
    out.set_item("final_energy", ran.final_energy)?;
    out.set_item("spent_main", ran.spent_main)?;
    out.set_item("spent_total", ran.spent_total)?;
    out.set_item("n_main", ran.n_main)?;
    out.set_item("accepted", ran.accepted)?;
    out.set_item("cluster_total", ran.cluster_total)?;
    out.set_item("cluster_count", ran.cluster_count)?;
    out.set_item("largest", ran.largest)?;
    out.set_item("records", PyArray1::from_vec(py, ran.records))?;
    out.set_item("indices", PyArray1::from_vec(py, ran.indices))?;
    out.set_item("charges", PyArray1::from_vec(py, ran.charges))?;
    out.set_item("best_polished", PyArray1::from_vec(py, ran.best_polished))?;
    out.set_item("polish_sweeps", ran.polish_sweeps)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2x2 periodic-free square, 0-1, 0-2, 1-3, 2-3, unit couplings.
    fn square(n_states: usize) -> (Vec<i64>, Vec<i64>, Vec<f64>, Vec<f64>) {
        let offsets = vec![0, 2, 4, 6, 8];
        let neighbours = vec![1, 2, 0, 3, 0, 3, 1, 2];
        (offsets, neighbours, vec![1.0; 8], vec![0.0; 4 * n_states])
    }

    fn plan<'a>(moves: &'a [u8], temperatures: &'a [f64], budget: u64, n_main: usize) -> Plan<'a> {
        Plan {
            moves,
            temperatures,
            budget,
            n_main,
            lead: 0,
            thin: 1,
            record: true,
            track_best: true,
            polish: false,
            min_sites: 0,
        }
    }

    #[test]
    fn the_energy_is_the_hand_count() {
        let (offsets, neighbours, couplings, mut field) = square(2);
        field[1] = 0.5; // site 0 prefers label 1 by 0.5
        let lattice = Lattice {
            field: &field,
            n_states: 2,
            offsets: &offsets,
            neighbours: &neighbours,
            couplings: &couplings,
        };
        // Labels (1, 1, 0, 0): bonds 0-1 and 2-3 agree, field 0.5 at site 0.
        assert_eq!(energy_of(&[1, 1, 0, 0], &lattice), -2.5);
    }

    #[test]
    fn a_fixed_cost_budget_reads_the_schedule_index_per_step() {
        let (offsets, neighbours, couplings, field) = square(3);
        let lattice = Lattice {
            field: &field,
            n_states: 3,
            offsets: &offsets,
            neighbours: &neighbours,
            couplings: &couplings,
        };
        let temperatures = [3.0, 2.0, 1.0, 0.5];
        let mut state = vec![0, 1, 2, 0];
        // A sweep costs 4 + 8 = 12; a budget of 4 * 12 runs index k at step k.
        let ran = potts_loop_impl(
            &mut state,
            &lattice,
            &plan(&[SINGLE_SITE], &temperatures, 48, 0),
            7,
        )
        .unwrap();
        assert_eq!(ran.indices, vec![0, 1, 2, 3]);
        assert_eq!(ran.spent_main, 48);
        assert_eq!(ran.records.len(), 16);
    }

    #[test]
    fn the_tracked_energies_match_a_full_recount() {
        let (offsets, neighbours, couplings, mut field) = square(3);
        field[4] = 0.7;
        field[9] = f64::NEG_INFINITY;
        let lattice = Lattice {
            field: &field,
            n_states: 3,
            offsets: &offsets,
            neighbours: &neighbours,
            couplings: &couplings,
        };
        let temperatures = [2.0, 1.0, 0.5];
        for moves in [&[WOLFF][..], &[WOLFF_HEAT_BATH], &[SWENDSEN_WANG, WOLFF]] {
            let mut state = vec![0, 1, 2, 0];
            let ran = potts_loop_impl(
                &mut state,
                &lattice,
                &plan(moves, &temperatures, 0, 200),
                11,
            )
            .unwrap();
            let recount = energy_of(&state, &lattice);
            assert!((ran.final_energy - recount).abs() < 1e-12);
            assert!((ran.best_energy - energy_of(&ran.best, &lattice)).abs() < 1e-12);
            assert!(ran.best_energy <= ran.final_energy);
        }
    }

    #[test]
    fn one_seed_is_one_run_and_another_seed_another() {
        let (offsets, neighbours, couplings, field) = square(3);
        let lattice = Lattice {
            field: &field,
            n_states: 3,
            offsets: &offsets,
            neighbours: &neighbours,
            couplings: &couplings,
        };
        let temperatures = [1.0];
        let run = |seed: u64| {
            let mut state = vec![0, 1, 2, 0];
            potts_loop_impl(
                &mut state,
                &lattice,
                &plan(&[WOLFF, SINGLE_SITE], &temperatures, 0, 50),
                seed,
            )
            .unwrap()
        };
        assert_eq!(run(3), run(3));
        assert_ne!(run(3).records, run(4).records);
    }

    #[test]
    fn an_unknown_move_is_refused() {
        let (offsets, neighbours, couplings, field) = square(2);
        let lattice = Lattice {
            field: &field,
            n_states: 2,
            offsets: &offsets,
            neighbours: &neighbours,
            couplings: &couplings,
        };
        let mut state = vec![0, 1, 0, 1];
        assert!(potts_loop_impl(&mut state, &lattice, &plan(&[9], &[1.0], 0, 1), 0).is_err());
    }
}
