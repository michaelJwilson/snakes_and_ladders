//! Wolff single-cluster moves for the Potts model, both recolourings, ported
//! from `python/sal/sample/potts_mcmc/sweeps.py::wolff_sweep` and
//! `::wolff_heat_bath_sweep` (the oracles) and exposed to Python as
//! `sal.oxisal.wolff_sweeps`.
//!
//! **Why this one is a Rust port.** Issue #1362 measured the Python move at
//! 5.4 ms a step on a 64x64 lattice near the transition, about 745 ns per
//! site visit, against ~21 ns for the compiled Swendsen-Wang pass: the
//! cluster is grown one Python-level neighbour at a time, which is the
//! control flow over an adjacency root `CLAUDE.md` reserves this backend for.
//!
//! **The generator is the run's, through one seed.** The number of uniforms a
//! cluster consumes depends on the cluster, so the draws cannot be made in
//! Python as arrays the way `potts.rs` takes them without drawing one per
//! incident edge of the whole graph each step. The kernel takes instead one
//! draw of the run's `numpy.random.Generator` as a ChaCha8 seed, the protocol
//! `sample.chain._seed` states for the compiled chains (`chain.rs`): a seeded
//! generator still determines the result. The chain is therefore of the
//! oracle's law and is not the oracle's chain, and the referee is the
//! enumerated law, as for Swendsen-Wang (#754).
//!
//! **No per-step allocation.** The cluster's members are the work queue: the
//! grow loop reads `members` from the front and appends at the back, so one
//! `n_nodes` buffer the caller owns holds the queue and, after the step, the
//! cluster. A visited stamp per site, one epoch per step, replaces clearing.
//! Validation is lazy, per site and edge the cluster reaches, so a step costs
//! O(cluster) rather than O(graph).

use numpy::{PyReadonlyArray1, PyReadonlyArrayDyn, PyReadwriteArray1, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardUniform, Uniform};

/// The compressed adjacency and field one batch of Wolff steps reads.
pub struct Lattice<'a> {
    /// The unscaled log weights, `n_nodes * n_states`, row-major.
    pub field: &'a [f64],
    /// The alphabet size.
    pub n_states: usize,
    /// `n_nodes + 1` row bounds into `neighbours` and `couplings`.
    pub offsets: &'a [i64],
    /// The far site of each incident edge.
    pub neighbours: &'a [i64],
    /// Each incident edge's coupling; a negative one reached is refused.
    pub couplings: &'a [f64],
}

/// What one batch of steps reports, one entry per step.
pub struct Outcomes<'a> {
    /// The cluster's members; after the batch, the last step's cluster in
    /// `members[..sizes[last]]`. Length `n_nodes`.
    pub members: &'a mut [i64],
    /// The size of each step's cluster.
    pub sizes: &'a mut [i64],
    /// Whether the step proposed a label other than the cluster's own.
    pub proposals: &'a mut [bool],
    /// Whether the cluster's label changed.
    pub accepts: &'a mut [bool],
}

/// One label drawn with probability proportional to `exp(beta * sums[c])`,
/// by inversion: the weights are taken relative to the largest, so no `exp`
/// can overflow and none underflows the drawn label away, and one uniform
/// scaled by their total is walked through the cumulative sum. `sums` is
/// overwritten with those weights, so the callers' per-cluster scratch is
/// the draw's. A forbidden label (`-inf`) has weight zero and is never
/// drawn; a cluster forbidding every label keeps `current` and draws
/// nothing (#1146). One uniform per cluster where #1362 drew one per label
/// by Gumbel-max, two logarithms each: that was ~70% of the heat-bath
/// Swendsen-Wang pass at 64x64 near `beta_c` (#1364). Shared by the Wolff
/// and Swendsen-Wang heat-bath kernels.
#[inline]
pub(crate) fn heat_bath_label(
    sums: &mut [f64],
    beta: f64,
    current: usize,
    rng: &mut ChaCha8Rng,
) -> usize {
    let mut top = f64::NEG_INFINITY;
    for s in sums.iter_mut() {
        // A forbidden label stays forbidden at `beta = 0`, where the
        // product would be `NaN`.
        if *s != f64::NEG_INFINITY {
            *s *= beta;
        }
        top = top.max(*s);
    }
    if top == f64::NEG_INFINITY {
        return current;
    }
    let mut total = 0.0f64;
    for s in sums.iter_mut() {
        // The largest weight is one exactly; only the others pay an `exp`.
        *s = if *s == top { 1.0 } else { (*s - top).exp() };
        total += *s;
    }
    let uniform: f64 = StandardUniform.sample(rng);
    let mut remaining = uniform * total;
    // The last allowed label takes what rounding leaves past the total.
    let mut last = current;
    for (c, &weight) in sums.iter().enumerate() {
        if weight > 0.0 {
            if remaining < weight {
                return c;
            }
            remaining -= weight;
            last = c;
        }
    }
    last
}

/// Run `sizes.len()` Wolff steps on `state`, in place.
///
/// Each step grows one cluster from `root` (or a uniform seed where `root`
/// is negative) through like neighbours, bonding each like edge reached from
/// outside the cluster with probability `1 - exp(-beta J)`, and recolours it.
/// `heat_bath = false` is `_recolour`: a uniform label (or `proposed` where it
/// is non-negative) accepted on the field difference `beta sum_C (h[i, c'] -
/// h[i, c])`. `heat_bath = true` draws the label from `exp(beta sum_C h[i,
/// c])` over all `n_states` ([`heat_bath_label`]). A forbidden
/// label (`-inf`) is never entered; a cluster forbidding every label keeps
/// its own, as the oracle (#1146).
///
/// # Errors
/// Returns `Err` for an empty alphabet, a field, offsets or outcome buffer of
/// the wrong length, a non-finite or negative `beta`, a `proposed` with the
/// heat bath or outside the alphabet, a `root` outside the graph, and, as the
/// cluster reaches them, a site label outside the alphabet, decreasing or
/// out-of-range offsets, a neighbour outside the graph, or a negative or
/// `NaN` coupling. A step refused part-way leaves its cluster unrecoloured;
/// the steps before it stand.
#[allow(clippy::too_many_arguments)]
pub fn wolff_sweeps_impl(
    state: &mut [i64],
    lattice: &Lattice<'_>,
    beta: f64,
    heat_bath: bool,
    root: i64,
    proposed: i64,
    seed: u64,
    out: &mut Outcomes<'_>,
) -> Result<(), String> {
    let n_nodes = state.len();
    let n_states = lattice.n_states;
    let (field, offsets, neighbours, couplings) = (
        lattice.field,
        lattice.offsets,
        lattice.neighbours,
        lattice.couplings,
    );
    let n_steps = out.sizes.len();
    if n_states == 0 {
        return Err("field is empty, so there are no labels to draw".to_string());
    }
    if n_nodes == 0 {
        return Err("state is empty, so there is no site to seed a cluster".to_string());
    }
    if field.len() != n_nodes * n_states {
        return Err(format!(
            "field has {} entries, expected {n_nodes} * {n_states} (one row per site)",
            field.len()
        ));
    }
    if offsets.len() != n_nodes + 1 {
        return Err(format!(
            "offsets has {} entries, expected n_nodes + 1 = {}",
            offsets.len(),
            n_nodes + 1
        ));
    }
    if neighbours.len() != couplings.len() {
        return Err(format!(
            "neighbours has {} entries and couplings {}; one each per incident edge",
            neighbours.len(),
            couplings.len()
        ));
    }
    if out.members.len() != n_nodes
        || out.proposals.len() != n_steps
        || out.accepts.len() != n_steps
    {
        return Err(format!(
            "members must hold {n_nodes} and proposals and accepts {n_steps}; got {}, {} and {}",
            out.members.len(),
            out.proposals.len(),
            out.accepts.len()
        ));
    }
    if !(beta.is_finite() && beta >= 0.0) {
        return Err(format!("beta must be finite and >= 0, got {beta}"));
    }
    if root >= n_nodes as i64 {
        return Err(format!("root {root} is outside [0, {n_nodes})"));
    }
    if proposed >= 0 && (heat_bath || proposed >= n_states as i64) {
        return Err(format!(
            "proposed {proposed} must lie in [0, {n_states}) and is not taken by the heat bath"
        ));
    }
    if n_steps >= u32::MAX as usize {
        return Err(format!("{n_steps} steps exceed the stamp's epochs"));
    }

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let draw = WolffDraw::new(n_nodes, n_states)?;
    // Allocated once per batch: the stamp, and the heat bath's per-label sums.
    let mut scratch = WolffScratch::new(n_nodes, n_states);
    for step in 0..n_steps {
        let (len, proposal, accept, _) = wolff_step(
            state,
            lattice,
            beta,
            heat_bath,
            root,
            proposed,
            &mut rng,
            &draw,
            &mut scratch,
            out.members,
        )?;
        out.proposals[step] = proposal;
        out.accepts[step] = accept;
        out.sizes[step] = len as i64;
    }
    Ok(())
}

/// The two uniform distributions a Wolff step draws its seed site and its
/// uniform proposal from, built once per run.
pub struct WolffDraw {
    sites: Uniform<usize>,
    labels: Uniform<usize>,
}

impl WolffDraw {
    /// Over `n_nodes` sites and `n_states` labels.
    ///
    /// # Errors
    /// Returns `Err` where either is zero.
    pub fn new(n_nodes: usize, n_states: usize) -> Result<Self, String> {
        Ok(Self {
            sites: Uniform::new(0, n_nodes).map_err(|e| e.to_string())?,
            labels: Uniform::new(0, n_states).map_err(|e| e.to_string())?,
        })
    }
}

/// A run's Wolff scratch: the visited stamp per site, its epoch, and the
/// heat bath's per-label sums. Held across steps so a step allocates nothing
/// and clears nothing (#1368).
pub struct WolffScratch {
    /// The epoch of the step that last reached each site.
    pub stamp: Vec<u32>,
    /// The current step's epoch; `stamp[i] == epoch` marks a member.
    pub epoch: u32,
    sums: Vec<f64>,
}

impl WolffScratch {
    /// For `n_nodes` sites and `n_states` labels.
    #[must_use]
    pub fn new(n_nodes: usize, n_states: usize) -> Self {
        Self {
            stamp: vec![0u32; n_nodes],
            epoch: 0,
            sums: vec![0.0f64; n_states],
        }
    }
}

/// One Wolff step of [`wolff_sweeps_impl`] on the caller's generator and
/// scratch: `(cluster size, proposed another label, label changed, the
/// cluster's label before the step)`, the
/// cluster in `members[..size]`. The batch kernel and the Rust loop
/// (`potts_loop.rs`, #1368) both call it, so the two draw alike.
///
/// # Errors
/// As [`wolff_sweeps_impl`], for what the cluster reaches.
#[allow(clippy::too_many_arguments)]
pub fn wolff_step(
    state: &mut [i64],
    lattice: &Lattice<'_>,
    beta: f64,
    heat_bath: bool,
    root: i64,
    proposed: i64,
    rng: &mut ChaCha8Rng,
    draw: &WolffDraw,
    scratch: &mut WolffScratch,
    members: &mut [i64],
) -> Result<(usize, bool, bool, i64), String> {
    let n_nodes = state.len();
    let n_states = lattice.n_states;
    let (field, offsets, neighbours, couplings) = (
        lattice.field,
        lattice.offsets,
        lattice.neighbours,
        lattice.couplings,
    );
    if scratch.epoch == u32::MAX {
        scratch.stamp.iter_mut().for_each(|s| *s = 0);
        scratch.epoch = 0;
    }
    scratch.epoch += 1;
    let epoch = scratch.epoch;
    let stamp = &mut scratch.stamp;
    let sums = &mut scratch.sums;
    let (sites, labels) = (&draw.sites, &draw.labels);
    let (proposal, accept);
    let seed_node = if root < 0 {
        sites.sample(rng)
    } else {
        root as usize
    };
    let colour = state[seed_node];
    if colour < 0 || colour >= n_states as i64 {
        return Err(format!(
            "state at node {seed_node} is {colour}, expected [0, {n_states})"
        ));
    }
    members[0] = seed_node as i64;
    stamp[seed_node] = epoch;
    let (mut head, mut len) = (0usize, 1usize);
    while head < len {
        let node = members[head] as usize;
        head += 1;
        let (lo, hi) = (offsets[node], offsets[node + 1]);
        if lo < 0 || hi < lo || hi as usize > neighbours.len() {
            return Err(format!(
                "offsets of node {node} are [{lo}, {hi}), outside [0, {}]",
                neighbours.len()
            ));
        }
        for position in lo as usize..hi as usize {
            let neighbour = neighbours[position];
            if neighbour < 0 || neighbour >= n_nodes as i64 {
                return Err(format!(
                    "neighbour {neighbour} at position {position} is outside [0, {n_nodes})"
                ));
            }
            let neighbour = neighbour as usize;
            if stamp[neighbour] == epoch || state[neighbour] != colour {
                continue;
            }
            let coupling = couplings[position];
            // `NaN` is refused beside a negative coupling: neither gives
            // `1 - exp(-beta J)` that is a probability.
            if coupling.is_nan() || coupling < 0.0 {
                return Err(format!(
                    "coupling {coupling} at position {position} is negative: the cluster \
                     moves refuse it"
                ));
            }
            let bond = -(-beta * coupling).exp_m1();
            let uniform: f64 = StandardUniform.sample(rng);
            if uniform < bond {
                stamp[neighbour] = epoch;
                members[len] = neighbour as i64;
                len += 1;
            }
        }
    }
    let cluster = &members[..len];

    let label = if heat_bath {
        sums.iter_mut().for_each(|s| *s = 0.0);
        for &member in cluster {
            let row = &field[member as usize * n_states..][..n_states];
            for (s, &h) in sums.iter_mut().zip(row) {
                *s += h;
            }
        }
        let label = heat_bath_label(sums, beta, colour as usize, rng) as i64;
        proposal = label != colour;
        accept = label != colour;
        label
    } else {
        let offer = if proposed >= 0 {
            proposed as usize
        } else {
            labels.sample(rng)
        };
        let accepted = if offer as i64 == colour {
            proposal = false;
            false
        } else {
            proposal = true;
            let (mut offered, mut current) = (0.0f64, 0.0f64);
            for &member in cluster {
                let row = member as usize * n_states;
                offered += field[row + offer];
                current += field[row + colour as usize];
            }
            if offered == f64::NEG_INFINITY {
                false
            } else if current == f64::NEG_INFINITY {
                true
            } else {
                let difference = beta * (offered - current);
                difference >= 0.0 || {
                    let uniform: f64 = StandardUniform.sample(rng);
                    uniform < difference.exp()
                }
            }
        };
        accept = accepted;
        if accepted {
            offer as i64
        } else {
            colour
        }
    };
    if label != colour {
        for &member in cluster {
            state[member as usize] = label;
        }
    }
    Ok((len, proposal, accept, colour))
}

/// PyO3 boundary for [`wolff_sweeps_impl`], `Err` mapped to a Python
/// `ValueError`; the GIL is released for the batch. `root` and `proposed`
/// are negative to draw them.
#[pyfunction]
#[pyo3(signature = (state, field, offsets, neighbours, couplings, beta, heat_bath, root, proposed, seed, members, sizes, proposals, accepts))]
#[allow(clippy::too_many_arguments)]
pub fn wolff_sweeps(
    py: Python<'_>,
    mut state: PyReadwriteArray1<'_, i64>,
    field: PyReadonlyArrayDyn<'_, f64>,
    offsets: PyReadonlyArray1<'_, i64>,
    neighbours: PyReadonlyArray1<'_, i64>,
    couplings: PyReadonlyArray1<'_, f64>,
    beta: f64,
    heat_bath: bool,
    root: i64,
    proposed: i64,
    seed: u64,
    mut members: PyReadwriteArray1<'_, i64>,
    mut sizes: PyReadwriteArray1<'_, i64>,
    mut proposals: PyReadwriteArray1<'_, bool>,
    mut accepts: PyReadwriteArray1<'_, bool>,
) -> PyResult<()> {
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
    let lattice = Lattice {
        field: field.as_slice()?,
        n_states,
        offsets: offsets.as_slice()?,
        neighbours: neighbours.as_slice()?,
        couplings: couplings.as_slice()?,
    };
    let mut out = Outcomes {
        members: members.as_slice_mut()?,
        sizes: sizes.as_slice_mut()?,
        proposals: proposals.as_slice_mut()?,
        accepts: accepts.as_slice_mut()?,
    };
    py.detach(|| {
        wolff_sweeps_impl(
            state, &lattice, beta, heat_bath, root, proposed, seed, &mut out,
        )
    })
    .map_err(PyValueError::new_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 4-site path 0-1-2-3, unit couplings, no field, `n_states` labels.
    fn path(n_states: usize) -> (Vec<i64>, Vec<i64>, Vec<f64>, Vec<f64>) {
        let offsets = vec![0, 1, 3, 5, 6];
        let neighbours = vec![1, 0, 2, 1, 3, 2];
        let couplings = vec![1.0; 6];
        (offsets, neighbours, couplings, vec![0.0; 4 * n_states])
    }

    type Run = Result<(Vec<i64>, Vec<i64>, Vec<bool>), String>;

    #[allow(clippy::too_many_arguments)]
    fn run(
        state: &mut [i64],
        field: &[f64],
        n_states: usize,
        couplings: &[f64],
        beta: f64,
        heat_bath: bool,
        root: i64,
        proposed: i64,
        n_steps: usize,
    ) -> Run {
        let (offsets, neighbours, _, _) = path(n_states);
        let lattice = Lattice {
            field,
            n_states,
            offsets: &offsets,
            neighbours: &neighbours,
            couplings,
        };
        let mut members = vec![0; state.len()];
        let mut sizes = vec![0; n_steps];
        let mut proposals = vec![false; n_steps];
        let mut accepts = vec![false; n_steps];
        let mut out = Outcomes {
            members: &mut members,
            sizes: &mut sizes,
            proposals: &mut proposals,
            accepts: &mut accepts,
        };
        wolff_sweeps_impl(
            state, &lattice, beta, heat_bath, root, proposed, 7, &mut out,
        )?;
        Ok((members, sizes, accepts))
    }

    /// An infinite coupling bonds every like edge with probability one, so a
    /// uniform state makes the whole path one cluster, and with no field the
    /// proposed label is always accepted.
    #[test]
    fn a_certain_bond_takes_the_whole_like_component() {
        let (_, _, _, field) = path(3);
        let mut state = vec![0; 4];
        let (members, sizes, accepts) = run(
            &mut state,
            &field,
            3,
            &[f64::INFINITY; 6],
            1.0,
            false,
            2,
            1,
            1,
        )
        .unwrap();
        assert_eq!(sizes, vec![4]);
        let mut sorted = members.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
        assert!(accepts[0]);
        assert_eq!(state, vec![1; 4]);
    }

    /// A zero coupling bonds nothing, and an unlike neighbour is never joined.
    #[test]
    fn a_zero_coupling_leaves_the_seed_alone() {
        let (_, _, _, field) = path(2);
        let mut state = vec![0, 0, 1, 0];
        let (_, sizes, _) = run(&mut state, &field, 2, &[0.0; 6], 1.0, false, 1, 1, 1).unwrap();
        assert_eq!(sizes, vec![1]);
        assert_eq!(state, vec![0, 1, 1, 0]);
    }

    /// A label every member forbids is never entered, under either recolouring.
    #[test]
    fn a_forbidden_label_is_never_entered() {
        let (_, _, _, mut field) = path(2);
        for node in 0..4 {
            field[node * 2 + 1] = f64::NEG_INFINITY;
        }
        for heat_bath in [false, true] {
            let mut state = vec![0; 4];
            let (_, sizes, _) = run(
                &mut state, &field, 2, &[0.5; 6], 1.0, heat_bath, -1, -1, 200,
            )
            .unwrap();
            assert_eq!(state, vec![0; 4]);
            assert_eq!(sizes.len(), 200);
        }
    }

    /// A negative coupling reached on a like edge is refused.
    #[test]
    fn a_negative_coupling_is_refused() {
        let (_, _, _, field) = path(2);
        let mut state = vec![0; 4];
        let err = run(&mut state, &field, 2, &[-1.0; 6], 1.0, false, 0, 1, 1).unwrap_err();
        assert!(err.contains("negative"), "{err}");
    }

    /// A proposed label is refused by the heat bath, which draws its own.
    #[test]
    fn the_heat_bath_takes_no_proposed_label() {
        let (_, _, _, field) = path(2);
        let mut state = vec![0; 4];
        assert!(run(&mut state, &field, 2, &[1.0; 6], 1.0, true, 0, 1, 1).is_err());
    }

    /// A strong field on label 1 sends the heat bath's cluster there almost
    /// surely: the conditional is `1 / (1 + exp(-40))` per step.
    #[test]
    fn the_heat_bath_follows_a_dominant_field() {
        let (_, _, _, mut field) = path(2);
        for node in 0..4 {
            field[node * 2 + 1] = 10.0;
        }
        let mut state = vec![0; 4];
        run(
            &mut state,
            &field,
            2,
            &[f64::INFINITY; 6],
            1.0,
            true,
            -1,
            -1,
            1,
        )
        .unwrap();
        assert_eq!(state, vec![1; 4]);
    }
}
