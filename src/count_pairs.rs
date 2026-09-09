//! Per-vertex simulation of the coupled model's two-channel counts, ported
//! from `python/snakes_and_ladders/sim/count_pairs.py` (the NumPy oracle) to
//! Rust, exposed to Python via PyO3 as
//! `snakes_and_ladders.oxi_snakes_and_ladders.simulate_count_pairs`.
//!
//! The declared 5,041-vertex instance is 1.008e8 count pairs
//! (`tests/regression/fixtures/spatio_sequential_counts/stress.yaml`), each a
//! negative-binomial total and a beta-binomial success count. That is three
//! non-uniform variates per pair and it is what a caller waits for before
//! anything is scored.
//!
//! **This module holds a generator, and that is a deliberate exception.**
//! `src/sampling.rs` states the repository's rule --- the uniforms are drawn
//! in Python and passed in, so a seeded `numpy.random.Generator` determines
//! the result --- and it does not apply here: passing 3.0e8 uniforms across
//! the boundary is 2.4 GB, several times the draw they produce. What replaces
//! it is a stronger reproducibility contract than a shared generator gives:
//! **vertex `v`'s counts depend on the fixture's seed and on `v`, and on
//! nothing else.** Not on the order vertices are visited in, not on how many
//! threads visit them, not on how many positions were drawn before. Each
//! vertex gets its own ChaCha8 stream keyed by `[seed, v]`, so the fixture is
//! the same draw whoever asks for it, and the fixture file records a digest of
//! the counts so a change to this file is visible in review.
//!
//! The oracle is therefore distributional and not bitwise: the NumPy
//! simulator draws the same distributions through NumPy's own generator, and
//! the two are compared on per-state sample means and dispersions at the ci
//! instance (`tests/regression/sim/test_count_pairs_rust.py`).
//!
//! **The output is written position-major in vertex blocks.** The rest of the
//! repository holds observations as `(S, n_nodes)`, while a vertex's stream
//! produces its `S` draws in order, so a vertex-at-a-time write would stride
//! the output by `n_nodes` and miss a cache line per draw. A block of
//! `VERTEX_BLOCK` vertices is drawn into a scratch buffer that fits in cache
//! and then written out transposed, 64 contiguous counts at a time.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use numpy::{PyReadonlyArray1, PyReadwriteArray1};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Beta, Binomial, Distribution, Gamma, Poisson};

/// Vertices drawn before the block is written out transposed. At the declared
/// `S = 20,000` the scratch is `64 * 20,000 * 2` counts, 5 MB, which is the
/// last size that stays inside the reference host's L3 while still amortizing
/// the transposed write over 64 contiguous entries.
const VERTEX_BLOCK: usize = 64;

/// The emission parameters of every class and state, as five flat `(M, K)` tables.
pub struct CountPairFamilies<'a> {
    /// Negative-binomial `r`.
    pub dispersion: &'a [f64],
    /// Negative-binomial `mu`.
    pub mean: &'a [f64],
    /// Beta-binomial `n`.
    pub trials: &'a [f64],
    /// Beta-binomial `a`.
    pub alpha: &'a [f64],
    /// Beta-binomial `b`.
    pub beta: &'a [f64],
}

/// Vertex `v`'s generator: ChaCha8 seeded by the fixture's seed and `v` alone.
///
/// The 32-byte key is the two numbers expanded by SplitMix64, which is what
/// `rand`'s own `seed_from_u64` does with one number; spelling it out here
/// keeps the keying a property of this file rather than of a helper's version.
fn stream(seed: u64, vertex: u64) -> ChaCha8Rng {
    let mut state = seed
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(vertex.wrapping_add(1).wrapping_mul(0xbf58_476d_1ce4_e5b9));
    let mut key = [0u8; 32];
    for chunk in key.chunks_exact_mut(8) {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        chunk.copy_from_slice(&(z ^ (z >> 31)).to_le_bytes());
    }
    ChaCha8Rng::from_seed(key)
}

/// Draw every vertex's counts from its own stream, position-major.
///
/// # Parameters
/// - `seed`: the fixture's seed; with the vertex it determines the draw.
/// - `states`: `M * S` hidden states, row-major, entries in `[0, K)`.
/// - `labels`: one class per vertex, entries in `[0, M)`.
/// - `families`: five `M * K` parameter tables.
/// - `n_positions`, `n_nodes`, `n_classes`, `n_states`: `S`, `V`, `M`, `K`.
/// - `totals`, `successes`: written, `S * V` each, position-major.
///
/// # Returns
/// `Ok(())`, or `Err` naming the first violated precondition --- including a
/// drawn count past `u16`, which is a fixture whose parameters have outgrown
/// the type its counts are held in rather than a type to widen silently.
#[allow(clippy::too_many_arguments)]
pub fn simulate_count_pairs_into(
    seed: u64,
    states: &[i64],
    labels: &[i64],
    families: &CountPairFamilies<'_>,
    n_positions: usize,
    n_nodes: usize,
    n_classes: usize,
    n_states: usize,
    totals: &mut [u16],
    successes: &mut [u16],
) -> Result<(), String> {
    let block = n_classes * n_states;
    if n_positions == 0 || n_nodes == 0 || block == 0 {
        return Err("a coupled instance has at least one of each dimension".to_string());
    }
    if states.len() != n_classes * n_positions {
        return Err(format!(
            "states has {} entries, expected M * S = {}",
            states.len(),
            n_classes * n_positions
        ));
    }
    if labels.len() != n_nodes {
        return Err(format!(
            "labels has {} entries, expected V = {n_nodes}",
            labels.len()
        ));
    }
    for (name, table) in [
        ("dispersion", families.dispersion),
        ("mean", families.mean),
        ("trials", families.trials),
        ("alpha", families.alpha),
        ("beta", families.beta),
    ] {
        if table.len() != block {
            return Err(format!(
                "{name} has {} entries, expected M * K = {block}",
                table.len()
            ));
        }
    }
    if totals.len() != n_positions * n_nodes || successes.len() != totals.len() {
        return Err(format!(
            "the outputs have {} and {} entries, expected S * V = {}",
            totals.len(),
            successes.len(),
            n_positions * n_nodes
        ));
    }
    if labels
        .iter()
        .any(|&label| label < 0 || label as usize >= n_classes)
    {
        return Err(format!("every label must lie in [0, {n_classes})"));
    }
    if states
        .iter()
        .any(|&state| state < 0 || state as usize >= n_states)
    {
        return Err(format!("every state must lie in [0, {n_states})"));
    }

    // One distribution object per (class, state) rather than per draw: the
    // negative binomial is a gamma mixture of Poissons, so its shape and
    // scale are fixed by the state and only the Poisson rate is drawn.
    let mut gamma = Vec::with_capacity(block);
    let mut rate = Vec::with_capacity(block);
    for index in 0..block {
        let (r, mu) = (families.dispersion[index], families.mean[index]);
        if r <= 0.0 || mu <= 0.0 || r.is_nan() || mu.is_nan() {
            return Err(format!(
                "state {index} has dispersion {r} and mean {mu}; both must be positive"
            ));
        }
        gamma.push(Gamma::new(r, mu / r).map_err(|error| error.to_string())?);
        rate.push(
            Beta::new(families.alpha[index], families.beta[index])
                .map_err(|error| error.to_string())?,
        );
    }

    let largest = u64::from(u16::MAX);
    let mut scratch_totals = vec![0u16; VERTEX_BLOCK * n_positions];
    let mut scratch_successes = vec![0u16; VERTEX_BLOCK * n_positions];
    for first in (0..n_nodes).step_by(VERTEX_BLOCK) {
        let width = VERTEX_BLOCK.min(n_nodes - first);
        for column in 0..width {
            let vertex = first + column;
            let class = labels[vertex] as usize;
            let mut rng = stream(seed, vertex as u64);
            let into_totals = &mut scratch_totals[column * n_positions..][..n_positions];
            let into_successes = &mut scratch_successes[column * n_positions..][..n_positions];
            for s in 0..n_positions {
                let index = class * n_states + states[class * n_positions + s] as usize;
                let mean = gamma[index].sample(&mut rng).max(f64::MIN_POSITIVE);
                let count = Poisson::new(mean)
                    .map_err(|error| error.to_string())?
                    .sample(&mut rng) as u64;
                let probability = rate[index].sample(&mut rng);
                let drawn = Binomial::new(families.trials[index] as u64, probability)
                    .map_err(|error| error.to_string())?
                    .sample(&mut rng);
                if count > largest || drawn > largest {
                    return Err(format!(
                        "vertex {vertex} drew {count} and {drawn}, past u16; the \
                         declared emission parameters have outgrown the fixture's dtype"
                    ));
                }
                into_totals[s] = count as u16;
                into_successes[s] = drawn as u16;
            }
        }
        for s in 0..n_positions {
            let row = s * n_nodes + first;
            for column in 0..width {
                totals[row + column] = scratch_totals[column * n_positions + s];
                successes[row + column] = scratch_successes[column * n_positions + s];
            }
        }
    }
    Ok(())
}

/// `simulate_count_pairs_into` as a Python binding.
///
/// # Returns
/// `None`; the counts are written into `totals` and `successes`.
///
/// # Errors
/// `ValueError` naming the first violated precondition.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn simulate_count_pairs(
    seed: u64,
    states: PyReadonlyArray1<'_, i64>,
    labels: PyReadonlyArray1<'_, i64>,
    dispersion: PyReadonlyArray1<'_, f64>,
    mean: PyReadonlyArray1<'_, f64>,
    trials: PyReadonlyArray1<'_, f64>,
    alpha: PyReadonlyArray1<'_, f64>,
    beta: PyReadonlyArray1<'_, f64>,
    n_positions: usize,
    n_nodes: usize,
    n_classes: usize,
    n_states: usize,
    mut totals: PyReadwriteArray1<'_, u16>,
    mut successes: PyReadwriteArray1<'_, u16>,
) -> PyResult<()> {
    let contiguous = |name: &str| PyValueError::new_err(format!("{name} must be C-contiguous"));
    let families = CountPairFamilies {
        dispersion: dispersion
            .as_slice()
            .map_err(|_| contiguous("dispersion"))?,
        mean: mean.as_slice().map_err(|_| contiguous("mean"))?,
        trials: trials.as_slice().map_err(|_| contiguous("trials"))?,
        alpha: alpha.as_slice().map_err(|_| contiguous("alpha"))?,
        beta: beta.as_slice().map_err(|_| contiguous("beta"))?,
    };
    simulate_count_pairs_into(
        seed,
        states.as_slice().map_err(|_| contiguous("states"))?,
        labels.as_slice().map_err(|_| contiguous("labels"))?,
        &families,
        n_positions,
        n_nodes,
        n_classes,
        n_states,
        totals.as_slice_mut().map_err(|_| contiguous("totals"))?,
        successes
            .as_slice_mut()
            .map_err(|_| contiguous("successes"))?,
    )
    .map_err(PyValueError::new_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draw(seed: u64, n_nodes: usize) -> (Vec<u16>, Vec<u16>) {
        let dispersion = [8.0, 8.0];
        let mean = [40.0, 90.0];
        let trials = [30.0, 30.0];
        let alpha = [6.0, 12.0];
        let beta = [14.0, 8.0];
        let n_positions = 200;
        let states: Vec<i64> = (0..n_positions).map(|s| (s % 2) as i64).collect();
        let labels = vec![0i64; n_nodes];
        let mut totals = vec![0u16; n_positions * n_nodes];
        let mut successes = vec![0u16; n_positions * n_nodes];
        simulate_count_pairs_into(
            seed,
            &states,
            &labels,
            &CountPairFamilies {
                dispersion: &dispersion,
                mean: &mean,
                trials: &trials,
                alpha: &alpha,
                beta: &beta,
            },
            n_positions,
            n_nodes,
            1,
            2,
            &mut totals,
            &mut successes,
        )
        .unwrap();
        (totals, successes)
    }

    #[test]
    fn a_vertex_draw_does_not_depend_on_how_many_vertices_are_drawn() {
        // The contract the per-vertex stream exists for: vertex 0's counts are
        // the same whether it is drawn alone or inside a block of 200, which
        // is what makes the fixture independent of the traversal.
        let (alone, alone_successes) = draw(20260908, 1);
        let (together, together_successes) = draw(20260908, 200);
        let n_positions = 200;

        for s in 0..n_positions {
            assert_eq!(alone[s], together[s * 200]);
            assert_eq!(alone_successes[s], together_successes[s * 200]);
        }
    }

    #[test]
    fn a_different_seed_is_a_different_draw() {
        let (first, _) = draw(20260908, 8);
        let (second, _) = draw(20260909, 8);

        assert_ne!(first, second);
    }

    #[test]
    fn the_sample_mean_is_the_declared_mean() {
        // 200 positions on 200 vertices, alternating between the two states,
        // is 20,000 draws per state: enough for a 3% check on a mean of 40.
        let (totals, successes) = draw(20260908, 200);
        let n_nodes = 200;
        let mut sums = [0.0f64; 2];
        let mut success_sums = [0.0f64; 2];
        for s in 0..200 {
            for v in 0..n_nodes {
                sums[s % 2] += f64::from(totals[s * n_nodes + v]);
                success_sums[s % 2] += f64::from(successes[s * n_nodes + v]);
            }
        }
        let draws = 100.0 * n_nodes as f64;

        assert!((sums[0] / draws - 40.0).abs() < 40.0 * 0.03);
        assert!((sums[1] / draws - 90.0).abs() < 90.0 * 0.03);
        // Beta-binomial means: n a / (a + b) = 30 * 6/20 and 30 * 12/20.
        assert!((success_sums[0] / draws - 9.0).abs() < 9.0 * 0.03);
        assert!((success_sums[1] / draws - 18.0).abs() < 18.0 * 0.03);
    }

    #[test]
    fn a_negative_dispersion_is_refused() {
        let mut totals = vec![0u16; 2];
        let mut successes = vec![0u16; 2];
        let refused = simulate_count_pairs_into(
            1,
            // M * S = 1, or the dimension check refuses before the dispersion
            // one does and the test passes on the wrong error.
            &[0],
            &[0, 0],
            &CountPairFamilies {
                dispersion: &[-1.0],
                mean: &[10.0],
                trials: &[5.0],
                alpha: &[1.0],
                beta: &[1.0],
            },
            1,
            2,
            1,
            1,
            &mut totals,
            &mut successes,
        );

        assert!(refused.unwrap_err().contains("must be positive"));
    }
}
