//! Criterion benchmarks for the `oxiphylo` bindings.
//!
//! `bench_double` is scaffolding, kept for the same demo binding
//! `src/lib.rs` still exposes. `bench_pruning_log_likelihood` measures the
//! real kernel (`src/pruning.rs`) at the same (taxa, site) sizes as
//! `tests/regression/fixtures/simulation_params*.yaml` -- 4 taxa / 200,000
//! sites and 8 taxa / 200,000 sites -- so this number is comparable to the
//! `pytest-benchmark` numbers in `tests/benchmarks/test_pruning_rust_bench.py`.
//! It calls `pruning_log_likelihood_impl` directly (the PyO3-free core, not
//! the `#[pyfunction]` wrapper) for the same link-time reason `cargo test`
//! does -- see `src/pruning.rs`'s module docs. CI runs `cargo bench` but
//! asserts nothing against the timings, since GitHub-hosted runner hardware
//! varies between runs. Compare locally with Criterion's `--save-baseline`
//! and `--baseline`.

use criterion::{criterion_group, criterion_main, Criterion};
use oxiphylo::double;
use oxiphylo::pruning::{pruning_log_likelihood_impl, LeafObservations};
use oxiphylo::sampling::sample_rows_impl;

fn bench_double(c: &mut Criterion) {
    c.bench_function("double", |b| b.iter(|| double(std::hint::black_box(21))));
}

/// A tiny deterministic PRNG (splitmix64) so the benchmark needs no `rand`
/// dependency -- only used to fill leaf states with realistic-looking data,
/// never for anything that must be reproducible science.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn next_state(&mut self, k: usize) -> i64 {
        (self.next_u64() % k as u64) as i64
    }
}

/// Build a balanced binary tree over `n_leaves` taxa (a power of two) with
/// `n_sites` random sites per leaf, in the post-order flat layout
/// `pruning_log_likelihood` expects: leaves first, each internal node after
/// both its children, root last.
fn build_balanced_tree(
    n_leaves: usize,
    n_sites: usize,
    k: usize,
    seed: u64,
) -> (Vec<f64>, Vec<Vec<usize>>, Vec<i64>, Vec<i64>) {
    assert!(n_leaves.is_power_of_two());
    let mut rng = SplitMix64::new(seed);

    let mut branch_length = Vec::new();
    let mut children: Vec<Vec<usize>> = Vec::new();
    let mut leaf_states: Vec<i64> = Vec::new();
    let mut leaf_row: Vec<i64> = Vec::new();

    // level holds the flat-array index of each node at the current level,
    // starting with n_leaves leaves.
    let mut level: Vec<usize> = Vec::with_capacity(n_leaves);
    for _ in 0..n_leaves {
        let idx = branch_length.len();
        branch_length.push(0.05 + (rng.next_u64() % 1000) as f64 / 5000.0); // 0.05..0.25
        children.push(vec![]);
        leaf_row.push((leaf_states.len() / n_sites) as i64);
        leaf_states.extend((0..n_sites).map(|_| rng.next_state(k)));
        level.push(idx);
    }

    while level.len() > 1 {
        let mut next_level = Vec::with_capacity(level.len() / 2);
        for pair in level.chunks(2) {
            let idx = branch_length.len();
            let branch = if next_level.is_empty() && level.len() == 2 {
                0.0 // will be overwritten for the true root below
            } else {
                0.05 + (rng.next_u64() % 1000) as f64 / 5000.0
            };
            branch_length.push(branch);
            children.push(pair.to_vec());
            leaf_row.push(-1);
            next_level.push(idx);
        }
        level = next_level;
    }

    (branch_length, children, leaf_states, leaf_row)
}

fn bench_pruning_log_likelihood(c: &mut Criterion) {
    let k = 4usize;
    let pi = vec![0.25, 0.25, 0.25, 0.25];

    let mut group = c.benchmark_group("pruning_log_likelihood");
    for &(n_leaves, n_sites, label) in &[
        (4usize, 200_000usize, "4taxa_200000sites"),
        (8usize, 200_000usize, "8taxa_200000sites"),
    ] {
        let (branch_length, children, leaf_states, leaf_row) =
            build_balanced_tree(n_leaves, n_sites, k, 20260930);
        group.bench_function(label, |b| {
            b.iter(|| {
                pruning_log_likelihood_impl(
                    std::hint::black_box(&branch_length),
                    std::hint::black_box(&children),
                    std::hint::black_box(LeafObservations {
                        states: &leaf_states,
                        n_sites,
                        row: &leaf_row,
                    }),
                    k,
                    &pi,
                    true,
                )
                .unwrap()
            })
        });
    }
    group.finish();
}

/// `sample_rows` at the two sizes issue #187's audit profiled: 200,000 and
/// 2,000,000 draws over a 4-category alphabet, the sizes
/// `simulate_alignment` reaches on the committed fixtures and one order
/// above.
///
/// This measures the kernel alone, which is the number to compare against
/// `tests/benchmarks/test_numerics_rust_bench.py` rather than to substitute
/// for it: the Python-visible speedup is smaller, because the arrays have to
/// cross the FFI boundary and the NumPy oracle's do not. Reporting only this
/// one would overstate what the port buys.
fn bench_sample_rows(c: &mut Criterion) {
    let n_categories = 4;
    let distributions: Vec<f64> = (0..n_categories * n_categories)
        .map(|i| {
            if i % (n_categories + 1) == 0 {
                0.7
            } else {
                0.1
            }
        })
        .collect();

    for &n_draws in &[200_000_usize, 2_000_000] {
        let mut rng = SplitMix64::new(20260904);
        let mut rows = Vec::with_capacity(n_draws);
        let mut draws = Vec::with_capacity(n_draws);
        for _ in 0..n_draws {
            rows.push(rng.next_state(n_categories));
            // The same [0, 1) construction numpy uses: 53 random bits scaled.
            draws.push((rng.next_u64() >> 11) as f64 / (1_u64 << 53) as f64);
        }

        c.bench_function(&format!("sample_rows/{n_draws}"), |b| {
            b.iter(|| {
                sample_rows_impl(
                    std::hint::black_box(&distributions),
                    n_categories,
                    std::hint::black_box(&rows),
                    std::hint::black_box(&draws),
                )
                .unwrap()
            })
        });
    }
}

criterion_group!(
    benches,
    bench_double,
    bench_pruning_log_likelihood,
    bench_sample_rows
);
criterion_main!(benches);
