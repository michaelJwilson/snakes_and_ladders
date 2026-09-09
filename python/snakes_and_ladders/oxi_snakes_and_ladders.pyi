"""Type stub for the compiled `snakes_and_ladders.oxi_snakes_and_ladders` Rust extension.

Hand-written, so it can drift: keep the signatures here matching the
`#[pyfunction]` definitions in src/lib.rs, src/pruning.rs, src/maxflow.rs,
src/sampling.rs, src/coupled.rs and src/count_pairs.rs. Issue #37 tracks putting `stubtest` in CI so the drift
is caught by a check rather than by whoever notices; until then,
`mypy --strict` catches only the direction where the stub is missing something
a caller uses, which is how `sample_rows` was caught.
"""

import numpy as np

def double(x: int) -> int: ...
def pruning_log_likelihood(
    branch_length: np.ndarray,
    children: list[list[int]],
    leaf_states: np.ndarray,
    leaf_row: list[int],
    k: int,
    pi: np.ndarray,
    rescale: bool,
) -> float: ...
def sample_rows(
    distributions: np.ndarray,
    n_categories: int,
    rows: np.ndarray,
    draws: np.ndarray,
    out: np.ndarray,
) -> None: ...
def max_flow(
    n_nodes: int,
    arcs: np.ndarray,
    capacity: np.ndarray,
    source: int,
    sink: int,
) -> float: ...
def ising_ground_state(
    n_nodes: int,
    field: np.ndarray,
    edges: np.ndarray,
    coupling: np.ndarray,
) -> np.ndarray: ...
def single_site_sweeps(
    state: np.ndarray,
    field: np.ndarray,
    offsets: np.ndarray,
    neighbours: np.ndarray,
    couplings: np.ndarray,
    draws: np.ndarray,
    n_sweeps: int,
) -> None: ...
def class_posteriors(
    totals: np.ndarray,
    successes: np.ndarray,
    labels: np.ndarray,
    total_table: np.ndarray,
    success_table: np.ndarray,
    log_initial: np.ndarray,
    log_transition: np.ndarray,
    n_positions: int,
    n_nodes: int,
    n_classes: int,
    n_states: int,
    posterior: np.ndarray,
    pairwise: np.ndarray,
    log_evidence: np.ndarray,
) -> None: ...
def external_field(
    totals: np.ndarray,
    successes: np.ndarray,
    total_table: np.ndarray,
    success_table: np.ndarray,
    weights: np.ndarray,
    n_positions: int,
    n_nodes: int,
    n_classes: int,
    n_states: int,
    field: np.ndarray,
) -> None: ...
def simulate_count_pairs(
    seed: int,
    states: np.ndarray,
    labels: np.ndarray,
    dispersion: np.ndarray,
    mean: np.ndarray,
    trials: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    n_positions: int,
    n_nodes: int,
    n_classes: int,
    n_states: int,
    totals: np.ndarray,
    successes: np.ndarray,
) -> None: ...
