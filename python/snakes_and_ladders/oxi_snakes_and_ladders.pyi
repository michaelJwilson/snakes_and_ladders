"""Type stub for the compiled `snakes_and_ladders.oxi_snakes_and_ladders` Rust extension.

Hand-written, so it can drift: keep the signatures here matching the
`#[pyfunction]` definitions in src/lib.rs, src/pruning.rs, src/maxflow.rs
and src/sampling.rs. Issue #37 tracks putting `stubtest` in CI so the drift
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
