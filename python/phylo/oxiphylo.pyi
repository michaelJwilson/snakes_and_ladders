"""Type stub for the compiled `phylo.oxiphylo` Rust extension.

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
    arcs: list[int],
    capacity: list[float],
    source: int,
    sink: int,
) -> float: ...
def ising_ground_state(
    n_nodes: int,
    field: list[float],
    edges: list[int],
    coupling: list[float],
) -> list[int]: ...
