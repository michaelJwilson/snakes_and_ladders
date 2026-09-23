"""Type stub for the compiled `snakes_and_ladders.oxi_snakes_and_ladders` Rust extension.

Hand-written, so it can drift: keep the signatures here matching the
`#[pyfunction]` definitions in src/lib.rs, src/pruning.rs, src/pruning_burn.rs,
src/maxflow.rs, src/sampling.rs, src/coupled.rs, src/count_pairs.rs and
src/bcjr.rs. Issue #37 tracks putting `stubtest` in CI so the drift
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

# Declared unconditionally, present only when the extension was built with the
# `sandbox` Cargo feature (`src/pruning_burn.rs`). A stub cannot be conditional
# on how the extension was compiled, and the module that calls this ---
# `snakes_and_ladders.sandbox.pruning_burn`, its only caller --- raises
# `ImportError` when the attribute is absent rather than reaching a missing
# symbol.
def pruning_gradient(
    branch_length: np.ndarray,
    children: list[list[int]],
    leaf_states: np.ndarray,
    leaf_row: list[int],
    k: int,
    pi: np.ndarray,
    weight: np.ndarray | None,
    rescale: bool,
) -> tuple[float, list[float]]: ...
def bcjr_forward_backward(
    next_state: np.ndarray,
    parity: np.ndarray,
    systematic_llr: np.ndarray,
    parity_llr: np.ndarray,
    apriori_llr: np.ndarray,
    terminated: bool,
) -> tuple[np.ndarray, np.ndarray, float]: ...
def tree_message_passing(
    cardinality: np.ndarray,
    variable_offsets: np.ndarray,
    variable_edges: np.ndarray,
    factor_offsets: np.ndarray,
    factor_edges: np.ndarray,
    edge_variable: np.ndarray,
    table_offsets: np.ndarray,
    tables: np.ndarray,
    width: int,
    maximum: bool,
) -> tuple[np.ndarray, np.ndarray]: ...
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
    reverse: np.ndarray | None = ...,
) -> tuple[float, np.ndarray]: ...
def ising_ground_state(
    n_nodes: int,
    field: np.ndarray,
    edges: np.ndarray,
    coupling: np.ndarray,
) -> np.ndarray: ...
def categorical_em_step(
    observations: np.ndarray,
    log_initial: np.ndarray,
    log_transition: np.ndarray,
    log_emission: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]: ...
def gaussian_em_step(
    observations: np.ndarray,
    log_initial: np.ndarray,
    log_transition: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]: ...
def count_cells(
    observations: np.ndarray, covariate: np.ndarray | None, stride: int
) -> tuple[np.ndarray, np.ndarray]: ...
def count_em_step(
    observations: np.ndarray,
    covariate: np.ndarray | None,
    stride: int,
    cells: np.ndarray,
    rows: np.ndarray,
    log_initial: np.ndarray,
    log_transition: np.ndarray,
    family: str,
    parameters: np.ndarray,
    tabled: np.ndarray,
    approx: bool = False,
    stirling_from: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]: ...
def hmm_viterbi(
    observations: np.ndarray,
    log_initial: np.ndarray,
    log_transition: np.ndarray,
    family: str,
    parameters: np.ndarray,
) -> tuple[np.ndarray, float]: ...
def hmm_score(
    observations: np.ndarray,
    log_initial: np.ndarray,
    log_transition: np.ndarray,
    family: str,
    parameters: np.ndarray,
) -> float: ...
def gaussian_hmm_statistics(
    observations: np.ndarray,
    log_initial: np.ndarray,
    log_transition: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]: ...
def bifurcation_integrate(
    rows: np.ndarray,
    offsets: np.ndarray,
    neighbours: np.ndarray,
    couplings: np.ndarray,
    x: np.ndarray,
    n_states: int,
    steps: int,
    dt: float,
    c0: float,
    a_end: float,
    discrete: bool,
) -> None: ...
def gaussian_mixture_em_step(
    values: np.ndarray,
    log_weight: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]: ...
def gaussian_mixture_gradient(
    values: np.ndarray,
    log_weight: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> tuple[float, np.ndarray]: ...
def gaussian_hmc(
    precision: np.ndarray,
    theta0: np.ndarray,
    n_samples: int,
    burn_in: int,
    step_size: float,
    n_steps: int,
    seed: int,
    store_chain: bool,
) -> tuple[np.ndarray, int, np.ndarray]: ...
def gaussian_leapfrog(
    precision: np.ndarray,
    theta: np.ndarray,
    momentum: np.ndarray,
    step_size: float,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]: ...
def bond_roots(n_nodes: int, first: np.ndarray, second: np.ndarray) -> np.ndarray: ...
def ising_ground_states(
    n_nodes: int,
    fields: np.ndarray,
    edges: np.ndarray,
    coupling: np.ndarray,
    threads: int | None = ...,
) -> np.ndarray: ...
def max_flow_declined(
    n_nodes: int,
    arcs: np.ndarray,
    capacity: np.ndarray,
    source: int,
    sink: int,
    reverse: np.ndarray | None = ...,
    algorithm: str = ...,
    threads: int | None = ...,
) -> tuple[float, np.ndarray]: ...
def ising_ground_state_declined(
    n_nodes: int,
    field: np.ndarray,
    edges: np.ndarray,
    coupling: np.ndarray,
    algorithm: str = ...,
    threads: int | None = ...,
) -> np.ndarray: ...
def single_site_sweeps(
    state: np.ndarray,
    field: np.ndarray,
    offsets: np.ndarray,
    neighbours: np.ndarray,
    couplings: np.ndarray,
    draws: np.ndarray,
    n_sweeps: int,
    beta: float,
    guard: float,
    first: int,
) -> int: ...
def swendsen_wang_sweep(
    state: np.ndarray,
    field: np.ndarray,
    edges: np.ndarray,
    bond_probability: np.ndarray,
    bond_draws: np.ndarray,
    colour_draws: np.ndarray,
    accept_draws: np.ndarray,
    labels: np.ndarray,
    guard: float,
    first: int,
) -> tuple[int, int]: ...
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
    exposure: np.ndarray | None,
    trial_counts: np.ndarray | None,
    n_positions: int,
    n_nodes: int,
    n_classes: int,
    n_states: int,
    totals: np.ndarray,
    successes: np.ndarray,
) -> None: ...
def negative_binomial_dispersions(
    tails: np.ndarray,
    total: np.ndarray,
    mean: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    tolerance: float,
    value: np.ndarray,
    at_boundary: np.ndarray,
    iterations: np.ndarray,
    residual: np.ndarray,
) -> None: ...
def beta_binomial_parameters(
    success: np.ndarray,
    failure: np.ndarray,
    depth: np.ndarray,
    total: np.ndarray,
    rate: np.ndarray,
    concentration: np.ndarray,
    bound: np.ndarray,
    log_low: np.ndarray,
    tolerance: float,
    max_iterations: int,
    max_bisections: int,
    margin: float,
    out: np.ndarray,
) -> None: ...

class LatticeCut:
    def __init__(
        self,
        n_nodes: int,
        first: np.ndarray,
        second: np.ndarray,
        coupling: np.ndarray,
    ) -> None: ...
    def expansion_source_side(
        self,
        values: np.ndarray,
        n_states: int,
        labels: np.ndarray,
        alpha: int,
        pinned: float,
    ) -> np.ndarray: ...
    def swap_source_side(
        self,
        values: np.ndarray,
        n_states: int,
        labels: np.ndarray,
        alpha: int,
        beta: int,
    ) -> np.ndarray: ...

def ragged_posteriors(
    log_density: np.ndarray,
    lengths: np.ndarray,
    log_initial: np.ndarray,
    log_transition: np.ndarray,
    gamma: np.ndarray,
    counts: np.ndarray,
    evidence: np.ndarray,
) -> None: ...
