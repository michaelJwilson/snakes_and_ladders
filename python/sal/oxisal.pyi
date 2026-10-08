"""Type stub for the compiled `sal.oxisal` Rust extension.

Hand-written, so it can drift: keep the signatures here matching the
`#[pyfunction]` definitions in src/lib.rs, src/pruning.rs, src/pruning_burn.rs,
src/maxflow.rs, src/sampling.rs, src/coupled.rs, src/count_pairs.rs and
src/bcjr.rs. Issue #37 tracks putting `stubtest` in CI so the drift
is caught by a check rather than by whoever notices; until then,
`mypy --strict` catches only the direction where the stub is missing something
a caller uses, which is how `sample_rows` was caught.
"""

from collections.abc import Mapping
from typing import Any

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
# `sal.sandbox.pruning_burn`, its only caller --- raises
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]: ...
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
def wolff_sweeps(
    state: np.ndarray,
    field: np.ndarray,
    offsets: np.ndarray,
    neighbours: np.ndarray,
    couplings: np.ndarray,
    beta: float,
    heat_bath: bool,
    root: int,
    proposed: int,
    seed: int,
    members: np.ndarray,
    sizes: np.ndarray,
    proposals: np.ndarray,
    accepts: np.ndarray,
) -> None: ...
def potts_loop(
    state: np.ndarray,
    field: np.ndarray,
    offsets: np.ndarray,
    neighbours: np.ndarray,
    couplings: np.ndarray,
    moves: list[int],
    temperatures: np.ndarray,
    budget: int,
    n_main: int,
    lead: int,
    thin: int,
    record: bool,
    track_best: bool,
    seed: int,
) -> dict[str, Any]: ...
def class_posteriors(
    totals: np.ndarray,
    successes: np.ndarray | None,
    labels: np.ndarray,
    total_table: np.ndarray,
    success_table: np.ndarray | None,
    log_initial: np.ndarray,
    log_transition: np.ndarray,
    n_positions: int,
    n_nodes: int,
    n_classes: int,
    n_states: int,
    posterior: np.ndarray,
    pairwise: np.ndarray,
    log_evidence: np.ndarray,
    exposure: np.ndarray | None = None,
    dispersion: np.ndarray | None = None,
    mean: np.ndarray | None = None,
    trials: np.ndarray | None = None,
    failure_table: np.ndarray | None = None,
    trial_table: np.ndarray | None = None,
    log_factorial: np.ndarray | None = None,
    log_rate: np.ndarray | None = None,
) -> None: ...
def external_field(
    totals: np.ndarray,
    successes: np.ndarray | None,
    total_table: np.ndarray,
    success_table: np.ndarray | None,
    weights: np.ndarray,
    n_positions: int,
    n_nodes: int,
    n_classes: int,
    n_states: int,
    field: np.ndarray,
    exposure: np.ndarray | None = None,
    dispersion: np.ndarray | None = None,
    mean: np.ndarray | None = None,
    trials: np.ndarray | None = None,
    failure_table: np.ndarray | None = None,
    trial_table: np.ndarray | None = None,
    log_factorial: np.ndarray | None = None,
    log_rate: np.ndarray | None = None,
) -> None: ...
def factorize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...
def coupled_log_prior(
    labels: np.ndarray, edges: np.ndarray, coupling: np.ndarray, beta: float
) -> float: ...
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
def negative_binomial_dispersions_exposed(
    tails: np.ndarray,
    weights: np.ndarray,
    exposures: np.ndarray,
    counts: np.ndarray,
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
    switch: np.ndarray | None = None,
    switch_kind: str = "stay_or_move",
    threads: int | None = None,
) -> None: ...
def ragged_posterior_probabilities(
    log_density: np.ndarray,
    lengths: np.ndarray,
    log_initial: np.ndarray,
    log_transition: np.ndarray,
    posterior: np.ndarray,
    pairs: np.ndarray,
    evidence: np.ndarray,
    threads: int | None = None,
) -> None: ...
def ragged_viterbi(
    log_density: np.ndarray,
    lengths: np.ndarray,
    log_initial: np.ndarray,
    log_transition: np.ndarray,
    path: np.ndarray,
    log_joint: np.ndarray,
    switch: np.ndarray | None = None,
    switch_kind: str = "stay_or_move",
) -> None: ...
def ragged_sample_paths(
    log_density: np.ndarray,
    lengths: np.ndarray,
    log_initial: np.ndarray,
    log_transition: np.ndarray,
    uniforms: np.ndarray,
    path: np.ndarray,
    log_joint: np.ndarray,
    log_evidence: np.ndarray,
    switch: np.ndarray | None = None,
    switch_kind: str = "stay_or_move",
) -> None: ...

class MetropolisWalk:
    def __init__(
        self,
        kernel: str,
        data: Mapping[str, Any],
        theta0: np.ndarray,
        step_size: float,
        seed: int,
        warmup: int,
        target_acceptance: float,
        step_jitter: float,
        constants: tuple[float, float, float],
        powers: list[int],
        temperature: float,
    ) -> None: ...
    def advance(
        self, n: int, store: bool, observe: bool
    ) -> tuple[np.ndarray, int, np.ndarray]: ...
    def state(self) -> tuple[np.ndarray, float, np.ndarray]: ...
    def set_state(
        self, position: np.ndarray, value: float, gradient: np.ndarray
    ) -> None: ...
    def advance_at(
        self, n: int, temperature: float
    ) -> tuple[np.ndarray, float, np.ndarray, int]: ...
    def statistics(
        self, index: int
    ) -> tuple[
        int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ]: ...
    @property
    def step_size(self) -> float: ...
    @property
    def mass_diagonal(self) -> np.ndarray: ...
    @property
    def flat(self) -> list[int]: ...
    @property
    def warmup_acceptance(self) -> float: ...

class HmcWalk:
    def __init__(
        self,
        kernel: str,
        data: Mapping[str, Any],
        theta0: np.ndarray,
        step_size: float,
        seed: int,
        warmup: int,
        target_acceptance: float,
        step_jitter: float,
        constants: tuple[float, float, float],
        powers: list[int],
        temperature: float,
        n_steps: int,
    ) -> None: ...
    def advance(
        self, n: int, store: bool, observe: bool
    ) -> tuple[np.ndarray, int, np.ndarray]: ...
    def state(self) -> tuple[np.ndarray, float, np.ndarray]: ...
    def set_state(
        self, position: np.ndarray, value: float, gradient: np.ndarray
    ) -> None: ...
    def advance_at(
        self, n: int, temperature: float
    ) -> tuple[np.ndarray, float, np.ndarray, int]: ...
    def statistics(
        self, index: int
    ) -> tuple[
        int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ]: ...
    @property
    def step_size(self) -> float: ...
    @property
    def mass_diagonal(self) -> np.ndarray: ...
    @property
    def flat(self) -> list[int]: ...
    @property
    def warmup_acceptance(self) -> float: ...

def leapfrog_trajectory(
    kernel: str,
    data: Mapping[str, Any],
    theta: np.ndarray,
    momentum: np.ndarray,
    step_size: float,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]: ...

class SupportedEnergy:
    def __init__(
        self, kernel: str, data: Mapping[str, Any], dimension: int
    ) -> None: ...
    def value_and_gradient(self, theta: np.ndarray) -> tuple[float, np.ndarray]: ...

def coded_encode(
    keys: np.ndarray, observed: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
def coded_log_emission(
    n_states: int,
    inverse: np.ndarray,
    out: np.ndarray,
    total_rows: np.ndarray | None = None,
    success_rows: np.ndarray | None = None,
    totals: np.ndarray | None = None,
    total_table: np.ndarray | None = None,
    exposure: np.ndarray | None = None,
    dispersion: np.ndarray | None = None,
    mean: np.ndarray | None = None,
    successes: np.ndarray | None = None,
    success_table: np.ndarray | None = None,
    trials: np.ndarray | None = None,
    failure_table: np.ndarray | None = None,
    trial_table: np.ndarray | None = None,
    log_factorial: np.ndarray | None = None,
    log_rate: np.ndarray | None = None,
) -> None: ...

# Deprecated for one cycle (#1340): use `sal.emissions.coded.log_emission`.
def dense_log_emission(
    n_states: int,
    family_order: bool,
    out: np.ndarray,
    totals: np.ndarray | None = None,
    total_table: np.ndarray | None = None,
    exposure: np.ndarray | None = None,
    dispersion: np.ndarray | None = None,
    mean: np.ndarray | None = None,
    successes: np.ndarray | None = None,
    success_table: np.ndarray | None = None,
    trials: np.ndarray | None = None,
    failure_table: np.ndarray | None = None,
    trial_table: np.ndarray | None = None,
    log_factorial: np.ndarray | None = None,
    log_rate: np.ndarray | None = None,
) -> None: ...
def count_mixture_value_and_gradient(
    log_weight: np.ndarray,
    grad_log_weight: np.ndarray,
    totals: np.ndarray | None = None,
    dispersion: np.ndarray | None = None,
    mean: np.ndarray | None = None,
    grad_dispersion: np.ndarray | None = None,
    grad_mean: np.ndarray | None = None,
    successes: np.ndarray | None = None,
    alpha: np.ndarray | None = None,
    beta: np.ndarray | None = None,
    trials: np.ndarray | None = None,
    grad_alpha: np.ndarray | None = None,
    grad_beta: np.ndarray | None = None,
) -> float: ...
def coded_log_emission_partials(
    total_rows: np.ndarray | None = None,
    totals: np.ndarray | None = None,
    exposure: np.ndarray | None = None,
    factor: np.ndarray | None = None,
    dispersion: np.ndarray | None = None,
    mean: np.ndarray | None = None,
    out_dispersion: np.ndarray | None = None,
    out_mean: np.ndarray | None = None,
    out_shift: np.ndarray | None = None,
    success_rows: np.ndarray | None = None,
    successes: np.ndarray | None = None,
    trials: np.ndarray | None = None,
    per_observation: bool = True,
    alpha: np.ndarray | None = None,
    beta: np.ndarray | None = None,
    rate: np.ndarray | None = None,
    out_first: np.ndarray | None = None,
    out_second: np.ndarray | None = None,
) -> None: ...
def coded_weighted_sum(
    n_states: int,
    values: np.ndarray,
    index: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray: ...
