"""Swendsen-Wang, Wolff and their heat-bath variants against exact lattice results (issue #1276).

Enumeration referees these moves on four sites (#1142, #1158); this file
referees them on the periodic ``64 x 64`` lattice per pull request and on
``128 x 128`` and ``256 x 256`` for a release, against closed forms from
outside the repository (`sal.sim.potts`):

- ``q = 2`` at 0.8, 1.0 and 1.2 times the critical coupling: the energy per
  site against Kaufman's exact torus value, and at 1.2 the magnetization
  against Yang's. Kaufman is exact at the simulated ``L``, so no
  finite-size correction enters the tolerance; Yang's value is the plane's,
  which the torus reaches to ``exp(-L / xi)`` with ``xi`` under 3 sites.
- ``q = 3, 4`` at the critical coupling: the energy per site at
  ``L = 64, 128, 256`` extrapolated as ``u_inf + a L^(-x)``, with
  ``x = d - y_t`` fixed at 4/5 and 1/2, against Baxter's value.

**The tolerance is declared before the run**: :data:`SIGMAS` standard errors
of the chain's own mean, the standard error read from its integrated
autocorrelation time (:func:`sal.sample.statistics.integrated_autocorrelation_time`),
and propagated through the weighted fit for the extrapolation. A chain whose
effective sample size falls under :data:`ESS_FLOOR` has no standard error
worth testing against and fails as not mixing, so a sampler that never moves
is refuted rather than passed on a zero tolerance.

The single-site heat bath is the control: run at the off-critical couplings
per pull request, and at the transition for a release at a stated number of
sweeps, where its autocorrelation time is 423 sweeps at ``L = 64``
(measured) against 5 for Swendsen-Wang.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from sal.backend import Backend
from sal.sample.potts_mcmc import PottsMove, sweep_for
from sal.sample.statistics import integrated_autocorrelation_time
from sal.sim.graph import BoundaryCondition, lattice_graph
from sal.sim.potts import (
    baxter_critical_energy,
    critical_coupling,
    energies,
    kaufman_energy,
    site_field,
    yang_magnetization,
)

#: Standard errors a chain's mean may sit from the exact value.
SIGMAS = 4.0
#: Effective draws below which a standard error is not trusted.
ESS_FLOOR = 50.0
#: Couplings as fractions of the critical one: disordered, critical, ordered.
FRACTIONS = (0.8, 1.0, 1.2)
CLUSTER_MOVES = (
    PottsMove.SWENDSEN_WANG,
    PottsMove.SWENDSEN_WANG_HEAT_BATH,
    PottsMove.WOLFF,
    PottsMove.WOLFF_HEAT_BATH,
)
SINGLE_CLUSTER = frozenset({PottsMove.WOLFF, PottsMove.WOLFF_HEAT_BATH})
#: The energy's finite-size exponent at the transition, ``d - y_t``
#: (den Nijs 1979; Nienhuis 1982): 4/5 at q = 3 and 1/2 at q = 4.
CRITICAL_EXPONENT = {3: 0.8, 4: 0.5}
RELEASE_SIDES = (64, 128, 256)


@dataclass(frozen=True)
class Run:
    """How long a chain runs: steps per record, records discarded, records kept."""

    steps: int
    burn_in: int
    records: int


@dataclass(frozen=True)
class Estimate:
    """A chain mean, its standard error, and its effective sample size."""

    mean: float
    error: float
    ess: float


def _run(move: PottsMove, fraction: float, n_nodes: int) -> Run:
    """The declared length of a ``q = 2`` chain.

    A Wolff step recolours one cluster, 31 sites at 0.8 of the transition
    against about 1,500 at it and 3,800 above it on ``64 x 64``, so in the
    disordered phase it records once per ``N / 64`` steps; every other move
    records once per sweep. The record counts were set to hold the effective
    sample size above :data:`ESS_FLOOR` at the largest autocorrelation times
    measured on ``64 x 64``, read before #1319 in Sokal's ``1/2 + sum rho``:
    Swendsen-Wang 5.3 sweeps, Wolff 8.5 records in the disordered phase and
    17.5 steps at the transition, the single-site sweep 1.8 off it. In
    ``1 + 2 sum rho`` these are twice that, and the Wolff chains carry 29.0
    to 49.3 effective draws, under the floor (#1320); they cost 6 to 7 s at
    the transition, so more records would cross the per-PR cap.

    Every chain at or above the transition starts cold: from a drawn state
    at the transition a Wolff chain's first clusters are a few sites, and 200
    steps left its mean 0.10 below Kaufman's at an effective sample size of 9.
    """
    if move in SINGLE_CLUSTER:
        if fraction < 1.0:
            return Run(steps=n_nodes // 64, burn_in=20, records=600)
        if fraction == 1.0:
            return Run(steps=1, burn_in=50, records=1_200)
        return Run(steps=1, burn_in=20, records=250)
    if move is PottsMove.SINGLE_SITE:
        if fraction == 1.0:
            return Run(steps=1, burn_in=2_000, records=100_000)
        return Run(steps=1, burn_in=100, records=2_000)
    return Run(steps=1, burn_in=100, records=1_500)


def _estimate(series: np.ndarray) -> Estimate:
    tau = integrated_autocorrelation_time(series)
    n_records = series.shape[0]
    return Estimate(
        mean=float(series.mean()),
        error=float(series.std() * np.sqrt(tau / n_records)),
        ess=n_records / tau,
    )


def _chain(
    move: PottsMove,
    n_states: int,
    side: int,
    coupling: float,
    run: Run,
    *,
    cold: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """The energy per site and the order parameter after every recorded step.

    The order parameter is ``(q max_k f_k - 1) / (q - 1)`` over the label
    fractions ``f``, which at ``q = 2`` is ``|2 f_0 - 1|``, Yang's
    magnetization. ``cold`` starts every site at label 0 rather than drawn.
    """
    graph = lattice_graph((side, side), BoundaryCondition.PERIODIC, coupling)
    field = np.zeros(n_states)
    rows = site_field(field, graph.n_nodes)
    offsets, neighbours, couplings = graph.compressed_adjacency()
    advance = sweep_for(move, graph, rows, offsets, neighbours, couplings, Backend.RUST)
    state = (
        np.zeros(graph.n_nodes, dtype=np.int64)
        if cold
        else np.ascontiguousarray(
            rng.integers(0, n_states, size=graph.n_nodes), dtype=np.int64
        )
    )
    for _ in range(run.burn_in * run.steps):
        advance(state, rng, 1.0)
    energy = np.empty(run.records)
    order = np.empty(run.records)
    for record in range(run.records):
        for _ in range(run.steps):
            advance(state, rng, 1.0)
        energy[record] = energies(graph, field, state[None])[0] / graph.n_nodes
        largest = np.bincount(state, minlength=n_states).max() / graph.n_nodes
        order[record] = (n_states * largest - 1.0) / (n_states - 1.0)
    return energy, order


def _assert_within(estimate: Estimate, exact: float, label: str) -> None:
    assert estimate.ess >= ESS_FLOOR, (
        f"{label}: not mixing, effective sample size {estimate.ess:.1f}"
    )
    gap = estimate.mean - exact
    assert abs(gap) <= SIGMAS * estimate.error, (
        f"{label}: {estimate.mean:.6f} against exact {exact:.6f}, "
        f"{gap / estimate.error:+.2f} standard errors"
    )


def _check_ising(move: PottsMove, fraction: float, side: int) -> None:
    coupling = fraction * critical_coupling(2)
    run = _run(move, fraction, side * side)
    rng = np.random.default_rng([1276, side, round(10 * fraction), len(move.value)])
    energy, order = _chain(move, 2, side, coupling, run, cold=fraction >= 1.0, rng=rng)

    _assert_within(_estimate(energy), kaufman_energy(coupling, side), "energy")
    if fraction > 1.0:
        _assert_within(_estimate(order), yang_magnetization(coupling), "magnetization")


@pytest.mark.oracle
def test_the_referee_refutes_a_chain_one_percent_off_the_coupling() -> None:
    # The power of the referee: Swendsen-Wang run at 0.99 of the critical
    # coupling and judged against Kaufman's value at it. The exact gap is
    # 0.0374 per site, 23.6 standard errors of the chain's mean (#1319).
    coupling = critical_coupling(2)
    run = _run(PottsMove.SWENDSEN_WANG, 1.0, 64 * 64)
    rng = np.random.default_rng([1276, 64, 10, 99])
    energy, _ = _chain(
        PottsMove.SWENDSEN_WANG, 2, 64, 0.99 * coupling, run, cold=True, rng=rng
    )

    with pytest.raises(AssertionError, match="standard errors"):
        _assert_within(_estimate(energy), kaufman_energy(coupling, 64), "energy")


#: Cases under :data:`ESS_FLOOR` once the effective sample size is read as
#: ``n / (1 + 2 sum rho)`` (#1319): 29.0 to 49.3 effective draws, z within
#: 0.93 corrected standard errors. The chains are short for the floor (#1320).
SHORT_OF_THE_FLOOR = {
    (PottsMove.WOLFF, 0.8),
    (PottsMove.WOLFF, 1.0),
    (PottsMove.WOLFF, 1.2),
    (PottsMove.WOLFF_HEAT_BATH, 1.0),
    (PottsMove.WOLFF_HEAT_BATH, 1.2),
}


def _cluster_cases() -> list[object]:
    return [
        pytest.param(
            move,
            fraction,
            id=f"{move}-{fraction}",
            marks=pytest.mark.xfail(
                reason="under ESS_FLOOR since #1319 (#1320)", strict=True
            )
            if (move, fraction) in SHORT_OF_THE_FLOOR
            else (),
        )
        for fraction in FRACTIONS
        for move in CLUSTER_MOVES
    ]


@pytest.mark.oracle
@pytest.mark.parametrize(("move", "fraction"), _cluster_cases())
def test_cluster_move_reaches_kaufman_and_yang_on_64_squared(
    move: PottsMove, fraction: float
) -> None:
    _check_ising(move, fraction, 64)


@pytest.mark.oracle
@pytest.mark.parametrize("fraction", [0.8, 1.2])
def test_single_site_control_reaches_kaufman_and_yang_off_the_transition(
    fraction: float,
) -> None:
    _check_ising(PottsMove.SINGLE_SITE, fraction, 64)


@pytest.mark.oracle
@pytest.mark.release
def test_single_site_control_at_the_transition_in_100k_sweeps() -> None:
    # At the transition the control is held to the same referee at a stated
    # 100,000 sweeps on 64 x 64; where it does not mix in them, the effective
    # sample size check fails and says so.
    _check_ising(PottsMove.SINGLE_SITE, 1.0, 64)


@pytest.mark.oracle
@pytest.mark.release
@pytest.mark.parametrize("side", [128, 256])
@pytest.mark.parametrize("fraction", FRACTIONS)
@pytest.mark.parametrize("move", CLUSTER_MOVES, ids=str)
def test_cluster_move_reaches_kaufman_and_yang_at_release_sizes(
    move: PottsMove, fraction: float, side: int
) -> None:
    _check_ising(move, fraction, side)


def _baxter_run(move: PottsMove, n_states: int) -> Run:
    """The declared length of a critical ``q = 3, 4`` chain.

    Stated, not measured: the release tier has not run (#1276). A count too
    short fails :data:`ESS_FLOOR` and says the chain is not mixing.
    """

    if move in SINGLE_CLUSTER:
        return Run(steps=1, burn_in=500, records=2_000 if n_states == 3 else 4_000)
    return Run(steps=1, burn_in=500, records=4_000 if n_states == 3 else 15_000)


def _extrapolate(
    sides: np.ndarray, estimates: list[Estimate], exponent: float
) -> Estimate:
    """``u_inf`` of the weighted least-squares fit ``u(L) = u_inf + a L^(-x)``, and its error."""
    design = np.column_stack([np.ones(sides.shape[0]), sides ** (-exponent)])
    weights = np.array([estimate.error**-2 for estimate in estimates])
    means = np.array([estimate.mean for estimate in estimates])
    covariance = np.linalg.inv(design.T @ (weights[:, None] * design))
    fit = covariance @ design.T @ (weights * means)
    return Estimate(
        mean=float(fit[0]),
        error=float(np.sqrt(covariance[0, 0])),
        ess=min(estimate.ess for estimate in estimates),
    )


@pytest.mark.oracle
@pytest.mark.release
@pytest.mark.parametrize("n_states", [3, 4])
@pytest.mark.parametrize("move", CLUSTER_MOVES, ids=str)
def test_cluster_move_extrapolates_to_baxter_at_the_transition(
    move: PottsMove, n_states: int
) -> None:
    coupling = critical_coupling(n_states)
    run = _baxter_run(move, n_states)
    estimates = []
    for side in RELEASE_SIDES:
        rng = np.random.default_rng([1276, side, n_states, len(move.value)])
        energy, _ = _chain(move, n_states, side, coupling, run, cold=False, rng=rng)
        estimates.append(_estimate(energy))
    limit = _extrapolate(
        np.array(RELEASE_SIDES, dtype=float), estimates, CRITICAL_EXPONENT[n_states]
    )

    _assert_within(limit, baxter_critical_energy(n_states), "critical energy")
