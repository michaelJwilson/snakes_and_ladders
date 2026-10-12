"""Every Potts cluster move against the single-site heat bath on per-site fields and forbidden labels (issue #1278).

Enumeration referees these moves on four sites (#1142, #1146, #1154); this
file referees them on the periodic ``64 x 64``, ``q = 3`` lattice per pull
request, and on ``128 x 128`` for a release, against the single-site heat
bath (Glauber) at equal ``beta``. Glauber is the reference because its
detailed balance is the simplest to verify and enumeration already pins it.

- **Instances:** a per-site field ``N(0, sigma^2)`` per (site, label), and a
  forbidden fraction of (site, label) pairs set to ``-inf``
  (:func:`sal.sim.potts.forbid`), at :data:`FRACTIONS` of the critical
  coupling. Per pull request: ``sigma = 0.3`` with 5% forbidden and
  ``sigma = 1`` with none; the other two corners and ``L = 128`` are release.
- **Observables:** the energy per site, the label occupancies, and the
  two-point function ``P(s_i = s_(i+r))`` at ``r = 1`` and ``4``, averaged
  over both lattice axes.
- **Test, declared before the run:** per observable, the difference of the
  two chains' means over ``sqrt(se_a^2 + se_b^2)``, each ``se`` read from
  the chain's integrated autocorrelation time, must lie within
  :data:`SIGMAS`. At 4 standard errors a comparison fails by chance with
  probability 6.3e-5; over the 252 per-pull-request comparisons the expected
  count of chance failures is 0.016. A chain whose effective sample size
  falls under :data:`ESS_FLOOR` fails as not mixing, as in #1276.
- **Forbidden pairs:** every recorded draw of every move, Glauber included,
  is counted against the mask; the count must be zero.

Every chain starts from the same state: each site on the label with the
largest summed allowed field over the lattice, or, where its own row
forbids that label, on its own largest allowed entry. Above the transition
the summed field selects one ordered phase by a margin of ``exp(beta dH)``,
and a single-site chain started elsewhere would coarsen too slowly to reach it.

**Status: draft, every test release-marked.** The first ``L = 64`` run
(43 tests, 8 workers on 4 cores) failed 25 and took 4 to 27 s per test,
over the per-PR cap: at ``1.5 beta_c`` Glauber's effective sample size is
9.8 on an occupancy, under :data:`ESS_FLOOR`; Wolff, Wolff heat bath and
Niedermayer fall to 8 to 45; Niedermayer at ``1.5 beta_c``, ``sigma = 1``
sits on label 2 alone against Glauber's occupancies 0.300 and 0.315, an
energy gap of 206 standard errors. The tolerance and the floor are not
changed to admit these; the schedules and the Niedermayer gap are open.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import pytest
from sal.backend import Backend
from sal.sample.potts_mcmc import PottsMove
from sal.sample.potts_mcmc.chains import sweep_for
from sal.sample.statistics import integrated_autocorrelation_time
from sal.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from sal.sim.potts import critical_coupling, energies, forbid, site_field

N_STATES = 3
#: Standard errors a difference of two chains' means may reach.
SIGMAS = 4.0
#: Effective draws below which a standard error is not trusted.
ESS_FLOOR = 50.0
#: Couplings as fractions of the critical one.
FRACTIONS = (0.5, 1.0, 1.5)
#: Separations of the two-point function.
DISTANCES = (1, 4)
MOVES = (
    PottsMove.SWENDSEN_WANG,
    PottsMove.SWENDSEN_WANG_HEAT_BATH,
    PottsMove.WOLFF,
    PottsMove.WOLFF_HEAT_BATH,
    PottsMove.GHOST_SPIN,
    PottsMove.LABEL_DIRECTED,
    PottsMove.NIEDERMAYER,
)
#: The moves that recolour one cluster per call rather than sweep the lattice.
SINGLE_CLUSTER = frozenset(
    {PottsMove.WOLFF, PottsMove.WOLFF_HEAT_BATH, PottsMove.NIEDERMAYER}
)
#: Calls a single-cluster move makes to measure its mean cluster size.
PILOT = 50
#: (sigma, forbidden fraction) run per pull request, and for a release.
PER_PR = ((0.3, 0.05), (1.0, 0.0))
RELEASE = ((0.3, 0.0), (1.0, 0.05))
OBSERVABLES = (
    "energy",
    *(f"occupancy {k}" for k in range(N_STATES)),
    *(f"C({r})" for r in DISTANCES),
)


@dataclass(frozen=True)
class Instance:
    """A periodic lattice, its per-site field rows, and its allowed mask."""

    side: int
    sigma: float
    forbidden: float
    rows: np.ndarray
    allowed: np.ndarray
    start: np.ndarray


@dataclass(frozen=True)
class Draws:
    """A chain's observables per record, its forbidden visits, its mean cluster size."""

    series: np.ndarray
    forbidden_visits: int
    mean_cluster_size: float


@functools.cache
def _instance(side: int, sigma: float, forbidden: float) -> Instance:
    rng = np.random.default_rng(
        [1278, side, round(100 * sigma), round(100 * forbidden)]
    )
    n_nodes = side * side
    field = rng.normal(0.0, sigma, (n_nodes, N_STATES))
    allowed = rng.random((n_nodes, N_STATES)) >= forbidden
    # A site forbidding every label is not a Potts instance; allow its first.
    allowed[~allowed.any(axis=1), 0] = True
    rows = site_field(forbid(field, allowed), n_nodes, n_states=N_STATES)
    phase = int(np.argmax(np.where(allowed, field, 0.0).sum(axis=0)))
    start = np.where(allowed[:, phase], phase, np.argmax(rows, axis=1))
    return Instance(
        side=side,
        sigma=sigma,
        forbidden=forbidden,
        rows=rows,
        allowed=allowed,
        start=np.ascontiguousarray(start, dtype=np.int64),
    )


def _schedule(
    move: PottsMove, mean_cluster: float, n_nodes: int
) -> tuple[int, int, int]:
    """Steps per record, records discarded, records kept.

    A single-cluster move records once per ``N`` sites it recolours in
    expectation, the mean cluster size read from a pilot of
    :data:`PILOT` calls: on ``64 x 64`` at ``sigma = 0.3`` a Wolff cluster
    holds about 2,200 sites at the transition and 3 at ``sigma = 1``, so a
    fixed step count is either a sweep or a thousandth of one. Every other
    move records once per sweep.
    """
    if move is PottsMove.SINGLE_SITE:
        return 1, 200, 6_000
    if move in SINGLE_CLUSTER:
        return max(1, round(n_nodes / max(mean_cluster, 1.0))), 20, 600
    return 1, 100, 2_000


def _observe(instance: Instance, state: np.ndarray, field: np.ndarray) -> np.ndarray:
    graph = _graph(instance.side)
    n_nodes = graph.n_nodes
    grid = state.reshape(instance.side, instance.side)
    values = [energies(graph, field, state[None])[0] / n_nodes]
    values.extend(np.bincount(state, minlength=N_STATES) / n_nodes)
    for distance in DISTANCES:
        same = (grid == np.roll(grid, distance, axis=0)).mean()
        same += (grid == np.roll(grid, distance, axis=1)).mean()
        values.append(same / 2.0)
    return np.asarray(values)


@functools.cache
def _graph(side: int) -> PottsGraph:
    return lattice_graph((side, side), BoundaryCondition.PERIODIC, 1.0)


def _draws(instance: Instance, move: PottsMove, fraction: float) -> Draws:
    graph = _graph(instance.side)
    beta = fraction * critical_coupling(N_STATES)
    offsets, neighbours, couplings = graph.compressed_adjacency()
    advance = sweep_for(
        move, graph, instance.rows, offsets, neighbours, couplings, Backend.RUST
    )
    rng = np.random.default_rng(
        [1278, instance.side, round(10 * fraction), len(move.value)]
    )
    sites = np.arange(graph.n_nodes)
    state = instance.start.copy()
    pilot = 0
    if move in SINGLE_CLUSTER:
        pilot = sum(advance(state, rng, beta) for _ in range(PILOT))
    steps, burn_in, records = _schedule(move, pilot / PILOT, graph.n_nodes)
    for _ in range(burn_in * steps):
        advance(state, rng, beta)
    # The energy is scored on the finite field: a -inf entry would make a
    # forbidden visit an infinite energy rather than a count.
    finite = np.where(instance.allowed, instance.rows, 0.0)
    series = np.empty((records, len(OBSERVABLES)))
    visits = 0
    clusters = 0
    for record in range(records):
        for _ in range(steps):
            clusters += advance(state, rng, beta)
        visits += int((~instance.allowed[sites, state]).sum())
        series[record] = _observe(instance, state, finite)
    return Draws(
        series=series,
        forbidden_visits=visits,
        mean_cluster_size=clusters / (records * steps),
    )


@functools.cache
def _glauber(side: int, sigma: float, forbidden: float, fraction: float) -> Draws:
    return _draws(_instance(side, sigma, forbidden), PottsMove.SINGLE_SITE, fraction)


def _estimates(
    series: np.ndarray, label: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per observable: the mean, its standard error, and the autocorrelation time."""
    taus = np.array(
        [integrated_autocorrelation_time(series[:, j]) for j in range(series.shape[1])]
    )
    ess = series.shape[0] / taus
    worst = int(np.argmin(ess))
    assert ess[worst] >= ESS_FLOOR, (
        f"{label}: not mixing, effective sample size {ess[worst]:.1f} "
        f"on {OBSERVABLES[worst]}"
    )
    errors = series.std(axis=0) * np.sqrt(taus / series.shape[0])
    return series.mean(axis=0), errors, taus


def _cross_check(
    side: int, sigma: float, forbidden: float, fraction: float, move: PottsMove
) -> None:
    reference = _glauber(side, sigma, forbidden, fraction)
    draws = _draws(_instance(side, sigma, forbidden), move, fraction)

    assert reference.forbidden_visits == 0, "glauber visited a forbidden label"
    assert draws.forbidden_visits == 0, f"{move} visited a forbidden label"
    mean_a, error_a, _ = _estimates(reference.series, "glauber")
    mean_b, error_b, _ = _estimates(draws.series, str(move))
    scale = np.sqrt(error_a**2 + error_b**2)
    gap = mean_b - mean_a
    within = np.abs(gap) <= SIGMAS * scale
    failures = [
        f"{OBSERVABLES[j]}: {mean_b[j]:.6f} against {mean_a[j]:.6f}, "
        f"{gap[j] / scale[j]:+.2f} standard errors"
        for j in np.flatnonzero(~within)
    ]
    assert not failures, f"{move} at {fraction} beta_c: " + "; ".join(failures)


@pytest.mark.oracle
@pytest.mark.release
def test_the_cross_check_refutes_a_chain_at_a_coupling_five_percent_off() -> None:
    # The power of the test: Swendsen-Wang at 1.05 of the coupling judged
    # against Glauber at it, on the per-pull-request corner with forbidden
    # labels, at the transition.
    sigma, forbidden = PER_PR[0]
    instance = _instance(64, sigma, forbidden)
    reference = _glauber(64, sigma, forbidden, 1.0)
    shifted = _draws(instance, PottsMove.SWENDSEN_WANG, 1.05)
    mean_a, error_a, _ = _estimates(reference.series, "glauber")
    mean_b, error_b, _ = _estimates(shifted.series, "shifted")
    z = (mean_b - mean_a) / np.sqrt(error_a**2 + error_b**2)
    assert np.abs(z[0]) > SIGMAS, f"energy gap {z[0]:+.2f} standard errors"


@pytest.mark.oracle
@pytest.mark.release
@pytest.mark.parametrize("fraction", FRACTIONS)
@pytest.mark.parametrize(("sigma", "forbidden"), PER_PR)
@pytest.mark.parametrize("move", MOVES, ids=str)
def test_cluster_move_agrees_with_glauber_on_64_squared(
    move: PottsMove, sigma: float, forbidden: float, fraction: float
) -> None:
    _cross_check(64, sigma, forbidden, fraction, move)


@pytest.mark.oracle
@pytest.mark.release
@pytest.mark.parametrize("fraction", FRACTIONS)
@pytest.mark.parametrize(("sigma", "forbidden"), RELEASE)
@pytest.mark.parametrize("move", MOVES, ids=str)
def test_cluster_move_agrees_with_glauber_on_the_release_corners_of_64_squared(
    move: PottsMove, sigma: float, forbidden: float, fraction: float
) -> None:
    _cross_check(64, sigma, forbidden, fraction, move)


@pytest.mark.oracle
@pytest.mark.release
@pytest.mark.parametrize("fraction", FRACTIONS)
@pytest.mark.parametrize(("sigma", "forbidden"), PER_PR + RELEASE)
@pytest.mark.parametrize("move", MOVES, ids=str)
def test_cluster_move_agrees_with_glauber_on_128_squared(
    move: PottsMove, sigma: float, forbidden: float, fraction: float
) -> None:
    _cross_check(128, sigma, forbidden, fraction, move)
