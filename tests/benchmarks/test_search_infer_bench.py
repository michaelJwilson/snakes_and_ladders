"""Benchmarks for topology search.

See tests/regression/test_search_infer.py for correctness. The unit that
matters is one candidate fit, because that is what the budget counts and
where a search spends its time: generating a neighbourhood is combinatorics
on a handful of nodes, while fitting one candidate is an optimization. Both
are measured here, so the ratio between them says whether that assumption
still holds.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.search import infer as infer_module
from snakes_and_ladders.search.infer import (
    Model,
    MoveSet,
    infer,
    parsimony_search,
    score_topology,
)
from snakes_and_ladders.search.topology import (
    nni_neighbours,
    random_topology,
    spr_neighbours,
)
from snakes_and_ladders.sim.simulate import simulate_alignment

from tests._fixtures import EIGHT_TAXA, SMALL_SITES, load_fixture

_SITES = 2000


def _alignment() -> tuple[dict[str, np.ndarray], int]:
    params = load_fixture(SMALL_SITES)
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        rng=np.random.default_rng(params.seed),
        n_sites=_SITES,
    )
    return dict(dataset.alignment), params.k


def test_one_candidate_fit_benchmark(benchmark: BenchmarkFixture) -> None:
    """The unit the search budget is denominated in."""
    alignment, k = _alignment()
    topology = random_topology(sorted(alignment), np.random.default_rng(1))

    result = benchmark(score_topology, topology, alignment, k)

    # Benchmarks assert finiteness only; correctness is pinned in
    # tests/regression/test_search_infer.py.
    assert math.isfinite(result)
    assert result < 0.0


def test_neighbourhood_generation_benchmark(benchmark: BenchmarkFixture) -> None:
    """Move generation alone, to show it is not where the time goes."""
    alignment, _ = _alignment()
    topology = random_topology(sorted(alignment), np.random.default_rng(1))

    neighbours = benchmark(lambda: list(nni_neighbours(topology)))

    assert len(neighbours) > 0


def test_hill_climb_benchmark(benchmark: BenchmarkFixture) -> None:
    """A whole search, so the per-fit number can be checked against a run."""
    alignment, k = _alignment()

    result = benchmark(
        infer, alignment, k, rng=np.random.default_rng(1), moves=MoveSet.NNI
    )

    assert result.converged
    assert math.isfinite(result.log_likelihood)


# --- what carries between neighbours (issue #289) --------------------------


def _eight_taxa() -> tuple[dict[str, np.ndarray], int]:
    params = load_fixture(EIGHT_TAXA)
    dataset = simulate_alignment(
        params.tau,
        params.k,
        params.pi,
        np.random.default_rng(params.seed),
        n_sites=1000,
    )
    return dict(dataset.alignment), params.k


@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
def test_one_neighbour_fit_benchmark(benchmark: BenchmarkFixture, warm: bool) -> None:
    """One SPR neighbour fitted cold, and from its parent's lengths."""
    alignment, k = _eight_taxa()
    start = random_topology(sorted(alignment), np.random.default_rng(1))
    parent = infer_module._score(Model.JC, start, k, alignment)
    neighbour = next(iter(spr_neighbours(start)))

    fitted = benchmark(
        infer_module._score, Model.JC, neighbour, k, alignment, parent if warm else None
    )

    benchmark.extra_info["likelihood_evaluations"] = fitted.evaluations
    assert math.isfinite(fitted.value)


# 113.8 s cold, 68.1 s warm, 7.3 s lazy on the reference host, down from
# 180.2 s, 117.6 s and 10.6 s (issue #397). Every case stays at the release
# gate, for two different reasons.
#
# Cold and warm are over the 10 s per-pull-request cap by an order of
# magnitude, and nothing found closes that. The evaluation is bound by its
# backward pass -- 59% of one forward-and-backward after the change, against
# 5% for the leaf gather it now uses -- and what is left of that is the
# autograd graph of a fit that has to be differentiable. The next step is a
# hand-written gradient, not another rewrite of the forward pass.
#
# Lazy is 7.3 s and under the cap, and stays here anyway. Of that, 5.0 s is
# the five rounds `pytest-benchmark` timed and 2.3 s is building the fixture;
# a round ranged 0.60 s to 1.30 s, so a session of worst rounds lands near
# 8.8 s, 12% inside the cap. Spending 7.3 s of the class's 300 s budget for
# that margin buys nothing the tier does not already have:
# `test_hill_climb_benchmark` above runs the same search per pull request at
# the smaller fixture, and `tests/regression/search/test_search_exhaustive.py`
# holds the optimum against enumeration.
@pytest.mark.release
@pytest.mark.parametrize(
    ("warm_start", "lazy_top"),
    [(False, None), (True, None), (True, 1)],
    ids=["cold", "warm", "warm-lazy1"],
)
def test_nni_hill_climb_eight_taxa_benchmark(
    benchmark: BenchmarkFixture, warm_start: bool, lazy_top: int | None
) -> None:
    """The whole NNI search at eight taxa, in the three ways it can be run.

    Wall-clock beside the counts the search itself reports, so the ratio can
    be read in either unit -- and the unit that matters is the counts, which
    issue #397 left unchanged while wall clock fell by a third: the search
    reaches the same optimum in the same 80 evaluations, 81 fits and 9 moves.
    Correctness is `tests/regression/search/test_search_infer.py`'s, where
    enumeration over the eight-taxon topologies referees the optimum.
    """
    alignment, k = _eight_taxa()

    result = benchmark(
        infer,
        alignment,
        k,
        rng=np.random.default_rng(1),
        moves=MoveSet.NNI,
        max_evaluations=300,
        warm_start=warm_start,
        lazy_top=lazy_top,
    )

    benchmark.extra_info["fits"] = result.fits
    benchmark.extra_info["likelihood_evaluations"] = result.likelihood_evaluations
    assert result.converged


@pytest.mark.parametrize("moves", [MoveSet.NNI, MoveSet.SPR])
def test_parsimony_hill_climb_eight_taxa_benchmark(
    benchmark: BenchmarkFixture, moves: MoveSet
) -> None:
    """The same climb on the Fitch score (issue #335), where a candidate is one pass and not a fit."""
    alignment, k = _eight_taxa()

    result = benchmark(
        parsimony_search,
        alignment,
        k,
        rng=np.random.default_rng(1),
        moves=moves,
        max_evaluations=1000,
    )

    benchmark.extra_info["evaluations"] = result.evaluations
    assert result.converged
