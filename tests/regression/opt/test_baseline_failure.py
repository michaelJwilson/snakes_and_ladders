"""The sizing harness, against an analytic family and against #194's number.

Issue #596. Two kinds of check, because the harness makes two kinds of claim:

* **the arithmetic** is checked against a family whose success probability is
  written down, so the size the sweep reports is the size the binomial
  predicts, and a probe that counted its starts or its reaches wrongly fails
  here rather than on a research problem;
* **the use** is checked by reproducing the measurement that motivated the
  ticket --- random-restart hill climbing at **1.000** on the 7-taxon tree,
  60 decisions, against a single descent's 0.48 (`STATUS.md`, issue #194). A
  harness that cannot reproduce the finding it was built for is not measuring
  what it claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from snakes_and_ladders.learn.failure import (
    FailureCurve,
    failure_curve,
    probe_failure,
)
from snakes_and_ladders.learn.rollout import greedy_rollout
from snakes_and_ladders.opt.budget import Budget, Outcome, OverspendError, restarts
from snakes_and_ladders.search.infer import MoveSet
from snakes_and_ladders.search.rl import RewardModel, TreeEnvironment
from snakes_and_ladders.search.topology import enumerate_topologies
from snakes_and_ladders.sim.params import SimulationParams, load_simulation_params
from snakes_and_ladders.sim.simulate import simulate_alignment
from snakes_and_ladders.sim.tree import edges

#: The instance #194 measured, and the budget it measured at: a greedy descent
#: stops after about four decisions, so 60 buys roughly fifteen restarts.
FIXTURE = Path("tests/regression/fixtures/tree_search/release.yaml")
DECISIONS = 60
DESCENT_COST = 4


@dataclass(frozen=True)
class _Basin:
    """An instance whose basin fraction is declared rather than measured."""

    fraction: float


def _draw(instance: _Basin, budget: Budget, rng: np.random.Generator) -> Outcome:
    """Reach the target with the declared probability, spending the budget.

    Module level and not a closure, so it is picklable the way
    `opt/budget.py` requires of a method.
    """
    reached = float(rng.random()) < instance.fraction
    return Outcome(0.0 if reached else 1.0, budget.size)


def _overspender(instance: _Basin, budget: Budget, rng: np.random.Generator) -> Outcome:
    """A method that reports more than it was given."""
    del instance, rng
    return Outcome(0.0, budget.size + 1)


def _family(size: int) -> tuple[_Basin, float]:
    """Basin fraction `0.5 ** (size - 1)`, and a target of zero."""
    return _Basin(0.5 ** (size - 1)), 0.0


def _never_fails(size: int) -> tuple[_Basin, float]:
    """A family whose baseline reaches the target from every start, at any size."""
    del size
    return _Basin(1.0), 0.0


@pytest.mark.analytic
def test_a_probe_counts_what_it_ran_and_what_it_reached() -> None:
    # The arithmetic, against a declared probability: at a basin fraction of
    # one every start reaches, at zero none does, and the fraction is over the
    # starts the probe ran rather than over the starts it was asked for.
    always = probe_failure(
        _draw,
        _Basin(1.0),
        target=0.0,
        budget=Budget("decisions", 10),
        starts=8,
        rng=np.random.default_rng(0),
        size=1,
    )
    assert (always.reached, always.starts, always.asked) == (8, 8, 8)
    assert always.fraction == 1.0
    assert not always.truncated

    never = probe_failure(
        _draw,
        _Basin(0.0),
        target=0.0,
        budget=Budget("decisions", 10),
        starts=8,
        rng=np.random.default_rng(0),
        size=1,
    )
    assert never.reached == 0
    assert never.fraction == 0.0
    assert never.best == 1.0


@pytest.mark.analytic
def test_the_sweep_stops_at_the_size_the_binomial_predicts() -> None:
    # `0.5 ** (size - 1)` is 1 at size 1, so the sweep cannot fail there; at
    # size 2 it is one half and 32 starts miss the target with probability
    # `1 - 0.5 ** 32`, so the first failure is size 2 for any seed. Asserting
    # the *size* rather than the fraction is what makes this a test of the
    # sweep and not of the draw.
    curve = failure_curve(
        _draw,
        _family,
        sizes=(1, 2, 3, 4),
        budget=Budget("decisions", 10),
        starts=32,
        rng=np.random.default_rng(596),
    )

    assert isinstance(curve, FailureCurve)
    failed = curve.first_failure
    assert failed is not None
    assert failed.size == 2
    assert len(curve.probes) == 2, "the sweep stops at the first failure"
    assert curve.probes[0].fraction == 1.0


@pytest.mark.analytic
def test_the_whole_curve_is_available_when_the_sweep_is_asked_for_it() -> None:
    # A report carries the curve, not only its first failure, and the
    # fractions must fall with the size for a family whose basin halves.
    curve = failure_curve(
        _draw,
        _family,
        sizes=(1, 2, 3, 4),
        budget=Budget("decisions", 10),
        starts=64,
        rng=np.random.default_rng(596),
        stop_at_failure=False,
    )

    fractions = [probe.fraction for probe in curve.probes]
    assert len(fractions) == 4
    assert fractions[0] == 1.0
    assert fractions[-1] < fractions[1] < fractions[0]


@pytest.mark.smoke
def test_a_baseline_that_never_fails_reports_no_failure() -> None:
    # The outcome issue #596 says to bring the curve for: the baseline held at
    # every size measured, so the gate cannot be argued here. `None` is the
    # finding and not a missing result.
    curve = failure_curve(
        _draw,
        _never_fails,
        sizes=(1, 2, 3),
        budget=Budget("decisions", 10),
        starts=16,
        rng=np.random.default_rng(1),
    )

    assert curve.first_failure is None
    assert len(curve.probes) == 3


@pytest.mark.smoke
def test_the_clock_truncates_a_probe_and_the_probe_says_so() -> None:
    # A fraction read off a sample the run did not take is the silent failure
    # a ceiling would otherwise introduce, so a truncated probe reports the
    # starts it managed and marks itself.
    probe = probe_failure(
        _draw,
        _Basin(1.0),
        target=0.0,
        budget=Budget("decisions", 10),
        starts=10_000_000,
        rng=np.random.default_rng(0),
        size=1,
        seconds=0.05,
    )

    assert probe.truncated
    assert probe.starts < probe.asked
    assert probe.fraction == 1.0


@pytest.mark.smoke
def test_the_harness_refuses_what_it_cannot_measure() -> None:
    budget = Budget("decisions", 10)
    with pytest.raises(ValueError, match="at least one start"):
        probe_failure(
            _draw,
            _Basin(1.0),
            target=0.0,
            budget=budget,
            starts=0,
            rng=np.random.default_rng(0),
            size=1,
        )
    with pytest.raises(ValueError, match="ceiling is positive"):
        probe_failure(
            _draw,
            _Basin(1.0),
            target=0.0,
            budget=budget,
            starts=1,
            rng=np.random.default_rng(0),
            size=1,
            seconds=0.0,
        )
    with pytest.raises(ValueError, match="at least one size"):
        failure_curve(
            _draw,
            _family,
            sizes=(),
            budget=budget,
            starts=1,
            rng=np.random.default_rng(0),
        )
    with pytest.raises(ValueError, match=r"threshold lies in \(0, 1\]"):
        failure_curve(
            _draw,
            _family,
            sizes=(1,),
            budget=budget,
            starts=1,
            rng=np.random.default_rng(0),
            threshold=1.5,
        )
    with pytest.raises(OverspendError, match="against a budget"):
        probe_failure(
            _overspender,
            _Basin(1.0),
            target=0.0,
            budget=budget,
            starts=1,
            rng=np.random.default_rng(0),
            size=1,
        )


# --- the use: #194's measurement, reproduced through the harness -------------


def _tree_instance() -> tuple[TreeEnvironment, float]:
    """The 7-taxon environment and its enumerated maximum."""
    params: SimulationParams = load_simulation_params(FIXTURE)
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        rng=np.random.default_rng(params.seed),
        n_sites=params.n_sites,
    )
    environment = TreeEnvironment(
        dict(dataset.alignment),
        params.k,
        np.asarray(params.pi),
        branch_length=float(
            np.mean([child.branch_length for _, child in edges(params.tau)])
        ),
        reward=RewardModel.KNOWN,
        moves=MoveSet.NNI,
    )
    taxa = sorted(dataset.alignment)
    maximum = max(
        environment.score(topology) for topology in enumerate_topologies(taxa)
    )
    return environment, maximum


def _descent(
    instance: TreeEnvironment, budget: Budget, rng: np.random.Generator
) -> Outcome:
    """One greedy descent from a random start, scored as a negative likelihood.

    The harness minimizes, and the environment maximizes a log-likelihood, so
    the sign is flipped here rather than in the harness: an `Outcome` is lower
    is better everywhere in `opt/budget.py`.
    """
    state = instance.reset(rng)
    episode = greedy_rollout(instance, state, budget.size)
    best = max(instance.score(visited) for visited in episode.states)
    return Outcome(-best, max(len(episode.actions), 1))


@pytest.mark.release
@pytest.mark.oracle
def test_the_harness_reproduces_the_restart_baseline_on_the_seven_taxon_tree() -> None:
    # The finding that motivated issue #596, through the harness rather than
    # through a loop written for it: a single greedy descent reaches the
    # enumerated maximum from about half the starts and random restarts at the
    # same 60-decision budget reach it from all of them. So the baseline
    # Milestone 2.1 has to beat here is 1.000, and the harness says so.
    environment, maximum = _tree_instance()
    budget = Budget("decisions", DECISIONS)

    single = probe_failure(
        _descent,
        environment,
        target=-maximum,
        budget=budget,
        starts=50,
        rng=np.random.default_rng(596),
        size=7,
    )
    restarted = probe_failure(
        restarts(_descent, DESCENT_COST),
        environment,
        target=-maximum,
        budget=budget,
        starts=50,
        rng=np.random.default_rng(596),
        size=7,
    )

    assert 0.35 <= single.fraction <= 0.62, single.fraction
    assert restarted.fraction == 1.0
    assert restarted.fraction > single.fraction
