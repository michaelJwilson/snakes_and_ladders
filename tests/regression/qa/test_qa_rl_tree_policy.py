"""The number `rl_tree_policy` plots, pinned (issue #729, step 4).

The renderer had 48 of 75 statements unreached. With no oracle of its own
(the rates are refereed in `learn.reinforce` and `learn.rollout`), pinned:
the rates `STATUS.md` and `docs/experiments/005-tree-policy-features.md`
record on the 7-taxon fixture, policy 0.487 against greedy 0.480. `release`:
eight policies at 640 episodes on `tree_search/release.yaml` take 92.6 s
(the manifest pays 101.4 s); `ci` takes 20.2 s, over the 10 s cap, and pins
nothing, since at five taxa everything reaches the maximum (issue #177).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest
from sal.qa.rl_tree_policy import (
    BATCH,
    ITERATIONS,
    ROLLOUTS_PER_START,
    STARTS,
    TRAINING_SEEDS,
    measure,
)
from sal.sim.fixtures import fixture

pytestmark = pytest.mark.release

FIXTURE = fixture("tree_search", "release")


class Comparison(NamedTuple):
    """What `measure` returns, named so the tests below read."""

    scores: np.ndarray
    optima: np.ndarray
    greedy: float
    learned: list[float]
    episodes: int
    taxa: int
    control: float


@pytest.fixture(scope="module")
def comparison() -> Comparison:
    """One comparison, run once when selected; at module level collection would pay it."""
    return Comparison(*measure(FIXTURE.params))


@pytest.mark.smoke
@pytest.mark.snapshot
def test_panel_b_plots_the_rates_the_experiment_record_reports(
    comparison: Comparison,
) -> None:
    """The figure's answer, to be conserved.

    8 seeds x 50 starts x 16 rollouts: 0.486875 against 0.480000 (record 0.487,
    3e-4 off). Per seed by equality: integer counts of 800 episodes.
    """
    assert (TRAINING_SEEDS, STARTS, ROLLOUTS_PER_START) == (8, 50, 16)
    assert comparison.episodes == ITERATIONS * BATCH == 640

    assert comparison.greedy == 0.48
    assert [round(rate * 800) for rate in comparison.learned] == [
        398,
        375,
        373,
        413,
        396,
        402,
        375,
        384,
    ]
    assert float(np.mean(comparison.learned)) == pytest.approx(0.486875, abs=0.0)
    assert float(np.mean(comparison.learned)) == pytest.approx(0.487, abs=3e-4)


@pytest.mark.smoke
@pytest.mark.snapshot
def test_the_untrained_control_separates_learning_from_the_baseline(
    comparison: Comparison,
) -> None:
    """The control the caption reads, to be conserved.

    Untrained: 0.0175 of 800 episodes, 28x below; "ties greedy" needs it.
    """
    assert round(comparison.control * 800) == 14
    assert comparison.control == pytest.approx(0.0175, abs=0.0)
    assert float(np.mean(comparison.learned)) > 25.0 * comparison.control


@pytest.mark.smoke
@pytest.mark.snapshot
def test_panel_a_plots_the_enumerated_surface_and_its_local_optima(
    comparison: Comparison,
) -> None:
    """The environment panel (a) draws, to be conserved.

    945 topologies, 10 NNI local optima, best -16433.2666633598 (1e-12 relative).
    """
    assert comparison.taxa == 7
    assert comparison.scores.shape == (945,)
    assert comparison.optima.shape == (10,)

    assert float(comparison.scores[-1]) == pytest.approx(-16433.2666633598, rel=1e-12)
    assert float(comparison.optima[-1]) == pytest.approx(
        float(comparison.scores[-1]), rel=1e-12
    )
    assert bool(np.all(np.diff(comparison.scores) >= 0.0))
