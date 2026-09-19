"""The number `rl_tree_policy` plots, pinned (issue #729, step 4).

The renderer had 75 statements, 48 of them reached by nothing and the rest
by another module's import. A renderer has no oracle --- the rates are what
`learn.reinforce` and `learn.rollout` produced, and those are held to
enumeration in their own modules --- so what is pinned here is that the
figure keeps plotting the rates `STATUS.md` and
`docs/experiments/005-tree-policy-features.md` record for the improvement
feature on the 7-taxon fixture: the policy at **0.487** against greedy's
**0.480**, a tie, which is the negative result the figure exists to report.

**The instance is the rendered one, so the test is `release`.** `measure`
trains eight policies at 640 episodes each on `tree_search/release.yaml`,
which is 92.6 s on the 4-core reference host --- the same run the manifest
pays 101.4 s for. The `ci` instance would run in 20.2 s and still not fit
`DEV.md`'s 10 s per-PR cap, and it pins nothing: at five taxa greedy and
every trained policy reach the maximum from all 50 starts, which is the
case issue #177 built this fixture to escape.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest
from snakes_and_ladders.qa.rl_tree_policy import (
    BATCH,
    ITERATIONS,
    ROLLOUTS_PER_START,
    STARTS,
    TRAINING_SEEDS,
    measure,
)
from snakes_and_ladders.sim.fixtures import fixture

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
    """One comparison, run once, and only where a test of it is selected.

    A module-level call would be paid at *collection*, so every per-pull-request
    run would train the eight policies this module is `release` to avoid.
    """
    return Comparison(*measure(FIXTURE.params))


@pytest.mark.smoke
@pytest.mark.snapshot
def test_panel_b_plots_the_rates_the_experiment_record_reports(
    comparison: Comparison,
) -> None:
    """The figure's answer, to be conserved.

    Over eight training seeds at 640 episodes, 50 shared starts and 16
    rollouts each, the learned policy reaches the enumerated maximum from
    **0.486875** of its episodes against greedy's **0.480000**. The
    experiment record's sixteen seeds give 0.487 against 0.480, so the eight
    drawn here reproduce it to **3e-4**. Pinned per seed by equality: each
    rate is an integer count of 800 episodes and admits no tolerance.
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

    An untrained policy reaches the maximum from **0.0175** of its 800
    episodes, 28 times below the trained one. Without it "the policy ties
    greedy" would be consistent with a policy that learned nothing at all,
    since greedy is what the single feature amounts to.
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

    All **945** unrooted topologies of the 7-taxon fixture, scored at the
    generating tree's mean branch length, of which **10** admit no improving
    NNI move and are where every episode ends. The best score is
    **-16433.2666633598**, pinned to 1e-12 relative, and it is the value
    both rates above are the frequency of reaching.
    """
    assert comparison.taxa == 7
    assert comparison.scores.shape == (945,)
    assert comparison.optima.shape == (10,)

    assert float(comparison.scores[-1]) == pytest.approx(-16433.2666633598, rel=1e-12)
    assert float(comparison.optima[-1]) == pytest.approx(
        float(comparison.scores[-1]), rel=1e-12
    )
    assert bool(np.all(np.diff(comparison.scores) >= 0.0))
