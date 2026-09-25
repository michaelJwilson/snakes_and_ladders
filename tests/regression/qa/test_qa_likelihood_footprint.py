"""The footprint table publishes a model; this is what pins it to measurement.

The table is computed from shapes, since a ``tracemalloc`` peak does not
survive a rebuild on another machine (`docs/CLAUDE.md`), so every figure is
compared against the allocator's count within 5%: matrix-product transients
and interpreter overhead are omitted, and a missing term falls outside the band.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest
from sal.learn.tree import with_uniform_branch_lengths
from sal.likelihood import pruning
from sal.qa.likelihood_footprint import (
    DECLARED_MAXIMUM,
    MEASURED_SIZES,
    MEMORY_BUDGET_BYTES,
    caterpillar,
    evaluation_bytes,
    measure,
    simulation_bytes,
    warm_up,
)
from sal.sim.fixtures import fixture
from sal.sim.simulate import simulate_alignment
from sal.sim.tree import Node

from tests._rows import every_value

#: The band the model is held to. Wide enough for the transients it does not
#: model, narrow enough that a missing array term fails: at 4 states the
#: evaluator's own array is 8 times the simulator's, so omitting either shows
#: as a factor, not a percentage.
TOLERANCE = 0.05

FIXED_TAXA = 16
FIXED_SITES = 2_000

#: The alphabet the table is computed at, read from the tree fixture the
#: figure is rendered from rather than restated here.
N_STATES = fixture("tree_jc", "ci").params.k


@pytest.fixture(autouse=True, scope="module")
def _warmed() -> None:
    """Absorb the cold-call inflation before any assertion reads a peak (see `warm_up`)."""
    warm_up()


def _balanced(n_taxa: int) -> Node:
    """A balanced binary topology on ``n_taxa`` leaves, a power of two.

    Depth ``log2(n_taxa)`` against the caterpillar's ``n_taxa - 2``.
    """
    level: list[Node] = [
        Node(name=f"t{index}", branch_length=None) for index in range(n_taxa)
    ]
    while len(level) > 3:
        level = [
            Node(name="i", branch_length=None, children=(left, right))
            for left, right in zip(level[::2], level[1::2], strict=True)
        ]
    return Node(name="root", branch_length=None, children=tuple(level))


@pytest.mark.oracle
def test_the_published_simulation_figure_matches_the_allocator() -> None:
    """Every cell of the Simulate column, against `tracemalloc`.

    `(2n - 1) x L x 8`: internal states are kept for the ancestral truth.
    """

    def check(size: tuple[int, int]) -> None:
        measured, _ = measure(*size, N_STATES)
        assert measured / simulation_bytes(*size) == pytest.approx(1.0, rel=TOLERANCE)

    every_value(MEASURED_SIZES, check)


@pytest.mark.oracle
def test_the_published_evaluation_figure_matches_the_allocator() -> None:
    """Every cell of the Evaluate column, against `tracemalloc`.

    `(2n - 2) x L x k x 8`: on a caterpillar all but the root are open at once.
    """

    def check(size: tuple[int, int]) -> None:
        _, measured = measure(*size, N_STATES)
        assert measured / evaluation_bytes(*size, N_STATES) == pytest.approx(
            1.0, rel=TOLERANCE
        )

    every_value(MEASURED_SIZES, check)


@pytest.mark.analytic
def test_a_balanced_topology_costs_strictly_less_than_the_caterpillar() -> None:
    """The table reports the worst case, and this is what says so.

    Same taxa, sites and data; only depth differs.
    """
    pi = np.full(4, 0.25)
    peaks: list[float] = []
    for topology in (_balanced(FIXED_TAXA), caterpillar(FIXED_TAXA)):
        tau = with_uniform_branch_lengths(topology, 0.1)
        alignment = dict(
            simulate_alignment(
                tau=tau, k=4, pi=pi, rng=np.random.default_rng(1), n_sites=FIXED_SITES
            ).alignment
        )
        tracemalloc.start()
        pruning.log_likelihood(tau, 4, pi, alignment)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks.append(float(peak))

    balanced_peak, caterpillar_peak = peaks
    assert balanced_peak < caterpillar_peak


@pytest.mark.smoke
def test_the_declared_maximum_sits_inside_the_memory_requirement() -> None:
    """The bound `ROADMAP.md` §1.2 states, at the corner it states it for.

    An order of magnitude of headroom, not an exact figure.
    """
    taxa, sites = DECLARED_MAXIMUM
    total = simulation_bytes(taxa, sites) + evaluation_bytes(taxa, sites, N_STATES)
    assert total < MEMORY_BUDGET_BYTES / 10


@pytest.mark.smoke
def test_the_check_would_fail_on_a_model_missing_a_term() -> None:
    """The tolerance rejects the error it exists to reject.

    Dropping internal nodes or the state axis are factors, outside 5%.
    """
    taxa, sites = 20, 2_000
    leaves_only = taxa * sites * 8
    without_states = (2 * taxa - 2) * sites * 8

    assert leaves_only / simulation_bytes(taxa, sites) < 1.0 - TOLERANCE
    assert without_states / evaluation_bytes(taxa, sites, N_STATES) < 1.0 - TOLERANCE
