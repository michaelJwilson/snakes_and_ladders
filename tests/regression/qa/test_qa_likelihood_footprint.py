"""The footprint table publishes a model; this is what pins it to measurement.

`phylo.qa.likelihood_footprint` computes what the arrays occupy from their
shapes, because `docs/CLAUDE.md` admits only a number that survives a rebuild on
another machine and a ``tracemalloc`` peak does not — an earlier draft published
one and the continuous-integration runner regenerated a different table. That
moves the burden here: a model nothing checks is arithmetic, not a measurement,
so every published figure is compared against the allocator's own count.

The comparison is a ratio within a stated tolerance rather than an equality. The
model omits the transient a matrix product allocates and the interpreter's own
overhead, both bounded and neither worth modelling; what matters is that nothing
*unaccounted for* dominates, and a 5% band says so while catching a term left
out.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest
from phylo.likelihood import pruning
from phylo.qa.likelihood_footprint import (
    DECLARED_MAXIMUM,
    MEASURED_SIZES,
    MEMORY_BUDGET_BYTES,
    caterpillar,
    evaluation_bytes,
    measure,
    simulation_bytes,
    warm_up,
)
from phylo.search.rl import with_uniform_branch_lengths
from phylo.sim.simulate import simulate_alignment
from phylo.sim.tree import Node

#: The band the model is held to. Wide enough for the transients it does not
#: model, narrow enough that a missing array term fails: at 4 states the
#: evaluator's own array is 8 times the simulator's, so omitting either shows
#: as a factor, not a percentage.
TOLERANCE = 0.05

FIXED_TAXA = 16
FIXED_SITES = 2_000


@pytest.fixture(autouse=True, scope="module")
def _warmed() -> None:
    """Absorb the cold-call inflation before any assertion reads a peak.

    `phylo.qa.likelihood_footprint.warm_up` states why; calling it from here
    rather than restating the reason keeps one home for the fact.
    """
    warm_up()


def _balanced(n_taxa: int) -> Node:
    """A balanced binary topology on ``n_taxa`` leaves, a power of two.

    The shallow counterpart of `caterpillar`: depth ``log2(n_taxa)`` against
    ``n_taxa - 2``, which is what makes it the cheaper case for an evaluator
    holding one partial per open node.
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


@pytest.mark.parametrize(
    "size", MEASURED_SIZES, ids=lambda s: f"{s[0]}taxa_{s[1]}sites"
)
def test_the_published_simulation_figure_matches_the_allocator(
    size: tuple[int, int],
) -> None:
    """Every cell of the Simulate column, against `tracemalloc`.

    The model is `(2n - 1) x L x 8`: the simulator retains every node's states,
    internal ones included, because the ancestral truth is what the validation
    tests compare against. A simulator that kept only the leaves would come in
    at half this and fail here rather than quietly making the table wrong.
    """
    measured, _ = measure(*size)
    assert measured / simulation_bytes(*size) == pytest.approx(1.0, rel=TOLERANCE)


@pytest.mark.parametrize(
    "size", MEASURED_SIZES, ids=lambda s: f"{s[0]}taxa_{s[1]}sites"
)
def test_the_published_evaluation_figure_matches_the_allocator(
    size: tuple[int, int],
) -> None:
    """Every cell of the Evaluate column, against `tracemalloc`.

    The model is `(2n - 2) x L x k x 8`: on a caterpillar every node but the
    root is open at the deepest point of the post-order, which is the claim
    that makes this the worst case and the table a bound.
    """
    _, measured = measure(*size)
    assert measured / evaluation_bytes(*size) == pytest.approx(1.0, rel=TOLERANCE)


def test_a_balanced_topology_costs_strictly_less_than_the_caterpillar() -> None:
    """The table reports the worst case, and this is what says so.

    Same taxa, same sites, same data volume: only the depth differs. If a
    balanced tree were not cheaper, the caterpillar would not be the bound the
    caption claims and the last row would understate the requirement.
    """
    pi = np.full(4, 0.25)
    peaks: list[float] = []
    for topology in (_balanced(FIXED_TAXA), caterpillar(FIXED_TAXA)):
        tau = with_uniform_branch_lengths(topology, 0.1)
        alignment = dict(
            simulate_alignment(
                tau=tau, k=4, pi=pi, seed=1, n_sites=FIXED_SITES
            ).alignment
        )
        tracemalloc.start()
        pruning.log_likelihood(tau, 4, pi, alignment)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks.append(float(peak))

    balanced_peak, caterpillar_peak = peaks
    assert balanced_peak < caterpillar_peak


def test_the_declared_maximum_sits_inside_the_memory_requirement() -> None:
    """The bound `ROADMAP.md` §1.2 states, at the corner it states it for.

    Asserted as an order of magnitude of headroom rather than an exact figure:
    the margin is what the requirement is about, and a tight assertion here
    would fail for a change to the representation rather than for a change that
    breaks the requirement.
    """
    taxa, sites = DECLARED_MAXIMUM
    total = simulation_bytes(taxa, sites) + evaluation_bytes(taxa, sites)
    assert total < MEMORY_BUDGET_BYTES / 10


def test_the_check_would_fail_on_a_model_missing_a_term() -> None:
    """The tolerance rejects the error it exists to reject.

    A 5% band is only a check if a plausible mistake lands outside it. Dropping
    the internal nodes from the simulator, or the state axis from the
    evaluator, are the two mistakes available, and both are factors.
    """
    taxa, sites = 20, 2_000
    leaves_only = taxa * sites * 8
    without_states = (2 * taxa - 2) * sites * 8

    assert leaves_only / simulation_bytes(taxa, sites) < 1.0 - TOLERANCE
    assert without_states / evaluation_bytes(taxa, sites) < 1.0 - TOLERANCE
