"""The schedule seam: the alternatives, what each guarantees, and every graph.

Issue #592. Three things are checked and they are different things. That the
refactor changed no number --- the two schedules that predate it reproduce
their old output bitwise, which the rest of
`tests/regression/likelihood/test_message_passing.py` already asserts against
the dictionary reference. That the new schedules are *right*, which for the
upward pass means agreeing with an oracle written in another module
(`likelihood.pruning`) and, for the iterative pair, reaching one fixed point.
And that the seam reaches the graphs it claims to, which is the last test:
every adapter in `sim.factor_graph`, run under every schedule its graph
admits.

A partial schedule is the interesting case. `upward` is exact where it speaks
and silent elsewhere, so the test asserts the silence too: a marginal it does
not compute is absent, and reading one raises rather than returning a number.
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.likelihood.message_passing import sum_product
from snakes_and_ladders.likelihood.pruning import log_likelihood
from snakes_and_ladders.likelihood.schedule import (
    SCHEDULES,
    DownwardSchedule,
    FloodingSchedule,
    Guarantee,
    MessageSchedule,
    SequentialSchedule,
    TreeSchedule,
    UpwardSchedule,
    resolve,
)
from snakes_and_ladders.sim.factor_graph import (
    Factor,
    FactorGraph,
    Variable,
    from_potts,
    from_tree,
)
from snakes_and_ladders.sim.graph import BoundaryCondition, lattice_graph
from snakes_and_ladders.sim.jc import jc_transition_probabilities
from snakes_and_ladders.sim.simulate import simulate_alignment
from snakes_and_ladders.sim.tree import preorder

from tests._fixtures import SMALL_SITES, load_fixture

TREE_SCHEDULES = ("tree", "upward", "downward")
LOOPY_SCHEDULES = ("flooding", "sequential")


def _chain(n: int, cardinality: int = 3, seed: int = 5) -> FactorGraph:
    rng = np.random.default_rng(seed)
    return FactorGraph(
        [Variable(f"v{i}", cardinality) for i in range(n)],
        [
            Factor(
                f"f{i}",
                (f"v{i}", f"v{i + 1}"),
                rng.normal(size=(cardinality, cardinality)),
            )
            for i in range(n - 1)
        ],
    )


def _star(n: int, cardinality: int = 3, seed: int = 7) -> FactorGraph:
    """A tree whose root has degree ``n``, which a chain never exercises."""
    rng = np.random.default_rng(seed)
    return FactorGraph(
        [Variable("hub", cardinality)]
        + [Variable(f"v{i}", cardinality) for i in range(n)],
        [
            Factor(
                f"f{i}",
                ("hub", f"v{i}"),
                rng.normal(size=(cardinality, cardinality)),
            )
            for i in range(n)
        ],
    )


def _lattice(extent: int = 3, cardinality: int = 2) -> FactorGraph:
    rng = np.random.default_rng(11)
    graph = lattice_graph((extent, extent), BoundaryCondition.OPEN, 0.4)
    return from_potts(graph, rng.normal(size=(graph.n_nodes, cardinality)))


@pytest.mark.structural
def test_every_registered_name_resolves_to_its_class() -> None:
    assert set(SCHEDULES) == {str(member) for member in MessageSchedule}
    for kind in (
        TreeSchedule,
        UpwardSchedule,
        DownwardSchedule,
        FloodingSchedule,
        SequentialSchedule,
    ):
        assert isinstance(resolve(kind().name), kind)
        assert resolve(kind()) is not None


@pytest.mark.edge_case
def test_a_schedule_named_by_a_plain_string_is_the_one_asked_for() -> None:
    # Before the seam, `sum_product` compared the argument with `is` against a
    # `StrEnum` member, so a plain string --- which compares *equal* to one but
    # is not it --- silently fell through to the other branch and ran a
    # different algorithm. `resolve` makes the two spellings one call.
    graph = _chain(6)

    named = sum_product(graph, schedule="tree")
    member = sum_product(graph, schedule=MessageSchedule.TREE)
    instance = sum_product(graph, schedule=TreeSchedule())

    assert named.log_partition == member.log_partition == instance.log_partition
    assert named.guarantee is Guarantee.EXACT


@pytest.mark.edge_case
def test_an_unregistered_schedule_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="unknown schedule 'gibbs'"):
        sum_product(_chain(4), schedule="gibbs")


@pytest.mark.oracle
@pytest.mark.parametrize(
    "graph", [_chain(8), _star(12), _chain(3)], ids=["chain", "star", "triple"]
)
def test_the_upward_pass_alone_recovers_the_log_partition(graph: FactorGraph) -> None:
    # Felsenstein pruning is the leaf-to-root pass, and pruning returns the
    # likelihood: so half the messages must give the whole of `log Z`. The two
    # reach it by different arithmetic --- the two-pass schedule sums a Bethe
    # free energy over the beliefs, the upward pass sums the normalizers it
    # discarded --- so they agree to rounding and not bitwise. Measured at one
    # unit in the last place on these three; the bound is well inside it.
    both = sum_product(graph, schedule="tree")
    half = sum_product(graph, schedule="upward")

    assert half.log_partition == pytest.approx(both.log_partition, rel=1e-12)
    assert half.guarantee is Guarantee.PARTIAL


@pytest.mark.structural
def test_the_upward_pass_defines_the_root_marginal_and_no_other() -> None:
    graph = _star(6)

    half = sum_product(graph, schedule="upward")
    whole = sum_product(graph, schedule="tree")

    assert set(half.variable) == {"hub"}
    np.testing.assert_allclose(half.variable["hub"], whole.variable["hub"], atol=1e-12)
    with pytest.raises(KeyError):
        half.variable["v0"]


@pytest.mark.structural
def test_the_downward_pass_reports_no_log_partition_rather_than_a_wrong_one() -> None:
    # It has seen no evidence from below, so neither route to `log Z` is open.
    # A number here would be a number for something it did not compute.
    half = sum_product(_chain(5), schedule="downward")

    assert np.isnan(half.log_partition)
    assert half.guarantee is Guarantee.PARTIAL
    assert set(half.variable) == {f"v{i}" for i in range(5)}


@pytest.mark.mathematical
@pytest.mark.parametrize("schedule", LOOPY_SCHEDULES)
def test_an_iterative_schedule_reaches_the_exact_answer_on_a_tree(
    schedule: str,
) -> None:
    graph = _chain(7)
    exact = sum_product(graph, schedule="tree")

    settled = sum_product(graph, schedule=schedule, max_iterations=2000)

    assert settled.log_partition == pytest.approx(exact.log_partition, abs=1e-9)
    for name, row in exact.variable.items():
        np.testing.assert_allclose(settled.variable[name], row, atol=1e-8)


@pytest.mark.mathematical
def test_the_two_iterative_schedules_find_the_same_fixed_point_on_a_loopy_graph() -> (
    None
):
    # Jacobi and Gauss-Seidel over the same update have the same stationary
    # points; only the path differs. On a graph where both settle, the answers
    # must agree -- if they did not, one of them is not computing Bethe.
    graph = _lattice()

    jacobi = sum_product(graph, schedule="flooding", max_iterations=5000)
    seidel = sum_product(graph, schedule="sequential", max_iterations=5000)

    assert seidel.log_partition == pytest.approx(jacobi.log_partition, abs=1e-6)
    for name, row in jacobi.variable.items():
        np.testing.assert_allclose(seidel.variable[name], row, atol=1e-5)


@pytest.mark.edge_case
@pytest.mark.parametrize("schedule", TREE_SCHEDULES)
def test_a_tree_schedule_is_refused_on_a_loopy_graph(schedule: str) -> None:
    with pytest.raises(ValueError, match="exact only on a tree"):
        sum_product(_lattice(), schedule=schedule)


@pytest.mark.structural
@pytest.mark.parametrize("schedule", TREE_SCHEDULES + LOOPY_SCHEDULES)
def test_every_schedule_runs_on_a_tree_adapted_from_another_problem(
    schedule: str,
) -> None:
    # `sim.factor_graph` adapts six problem types onto one representation, so
    # the seam reaches all of them without a seventh adapter. A chain built
    # through `from_potts` is a Potts model to its own module and a factor
    # graph here, and every schedule must take it.
    graph = from_potts(
        lattice_graph((6,), BoundaryCondition.OPEN, 0.5),
        np.random.default_rng(3).normal(size=(6, 2)),
    )

    marginals = sum_product(graph, schedule=schedule, max_iterations=2000)

    assert marginals.guarantee is resolve(schedule).guarantee
    assert set(marginals.variable) <= {v.name for v in graph.variables}


@pytest.mark.oracle
def test_the_upward_pass_is_felsenstein_pruning_on_a_real_tree() -> None:
    # The strongest claim in #592, against an oracle that shares no code with
    # the schedule: `likelihood.pruning.log_likelihood` is the leaf-to-root
    # recursion written directly on the tree, and the upward schedule is that
    # recursion expressed as half a message-passing plan. Summed over sites
    # they are the same likelihood.
    params = load_fixture(SMALL_SITES)
    dataset = simulate_alignment(
        params.tau, params.k, params.pi, np.random.default_rng(params.seed), n_sites=7
    )
    alignment = dict(dataset.alignment)
    transitions = {
        node.name: jc_transition_probabilities(node.branch_length, params.k)
        for node in preorder(params.tau)
        if node.branch_length is not None
    }

    total = 0.0
    for site_index in range(7):
        site = {name: int(states[site_index]) for name, states in alignment.items()}
        graph = from_tree(params.tau, params.k, params.pi, site, transitions)
        half = sum_product(graph, schedule="upward")
        assert half.guarantee is Guarantee.PARTIAL
        total += half.log_partition

    assert total == pytest.approx(
        log_likelihood(params.tau, params.k, params.pi, alignment), rel=1e-13
    )
