"""The schedule seam: the alternatives, what each guarantees, and every graph.

Issue #592. Three things are checked and they are different things. That the
refactor changed no number --- the two schedules that predate it reproduce
their old output bitwise, which
`tests/regression/likelihood/test_message_passing.py` asserts against the
dictionary reference. That the new schedules are *right*, which for the
upward pass means agreeing with an oracle written in another module
(`likelihood.pruning`) and, for the iterative three, reaching belief
propagation's fixed point: both are asserted in `test_message_passing.py`,
parametrised over the schedule (issue #982). And that the seam reaches the
graphs it claims to: every adapter in `sim.factor_graph`, run under every
schedule its graph admits.

A partial schedule is the interesting case. `upward` is exact where it speaks
and silent elsewhere, so the test asserts the silence too: a marginal it does
not compute is absent, and reading one raises rather than returning a number.
"""

from __future__ import annotations

from collections.abc import Generator

import numpy as np
import pytest
from snakes_and_ladders.likelihood.message_passing import sum_product
from snakes_and_ladders.likelihood.schedule import (
    SCHEDULES,
    DownwardMessageSchedule,
    FloodingMessageSchedule,
    Guarantee,
    Layout,
    MessageScheduleName,
    SequentialMessageSchedule,
    Step,
    TreeMessageSchedule,
    UpwardMessageSchedule,
    resolve,
)
from snakes_and_ladders.sim.factor_graph import (
    Factor,
    FactorGraph,
    Variable,
    from_potts,
)
from snakes_and_ladders.sim.graph import BoundaryCondition, lattice_graph

TREE_SCHEDULES = ("tree", "upward", "downward")
LOOPY_SCHEDULES = ("flooding", "sequential", "residual")


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


@pytest.mark.smoke
def test_every_registered_name_resolves_to_its_class() -> None:
    assert set(SCHEDULES) == {str(member) for member in MessageScheduleName}
    for kind in (
        TreeMessageSchedule,
        UpwardMessageSchedule,
        DownwardMessageSchedule,
        FloodingMessageSchedule,
        SequentialMessageSchedule,
    ):
        assert isinstance(resolve(kind().name), kind)
        assert resolve(kind()) is not None


@pytest.mark.smoke
def test_a_schedule_named_by_a_plain_string_is_the_one_asked_for() -> None:
    # Before the seam, `sum_product` compared the argument with `is` against a
    # `StrEnum` member, so a plain string --- which compares *equal* to one but
    # is not it --- silently fell through to the other branch and ran a
    # different algorithm. `resolve` makes the two spellings one call.
    graph = _chain(6)

    named = sum_product(graph, schedule="tree")
    member = sum_product(graph, schedule=MessageScheduleName.TREE)
    instance = sum_product(graph, schedule=TreeMessageSchedule())

    assert named.log_partition == member.log_partition == instance.log_partition
    assert named.guarantee is Guarantee.EXACT


@pytest.mark.smoke
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


@pytest.mark.smoke
def test_the_upward_pass_defines_the_root_marginal_and_no_other() -> None:
    graph = _star(6)

    half = sum_product(graph, schedule="upward")
    whole = sum_product(graph, schedule="tree")

    assert set(half.variable) == {"hub"}
    np.testing.assert_allclose(half.variable["hub"], whole.variable["hub"], atol=1e-12)
    with pytest.raises(KeyError):
        half.variable["v0"]


@pytest.mark.smoke
def test_the_downward_pass_reports_no_log_partition_rather_than_a_wrong_one() -> None:
    # It has seen no evidence from below, so neither route to `log Z` is open.
    # A number here would be a number for something it did not compute.
    half = sum_product(_chain(5), schedule="downward")

    assert np.isnan(half.log_partition)
    assert half.guarantee is Guarantee.PARTIAL
    assert set(half.variable) == {f"v{i}" for i in range(5)}


@pytest.mark.analytic
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


@pytest.mark.smoke
@pytest.mark.parametrize("schedule", TREE_SCHEDULES)
def test_a_tree_schedule_is_refused_on_a_loopy_graph(schedule: str) -> None:
    with pytest.raises(ValueError, match="exact only on a tree"):
        sum_product(_lattice(), schedule=schedule)


@pytest.mark.smoke
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


@pytest.mark.analytic
def test_the_residual_schedule_sweeps_once_in_order_then_follows_the_residual() -> None:
    # Every factor starts at infinite priority, so the first sweep is the
    # sequential order whatever residuals come back; after it the heap sends
    # the factor whose inputs moved most. Read on the layout directly, feeding
    # the residuals by hand (#825).
    layout = Layout(_lattice())
    schedule = resolve("residual")
    assert schedule.adaptive
    assert schedule.sweep_length(layout) == len(layout.factor_edges)
    per_factor = layout.sequential_steps()

    def which(step: Step) -> int:
        # A factor's step is identified by the edges its sends write.
        written = tuple(int(edge) for sends in step[1] for edge in sends.targets)
        return next(
            position
            for position, own in enumerate(per_factor)
            if tuple(int(edge) for sends in own[1] for edge in sends.targets) == written
        )

    steps = schedule.steps(layout)
    assert isinstance(steps, Generator)
    first = [which(next(steps))]
    first += [which(steps.send(1.0)) for _ in range(len(per_factor) - 1)]
    assert first == list(range(len(per_factor)))
    # A residual of 5 on the next factor sent bumps its neighbours above every
    # other pending factor, so the step after is one of them.
    neighbours = layout.factor_neighbours()
    factor = which(steps.send(0.0))
    assert which(steps.send(5.0)) in neighbours[factor]
