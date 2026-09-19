"""The Rust tree schedule against the NumPy oracle and against enumeration (issue #754).

`likelihood/CLAUDE.md`: the reference implementation is the oracle and it
stays, and a backend is accepted or rejected against a stated bound rather
than adjusted until it matches. Two claims are separated here rather than
merged into the looser one.

* **Exactly equal** where the output is a decision rather than a float: the
  max-product assignment, the guarantee, the pass count and which marginals
  are defined are the NumPy route's on every fixture below.
* **Inside `CROSS_DEVICE_RTOL_FLOAT64`** on the marginals and `log Z`. The
  kernel's arithmetic is the oracle's operation for operation --- the same
  sums in the same order --- and what is left is that NumPy's vectorized
  `exp` and `log` differ from `libm`'s in the last place, which
  `snakes_and_ladders.backend` states as the standing reason a compiled
  route is not bitwise. Realized over the fixtures below: **3.3e-15**
  absolute and **7.2e-15** relative on the marginals, **1.4e-14** absolute
  and **2.0e-16** relative on `log Z`.

Both routes are then pinned to enumeration, which shares no recursion with
either, so the pair is not established by agreeing with each other alone.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from snakes_and_ladders.backend import Backend
from snakes_and_ladders.likelihood import message_passing_reference as reference
from snakes_and_ladders.likelihood import message_passing_rust
from snakes_and_ladders.likelihood.device import CROSS_DEVICE_RTOL_FLOAT64
from snakes_and_ladders.likelihood.message_passing import (
    Marginals,
    MessageScheduleName,
    _run,
    max_product,
    sum_product,
)
from snakes_and_ladders.likelihood.schedule import Layout
from snakes_and_ladders.sim.factor_graph import (
    Factor,
    FactorGraph,
    Variable,
    from_hmm,
    from_potts,
)
from snakes_and_ladders.sim.graph import BoundaryCondition, PottsGraph, lattice_graph

#: The field and the tree `test_message_passing.py` runs the same claims on,
#: so the two files exercise one graph and not two.
FIELD = np.array([0.3, -0.7, 0.15])
TREE = PottsGraph(
    n_nodes=6,
    edges=((0, 1), (0, 2), (1, 3), (1, 4), (2, 5)),
    coupling=(0.8, -0.4, 1.2, 0.3, 0.9),
)
LOOPY = lattice_graph((3, 3), coupling=0.4, boundary=BoundaryCondition.OPEN)

#: What a marginal is held to against enumeration, as `test_message_passing.py`
#: states it.
RTOL = 1e-11
ATOL = 1e-12


def _chain(length: int, states: int, seed: int) -> FactorGraph:
    rng = np.random.default_rng(seed)
    return from_hmm(
        np.log(rng.dirichlet(np.ones(states))),
        np.log(rng.dirichlet(np.ones(states), size=states)),
        rng.normal(size=(length, states)),
    )


def _random_tree(
    n_variables: int, cardinality: int, degree: int, seed: int
) -> FactorGraph:
    """A tree of factors of one degree, each joining a new variable to the graph.

    Degree three and four are what a chain does not exercise: a factor there
    reduces two or three axes to send along one, which is the reduction the
    kernel walks in row-major order and NumPy reaches through
    ``transpose(...).reshape(n, c, -1)``.
    """
    rng = np.random.default_rng(seed)
    variables = [Variable(f"v{i}", cardinality) for i in range(n_variables)]
    factors = []
    attached = [0]
    for first in range(1, n_variables, degree - 1):
        fresh = list(range(first, min(first + degree - 1, n_variables)))
        members = (int(rng.choice(attached)), *fresh)
        factors.append(
            Factor(
                f"f{first}",
                tuple(f"v{member}" for member in members),
                rng.normal(size=(cardinality,) * len(members)),
            )
        )
        attached.extend(fresh)
    return FactorGraph(variables, factors)


#: Every tree the two routes are compared on. The chain is the shape the
#: stress profile ranks; the Potts tree and its Forney form are what
#: `test_message_passing.py` pins bitwise against the dictionary reference;
#: the random trees carry the degrees and the padding a chain has not.
TREES = {
    "chain-200x4": _chain(200, 4, 0),
    "chain-20x2": _chain(20, 2, 3),
    "chain-50x8": _chain(50, 8, 5),
    "potts-tree": from_potts(TREE, FIELD),
    "forney-form": from_potts(TREE, FIELD).forney(),
    "tree-degree-3": _random_tree(41, 3, 3, 2),
    "tree-degree-4": _random_tree(40, 5, 4, 7),
}


def _assert_agrees(realized: Marginals, expected: Marginals, tag: str) -> None:
    """The decisions equal, the floats inside the declared tolerance."""
    assert realized.iterations == expected.iterations
    assert realized.guarantee is expected.guarantee
    assert realized.tree == expected.tree
    assert set(realized.variable) == set(expected.variable)
    assert set(realized.factor) == set(expected.factor)
    for name, values in expected.variable.items():
        np.testing.assert_allclose(
            realized.variable[name],
            values,
            rtol=CROSS_DEVICE_RTOL_FLOAT64,
            atol=0.0,
            err_msg=f"{tag}: variable {name}",
        )
    for name, values in expected.factor.items():
        np.testing.assert_allclose(
            realized.factor[name],
            values,
            rtol=CROSS_DEVICE_RTOL_FLOAT64,
            atol=0.0,
            err_msg=f"{tag}: factor {name}",
        )
    if math.isnan(expected.log_partition):
        assert math.isnan(realized.log_partition)
    else:
        assert realized.log_partition == pytest.approx(
            expected.log_partition, rel=CROSS_DEVICE_RTOL_FLOAT64
        )


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.backend
@pytest.mark.parametrize("tag", sorted(TREES))
def test_the_rust_tree_schedule_agrees_with_the_numpy_oracle(tag: str) -> None:
    # The oracle is the NumPy route of the same function, which
    # `test_message_passing.py` pins bitwise against the dictionary reference
    # and against enumeration; this is the one hop the kernel adds.
    graph = TREES[tag]

    _assert_agrees(
        sum_product(graph, backend=Backend.RUST),
        sum_product(graph, backend=Backend.PYTHON),
        tag,
    )


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.backend
@pytest.mark.parametrize("tag", sorted(TREES))
def test_the_rust_max_product_decodes_the_oracle_s_assignment(tag: str) -> None:
    # The assignment is a decision and not a float, so it is asserted equal
    # rather than close: a last-place difference in a max-marginal that moved
    # an argmax would be a different MAP and is not tolerable at any bound.
    graph = TREES[tag]

    assignment, marginals = max_product(graph, backend=Backend.RUST)
    expected_assignment, expected = max_product(graph, backend=Backend.PYTHON)

    assert assignment == expected_assignment
    _assert_agrees(marginals, expected, tag)


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.backend
@pytest.mark.parametrize("tag", sorted(TREES))
@pytest.mark.parametrize("maximum", [False, True])
def test_the_kernel_writes_the_oracle_s_messages_edge_for_edge(
    tag: str, maximum: bool
) -> None:
    # The kernel against the loop it replaces, message for message rather than
    # belief for belief: the beliefs sum the messages, and a sum can agree
    # where its terms do not.
    graph = TREES[tag]

    to_variable, to_factor = message_passing_rust.tree_messages(
        Layout(graph), maximum=maximum
    )
    _, expected_variable, expected_factor, *_ = _run(
        graph, MessageScheduleName.TREE, maximum, 0.5, 1e-10, 500, Backend.PYTHON
    )

    np.testing.assert_allclose(
        to_variable, expected_variable, rtol=CROSS_DEVICE_RTOL_FLOAT64, atol=0.0
    )
    np.testing.assert_allclose(
        to_factor, expected_factor, rtol=CROSS_DEVICE_RTOL_FLOAT64, atol=0.0
    )


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.backend
def test_the_rust_route_reproduces_the_dictionary_reference() -> None:
    # The oracle's oracle (issue #341): the dictionary-per-message
    # implementation the edge-array layout replaced, which shares no array
    # with either route.
    graph = from_potts(TREE, FIELD)

    _assert_agrees(
        sum_product(graph, backend=Backend.RUST), reference.sum_product(graph), "potts"
    )


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.backend
@pytest.mark.parametrize(("states", "length"), [(2, 4), (3, 3)])
def test_the_rust_marginals_and_log_z_are_the_enumeration(
    states: int, length: int
) -> None:
    # Exact on a tree, asserted against the sum over every configuration
    # rather than against the other route: `log Z` and every marginal.
    graph = _chain(length, states, 11)
    weights = np.array(
        [
            graph.log_density(
                {f"z{position}": state for position, state in enumerate(assignment)}
            )
            for assignment in itertools.product(range(states), repeat=length)
        ]
    )
    peak = weights.max()
    evidence = peak + np.log(np.exp(weights - peak).sum())

    marginals = sum_product(graph, backend=Backend.RUST)

    assert marginals.log_partition == pytest.approx(evidence, rel=RTOL, abs=ATOL)
    for position in range(length):
        expected = np.zeros(states)
        for index, assignment in enumerate(
            itertools.product(range(states), repeat=length)
        ):
            expected[assignment[position]] += float(np.exp(weights[index] - evidence))
        np.testing.assert_allclose(
            marginals.variable[f"z{position}"], expected, rtol=RTOL, atol=ATOL
        )


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.backend
def test_the_rust_max_product_is_the_enumerated_mode() -> None:
    states, length = 3, 5
    graph = _chain(length, states, 13)
    best = max(
        itertools.product(range(states), repeat=length),
        key=lambda assignment: graph.log_density(
            {f"z{position}": state for position, state in enumerate(assignment)}
        ),
    )

    assignment, marginals = max_product(graph, backend=Backend.RUST)

    assert tuple(assignment[f"z{position}"] for position in range(length)) == best
    assert marginals.log_partition == pytest.approx(
        graph.log_density(
            {f"z{position}": state for position, state in enumerate(best)}
        ),
        rel=RTOL,
    )


@pytest.mark.smoke
@pytest.mark.backend
def test_a_schedule_with_no_kernel_takes_the_numpy_route_bitwise() -> None:
    # Flooding answers `compiled` false, so naming the Rust backend leaves it
    # on the NumPy route rather than refusing the call or changing a number.
    graph = from_potts(LOOPY, FIELD)
    assert not graph.is_tree()

    realized = sum_product(
        graph, schedule=MessageScheduleName.FLOODING, backend=Backend.RUST
    )
    expected = sum_product(
        graph, schedule=MessageScheduleName.FLOODING, backend=Backend.PYTHON
    )

    assert realized.log_partition == expected.log_partition
    for name, values in expected.variable.items():
        np.testing.assert_array_equal(realized.variable[name], values, err_msg=name)


@pytest.mark.smoke
@pytest.mark.backend
def test_the_rust_route_refuses_a_loopy_graph_as_the_numpy_route_does() -> None:
    # The refusal is the schedule's and comes before either route, so a
    # backend does not decide which graphs the tree schedule accepts.
    graph = from_potts(LOOPY, FIELD)

    with pytest.raises(ValueError, match="exact only on a tree"):
        sum_product(graph, backend=Backend.RUST)


@pytest.mark.smoke
@pytest.mark.backend
def test_a_backend_this_function_does_not_have_is_refused() -> None:
    with pytest.raises(ValueError, match="message passing runs on"):
        sum_product(TREES["potts-tree"], backend=Backend.NUMBA)
