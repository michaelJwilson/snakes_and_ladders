"""The Rust tree schedule against the NumPy oracle and against enumeration (issue #754).

Decisions are exactly equal: the max-product assignment, the guarantee, the
pass count and which marginals are defined. Floats are inside
`CROSS_DEVICE_RTOL_FLOAT64`: the same sums in the same order, with NumPy's
vectorized `exp` and `log` differing from `libm`'s in the last place
(`sal.backend`). Realized: 3.3e-15 absolute and 7.2e-15
relative on the marginals, 1.4e-14 and 2.0e-16 on `log Z`. The default is
held to enumeration in
`test_message_passing.py::test_every_chain_evaluator_is_the_path_enumeration` (#982).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from sal.backend import Backend
from sal.likelihood import message_passing_reference as reference
from sal.likelihood.device import CROSS_DEVICE_RTOL_FLOAT64
from sal.likelihood.message_passing import (
    Marginals,
    MaxMarginals,
    MessageScheduleName,
    _run,
    max_product,
    sum_product,
)
from sal.likelihood.message_passing import rust as message_passing_rust
from sal.likelihood.schedule import Layout
from sal.sim.factor_graph import (
    Factor,
    FactorGraph,
    Variable,
    from_hmm,
    from_potts,
)
from sal.sim.graph import BoundaryCondition, lattice_graph

from tests.regression.likelihood.conftest import FIELD, TREE

#: The field and the tree `test_message_passing.py` runs the same claims on,
#: so the two files exercise one graph and not two.
LOOPY = lattice_graph((3, 3), coupling=0.4, boundary=BoundaryCondition.OPEN)


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

    Degrees three and four reduce two or three axes: the row-major walk.
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


def _value(beliefs: Marginals | MaxMarginals) -> float:
    """The scalar each result carries: ``log Z``, or the MAP's log-weight (#1090)."""
    return (
        beliefs.log_partition
        if isinstance(beliefs, Marginals)
        else beliefs.map_log_weight
    )


def _assert_agrees(
    realized: Marginals | MaxMarginals, expected: Marginals | MaxMarginals, tag: str
) -> None:
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
    if math.isnan(_value(expected)):
        assert math.isnan(_value(realized))
    else:
        assert _value(realized) == pytest.approx(
            _value(expected), rel=CROSS_DEVICE_RTOL_FLOAT64
        )


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.backend
@pytest.mark.parametrize("tag", sorted(TREES))
@pytest.mark.parametrize("maximum", [False, True])
def test_the_kernel_writes_the_oracle_s_messages_edge_for_edge(
    tag: str, maximum: bool
) -> None:
    # Message for message, then belief for belief: a sum can agree where its
    # terms do not. The oracle is the NumPy route, pinned in `test_message_passing.py`.
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

    if not maximum:
        _assert_agrees(
            sum_product(graph, backend=Backend.RUST),
            sum_product(graph, backend=Backend.PYTHON),
            tag,
        )
        return
    # The assignment is a decision and not a float, so it is asserted equal
    # rather than close: a last-place difference in a max-marginal that moved
    # an argmax would be a different MAP and is not tolerable at any bound.
    assignment, marginals = max_product(graph, backend=Backend.RUST)
    expected_assignment, expected = max_product(graph, backend=Backend.PYTHON)
    assert assignment == expected_assignment
    _assert_agrees(marginals, expected, tag)


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
