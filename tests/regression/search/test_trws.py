"""TRW-S (`search.trws`): a lower bound at every iteration, and the local-polytope value at convergence (issue #1060).

Referees: enumeration through `likelihood.potts.log_weights` bounds every
trace entry from above; on a tree or a chain the local polytope is tight, so
the bound meets the enumerated optimum; at convergence the bound equals
`search.tightening.dual_bound`'s pairwise value, a second coordinate ascent
on the same dual that shares no code with this one; the trace does not
decrease, Kolmogorov's (2006) guarantee; the compiled kernel is the Python
reference bitwise. `spatio_only/release`'s optimum, the two-state graph cut
(`test_ground_state.py`), is met by the bound at 5,041 sites.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from sal.backend import Backend
from sal.enumeration import configurations
from sal.likelihood.potts import log_weights
from sal.search.potts_starts import spatio_rung, tiling_rung
from sal.search.tightening import dual_bound
from sal.search.trws import chain_layout, trws
from sal.search.trws.numba import trws_iterations_checked
from sal.sim.fixtures import fixture
from sal.sim.graph import (
    BoundaryCondition,
    PottsGraph,
    lattice_graph,
    triangular_lattice_graph,
)
from sal.sim.potts import energy

from tests._rows import every_row, every_value

#: The tree `test_tightening.py` runs on: six nodes, five edges, couplings of
#: mixed sign.
TREE = PottsGraph(
    n_nodes=6,
    edges=((0, 1), (0, 2), (1, 3), (1, 4), (2, 5)),
    coupling=(0.8, -0.4, 1.2, 0.3, 0.9),
)

#: `spatio_only/release`'s ten-state optimum: the two-state graph cut's energy
#: (`RELEASE_Q2_ENERGY` in `test_ground_state.py`), which issue #1041's
#: reduction makes the ten-state one.
RELEASE_OPTIMUM = -10454.1562900565

#: Agreement asked of two routes to one LP value: the plan's 1e-9 relative.
LP_AGREEMENT = 1e-9

#: The triangular antiferromagnets of `_instances`, whose pairwise LP is loose.
FRUSTRATED = ("triangular-0", "triangular-1", "triangular-2")
#: The two of them on which TRW-S converges below `dual_bound`'s value.
STALLED = ("triangular-0", "triangular-2")


def _optimum(graph: PottsGraph, field: np.ndarray, n_states: int) -> float:
    """``min_x E(x)`` by enumeration: the negated largest log weight."""
    every = configurations(n_states, graph.n_nodes, limit=600_000)
    return -float(log_weights(graph, field, every).max())


def _mixed(graph: PottsGraph, seed: int) -> PottsGraph:
    """``graph`` with standard-normal couplings, of either sign."""
    coupling = np.random.default_rng(seed).normal(size=len(graph.edges))
    return PottsGraph(graph.n_nodes, graph.edges, tuple(coupling.tolist()))


def _instances() -> list[tuple[str, PottsGraph, np.ndarray, int]]:
    """Small graphs enumeration reaches: square and triangular, open and periodic, mixed signs.

    The triangular antiferromagnet is frustrated, so its pairwise LP is loose
    and the bound sits strictly below the optimum; the rest carry both.
    """
    rows = []
    for seed in range(3):
        rng = np.random.default_rng(1060 + seed)
        square = _mixed(lattice_graph((3, 3), BoundaryCondition.PERIODIC, 1.0), seed)
        rows.append((f"square-{seed}", square, rng.normal(size=(9, 3)), 3))
        strip = _mixed(lattice_graph((2, 4), BoundaryCondition.OPEN, 1.0), seed + 10)
        rows.append((f"strip-{seed}", strip, rng.normal(size=(8, 3)), 3))
        frustrated = triangular_lattice_graph(
            (3, 3), BoundaryCondition.PERIODIC, -1.0 - seed
        )
        rows.append(
            (f"triangular-{seed}", frustrated, 0.3 * rng.normal(size=(9, 3)), 3)
        )
    return rows


@pytest.mark.critical
@pytest.mark.oracle
def test_every_iterate_bounds_the_enumerated_optimum() -> None:
    # The chains sum to the reparametrized energy for any messages, so every
    # trace entry is a bound; enumeration checks it entry by entry, and the
    # decoded energy from above.
    def check(name: str, graph: PottsGraph, field: np.ndarray, n_states: int) -> None:
        optimum = _optimum(graph, field, n_states)
        result = trws(graph, field, max_iterations=300, tolerance=0.0)
        scale = 1e-12 * max(1.0, abs(optimum))

        assert result.trace.max() <= optimum + scale, name
        assert result.energy >= optimum - scale, name

    every_row(_instances(), check)


@pytest.mark.analytic
def test_the_bound_does_not_decrease() -> None:
    # Kolmogorov (2006): the chain bound is non-decreasing under the
    # sequential updates. Held to 1e-12 relative for the last bits of a sum.
    def check(name: str, graph: PottsGraph, field: np.ndarray, _n_states: int) -> None:
        trace = trws(graph, field, max_iterations=300, tolerance=0.0).trace
        steps = np.diff(trace)
        lengths.append(trace.shape[0])

        assert (steps >= -1e-12 * max(1.0, float(np.abs(trace).max()))).all(), name

    # A run whose gap closes stops there, so the loose instances carry the
    # long traces; the claim is empty unless some run is long.
    lengths: list[int] = []
    every_row(_instances(), check)

    assert max(lengths) >= 100


@pytest.mark.analytic
def test_the_bound_is_loose_on_the_frustrated_lattice() -> None:
    # The pairwise relaxation cannot see a triangle's frustration, so the
    # bound stays below the optimum, by 0.02 to 1.6 here: an instance that
    # separates the bound from the optimum is one the agreement tests below
    # can fail on.
    def check(name: str, graph: PottsGraph, field: np.ndarray, n_states: int) -> None:
        optimum = _optimum(graph, field, n_states)

        assert trws(graph, field).bound < optimum - 1e-3, name

    every_row([row for row in _instances() if row[0] in FRUSTRATED], check)


@pytest.mark.oracle
def test_on_a_tree_or_a_chain_the_bound_is_the_optimum() -> None:
    # A tree's local polytope is its marginal polytope, so the LP is tight
    # and the bound meets the enumerated optimum; the decoded labelling
    # attains it.
    chain = _mixed(lattice_graph((8,), BoundaryCondition.OPEN, 1.0), 7)
    rows = [(TREE, seed) for seed in range(3)] + [(chain, seed) for seed in range(3)]

    def check(graph: PottsGraph, seed: int) -> None:
        field = np.random.default_rng(seed).normal(size=(graph.n_nodes, 3))
        optimum = _optimum(graph, field, 3)
        result = trws(graph, field)
        scale = 1e-12 * max(1.0, abs(optimum))

        assert abs(result.bound - optimum) <= scale
        assert abs(result.energy - optimum) <= scale
        assert result.termination.converged

    every_row(rows, check)


@pytest.mark.oracle
def test_at_convergence_the_bound_is_dual_bounds_pairwise_value() -> None:
    # Two coordinate ascents on the one local-polytope dual: `dual_bound`
    # moves a cluster's share against its two sites, this passes messages
    # along chains. Where both reach the LP maximum they read one value, to
    # 1e-9 relative: every square and strip instance, one frustrated one,
    # and the two declared CI fixtures.
    spatio = spatio_rung(fixture("spatio_only", "ci").params, "ci")
    tiling = tiling_rung(fixture("spatio_tiling", "ci").params, "ci")
    rows = [row for row in _instances() if row[0] not in STALLED] + [
        ("spatio_only/ci", spatio.graph, spatio.field, spatio.n_states),
        ("spatio_tiling/ci", tiling.graph, tiling.field, tiling.n_states),
    ]

    def check(name: str, graph: PottsGraph, field: np.ndarray, _n_states: int) -> None:
        result = trws(graph, field)
        reference = dual_bound(graph, field, iterations=5000)

        assert result.termination.converged, name
        assert reference.termination is not None
        assert reference.termination.converged, name
        assert abs(result.bound - reference.bound) <= LP_AGREEMENT * abs(
            reference.bound
        ), name

    every_row(rows, check)


@pytest.mark.oracle
@pytest.mark.warning
def test_on_two_frustrated_instances_both_ascents_stop_short_of_each_other() -> None:
    # Coordinate ascent on the LP dual stops where no block move raises it,
    # which need not be the maximum (Kolmogorov 2006, weak tree agreement).
    # On `triangular-0` and `triangular-2` TRW-S converges 0.0160 and 0.0067
    # below `dual_bound`; the explicit LP, solved by HiGHS
    # (`tests/validation/test_highs.py`, #1063), is -1.14911 and -1.40024,
    # above both, so neither is the LP value there. Both stay bounds.
    def check(name: str, graph: PottsGraph, field: np.ndarray, n_states: int) -> None:
        result = trws(graph, field)
        reference = dual_bound(graph, field, iterations=5000)

        assert result.termination.converged, name
        assert result.bound < reference.bound - 1e-3, name
        assert reference.bound <= _optimum(graph, field, n_states), name

    every_row([row for row in _instances() if row[0] in STALLED], check)


@pytest.mark.oracle
def test_the_tiling_bound_is_below_its_enumerated_optimum() -> None:
    # `spatio_tiling/ci`, 3**12 labellings: the declared instance whose
    # optimum uses all three states, where no two-state cut reaches it.
    params = fixture("spatio_tiling", "ci").params
    rung = tiling_rung(params, "ci")
    optimum = _optimum(rung.graph, rung.field, rung.n_states)
    result = trws(rung.graph, rung.field, max_iterations=300, tolerance=0.0)

    assert result.trace.max() <= optimum + 1e-12 * abs(optimum)
    assert result.energy >= optimum - 1e-12 * abs(optimum)


@pytest.mark.release
@pytest.mark.oracle
def test_the_bound_meets_the_graph_cut_optimum_at_release() -> None:
    # `spatio_only/release`, 5,041 sites at ten states: the optimum is the
    # two-state graph cut's (issue #1041). The bound meets it and the decoded
    # labelling attains it, so the LP is tight there and TRW-S certifies the
    # optimum on its own.
    rung = spatio_rung(fixture("spatio_only", "release").params, "release")
    result = trws(rung.graph, rung.field)
    scale = 1e-9 * abs(RELEASE_OPTIMUM)

    assert abs(result.bound - RELEASE_OPTIMUM) <= scale
    assert abs(result.energy - RELEASE_OPTIMUM) <= scale
    assert result.termination.converged


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.backend
def test_numba_reproduces_the_python_reference_bitwise() -> None:
    # Same passes, bound and decode in the same order, operation for
    # operation: the trace, the labelling and the energy are identical.
    doubled = lattice_graph((2, 3), BoundaryCondition.PERIODIC, 0.7)
    isolated = PottsGraph(4, ((0, 2), (2, 1)), (0.5, -0.3))
    ci = spatio_rung(fixture("spatio_only", "ci").params, "ci")
    rows = [(name, graph, field) for name, graph, field, _ in _instances()] + [
        ("doubled", doubled, np.random.default_rng(3).normal(size=(6, 4))),
        ("isolated", isolated, np.random.default_rng(4).normal(size=(4, 2))),
        ("shared-field", doubled, np.array([0.1, -0.2, 0.3])),
        ("spatio_only/ci", ci.graph, ci.field),
    ]

    def check(name: str, graph: PottsGraph, field: np.ndarray) -> None:
        compiled = trws(graph, field, max_iterations=60, tolerance=0.0)
        reference = trws(
            graph, field, max_iterations=60, tolerance=0.0, backend=Backend.PYTHON
        )

        assert np.array_equal(compiled.trace, reference.trace), name
        assert np.array_equal(compiled.labelling, reference.labelling), name
        assert compiled.energy == reference.energy, name
        assert compiled.termination == reference.termination, name

    every_row(rows, check)


@pytest.mark.oracle
@pytest.mark.backend
def test_both_backends_stop_at_the_same_iteration() -> None:
    # The stopping rule is the kernel's and the reference's alike: at the
    # default tolerance both converge after the same count.
    def check(name: str, graph: PottsGraph, field: np.ndarray, _n_states: int) -> None:
        compiled = trws(graph, field)
        reference = trws(graph, field, backend=Backend.PYTHON)

        assert compiled.termination == reference.termination, name
        assert compiled.bound == reference.bound, name

    every_row(_instances(), check)


@pytest.mark.analytic
def test_the_chains_cover_every_edge_once_and_each_site_max_in_out_times() -> None:
    # The decomposition the bound is summed over: each edge in exactly one
    # chain, walked lower end to higher, and site s in max(in, out, 1)
    # chains, which is what makes `1 / n_s` the weight that sums to one.
    graphs = [
        triangular_lattice_graph((4, 5), BoundaryCondition.PERIODIC, 1.0),
        lattice_graph((2, 3), BoundaryCondition.PERIODIC, 1.0),
        PottsGraph(5, ((3, 1), (1, 4), (0, 4)), (1.0, 1.0, 1.0)),
    ]

    def check(graph: PottsGraph) -> None:
        layout = chain_layout(graph)
        n_edges = len(graph.edges)
        counts = np.zeros(graph.n_nodes, dtype=np.int64)

        assert np.array_equal(np.sort(layout.chain_edges), np.arange(n_edges))
        for chain, head in enumerate(layout.chain_heads.tolist()):
            edges = layout.chain_edges[
                layout.chain_offsets[chain] : layout.chain_offsets[chain + 1]
            ]
            walk = [head, *layout.ends[edges, 1].tolist()]
            assert all(
                layout.ends[edge, 0] == walk[step] for step, edge in enumerate(edges)
            )
            counts[walk] += 1
        assert np.allclose(counts, 1.0 / layout.weight, rtol=0.0, atol=0.0)

    every_value(graphs, check)


@pytest.mark.smoke
def test_the_labelling_energy_is_sim_potts_energy() -> None:
    graph, field = _instances()[0][1:3]
    result = trws(graph, field)

    assert result.energy == energy(graph, field, result.labelling)
    assert result.gap == result.energy - result.bound
    assert result.iterations == result.trace.shape[0]


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("graph", "field", "keywords", "message"),
    [
        (TREE, np.zeros((5, 3)), {}, r"field has shape \(5, 3\)"),
        (
            PottsGraph(2, ((1, 1),), (1.0,)),
            np.zeros(2),
            {},
            "edge 0 joins site 1 to itself",
        ),
        (TREE, np.zeros(3), {"max_iterations": 0}, "max_iterations is a loop's cap"),
        (TREE, np.zeros(3), {"tolerance": -1.0}, "tolerance must be >= 0"),
        (
            TREE,
            np.zeros(3),
            {"max_iterations": 0, "backend": Backend.PYTHON},
            "max_iterations is a loop's cap",
        ),
        (
            TREE,
            np.zeros(3),
            {"tolerance": -1.0, "backend": Backend.PYTHON},
            "tolerance must be >= 0",
        ),
        (TREE, np.zeros(3), {"backend": Backend.RUST}, "TRW-S runs on numba or python"),
    ],
)
def test_the_gateway_refuses(
    graph: PottsGraph,
    field: np.ndarray,
    keywords: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        trws(graph, field, **keywords)  # type: ignore[arg-type]


def _kernel_arguments() -> dict[str, object]:
    layout = chain_layout(TREE)
    return {
        "unary": np.zeros((6, 3)),
        "offsets": layout.offsets,
        "slots": layout.slots,
        "neighbours": layout.neighbours,
        "ends": layout.ends,
        "coupling": layout.coupling,
        "weight": layout.weight,
        "chain_offsets": layout.chain_offsets,
        "chain_heads": layout.chain_heads,
        "chain_edges": layout.chain_edges,
        "max_iterations": 4,
        "tolerance": 0.0,
    }


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"unary": np.zeros(6)}, r"unary must be 2-D.*got shape \(6,\)"),
        ({"unary": np.zeros((6, 0))}, r"unary must be 2-D.*got shape \(6, 0\)"),
        ({"ends": np.zeros((5, 3))}, r"ends must be \(n_edges, 2\)"),
        ({"coupling": np.zeros(4)}, r"coupling has shape \(4,\) and ends 5 rows"),
        (
            {"ends": np.array([[1, 0], [0, 2], [1, 3], [1, 4], [2, 5]])},
            r"ends must be \(lower, higher\) pairs",
        ),
        ({"offsets": np.zeros(6, dtype=np.int64)}, r"offsets has shape \(6,\)"),
        (
            {"offsets": np.array([0, 3, 2, 5, 7, 9, 10])},
            "offsets must start at 0 and be non-decreasing",
        ),
        ({"slots": np.arange(8)}, r"slots has shape \(8,\)"),
        ({"neighbours": np.zeros(9, dtype=np.int64)}, r"neighbours has shape \(9,\)"),
        ({"slots": np.arange(10) + 1}, r"slots must index entries in \[0, 10\)"),
        ({"weight": np.ones(5)}, r"weight has shape \(5,\), expected \(6,\)"),
        ({"chain_offsets": np.zeros(2, dtype=np.int64)}, "chain_offsets has shape"),
        (
            {"chain_edges": np.array([0, 1, 2, 3, 9])},
            r"chain_edges must index edges in \[0, 5\)",
        ),
        ({"max_iterations": 0}, "max_iterations must be >= 1, got 0"),
        ({"tolerance": float("nan")}, "tolerance must be >= 0, got nan"),
    ],
)
def test_each_kernel_shape_error_names_wanted_and_given(
    change: dict[str, object], message: str
) -> None:
    arguments = _kernel_arguments()
    arguments.update(change)
    with pytest.raises(ValueError, match=message):
        trws_iterations_checked(**arguments)  # type: ignore[arg-type]


@pytest.mark.smoke
def test_every_bound_refuses_a_cap_below_one_alike() -> None:
    # Issue #1089: `dual_bound(iterations=0)` returned a bound of -inf where
    # TRW-S refused; both refuse through `check_cap`.
    field = np.zeros(3)
    calls: tuple[Callable[[], object], ...] = (
        lambda: trws(TREE, field, max_iterations=0),
        lambda: dual_bound(TREE, field, iterations=0, plaquettes=()),
    )
    for call in calls:
        with pytest.raises(ValueError, match="a loop's cap and must be at least 1"):
            call()
