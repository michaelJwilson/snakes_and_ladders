"""`SparseIncidence` against a dictionary of lists, and against its consumers.

The oracle is the obvious implementation: a Python dictionary from row to the
list of columns given in the caller's order, built by a loop. It is what every
consumer wrote before issue #586, it is wrong in no way except speed, and it
decides the two things the compressed layout is easy to get wrong --- which
row an entry landed in, and in what order within the row.

The consumers are pinned separately, because the point of lifting the layout
was that three classes kept deriving it: each is asserted to produce, through
the structure, what its own hand-written derivation produced.
"""

from __future__ import annotations

import numpy as np
import pytest
from sal.incidence import SparseIncidence
from sal.sim.factor_graph import Factor, FactorGraph, Variable
from sal.sim.graph import BoundaryCondition, lattice_graph
from sal.sim.ldpc import gallager_code
from scipy.sparse import csr_matrix

from tests._rows import every_row


def _rows_of(
    n_rows: int, rows: np.ndarray, columns: np.ndarray, *, ascending: bool
) -> list[list[int]]:
    """The oracle: each row's columns, by a Python loop over the pairs."""
    out: list[list[int]] = [[] for _ in range(n_rows)]
    for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
        out[row].append(column)
    return [sorted(row) if ascending else row for row in out]


def _pairs(
    seed: int, n_rows: int, n_cols: int, n_entries: int
) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    return (
        rng.integers(0, n_rows, size=n_entries),
        rng.integers(0, n_cols, size=n_entries),
    )


@pytest.mark.oracle
def test_the_rows_are_the_loop_s_rows() -> None:
    def check(ascending: bool, n_rows: int, n_cols: int, n_entries: int) -> None:
        rows, columns = _pairs(2026, n_rows, n_cols, n_entries)
        incidence = SparseIncidence.from_pairs(
            n_rows, n_cols, rows, columns, ascending=ascending
        )

        recovered = [incidence.row(index).tolist() for index in range(n_rows)]

        assert recovered == _rows_of(n_rows, rows, columns, ascending=ascending)
        assert incidence.degrees.tolist() == [len(row) for row in recovered]
        assert incidence.n_entries == n_entries

    sizes = [(7, 5, 30), (40, 9, 200)]
    every_row(
        ((ascending, *size) for ascending in (False, True) for size in sizes), check
    )


@pytest.mark.oracle
def test_gather_carries_a_per_entry_value_into_the_layout() -> None:
    # The property the couplings and the messages both rest on: a value the
    # caller holds against its own pair list is read against the layout's
    # entries, and the pairing survives the sort.
    rows, columns = _pairs(7, 12, 12, 60)
    incidence = SparseIncidence.from_pairs(12, 12, rows, columns)
    values = np.arange(60) * 3.5

    gathered = incidence.gather(values)

    for entry, (row, column, value) in enumerate(
        zip(
            incidence.rows.tolist(),
            incidence.indices.tolist(),
            gathered.tolist(),
            strict=True,
        )
    ):
        source = int(incidence.order[entry])
        assert (row, column) == (int(rows[source]), int(columns[source]))
        assert value == values[source]


@pytest.mark.analytic
def test_the_transpose_is_the_relation_read_the_other_way() -> None:
    rows, columns = _pairs(19, 15, 11, 90)
    incidence = SparseIncidence.from_pairs(15, 11, rows, columns)
    values = np.arange(90, dtype=np.float64)

    other, read = incidence.transpose()

    assert np.array_equal(other.dense(), incidence.dense().T)
    assert other.n_rows == incidence.n_cols
    assert other.n_cols == incidence.n_rows
    # One per-entry array, read from both ends through the permutation: no
    # second copy, which is what `ParityCheck.check_order` buys.
    assert np.array_equal(incidence.gather(values)[read], other.gather(values))
    assert np.array_equal(other.transpose()[0].dense(), incidence.dense())


@pytest.mark.oracle
def test_dense_counts_a_repeated_entry_twice() -> None:
    # A periodic lattice of extent two doubles a bond, and the doubling is the
    # model rather than a duplicate; a layout that silently deduplicated would
    # halve that coupling.
    incidence = SparseIncidence.from_pairs(
        2, 2, np.array([0, 0, 1]), np.array([1, 1, 0])
    )

    assert incidence.dense().tolist() == [[0, 2], [1, 0]]
    assert incidence.degrees.tolist() == [2, 1]


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("kwargs", "rows", "columns", "message"),
    [
        ({}, [0, 1], [0], "same length"),
        ({}, [0, 2], [0, 0], "outside the 2 x 2 relation"),
        ({}, [0, 1], [0, -1], "outside the 2 x 2 relation"),
        ({"every_row": True}, [0, 0], [0, 1], "row 1 carries none"),
        ({"distinct": True}, [0, 0], [1, 1], "listed twice"),
    ],
)
def test_a_relation_that_is_not_one_is_refused(
    kwargs: dict[str, bool], rows: list[int], columns: list[int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        SparseIncidence.from_pairs(2, 2, np.array(rows), np.array(columns), **kwargs)


@pytest.mark.smoke
def test_a_duplicate_is_caught_when_the_two_are_not_adjacent() -> None:
    # Without `ascending` the entries keep the caller's order, so two equal
    # pairs need not end up side by side; an adjacency test would pass the
    # duplicate it exists to refuse.
    with pytest.raises(ValueError, match="listed twice"):
        SparseIncidence.from_pairs(
            1, 3, np.array([0, 0, 0]), np.array([2, 1, 2]), distinct=True
        )


@pytest.mark.smoke
def test_the_parity_check_s_two_orientations_are_the_structure_s() -> None:
    # What `ParityCheck.from_edges` derived by `lexsort` and `searchsorted`
    # before it held the structure, asserted field for field.
    code = gallager_code(120, 3, 6, np.random.default_rng(5))
    variable, check = code.edge_variable, code.edge_check

    assert np.array_equal(np.lexsort((check, variable)), np.arange(variable.size))
    assert np.array_equal(
        np.searchsorted(variable, np.arange(code.n_bits + 1)), code.variable_offsets
    )
    order = np.argsort(check, kind="stable")
    assert np.array_equal(order, code.check_order)
    assert np.array_equal(
        np.searchsorted(check[order], np.arange(code.n_checks + 1)), code.check_offsets
    )


@pytest.mark.smoke
def test_the_potts_adjacency_is_derived_once_and_is_read_only() -> None:
    # The fix the survey's finding asked for: every caller gets the same
    # arrays, so a caller that wrote into them would be editing the graph.
    graph = lattice_graph((8, 8), BoundaryCondition.PERIODIC, 1.0)

    first, second = graph.compressed_adjacency(), graph.compressed_adjacency()

    for one, two in zip(first, second, strict=True):
        assert one is two
        with pytest.raises(ValueError, match="read-only"):
            one[0] = 0


@pytest.mark.oracle
def test_the_potts_adjacency_keeps_each_node_s_edge_order() -> None:
    # The contract a sweep rests on: a site consumes one draw per neighbour in
    # the graph's edge order, so a permutation within a row would change which
    # draw a site sees without changing any distribution a test could catch.
    graph = lattice_graph((4, 5), BoundaryCondition.OPEN, 0.75)
    offsets, neighbours, couplings = graph.compressed_adjacency()

    expected: list[list[int]] = [[] for _ in range(graph.n_nodes)]
    weights: list[list[float]] = [[] for _ in range(graph.n_nodes)]
    for (first, second), coupling in graph.weighted_edges():
        expected[first].append(second)
        weights[first].append(coupling)
        expected[second].append(first)
        weights[second].append(coupling)

    for node in range(graph.n_nodes):
        start, stop = int(offsets[node]), int(offsets[node + 1])
        assert neighbours[start:stop].tolist() == expected[node]
        assert couplings[start:stop].tolist() == weights[node]


@pytest.mark.oracle
def test_the_factor_graph_s_degrees_are_the_scan_s() -> None:
    # `degree` and `neighbours` scanned every factor per query; the scan is
    # the oracle, and it is quadratic, which is why it is no longer the code.
    variables = [Variable(f"v{index}", 2) for index in range(12)]
    factors = [
        Factor(f"f{index}", (f"v{index}", f"v{(index * 5 + 1) % 12}"), np.zeros((2, 2)))
        for index in range(12)
    ]
    graph = FactorGraph(variables, factors)

    for variable in graph.variables:
        scanned = [f for f in graph.factors if variable.name in f.variables]
        assert [f.name for f in graph.neighbours(variable.name)] == [
            f.name for f in scanned
        ]
        assert graph.degree(variable.name) == len(scanned)


@pytest.mark.smoke
def test_an_unknown_variable_has_no_degree() -> None:
    graph = FactorGraph(
        [Variable("a", 2), Variable("b", 2)],
        [Factor("f", ("a", "b"), np.zeros((2, 2)))],
    )

    with pytest.raises(KeyError):
        graph.degree("c")


def _distinct_pairs(
    seed: int, n_rows: int, n_cols: int, n_entries: int
) -> tuple[np.ndarray, np.ndarray]:
    """Random entries of the relation, no pair repeated."""
    rng = np.random.default_rng(seed)
    flat = rng.choice(n_rows * n_cols, size=n_entries, replace=False)
    return flat // n_cols, flat % n_cols


@pytest.mark.oracle
def test_the_compressed_layout_is_scipys_csr_on_the_same_pairs() -> None:
    """`scipy.sparse.csr_matrix` builds the same layout by another route.

    Equal entry for entry on 30 in 7 x 5, 200 in 40 x 9, 512 in 64 x 64; no repeats.
    """

    def check(n_rows: int, n_cols: int, n_entries: int) -> None:
        rows, columns = _distinct_pairs(31 + n_rows, n_rows, n_cols, n_entries)
        incidence = SparseIncidence.from_pairs(
            n_rows, n_cols, rows, columns, ascending=True, distinct=True
        )
        reference = csr_matrix(
            (np.ones(n_entries, dtype=np.int64), (rows, columns)),
            shape=(n_rows, n_cols),
        )

        assert incidence.n_entries == n_entries == int(reference.nnz)
        assert np.array_equal(incidence.offsets, reference.indptr)
        assert np.array_equal(incidence.indices, reference.indices)
        assert np.array_equal(incidence.degrees, np.diff(reference.indptr))
        assert np.array_equal(incidence.dense(), reference.toarray())

    every_row([(7, 5, 30), (40, 9, 200), (64, 64, 512)], check)
