"""`bond_roots` compiled against the Python union-find loop (issue #986).

The roots, not only the partition, are pinned bitwise: which node labels a
component is `union_roots`' rule, and the Swendsen--Wang recolouring order
is taken in root order. One bond draw at beta = 1 on an open lattice at
three temperatures' worth of bond densities, and a random multigraph with
repeated and self bonds.
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.backend import Backend
from snakes_and_ladders.sample.potts_mcmc import bond_roots
from snakes_and_ladders.sim.graph import BoundaryCondition, lattice_graph

from tests._rows import every_value


@pytest.mark.oracle
def test_the_compiled_roots_are_the_python_loop_s_on_a_lattice() -> None:
    def check(density: float) -> None:
        graph = lattice_graph((40, 40), BoundaryCondition.OPEN, 1.0)
        rng = np.random.default_rng(986)
        bonds = graph.edge_index[rng.random(len(graph.edges)) < density]
        np.testing.assert_array_equal(
            bond_roots(graph.n_nodes, bonds, backend=Backend.RUST),
            bond_roots(graph.n_nodes, bonds, backend=Backend.PYTHON),
        )

    every_value([0.2, 0.5, 0.8], check)


@pytest.mark.oracle
def test_the_compiled_roots_are_the_python_loop_s_on_repeated_bonds() -> None:
    rng = np.random.default_rng(9861)
    bonds = rng.integers(0, 300, size=(900, 2))
    np.testing.assert_array_equal(
        bond_roots(300, bonds, backend=Backend.RUST),
        bond_roots(300, bonds, backend=Backend.PYTHON),
    )
