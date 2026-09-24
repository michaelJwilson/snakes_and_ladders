"""The one Potts energy, against the oracle it replaced four loops with.

`sim.potts.energies` replaced three scorers (#277); `likelihood.potts.log_weights`
stays separate as its referee. Orders differ (a gather and pairwise sum against a
term per edge), so relative `1e-12`, not bitwise (#341). Instances: a chain, a
lattice, a mixed-sign glass, a per-site field, and one of the release-q10 rung's
size, where the edge term moved with the BLAS thread count (#1044).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from snakes_and_ladders.fixtures import load_params
from snakes_and_ladders.likelihood.potts import log_weights
from snakes_and_ladders.sim.fixtures import FIXTURES_DIR, fixture
from snakes_and_ladders.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from snakes_and_ladders.sim.potts import energies
from snakes_and_ladders.sim.potts_chain import PottsParams

#: What two exact routes over the same weights may differ by: the last bits
#: of a `float64` reduction, which belong to the host and not to the tree.
_EXACT = 1e-12

#: The release-q10 rung's site count (issue #1044), 71 x 71 = 5,041, the
#: size at which a threaded BLAS split the edge term; periodic, 10,082 edges.
_RELEASE_SIDE = 71


def _release_sized() -> tuple[PottsGraph, np.ndarray, int]:
    """A periodic lattice of the release-q10 rung's size, couplings of both signs."""
    rng = np.random.default_rng(1044)
    lattice = lattice_graph(
        (_RELEASE_SIDE, _RELEASE_SIDE), BoundaryCondition.PERIODIC, 1.0
    )
    coupling = tuple(float(j) for j in rng.normal(size=len(lattice.edges)))
    graph = PottsGraph(lattice.n_nodes, lattice.edges, coupling)
    return graph, rng.normal(size=(graph.n_nodes, 10)), 10


def _instances() -> dict[str, tuple[PottsGraph, np.ndarray, int]]:
    """The declared instances, as ``(graph, field, n_states)``."""
    chain = load_params(FIXTURES_DIR / "potts_chain/ci.yaml", PottsParams)
    lattice = fixture("potts_lattice", "ci").params
    glass_params = fixture("planted_glass", "ci").params
    (frustration,) = glass_params.glass_frustrations
    glass = glass_params.glass(frustration, np.random.default_rng(glass_params.seed))
    spatio = fixture("spatio_only", "ci").params
    return {
        "potts_chain": (
            lattice_graph(
                (chain.chain_length,), BoundaryCondition.OPEN, chain.coupling
            ),
            chain.field,
            chain.n_states,
        ),
        "potts_lattice": (
            lattice_graph(lattice.shape, lattice.boundary, lattice.coupling),
            lattice.field,
            lattice.n_states,
        ),
        "planted_glass": (glass.graph, np.zeros(2), 2),
        "spatio_only": (spatio.graph, spatio.field, spatio.n_classes),
        "release_sized": _release_sized(),
    }


INSTANCES = _instances()


@pytest.mark.oracle
@pytest.mark.parametrize("name", sorted(INSTANCES))
def test_the_energy_is_the_negated_log_weight_of_the_configuration(name: str) -> None:
    graph, field, n_states = INSTANCES[name]
    states = np.random.default_rng(11).integers(0, n_states, size=(64, graph.n_nodes))

    assert_allclose(
        energies(graph, field, states),
        -log_weights(graph, field, states),
        rtol=_EXACT,
        atol=0.0,
    )


@pytest.mark.oracle
@pytest.mark.parametrize("name", sorted(INSTANCES))
def test_one_labelling_is_the_block_of_one(name: str) -> None:
    # The reduction the single-labelling callers take: `state[None]` and
    # element zero. A block scored row by row and a block scored at once
    # must agree bitwise, or `sim.potts.energy` and the sampler's
    # `energies` are two functions again.
    graph, field, n_states = INSTANCES[name]
    states = np.random.default_rng(12).integers(0, n_states, size=(8, graph.n_nodes))

    block = energies(graph, field, states)

    assert np.array_equal(
        block, np.array([energies(graph, field, state[None])[0] for state in states])
    )


@pytest.mark.smoke
def test_a_configuration_of_the_wrong_width_is_refused() -> None:
    # Broadcast rather than refused, this returns energies for a different
    # model -- the failure `log_weights` refuses the same way.
    graph = lattice_graph((4,), BoundaryCondition.OPEN, 0.5)

    with pytest.raises(ValueError, match="columns for a graph of 4 nodes"):
        energies(graph, np.zeros(2), np.zeros((3, 5), dtype=np.int64))


#: The child's script: the release-sized instance scored one labelling at a
#: time, as the annealed runs call it, and as blocks of 6 and 64, as the
#: tempering and annealed-importance callers do; printed as hex, bit for bit.
_SCORE = """
import sys
sys.path.insert(0, {here!r})
from test_potts_energy import _release_sized
import numpy as np
from snakes_and_ladders.sim.potts import energies
graph, field, n_states = _release_sized()
states = np.random.default_rng(3).integers(0, n_states, size=(64, graph.n_nodes))
values = [energies(graph, field, state[None])[0] for state in states[:8]]
values += list(energies(graph, field, states[:6])) + list(energies(graph, field, states))
print(" ".join(float(value).hex() for value in values))
"""


def _scored_under(threads: int) -> str:
    """The child's energies with every BLAS pool held to ``threads``."""
    # Set for the child alone: the owner's constraint on #1044 is that no
    # thread setting changes anywhere the package or its callers can see.
    limits = {
        name: str(threads)
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    }
    script = _SCORE.format(here=str(Path(__file__).resolve().parent))
    return subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, **limits},
    ).stdout


@pytest.mark.smoke
def test_the_energy_does_not_move_with_the_blas_thread_count() -> None:
    # Issue #1044: through a BLAS gemv the edge term split across threads and
    # the n = 1 energy moved by 4.5e-13 between one thread and four at this
    # size. A reduction outside BLAS has one order at any thread count.
    assert _scored_under(1) == _scored_under(4)
