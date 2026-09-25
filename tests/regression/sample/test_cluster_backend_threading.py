"""`cluster_backend` reaches the Swendsen-Wang pass from every Potts sampler that runs one (issue #1059).

`sweep_for` took `cluster_backend` and five samplers never handed it one, so
their cluster moves ran the Python pass whatever a caller wanted. Each now
takes it with the default it had: named `PYTHON` the run is the default run
bitwise, and named `RUST` it is another run of the same law, drawn in another
order (`_cluster_pass_rust`), which is what shows the argument arrived.
"""

from __future__ import annotations

import pickle
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from sal.backend import Backend
from sal.sample.annealed import (
    annealed_importance_sampling,
    population_annealing,
    simulated_tempering,
)
from sal.sample.potts_mcmc import PottsMove, sample_potts_pair
from sal.sample.tempered import tempered_potts_pair
from sal.sim.graph import BoundaryCondition, lattice_graph

GRAPH = lattice_graph((4, 4), BoundaryCondition.OPEN, 0.9)
#: A per-site field, so each cluster carries an accept step and the two
#: passes consume their uniforms in different orders.
FIELD = np.random.default_rng(7).normal(scale=0.5, size=(16, 3))
BETAS = (0.0, 0.5, 1.0)
MOVE = PottsMove.SWENDSEN_WANG


def _run(name: str, cluster_backend: Backend | None) -> object:
    keywords: dict[str, Any] = (
        {} if cluster_backend is None else {"cluster_backend": cluster_backend}
    )
    rng = np.random.default_rng(1059)
    runs: dict[str, Callable[[], object]] = {
        "annealed_importance_sampling": lambda: annealed_importance_sampling(
            GRAPH, FIELD, BETAS, rng, 4, move=MOVE, **keywords
        ),
        "population_annealing": lambda: population_annealing(
            GRAPH, FIELD, BETAS, rng, 4, move=MOVE, **keywords
        ),
        "simulated_tempering": lambda: simulated_tempering(
            GRAPH, FIELD, BETAS, np.zeros(3), rng, 20, move=MOVE, **keywords
        ),
        "tempered_potts_pair": lambda: tempered_potts_pair(
            GRAPH, FIELD, (1.0, 2.0), rng, 20, move=MOVE, houdayer=False, **keywords
        ),
        "sample_potts_pair": lambda: sample_potts_pair(
            GRAPH, FIELD, MOVE, rng, 20, houdayer=False, **keywords
        ),
    }
    return runs[name]()


NAMES = [
    "annealed_importance_sampling",
    "population_annealing",
    "simulated_tempering",
    "tempered_potts_pair",
    "sample_potts_pair",
]


@pytest.mark.backend
@pytest.mark.smoke
@pytest.mark.parametrize("name", NAMES)
def test_the_cluster_backend_reaches_the_pass_and_the_default_is_unchanged(
    name: str,
) -> None:
    default = pickle.dumps(_run(name, None))
    assert pickle.dumps(_run(name, Backend.PYTHON)) == default
    assert pickle.dumps(_run(name, Backend.RUST)) != default
