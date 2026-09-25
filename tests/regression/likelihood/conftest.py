"""What several likelihood test modules share, declared once.

`TREE` and `FIELD` are the six-node tree the message-passing tests are exact
on, once written in three modules that agreed by coincidence (issue #863,
design-audit row R19); `ROUTES` is every route to the tree likelihood;
`random_hmm` and `CHAIN_CASES` are the categorical chains path enumeration
referees (issue #982). Six nodes give branching, a mixed-sign coupling makes
messages non-monotone, and ``3 ** 6`` configurations make enumeration the
referee. Loopy graphs are built per module from `sim.graph.lattice_graph`, so
each module names its problem by import (`tests/_problems.py`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from sal.emissions import CategoricalEmission
from sal.likelihood import (
    pruning,
    pruning_analytic,
    pruning_rust,
    pruning_torch,
)
from sal.likelihood.brute_force import brute_force_log_likelihood
from sal.sim.graph import PottsGraph
from sal.sim.hmm import HmmParams
from sal.sim.tree import Node

#: The per-state field over `TREE`; three states, none of them favoured
#: enough to make a marginal degenerate.
FIELD = np.array([0.3, -0.7, 0.15])

#: Six nodes, five edges, one root with two children.
TREE = PottsGraph(
    n_nodes=6,
    edges=((0, 1), (0, 2), (1, 3), (1, 4), (2, 5)),
    coupling=(0.8, -0.4, 1.2, 0.3, 0.9),
)


#: A route from a tree, an alphabet, a root distribution and an alignment to
#: a log-likelihood; keywords (``rescale``) pass through where the route has
#: them. What every route is, once its branch lengths are read off the tree.
Route = Callable[..., Any]


def _torch(
    tau: Node,
    k: int,
    pi: np.ndarray,
    alignment: dict[str, np.ndarray],
    **keywords: Any,
) -> Any:
    """`pruning_torch`, with its branch lengths read off ``tau``."""
    return pruning_torch.log_likelihood(
        tau, k, pi, alignment, pruning_torch.branch_lengths_from_tree(tau), **keywords
    )


def _analytic(
    tau: Node,
    k: int,
    pi: np.ndarray,
    alignment: dict[str, np.ndarray],
    **keywords: Any,
) -> Any:
    """`pruning_analytic`, with its branch lengths read off ``tau``."""
    return pruning_analytic.log_likelihood(
        tau, k, pi, alignment, pruning_torch.branch_lengths_from_tree(tau), **keywords
    )


#: Every route that computes the tree likelihood, by the name a failure
#: reports. Moved here from `test_likelihood_validation.py` (issue #982) so
#: the refusals and the brute-force and rescaling checks in
#: `test_likelihood_pruning.py` ask every route the same questions.
ROUTES: dict[str, Route] = {
    "pruning": pruning.log_likelihood,
    "brute_force": brute_force_log_likelihood,
    "pruning_rust": pruning_rust.log_likelihood,
    "pruning_torch": _torch,
    "pruning_analytic": _analytic,
}


def random_hmm(n_states: int, n_symbols: int, length: int, seed: int) -> HmmParams:
    """A categorical HMM with Dirichlet(1) rows, drawn from ``seed``."""
    rng = np.random.default_rng(seed)
    return HmmParams(
        n_states=n_states,
        lengths=(length,),
        initial=rng.dirichlet(np.ones(n_states)),
        transition=rng.dirichlet(np.ones(n_states), size=n_states),
        emissions=CategoricalEmission(rng.dirichlet(np.ones(n_symbols), size=n_states)),
        seed=seed,
        tolerance=1e-12,
    )


#: ``(n_states, n_symbols, length, seed)``: chains of 32, 81, 64 and 64 paths,
#: small enough to enumerate. The observations are drawn from ``seed`` too.
CHAIN_CASES = [(2, 2, 5, 1), (3, 2, 4, 2), (2, 4, 6, 3), (4, 3, 3, 4)]
