"""Pre-order sequence simulation under the k-state Jukes-Cantor model.

Implements alg. (simulate) of ``docs/tex/main.tex`` (Sec. "Simulation"):
draw the root state from pi, then walk the tree in pre-order, drawing each
node's state from the transition-probability row of its parent's state. All
sites are drawn independently and in parallel via vectorized NumPy
sampling, rather than one Python-level loop per site.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from snakes_and_ladders.numerics_rust import sample_rows
from snakes_and_ladders.sim.gtr import reversible_transition_probabilities
from snakes_and_ladders.sim.jc import jc_transition_probabilities
from snakes_and_ladders.sim.newick import to_newick
from snakes_and_ladders.sim.tree import Node, preorder


@dataclass(frozen=True)
class SimulatedDataset:
    """A simulated alignment together with the parameters that generated it.

    Ground truth ships with the data: a dataset without its generating
    ``(tau, k, pi, n_sites)`` is not validation-usable. The seed is not
    among them. It is declared in the fixture and read from there --
    `qa.sim_problem_sizes` prints `params.seed`, never a dataset's -- and a
    generator cannot be asked which seed made it, so recording one here
    would mean carrying a second source for one fact (issue #240).

    Parameters
    ----------
    alignment : dict[str, np.ndarray]
        Leaf name to its simulated states, each of shape (n_sites,) with
        entries in ``[0, k)``.
    node_states : dict[str, np.ndarray]
        Every node's (leaf and internal) simulated states, same shape and
        encoding as ``alignment``. The ancestral truth used to validate
        simulated substitution frequencies against the analytic model.
    newick : str
        The topology, with branch lengths, in Newick format.
    tau : Node
        The topology that was simulated over.
    k : int
        Number of states.
    pi : np.ndarray
        Root state distribution used.
    n_sites : int
        Number of sites simulated.
    """

    alignment: dict[str, np.ndarray]
    node_states: dict[str, np.ndarray]
    newick: str
    tau: Node
    k: int
    pi: np.ndarray
    n_sites: int


def simulate_alignment(
    tau: Node,
    k: int,
    pi: np.ndarray,
    rng: np.random.Generator,
    n_sites: int,
    rate_matrix: np.ndarray | None = None,
) -> SimulatedDataset:
    """Simulate an alignment under the k-state Jukes-Cantor model.

    Parameters
    ----------
    tau : Node
        Root of the topology, with branch lengths attached to each
        non-root node.
    k : int
        Number of states.
    pi : np.ndarray
        Root state distribution, shape (k,).
    rng : np.random.Generator
        Passed in rather than seeded here, so a caller drawing an *ensemble*
        gets independent datasets rather than the same one repeatedly --- the
        mistake a `seed` parameter invites, which `sim/CLAUDE.md` forbids and
        `qa.rl_reward_surface` has made once.
    n_sites : int
        Number of alignment columns to simulate.
    rate_matrix : np.ndarray | None
        Reversible rate matrix, shape ``(k, k)``, whose stationary
        distribution is ``pi``. ``None`` keeps the Jukes-Cantor closed form,
        which is the existing behaviour and stays the default: root
        ``CLAUDE.md`` forbids silent behaviour changes, and the substitution
        model is the last thing worth changing silently.

    Returns
    -------
    SimulatedDataset
        The simulated alignment, ancestral states, and generating truth.
    """
    if pi.shape != (k,):
        msg = f"pi has shape {pi.shape}, expected ({k},)"
        raise ValueError(msg)

    node_states: dict[str, np.ndarray] = {}

    def _walk(node: Node, parent_states: np.ndarray | None) -> None:
        if parent_states is None:
            states = rng.choice(k, size=n_sites, p=pi)
        else:
            if node.branch_length is None:
                msg = f"non-root node {node.name!r} has no branch_length"
                raise ValueError(msg)
            transition = (
                jc_transition_probabilities(node.branch_length, k=k)
                if rate_matrix is None
                else reversible_transition_probabilities(
                    rate_matrix, pi, node.branch_length
                )
            )
            states = sample_rows(rng, transition, parent_states)
        node_states[node.name] = states
        for child in node.children:
            _walk(child, states)

    _walk(tau, None)

    alignment = {
        node.name: node_states[node.name] for node in preorder(tau) if node.is_leaf
    }

    return SimulatedDataset(
        alignment=alignment,
        node_states=node_states,
        newick=to_newick(tau),
        tau=tau,
        k=k,
        pi=pi,
        n_sites=n_sites,
    )
