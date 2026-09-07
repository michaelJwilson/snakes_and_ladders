"""The phylogenetic environment: tree search as the MDP ``docs/tex`` states.

This is the application instance of :class:`snakes_and_ladders.learn.environment.Environment`,
and it lives here rather than in ``snakes_and_ladders.learn`` for the reason the
phylogenetic ``Objective`` lives in ``snakes_and_ladders.likelihood``: ``learn/`` may
import no application module, so the direction of the dependency has to run
application to infrastructure. An agent developed inside ``learn/`` would be
an agent shaped by trees, which is the thing that module exists to prevent.

The MDP is the one the technical document specifies. A **state** is a
topology; an **action** is a neighbour of it under NNI or SPR; the **reward**
is the improvement in log-likelihood; an **episode** ends on a step budget or
at a topology no move improves.

**Two reward models, and the choice is the point.**

``RewardModel.FITTED`` is the honest quantity: the *maximized* log-likelihood,
one L-BFGS solve per candidate. ``RewardModel.KNOWN`` evaluates the
likelihood at fixed, known parameters instead --- issue #131's simplification,
and the difference between a millisecond and a fifth of a second per
candidate.

There is a wrinkle worth stating plainly, because it is the same wrinkle
``opt/CLAUDE.md`` records about discrete moves. "The known parameters" do not
transfer across topologies: a different topology has different branches, so a
truth expressed as branch lengths on one tree means nothing on another. The
only thing that does transfer is a single scalar, so ``KNOWN`` scores every
candidate at one fixed branch length. That is a different surface from the
fitted one, not an approximation of it that happens to be cheap, and
:mod:`snakes_and_ladders.qa.rl_reward_surface` measures how far apart the two are rather
than assuming the gap is benign.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum

import numpy as np
import torch

from snakes_and_ladders.likelihood.pruning import log_likelihood
from snakes_and_ladders.search.infer import Model, MoveSet, score_topology
from snakes_and_ladders.search.topology import (
    Topology,
    leaf_bipartitions,
    nni_neighbours,
    random_topology,
    spr_neighbours,
)
from snakes_and_ladders.sim.tree import Node


class RewardModel(StrEnum):
    """Which log-likelihood the reward is a difference of.

    ``KNOWN`` evaluates at a fixed branch length with no optimization.
    ``FITTED`` maximizes over branch lengths per candidate, which is the
    quantity a phylogenetic search cares about. Measured on the 5-taxon
    fixture, it costs 113.7 ms against 352 us: a factor of 323.
    """

    KNOWN = "known"
    FITTED = "fitted"


def with_uniform_branch_lengths(topology: Topology, branch_length: float) -> Node:
    """``topology`` with every branch set to ``branch_length``.

    The only form in which "known parameters" survives a change of topology:
    a branch length is attached to an edge, and a different topology has
    different edges, so a single scalar is all that carries over.

    Parameters
    ----------
    topology : Topology
        The topology to label. Its own branch lengths, if any, are replaced.
    branch_length : float
        Length given to every non-root branch.

    Returns
    -------
    Node
        A tree with the same shape and uniform branch lengths.

    Raises
    ------
    ValueError
        If ``branch_length`` is not positive; a zero-length branch makes two
        nodes the same node and the likelihood no longer identifies the
        topology.
    """
    if branch_length <= 0.0:
        msg = f"branch_length must be > 0, got {branch_length}"
        raise ValueError(msg)

    def label(node: Node, *, root: bool) -> Node:
        return Node(
            name=node.name,
            branch_length=None if root else branch_length,
            children=tuple(label(child, root=False) for child in node.children),
        )

    return label(topology, root=True)


class TopologyEnvironment:
    """Tree search as a Markov decision process.

    Parameters
    ----------
    alignment : Mapping[str, np.ndarray]
        Observed states per taxon, each of shape ``(n_sites,)``.
    k : int
        Number of states.
    pi : np.ndarray
        Root distribution, shape ``(k,)``. Used only by ``KNOWN``; the fitted
        reward estimates it or, under Jukes-Cantor, knows it is uniform.
    branch_length : float
        The fixed branch length ``KNOWN`` scores at. Ignored by ``FITTED``.
        :mod:`snakes_and_ladders.qa.rl_reward_surface` measures how much the resulting
        surface depends on this: on the 6-taxon fixture the ranking's argmax
        is unchanged across ``0.02`` to ``1.00``.
    model : Model
        Substitution model for the fitted reward.
    reward : RewardModel
        Which log-likelihood to difference.
    moves : MoveSet
        Neighbourhood to propose from.

    Raises
    ------
    ValueError
        If the alignment has fewer than 4 taxa, below which no unrooted
        topology has a neighbour; or if ``KNOWN`` is combined with a model
        other than Jukes-Cantor, which the closed-form scorer does not cover.
    """

    def __init__(
        self,
        alignment: Mapping[str, np.ndarray],
        k: int,
        pi: np.ndarray,
        branch_length: float,
        model: Model = Model.JC,
        reward: RewardModel = RewardModel.KNOWN,
        moves: MoveSet = MoveSet.NNI,
    ) -> None:
        if len(alignment) < 4:
            msg = f"need at least 4 taxa to search, got {len(alignment)}"
            raise ValueError(msg)
        if reward is RewardModel.KNOWN and model is not Model.JC:
            # The closed-form scorer is the NumPy Jukes-Cantor pruning path,
            # which is where the speed comes from. A general-Q known-parameter
            # reward is worth having and is not this PR's.
            msg = (
                f"the {RewardModel.KNOWN.value} reward is implemented for "
                f"{Model.JC.value} only, got {model.value}"
            )
            raise ValueError(msg)

        self._alignment = dict(alignment)
        self._k = k
        self._pi = np.asarray(pi, dtype=np.float64)
        self._branch_length = branch_length
        self._model = model
        self._reward = reward
        self._moves = moves
        self._scores: dict[frozenset[frozenset[str]], float] = {}
        self._evaluations = 0

    @property
    def evaluations(self) -> int:
        """Distinct topologies scored so far, which is what a budget counts.

        Counted on cache misses, not on calls: a topology reached twice by
        different move sequences costs once, which is the same accounting
        ``snakes_and_ladders.search.infer`` uses and the reason both are stated in
        candidate scorings rather than seconds.
        """
        return self._evaluations

    def score(self, topology: Topology) -> float:
        """Log-likelihood of ``topology`` under this environment's reward model.

        Memoized on ``leaf_bipartitions``, which is rooting- and
        child-order-independent, so the same topology proposed by two
        different moves is recognized as the same work.
        """
        key = leaf_bipartitions(topology)
        cached = self._scores.get(key)
        if cached is not None:
            return cached
        if self._reward is RewardModel.KNOWN:
            value = log_likelihood(
                with_uniform_branch_lengths(topology, self._branch_length),
                self._k,
                self._pi,
                self._alignment,
            )
        else:
            value = score_topology(topology, self._alignment, self._k, self._model)
        self._scores[key] = value
        self._evaluations += 1
        return value

    def reset(self, rng: np.random.Generator) -> Topology:
        """Draw a starting topology uniformly over the leaf set's topologies."""
        return random_topology(sorted(self._alignment), rng)

    def actions(self, state: Topology) -> Sequence[Topology]:
        """The neighbourhood of ``state``, deduplicated.

        An action *is* the neighbour topology: a move is fully described by
        where it lands, and carrying the move separately would let the two
        disagree.
        """
        neighbourhood = nni_neighbours if self._moves is MoveSet.NNI else spr_neighbours
        seen = {leaf_bipartitions(state)}
        unique: list[Topology] = []
        for neighbour in neighbourhood(state):
            key = leaf_bipartitions(neighbour)
            if key not in seen:
                seen.add(key)
                unique.append(neighbour)
        return unique

    def step(self, state: Topology, action: Topology) -> tuple[Topology, float]:
        """Move to ``action`` and return it with the improvement it bought."""
        return action, self.score(action) - self.score(state)

    def features(self, state: Topology, actions: Sequence[Topology]) -> torch.Tensor:
        """``(len(actions), 1)``: the improvement each move would buy.

        One feature, deliberately. The policy is then a Boltzmann
        distribution over moves whose single weight is an inverse
        temperature, so the greedy searcher is its zero-temperature limit
        exactly as on the Potts landscape, and what the agent can learn is
        precisely *how much* it should accept a move that loses
        log-likelihood in order to reach a better local maximum. That is the
        question `docs/tex` raises about credit assignment, isolated to one
        parameter.

        Richer features -- split support, subtree sizes, alignment summaries
        -- are worth having, and are worth having *after* the reward surface
        they would be scored against has been validated, which is what
        :mod:`snakes_and_ladders.qa.rl_reward_surface` does.
        """
        current = self.score(state)
        rows = [[self.score(action) - current] for action in actions]
        return torch.tensor(rows, dtype=torch.float64).reshape(len(actions), 1)

    def n_features(self) -> int:
        """One: the improvement a move would buy."""
        return 1

    def is_terminal(self, state: Topology) -> bool:
        """Whether no move out of ``state`` improves the score.

        A property of the state, not of the policy, so the agent and the
        greedy baseline stop in the same places and are comparable at a
        matched budget.
        """
        current = self.score(state)
        return not any(self.score(action) > current for action in self.actions(state))
