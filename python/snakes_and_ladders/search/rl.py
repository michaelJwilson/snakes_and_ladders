"""The phylogenetic environment: tree search as the MDP ``sec:policy-gradient`` of ``docs/tex/textbook.tex`` states.

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

**Two feature sets, and the second is issue #328.** With the improvement a
move buys as the only feature, the policy is a Boltzmann distribution over
moves whose one weight is an inverse temperature, and nothing it learns can
rank two moves differently from hill climbing. ``FeatureSet.FULL`` adds what
a move already computes without a fit: the parsimony change, the
pattern support of the split it breaks and of the split it makes, and the
sizes of the subtrees it exchanges. Every column is standardized within the
neighbourhood, and a column constant across a neighbourhood is excluded by
the unidentifiable-constant rule of ``learn/CLAUDE.md`` rather than carried
--- the normalized Robinson--Foulds distance from the state is the standing
example, since every NNI neighbour sits at the same distance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum

import numpy as np
import torch

from snakes_and_ladders.likelihood.parsimony import fitch_score
from snakes_and_ladders.likelihood.pruning import log_likelihood
from snakes_and_ladders.likelihood.pruning_torch import branch_order
from snakes_and_ladders.likelihood.pruning_torch import (
    log_likelihood as log_likelihood_torch,
)
from snakes_and_ladders.search.infer import Model, MoveSet, score_topology
from snakes_and_ladders.search.support import split_pattern_support
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


class FeatureSet(StrEnum):
    """Which columns :meth:`TopologyEnvironment.features` returns per move.

    ``IMPROVEMENT`` is the one column the environment had before issue
    #328, unstandardized, so every number pinned against it stands.
    ``FULL`` is the seven columns :data:`FEATURE_NAMES` lists, each
    standardized within the neighbourhood.
    """

    IMPROVEMENT = "improvement"
    FULL = "full"


#: The columns of each feature set, in order. Every one varies across an NNI
#: neighbourhood on the fixtures the suite uses; ``test_search_tree_policy``
#: asserts it, and refuses a planted column that does not.
FEATURE_NAMES: dict[FeatureSet, tuple[str, ...]] = {
    FeatureSet.IMPROVEMENT: ("improvement",),
    FeatureSet.FULL: (
        "improvement",
        "parsimony_change",
        "support_broken",
        "support_made",
        "subtree_detached",
        "subtree_attached",
        "subtree_imbalance",
    ),
}


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


def exchanged_subtrees(
    state_splits: frozenset[frozenset[str]], action_splits: frozenset[frozenset[str]]
) -> tuple[frozenset[str], frozenset[str]]:
    """The leaf sets a move detaches and attaches, from the two topologies' splits.

    A move breaks the splits in ``state_splits`` absent from
    ``action_splits`` and makes the converse. The **detached** set is the
    leaves in every broken split and no made one; the **attached** set is
    the leaves in every made split and no broken one. For an NNI move,
    which breaks one split ``S`` and makes one ``S'``, these are
    ``S - S'`` and ``S' - S``: two of the four subtrees around the edge, and
    the move is exactly their exchange (the other two exchanging is the same
    move). Which two is fixed by the split canonicalization of
    :func:`snakes_and_ladders.search.topology.leaf_bipartitions`: the anchor leaf's
    subtree is the one in neither split.

    Parameters
    ----------
    state_splits, action_splits : frozenset[frozenset[str]]
        ``leaf_bipartitions`` of the state and of the neighbour.

    Returns
    -------
    tuple[frozenset[str], frozenset[str]]
        The detached and the attached leaf sets. Empty when the topologies
        are the same, since then nothing is broken or made.
    """
    broken = state_splits - action_splits
    made = action_splits - state_splits
    if not broken or not made:
        return frozenset(), frozenset()
    detached = frozenset.intersection(*broken) - frozenset.union(*made)
    attached = frozenset.intersection(*made) - frozenset.union(*broken)
    return detached, attached


def standardize(rows: torch.Tensor) -> torch.Tensor:
    """Each column centred and scaled to unit spread across the rows.

    Population standard deviation, so one row standardizes to zero rather
    than dividing by zero; a column with no spread becomes zero, which is
    what a constant feature is worth to a softmax anyway.

    Parameters
    ----------
    rows : torch.Tensor
        Shape ``(n_actions, n_features)``.

    Returns
    -------
    torch.Tensor
        The same shape, each column of mean zero and, where it varies,
        unit standard deviation.
    """
    centred = rows - rows.mean(dim=0, keepdim=True)
    spread = rows.std(dim=0, unbiased=False, keepdim=True)
    return torch.where(
        spread > 0.0, centred / torch.where(spread > 0.0, spread, 1.0), 0.0
    )


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
        Substitution model for the fitted reward, and for the known one.
    reward : RewardModel
        Which log-likelihood to difference.
    moves : MoveSet
        Neighbourhood to propose from.
    features : FeatureSet
        Which columns :meth:`features` returns. The single improvement
        column by default, which is the environment before issue #328.
    rate_matrix : np.ndarray | None
        The rate matrix ``Q``, shape ``(k, k)``, the known reward scores a
        general model at; ``pi`` is its stationary distribution. Required
        with ``KNOWN`` under ``GTR``, refused under ``JC``, whose ``Q`` is
        fixed by ``k``.

    Raises
    ------
    ValueError
        If the alignment has fewer than 4 taxa, below which no unrooted
        topology has a neighbour; if ``KNOWN`` under ``GTR`` is given no
        rate matrix, or one of the wrong shape; or if a rate matrix is
        given under ``JC``.
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
        features: FeatureSet = FeatureSet.IMPROVEMENT,
        rate_matrix: np.ndarray | None = None,
    ) -> None:
        if len(alignment) < 4:
            msg = f"need at least 4 taxa to search, got {len(alignment)}"
            raise ValueError(msg)
        if model is Model.JC and rate_matrix is not None:
            msg = f"{Model.JC.value} fixes its rate matrix; none may be given"
            raise ValueError(msg)
        if reward is RewardModel.KNOWN and model is not Model.JC:
            if rate_matrix is None:
                msg = (
                    f"the {RewardModel.KNOWN.value} reward under {model.value} "
                    "needs a rate_matrix"
                )
                raise ValueError(msg)
            if np.shape(rate_matrix) != (k, k):
                msg = (
                    f"rate_matrix must have shape {(k, k)}, got {np.shape(rate_matrix)}"
                )
                raise ValueError(msg)

        self._alignment = dict(alignment)
        self._k = k
        self._pi = np.asarray(pi, dtype=np.float64)
        self._branch_length = branch_length
        self._model = model
        self._reward = reward
        self._moves = moves
        self._features = features
        self._rate_matrix = (
            None
            if rate_matrix is None
            else torch.as_tensor(np.asarray(rate_matrix, dtype=np.float64))
        )
        self._scores: dict[frozenset[frozenset[str]], float] = {}
        self._parsimony: dict[frozenset[frozenset[str]], int] = {}
        self._support: dict[frozenset[str], float] = {}
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

    @property
    def feature_set(self) -> FeatureSet:
        """Which columns :meth:`features` returns."""
        return self._features

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
        if self._reward is RewardModel.FITTED:
            value = score_topology(topology, self._alignment, self._k, self._model)
        elif self._rate_matrix is None:
            value = log_likelihood(
                with_uniform_branch_lengths(topology, self._branch_length),
                self._k,
                self._pi,
                self._alignment,
            )
        else:
            # The general-Q path: the differentiable recursion at the fixed
            # length on every branch, gradients unused. The same arithmetic
            # `score_topology` fits through, so the two surfaces differ only
            # in whether the lengths are optimized.
            lengths = torch.full(
                (len(branch_order(topology)),),
                self._branch_length,
                dtype=torch.float64,
            )
            with torch.no_grad():
                value = float(
                    log_likelihood_torch(
                        topology,
                        self._k,
                        self._pi,
                        self._alignment,
                        lengths,
                        rate_matrix=self._rate_matrix,
                    )
                )
        self._scores[key] = value
        self._evaluations += 1
        return value

    def parsimony(self, topology: Topology) -> int:
        """Fitch parsimony score of ``topology``, memoized as :meth:`score` is."""
        key = leaf_bipartitions(topology)
        cached = self._parsimony.get(key)
        if cached is None:
            cached = fitch_score(topology, self._alignment, self._k)
            self._parsimony[key] = cached
        return cached

    def split_support(self, split: frozenset[str]) -> float:
        """Fraction of sites compatible with ``split``, memoized per split.

        :func:`snakes_and_ladders.search.support.split_pattern_support`, which
        needs no fit and no resample: one pass over the alignment per
        distinct split, and a 7-taxon search meets at most 56 of them.
        """
        cached = self._support.get(split)
        if cached is None:
            cached = split_pattern_support(split, self._alignment, self._k)
            self._support[split] = cached
        return cached

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

    def raw_features(
        self, state: Topology, actions: Sequence[Topology]
    ) -> torch.Tensor:
        """The :data:`FEATURE_NAMES` columns of the full set, unstandardized.

        Per move: the improvement it buys; the change in Fitch parsimony
        score; the pattern support of the split it breaks and of the split
        it makes, each a mean where an SPR move breaks and makes several;
        the sizes of the subtree it detaches and the subtree it attaches
        (:func:`exchanged_subtrees`), and the absolute difference of the
        two. Every column is in the move's own units, which is what the
        pins in the regression suite read; :meth:`features` standardizes.
        """
        current = self.score(state)
        current_parsimony = self.parsimony(state)
        state_splits = leaf_bipartitions(state)
        rows = []
        for action in actions:
            action_splits = leaf_bipartitions(action)
            broken = state_splits - action_splits
            made = action_splits - state_splits
            detached, attached = exchanged_subtrees(state_splits, action_splits)
            rows.append(
                [
                    self.score(action) - current,
                    float(self.parsimony(action) - current_parsimony),
                    float(np.mean([self.split_support(split) for split in broken])),
                    float(np.mean([self.split_support(split) for split in made])),
                    float(len(detached)),
                    float(len(attached)),
                    float(abs(len(detached) - len(attached))),
                ]
            )
        width = len(FEATURE_NAMES[FeatureSet.FULL])
        return torch.tensor(rows, dtype=torch.float64).reshape(len(actions), width)

    def features(self, state: Topology, actions: Sequence[Topology]) -> torch.Tensor:
        """``(len(actions), n_features())``: what the policy scores each move by.

        Under ``FeatureSet.IMPROVEMENT``, the one unstandardized column of
        the improvement each move would buy. The policy is then a Boltzmann
        distribution over moves whose single weight is an inverse
        temperature, so the greedy searcher is its zero-temperature limit
        exactly as on the Potts landscape, and what the agent can learn is
        precisely *how much* it should accept a move that loses
        log-likelihood in order to reach a better local maximum --- the
        question ``sec:policy-gradient`` of ``docs/tex/textbook.tex`` raises
        about credit assignment, isolated to one parameter.

        Under ``FeatureSet.FULL``, :meth:`raw_features` with every column
        standardized within the neighbourhood (:func:`standardize`), so the
        softmax sees comparable scales: an improvement is tens of
        log-likelihood units, a support is a fraction, a subtree size an
        integer. Standardizing is a positive affine map per column, so the
        weight vector with zero on every column but the improvement still
        ranks moves as hill climbing does and the greedy searcher stays
        inside the policy class. Which columns vary within a neighbourhood
        is a property of the move set: under NNI all seven do on the
        suite's fixtures, while a distance from the state does not and is
        therefore not a column.
        """
        if self._features is FeatureSet.IMPROVEMENT:
            current = self.score(state)
            rows = [[self.score(action) - current] for action in actions]
            return torch.tensor(rows, dtype=torch.float64).reshape(len(actions), 1)
        return standardize(self.raw_features(state, actions))

    def n_features(self) -> int:
        """Width of :meth:`features`: one, or the seven of the full set."""
        return len(FEATURE_NAMES[self._features])

    def is_terminal(self, state: Topology) -> bool:
        """Whether no move out of ``state`` improves the score.

        A property of the state, not of the policy, so the agent and the
        greedy baseline stop in the same places and are comparable at a
        matched budget.
        """
        current = self.score(state)
        return not any(self.score(action) > current for action in self.actions(state))
