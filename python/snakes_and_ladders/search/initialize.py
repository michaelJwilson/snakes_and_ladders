"""Data-driven starts for the tree: from distances, and from the Hadamard spectrum.

The tree's answer to the question `opt/initialize.py` leaves open (issue
#364): its three initializers read the objective and never the data, and
its docstring says why a data-driven one cannot live there --- ``opt/`` may
import no application module, and a test asserts it. The mixture's
k-means++ sits beside the mixture; these sit here, beside the search they
also start, because a start for a tree is a topology *and* its lengths.

Each satisfies :class:`snakes_and_ladders.opt.initialize.Initializer` on the
nose, taking an ``Objective`` and refusing one whose parameters it cannot
interpret, exactly as ``KMeansPlusPlus`` does. Each also offers a
``tree`` method, the topology with lengths a search may begin from, since a
branch-length objective holds its topology fixed and the same estimate is a
start for the discrete search as well.

:class:`FromDistances` is the moment estimator: pairwise distances
(:mod:`snakes_and_ladders.likelihood.distance`) and neighbor joining
(:mod:`snakes_and_ladders.search.neighbor_joining`), the Jukes--Cantor
closed form by default and the log-det distance for the general model.
:class:`FromHadamard` is the spectral one: the closest tree of the Hadamard
conjugation (:mod:`snakes_and_ladders.likelihood.hadamard`) on the
two-state recoding, weights rescaled to the model's units. A length the
estimate puts at or below the floor is raised to it, because the fit runs in
log coordinates and a zero or negative length has none; the floor is a
declared parameter, never a silent default.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from snakes_and_ladders.likelihood.distance import DistanceKind, distance_matrix
from snakes_and_ladders.likelihood.hadamard import (
    binary_recoding,
    closest_tree,
    hadamard_conjugation,
    recoding_scale,
    sequence_spectrum,
    split_weights,
)
from snakes_and_ladders.likelihood.objective import (
    BranchLengthObjective,
    SubstitutionModelObjective,
)
from snakes_and_ladders.opt.objective import Objective
from snakes_and_ladders.search.neighbor_joining import neighbor_joining, split_lengths
from snakes_and_ladders.sim.gtr import n_exchangeabilities
from snakes_and_ladders.sim.tree import Node

TreeObjective = BranchLengthObjective | SubstitutionModelObjective


def _check_floor(minimum_length: float) -> None:
    if minimum_length <= 0.0:
        msg = f"minimum_length must be positive, got {minimum_length}"
        raise ValueError(msg)


def _tree_objective(objective: Objective, who: str) -> TreeObjective:
    if not isinstance(objective, BranchLengthObjective | SubstitutionModelObjective):
        msg = (
            f"{who} seeds a tree's branch lengths and does not know what "
            f"{type(objective).__name__}'s parameters mean"
        )
        raise TypeError(msg)
    return objective


def _floored(tau: Node, minimum_length: float) -> Node:
    """``tau`` with every branch length raised to at least ``minimum_length``."""
    return Node(
        name=tau.name,
        branch_length=None
        if tau.branch_length is None
        else max(tau.branch_length, minimum_length),
        children=tuple(_floored(child, minimum_length) for child in tau.children),
    )


def _on_topology(
    topology: Node, lengths: Mapping[frozenset[str], float], minimum_length: float
) -> Node:
    """``topology`` carrying ``lengths``' value on every split it has, floored.

    A split ``lengths`` lacks --- the estimate's topology differs there ---
    takes the floor: nothing in the data speaks to it under this estimate,
    and a short branch is the uninformative choice the objective itself
    makes. On a rooted binary topology the two branches below the root share
    one split and take half its length each, the convention
    :mod:`snakes_and_ladders.likelihood.objective` reports them by.
    """
    all_leaves = _leaves(topology)
    anchor = min(all_leaves)
    halve = (
        {child.name for child in topology.children}
        if len(topology.children) == 2
        else set()
    )

    def rebuild(node: Node) -> Node:
        if node is topology:
            length = None
        else:
            below = _leaves(node)
            split = below if anchor not in below else all_leaves - below
            length = max(lengths.get(split, minimum_length), minimum_length)
            if node.name in halve:
                length /= 2.0
        return Node(
            name=node.name,
            branch_length=length,
            children=tuple(rebuild(child) for child in node.children),
        )

    return rebuild(topology)


def _theta(objective: TreeObjective, tree: Node) -> torch.Tensor:
    """The objective's coordinates for ``tree``'s lengths, its own start elsewhere."""
    if isinstance(objective, BranchLengthObjective):
        return objective.theta_from_truth(tree)
    # Equal exchangeabilities and a uniform pi are the objective's own
    # initial(), Jukes--Cantor exactly; only the branch block changes.
    k = objective.k
    return objective.theta_from_truth(
        tree, np.ones(n_exchangeabilities(k)), np.full(k, 1.0 / k)
    )


def _leaves(node: Node) -> frozenset[str]:
    if node.is_leaf:
        return frozenset((node.name,))
    return frozenset().union(*(_leaves(child) for child in node.children))


class FromDistances:
    """Neighbor joining on pairwise distances, as the start of a fit or a search.

    Parameters
    ----------
    kind : DistanceKind
        The pairwise estimator: the Jukes--Cantor closed form, or the
        log-det distance for the general time-reversible model.
    minimum_length : float
        The floor every returned branch length is raised to, positive.

    Raises
    ------
    ValueError
        If the floor is not positive.
    """

    def __init__(
        self,
        kind: DistanceKind = DistanceKind.JUKES_CANTOR,
        minimum_length: float = 1e-4,
    ) -> None:
        _check_floor(minimum_length)
        self.kind = kind
        self.minimum_length = minimum_length

    def tree(self, alignment: Mapping[str, np.ndarray], k: int) -> Node:
        """The neighbor-joining tree of the alignment's distances, lengths floored.

        Parameters
        ----------
        alignment : Mapping[str, np.ndarray]
            Taxon name to states, at least 3 taxa.
        k : int
            Number of states.

        Returns
        -------
        Node
            An unrooted binary tree in the trifurcating-root convention with
            a branch length on every non-root node.
        """
        names, distances, _ = distance_matrix(alignment, k, self.kind)
        return _floored(neighbor_joining(names, distances), self.minimum_length)

    def starts(self, objective: Objective) -> list[torch.Tensor]:
        """One start: the neighbor-joining lengths on the objective's own topology.

        Returns
        -------
        list[torch.Tensor]
            Exactly one point, in the objective's unconstrained coordinates,
            the neighbor-joining length on every branch the two topologies
            share and the floor elsewhere.

        Raises
        ------
        TypeError
            If the objective is not one of the tree's.
        """
        tree_objective = _tree_objective(objective, "FromDistances")
        estimate = self.tree(tree_objective.alignment, tree_objective.k)
        placed = _on_topology(
            tree_objective.topology, split_lengths(estimate), self.minimum_length
        )
        return [_theta(tree_objective, placed)]


class FromHadamard:
    """The closest tree of the Hadamard conjugation, as the start of a fit or a search.

    The alignment is recoded to two states
    (:func:`~snakes_and_ladders.likelihood.hadamard.binary_recoding`), its
    sequence spectrum conjugated, and the ``n - 3`` largest compatible split
    weights taken as the tree, every weight divided by the recoding's scale
    so the lengths are in the model's units.

    Parameters
    ----------
    minimum_length : float
        The floor every returned branch length is raised to, positive.

    Raises
    ------
    ValueError
        If the floor is not positive.
    """

    def __init__(self, minimum_length: float = 1e-4) -> None:
        _check_floor(minimum_length)
        self.minimum_length = minimum_length

    def tree(self, alignment: Mapping[str, np.ndarray], k: int) -> Node:
        """The closest tree of the alignment's spectrum, lengths in the model's units.

        Parameters
        ----------
        alignment : Mapping[str, np.ndarray]
            Taxon name to states, 3 to
            :data:`~snakes_and_ladders.likelihood.hadamard.MAX_TAXA` taxa.
        k : int
            Number of states, even.

        Returns
        -------
        Node
            An unrooted binary tree in the trifurcating-root convention with
            a branch length on every non-root node.
        """
        names, spectrum = sequence_spectrum(binary_recoding(alignment, k))
        weights = split_weights(hadamard_conjugation(spectrum), names)
        scale = recoding_scale(k)
        return closest_tree(
            {split: weight / scale for split, weight in weights.items()},
            names,
            self.minimum_length,
        )

    def starts(self, objective: Objective) -> list[torch.Tensor]:
        """One start: the conjugation's split weights on the objective's own topology.

        Returns
        -------
        list[torch.Tensor]
            Exactly one point, in the objective's unconstrained coordinates,
            the split weight on every branch the objective's topology has and
            the floor where the weight is below it.

        Raises
        ------
        TypeError
            If the objective is not one of the tree's.
        """
        tree_objective = _tree_objective(objective, "FromHadamard")
        names, spectrum = sequence_spectrum(
            binary_recoding(tree_objective.alignment, tree_objective.k)
        )
        scale = recoding_scale(tree_objective.k)
        weights = {
            split: weight / scale
            for split, weight in split_weights(
                hadamard_conjugation(spectrum), names
            ).items()
        }
        # The spectrum carries a weight for every split of any topology, so
        # the placement reads it directly rather than through the closest tree.
        placed = _on_topology(tree_objective.topology, weights, self.minimum_length)
        return [_theta(tree_objective, placed)]
