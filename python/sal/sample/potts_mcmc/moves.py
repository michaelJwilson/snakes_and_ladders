"""The move sets a Potts chain proposes from, the learned-action vocabulary over them, and the refusal of a cluster move on an antiferromagnet.

The other submodules of :mod:`sal.sample.potts_mcmc` import
this one, and it imports neither of them.
"""

from __future__ import annotations

from enum import StrEnum

from sal.sim.graph import PottsGraph


class PottsMove(StrEnum):
    """Which Monte Carlo move set a chain proposes from.

    A ``StrEnum`` for the reason `sal.search.infer.MoveSet` is one: an
    unrecognized move is rejected by ``mypy --strict`` at the call site rather
    than by a branch that silently falls through to a default.
    """

    SINGLE_SITE = "single-site"
    SWENDSEN_WANG = "swendsen-wang"
    WOLFF = "wolff"
    LOCALLY_BALANCED = "locally-balanced"
    GIBBS_WITH_GRADIENTS = "gibbs-with-gradients"
    NIEDERMAYER = "niedermayer"
    # Issue #1041: the field as bonds to a ghost site per label, and
    # Fortuin-Kasteleyn clusters proposed onto one label at a time.
    GHOST_SPIN = "ghost-spin"
    LABEL_DIRECTED = "label-directed"


#: The move sets built on the Fortuin-Kasteleyn bond construction, which needs
#: every coupling non-negative. Named rather than written as "not single-site":
#: the gradient-informed moves are single-flip and run on an antiferromagnet,
#: and a negation would have refused them with the clusters. Niedermayer's
#: rule builds clusters on a coupling of either sign and is not in the set,
#: which is the whole reason issue #756 adds it.
_CLUSTER_MOVES = frozenset(
    {
        PottsMove.SWENDSEN_WANG,
        PottsMove.WOLFF,
        PottsMove.GHOST_SPIN,
        PottsMove.LABEL_DIRECTED,
    }
)


class MoveKind(StrEnum):
    """Which move a learned action applies, over the move set :class:`PottsMove` names.

    Issue #779 moved it here from ``sal.learn.potts_nd``, which
    spelled it out rather than importing it. It sits beside the moves it names
    so one vocabulary is defined once: ``SWEEP`` is a pass of
    :data:`PottsMove.SINGLE_SITE`, ``FLIP`` is one site of that pass, and
    ``WOLFF`` and ``SWENDSEN_WANG`` are the two cluster moves under their own
    names.
    """

    FLIP = "flip"
    SWEEP = "sweep"
    WOLFF = "wolff"
    SWENDSEN_WANG = "swendsen-wang"
    NIEDERMAYER = "niedermayer"


def refuse_negative_coupling(move: PottsMove, graph: PottsGraph) -> None:
    """Refuse a Fortuin-Kasteleyn cluster move on a graph with a negative coupling.

    One message for every entry point that runs one, rather than a copy of
    it apiece: the bond probability ``1 - exp(-J)`` is not a probability
    below zero, and an antiferromagnet has no like-spin clusters to flip.
    Niedermayer's rule is not in :data:`_CLUSTER_MOVES` and is the move for
    that case (issue #756).
    """
    if move in _CLUSTER_MOVES and min(graph.coupling, default=0.0) < 0.0:
        msg = (
            f"{move} needs every coupling >= 0: the bond probability "
            "1 - exp(-J) is not a probability for J < 0, and an "
            "antiferromagnet has no like-spin clusters to flip"
        )
        raise ValueError(msg)
