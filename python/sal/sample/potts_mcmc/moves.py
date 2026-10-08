"""The move sets a Potts chain proposes from, the learned-action vocabulary over them, and the refusal of a cluster move on an antiferromagnet.

The other submodules of :mod:`sal.sample.potts_mcmc` import
this one, and it imports neither of them.
"""

from __future__ import annotations

from collections.abc import Sequence
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
    # Issue #1142: the Fortuin-Kasteleyn bonds, each cluster relabelled by
    # the heat bath on its summed field rather than a uniform proposal.
    # Deprecated by #1317: aliases of ``SWENDSEN_WANG`` and ``WOLFF`` with
    # ``recolour=Recolour.HEAT_BATH``, kept for one deprecation cycle.
    SWENDSEN_WANG_HEAT_BATH = "swendsen-wang-heat-bath"
    WOLFF_HEAT_BATH = "wolff-heat-bath"


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
        PottsMove.SWENDSEN_WANG_HEAT_BATH,
        PottsMove.WOLFF_HEAT_BATH,
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


class Recolour(StrEnum):
    """How a cluster move draws the label of a cluster it has built (issue #1317).

    One argument on every cluster move rather than a paired enum member per
    move (meaning by type, #1091). ``HEAT_BATH`` draws the label
    ``proportional to exp(beta sum_C h[i, label])`` over the allowed labels,
    the exact conditional given the bonds; ``UNIFORM`` proposes a label
    uniformly and accepts it on the cluster's field difference.
    """

    HEAT_BATH = "heat-bath"
    UNIFORM = "uniform"


#: What every Potts entry point takes as ``move``: one move set, or a
#: sequence of them applied in order as one step, :data:`RungMoves`' per-step
#: form (issue #1317).
PottsMoves = PottsMove | Sequence[PottsMove]

#: Each cluster move with a heat-bath recolouring, and the member that runs it.
_HEAT_BATH_OF = {
    PottsMove.WOLFF: PottsMove.WOLFF_HEAT_BATH,
    PottsMove.SWENDSEN_WANG: PottsMove.SWENDSEN_WANG_HEAT_BATH,
    PottsMove.WOLFF_HEAT_BATH: PottsMove.WOLFF_HEAT_BATH,
    PottsMove.SWENDSEN_WANG_HEAT_BATH: PottsMove.SWENDSEN_WANG_HEAT_BATH,
}

#: The cluster moves whose label draw has no heat-bath form: Niedermayer's
#: uniform proposal, the ghost spin's bonded label, the directed target.
_NO_HEAT_BATH = frozenset(
    {PottsMove.NIEDERMAYER, PottsMove.GHOST_SPIN, PottsMove.LABEL_DIRECTED}
)


def move_set(
    move: PottsMoves, recolour: Recolour = Recolour.UNIFORM
) -> tuple[PottsMove, ...]:
    """``move`` as a non-empty tuple applied in order, each cluster move under ``recolour``.

    A single :class:`PottsMove` is checked first, as
    :func:`~sal.sample.potts_mcmc.chains.moves_per_rung` checks it: it is a
    ``str`` and so a ``Sequence``. ``recolour`` reaches the cluster moves of
    the set; a single-site or gradient-informed move has no cluster to
    recolour and is returned as given. The deprecated members
    ``WOLFF_HEAT_BATH`` and ``SWENDSEN_WANG_HEAT_BATH`` name their recolouring
    and keep it under either value, so a call written before #1317 runs
    bitwise as it did.

    Raises
    ------
    ValueError
        If ``move`` is empty, or ``recolour`` is ``HEAT_BATH`` on a cluster
        move with no heat-bath form, named.
    TypeError
        If an entry is not a ``PottsMove``.
    """
    moves = (move,) if isinstance(move, PottsMove) else tuple(move)
    if not moves:
        msg = "move is a PottsMove or a non-empty sequence of them"
        raise ValueError(msg)
    resolved = []
    for each in moves:
        if not isinstance(each, PottsMove):
            msg = f"move holds PottsMove, got {each!r}"
            raise TypeError(msg)
        if recolour is Recolour.HEAT_BATH and each in _NO_HEAT_BATH:
            msg = (
                f"{each} has no heat-bath recolouring: pass "
                "recolour=Recolour.UNIFORM, or use wolff or swendsen-wang"
            )
            raise ValueError(msg)
        heat_bath = recolour is Recolour.HEAT_BATH
        resolved.append(_HEAT_BATH_OF.get(each, each) if heat_bath else each)
    return tuple(resolved)


def refuse_negative_coupling(move: PottsMoves, graph: PottsGraph) -> None:
    """Refuse a Fortuin-Kasteleyn cluster move on a graph with a negative coupling.

    One message for every entry point that runs one, rather than a copy of
    it apiece: the bond probability ``1 - exp(-J)`` is not a probability
    below zero, and an antiferromagnet has no like-spin clusters to flip.
    Niedermayer's rule is not in :data:`_CLUSTER_MOVES` and is the move for
    that case (issue #756).
    """
    clusters = [each for each in move_set(move) if each in _CLUSTER_MOVES]
    if clusters and min(graph.coupling, default=0.0) < 0.0:
        msg = (
            f"{clusters[0]} needs every coupling >= 0: the bond probability "
            "1 - exp(-J) is not a probability for J < 0, and an "
            "antiferromagnet has no like-spin clusters to flip"
        )
        raise ValueError(msg)
