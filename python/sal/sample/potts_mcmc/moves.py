"""The move sets a Potts chain proposes from, the learned-action vocabulary over them, and the refusal of a cluster move on an antiferromagnet.

The other submodules of :mod:`sal.sample.potts_mcmc` import
this one, and it imports neither of them.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import EnumType, StrEnum

from sal.sim.graph import PottsGraph

#: The members that left for :mod:`sal.sandbox.potts_moves` under its
#: "unsupported" rule (issue #1365), refused by name.
_SANDBOXED_MOVES = frozenset({"NIEDERMAYER", "GHOST_SPIN", "LABEL_DIRECTED"})


class _MoveType(EnumType):
    """:class:`PottsMove`'s metaclass: a sandboxed member's ``AttributeError`` names its new home."""

    def __getattr__(cls, name: str) -> object:
        if name in _SANDBOXED_MOVES:
            msg = (
                f"PottsMove.{name} is no longer supported: it is "
                f"sal.sandbox.potts_moves.SandboxMove.{name} (issue #1365)"
            )
            raise AttributeError(msg)
        raise AttributeError(name)


class PottsMove(StrEnum, metaclass=_MoveType):
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
    # Issue #1142: the Fortuin-Kasteleyn bonds, each cluster relabelled by
    # the heat bath on its summed field rather than a uniform proposal.
    # Deprecated by #1317: aliases of ``SWENDSEN_WANG`` and ``WOLFF`` with
    # ``recolour=Recolour.HEAT_BATH``, kept for one deprecation cycle.
    SWENDSEN_WANG_HEAT_BATH = "swendsen-wang-heat-bath"
    WOLFF_HEAT_BATH = "wolff-heat-bath"


#: The move sets built on the Fortuin-Kasteleyn bond construction, which needs
#: every coupling non-negative. Named rather than written as "not single-site":
#: the gradient-informed moves are single-flip and run on an antiferromagnet,
#: and a negation would have refused them with the clusters.
_CLUSTER_MOVES = frozenset(
    {
        PottsMove.SWENDSEN_WANG,
        PottsMove.WOLFF,
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


class Recolour(StrEnum):
    """How a cluster move draws the label of a cluster it has built (issue #1317).

    One argument on every cluster move rather than a paired enum member per
    move (meaning by type, #1091). ``HEAT_BATH`` draws the label
    ``proportional to exp(beta sum_C h[i, label])`` over the allowed labels,
    the exact conditional given the bonds; ``UNIFORM`` proposes a label
    uniformly and accepts it on the cluster's field difference; ``PER_MOVE``
    takes ``HEAT_BATH`` for every supported cluster move (issue #1323): the
    uniform proposal freezes under a strong field. The moves with no heat-bath
    form left for :mod:`sal.sandbox.potts_moves` (issue #1365).
    """

    HEAT_BATH = "heat-bath"
    UNIFORM = "uniform"
    # Issue #1323: each move's own default, the entry points' default.
    PER_MOVE = "per-move"


#: What every Potts entry point takes as ``move``: one move set, or a
#: sequence of them applied in order as one step, :data:`~sal.sandbox.potts_tempering.RungMoves`' per-step
#: form (issue #1317).
PottsMoves = PottsMove | Sequence[PottsMove]

#: Each cluster move with a heat-bath recolouring, and the member that runs it.
_HEAT_BATH_OF = {
    PottsMove.WOLFF: PottsMove.WOLFF_HEAT_BATH,
    PottsMove.SWENDSEN_WANG: PottsMove.SWENDSEN_WANG_HEAT_BATH,
    PottsMove.WOLFF_HEAT_BATH: PottsMove.WOLFF_HEAT_BATH,
    PottsMove.SWENDSEN_WANG_HEAT_BATH: PottsMove.SWENDSEN_WANG_HEAT_BATH,
}

#: The cluster moves a bare :class:`PottsMove` composes with a single-site
#: Gibbs sweep per step (issue #1323): a cluster move alone is not ergodic
#: under a forbidden label. A sequence is the move set exactly as given.
_COMPOSED = frozenset({PottsMove.WOLFF, PottsMove.SWENDSEN_WANG})


def composed(move: PottsMove) -> tuple[PottsMove, ...]:
    """A bare ``move`` as the set it runs: Wolff and Swendsen-Wang with a Gibbs sweep (issue #1323)."""
    return (move, PottsMove.SINGLE_SITE) if move in _COMPOSED else (move,)


def move_set(
    move: PottsMoves, recolour: Recolour = Recolour.PER_MOVE
) -> tuple[PottsMove, ...]:
    """``move`` as a non-empty tuple applied in order, each cluster move under ``recolour``.

    A single :class:`PottsMove` is checked first, as
    :func:`~sal.sandbox.potts_tempering.moves_per_rung` checks it: it is a
    ``str`` and so a ``Sequence``. Composition is read from the type (issue
    #1323): a bare ``WOLFF`` or ``SWENDSEN_WANG`` is the move composed with a
    single-site Gibbs sweep, ``(move, SINGLE_SITE)``, and a sequence is the
    set exactly as given, so ``move=[PottsMove.WOLFF]`` with
    ``recolour=Recolour.UNIFORM`` is the move before #1323. The result is a
    sequence, so resolving it again under the same ``recolour`` returns it
    unchanged. ``recolour`` reaches the cluster moves of
    the set; a single-site or gradient-informed move has no cluster to
    recolour and is returned as given. The deprecated members
    ``WOLFF_HEAT_BATH`` and ``SWENDSEN_WANG_HEAT_BATH`` name their recolouring
    and keep it under either value, so a call written before #1317 runs
    bitwise as it did.

    Raises
    ------
    ValueError
        If ``move`` is empty.
    TypeError
        If an entry is not a ``PottsMove``.
    """
    moves = composed(move) if isinstance(move, PottsMove) else tuple(move)
    if not moves:
        msg = "move is a PottsMove or a non-empty sequence of them"
        raise ValueError(msg)
    resolved = []
    for each in moves:
        if not isinstance(each, PottsMove):
            msg = f"move holds PottsMove, got {each!r}"
            raise TypeError(msg)
        heat_bath = recolour is not Recolour.UNIFORM
        resolved.append(_HEAT_BATH_OF.get(each, each) if heat_bath else each)
    return tuple(resolved)


def refuse_negative_coupling(move: PottsMoves, graph: PottsGraph) -> None:
    """Refuse a Fortuin-Kasteleyn cluster move on a graph with a negative coupling.

    One message for every entry point that runs one, rather than a copy of
    it apiece: the bond probability ``1 - exp(-J)`` is not a probability
    below zero, and an antiferromagnet has no like-spin clusters to flip.
    Niedermayer's rule, the move for that case (issue #756), is
    :mod:`sal.sandbox.potts_moves` (issue #1365).
    """
    clusters = [each for each in move_set(move) if each in _CLUSTER_MOVES]
    if clusters and min(graph.coupling, default=0.0) < 0.0:
        msg = (
            f"{clusters[0]} needs every coupling >= 0: the bond probability "
            "1 - exp(-J) is not a probability for J < 0, and an "
            "antiferromagnet has no like-spin clusters to flip"
        )
        raise ValueError(msg)
