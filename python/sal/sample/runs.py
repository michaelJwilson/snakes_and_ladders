"""One result type per sampler concept (issue #1090).

Five annealers returned five shapes: the best point was ``labelling``,
``state``, ``theta`` or ``topology``, the cost was ``site_visits``,
``force_evaluations`` or not reported, and none said why it stopped. A
caller comparing two annealers wrote an adapter per pair.

:class:`Annealed` is the shape they share: the best point visited, where the
chain ended, what the run spent in a declared :class:`~sal.cost.Cost` unit,
and its :class:`~sal.opt.termination.Termination`. Each annealer's result is
a thin subclass adding the value in its own sign --- an ``energy`` it
minimizes or a ``log_*`` it maximizes, per root ``CLAUDE.md`` --- and the
per-step record it keeps. The value stays on the subclass because one
field name across the two signs is the drift #1089 removed.
"""

from __future__ import annotations

from dataclasses import dataclass

from sal.cost import Cost
from sal.opt.termination import Termination


@dataclass(frozen=True, kw_only=True)
class Annealed[T]:
    """What an annealing run found, where it ended, and what it cost.

    Parameters
    ----------
    best : T
        The best point visited. The *best* rather than the last: the final
        steps run cold but not at zero, so the chain can leave it.
    final : T
        Where the chain ended, so a caller can see whether the best was the
        end or a point passed through.
    spent : int
        What the run cost, in ``unit``.
    unit : Cost
        The unit ``spent`` is counted in, so two annealers are compared on a
        budget they both declare.
    termination : Termination
        Why the run stopped. An annealer runs its schedule to the end, so
        this is the schedule's length and not converged.
    """

    best: T
    final: T
    spent: int
    unit: Cost
    termination: Termination
