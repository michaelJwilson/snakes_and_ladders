"""Whether a loop finished and why, in one answer on every optimizer's result.

Issue #860. "Did it finish" reached a caller five ways
(`docs/reviews/2026-09-20-design.md`): a ``converged: bool`` field on four
results, a raise on exhaustion on two, an exception caught to a bool on one,
an iteration count the caller compared against the cap on four, and a derived
``optimal`` property on one --- with six results answering not at all. Five
encodings are five readings, and a caller that applies one result's to
another reads a fit that ran out of iterations as a fit that converged.

The answer is **additive**. Every result keeps the fields it had, with their
values, and gains a ``termination``. Since issue #1085 it is **required**, a
keyword-only field every producer fills, so a caller handles one case and not
three: a loop of a fixed step count ends on its budget, and says so, rather
than leaving the field ``None`` for *not known*.

A raise is not replaced by a field. Where a module refuses to return an
unconverged number it still refuses (`likelihood/CLAUDE.md`);
:attr:`Stop.REFUSED` is for a result that exists *because* a refusal was
caught, which is what `search.ground_state.run_max_product` does with the
flooding schedule's `ConvergenceError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Stop(StrEnum):
    """The branch a run left its loop by."""

    CONVERGED = "converged"
    """The run met its own stopping criterion: a tolerance, or a sweep that changed nothing."""

    BUDGET = "budget"
    """The count the run was given ran out, whether or not a criterion was tested."""

    REFUSED = "refused"
    """A refusal ended the run and the caller caught it: there is a result, and it is not a fit."""


@dataclass(frozen=True)
class Termination:
    """How a run ended, and after how many of the loop's own iterations.

    Parameters
    ----------
    converged : bool
        Whether the run met its own stopping criterion rather than running
        out. It is a statement about that criterion and about nothing else:
        `opt/CLAUDE.md` states that a converged fit is a statement about the
        gradient, never about the global minimum, and the same reading holds
        for every criterion here.
    iterations : int
        Iterations of the producer's own loop --- EM steps, coordinate-ascent
        sweeps, expansion cycles, hill-climbing rounds. The unit is the
        loop's, not the budget's: what a run *spent* is reported where a
        budget is held equal (`opt.budget`), and a result that counts in one
        unit does not answer in the other.
    reason : Stop
        The branch it left by. ``converged`` is :attr:`Stop.CONVERGED` and
        nothing else, so the two cannot drift apart and a reader may use
        either.

    Raises
    ------
    ValueError
        If ``converged`` and ``reason`` disagree, or ``iterations`` is
        negative.
    """

    converged: bool
    iterations: int
    reason: Stop

    def __post_init__(self) -> None:
        if self.converged != (self.reason is Stop.CONVERGED):
            msg = (
                f"converged={self.converged} and reason={self.reason} disagree: "
                f"a run converged exactly when it left by {Stop.CONVERGED}"
            )
            raise ValueError(msg)
        if self.iterations < 0:
            msg = f"a run takes at least no iterations, got {self.iterations}"
            raise ValueError(msg)

    @classmethod
    def after(cls, iterations: int, *, converged: bool) -> Termination:
        """The two branches a bounded loop has: its criterion, or its cap.

        The form every loop here ends in, so a producer states the count once
        and the reason follows from the flag it already kept.
        """
        return cls(
            converged=converged,
            iterations=iterations,
            reason=Stop.CONVERGED if converged else Stop.BUDGET,
        )
