"""The alternation every expectation-maximization run here performs, once.

Three loops carried it --- `opt/hmm.py`'s `baum_welch_family`,
`opt/mixture.py`'s and `opt/emission_mixture.py`'s --- and differed in what
one step computes and in nothing else: each ran a step, read a
log-likelihood off the parameters that step was handed, and stopped when the
value stopped moving relative to its own magnitude (issue #859).

**A shared loop is not a shared rung.** `infra/ladder.py` holds the two
mixture fits as two rungs and they stay two: two entry points, two default
sets, two result types, two oracle pins. What is here is the alternation, and
no oracle is pinned against it. The E step, the M step, the report the M step
returns and the result built at the end stay with the caller, which is why
this module imports neither `torch` nor a model.

**The convergence test is relative**, as every threshold in `opt/` is: an
absolute one does not transfer across data sizes (`DEV.md`, issue #111).

**One budget, stated once** (issue #1059). Every EM entry point takes an
:class:`EmConfig` as ``config=`` and hands it here; a caller changes one field
with :func:`dataclasses.replace`, and the config is frozen, so no caller's
change reaches another's.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from sal.opt.termination import Termination


@dataclass(frozen=True)
class EmConfig:
    """How long an EM run may go, and when it has settled.

    Parameters
    ----------
    max_iterations : int
        Maximum EM iterations. Zero runs no step.
    tolerance : float
        Stop when the log-likelihood changes by less than this *relative* to
        its magnitude.
    """

    max_iterations: int = 500
    tolerance: float = 1e-12


#: The default: `opt.mixture`'s and `baum_welch`'s budget.
EM = EmConfig()

#: The count mixture's budget: `opt.emission_mixture`, `opt.split_merge` and
#: `sandbox.annealed_em` default to it, so their results are the ones they
#: returned before `EmConfig` existed (issue #1059, open question Q4b on
#: whether they move to `EM`).
EMISSION_MIXTURE_EM = EmConfig(max_iterations=200, tolerance=1e-10)


def em_loop[State](
    step: Callable[[State], tuple[State, float]],
    start: State,
    *,
    config: EmConfig,
    previous: float = -float("inf"),
) -> tuple[State, float, Termination]:
    """Alternate ``step`` until the log-likelihood settles or the budget runs out.

    Parameters
    ----------
    step : Callable[[State], tuple[State, float]]
        One E step and one M step. It returns the state its M step produced
        and the log-likelihood **at the state it was handed**, which is the
        order all three callers compute them in: the E step's evidence is
        what the M step is then given.
    start : State
        The starting parameters, in whatever form the caller's ``step``
        reads. Opaque here.
    config : EmConfig
        The budget and the relative tolerance. ``max_iterations=0`` runs no
        step and reports ``previous``.
    previous : float
        The log-likelihood at the state before ``start``, which the first
        step's value is tested against. ``-inf`` for a loop that begins at
        ``start``; a caller that has already stepped once (the tempered steps
        of issue #903) hands over its last value, so the loop tests the same
        change it would have tested had it run that step itself.

    Returns
    -------
    tuple[State, float, Termination]
        The last state, the last log-likelihood, and how the loop ended: the
        iterations run, and :attr:`Stop.CONVERGED` when the relative test
        stopped it or :attr:`Stop.BUDGET` when ``config.max_iterations`` did (issue
        #860). The caller builds its own result type from them and reports
        whatever its M steps said along the way.
    """
    state = start
    log_likelihood = previous
    iterations = 0
    converged = False
    while iterations < config.max_iterations:
        iterations += 1
        state, log_likelihood = step(state)
        if abs(log_likelihood - previous) <= config.tolerance * abs(log_likelihood):
            converged = True
            break
        previous = log_likelihood
    return state, log_likelihood, Termination.after(iterations, converged=converged)


class InnerSolve(Protocol):
    """What an M step reports of its own loop: :class:`sal.emissions.Reestimate` among others.

    Structural, so this module stays free of ``torch`` and of every model.
    """

    @property
    def converged(self) -> bool:
        """Whether the inner solve met its criterion."""
        ...

    @property
    def iterations(self) -> int:
        """Iterations the inner solve took; ``0`` for a closed form."""
        ...


def inner_termination(step: InnerSolve) -> Termination:
    """An M step's inner solve as a :class:`Termination`, the form every loop states (issue #1165).

    ``Reestimate`` reports ``converged`` and ``iterations`` and lives in
    :mod:`sal.emissions.base`, which imports no inference directory
    (``test_directory_imports.py``), so the conversion lives here rather than
    as a property on it. A closed-form step reads as converged after none.
    """
    return Termination.after(step.iterations, converged=step.converged)
