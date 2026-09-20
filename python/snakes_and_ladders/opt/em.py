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
"""

from __future__ import annotations

from collections.abc import Callable


def em_loop[State](
    step: Callable[[State], tuple[State, float]],
    start: State,
    *,
    tolerance: float,
    max_iterations: int,
) -> tuple[State, float, int]:
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
    tolerance : float
        Stop when the log-likelihood changes by less than this *relative* to
        its magnitude.
    max_iterations : int
        Maximum iterations. Zero runs no step and reports ``-inf``.

    Returns
    -------
    tuple[State, float, int]
        The last state, the last log-likelihood, and the iterations run. The
        caller builds its own result type from them and reports whatever its
        M steps said along the way.
    """
    state = start
    previous = -float("inf")
    log_likelihood = previous
    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        state, log_likelihood = step(state)
        if abs(log_likelihood - previous) <= tolerance * abs(log_likelihood):
            break
        previous = log_likelihood
    return state, log_likelihood, iterations
