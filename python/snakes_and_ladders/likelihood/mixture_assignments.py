"""Every component assignment, enumerated: the brute force a mixture's E step is held to.

Exponential in the number of observations and deliberately so, on the footing
:mod:`snakes_and_ladders.likelihood.hmm_paths` occupies for a chain. The
mixture's evidence and its responsibilities have a one-line factorized form
because the observations are independent, and that form is what
:mod:`snakes_and_ladders.opt.mixture` computes; summing ``k ** n`` assignments
term by term uses none of it, so agreement between the two is evidence rather
than a restatement.

What it catches is what a factorization can get wrong and still look right: a
normalization over the wrong axis, a weight broadcast against the components
rather than along them, a log-sum-exp shift shared where it may not be. Each
of those leaves the per-observation form self-consistent and moves the number
this module returns.

The mixture is a density model over the reals, so the evidence assembled here
carries no upper bound of one (``likelihood/CLAUDE.md``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from snakes_and_ladders.emissions import GaussianEmission
from snakes_and_ladders.enumeration import (
    MAX_ENUMERABLE_CONFIGURATIONS,
    accumulate,
    assignment_table,
    normalize,
)


@dataclass(frozen=True)
class AssignmentEnumeration:
    """The mixture's evidence and E step, summed over every assignment.

    Parameters
    ----------
    log_evidence : float
        ``log sum_z P(z) p(y | z)`` over all ``k ** n`` assignments. A log
        density, so it may be positive.
    responsibilities : np.ndarray
        ``P(z_i = k | y)``, shape ``(n_samples, n_components)``, each row
        summing to 1, obtained by marginalizing the enumerated posterior
        rather than by normalizing one observation at a time.
    assignment : np.ndarray
        The assignment of highest posterior probability, shape
        ``(n_samples,)``.
    """

    log_evidence: float
    responsibilities: np.ndarray
    assignment: np.ndarray


def enumerate_mixture_assignments(
    weights: np.ndarray,
    components: GaussianEmission,
    observations: np.ndarray,
    *,
    max_assignments: int = MAX_ENUMERABLE_CONFIGURATIONS,
) -> AssignmentEnumeration:
    """Sum the mixture's joint over all ``k ** n`` component assignments.

    Parameters
    ----------
    weights : np.ndarray
        Mixing weights, shape ``(n_components,)``.
    components : GaussianEmission
        The component densities.
    observations : np.ndarray
        Observations, shape ``(n_samples,)``.
    max_assignments : int
        Refuse above this many assignments.

    Returns
    -------
    AssignmentEnumeration

    Raises
    ------
    ValueError
        If ``observations`` is empty, ``weights`` does not match
        ``components``, or the enumeration would exceed ``max_assignments``.
    """
    values = np.asarray(observations, dtype=np.float64).reshape(-1)
    log_weight = np.log(np.asarray(weights, dtype=np.float64).reshape(-1))
    n_samples = int(values.shape[0])
    n_components = int(log_weight.shape[0])
    if n_samples == 0:
        msg = "observations must be non-empty"
        raise ValueError(msg)
    if n_components != components.n_states:
        msg = (
            f"weights carry {n_components} components and the family "
            f"{components.n_states}"
        )
        raise ValueError(msg)

    # (n_samples, n_components): log w_k + log N(y_i; mu_k, s_k), the only
    # per-observation quantity used. Everything below is a sum over whole
    # assignments.
    scored = log_weight + components.log_density(
        torch.as_tensor(values, dtype=torch.float64)
    ).numpy().reshape(n_samples, n_components)

    # The shared enumeration of :mod:`snakes_and_ladders.enumeration` (issue
    # #387): the table is every assignment as the digits of its index in base
    # ``n_components``, which is ``itertools.product`` without the
    # Python-level loop --- at 65,536 assignments the Python loop cost 0.4427
    # s per call and this costs 0.0222 s, 19.9x, for bitwise-identical
    # responsibilities; a test that calls the oracle forty times is the
    # difference between 18.80 s and 0.99 s, either side of the 10 s per-test
    # cap (``CLAUDE.md``, Vectorization). The sum below is still over whole
    # assignments --- no per-observation factorization is used, which is the
    # entire point of this module.
    assignments = assignment_table(
        n_components,
        n_samples,
        what=f"{n_components}**{n_samples} component assignments",
        limit=max_assignments,
    )
    log_joint = scored[np.arange(n_samples)[None, :], assignments].sum(axis=1)
    _, posterior, log_evidence = normalize(log_joint)

    return AssignmentEnumeration(
        log_evidence=log_evidence,
        responsibilities=accumulate(assignments, posterior, n_components),
        assignment=assignments[int(log_joint.argmax())].copy(),
    )
