"""The rectangular Baum-Welch, conserved as the referee of the ragged one.

Issue #666, under `sandbox/CLAUDE.md`'s **superseded in capability** clause: the
route that replaced this one expresses something it cannot --- segments of
unequal length --- and is not faster, so there is nothing to measure and the
question is only what this still answers. It answers the **equal-length** case,
and `opt.hmm.baum_welch_family` must reproduce it **bit for bit** there.

That is a sharper referee than it looks. The ragged path pads to the longest
segment and masks; at equal lengths the mask is everywhere true, so every
`torch.where` must fall through to exactly the arithmetic below, in the same
order. A masked implementation that quietly reassociates a sum, or gathers an
evidence from the wrong column, differs here in the last bits and nowhere else.

Verbatim from `opt/hmm.py` as of `main`, renamed. It is not maintained: it is
the thing the maintained route is checked against.
"""

from __future__ import annotations

import numpy as np
import torch

from sal.emissions import EmissionFamily
from sal.opt.em import EM, EmConfig
from sal.opt.hmm import EmFit
from sal.opt.termination import Termination


def baum_welch_rectangular(
    observations: np.ndarray,
    log_initial: torch.Tensor,
    log_transition: torch.Tensor,
    emissions: EmissionFamily,
    config: EmConfig = EM,
    covariate: np.ndarray | None = None,
) -> EmFit:
    """Baum-Welch over any emission family, with no autodiff involved.

    The E step is the model: forward and backward messages in log space,
    identical whatever a state emits, and so is the M step for the initial
    distribution and the transitions, both simplex-valued for every family.
    Only the emission M step differs, and it is delegated to the family.

    Parameters
    ----------
    observations : np.ndarray
        Observations, shape ``(n_sequences, length)`` followed by whatever
        trailing axes the family's observation carries --- none for a scalar
        observation, a channel axis for a family over a pair of counts.
        Symbol indices or real values, as the family says.
    emissions : EmissionFamily
        Starting emission family.
    config : EmConfig
        The EM budget and its relative tolerance; :data:`~sal.opt.em.EM`,
        500 iterations at 1e-12, by default. Relative, since an absolute
        tolerance does not transfer across data sizes (``DEV.md``, issue #111).
    log_initial, log_transition : torch.Tensor
        Starting parameters, as log-probabilities. ``log_transition`` is either
        ``(m, m)``, one kernel for the whole chain, or ``(length - 1, m, m)``,
        one per step (issue #658).

        **A per-step kernel is conditioned on, not fitted.** It carries
        ``(length - 1) * m * (m - 1)`` free values against ``length - 1``
        transitions per sequence, so at the sequence counts this repository
        runs it is not identifiable and an M step that re-estimated it would
        return the posterior it was handed. Given one, this function holds it
        fixed and fits the initial distribution and the emissions --- the same
        standing a covariate has, and for the same reason. Given a single
        matrix it fits that matrix, exactly as before.
    covariate : np.ndarray | None
        What each observation is scored against, carrying the observations'
        leading ``(n_sequences, length)`` -- an exposure for a rate family, a trial count for a
        bounded one (issue #652). It reaches both seams of the loop, the E
        step's scoring and the emission M step, because a fit that scores
        against an exposure and re-estimates without it is fitting two
        different models. ``None`` is the model this function had before.
        The family broadcasts a covariate along the states, so it wants a trailing singleton axis; the covariate is stored with the observations' own axes and the singleton is added here, where the observation layout is known. A caller should not have to carry a shape that exists for the family's broadcast.

    Returns
    -------
    EmFit
        The fitted parameters, the final log-likelihood, and whether any M
        step reported a parameter at the edge of what the data identifies.

    Raises
    ------
    ValueError
        If the family refuses its own re-estimate. A Gaussian family does so
        when a state's variance reaches its floor, which is an approach to a
        degenerate optimum rather than a convergence, and is reported as such
        rather than clamped away.
    """
    data = torch.as_tensor(observations, dtype=emissions.observation_dtype)
    # The leading two axes are the sequence and the position. What follows them
    # is the family's own: none where an observation is a scalar, and one or
    # more where it is not --- a family over a pair of counts carries a channel
    # axis. Unpacking the whole shape refused every such family outright, so a
    # family could be made to condition on a covariate and still not be
    # fittable here (issue #658).
    n_sequences, length = data.shape[:2]
    m = emissions.n_states
    varying = log_transition.shape != (m, m)
    if varying and log_transition.shape != (max(length - 1, 0), m, m):
        msg = (
            f"log_transition {tuple(log_transition.shape)} is neither ({m}, {m}) "
            f"nor ({max(length - 1, 0)}, {m}, {m}) for a chain of {length} "
            f"positions over {m} states"
        )
        raise ValueError(msg)
    kernels = (
        log_transition if varying else log_transition.expand(max(length - 1, 0), m, m)
    )
    # The trailing singleton the single-channel families broadcast over their
    # states with is added only where the covariate has no axes of its own; a
    # covariate carrying the family's axes is passed through, because the
    # singleton then belongs inside each of them and the family is what puts it
    # there (issue #658).
    exposure: torch.Tensor | None = None
    if covariate is not None:
        exposure = torch.as_tensor(covariate, dtype=torch.float64)
        if exposure.ndim == data.ndim == 2:
            exposure = exposure[..., None]

    previous = -float("inf")
    iterations, converged = 0, False
    log_likelihood = previous
    at_boundary = False
    for iterations in range(1, config.max_iterations + 1):  # noqa: B007
        # --- E step: forward and backward messages in log space ----------
        emit = emissions.log_density(data, covariate=exposure)
        alpha = torch.empty((n_sequences, length, m), dtype=log_initial.dtype)
        alpha[:, 0] = log_initial.unsqueeze(0) + emit[:, 0]
        for t in range(1, length):
            kernel = kernels[t - 1] if varying else log_transition
            alpha[:, t] = (
                torch.logsumexp(
                    alpha[:, t - 1].unsqueeze(2) + kernel.unsqueeze(0), dim=1
                )
                + emit[:, t]
            )
        beta = torch.zeros((n_sequences, length, m), dtype=log_initial.dtype)
        for t in range(length - 2, -1, -1):
            kernel = kernels[t] if varying else log_transition
            beta[:, t] = torch.logsumexp(
                kernel.unsqueeze(0) + (emit[:, t + 1] + beta[:, t + 1]).unsqueeze(1),
                dim=2,
            )

        evidence = torch.logsumexp(alpha[:, -1], dim=1)
        log_likelihood = float(evidence.sum())

        gamma = alpha + beta - evidence[:, None, None]
        xi = (
            alpha[:, :-1].unsqueeze(3)
            + kernels.unsqueeze(0)
            + (emit[:, 1:] + beta[:, 1:]).unsqueeze(2)
            - evidence[:, None, None, None]
        )

        # --- M step: normalized expected counts, then the family's own ---
        log_initial = torch.logsumexp(gamma[:, 0], dim=0) - torch.log(
            torch.tensor(float(n_sequences), dtype=gamma.dtype)
        )
        if not varying:
            transition_counts = torch.logsumexp(xi.reshape(-1, m, m), dim=0)
            log_transition = transition_counts - torch.logsumexp(
                transition_counts, dim=1, keepdim=True
            )
            kernels = log_transition.expand(max(length - 1, 0), m, m)
        step = emissions.reestimate(data, torch.exp(gamma), covariate=exposure)
        if not step.converged:
            msg = (
                f"the emission M step did not settle after {step.iterations} "
                f"iterations, at a relative change of {step.residual:.3e}: a "
                f"parameter read off iterations that never converged is not an "
                f"estimate, and a monotone outer likelihood would not have "
                f"shown it"
            )
            raise ValueError(msg)
        emissions = step.emissions
        at_boundary = at_boundary or step.at_boundary

        if abs(log_likelihood - previous) <= config.tolerance * abs(log_likelihood):
            converged = True
            break
        previous = log_likelihood

    return EmFit(
        log_initial=log_initial,
        log_transition=log_transition,
        emissions=emissions,
        log_likelihood=log_likelihood,
        emission_at_boundary=at_boundary,
        termination=Termination.after(iterations, converged=converged),
    )
