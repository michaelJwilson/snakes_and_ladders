"""Deterministic annealing EM, for a mixture of emissions and for an HMM: declined, conserved (issues #903, #916, #1171).

The E step tempered along a schedule ending at one, then plain
expectation-maximization to its tolerance (Ueda & Nakano, 1998). For a
mixture the tempered responsibilities are ``softmax((log w_k + log p_k(x)) /
T)``. For an HMM the tempered E step is the existing forward-backward on
``log_density / T``, ``log_initial / T`` and ``log_transition / T``, with no
new kernel, handed to :func:`~sal.opt.hmm.baum_welch_family` through its
``e_step`` hook (issue #1170). Both were built to reach a higher maximum than
plain EM from the same start, and neither does, so neither is in ``opt`` and
both live here, on this module's rule for a declined route that was finished
(`sandbox/CLAUDE.md`).

**The mixture (#903).** On `emission_mixture/ci` annealed and plain EM reach
-7,836.806 from every one of six `data` starts, annealed in 56 to 58
iterations against plain's 27 to 160.

**The HMM (#1171).** The numbers are
``tests/regression/sandbox/test_annealed_em.py``'s release experiment, ten
random starts on a six-state Gaussian HMM whose plain EM reaches several
maxima.

**What it referees.** At ``T = 1`` a tempered step is plain EM's step: for
the mixture an empty schedule and a schedule of one step at one reproduce
:func:`~sal.opt.emission_mixture.expectation_maximization` bitwise, the
tolerance loop being the same :func:`~sal.opt.em.em_loop` handed the last
tempered value as ``previous``; for the HMM the tempered E step at one is
:func:`sal.likelihood.ragged.posteriors` bitwise. Each fit reports one stage
per tempered step and one for the plain EM after them (issue #1171).

**What would bring it back.** A draw where plain EM stops below annealed EM
from the same start, beyond EM's tolerance, over seeds and at the stress
size.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import torch

from sal.emissions import EmissionFamily
from sal.likelihood.ragged import posteriors
from sal.opt.em import EM, EMISSION_MIXTURE_EM, EmConfig, em_loop
from sal.opt.emission_mixture import EmissionMixtureFit
from sal.opt.hmm import EmFit, Posteriors, baum_welch_family
from sal.opt.m_step import MStep
from sal.opt.mixture import mixture_log_likelihood, responsibilities_torch
from sal.opt.termination import Termination
from sal.ragged import Ragged
from sal.sample.schedule import TempSchedule, ladder
from sal.track import current

#: One tempered step's stage: a loop of one step that ends on its count.
_TEMPERED = Termination.after(1, converged=False)


def log_tempered_evidence(joint: torch.Tensor, temperature: float) -> float:
    """``sum_i T log sum_k (w_k p_k(x_i))^(1/T)``, the value a tempered EM step ascends.

    ``joint`` is ``log w_k + log p_k(x_i)``, shape ``(n_samples,
    n_components)``. At ``T = 1`` it is the mixture log-likelihood. At fixed
    ``T`` it is ``max_q sum_i [sum_k q_ik log(w_k p_k(x_i)) + T H(q_i)]``, the
    maximum over ``q`` attained at the tempered responsibilities, so a
    tempered E step followed by an M step that maximizes the expected
    complete-data term cannot lower it (Ueda & Nakano, 1998). Maximized, so
    named ``log_*`` (root ``CLAUDE.md``); ``free_energy`` before #1171.

    Returns
    -------
    float
    """
    return float(temperature * torch.logsumexp(joint / temperature, dim=-1).sum())


@dataclass(frozen=True)
class AnnealedFit[FitT]:
    """An annealed EM fit and the tempered steps that led to it.

    Parameters
    ----------
    fit : FitT
        The fit: an :class:`~sal.opt.emission_mixture.EmissionMixtureFit` or
        an :class:`~sal.opt.hmm.EmFit`, its ``stages`` one per tempered step
        and then the plain EM's, its ``spent`` their total.
    temperatures : tuple[float, ...]
        The tempered steps' temperatures, in order.
    log_tempered_evidences : tuple[float, ...]
        :func:`log_tempered_evidence` at each step's temperature, at the state
        that step was handed.
    """

    fit: FitT
    temperatures: tuple[float, ...]
    log_tempered_evidences: tuple[float, ...]


def _schedule(
    temperatures: TempSchedule | Sequence[float], config: EmConfig
) -> tuple[float, ...]:
    """The temperatures, refused unless positive, finite and within the budget."""
    schedule = ladder(temperatures)
    for value in schedule:
        if not (math.isfinite(value) and value > 0.0):
            msg = f"a temperature is positive and finite, got {value}"
            raise ValueError(msg)
    if len(schedule) > config.max_iterations:
        msg = (
            f"{len(schedule)} tempered steps do not fit in "
            f"max_iterations={config.max_iterations}"
        )
        raise ValueError(msg)
    return schedule


def annealed_expectation_maximization(
    observations: np.ndarray | torch.Tensor,
    weights: torch.Tensor,
    components: EmissionFamily,
    temperatures: TempSchedule | Sequence[float],
    *,
    config: EmConfig = EMISSION_MIXTURE_EM,
    m_step: MStep | None = None,
) -> AnnealedFit[EmissionMixtureFit]:
    """One E and one M step at each temperature, then plain EM at one to ``config.tolerance``.

    The tempered steps count against ``config.max_iterations`` and are not tested
    for convergence. Each is recorded into the enclosing ``track`` run at its
    index: its temperature, the log-likelihood and
    :func:`log_tempered_evidence` at the state it was handed.

    Parameters
    ----------
    temperatures : TempSchedule | Sequence[float]
        One temperature per tempered step, read by
        :func:`sal.sample.schedule.ladder`: a schedule step by step, a
        sequence as given.
    m_step : MStep | None
        The component M step, as
        :func:`~sal.opt.emission_mixture.expectation_maximization` takes it;
        ``None`` is the family's ``reestimate``.

    Returns
    -------
    AnnealedFit[EmissionMixtureFit]

    Raises
    ------
    ValueError
        If a temperature is not positive and finite, there are more of them
        than ``config.max_iterations``, or a component's M step did not converge.
    """
    schedule = _schedule(temperatures, config)
    values = torch.as_tensor(observations, dtype=torch.float64)
    boundary = False
    attempt = 0
    evidences: list[float] = []

    def step(
        state: tuple[torch.Tensor, EmissionFamily, torch.Tensor],
        temperature: float = 1.0,
    ) -> tuple[tuple[torch.Tensor, EmissionFamily, torch.Tensor], float]:
        """One E step at ``temperature``, one M step, and the log-likelihood at the state given.

        At one it is `opt.emission_mixture.expectation_maximization`'s step,
        term for term.
        """
        nonlocal boundary, attempt
        attempt += 1
        present, family, _ = state
        log_weight = torch.log(present)
        log_likelihood = float(mixture_log_likelihood(values, log_weight, family))
        if temperature == 1.0:
            posterior = responsibilities_torch(values, log_weight, family)
        else:
            joint = log_weight + family.log_density(values)
            evidences.append(log_tempered_evidence(joint, temperature))
            posterior = torch.softmax(joint / temperature, dim=-1)
        reestimated = (
            family.reestimate(values, posterior)
            if m_step is None
            else m_step(family, values, posterior, None)
        )
        if not reestimated.converged:
            msg = (
                f"a component's M step did not settle at EM iteration "
                f"{attempt}: residual {reestimated.residual:.3e} after "
                f"{reestimated.iterations} inner iterations"
            )
            raise ValueError(msg)
        boundary = boundary or reestimated.at_boundary
        return (posterior.mean(dim=0), reestimated.emissions, posterior), log_likelihood

    start = (
        weights,
        components,
        torch.empty((values.shape[0], components.n_states), dtype=torch.float64),
    )
    tracked = current()
    previous = -float("inf")
    for index, temperature in enumerate(schedule):
        start, previous = step(start, temperature)
        if temperature == 1.0:
            # At one the tempered evidence is the log-likelihood, the same sum.
            evidences.append(previous)
        tracked.record(
            index,
            log_likelihood=previous,
            log_tempered_evidence=evidences[-1],
            temperature=temperature,
        )
    (weights, components, posterior), log_likelihood, termination = em_loop(
        step,
        start,
        config=replace(config, max_iterations=config.max_iterations - len(schedule)),
        previous=previous,
    )
    fit = EmissionMixtureFit(
        weights=weights,
        components=components,
        responsibilities=posterior,
        log_likelihood=log_likelihood,
        at_boundary=boundary,
        termination=termination,
        spent=len(schedule) + termination.iterations,
        stages=(*(_TEMPERED for _ in schedule), termination),
    )
    return AnnealedFit(fit, schedule, tuple(evidences))


@dataclass(frozen=True)
class TemperedEStep:
    """An :class:`~sal.opt.hmm.EStep` that runs forward-backward on every log input divided by ``temperature``.

    ``log_density / T``, ``log_initial / T`` and ``log_transition / T``,
    handed to :func:`sal.likelihood.ragged.posteriors`: the posterior of the
    tempered joint ``p(x, z)^(1/T)``, normalized over paths. Its summed
    evidence is ``log Z_T``, and ``T log Z_T`` is the HMM's
    :func:`log_tempered_evidence`. At one every division is by one, and the
    step is :func:`~sal.likelihood.ragged.posteriors` bitwise.

    Parameters
    ----------
    temperature : float
        ``T``, positive.
    """

    temperature: float

    def __call__(
        self,
        log_density: Ragged,
        log_initial: np.ndarray,
        log_transition: np.ndarray,
    ) -> Posteriors:
        """The tempered posteriors, as :func:`~sal.likelihood.ragged.posteriors` returns them."""
        t = self.temperature
        return posteriors(
            Ragged(log_density.values / t, log_density.lengths),
            log_initial / t,
            log_transition / t,
        )


def annealed_baum_welch(
    observations: np.ndarray | Ragged,
    log_initial: torch.Tensor,
    log_transition: torch.Tensor,
    components: EmissionFamily,
    temperatures: TempSchedule | Sequence[float],
    config: EmConfig = EM,
    *,
    covariate: np.ndarray | Ragged | None = None,
    fit_transition: bool = True,
    m_step: MStep | None = None,
) -> AnnealedFit[EmFit]:
    """Baum-Welch with one tempered E step and one M step at each temperature, then plain EM (issue #1171).

    :func:`~sal.opt.hmm.baum_welch_family`'s arguments, in its order, with
    the schedule after the family. Each tempered step is one iteration of
    :func:`~sal.opt.hmm.baum_welch_family` given a :class:`TemperedEStep`,
    so the M step --- the closed-form initial distribution and transition,
    and the family's ``reestimate`` or ``m_step`` --- reads the tempered
    posterior. The tempered steps count against ``config.max_iterations``,
    and the plain EM after them runs the rest from a fresh tolerance test.
    Each step is recorded into the enclosing ``track`` run at its index, as
    :func:`annealed_expectation_maximization` records it.

    Returns
    -------
    AnnealedFit[EmFit]
        The fit, its ``stages`` one per tempered step and then the plain
        EM's, ``spent`` their total, ``at_boundary`` and ``frozen`` over
        every stage.

    Raises
    ------
    ValueError
        If a temperature is not positive and finite, there are more of them
        than ``config.max_iterations``, or an M step did not converge.
    """
    schedule = _schedule(temperatures, config)
    one = replace(config, max_iterations=1)
    tracked = current()
    evidences: list[float] = []
    boundary = False
    frozen: set[int] = set()
    for index, temperature in enumerate(schedule):
        stepped = baum_welch_family(
            observations,
            log_initial,
            log_transition,
            components,
            one,
            covariate=covariate,
            fit_transition=fit_transition,
            e_step=TemperedEStep(temperature),
            m_step=m_step,
        )
        # The hook's summed evidence is log Z_T at the state the step was handed.
        evidences.append(temperature * stepped.log_likelihood)
        tracked.record(
            index, log_tempered_evidence=evidences[-1], temperature=temperature
        )
        log_initial, log_transition = stepped.log_initial, stepped.log_transition
        components = stepped.components
        boundary = boundary or stepped.at_boundary
        frozen.update(stepped.frozen)
    final = baum_welch_family(
        observations,
        log_initial,
        log_transition,
        components,
        replace(config, max_iterations=config.max_iterations - len(schedule)),
        covariate=covariate,
        fit_transition=fit_transition,
        m_step=m_step,
    )
    fit = replace(
        final,
        at_boundary=boundary or final.at_boundary,
        frozen=tuple(sorted(frozen.union(final.frozen))),
        spent=len(schedule) + final.spent,
        stages=(*(_TEMPERED for _ in schedule), final.termination),
    )
    return AnnealedFit(fit, schedule, tuple(evidences))
