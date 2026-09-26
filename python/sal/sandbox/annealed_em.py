"""Deterministic annealing EM for a mixture of emissions: declined, conserved (issues #903, #916).

The E step tempered, ``softmax((log w_k + log p_k(x)) / T)``, along a schedule
ending at one, then plain expectation-maximization to its tolerance (Ueda &
Nakano, 1998). It was built to reach a higher maximum than plain EM from the
same start. On `emission_mixture/ci` it does not: annealed and plain EM reach
-7,836.806 from every one of six `data` starts, annealed in 56 to 58
iterations against plain's 27 to 160 (#903). So it is not in
`opt.emission_mixture` and lives here, on this module's rule for a declined
route that was finished (`sandbox/CLAUDE.md`).

**What it referees.** At ``T = 1`` a tempered step is plain EM's step, so an
empty schedule and a schedule of one step at one reproduce
:func:`~sal.opt.emission_mixture.expectation_maximization`
bitwise: the tolerance loop is the same
:func:`~sal.opt.em.em_loop`, handed the last tempered value as
``previous`` so it tests the change it would have tested itself.

**What would bring it back.** A draw where plain EM stops below annealed EM
from the same start, beyond EM's tolerance, measured at the stress size.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import torch

from sal.emissions import EmissionFamily
from sal.opt.em import EMISSION_MIXTURE_EM, EmConfig, em_loop
from sal.opt.emission_mixture import EmissionMixtureFit
from sal.opt.mixture import mixture_log_likelihood, responsibilities_torch
from sal.opt.termination import Termination
from sal.track import current


def free_energy(joint: torch.Tensor, temperature: float) -> float:
    """``F_T = sum_i T log sum_k (w_k p_k(x_i))^(1/T)``, the value a tempered EM step ascends.

    ``joint`` is ``log w_k + log p_k(x_i)``, shape ``(n_samples,
    n_components)``. At ``T = 1`` it is the mixture log-likelihood. At fixed
    ``T`` it is ``max_q sum_i [sum_k q_ik log(w_k p_k(x_i)) + T H(q_i)]``, the
    maximum over ``q`` attained at the tempered responsibilities, so a
    tempered E step followed by an M step that maximizes the expected
    complete-data term cannot lower it (Ueda & Nakano, 1998).

    Returns
    -------
    float
    """
    return float(temperature * torch.logsumexp(joint / temperature, dim=-1).sum())


@dataclass(frozen=True)
class AnnealedFit:
    """An annealed EM fit and the tempered steps that led to it.

    Parameters
    ----------
    fit : EmissionMixtureFit
        The fit, its iterations counting the tempered steps.
    temperatures : tuple[float, ...]
        The tempered steps' temperatures, in order.
    free_energies : tuple[float, ...]
        :func:`free_energy` at each step's temperature, at the state that step
        was handed.
    """

    fit: EmissionMixtureFit
    temperatures: tuple[float, ...]
    free_energies: tuple[float, ...]


def annealed_expectation_maximization(
    observations: np.ndarray | torch.Tensor,
    weights: torch.Tensor,
    components: EmissionFamily,
    temperatures: Sequence[float],
    *,
    config: EmConfig = EMISSION_MIXTURE_EM,
) -> AnnealedFit:
    """One E and one M step at each temperature, then plain EM at one to ``config.tolerance``.

    The tempered steps count against ``config.max_iterations`` and are not tested
    for convergence. Each is recorded into the enclosing ``track`` run at its
    index: its temperature, the log-likelihood and :func:`free_energy` at the
    state it was handed.

    Returns
    -------
    AnnealedFit

    Raises
    ------
    ValueError
        If a temperature is not positive and finite, there are more of them
        than ``config.max_iterations``, or a component's M step did not converge.
    """
    schedule = [float(t) for t in temperatures]
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
    values = torch.as_tensor(observations, dtype=torch.float64)
    boundary = False
    attempt = 0
    free_energies: list[float] = []

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
            free_energies.append(free_energy(joint, temperature))
            posterior = torch.softmax(joint / temperature, dim=-1)
        reestimated = family.reestimate(values, posterior)
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
            # At one the free energy is the log-likelihood, the same sum.
            free_energies.append(previous)
        tracked.record(
            index,
            log_likelihood=previous,
            free_energy=free_energies[-1],
            temperature=temperature,
        )
    (weights, components, posterior), log_likelihood, termination = em_loop(
        step,
        start,
        config=replace(config, max_iterations=config.max_iterations - len(schedule)),
        previous=previous,
    )
    if schedule:
        termination = Termination.after(
            termination.iterations + len(schedule), converged=termination.converged
        )
    fit = EmissionMixtureFit(
        weights=weights,
        components=components,
        responsibilities=posterior,
        log_likelihood=log_likelihood,
        iterations=termination.iterations,
        at_boundary=boundary,
        termination=termination,
    )
    return AnnealedFit(fit, tuple(schedule), tuple(free_energies))
