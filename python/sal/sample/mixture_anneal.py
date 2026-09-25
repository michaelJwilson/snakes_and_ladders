"""Simulated annealing over a mixture's component assignments (issues #901, #1059).

A mixture of independent observations is a factor graph with unary factors
alone, so a heat-bath sweep over the assignments is one independent draw per
observation. :func:`anneal_assignments` runs those sweeps down a falling
temperature, the components re-estimated by the family's own M step at each
sweep's draw. It draws, so it is a sampler and lives here; it sat in
:mod:`sal.opt.emission_mixture` beside the EM loop until #1059.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from sal.emissions import EmissionFamily
from sal.opt.mixture import component_log_density
from sal.track import current


def draw_assignments(
    joint: torch.Tensor, temperature: float, rng: np.random.Generator
) -> np.ndarray:
    """One heat-bath sweep over the assignments: each observation's component drawn from ``softmax(joint / T)``.

    A mixture of independent observations is a factor graph with unary
    factors alone, so a sweep of Gibbs over the assignments is one
    independent draw per observation (issue #901). At ``T = 1`` the draw is
    from the posterior; as ``T -> 0`` the distribution is the indicator of
    the argmax, and the draw is classification EM's hard assignment.

    Parameters
    ----------
    joint : torch.Tensor
        ``log w_k + log p_k(x_i)``, shape ``(n_samples, n_components)``.
    temperature : float
        Positive and finite.
    rng : np.random.Generator
        One uniform per observation, drawn by inverse distribution function.

    Returns
    -------
    np.ndarray
        The component of each observation, shape ``(n_samples,)``.
    """
    probabilities = torch.softmax(joint / temperature, dim=-1).numpy()
    cumulative = np.cumsum(probabilities, axis=1)
    uniform = np.asarray(rng.random(probabilities.shape[0]), dtype=np.float64)
    # The first component whose cumulative probability passes the uniform;
    # the last column is clipped so rounding in the cumulative sum cannot
    # step past it.
    drawn = (cumulative < uniform[:, None]).sum(axis=1)
    return np.asarray(np.minimum(drawn, probabilities.shape[1] - 1), dtype=np.int64)


@dataclass(frozen=True)
class AnnealedAssignments:
    """What :func:`anneal_assignments` visited, and the best of it.

    Parameters
    ----------
    weights : torch.Tensor
        The best state's mixing weights.
    components : EmissionFamily
        The best state's components.
    log_likelihood : float
        The best state's mixture log-likelihood, at temperature one.
    best_step : int
        The sweep that produced it.
    temperatures : tuple[float, ...]
        Each sweep's temperature.
    log_likelihoods : tuple[float, ...]
        The log-likelihood after each sweep.
    path : tuple[EmissionFamily, ...]
        Each sweep's components.
    """

    weights: torch.Tensor
    components: EmissionFamily
    log_likelihood: float
    best_step: int
    temperatures: tuple[float, ...]
    log_likelihoods: tuple[float, ...]
    path: tuple[EmissionFamily, ...]


def anneal_assignments(
    observations: np.ndarray | torch.Tensor,
    weights: torch.Tensor,
    components: EmissionFamily,
    temperatures: Sequence[float],
    rng: np.random.Generator,
    *,
    covariate: np.ndarray | torch.Tensor | None = None,
) -> AnnealedAssignments:
    """Simulated annealing over the component assignments, the components re-estimated at each sweep's.

    Per temperature: one :func:`draw_assignments` sweep, then the family's
    own ``reestimate`` at the one-hot posterior of the draw and the weights
    at its counts, then the log-likelihood at temperature one. The best
    state visited is kept, as
    :class:`~sal.sample.gibbs.Annealed` keeps its best. The
    count-pair likelihood is scored itself; nothing here reads a surrogate.
    ``temperatures`` is a schedule's values, as
    :func:`~sal.sample.schedule.ladder` returns them. Each sweep is recorded
    into the enclosing ``track`` run at its index.

    **A component that draws no observation keeps a soft claim.** A hard
    draw can leave a component with none, and the family's M step has
    nothing to solve on. Such a component's column is its tempered
    responsibility instead of the empty indicator, and its weight is that
    column's mean, so it survives the sweep and is not frozen at zero weight.

    ``covariate`` is
    :func:`~sal.opt.emission_mixture.expectation_maximization`'s: scored at every
    sweep and conditioned on in every M step (issue #933).

    Returns
    -------
    AnnealedAssignments

    Raises
    ------
    ValueError
        If ``temperatures`` is empty or holds a value that is not positive and
        finite, or a component's M step did not converge.
    """
    schedule = [float(t) for t in temperatures]
    if not schedule:
        msg = "an anneal needs at least one temperature"
        raise ValueError(msg)
    for value in schedule:
        if not (math.isfinite(value) and value > 0.0):
            msg = f"a temperature is positive and finite, got {value}"
            raise ValueError(msg)
    values = torch.as_tensor(observations, dtype=torch.float64)
    conditioned = (
        None if covariate is None else torch.as_tensor(covariate, dtype=torch.float64)
    )
    n_components = components.n_states
    tracked = current()
    joint = torch.log(weights) + component_log_density(components, values, conditioned)
    best: tuple[torch.Tensor, EmissionFamily, float, int] | None = None
    trace: list[float] = []
    path: list[EmissionFamily] = []
    for step, temperature in enumerate(schedule):
        drawn = draw_assignments(joint, temperature, rng)
        posterior = torch.nn.functional.one_hot(
            torch.as_tensor(drawn), n_components
        ).to(torch.float64)
        empty = posterior.sum(dim=0) == 0.0
        if bool(empty.any()):
            soft = torch.softmax(joint / temperature, dim=-1)
            posterior[:, empty] = soft[:, empty]
        reestimated = (
            components.reestimate(values, posterior)
            if conditioned is None
            else components.reestimate(values, posterior, conditioned)
        )
        if not reestimated.converged:
            msg = f"a component's M step did not settle at sweep {step}"
            raise ValueError(msg)
        components = reestimated.emissions
        weights = posterior.sum(dim=0) / posterior.sum()
        joint = torch.log(weights) + component_log_density(
            components, values, conditioned
        )
        log_likelihood = float(torch.logsumexp(joint, dim=-1).sum())
        tracked.record(step, log_likelihood=log_likelihood, temperature=temperature)
        trace.append(log_likelihood)
        path.append(components)
        if best is None or log_likelihood > best[2]:
            best = (weights, components, log_likelihood, step)
    assert best is not None
    return AnnealedAssignments(
        weights=best[0],
        components=best[1],
        log_likelihood=best[2],
        best_step=best[3],
        temperatures=tuple(schedule),
        log_likelihoods=tuple(trace),
        path=tuple(path),
    )
