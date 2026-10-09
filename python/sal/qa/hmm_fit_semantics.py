"""Baum-Welch on the count-pair HMM under sal's semantics and a downstream caller's (issue #1412).

A downstream package fits the count-pair HMM (:class:`~sal.emissions.CountPairEmission`,
independent form: a negative-binomial total over an exposure, beta-binomial
successes over a trial count) with its own Baum-Welch. This module fits the
same cell with :func:`~sal.opt.hmm.estimation.baum_welch_family` under each
:class:`Semantics`, from one start, and reports what each reached and what it
cost. ``docs/experiments/039`` carries the numbers, and the downstream
package's own fit on its instances as the reference.

**What sal expresses of the downstream semantics:** one dispersion and one
concentration shared by every state (:func:`tied_m_step`, through the
``m_step`` seam), the transition held fixed (``fit_transition=False``), and a
relative log-likelihood tolerance. :data:`GAPS` lists what it does not.

**The cell** is :mod:`sal.sim.count_hmm_cell`'s; its ``downstream`` variant
mirrors the downstream instances' shape and statistics.

Run as ``python -m sal.qa.hmm_fit_semantics [tier] [variant]``; two threads.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from sal.emissions import (
    BetaBinomialEmission,
    CountPairEmission,
    EmissionFamily,
    NegativeBinomialEmission,
)
from sal.emissions.base import Reestimate
from sal.likelihood.ragged import posteriors
from sal.opt.em import EmConfig
from sal.opt.hmm.estimation import EmFit, baum_welch_family
from sal.ragged import Ragged
from sal.sim.count_hmm_cell import CountHmm

__all__ = [
    "DOWNSTREAM",
    "GAPS",
    "SAL",
    "Fitted",
    "Semantics",
    "fit",
    "missed",
    "start",
    "tied_m_step",
]

#: The downstream caller's starting dispersion ``1 / r`` and concentration.
START_INVERSE_DISPERSION = 0.5
START_CONCENTRATION = 1000.0

#: The seed :func:`start` perturbs the levels with.
START_SEED = 1412


@dataclass(frozen=True)
class Semantics:
    """One reading of "fit the count-pair HMM": what is shared, held and when it stops.

    Parameters
    ----------
    name : str
    tied : bool
        One dispersion and one concentration across states.
    fit_transition : bool
        Whether the M step re-estimates the ``(K, K)`` transition.
    tolerance : float
        The relative log-likelihood change the EM loop stops at.
    max_iterations : int
    """

    name: str
    tied: bool
    fit_transition: bool
    tolerance: float
    max_iterations: int = 500


#: sal's own: a dispersion and a concentration per state, the transition fitted.
SAL = Semantics("sal", tied=False, fit_transition=True, tolerance=1e-10)

#: The downstream caller's, as far as sal expresses it.
DOWNSTREAM = Semantics("downstream", tied=True, fit_transition=False, tolerance=1e-10)

#: What the downstream fit does that ``baum_welch_family`` cannot be asked for.
GAPS: tuple[str, ...] = (
    "an initial distribution held fixed: the M step always re-estimates it",
    "a joint quasi-Newton M step over every emission parameter, posteriors "
    "refreshed every second quasi-Newton step, stopping where its line search fails",
    "a tied count pair: CountPairEmission builds untied channels, so tying "
    "goes through the m_step seam, on the Python solve",
    "a covariate the caller rescales per sequence group from the decode "
    "(a library-size shift): expressible through update, not natively",
    "the phased chain: 2K states under a per-position Kronecker switch; "
    "likelihood.ragged.posteriors takes the switch, baum_welch_family does not",
)


@dataclass(frozen=True)
class Fitted:
    """One fit: what it reached, how it stopped, and where its seconds went."""

    semantics: str
    log_likelihood: float
    missed: int
    iterations: int
    termination: str
    seconds: float
    m_step_seconds: float
    dispersion: np.ndarray
    concentration: np.ndarray

    @property
    def per_iteration(self) -> float:
        """Seconds per EM iteration."""
        return self.seconds / max(self.iterations, 1)


def tied_m_step(
    components: EmissionFamily,
    observations: torch.Tensor,
    posterior: torch.Tensor,
    covariate: torch.Tensor | None,
) -> Reestimate[EmissionFamily]:
    """The count pair's M step with one dispersion and one concentration across states.

    Each channel's tied solve (:class:`~sal.emissions.NegativeBinomialEmission`
    and :class:`~sal.emissions.BetaBinomialEmission` with ``tied=True``) on
    the weights the E step handed over.
    """
    assert isinstance(components, CountPairEmission)
    assert covariate is not None
    values = observations.reshape(-1, 2).to(torch.float64)
    weights = posterior.reshape(-1, components.n_states)
    given = covariate.reshape(-1, 2)
    total, successes = components.total, components.successes
    assert successes is not None
    k = components.n_states
    depth = NegativeBinomialEmission(
        torch.full((k,), float(total.dispersion.mean())), total.mean, tied=True
    ).reestimate(values[:, 0], weights, given[:, :1])
    shared = float(components.concentration.mean())
    rate = components.rate
    allele = BetaBinomialEmission(
        successes.trials, rate * shared, (1.0 - rate) * shared, tied=True
    ).reestimate(values[:, 1], weights, given[:, 1:])
    return Reestimate(
        CountPairEmission(
            depth.components.dispersion,
            depth.components.mean,
            allele.components.alpha,
            allele.components.beta,
            successes.trials,
            joint=False,
        ),
        converged=depth.converged and allele.converged,
        at_boundary=depth.at_boundary or allele.at_boundary,
        iterations=max(depth.iterations, allele.iterations),
        residual=max(depth.residual, allele.residual),
    )


def start(
    cell: CountHmm, k: int, stay: float
) -> tuple[torch.Tensor, torch.Tensor, CountPairEmission]:
    """One start for every semantics: the ``k`` most occupied levels' means and rates, perturbed.

    Each log mean moves by ``N(0, 0.1)`` and each rate's logit by
    ``N(0, 0.5)``, from ``default_rng(START_SEED)``: a start at the truth
    settles in three iterations and separates nothing. Dispersion and concentration at the downstream caller's starting values,
    the initial distribution uniform, and the transition staying with
    probability ``stay``.
    """
    order = np.argsort(-np.bincount(cell.states, minlength=cell.components.n_states))[
        :k
    ]
    truth = cell.components
    rng = np.random.default_rng(START_SEED)
    mean = truth.total.mean.numpy()[order] * np.exp(rng.normal(0.0, 0.1, k))
    logit = np.log(truth.rate.numpy()[order] / (1.0 - truth.rate.numpy()[order]))
    rate = 1.0 / (1.0 + np.exp(-(logit + rng.normal(0.0, 0.5, k))))
    family = CountPairEmission(
        np.full(k, 1.0 / START_INVERSE_DISPERSION),
        mean,
        rate * START_CONCENTRATION,
        (1.0 - rate) * START_CONCENTRATION,
        np.ones(k, dtype=np.int64),
        joint=False,
    )
    transition = np.full((k, k), (1.0 - stay) / (k - 1))
    np.fill_diagonal(transition, stay)
    return (
        torch.full((k,), -float(np.log(k)), dtype=torch.float64),
        torch.as_tensor(np.log(transition)),
        family,
    )


def missed(cell: CountHmm, fitted: EmFit) -> int:
    """Positions whose posterior-argmax state is not the generating level, under the best one-to-one matching."""
    emit = fitted.components.log_density(
        torch.as_tensor(cell.observations), covariate=torch.as_tensor(cell.covariate)
    )
    found = posteriors(
        Ragged(np.ascontiguousarray(emit.numpy(), dtype=np.float64), cell.lengths),
        fitted.log_initial.numpy(),
        fitted.log_transition.numpy(),
    )
    label = np.asarray(found.log_posterior).argmax(axis=1)
    k, levels = fitted.components.n_states, int(cell.states.max()) + 1
    confusion = np.zeros((k, levels), dtype=np.int64)
    np.add.at(confusion, (label, cell.states), 1)
    rows, cols = linear_sum_assignment(-confusion)
    return int(cell.states.size - confusion[rows, cols].sum())


def fit(cell: CountHmm, k: int, semantics: Semantics, stay: float) -> Fitted:
    """``baum_welch_family`` on ``cell`` with ``k`` states under ``semantics``, from :func:`start`."""
    log_initial, log_transition, family = start(cell, k, stay)
    spent = {"m_step": 0.0}

    def timed(components: EmissionFamily, observations: torch.Tensor, posterior: torch.Tensor,
              covariate: torch.Tensor | None) -> Reestimate[Any]:  # fmt: skip
        opened = time.perf_counter()
        step = (
            tied_m_step(components, observations, posterior, covariate)
            if semantics.tied
            else components.reestimate(observations, posterior, covariate=covariate)
        )
        spent["m_step"] += time.perf_counter() - opened
        return step

    opened = time.perf_counter()
    fitted = baum_welch_family(
        Ragged(np.ascontiguousarray(cell.observations.astype(np.int64)), cell.lengths),
        log_initial,
        log_transition,
        family,
        EmConfig(
            max_iterations=semantics.max_iterations, tolerance=semantics.tolerance
        ),
        covariate=Ragged(np.ascontiguousarray(cell.covariate), cell.lengths),
        fit_transition=semantics.fit_transition,
        m_step=timed,
    )
    seconds = time.perf_counter() - opened
    pair = fitted.components
    assert isinstance(pair, CountPairEmission)
    return Fitted(
        semantics.name,
        fitted.log_likelihood,
        missed(cell, fitted),
        fitted.spent,
        fitted.termination.reason.value,
        seconds,
        spent["m_step"],
        1.0 / pair.total.dispersion.numpy(),
        pair.concentration.numpy(),
    )


def main(argv: list[str]) -> None:
    """Fit the fixture under each semantics and print one row per fit."""
    from sal.sim.fixtures import fixture

    tier = argv[0] if argv else "ci"
    params = fixture("count_hmm_reference", tier).params
    if len(argv) > 1:
        params = params.variant(argv[1])
    cell = params.instance()
    limit = int(argv[2]) if len(argv) > 2 else 500
    for semantics in (SAL, DOWNSTREAM):
        row = fit(cell, params.n_fit_states, Semantics(**{**semantics.__dict__, "max_iterations": limit}),
                  params.stay)  # fmt: skip
        print(
            f"{row.semantics:10s} log L {row.log_likelihood:.4f}  missed {row.missed}/{cell.states.size}"
            f"  {row.iterations} it ({row.termination})  {row.seconds:.2f} s"
            f"  M step {row.m_step_seconds:.2f} s  {row.per_iteration * 1e3:.1f} ms/it",
            flush=True,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
