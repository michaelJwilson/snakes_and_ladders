"""Split-and-merge EM for a mixture of emissions (Ueda, Nakano, Ghahramani & Hinton, 2000).

EM converges to a fixed point, and on a mixture a common one is wrong in a
way EM cannot repair: one component sitting on two generating ones while two
share a third. Moving one component across the space between them lowers the
likelihood on the way, so no EM step takes it. A split-and-merge move takes
it in one step: merge two components, split a third, run EM, and keep the
result only if the log-likelihood rose (issue #904).

**Candidates are ranked, not searched.** A merge pair is ranked by the cosine
of the two components' responsibility vectors: two components claiming the
same observations. A split candidate is ranked by the local Kullback--Leibler
divergence between the observations a component owns, weighted by its
responsibilities, and the component's own density there: a component whose
density disagrees with the data it holds. Each merge pair in order is paired
with the best-ranked split candidate outside it, and the first
``candidates`` triples are tried in that order.

**A move is a re-weighting of the responsibilities, so the family's own M
step realizes it.** The merged component takes the sum of the two columns;
the split component's column is divided between two slots by a
two-component seeding at two of its observations; every other column is
kept. One call of the family's ``reestimate`` then builds all three, which
is why nothing here reaches inside a family. Partial EM follows on the three
affected components, their total responsibility on each observation held at
what it was
(:func:`~sal.opt.emission_mixture.partial_expectation_maximization`),
then full EM to its tolerance
(:func:`~sal.opt.emission_mixture.expectation_maximization`).

**The criteria and the move are arithmetic on arrays; the EM is
``opt.emission_mixture``'s** (issue #1011). No derivative is taken here, so
the module holds NumPy arrays and imports no torch: the responsibilities and
log-densities are read out of the fit once per round, and the two E and M
steps the family's torch API requires run behind
:func:`~sal.opt.emission_mixture.log_densities`,
:func:`~sal.opt.emission_mixture.responsibilities_at` and the
partial EM.

**The likelihood never falls, by construction:** a move whose full EM does
not end above the fit it started from is refused and the fit kept. What the
moves buy is recorded per candidate, accepted or refused, and each is
recorded into the enclosing ``track`` run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sal.emissions import EmissionFamily
from sal.opt.emission_mixture import (
    ComponentsAt,
    EmissionMixtureFit,
    expectation_maximization,
    log_densities,
    partial_expectation_maximization,
    responsibilities_at,
)
from sal.track import current


def merge_criterion(posterior: np.ndarray) -> np.ndarray:
    """The cosine of every pair of responsibility columns, shape ``(K, K)``, ``-inf`` on the diagonal.

    Parameters
    ----------
    posterior : np.ndarray
        Responsibilities, shape ``(n_samples, K)``.

    Returns
    -------
    np.ndarray
        ``J_merge(i, j) = r_i . r_j / (|r_i| |r_j|)``; a component is never
        merged with itself. A column of zeros has no direction and scores
        ``nan``.
    """
    posterior = np.asarray(posterior, dtype=np.float64)
    norms = np.linalg.norm(posterior, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine: np.ndarray = (posterior.T @ posterior) / np.outer(norms, norms)
    np.fill_diagonal(cosine, -np.inf)
    return cosine


def split_criterion(posterior: np.ndarray, log_densities: np.ndarray) -> np.ndarray:
    """The local Kullback--Leibler divergence of each component from the data it owns, shape ``(K,)``.

    ``f_k(x_i) = r_ik / sum_i r_ik`` is the data component ``k`` owns, as a
    distribution over the observations, and ``J_split(k) = sum_i f_k(x_i)
    log(f_k(x_i) / p_k(x_i))``. For a discrete family ``p_k(x_i)`` is a
    probability and the divergence is the one Ueda et al. state for a
    density; two observations with the same value are counted apart, which
    raises every component's divergence alike.

    Parameters
    ----------
    posterior : np.ndarray
        Responsibilities, shape ``(n_samples, K)``.
    log_densities : np.ndarray
        ``log p_k(x_i)``, shape ``(n_samples, K)``.

    Returns
    -------
    np.ndarray
    """
    posterior = np.asarray(posterior, dtype=np.float64)
    owned = posterior / posterior.sum(axis=0, keepdims=True)
    # An observation a component does not own contributes nothing, and its
    # ``0 * log 0`` is never read.
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(owned > 0.0, owned * (np.log(owned) - log_densities), 0.0)
    return terms.sum(axis=0)


def candidate_moves(
    posterior: np.ndarray, log_densities: np.ndarray, candidates: int
) -> list[tuple[int, int, int]]:
    """The first ``candidates`` ``(i, j, k)`` triples: merge ``i`` and ``j``, split ``k``.

    Merge pairs in decreasing :func:`merge_criterion`, each with the
    highest-ranked :func:`split_criterion` component outside the pair; ties
    fall to the lower index, so the order is a function of the two criteria.

    Returns
    -------
    list[tuple[int, int, int]]
        With ``i < j`` and ``k`` neither; empty below three components.
    """
    k = posterior.shape[1]
    if k < 3:
        return []
    merge = merge_criterion(posterior)
    split = split_criterion(posterior, log_densities)
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    pairs.sort(key=lambda pair: (-float(merge[pair]), pair))
    order = sorted(range(k), key=lambda c: (-float(split[c]), c))
    moves = []
    for i, j in pairs[:candidates]:
        chosen = next(c for c in order if c not in (i, j))
        moves.append((i, j, chosen))
    return moves


@dataclass(frozen=True)
class SplitMergeStep:
    """One candidate move, tried from the fit it names.

    Parameters
    ----------
    merge : tuple[int, int]
        The components merged, into the first slot.
    split : int
        The component split, into its own slot and the second merged one's.
    before : float
        The log-likelihood of the fit the move started from.
    after : float
        The log-likelihood full EM reached after the move.
    accepted : bool
        Whether ``after`` is above ``before``, so the move's fit was kept.
    """

    merge: tuple[int, int]
    split: int
    before: float
    after: float
    accepted: bool


@dataclass(frozen=True)
class SplitMerge:
    """What split-and-merge EM ended on, and every move it tried.

    Parameters
    ----------
    fit : EmissionMixtureFit
        The last accepted fit, or the one handed in if no move was accepted.
    steps : tuple[SplitMergeStep, ...]
        Every candidate tried, in order.
    """

    fit: EmissionMixtureFit
    steps: tuple[SplitMergeStep, ...]

    @property
    def accepted(self) -> int:
        """Moves kept."""
        return sum(step.accepted for step in self.steps)


def _split_seeds(values: np.ndarray, owned: np.ndarray) -> np.ndarray:
    """The two observations a split is seeded at: the most owned, and the most owned far from it.

    The first is the observation of highest responsibility; the second
    maximizes the responsibility times the squared distance to the first, a
    deterministic k-means++ step within the component. Two observations of
    highest responsibility sit together at the component's mode and seed two
    copies of one component, which partial EM cannot then tell apart.
    """
    first = int(np.argmax(owned))
    distance = ((values - values[first]) ** 2).sum(axis=-1)
    second = int(np.argmax(owned * distance))
    return values[[first, second]]


def _moved(
    fit: EmissionMixtureFit,
    values: np.ndarray,
    at: ComponentsAt,
    move: tuple[int, int, int],
    partial_iterations: int,
) -> tuple[np.ndarray, EmissionFamily]:
    """The weights and components a move and its partial EM hand to full EM.

    Raises
    ------
    ValueError
        If a component's M step did not converge.
    """
    i, j, k = move
    posterior = fit.responsibilities.detach().numpy()
    pair = values.reshape(values.shape[0], -1)
    halves = responsibilities_at(
        values, np.full(2, 0.5), at(_split_seeds(pair, posterior[:, k]))
    )
    moved = posterior.copy()
    moved[:, i] = posterior[:, i] + posterior[:, j]
    moved[:, j] = posterior[:, k] * halves[:, 0]
    moved[:, k] = posterior[:, k] * halves[:, 1]
    affected = [i, j, k]
    # The three components' total claim on each observation, which partial
    # EM redistributes among them and never changes.
    mass = posterior[:, affected].sum(axis=1, keepdims=True)
    try:
        return partial_expectation_maximization(
            values, moved, fit.components, affected, mass, partial_iterations
        )
    except ValueError as error:
        msg = f"{error}, in move {move}"
        raise ValueError(msg) from error


def split_and_merge(
    fit: EmissionMixtureFit,
    observations: np.ndarray,
    at: ComponentsAt,
    *,
    candidates: int,
    partial_iterations: int = 3,
    max_iterations: int = 200,
    tolerance: float = 1e-10,
) -> SplitMerge:
    """Split-and-merge moves from a converged EM fit until none of the top ``candidates`` raises the likelihood.

    From the current fit, :func:`candidate_moves` ranks the moves; each is
    realized, polished by ``partial_iterations`` steps of partial EM on its
    three components and by full EM to ``tolerance``, and kept if its
    log-likelihood is above the current fit's. After a kept move the
    ranking starts again from the new fit; a round in which none of the
    ``candidates`` is kept ends the search. Every step is recorded into the
    enclosing ``track`` run at its index: the log-likelihood reached and
    whether the move was kept.

    Parameters
    ----------
    fit : EmissionMixtureFit
        A converged fit, whose ``responsibilities`` are the posterior its
        last M step consumed.
    observations : np.ndarray
        The observations ``fit`` was fitted to.
    at : ComponentsAt
        The seam a split seeds its two halves by.
    candidates : int
        Moves tried per round, at least 1.
    partial_iterations : int
        Partial EM steps on the three affected components, after the M step
        that realizes the move.
    max_iterations, tolerance
        Full EM's, as
        :func:`~sal.opt.emission_mixture.expectation_maximization`
        takes them.

    Returns
    -------
    SplitMerge

    Raises
    ------
    ValueError
        If ``candidates`` is below 1 or ``partial_iterations`` negative, or a
        component's M step did not converge.
    """
    if candidates < 1 or partial_iterations < 0:
        msg = (
            f"candidates must be at least 1 and partial_iterations non-negative, "
            f"got {candidates} and {partial_iterations}"
        )
        raise ValueError(msg)
    values = np.asarray(observations, dtype=np.float64)
    tracked = current()
    steps: list[SplitMergeStep] = []
    improved = True
    while improved:
        improved = False
        posterior = fit.responsibilities.detach().numpy()
        scored = log_densities(values, fit.components)
        for move in candidate_moves(posterior, scored, candidates):
            weights, family = _moved(fit, values, at, move, partial_iterations)
            full = expectation_maximization(
                values,
                weights,
                family,
                max_iterations=max_iterations,
                tolerance=tolerance,
            )
            accepted = full.log_likelihood > fit.log_likelihood
            steps.append(
                SplitMergeStep(
                    merge=(move[0], move[1]),
                    split=move[2],
                    before=fit.log_likelihood,
                    after=full.log_likelihood,
                    accepted=accepted,
                )
            )
            tracked.record(
                len(steps) - 1,
                log_likelihood=full.log_likelihood,
                accepted=float(accepted),
            )
            if accepted:
                fit = full
                improved = True
                break
    return SplitMerge(fit=fit, steps=tuple(steps))
