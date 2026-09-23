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
what it was, then full EM to its tolerance
(:func:`~snakes_and_ladders.opt.emission_mixture.expectation_maximization`).

**The likelihood never falls, by construction:** a move whose full EM does
not end above the fit it started from is refused and the fit kept. What the
moves buy is recorded per candidate, accepted or refused, and each is
recorded into the enclosing ``track`` run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from snakes_and_ladders.emissions import EmissionFamily
from snakes_and_ladders.opt.emission_mixture import (
    ComponentsAt,
    EmissionMixtureFit,
    expectation_maximization,
)
from snakes_and_ladders.opt.mixture import responsibilities
from snakes_and_ladders.track import current


def merge_criterion(posterior: torch.Tensor) -> torch.Tensor:
    """The cosine of every pair of responsibility columns, shape ``(K, K)``, ``-inf`` on the diagonal.

    Parameters
    ----------
    posterior : torch.Tensor
        Responsibilities, shape ``(n_samples, K)``.

    Returns
    -------
    torch.Tensor
        ``J_merge(i, j) = r_i . r_j / (|r_i| |r_j|)``; a component is never
        merged with itself.
    """
    norms = torch.linalg.vector_norm(posterior, dim=0)
    cosine = (posterior.T @ posterior) / torch.outer(norms, norms)
    cosine.fill_diagonal_(-float("inf"))
    return cosine


def split_criterion(
    posterior: torch.Tensor, log_densities: torch.Tensor
) -> torch.Tensor:
    """The local Kullback--Leibler divergence of each component from the data it owns, shape ``(K,)``.

    ``f_k(x_i) = r_ik / sum_i r_ik`` is the data component ``k`` owns, as a
    distribution over the observations, and ``J_split(k) = sum_i f_k(x_i)
    log(f_k(x_i) / p_k(x_i))``. For a discrete family ``p_k(x_i)`` is a
    probability and the divergence is the one Ueda et al. state for a
    density; two observations with the same value are counted apart, which
    raises every component's divergence alike.

    Parameters
    ----------
    posterior : torch.Tensor
        Responsibilities, shape ``(n_samples, K)``.
    log_densities : torch.Tensor
        ``log p_k(x_i)``, shape ``(n_samples, K)``.

    Returns
    -------
    torch.Tensor
    """
    owned = posterior / posterior.sum(dim=0, keepdim=True)
    terms = torch.where(
        owned > 0.0, owned * (torch.log(owned) - log_densities), torch.zeros_like(owned)
    )
    return terms.sum(dim=0)


def candidate_moves(
    posterior: torch.Tensor, log_densities: torch.Tensor, candidates: int
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


def _split_seeds(values: torch.Tensor, owned: torch.Tensor) -> np.ndarray:
    """The two observations a split is seeded at: the most owned, and the most owned far from it.

    The first is the observation of highest responsibility; the second
    maximizes the responsibility times the squared distance to the first, a
    deterministic k-means++ step within the component. Two observations of
    highest responsibility sit together at the component's mode and seed two
    copies of one component, which partial EM cannot then tell apart.
    """
    first = int(torch.argmax(owned))
    distance = ((values - values[first]) ** 2).sum(dim=-1)
    second = int(torch.argmax(owned * distance))
    return values[[first, second]].numpy()


def _moved(
    fit: EmissionMixtureFit,
    values: torch.Tensor,
    at: ComponentsAt,
    move: tuple[int, int, int],
    partial_iterations: int,
) -> tuple[torch.Tensor, EmissionFamily]:
    """The weights and components a move and its partial EM hand to full EM.

    Raises
    ------
    ValueError
        If a component's M step did not converge.
    """
    i, j, k = move
    posterior = fit.responsibilities.clone()
    pair = values.reshape(values.shape[0], -1)
    seeded = at(_split_seeds(pair, posterior[:, k]))
    halves = responsibilities(
        values, torch.log(torch.full((2,), 0.5, dtype=torch.float64)), seeded
    )
    moved = posterior.clone()
    moved[:, i] = posterior[:, i] + posterior[:, j]
    moved[:, j] = posterior[:, k] * halves[:, 0]
    moved[:, k] = posterior[:, k] * halves[:, 1]
    affected = [i, j, k]
    # The three components' total claim on each observation, which partial
    # EM redistributes among them and never changes.
    mass = posterior[:, affected].sum(dim=1, keepdim=True)
    family = fit.components
    for iteration in range(partial_iterations + 1):
        reestimated = family.reestimate(values, moved)
        if not reestimated.converged:
            msg = (
                f"a component's M step did not settle in the partial EM of "
                f"move {move}, iteration {iteration}"
            )
            raise ValueError(msg)
        family = reestimated.emissions
        weights = moved.mean(dim=0)
        if iteration == partial_iterations:
            break
        joint = torch.log(weights) + family.log_density(values)
        local = joint[:, affected]
        moved = moved.clone()
        moved[:, affected] = mass * torch.softmax(local, dim=-1)
    return weights, family


def split_and_merge(
    fit: EmissionMixtureFit,
    observations: np.ndarray | torch.Tensor,
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
    observations : np.ndarray | torch.Tensor
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
        :func:`~snakes_and_ladders.opt.emission_mixture.expectation_maximization`
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
    values = torch.as_tensor(observations, dtype=torch.float64)
    tracked = current()
    steps: list[SplitMergeStep] = []
    improved = True
    while improved:
        improved = False
        log_densities = fit.components.log_density(values)
        for move in candidate_moves(fit.responsibilities, log_densities, candidates):
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
