"""Fitting the coupled spatio-sequential model: block ascent, three label solvers, and the annealed start (issue #306).

The block structure of ``eq:joint`` makes maximum likelihood two alternating
problems. Given the labels, each class is a hidden Markov chain over its
members' observations and expectation--maximization fits its emissions, its
initial distribution and the shared self-transition -- the E step is
:func:`~snakes_and_ladders.likelihood.spatio_sequential.class_posteriors`
and the M step is each family's own ``reestimate`` (the M-step identity in the textbook).
Given the parameters, the posterior defines a per-node, per-class external
field ``H`` (the external-field equation of the textbook) and the labels are the ground state of a
Potts model in that field (the label ground-state equation of the textbook): exactly the problem
:func:`~snakes_and_ladders.search.alpha_expansion.alpha_expansion` solves
with a bound, :func:`~snakes_and_ladders.search.alpha_expansion.iterated_conditional_modes`
descends, and a Wolff cluster move in a field samples.

Every block is held to one property: ``log p(x, l | theta)`` with the chains
marginalized does not decrease. The two exact solvers give that by
construction; the annealed Wolff move is a *proposer* here, its best visited
labelling taken only when it improves the joint (open question 1 of #306).
Whether a cluster move escapes what single-site descent freezes into is a
measurement, made past enumeration on a lattice with planted labels.

The annealed start, ``Graph_BurnIn++`` (the textbook's burn-in algorithm), runs the
same blocks while the inverse temperature rises from zero, so the labels are
nearly free while the chains and emissions are fitted to what the data alone
supports, and the spatial prior tightens as the classes separate. Its
seeding, ``Emission_Mixture++``, is k-means++ under the family's own
negative log-density, in :mod:`snakes_and_ladders.opt.mixture`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from itertools import permutations

import numpy as np
import torch

from snakes_and_ladders.emissions import (
    CategoricalEmission,
    EmissionFamily,
    GaussianEmission,
)
from snakes_and_ladders.likelihood.forward_backward import sample_path
from snakes_and_ladders.likelihood.spatio_sequential import (
    ClassPosteriors,
    class_log_density,
    class_posteriors,
    external_field,
    labelled_log_likelihood,
)
from snakes_and_ladders.opt.mixture import emission_mixture_plus_plus
from snakes_and_ladders.opt.schedule import Schedule
from snakes_and_ladders.search.alpha_expansion import (
    alpha_expansion,
    energy,
)
from snakes_and_ladders.sim.graph import PottsGraph
from snakes_and_ladders.sim.spatio_sequential import SpatioSequentialParams


class LabelSolver(StrEnum):
    """How the label block is solved."""

    ALPHA_EXPANSION = "alpha_expansion"
    """Exact two-state expansions cycled to a local minimum within the bound."""

    ICM = "icm"
    """Single-site descent to a local minimum."""

    WOLFF = "wolff"
    """Cluster moves in the field on an annealing schedule, best labelling kept."""


@dataclass(frozen=True)
class SpatioSequentialFit:
    """What block ascent returns.

    Parameters
    ----------
    params : SpatioSequentialParams
        The fitted truth: emissions, initial distributions and self-transition.
    labels : np.ndarray
        The fitted labels, shape ``(n_nodes,)``.
    log_likelihoods : np.ndarray
        ``log p(x, l | theta)`` after every block, shape ``(2 * n_blocks + 1,)``:
        the start, then after each EM half and each label half. Non-decreasing.
    field : np.ndarray
        The external field at the fitted labels and parameters, ``(n_nodes, M)``.
    """

    params: SpatioSequentialParams
    labels: np.ndarray
    log_likelihoods: np.ndarray
    field: np.ndarray


def m_step(
    params: SpatioSequentialParams,
    observations: np.ndarray,
    labels: np.ndarray,
    posteriors: ClassPosteriors,
) -> SpatioSequentialParams:
    """Re-estimate every class's emissions, ``Pi_m`` and the shared ``t`` from the E step.

    A class with no members keeps its emissions: there is nothing to
    re-estimate them from, and its chain posterior is its prior.
    """
    labels = np.asarray(labels, dtype=np.int64)
    emissions: list[EmissionFamily] = []
    for m, family in enumerate(params.emissions):
        members = np.flatnonzero(labels == m)
        if members.size == 0:
            emissions.append(family)
            continue
        block = torch.as_tensor(
            np.moveaxis(observations[:, members], 1, 0), dtype=family.observation_dtype
        )  # (n_m, S), plus any channel axes the family's observation carries
        weights = torch.as_tensor(posteriors.posterior[m])[None].expand(
            members.size, -1, -1
        )  # (n_m, S, K)
        emissions.append(family.reestimate(block, weights).emissions)
    initial = np.maximum(posteriors.posterior[:, 0, :], 1e-12)
    initial = initial / initial.sum(axis=1, keepdims=True)
    if params.n_positions > 1:
        stays = sum(
            float(np.trace(posteriors.pairwise[m, s]))
            for m in range(params.n_classes)
            for s in range(params.n_positions - 1)
        )
        total = params.n_classes * (params.n_positions - 1)
        self_transition = min(max(stays / total, 1e-6), 1 - 1e-6)
    else:
        self_transition = params.self_transition
    return replace(
        params,
        initial=initial,
        self_transition=self_transition,
        emissions=tuple(emissions),
    )


def _wolff_update(
    labels: np.ndarray,
    graph: PottsGraph,
    field: np.ndarray,
    beta: float,
    rng: np.random.Generator,
) -> None:
    """One cluster grown on ``1 - exp(-beta J)`` and recoloured on the field alone (the textbook's Wolff-in-a-field algorithm).

    ``field`` is ``-H`` per node and class, so the accept step is Metropolis
    on ``beta * sum_C (H[n, new] - H[n, old])``.
    """
    n_nodes = labels.shape[0]
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n_nodes)]
    for (first, second), coupling in graph.weighted_edges():
        adjacency[first].append((second, coupling))
        adjacency[second].append((first, coupling))
    root = int(rng.integers(n_nodes))
    colour = int(labels[root])
    members = [root]
    inside = np.zeros(n_nodes, dtype=bool)
    inside[root] = True
    frontier = [root]
    while frontier:
        node = frontier.pop()
        for neighbour, coupling in adjacency[node]:
            if inside[neighbour] or labels[neighbour] != colour:
                continue
            if rng.random() < 1.0 - np.exp(-beta * coupling):
                inside[neighbour] = True
                members.append(neighbour)
                frontier.append(neighbour)
    proposed = int(rng.integers(field.shape[1]))
    if proposed == colour:
        return
    cluster = np.array(members, dtype=np.int64)
    difference = beta * float((field[cluster, proposed] - field[cluster, colour]).sum())
    if difference >= 0.0 or rng.random() < np.exp(difference):
        labels[cluster] = proposed


def label_step(
    params: SpatioSequentialParams,
    observations: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    solver: LabelSolver,
    *,
    field: np.ndarray | None = None,
    wolff_schedule: Schedule | None = None,
    wolff_moves_per_step: int = 4,
) -> np.ndarray:
    """One label block: the ground state of the Potts model in the field, by ``solver``.

    The energy minimized is ``-beta sum J delta(l, l') + sum_n H[n, l_n]``,
    so the solvers see couplings ``beta * J`` and a per-node field ``-H``.
    The two exact solvers start from ``labels``; Wolff anneals from them on
    ``wolff_schedule`` (temperatures multiply ``beta``'s inverse) and returns
    the lowest-energy labelling it visited.
    """
    labels = np.asarray(labels, dtype=np.int64).copy()
    if field is None:
        field = external_field(params, observations, labels)
    graph = params.scaled_graph()
    potential = -field
    if solver is LabelSolver.ALPHA_EXPANSION:
        return np.asarray(
            alpha_expansion(graph, potential, params.n_classes, start=labels).labelling
        )
    if solver is LabelSolver.ICM:
        current = labels.copy()
        best = energy(graph, potential, current)
        for _ in range(200):
            moved = False
            for node in rng.permutation(graph.n_nodes):
                for label in range(params.n_classes):
                    if label == current[node]:
                        continue
                    trial = current.copy()
                    trial[node] = label
                    value = energy(graph, potential, trial)
                    if value < best - 1e-12:
                        best, current, moved = value, trial, True
            if not moved:
                break
        return current
    if wolff_schedule is None:
        msg = "the Wolff solver needs a schedule"
        raise ValueError(msg)
    best_labels = labels.copy()
    best_value = energy(graph, potential, labels)
    for step in range(wolff_schedule.n_steps):
        temperature = wolff_schedule(step)
        for _ in range(wolff_moves_per_step):
            _wolff_update(
                labels, params.graph, potential, params.beta / temperature, rng
            )
        value = energy(graph, potential, labels)
        if value < best_value:
            best_value, best_labels = value, labels.copy()
    return best_labels


def fit_spatio_sequential(
    params: SpatioSequentialParams,
    observations: np.ndarray,
    rng: np.random.Generator,
    *,
    solver: LabelSolver = LabelSolver.ALPHA_EXPANSION,
    n_blocks: int = 10,
    labels: np.ndarray | None = None,
    fit_parameters: bool = True,
    wolff_schedule: Schedule | None = None,
) -> SpatioSequentialFit:
    """Block-coordinate ascent on ``log p(x, l | theta)``.

    Parameters
    ----------
    params : SpatioSequentialParams
        The starting parameters; the fitted ones replace the emissions,
        ``initial`` and ``self_transition``.
    observations : np.ndarray
        Shape ``(S, n_nodes)``.
    rng : np.random.Generator
        Draws the starting labels when none are given, and the Wolff moves.
    solver : LabelSolver
        The label block.
    n_blocks : int
        EM-then-label rounds; at least one.
    labels : np.ndarray | None
        Starting labels; ``None`` draws them uniformly.
    fit_parameters : bool
        ``False`` holds ``params`` fixed and runs the label block alone, the
        setting in which the label step is pinned against enumeration.
    wolff_schedule : Schedule | None
        Required by the Wolff solver.

    Raises
    ------
    ValueError
        If ``n_blocks < 1``, or a re-estimated family did not converge.
    """
    if n_blocks < 1:
        msg = f"at least one block, got {n_blocks}"
        raise ValueError(msg)
    n_nodes = params.graph.n_nodes
    current = (
        rng.integers(0, params.n_classes, size=n_nodes)
        if labels is None
        else np.asarray(labels, dtype=np.int64).copy()
    )
    values = [labelled_log_likelihood(params, observations, current)]
    for _ in range(n_blocks):
        posteriors = class_posteriors(params, observations, current)
        if fit_parameters:
            params = m_step(params, observations, current, posteriors)
            posteriors = class_posteriors(params, observations, current)
        values.append(labelled_log_likelihood(params, observations, current))
        field = external_field(params, observations, current, posteriors.posterior)
        proposed = label_step(
            params,
            observations,
            current,
            rng,
            solver,
            field=field,
            wolff_schedule=wolff_schedule,
        )
        candidate = labelled_log_likelihood(params, observations, proposed)
        if candidate >= values[-1]:
            current = proposed
            values.append(candidate)
        else:
            values.append(values[-1])
    field = external_field(params, observations, current)
    return SpatioSequentialFit(params, current, np.array(values), field)


def label_accuracy(fitted: np.ndarray, planted: np.ndarray, n_classes: int) -> float:
    """The fraction of nodes labelled as planted, up to the best permutation of classes."""
    fitted = np.asarray(fitted)
    planted = np.asarray(planted)
    best = 0.0
    for order in permutations(range(n_classes)):
        mapping = np.array(order)
        best = max(best, float((mapping[fitted] == planted).mean()))
    return best


def seed_emissions(
    params: SpatioSequentialParams, observations: np.ndarray, rng: np.random.Generator
) -> SpatioSequentialParams:
    """``Emission_Mixture++`` for every class: seeds under the family's own negative log-density.

    A categorical family is seeded from symbols (a smoothed one-hot row per
    seed); a Gaussian one from values (the seed as the mean, the pooled scale).
    Other families keep their parameters, which the notebook records.
    """
    emissions: list[EmissionFamily] = []
    flat = observations.reshape(-1)
    for family in params.emissions:
        if isinstance(family, CategoricalEmission):
            n_symbols = int(family.matrix.shape[1])

            def score(
                seed: float, values: np.ndarray, n_symbols: int = n_symbols
            ) -> np.ndarray:
                row = np.full(n_symbols, 0.1 / (n_symbols - 1))
                row[int(seed)] = 0.9
                return -np.log(row[values.astype(np.int64)])

            seeds = emission_mixture_plus_plus(flat, params.n_states, score, rng)
            matrix = np.full((params.n_states, n_symbols), 0.1 / (n_symbols - 1))
            for k, seed in enumerate(seeds):
                matrix[k, int(seed)] = 0.9
            emissions.append(CategoricalEmission(matrix))
        elif isinstance(family, GaussianEmission):
            pooled = float(np.std(flat)) or 1.0

            def gaussian_score(
                seed: float, values: np.ndarray, pooled: float = pooled
            ) -> np.ndarray:
                return 0.5 * ((values - seed) / pooled) ** 2

            seeds = emission_mixture_plus_plus(
                flat, params.n_states, gaussian_score, rng
            )
            emissions.append(
                GaussianEmission(
                    np.sort(np.asarray(seeds, dtype=float)),
                    np.full(params.n_states, pooled),
                    float(family.variance_floor),
                )
            )
        else:
            emissions.append(family)
    return replace(params, emissions=tuple(emissions))


def graph_burn_in(
    params: SpatioSequentialParams,
    observations: np.ndarray,
    rng: np.random.Generator,
    schedule: Schedule,
    *,
    wolff_moves_per_step: int = 4,
) -> SpatioSequentialFit:
    """``Graph_BurnIn++`` (the textbook's burn-in algorithm): the blocks while the inverse temperature rises.

    ``schedule`` gives a *temperature* per step; the inverse temperature used
    is ``params.beta / T``, so a schedule ending at ``T = 1`` ends at the
    model's own ``beta``. Each step: one Wolff pass on the labels in the
    current field, a block Gibbs draw of every class's chain, an E step and an
    M step, then the field. Emissions are seeded once by ``Emission_Mixture++``
    and labels uniformly; the returned fit is the state at the schedule's end,
    to be polished by :func:`fit_spatio_sequential`.
    """
    params = seed_emissions(params, observations, rng)
    labels = rng.integers(0, params.n_classes, size=params.graph.n_nodes)
    field = external_field(params, observations, labels)
    values = [labelled_log_likelihood(params, observations, labels)]
    for step in range(schedule.n_steps):
        beta = params.beta / schedule(step)
        for _ in range(wolff_moves_per_step):
            _wolff_update(labels, params.graph, -field, beta, rng)
        density = class_log_density(params, observations, labels)
        for m in range(
            params.n_classes
        ):  # block Gibbs over the chains, recorded nowhere
            sample_path(
                density[m], np.log(params.initial[m]), np.log(params.transition), rng
            )
        posteriors = class_posteriors(params, observations, labels)
        params = m_step(params, observations, labels, posteriors)
        posteriors = class_posteriors(params, observations, labels)
        field = external_field(params, observations, labels, posteriors.posterior)
        values.append(labelled_log_likelihood(params, observations, labels))
    return SpatioSequentialFit(params, labels, np.array(values), field)
