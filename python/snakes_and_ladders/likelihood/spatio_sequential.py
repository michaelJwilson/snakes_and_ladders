"""Exact quantities for the coupled spatio-sequential model, by enumeration.

The oracle the rest of issue #290 is held to: the marginal likelihood, the
label posterior, the per-class state posterior, and the state posterior given
a labelling, each a sum over every joint assignment of ``eq:joint``.
Deliberately exponential (``likelihood/CLAUDE.md``): it is an oracle because
it shares no recursion with what it referees.

It is itself pinned two ways. Its unnormalized log-density is the factor
graph's :meth:`~snakes_and_ladders.sim.factor_graph.FactorGraph.log_density`,
assignment by assignment; and its evidence equals a second route that shares
no code with it -- given the labels the chains decouple, so
``p(x) = sum_l p(l) prod_m p(x_{.,l=m} | chain m)`` with each inner term from
the forward recursion of :mod:`snakes_and_ladders.opt.hmm`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch

from snakes_and_ladders.backend import Backend
from snakes_and_ladders.enumeration import (
    configurations,
    refuse_oversized,
)
from snakes_and_ladders.likelihood.forward_backward import forward_backward
from snakes_and_ladders.numerics import logsumexp
from snakes_and_ladders.opt.hmm import forward_log_likelihood_from_density
from snakes_and_ladders.sim.spatio_sequential import (
    SpatioSequentialParams,
    gated_log_density,
)


@dataclass(frozen=True)
class ExactSpatioSequential:
    """What enumeration returns.

    Parameters
    ----------
    log_evidence : float
        ``log p(x)``, summed over every labelling and every joint chain path.
    label_posterior : np.ndarray
        ``p(l_n = m | x)``, shape ``(n_nodes, M)``.
    state_posterior : np.ndarray
        ``p(k_{s,m} | x)``, shape ``(M, S, K)``.
    log_prior_normalizer : float
        ``log Z_Potts`` of the Potts term of ``eq:joint``, the constant the factor
        graph's log-density omits.
    """

    log_evidence: float
    label_posterior: np.ndarray
    state_posterior: np.ndarray
    log_prior_normalizer: float


def _labellings(params: SpatioSequentialParams) -> np.ndarray:
    # `_log_joint` refuses the product of the two factors first, which is
    # what a caller pays for; the cap here reaches only the callers that
    # enumerate one factor alone.
    return configurations(
        params.n_classes,
        params.graph.n_nodes,
        what=f"{params.n_classes}**{params.graph.n_nodes} labellings",
    )


def _paths(params: SpatioSequentialParams) -> np.ndarray:
    return configurations(
        params.n_states,
        params.n_positions,
        what=f"{params.n_states}**{params.n_positions} paths",
    )


def log_prior(params: SpatioSequentialParams, labellings: np.ndarray) -> np.ndarray:
    """Unnormalized ``log p(l)``, the Potts term of ``eq:joint``, per labelling, shape ``(n_labellings,)``."""
    total = np.zeros(labellings.shape[0])
    for (first, second), coupling in params.graph.weighted_edges():
        total += (
            params.beta * coupling * (labellings[:, first] == labellings[:, second])
        )
    return total


def log_chain(params: SpatioSequentialParams, m: int, paths: np.ndarray) -> np.ndarray:
    """``log p(k_{.,m})`` per path, written out term by term, shape ``(n_paths,)``."""
    log_transition = np.log(params.transition)
    total = np.log(params.initial[m])[paths[:, 0]]
    for s in range(1, params.n_positions):
        total = total + log_transition[paths[:, s - 1], paths[:, s]]
    return np.asarray(total)


def _log_joint(
    params: SpatioSequentialParams, observations: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``log p(x, l, k)`` up to ``log Z_Potts`` over every assignment.

    Returns the labellings ``(L, n_nodes)``, the paths ``(P, S)`` and the table
    of shape ``(L, P, ..., P)`` with one path axis per class, in that order.
    """
    # The joint space is refused before either factor is built, so the
    # refusal names the product a caller pays for rather than one factor.
    n_labellings = params.n_classes**params.graph.n_nodes
    n_paths = params.n_states**params.n_positions
    refuse_oversized(
        n_labellings * n_paths**params.n_classes,
        what=(
            f"{params.n_classes}**{params.graph.n_nodes} labellings times "
            f"{params.n_states}**{params.n_positions} paths per class for "
            f"{params.n_classes} classes"
        ),
    )
    labellings = _labellings(params)
    paths = _paths(params)
    gated = gated_log_density(params, observations)  # (n_nodes, S, M, K)
    # emission[n, m, p]: what node n contributes if it belongs to class m and
    # class m's chain follows path p.
    emission = np.zeros((params.graph.n_nodes, params.n_classes, n_paths))
    for n in range(params.graph.n_nodes):
        for m in range(params.n_classes):
            emission[n, m] = gated[n, np.arange(params.n_positions), m, :][
                np.arange(params.n_positions), paths
            ].sum(axis=1)
    table = np.empty((n_labellings,) + (n_paths,) * params.n_classes)
    prior = log_prior(params, labellings)
    chain = [log_chain(params, m, paths) for m in range(params.n_classes)]
    for index, labelling in enumerate(labellings):
        per_class = []
        for m in range(params.n_classes):
            members = np.flatnonzero(labelling == m)
            per_class.append(chain[m] + emission[members, m, :].sum(axis=0))
        total = np.array(prior[index])
        for m in range(params.n_classes):
            shape = [1] * params.n_classes
            shape[m] = n_paths
            total = total + per_class[m].reshape(shape)
        table[index] = total
    return labellings, paths, table


def log_joint_at(
    params: SpatioSequentialParams,
    observations: np.ndarray,
    labels: np.ndarray,
    states: np.ndarray,
) -> float:
    """``eq:joint`` at one assignment, up to ``log Z_Potts``, written out term by term."""
    labellings = np.asarray(labels, dtype=np.int64)[None, :]
    total = float(log_prior(params, labellings)[0])
    gated = gated_log_density(params, observations)
    for m in range(params.n_classes):
        total += float(log_chain(params, m, np.asarray(states[m])[None, :])[0])
        for n in np.flatnonzero(labels == m):
            for s in range(params.n_positions):
                total += float(gated[n, s, m, states[m, s]])
    return total


def enumerate_spatio_sequential(
    params: SpatioSequentialParams, observations: np.ndarray
) -> ExactSpatioSequential:
    """Sum ``eq:joint`` over every assignment.

    Raises
    ------
    ValueError
        Above :data:`~snakes_and_ladders.enumeration.MAX_ENUMERABLE_CONFIGURATIONS`
        joint assignments.
    """
    labellings, paths, table = _log_joint(params, observations)
    flat = table.reshape(1, -1)
    log_z_prior = float(logsumexp(log_prior(params, labellings), axis=0))
    log_total = float(logsumexp(flat, axis=1)[0])
    weights = np.exp(table - log_total)  # posterior over (l, k_1, ..., k_M)

    label_posterior = np.zeros((params.graph.n_nodes, params.n_classes))
    per_labelling = weights.reshape(labellings.shape[0], -1).sum(axis=1)
    for index, labelling in enumerate(labellings):
        label_posterior[np.arange(params.graph.n_nodes), labelling] += per_labelling[
            index
        ]

    state_posterior = np.zeros((params.n_classes, params.n_positions, params.n_states))
    for m in range(params.n_classes):
        axes = tuple(axis for axis in range(table.ndim) if axis != m + 1)
        per_path = weights.sum(axis=axes)  # (P,)
        for s in range(params.n_positions):
            np.add.at(state_posterior[m, s], paths[:, s], per_path)

    return ExactSpatioSequential(
        log_evidence=log_total - log_z_prior,
        label_posterior=label_posterior,
        state_posterior=state_posterior,
        log_prior_normalizer=log_z_prior,
    )


def conditional_state_posterior(
    params: SpatioSequentialParams, observations: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """``Q(k_{s,m} | l, x)`` by enumeration over paths, shape ``(M, S, K)``.

    Given the labels the classes decouple, so each is a sum over its own
    ``K**S`` paths. This is what part 3's per-class forward--backward is
    pinned to.
    """
    labels = np.asarray(labels, dtype=np.int64)
    paths = _paths(params)
    gated = gated_log_density(params, observations)
    posterior = np.zeros((params.n_classes, params.n_positions, params.n_states))
    for m in range(params.n_classes):
        members = np.flatnonzero(labels == m)
        scores = log_chain(params, m, paths)
        for n in members:
            scores = scores + gated[n, np.arange(params.n_positions), m, :][
                np.arange(params.n_positions), paths
            ].sum(axis=1)
        weights = np.exp(scores - logsumexp(scores, axis=0))
        for s in range(params.n_positions):
            np.add.at(posterior[m, s], paths[:, s], weights)
    return posterior


def log_evidence_by_forward(
    params: SpatioSequentialParams, observations: np.ndarray
) -> float:
    """``log p(x)`` by a route that shares no code with the enumeration.

    Sums over labellings; for each, the classes decouple and every class's
    evidence is the forward recursion on the product of its members'
    emission scores.
    """
    labellings = _labellings(params)
    prior = log_prior(params, labellings)
    log_z_prior = float(logsumexp(prior, axis=0))
    gated = gated_log_density(params, observations)
    log_transition = torch.log(torch.as_tensor(params.transition))
    terms = np.empty(labellings.shape[0])
    for index, labelling in enumerate(labellings):
        total = prior[index] - log_z_prior
        for m in range(params.n_classes):
            members = np.flatnonzero(labelling == m)
            density = gated[members, :, m, :].sum(axis=0)  # (S, K)
            total += float(
                forward_log_likelihood_from_density(
                    torch.as_tensor(density)[None],
                    torch.log(torch.as_tensor(params.initial[m])),
                    log_transition,
                )
            )
        terms[index] = total
    return float(logsumexp(terms, axis=0))


# --- the E step, the field and the joint given a labelling (issue #306) ----


#: Vertices whose emission scores are evaluated at once. The table
#: :func:`gated_log_density` returns is ``(n_nodes, S, M, K)``, which at the
#: declared 5,041-vertex instance is 1.0e10 entries and 80 GB: the E step and
#: the field therefore never build it, and walk the vertices in blocks whose
#: largest intermediate, ``(S, block, K)``, stays in the tens of megabytes at
#: every declared size. The answer does not depend on the block --- only the
#: order the members' scores are summed in, which moves the result by less
#: than the tolerance the Rust backend is pinned at.
VERTEX_BLOCK = 256


def _blocks(members: np.ndarray) -> Iterator[np.ndarray]:
    """``members`` in contiguous blocks of at most :data:`VERTEX_BLOCK`."""
    for start in range(0, members.size, VERTEX_BLOCK):
        yield members[start : start + VERTEX_BLOCK]


def covariate_block(
    params: SpatioSequentialParams, members: np.ndarray
) -> torch.Tensor | None:
    """``params.covariate`` for one block of vertices, ready for a family.

    Every seam here slices the covariate the way it slices the observations ---
    a column selection over the vertex axis. Written once because three seams
    do it, and a covariate sliced differently from the block it accompanies is
    a fit conditioning on the wrong exposures that converges anyway (#658).

    The singleton is added **only** where the covariate has no axes of its own
    --- :mod:`snakes_and_ladders.emissions` states the contract and enforces
    it, and #677 stopped the five seams that slice a covariate from each
    carrying their own copy of it. What is local here: a covariate that
    carries the family's axes is passed through, because
    :func:`~snakes_and_ladders.sim.count_pairs.split_covariate` is what puts
    the singleton inside each channel, and appending it here made a
    ``(S, V, 2)`` covariate ``(S, V, 2, 1)`` (#670).

    Returns
    -------
    torch.Tensor | None
        ``(S, len(members), 1)`` for a scalar-observation family and
        ``(S, len(members), ...)`` for one with its own axes, or ``None``
        where the params carry no covariate.
    """
    if params.covariate is None:
        return None
    block = params.covariate[:, members]
    return torch.as_tensor(block[..., None] if block.ndim == 2 else block)


def class_log_density(
    params: SpatioSequentialParams, observations: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """Per class, the summed emission scores of its members, shape ``(M, S, K)``.

    Given the labels the classes decouple, and each class's chain sees the
    product of its members' emissions -- a class with no members sees a flat
    score and its posterior is its prior.

    The sum is accumulated over blocks of members rather than over
    :func:`gated_log_density`'s whole table, which is what keeps it usable at
    the sizes `ROADMAP.md` declares; see :data:`VERTEX_BLOCK`.

    ``params.covariate`` is sliced by the same block as the observations, so a
    family scores each member against that member's own exposure (issue #652).
    The block is a column selection, which is why the covariate is stored with
    the observations' axes and not the class's.
    """
    labels = np.asarray(labels, dtype=np.int64)
    density = np.zeros((params.n_classes, params.n_positions, params.n_states))
    for m, family in enumerate(params.emissions):
        for block in _blocks(np.flatnonzero(labels == m)):
            scores = family.log_density(
                torch.as_tensor(observations[:, block], dtype=family.observation_dtype),
                covariate=covariate_block(params, block),
            )  # (S, block, K)
            density[m] += scores.detach().numpy().sum(axis=1)
    return density


@dataclass(frozen=True)
class ClassPosteriors:
    """The E step of the coupled model, given a labelling.

    Parameters
    ----------
    posterior : np.ndarray
        ``Q(k_{s,m} | l, x)``, shape ``(M, S, K)``.
    pairwise : np.ndarray
        ``Q(k_{s-1,m}, k_{s,m} | l, x)``, shape ``(M, S - 1, K, K)``.
    log_evidence : np.ndarray
        Per class, ``log p(x_{., l = m} | chain m)``, shape ``(M,)``.
    """

    posterior: np.ndarray
    pairwise: np.ndarray
    log_evidence: np.ndarray


def class_posteriors(
    params: SpatioSequentialParams,
    observations: np.ndarray,
    labels: np.ndarray,
    *,
    backend: Backend = Backend.PYTHON,
) -> ClassPosteriors:
    """Forward--backward on every class's chain over its members' summed scores."""
    if backend is Backend.RUST:
        from snakes_and_ladders.likelihood import spatio_sequential_rust

        return spatio_sequential_rust.class_posteriors(params, observations, labels)
    _refuse_backend(backend)
    density = class_log_density(params, observations, labels)
    log_transition = np.log(params.transition)
    posterior = np.empty_like(density)
    pairwise = np.empty(
        (
            params.n_classes,
            max(params.n_positions - 1, 0),
            params.n_states,
            params.n_states,
        )
    )
    evidence = np.empty(params.n_classes)
    for m in range(params.n_classes):
        run = forward_backward(density[m], np.log(params.initial[m]), log_transition)
        posterior[m] = run.posterior
        pairwise[m] = run.pairwise
        evidence[m] = run.log_evidence
    return ClassPosteriors(posterior, pairwise, evidence)


def external_field(
    params: SpatioSequentialParams,
    observations: np.ndarray,
    labels: np.ndarray,
    posterior: np.ndarray | None = None,
    *,
    backend: Backend = Backend.PYTHON,
) -> np.ndarray:
    """`    `H_nm`` of the external-field equation of the textbook: minus the posterior-expected emission score, shape ``(n_nodes, M)``.

    ``posterior`` defaults to the E step at ``labels``; passing one computed
    under other parameters is the ``theta'`` of the equation.

    This scores through its **own** ``log_density`` rather than through
    :func:`class_log_density`, so threading ``params.covariate`` here is not
    tidiness (issue #658):
    :func:`~snakes_and_ladders.search.spatio_sequential.fit_spatio_sequential`
    passes a ``posterior`` computed *with* the covariate, and until this the
    field was computed *without* --- so the step proposed labels under one
    model and accepted them under another. The ascent stays monotone either
    way, which is why nothing failed.
    """
    if backend is Backend.RUST:
        from snakes_and_ladders.likelihood import spatio_sequential_rust

        return spatio_sequential_rust.external_field(
            params, observations, labels, posterior
        )
    _refuse_backend(backend)
    if posterior is None:
        posterior = class_posteriors(params, observations, labels).posterior
    # The vertices the observations carry, not the graph's: a slice of a
    # declared instance is scored against the same parameters, and the field
    # is over what was observed.
    n_nodes = int(observations.shape[1])
    field = np.empty((n_nodes, params.n_classes))
    every = np.arange(n_nodes)
    for m, family in enumerate(params.emissions):
        for block in _blocks(every):
            scores = family.log_density(
                torch.as_tensor(observations[:, block], dtype=family.observation_dtype),
                covariate=covariate_block(params, block),
            )  # (S, block, K)
            field[block, m] = -np.einsum(
                "sbk,sk->b", scores.detach().numpy(), posterior[m]
            )
    return field


def labelled_log_likelihood(
    params: SpatioSequentialParams,
    observations: np.ndarray,
    labels: np.ndarray,
    *,
    backend: Backend = Backend.PYTHON,
) -> float:
    """``log p(x, l | theta)`` with the chains marginalized, up to ``log Z_Potts``.

    The quantity a block ascent must not decrease. The Potts normalizer is
    constant across the blocks (``beta`` and ``J`` are not fitted) and
    intractable past enumeration, so it is left out; add
    :attr:`ExactSpatioSequential.log_prior_normalizer` where enumeration
    reaches, which is how the test pins this against the oracle.
    """
    if backend is Backend.RUST:
        from snakes_and_ladders.likelihood import spatio_sequential_rust

        return spatio_sequential_rust.labelled_log_likelihood(
            params, observations, labels
        )
    _refuse_backend(backend)
    own = float(log_prior(params, np.asarray(labels, dtype=np.int64)[None, :])[0])
    evidence = class_posteriors(params, observations, labels).log_evidence
    return own + float(evidence.sum())


def _refuse_backend(backend: Backend) -> None:
    """The coupled E step runs on NumPy, the oracle, or on the tabulated Rust kernel.

    One enum names the kernel, as it does for `maxflow` and `count_pairs`
    (#819); the frozen triple of callables this replaced spelled the same
    choice a second way for one problem (#828).
    """
    if backend is not Backend.PYTHON:
        msg = (
            f"the coupled model runs on {Backend.PYTHON} or {Backend.RUST}, "
            f"not {backend}"
        )
        raise ValueError(msg)


def map_labelling(
    params: SpatioSequentialParams, observations: np.ndarray
) -> np.ndarray:
    """The labelling of highest ``p(l | x)``, by enumeration: the oracle a label step is held to."""
    labellings, _, table = _log_joint(params, observations)
    per_labelling = logsumexp(table.reshape(labellings.shape[0], -1), axis=1)
    return np.asarray(labellings[int(np.argmax(per_labelling))])


def marginal_log_likelihood_torch(
    params: SpatioSequentialParams, observations: np.ndarray, labels: np.ndarray
) -> torch.Tensor:
    """``log p(x | l, theta)`` as a differentiable scalar, through the forward recursion.

    The left side of the M-step identity of the textbook's coupled-model section: its gradient with respect to a
    family's parameters is what the posterior-weighted score must equal.

    It conditions on ``params.covariate`` like every other seam (issue #658):
    an identity checked between a covaried score and an uncovaried likelihood
    is an identity between two different models.
    """
    labels = np.asarray(labels, dtype=np.int64)
    log_transition = torch.log(torch.as_tensor(params.transition))
    total = torch.zeros((), dtype=torch.float64)
    for m, family in enumerate(params.emissions):
        members = np.flatnonzero(labels == m)
        scores = family.log_density(
            torch.as_tensor(observations[:, members], dtype=family.observation_dtype),
            covariate=covariate_block(params, members),
        )  # (S, n_m, K)
        density = scores.sum(dim=1)[None]  # (1, S, K)
        total = total + forward_log_likelihood_from_density(
            density, torch.log(torch.as_tensor(params.initial[m])), log_transition
        )
    return total
