"""Felsenstein pruning under JAX: the value and its gradient in one compiled program (issue #1005).

The recursion is :func:`snakes_and_ladders.likelihood.pruning_torch.log_likelihood`'s
operation for operation --- the same post-order schedule, the same transition
matrices (``eq:jc`` in closed form, or ``expm(Q t)``), the same rescaling by
each node's per-site maximum --- so the two agree to rounding. What differs
is where it runs: the post-order is unrolled while ``jit`` traces it, since a
topology is fixed for the life of an objective, and ``jax.value_and_grad``
differentiates the unrolled program. One program is compiled per topology,
alphabet and model and cached on them (the ``*_program`` functions), and the alignment,
the branch lengths and the model are its arguments, so a second objective on
the same tree reuses it.

The objectives in :mod:`snakes_and_ladders.likelihood.objective` map their
``theta`` into this program's arguments inside the same compiled function
(:func:`branch_length_program`, :func:`substitution_model_program`), so the
gradient is in ``theta`` directly and nothing crosses back to PyTorch but the
two results.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from snakes_and_ladders.likelihood.patterns import check_weights
from snakes_and_ladders.likelihood.pruning_common import (
    check_alignment_covers,
    leaf_indicator_array,
)
from snakes_and_ladders.likelihood.pruning_torch import _traversal
from snakes_and_ladders.sim.tree import Node

#: A post-order schedule as :func:`pruning_torch._traversal` builds it: per
#: node, its slot, its leaf name or ``None``, and ``(child slot, branch)``
#: pairs.
type Steps = tuple[tuple[int, str | None, tuple[tuple[int, int], ...]], ...]


def _jax() -> Any:
    import jax  # a core dependency, imported where it is used

    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
    return jax


def steps(tau: Node) -> Steps:
    """``tau``'s post-order schedule, the key its program is compiled and cached on."""
    return _traversal(tau).steps


def placed(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """``arrays`` placed on the device once, so every call reads the same buffers."""
    jax = _jax()
    return {key: jax.device_put(value) for key, value in arrays.items()}


def leaves(
    tau: Node, k: int, alignment: Mapping[str, np.ndarray], weights: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """The leaf indicator partials in post-order, ``(n_leaves, n_sites, k)``, and the site weights.

    Built once per objective: they are the data, constant in everything a
    fit moves.
    """
    traversal = _traversal(tau)
    check_alignment_covers(traversal.leaf_names, alignment)
    n_sites = int(np.asarray(alignment[traversal.leaf_names[0]]).shape[0])
    stacked = np.stack(
        [
            leaf_indicator_array(np.asarray(alignment[name]), n_sites, k)
            for name in traversal.leaf_names
        ]
    )
    weight = check_weights(weights, n_sites)
    return stacked, (np.ones(n_sites) if weight is None else np.asarray(weight, float))


def _log_likelihood(
    jax: Any, steps: Steps, k: int, rescale: bool
) -> Callable[[Any, Any, Any, Any, Any], Any]:
    """``(lengths, pi, rate, leaves, weight) -> log L``, traced over ``steps``."""
    jnp = jax.numpy
    leaf_index: dict[int, int] = {}
    for slot, name, _ in steps:
        if name is not None:
            leaf_index[slot] = len(leaf_index)

    def transitions(lengths: Any, rate: Any) -> Any:
        if rate is None:
            # eq:jc, as pruning_torch._jc_transition_probabilities.
            decay = jnp.exp(-k * lengths / (k - 1))[..., None, None]
            off = (1.0 - decay) / k
            diagonal = 1.0 / k + (k - 1) / k * decay
            eye = jnp.eye(k)
            return off * (1.0 - eye) + diagonal * eye
        return jax.scipy.linalg.expm(rate * lengths[..., None, None])

    def log_likelihood(lengths: Any, pi: Any, rate: Any, data: Any, weight: Any) -> Any:
        matrices = transitions(lengths, rate)
        log_scale = jnp.zeros(data.shape[1])
        partials: dict[int, Any] = {}
        for slot, name, children in steps:
            if name is not None:
                partials[slot] = data[leaf_index[slot]]
                continue
            partial = None
            for child_slot, branch in children:
                # message[s, i] = sum_j P_ij(t) * L_child(s, j) -- eq:pruning.
                message = partials.pop(child_slot) @ matrices[branch].T
                partial = message if partial is None else partial * message
            if partial is None:
                partial = jnp.ones((data.shape[1], k))
            if rescale:
                # As pruning_common.rescale_partial: a vanished scale is
                # replaced by one, so log(0) reaches the total.
                scale = partial.max(axis=1)
                safe = jnp.where(scale > 0, scale, 1.0)
                partial = partial / safe[:, None]
                log_scale = log_scale + jnp.log(safe)
            partials[slot] = partial
        (root,) = partials.values()
        return jnp.dot(weight, jnp.log(root @ pi) + log_scale)

    return log_likelihood


@functools.cache
def branch_length_program(
    steps: Steps,
    k: int,
    n_branches: int,
    source: tuple[int, ...],
    halves: tuple[int, ...],
) -> Any:
    """``jit(value_and_grad)`` of the JC negative log-likelihood in ``theta``, per topology.

    ``theta`` maps to one length per branch as
    :meth:`BranchLengthObjective.branch_lengths` maps it: ``exp`` of the
    estimable entries, gathered by ``source``, with the confounded root pair
    (``halves``) each at half their estimable sum.
    """
    jax = _jax()
    jnp = jax.numpy
    log_likelihood = _log_likelihood(jax, steps, k, rescale=True)
    gather = np.asarray(source)
    factor = np.ones(n_branches)
    factor[list(halves)] = 0.5

    def negative(theta: Any, pi: Any, data: Any, weight: Any) -> Any:
        lengths = jnp.exp(theta)[gather] * factor
        return -log_likelihood(lengths, pi, None, data, weight)

    return jax.jit(jax.value_and_grad(negative))


@functools.cache
def substitution_model_program(
    steps: Steps,
    k: int,
    n_branches: int,
    source: tuple[int, ...],
    halves: tuple[int, ...],
    n_branch_parameters: int,
) -> Any:
    """``jit(value_and_grad)`` of the GTR negative log-likelihood in ``theta``, per topology.

    ``theta`` is branch lengths, free exchangeabilities and free ``pi``, as
    :class:`SubstitutionModelObjective` lays it out, and the rate matrix is
    built as :meth:`SubstitutionModelObjective.rate_matrix` builds it.
    """
    jax = _jax()
    jnp = jax.numpy
    log_likelihood = _log_likelihood(jax, steps, k, rescale=True)
    gather = np.asarray(source)
    factor = np.ones(n_branches)
    factor[list(halves)] = 0.5
    rows, columns = np.triu_indices(k, k=1)
    n_exchange = rows.size - 1

    def negative(theta: Any, data: Any, weight: Any) -> Any:
        branches = theta[:n_branch_parameters]
        free_exchange = theta[n_branch_parameters : n_branch_parameters + n_exchange]
        free_pi = theta[n_branch_parameters + n_exchange :]
        pi = jnp.exp(jax.nn.log_softmax(jnp.concatenate([jnp.zeros(1), free_pi])))
        values = jnp.concatenate([jnp.exp(free_exchange), jnp.ones(1)])
        upper = jnp.zeros((k, k)).at[rows, columns].set(values)
        rate = (upper + upper.T) * pi[None, :]
        rate = rate - jnp.diag(rate.sum(axis=1))
        rate = rate / -(pi * jnp.diagonal(rate)).sum()
        lengths = jnp.exp(branches)[gather] * factor
        return -log_likelihood(lengths, pi, rate, data, weight)

    return jax.jit(jax.value_and_grad(negative))
