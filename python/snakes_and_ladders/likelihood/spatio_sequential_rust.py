"""The coupled E step and its field in Rust, pinned to the NumPy oracle (issue #399).

The same two quantities :mod:`snakes_and_ladders.likelihood.spatio_sequential`
computes --- ``class_posteriors`` and ``external_field`` --- over the
two-channel count emission of :mod:`snakes_and_ladders.sim.count_pairs`,
through ``src/coupled.rs``. The NumPy path stays and is the oracle
(``likelihood/CLAUDE.md``); this is pinned against it at the ci instance and
on a 64-vertex slice of the declared 5,041-vertex one.

**Why there is a Rust path at all.** `cProfile` on the NumPy path at the 5K
instance, bin factor 10: one ``class_posteriors`` is 20.1 s, of which
``torch.lgamma`` is 10.4 s of self time (51.5%) and the two emission
``log_density`` bodies 18.9 s cumulative (94.3%); one ``external_field`` is
191.3 s, of which ``torch.lgamma`` is 102.8 s (53.7%) and the emission
densities 188.1 s (98.3%). Everything else --- forward--backward, the
``einsum``, the boundary --- is under 4%. The measured speedups are in
``STATUS.md``.

**The counts index a table instead of calling `lgamma`.** Both channels'
observations are integers in a range of a few thousand, so
``log p(count | class, state)`` is a function of an integer and is tabulated
once per call by the families themselves. The kernel then does two loads and
an add where the oracle does three ``lgamma`` calls, and the tables are the
oracle's own arithmetic rather than a second implementation of it. The tables
are count-major, ``[count, M, K]``, for the reason ``src/coupled.rs`` states.

**One crossing per call, contiguous.** The counts cross as the ``(S, V)``
arrays they are already held in, cast to ``uint16`` --- which is the same
statement the table makes, that the counts are small --- and the results are
written into arrays allocated here.
"""

from __future__ import annotations

import numpy as np
import torch

from snakes_and_ladders import oxi_snakes_and_ladders
from snakes_and_ladders.likelihood.spatio_sequential import (
    ClassPosteriors,
    CoupledBackend,
    log_prior,
)
from snakes_and_ladders.sim.count_pairs import (
    SUCCESSES,
    TOTAL,
    IndependentCountPair,
)
from snakes_and_ladders.sim.spatio_sequential import SpatioSequentialParams


def _families(params: SpatioSequentialParams) -> list[IndependentCountPair]:
    """The two-channel families of every class, at their own type.

    Raises
    ------
    TypeError
        If a class does not carry one. The kernel tabulates two channels by
        their integer counts, and a family that is not a pair of count
        families has nothing to tabulate.
    """
    families = []
    for m, family in enumerate(params.emissions):
        if not isinstance(family, IndependentCountPair):
            msg = (
                f"class {m} carries a {type(family).__name__}; the Rust E step "
                f"is over the two-channel count emission"
            )
            raise TypeError(msg)
        families.append(family)
    return families


def emission_tables(
    params: SpatioSequentialParams, observations: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Both channels' log-density for every class, state and count in the data.

    Parameters
    ----------
    params : SpatioSequentialParams
        Its emissions the two-channel families.
    observations : np.ndarray
        Shape ``(S, n_nodes, 2)``; only the largest count in each channel is
        read, since that is the extent each table must cover.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The first channel's table and the second's, each ``(count, M, K)``
        contiguous ``float64``.
    """
    families = _families(params)
    tables = []
    for channel in (TOTAL, SUCCESSES):
        extent = int(observations[..., channel].max()) + 1
        counts = torch.arange(extent, dtype=torch.float64)
        table = np.empty((extent, params.n_classes, params.n_states))
        for m, family in enumerate(families):
            side = family.total if channel == TOTAL else family.successes
            table[:, m, :] = side.log_density(counts).numpy()
        tables.append(np.ascontiguousarray(table))
    return tables[0], tables[1]


def _counts(observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Both channels as contiguous ``uint16``, or a refusal naming the overflow."""
    largest = int(observations.max())
    if largest > np.iinfo(np.uint16).max:
        msg = (
            f"a count of {largest} does not fit uint16; the kernel indexes a "
            f"table by the count, so a range this wide is a different kernel"
        )
        raise ValueError(msg)
    return (
        np.ascontiguousarray(observations[..., TOTAL], dtype=np.uint16),
        np.ascontiguousarray(observations[..., SUCCESSES], dtype=np.uint16),
    )


def class_posteriors(
    params: SpatioSequentialParams, observations: np.ndarray, labels: np.ndarray
) -> ClassPosteriors:
    """Forward--backward on every class's chain over its members' summed scores.

    The signature and the return of
    :func:`snakes_and_ladders.likelihood.spatio_sequential.class_posteriors`,
    which is the oracle this is pinned to.

    Parameters
    ----------
    params : SpatioSequentialParams
        The model, its emissions the two-channel count families.
    observations : np.ndarray
        Shape ``(S, n_nodes, 2)``, integer counts.
    labels : np.ndarray
        One class per vertex, shape ``(n_nodes,)``.

    Returns
    -------
    ClassPosteriors
        The posterior ``(M, S, K)``, the pairwise ``(M, S - 1, K, K)`` and the
        per-class log evidence ``(M,)``.
    """
    totals, successes = _counts(observations)
    total_table, success_table = emission_tables(params, observations)
    n_positions, n_nodes = observations.shape[:2]
    posterior = np.empty((params.n_classes, n_positions, params.n_states))
    pairwise = np.empty(
        (
            params.n_classes,
            max(n_positions - 1, 0),
            params.n_states,
            params.n_states,
        )
    )
    log_evidence = np.empty(params.n_classes)
    log_transition = np.ascontiguousarray(
        np.broadcast_to(
            np.log(params.transition),
            (params.n_classes, params.n_states, params.n_states),
        )
    )
    oxi_snakes_and_ladders.class_posteriors(
        totals.reshape(-1),
        successes.reshape(-1),
        np.ascontiguousarray(labels, dtype=np.int64),
        total_table.reshape(-1),
        success_table.reshape(-1),
        np.ascontiguousarray(np.log(params.initial)).reshape(-1),
        log_transition.reshape(-1),
        n_positions,
        n_nodes,
        params.n_classes,
        params.n_states,
        posterior.reshape(-1),
        pairwise.reshape(-1),
        log_evidence,
    )
    return ClassPosteriors(posterior, pairwise, log_evidence)


def external_field(
    params: SpatioSequentialParams,
    observations: np.ndarray,
    labels: np.ndarray,
    posterior: np.ndarray | None = None,
) -> np.ndarray:
    """``H[n, m]``, minus the posterior-expected emission score, shape ``(n_nodes, M)``.

    The signature and the return of
    :func:`snakes_and_ladders.likelihood.spatio_sequential.external_field`.
    ``posterior`` defaults to this module's own E step at ``labels``.

    Returns
    -------
    np.ndarray
        Shape ``(n_nodes, M)``.
    """
    if posterior is None:
        posterior = class_posteriors(params, observations, labels).posterior
    totals, successes = _counts(observations)
    total_table, success_table = emission_tables(params, observations)
    n_positions, n_nodes = observations.shape[:2]
    field = np.empty((n_nodes, params.n_classes))
    # (M, S, K) to (S, M, K): the kernel wants one position's weights
    # contiguous beside the tables' rows, which are count-major.
    weights = np.ascontiguousarray(np.moveaxis(posterior, 0, 1))
    oxi_snakes_and_ladders.external_field(
        totals.reshape(-1),
        successes.reshape(-1),
        total_table.reshape(-1),
        success_table.reshape(-1),
        weights.reshape(-1),
        n_positions,
        n_nodes,
        params.n_classes,
        params.n_states,
        field.reshape(-1),
    )
    return field


def labelled_log_likelihood(
    params: SpatioSequentialParams, observations: np.ndarray, labels: np.ndarray
) -> float:
    """``log p(x, l | theta)`` with the chains marginalized, up to ``log Z_Potts``.

    The signature and the return of
    :func:`snakes_and_ladders.likelihood.spatio_sequential.labelled_log_likelihood`,
    over this module's E step. The Potts term is the oracle's own: it is a sum
    over edges and costs nothing beside the emission densities.

    Returns
    -------
    float
    """
    own = float(log_prior(params, np.asarray(labels, dtype=np.int64)[None, :])[0])
    return own + float(
        class_posteriors(params, observations, labels).log_evidence.sum()
    )


#: This module's implementations as one backend, for
#: :func:`snakes_and_ladders.search.spatio_sequential.fit_spatio_sequential`.
RUST_BACKEND = CoupledBackend(
    class_posteriors=class_posteriors,
    external_field=external_field,
    labelled_log_likelihood=labelled_log_likelihood,
)
