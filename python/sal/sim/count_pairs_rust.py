"""The count-pair simulator in Rust, checked against the NumPy one (issue #399).

`sim.count_pairs.simulate_count_pairs` is the oracle and it stays
(`sim/CLAUDE.md`, `likelihood/CLAUDE.md`); this draws the same distributions
through ``src/count_pairs.rs`` and is what the declared 5,041-vertex instance
is drawn by, because it is the draw whose reproducibility does not depend on
NumPy's generator being the one that made it.

**Same contract, different arithmetic.** Both simulators give vertex ``v`` a
stream of its own keyed by the fixture's seed and ``v``, so neither depends on
the order vertices are visited in. What they do not share is the generator
that turns that key into variates, so the two draws are *different samples of
the same distributions* and the comparison between them is distributional:
per-state sample means and dispersions in both channels, at the tolerances
``tests/regression/sim/test_count_pairs_rust.py`` states. The fixture records
a digest of the counts this module produces, so a change here is visible in
review rather than in a recovery number.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from sal import oxisal
from sal.fixtures import load_params
from sal.sim.count_pairs import (
    CountPairInstance,
    IndependentCountPair,
    SpatioSequentialCountsParams,
    chain_states,
    coarsen,
    counts_digest,
    planted_labels,
    split_covariate,
)


def _tables(declared: SpatioSequentialCountsParams) -> dict[str, np.ndarray]:
    """The five ``(M, K)`` parameter tables the kernel takes, flattened.

    Raises
    ------
    TypeError
        If a class does not carry the two-channel count emission: there is
        nothing else this kernel knows how to draw.
    """
    params = declared.model
    shape = (params.n_classes, params.n_states)
    tables = {
        name: np.empty(shape)
        for name in ("dispersion", "mean", "trials", "alpha", "beta")
    }
    for m, family in enumerate(params.emissions):
        if not isinstance(family, IndependentCountPair):
            msg = (
                f"class {m} carries a {type(family).__name__}; the Rust simulator "
                f"draws the two-channel count emission"
            )
            raise TypeError(msg)
        tables["dispersion"][m] = family.total.dispersion.numpy()
        tables["mean"][m] = family.total.mean.numpy()
        tables["trials"][m] = family.successes.trials.numpy()
        tables["alpha"][m] = family.successes.alpha.numpy()
        tables["beta"][m] = family.successes.beta.numpy()
    return {
        name: np.ascontiguousarray(table).reshape(-1) for name, table in tables.items()
    }


def _channels(
    covariate: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """The model's covariate as the kernel's two flat ``S * V`` arrays.

    The kernel takes one array per channel rather than the ``(S, V, 2)`` the
    model holds, because each is read at a different point of the draw --- the
    exposure scales the gamma, the trial count sizes the binomial --- and a
    strided read of one channel out of an interleaved pair would touch both
    (issue #671). Splitting here is one copy per call against ``S * V`` reads.

    The split and the refusal are
    :func:`~sal.sim.count_pairs.split_covariate`'s, which states
    the rank rule --- one simulator must not accept what the other refuses, and
    this twin carried a second check that refused what the NumPy draw accepted
    (issue #856). What is left here is the flatten the kernel's
    one-dimensional arguments take; a covariate of the wrong length is the
    kernel's own refusal. The pair stays a pair (issue #865): it is one
    caller's two kernel arguments, and the split it wraps is named by
    :class:`~sal.sim.count_pairs.ChannelCovariates`.

    Raises
    ------
    CovariateNotSupportedError
        If the covariate names no channel.
    """
    channels = split_covariate(IndependentCountPair, covariate)
    if channels.exposure is None or channels.trials is None:
        return None, None
    return (
        np.ascontiguousarray(channels.exposure, dtype=np.float64).reshape(-1),
        np.ascontiguousarray(channels.trials, dtype=np.float64).reshape(-1),
    )


def simulate_count_pairs(
    declared: SpatioSequentialCountsParams,
) -> CountPairInstance:
    """Draw the fine instance: the planted labels, every class's chain, and every vertex's counts.

    The signature and the return of
    :func:`sal.sim.count_pairs.simulate_count_pairs`. The
    labels and the chains are that module's --- they are ``V`` and ``M x S``
    values, not ``S x V``, and cost nothing --- so what crosses into Rust is
    the ``1.0e8`` emission draws alone.

    Parameters
    ----------
    declared : SpatioSequentialCountsParams
        The loaded fixture.

    Returns
    -------
    CountPairInstance
        ``factor = 1``, its observations ``(S, n_nodes, 2)`` ``uint16``.

    Raises
    ------
    CovariateNotSupportedError
        If the model's covariate does not carry the channel axis the pair
        family splits on.
    """
    params = declared.model
    n_nodes = params.graph.n_nodes
    labels = planted_labels(params, params.n_classes)
    states = chain_states(params, np.random.default_rng([declared.seed, n_nodes]))

    totals = np.empty((params.n_positions, n_nodes), dtype=np.uint16)
    successes = np.empty_like(totals)
    tables = _tables(declared)
    exposure, trial_counts = _channels(params.covariate)
    oxisal.simulate_count_pairs(
        declared.seed,
        np.ascontiguousarray(states, dtype=np.int64).reshape(-1),
        np.ascontiguousarray(labels, dtype=np.int64),
        tables["dispersion"],
        tables["mean"],
        tables["trials"],
        tables["alpha"],
        tables["beta"],
        exposure,
        trial_counts,
        params.n_positions,
        n_nodes,
        params.n_classes,
        params.n_states,
        totals.reshape(-1),
        successes.reshape(-1),
    )
    return CountPairInstance(
        factor=1,
        params=params,
        labels=labels,
        states=states,
        observations=np.stack([totals, successes], axis=-1),
    )


@lru_cache(maxsize=2)
def fine_instance(path: Path) -> CountPairInstance:
    """The fine instance a fixture file declares, drawn once and held.

    **This accessor lives beside the Rust simulator and not beside the
    parameters** because it answers "what is the fixture's draw", and the
    answer names a simulator: the file states a seed and the parameters, and
    the counts are what this simulator makes of them. The NumPy simulator in
    :mod:`sal.sim.count_pairs` stays as the oracle it is
    checked against, and is not what a caller of this gets.

    Cached on the file, so the ``S x V`` draw --- 384.6 MiB at the declared
    5,041-vertex instance --- is paid once per session however many callers
    read it, and every coarse instance is a bin of the same draw. Nothing is
    written to disk.

    Parameters
    ----------
    path : Path
        The fixture file, as
        :attr:`sal.sim.fixtures.Fixture.path` gives it.

    Returns
    -------
    CountPairInstance
        ``factor = 1``.

    Raises
    ------
    ValueError
        If the file records a digest and the draw does not match it. A
        changed simulator is then a failure at the fixture rather than a
        drift in whatever was measured on it.
    """
    declared = load_params(path, SpatioSequentialCountsParams)
    instance = simulate_count_pairs(declared)
    digest = counts_digest(instance)
    if declared.counts_digest is not None and digest != declared.counts_digest:
        msg = (
            f"{path}: the draw digests {digest}, and the file records "
            f"{declared.counts_digest}; the simulator or the parameters moved"
        )
        raise ValueError(msg)
    return instance


@lru_cache(maxsize=4)
def binned_instance(path: Path, factor: int) -> CountPairInstance:
    """The instance a fixture file declares at one bin factor.

    Returns
    -------
    CountPairInstance
        :func:`sal.sim.count_pairs.coarsen` of
        :func:`fine_instance`, cached on both arguments.
    """
    return coarsen(fine_instance(path), factor)
