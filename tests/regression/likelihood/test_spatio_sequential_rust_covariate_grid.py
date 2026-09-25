"""The Rust coupled E step under a continuous covariate (issue #1064).

A log-normal exposure per observation makes a table by count and distinct
exposure as large as the observations. Both channels are factored instead ---
the negative binomial's exposure and the beta-binomial's trial count --- and
pinned per score against the families' own ``log_density`` and per E step
against the NumPy oracle. The trial count's two tabulated layouts, ``range``
and ``distinct``, are pinned bitwise to the factored one, and the rows each
builds are pinned bitwise to themselves rebuilt. The covariate grid is
conserved in ``sal.sandbox.covariate_grid`` and tested in
``tests/regression/sandbox/test_covariate_grid.py``. Fixture:
``spatio_sequential_counts_covariate``.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from sal import oxisal
from sal.backend import Backend
from sal.emissions import (
    BetaBinomialEmission,
    EmissionFamily,
    NegativeBinomialEmission,
)
from sal.likelihood import spatio_sequential_rust as rust
from sal.likelihood.spatio_sequential import (
    class_posteriors,
    external_field,
)
from sal.search.spatio_sequential import fit_spatio_sequential
from sal.sim.count_pairs import (
    SUCCESSES,
    TOTAL,
    CountPairInstance,
    IndependentCountPair,
)
from sal.sim.count_pairs_rust import fine_instance
from sal.sim.fixtures import fixture
from sal.sim.spatio_sequential import SpatioSequentialParams

PROBLEM = "spatio_sequential_counts_covariate"

#: Relative agreement of one observation's score with the family's
#: ``log_density``. The kernel forms ``B + y ln c - (y + r) ln t``, one
#: logarithm per score, where the family forms ``A + r ln(r/t) + y ln(mu c/t)``;
#: ``y ln mu`` in ``B`` cancels against ``-y ln t``. Measured 262.9 ulp at the ci
#: instance (2.3 in the family's order, traded for speed; issue #1064).
_ULPS = 512
_PER_SCORE = _ULPS * np.finfo(np.float64).eps

#: Relative agreement of an evidence and a field with the NumPy oracle: the
#: tolerance the uncovaried and the #658 covariate comparisons declare
#: (`test_spatio_sequential_rust_covariate.py`). Measured 4.4e-15 on the
#: evidence and 2.4e-15 on the field, one thread.
_RELATIVE = 1e-11

#: The posteriors are probabilities, compared absolutely, at the same
#: declared tolerance as there. Measured 1.0e-09.
_ABSOLUTE = 1e-7

#: Relative tolerance on a recovered negative-binomial mean or beta-binomial
#: rate: the 5% the exact path's coupled recovery test uses
#: (`tests/regression/search/test_spatio_sequential_pair_fit.py`).
_RECOVERY = 0.05

#: Positions of the ci instance the per-score checks read, every
#: tenth: 100 positions and 6,400 observations.
_EVERY = 10


def _instance() -> CountPairInstance:
    return fine_instance(fixture(PROBLEM, "ci").path)


def _at(params: SpatioSequentialParams, position: int) -> SpatioSequentialParams:
    """The model at one position, its covariate sliced to it."""
    covariate = params.covariate
    assert covariate is not None
    return replace(params, n_positions=1, covariate=covariate[position : position + 1])


def _kernel_scores(
    params: SpatioSequentialParams, observations: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """Every observation's pair score as the kernel forms it, ``(V, M, K)`` at one position.

    The field at one position with all of a state's weight is minus that
    state's score, so ``K`` calls read every class and state.
    """
    n_classes, n_states = params.n_classes, params.n_states
    scores = np.empty((observations.shape[1], n_classes, n_states))
    for k in range(n_states):
        weights = np.zeros((n_classes, 1, n_states))
        weights[:, 0, k] = 1.0
        scores[:, :, k] = -rust.external_field(params, observations, labels, weights)
    return scores


def _family_scores(
    params: SpatioSequentialParams, observations: np.ndarray
) -> np.ndarray:
    """The same scores through each family's ``log_density``, ``(V, M, K)``."""
    covariate = params.covariate
    assert covariate is not None
    return np.stack(
        [
            family.log_density(
                torch.as_tensor(observations[0], dtype=torch.float64),
                covariate=torch.as_tensor(covariate[0]),
            ).numpy()
            for family in params.emissions
        ],
        axis=1,
    )


def _sides(params: SpatioSequentialParams, channel: int) -> list[EmissionFamily]:
    """One channel's family per class."""
    sides: list[EmissionFamily] = []
    for family in params.emissions:
        assert isinstance(family, IndependentCountPair)
        sides.append(family.total if channel == TOTAL else family.successes)
    return sides


@pytest.mark.oracle
def test_each_factored_score_is_the_familys_within_the_declared_ulp() -> None:
    # The factorization claims each score, not only their sum: A[y] plus the
    # two exposure terms is the family's log_density at that exposure.
    instance = _instance()
    worst = 0.0
    for position in range(0, instance.params.n_positions, _EVERY):
        params = _at(instance.params, position)
        observations = instance.observations[position : position + 1]
        got = _kernel_scores(params, observations, instance.labels)
        want = _family_scores(params, observations)
        worst = max(worst, float((np.abs(got - want) / np.abs(want)).max()))

    assert worst <= _PER_SCORE, f"{worst / np.finfo(np.float64).eps:.1f} ulp"


@pytest.mark.oracle
@pytest.mark.backend
@pytest.mark.parametrize("layout", rust.COVARIATE_ROWS)
def test_the_e_step_and_field_match_the_numpy_oracle(
    layout: rust.CovariateRows,
) -> None:
    instance = _instance()
    params, observations, labels = (
        instance.params,
        instance.observations,
        instance.labels,
    )

    expected = class_posteriors(params, observations, labels)
    actual = rust.class_posteriors(params, observations, labels, covariate_rows=layout)

    np.testing.assert_allclose(
        actual.log_evidence, expected.log_evidence, rtol=_RELATIVE
    )
    np.testing.assert_allclose(
        actual.posterior, expected.posterior, rtol=_RELATIVE, atol=_ABSOLUTE
    )
    np.testing.assert_allclose(
        actual.pairwise, expected.pairwise, rtol=_RELATIVE, atol=_ABSOLUTE
    )
    np.testing.assert_allclose(
        rust.external_field(
            params, observations, labels, expected.posterior, covariate_rows=layout
        ),
        external_field(params, observations, labels, expected.posterior),
        rtol=_RELATIVE,
    )


def _success_scores(
    params: SpatioSequentialParams,
    observations: np.ndarray,
    layout: rust.CovariateRows,
) -> np.ndarray:
    """The successes' score alone as the kernel forms it, ``(S, V, M, K)``.

    The field at one position with all of a state's weight, over a first
    channel of zeros: every class and state of every observation, in ``K``
    calls per position.
    """
    rows = rust.emission_rows(params, observations, covariate_rows=layout)
    arguments = {} if rows.trials is None else rows.trials.arguments()
    n_positions, n_nodes = observations.shape[:2]
    n_classes, n_states = params.n_classes, params.n_states
    scores = np.empty((n_positions, n_nodes, n_classes, n_states))
    for s in range(n_positions):
        if rows.trials is not None:
            arguments["trials"] = rows.trials.trials[s]
        for k in range(n_states):
            weights = np.zeros((1, n_classes, n_states))
            weights[0, :, k] = 1.0
            field = np.empty(n_nodes * n_classes)
            oxisal.external_field(
                np.zeros(n_nodes, dtype=np.uint32),
                np.ascontiguousarray(rows.success_rows[s]),
                np.zeros(n_classes * n_states),
                rows.success_table.reshape(-1),
                weights.reshape(-1),
                1,
                n_nodes,
                n_classes,
                n_states,
                field,
                **arguments,
            )
            scores[s, :, :, k] = -field.reshape(n_nodes, n_classes)
    return scores


@pytest.mark.oracle
@pytest.mark.parametrize("layout", rust.COVARIATE_ROWS)
def test_each_trial_count_score_is_the_familys_bitwise(
    layout: rust.CovariateRows,
) -> None:
    # The factored layout sums the nine `lgamma` terms in the family's order,
    # each tabulated at the family's own argument; the two tabulated layouts
    # read the family's `log_density` at the pair. Measured 0 ulp at every one
    # of the 6.4e6 scores of the ci instance, in every layout.
    instance = _instance()
    params, observations = instance.params, instance.observations
    covariate = params.covariate
    assert covariate is not None
    n_positions, n_nodes = observations.shape[:2]

    got = _success_scores(params, observations, layout)

    successes = torch.as_tensor(
        observations[..., SUCCESSES].reshape(-1), dtype=torch.float64
    )
    trials = torch.as_tensor(covariate[..., SUCCESSES].reshape(-1, 1))
    want = np.stack(
        [
            side.log_density(successes, covariate=trials)
            .numpy()
            .reshape(n_positions, n_nodes, -1)
            for side in _sides(params, SUCCESSES)
        ],
        axis=2,
    )
    assert np.array_equal(got, want)


@pytest.mark.smoke
@pytest.mark.backend
@pytest.mark.parametrize("layout", ["range", "distinct"])
def test_every_tabulated_layout_is_the_factored_e_step_and_field_bitwise(
    layout: rust.CovariateRows,
) -> None:
    # The scores are the same bits in every layout (above), and the kernel sums
    # them in one order: the E step and the field follow.
    instance = _instance()
    params, observations, labels = (
        instance.params,
        instance.observations,
        instance.labels,
    )

    factored = rust.class_posteriors(
        params, observations, labels, covariate_rows="factored"
    )
    tabulated = rust.class_posteriors(
        params, observations, labels, covariate_rows=layout
    )

    assert np.array_equal(tabulated.log_evidence, factored.log_evidence)
    assert np.array_equal(tabulated.posterior, factored.posterior)
    assert np.array_equal(tabulated.pairwise, factored.pairwise)
    assert np.array_equal(
        rust.external_field(
            params, observations, labels, factored.posterior, covariate_rows=layout
        ),
        rust.external_field(
            params, observations, labels, factored.posterior, covariate_rows="factored"
        ),
    )


@pytest.mark.smoke
@pytest.mark.backend
def test_first_appearance_codes_are_the_sorted_codes_bitwise() -> None:
    # `oxisal.factorize` codes the trial counts in the order they appear and
    # `np.unique` in sorted order: the table's rows are permuted and every row
    # an observation reads holds the same number.
    instance = _instance()
    params, observations, labels = (
        instance.params,
        instance.observations,
        instance.labels,
    )
    covariate = params.covariate
    assert covariate is not None
    hashed = rust.observation_rows(observations, covariate, covariate_rows="distinct")
    trials = covariate[..., SUCCESSES]
    distinct, codes = np.unique(trials.reshape(-1), return_inverse=True)
    successes = observations[..., SUCCESSES]
    rows = successes.astype(np.int64) * distinct.size + codes.reshape(trials.shape)
    sorted_rows = replace(
        hashed,
        successes=rust.ChannelRows(
            np.ascontiguousarray(rows, dtype=np.uint32),
            hashed.successes.extent,
            distinct,
        ),
    )
    levels = hashed.successes.levels
    assert levels is not None

    first = rust.class_posteriors(params, observations, labels, covariate_rows=hashed)
    second = rust.class_posteriors(
        params, observations, labels, covariate_rows=sorted_rows
    )

    assert not np.array_equal(levels, distinct)
    assert np.array_equal(np.sort(levels), distinct)
    assert np.array_equal(first.log_evidence, second.log_evidence)
    assert np.array_equal(first.posterior, second.posterior)
    assert np.array_equal(first.pairwise, second.pairwise)


@pytest.mark.smoke
@pytest.mark.backend
@pytest.mark.parametrize("layout", rust.COVARIATE_ROWS)
def test_rows_built_once_are_the_rows_built_per_call_bitwise(
    layout: rust.CovariateRows,
) -> None:
    # A fit builds the rows once and every E step builds only the tables; the
    # rows name no parameter, so the answer is the same bits at other
    # parameters than the ones current when they were built.
    instance = _instance()
    observations, labels = instance.observations, instance.labels
    params = replace(
        instance.params,
        emissions=tuple(
            IndependentCountPair(
                NegativeBinomialEmission(
                    family.total.dispersion, family.total.mean * 1.1
                ),
                family.successes,
            )
            for family in instance.params.emissions
            if isinstance(family, IndependentCountPair)
        ),
    )
    rows = rust.observation_rows(
        observations, instance.params.covariate, covariate_rows=layout
    )

    cached = rust.class_posteriors(params, observations, labels, covariate_rows=rows)
    built = rust.class_posteriors(params, observations, labels, covariate_rows=layout)

    assert np.array_equal(cached.log_evidence, built.log_evidence)
    assert np.array_equal(cached.posterior, built.posterior)
    assert np.array_equal(cached.pairwise, built.pairwise)
    assert np.array_equal(
        rust.external_field(params, observations, labels, covariate_rows=rows),
        rust.external_field(params, observations, labels, covariate_rows=layout),
    )


@pytest.mark.smoke
def test_rows_built_for_other_observations_are_refused() -> None:
    instance = _instance()
    params, observations = instance.params, instance.observations
    rows = rust.observation_rows(observations, params.covariate)

    with pytest.raises(ValueError, match="other observations"):
        rust.emission_rows(params, observations.copy(), covariate_rows=rows)


@pytest.mark.smoke
def test_an_unknown_layout_is_refused() -> None:
    instance = _instance()

    with pytest.raises(ValueError, match="covariate_rows"):
        rust.emission_rows(
            instance.params,
            instance.observations,
            covariate_rows="sorted",  # type: ignore[arg-type]
        )


@pytest.mark.end2end
def test_em_on_both_factored_channels_recovers_the_generating_parameters() -> None:
    # Block ascent on the Rust backend, both channels factored and the rows
    # built once for the fit, started from means 30% high and dispersions 30%
    # low; recovery is held to the exact path's 5%. Two blocks: the measured
    # fit reaches its value at the first and holds it through eight.
    instance = _instance()
    planted = instance.params
    start = replace(
        planted,
        emissions=tuple(
            IndependentCountPair(
                NegativeBinomialEmission(
                    family.total.dispersion * 0.7, family.total.mean * 1.3
                ),
                family.successes,
            )
            for family in planted.emissions
            if isinstance(family, IndependentCountPair)
        ),
    )

    fit = fit_spatio_sequential(
        start,
        instance.observations,
        np.random.default_rng(3),
        labels=instance.labels,
        n_blocks=2,
        backend=Backend.RUST,
        covariate_rows="factored",
    )

    assert np.array_equal(fit.labels, instance.labels)
    for truth, fitted in zip(planted.emissions, fit.params.emissions, strict=True):
        assert isinstance(truth, IndependentCountPair)
        assert isinstance(fitted, IndependentCountPair)
        for want, got in (
            (truth.total.mean, fitted.total.mean),
            (truth.total.dispersion, fitted.total.dispersion),
            (_rate(truth.successes), _rate(fitted.successes)),
        ):
            assert (torch.abs(got / want - 1.0) < _RECOVERY).all(), (want, got)


def _rate(family: BetaBinomialEmission) -> torch.Tensor:
    """The beta-binomial's mean success fraction ``a / (a + b)``."""
    return family.alpha / (family.alpha + family.beta)


@pytest.mark.smoke
def test_a_layout_on_the_numpy_oracle_is_refused() -> None:
    # The oracle builds no table, so a layout handed to it would be ignored.
    instance = _instance()

    with pytest.raises(ValueError, match="covariate_rows"):
        class_posteriors(
            instance.params,
            instance.observations,
            instance.labels,
            covariate_rows="factored",
        )


@pytest.mark.smoke
def test_a_negative_exposure_is_refused() -> None:
    instance = _instance()
    covariate = instance.params.covariate
    assert covariate is not None
    negative = covariate.copy()
    negative[0, 0, TOTAL] = -1.0

    with pytest.raises(ValueError, match="non-negative"):
        rust.emission_rows(
            replace(instance.params, covariate=negative), instance.observations
        )


@pytest.mark.smoke
@pytest.mark.critical
def test_the_factored_layout_rows_both_channels_by_count() -> None:
    # The rows are the counts in both channels, the covariate travels to the
    # kernel, and each table spans its own integer: 0.35 MB against the
    # 13.1 MB of the `range` table at stress.
    instance = _instance()
    params, observations = instance.params, instance.observations
    covariate = params.covariate
    assert covariate is not None

    rows = rust.emission_rows(params, observations, covariate_rows="factored")

    assert rows.exposure is not None
    assert rows.trials is not None
    assert np.array_equal(rows.total_rows, observations[..., TOTAL])
    assert np.array_equal(rows.success_rows, observations[..., SUCCESSES])
    assert np.array_equal(rows.trials.trials, covariate[..., SUCCESSES])
    successes = observations[..., SUCCESSES]
    assert rows.success_table.shape[0] == int(successes.max()) + 1
    assert rows.trials.trial.shape[0] == int(covariate[..., SUCCESSES].max()) + 1


@pytest.mark.smoke
@pytest.mark.critical
def test_by_default_the_trial_count_spans_its_range() -> None:
    # The row is `successes * n_codes + n - n_min`, every trial count from the
    # least to the greatest a code; the exposure is factored.
    instance = _instance()
    params, observations = instance.params, instance.observations
    covariate = params.covariate
    assert covariate is not None

    rows = rust.emission_rows(params, observations)

    trials = covariate[..., SUCCESSES].astype(np.int64)
    low, n_codes = int(trials.min()), int(trials.max() - trials.min()) + 1
    successes = observations[..., SUCCESSES].astype(np.int64)
    assert rows.exposure is not None
    assert rows.trials is None
    assert np.array_equal(rows.success_rows, successes * n_codes + trials - low)
    assert rows.success_table.shape[0] == (int(successes.max()) + 1) * n_codes


@pytest.mark.smoke
def test_the_distinct_layout_codes_in_first_appearance_order() -> None:
    # The row is `successes * n_distinct + code`, the code each trial count's
    # index among the distinct values in the order they first occur.
    instance = _instance()
    params, observations = instance.params, instance.observations
    covariate = params.covariate
    assert covariate is not None

    rows = rust.emission_rows(params, observations, covariate_rows="distinct")

    trials = covariate[..., SUCCESSES].reshape(-1)
    _, first = np.unique(trials, return_index=True)
    levels = trials[np.sort(first)]
    codes = {value: code for code, value in enumerate(levels)}
    successes = observations[..., SUCCESSES]
    expected = successes.astype(np.int64) * levels.size + np.array(
        [codes[value] for value in trials]
    ).reshape(successes.shape)
    assert np.array_equal(rows.success_rows, expected)
    assert rows.success_table.shape[0] == (int(successes.max()) + 1) * levels.size
