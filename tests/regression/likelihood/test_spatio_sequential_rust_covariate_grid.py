"""The Rust coupled E step under a continuous covariate (issue #1064).

A log-normal exposure per observation makes a table by count and distinct
exposure as large as the observations. The negative binomial's exposure is
factored instead, and pinned per score against the family's own
``log_density`` and per E step against the NumPy oracle. The trial count is
tabulated, on a grid where ``covariate_tolerance`` is set: an integer covariate
is exact on it, and a continuous one is held to the bound
``covariate_grid_error`` states. Fixture: ``spatio_sequential_counts_covariate``.
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
#: ``log_density``. The terms are the family's, in its order; what differs is
#: Rust's ``ln`` against torch's ``log``. Measured 2.3 ulp at the ci instance.
_ULPS = 8
_PER_SCORE = _ULPS * np.finfo(np.float64).eps

#: Relative agreement of an evidence and a field with the NumPy oracle: the
#: tolerance the uncovaried and the #658 covariate comparisons declare
#: (`test_spatio_sequential_rust_covariate.py`). Measured 4.4e-15 on the
#: evidence and 2.4e-15 on the field, one thread.
_RELATIVE = 1e-11

#: The posteriors are probabilities, compared absolutely, at the same
#: declared tolerance as there. Measured 1.0e-09.
_ABSOLUTE = 1e-7

#: A tolerance the trial count's grid is asked for. Any positive value codes
#: an integer covariate at scale one; this one is the value the benchmark uses.
_TOLERANCE = 1e-3

#: Relative tolerance on a recovered negative-binomial mean or beta-binomial
#: rate: the 5% the exact path's coupled recovery test uses
#: (`tests/regression/search/test_spatio_sequential_pair_fit.py`).
_RECOVERY = 0.05

#: Positions of the ci instance the per-score and grid checks read, every
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
def test_each_factored_score_is_the_familys_to_a_few_ulp() -> None:
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
def test_the_factored_e_step_and_field_match_the_numpy_oracle() -> None:
    instance = _instance()
    params, observations, labels = (
        instance.params,
        instance.observations,
        instance.labels,
    )

    expected = class_posteriors(params, observations, labels)
    actual = rust.class_posteriors(params, observations, labels)

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
        rust.external_field(params, observations, labels, expected.posterior),
        external_field(params, observations, labels, expected.posterior),
        rtol=_RELATIVE,
    )


@pytest.mark.backend
def test_an_integer_covariate_on_the_grid_is_the_exact_path_bitwise() -> None:
    # The trial count is an integer, coded at scale one, where the code is the
    # value: the grid changes the table's layout and none of its values.
    instance = _instance()
    params, observations, labels = (
        instance.params,
        instance.observations,
        instance.labels,
    )
    trials = params.covariate
    assert trials is not None

    exact = rust.class_posteriors(params, observations, labels)
    grid = rust.class_posteriors(
        params, observations, labels, covariate_tolerance=_TOLERANCE
    )
    sides = _sides(params, SUCCESSES)

    assert (
        rust.grid_scale(
            sides, observations[..., SUCCESSES], trials[..., SUCCESSES], _TOLERANCE
        )
        == 1.0
    )
    assert np.array_equal(grid.log_evidence, exact.log_evidence)
    assert np.array_equal(grid.posterior, exact.posterior)
    assert np.array_equal(grid.pairwise, exact.pairwise)
    assert np.array_equal(
        rust.external_field(
            params, observations, labels, exact.posterior, covariate_tolerance=1.0
        ),
        rust.external_field(params, observations, labels, exact.posterior),
    )
    assert not rust.covariate_grid_error(params, observations, _TOLERANCE).any()


@pytest.mark.oracle
def test_a_covariate_on_a_power_of_two_grid_is_tabulated_bitwise() -> None:
    # An exposure already a multiple of 1/4 is coded at scale 4, and
    # code / scale is exact in float64: every tabulated score is the family's
    # log_density at the observation's own exposure, bit for bit.
    instance = _instance()
    params = instance.params
    assert params.covariate is not None
    exposure = np.maximum(np.rint(params.covariate[..., TOTAL] * 4.0), 1.0) / 4.0
    values = instance.observations[::_EVERY, :, TOTAL]
    exposure = exposure[::_EVERY]
    sides = _sides(params, TOTAL)

    table, rows = rust._grid_rows(sides, params, values, exposure, 4.0)

    for m, side in enumerate(sides):
        direct = side.log_density(
            torch.as_tensor(values.reshape(-1), dtype=torch.float64),
            covariate=torch.as_tensor(exposure.reshape(-1, 1)),
        ).numpy()
        assert np.array_equal(table[rows.reshape(-1), m, :], direct)


@pytest.mark.oracle
@pytest.mark.parametrize("tolerance", [1.0, 0.25])
def test_a_continuous_covariate_on_the_grid_is_within_its_stated_bound(
    tolerance: float,
) -> None:
    # The grid applied to the continuous exposure, where the factorization is
    # what runs: the one continuous covariate the model carries. Measured at
    # tolerance 1.0: scale 64, largest bound 0.81 and largest change 0.48, a
    # class's evidence moved by 0.94 against 1,283 allowed; at 0.25: scale 256,
    # 0.20 and 0.11, and 0.21 against 317. Per score,
    # the change against the family's log_density is under the bound; per
    # class, the change in log evidence against the NumPy oracle is under the
    # sum of the bounds over its members.
    instance = _instance()
    params = replace(
        instance.params,
        n_positions=instance.params.n_positions // _EVERY,
        covariate=None
        if instance.params.covariate is None
        else instance.params.covariate[::_EVERY],
    )
    observations = instance.observations[::_EVERY]
    labels = instance.labels
    covariate = params.covariate
    assert covariate is not None
    sides = _sides(params, TOTAL)
    values, exposure = observations[..., TOTAL], covariate[..., TOTAL]

    scale = rust.grid_scale(sides, values, exposure, tolerance)
    table, rows = rust._grid_rows(sides, params, values, exposure, scale)
    bound = rust._grid_bound(sides, values, exposure, scale)
    assert bound.max() <= tolerance
    for m, side in enumerate(sides):
        direct = side.log_density(
            torch.as_tensor(values.reshape(-1), dtype=torch.float64),
            covariate=torch.as_tensor(exposure.reshape(-1, 1)),
        ).numpy()
        # The bound is first order and exact arithmetic's; where it is tight
        # the two scores' own rounding is added, as the per-score tolerance.
        moved = np.abs(table[rows.reshape(-1), m, :] - direct)
        rounding = _PER_SCORE * np.abs(direct)
        assert (moved <= bound.reshape(-1, 1) + rounding).all()

    exact = rust.emission_rows(params, observations)
    evidence = np.empty(params.n_classes)
    oxisal.class_posteriors(
        np.ascontiguousarray(rows, dtype=np.uint32).reshape(-1),
        exact.success_rows.reshape(-1),
        np.ascontiguousarray(labels, dtype=np.int64),
        table.reshape(-1),
        exact.success_table.reshape(-1),
        np.ascontiguousarray(np.log(params.initial)).reshape(-1),
        np.ascontiguousarray(
            np.broadcast_to(
                np.log(params.transition),
                (params.n_classes, params.n_states, params.n_states),
            )
        ).reshape(-1),
        params.n_positions,
        params.graph.n_nodes,
        params.n_classes,
        params.n_states,
        np.empty(params.n_classes * params.n_positions * params.n_states),
        np.empty(
            params.n_classes
            * (params.n_positions - 1)
            * params.n_states
            * params.n_states
        ),
        evidence,
    )
    oracle = class_posteriors(params, observations, labels).log_evidence
    for m in range(params.n_classes):
        allowed = float(bound[:, labels == m].sum())
        assert abs(evidence[m] - oracle[m]) <= allowed


@pytest.mark.end2end
def test_em_with_the_grid_recovers_the_generating_parameters() -> None:
    # Block ascent on the Rust backend, the trial count on the grid, started
    # from means 30% high and dispersions 30% low; recovery is held to the
    # exact path's 5%. Two blocks: the measured fit reaches its value at the
    # first and holds it through eight.
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
        covariate_tolerance=_TOLERANCE,
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
@pytest.mark.parametrize("tolerance", [0.0, -1.0, float("nan"), float("inf")])
def test_a_tolerance_that_is_not_positive_and_finite_is_refused(
    tolerance: float,
) -> None:
    instance = _instance()

    with pytest.raises(ValueError, match="covariate_tolerance"):
        rust.emission_rows(
            instance.params, instance.observations, covariate_tolerance=tolerance
        )


@pytest.mark.smoke
def test_a_tolerance_on_the_numpy_oracle_is_refused() -> None:
    # The oracle has no grid, so a tolerance handed to it would be ignored.
    instance = _instance()

    with pytest.raises(ValueError, match="covariate_tolerance"):
        class_posteriors(
            instance.params,
            instance.observations,
            instance.labels,
            covariate_tolerance=_TOLERANCE,
        )


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("scale", "match"),
    [
        (2.0**13, "GRID_TABLE_CEILING"),
        (2.0**20, "uint32"),
        (0.0, "scale=0.0"),
        (float("inf"), "scale=inf"),
    ],
)
def test_a_grid_too_fine_to_tabulate_is_refused_before_it_is_built(
    scale: float, match: str
) -> None:
    # At 2**13 the exposure's range spans 9.7e4 codes, 1.1e8 rows and 3.5 GB;
    # at 2**20 the rows pass what a uint32 index carries. Both are refused
    # before anything is allocated, and every message names the scale.
    instance = _instance()
    params = instance.params
    assert params.covariate is not None

    with pytest.raises(ValueError, match=match) as refused:
        rust._grid_rows(
            _sides(params, TOTAL),
            params,
            instance.observations[..., TOTAL],
            params.covariate[..., TOTAL],
            scale,
        )
    assert "scale" in str(refused.value)


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
def test_without_a_tolerance_the_trial_count_is_tabulated_as_before() -> None:
    # The `None` path is #658's: the distinct trial counts, factorized, and the
    # row `count * n_distinct + code`, restated here against the arrays.
    instance = _instance()
    params, observations = instance.params, instance.observations
    covariate = params.covariate
    assert covariate is not None

    rows = rust.emission_rows(params, observations)

    distinct, codes = np.unique(
        covariate[..., SUCCESSES].reshape(-1), return_inverse=True
    )
    successes = observations[..., SUCCESSES]
    expected_rows = successes.astype(np.int64) * distinct.size + codes.reshape(
        successes.shape
    )
    assert np.array_equal(rows.success_rows, expected_rows)
    assert rows.success_table.shape[0] == (int(successes.max()) + 1) * distinct.size
    assert rows.exposure is not None
    assert np.array_equal(rows.total_rows, observations[..., TOTAL])
