"""The covariate grid, conserved in the sandbox, on the continuous exposure (issue #1064).

No family reaches the grid: the negative binomial's exposure factors and the
beta-binomial's trial count is an integer with exact layouts
(``sal.sandbox.covariate_grid``). What it referees is held here, on the one
continuous covariate the model carries: each gridded score within its stated
bound of the family's ``log_density``, each class's log evidence within the
sum of its members' bounds of the NumPy oracle, and a power-of-two grid
bitwise; the log grid, which codes ``ln c``, likewise, with its level counts
read against the linear grid's at one tolerance. The refusals are pinned beside
them. Fixture: ``spatio_sequential_counts_covariate``.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from sal import oxisal
from sal.emissions import EmissionFamily
from sal.likelihood import spatio_sequential_rust as rust
from sal.likelihood.spatio_sequential import class_posteriors
from sal.sandbox import covariate_grid as grid
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

#: Relative agreement of one exact score with the family's ``log_density``,
#: added to the grid's first-order bound where it is tight: the 512 ulp the
#: live per-score check declares
#: (`test_spatio_sequential_rust_covariate_grid.py`).
_PER_SCORE = 512 * np.finfo(np.float64).eps

#: A tolerance a grid is asked for, refused on the integer trial count.
_TOLERANCE = 1e-3

#: Positions of the ci instance the grid checks read, every tenth: 100
#: positions and 6,400 observations.
_EVERY = 10


def _instance() -> CountPairInstance:
    return fine_instance(fixture(PROBLEM, "ci").path)


def _sides(params: SpatioSequentialParams, channel: int) -> list[EmissionFamily]:
    """One channel's family per class."""
    sides: list[EmissionFamily] = []
    for family in params.emissions:
        assert isinstance(family, IndependentCountPair)
        sides.append(family.total if channel == TOTAL else family.successes)
    return sides


#: A grid's scale from a tolerance, its table and rows at a scale, and its
#: per-observation bound at a scale: ``linear`` codes ``c``, ``log`` codes
#: ``ln c``.
GRIDS = {
    "linear": (grid.grid_scale, grid._grid_rows, grid._grid_bound),
    "log": (grid.log_grid_scale, grid._log_grid_rows, grid._log_grid_bound),
}


@pytest.mark.smoke
@pytest.mark.parametrize("kind", GRIDS)
def test_a_tolerance_on_the_integer_trial_count_is_refused(kind: str) -> None:
    # A table over an integer's values is exact, so no grid is asked for.
    instance = _instance()
    params, observations = instance.params, instance.observations
    covariate = params.covariate
    assert covariate is not None
    scale_of = GRIDS[kind][0]

    with pytest.raises(ValueError, match="integer"):
        scale_of(
            _sides(params, SUCCESSES),
            observations[..., SUCCESSES],
            covariate[..., SUCCESSES],
            _TOLERANCE,
        )


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

    table, rows = grid._grid_rows(sides, params, values, exposure, 4.0)

    for m, side in enumerate(sides):
        direct = side.log_density(
            torch.as_tensor(values.reshape(-1), dtype=torch.float64),
            covariate=torch.as_tensor(exposure.reshape(-1, 1)),
        ).numpy()
        assert np.array_equal(table[rows.reshape(-1), m, :], direct)


@pytest.mark.oracle
@pytest.mark.parametrize("tolerance", [1.0, 0.25])
@pytest.mark.parametrize("kind", GRIDS)
def test_a_continuous_covariate_on_the_grid_is_within_its_stated_bound(
    kind: str, tolerance: float
) -> None:
    # The grid applied to the continuous exposure, where the factorization is
    # what runs: the one continuous covariate the model carries. Measured on
    # the linear grid at tolerance 1.0: scale 64, largest bound 0.81 and
    # largest change 0.48, a class's evidence moved by 0.94 against 1,283
    # allowed; at 0.25: scale 256, 0.20 and 0.11, and 0.21 against 317. On
    # the log grid at 1.0: scale 32, 0.68 and 0.63, and 0.95 against 1,101;
    # at 0.25: scale 128, 0.17 and 0.15, and 0.47 against 276. Per score, the
    # change against the family's log_density is under the bound; per
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

    scale_of, rows_of, bound_of = GRIDS[kind]

    scale = scale_of(sides, values, exposure, tolerance)
    table, rows = rows_of(sides, params, values, exposure, scale)
    bound = bound_of(sides, values, exposure, scale)
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
        **({} if exact.trials is None else exact.trials.arguments()),
    )
    oracle = class_posteriors(params, observations, labels).log_evidence
    for m in range(params.n_classes):
        allowed = float(bound[:, labels == m].sum())
        assert abs(evidence[m] - oracle[m]) <= allowed


@pytest.mark.smoke
@pytest.mark.parametrize("tolerance", [0.0, -1.0, float("nan"), float("inf")])
@pytest.mark.parametrize("kind", GRIDS)
def test_a_tolerance_that_is_not_positive_and_finite_is_refused(
    kind: str, tolerance: float
) -> None:
    instance = _instance()
    params = instance.params
    assert params.covariate is not None
    scale_of = GRIDS[kind][0]

    with pytest.raises(ValueError, match="covariate_tolerance"):
        scale_of(
            _sides(params, TOTAL),
            instance.observations[..., TOTAL],
            params.covariate[..., TOTAL],
            tolerance,
        )


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("scale", "match"),
    [
        (2.0**13, "COVARIATE_TABLE_CEILING"),
        (2.0**20, "uint32"),
        (0.0, "scale=0.0"),
        (float("inf"), "scale=inf"),
    ],
)
@pytest.mark.parametrize("kind", GRIDS)
def test_a_grid_too_fine_to_tabulate_is_refused_before_it_is_built(
    kind: str, scale: float, match: str
) -> None:
    # At 2**13 the exposure's range spans 9.7e4 linear codes, 1.1e8 rows and
    # 3.5 GB, and 3.5e4 log codes and 1.3 GB; at 2**20 either grid's rows pass
    # what a uint32 index carries. Both are refused before anything is
    # allocated, and every message names the scale.
    instance = _instance()
    params = instance.params
    assert params.covariate is not None
    rows_of = GRIDS[kind][1]

    with pytest.raises(ValueError, match=match) as refused:
        rows_of(
            _sides(params, TOTAL),
            params,
            instance.observations[..., TOTAL],
            params.covariate[..., TOTAL],
            scale,
        )
    assert "scale" in str(refused.value)


@pytest.mark.oracle
def test_a_covariate_on_the_log_grid_is_tabulated_bitwise() -> None:
    # An exposure already exp(k / 4) is coded k at scale 4 and tabulated at
    # exp((low + i) / 4), the same float64 operations on the same integer:
    # the level is the observation's own exposure, so every tabulated score
    # is the family's log_density there bit for bit. exp is not exact, and
    # this does not need it to be.
    instance = _instance()
    params = instance.params
    assert params.covariate is not None
    exposure = np.exp(np.rint(np.log(params.covariate[::_EVERY, :, TOTAL]) * 4.0) / 4.0)
    values = instance.observations[::_EVERY, :, TOTAL]
    sides = _sides(params, TOTAL)

    table, rows = grid._log_grid_rows(sides, params, values, exposure, 4.0)

    assert not grid._log_grid_bound(sides, values, exposure, 4.0).any()
    for m, side in enumerate(sides):
        direct = side.log_density(
            torch.as_tensor(values.reshape(-1), dtype=torch.float64),
            covariate=torch.as_tensor(exposure.reshape(-1, 1)),
        ).numpy()
        assert np.array_equal(table[rows.reshape(-1), m, :], direct)


@pytest.mark.oracle
def test_an_unobserved_exposure_is_the_log_grids_own_level() -> None:
    # Zero marks the count unobserved and has no logarithm: it is coded on a
    # level of its own at 0.0, where the family scores log 1 at every state,
    # and its bound is zero. The observed exposures keep their codes, moved
    # up by one.
    instance = _instance()
    params = instance.params
    assert params.covariate is not None
    exposure = params.covariate[::_EVERY, :, TOTAL].copy()
    exposure[:, ::7] = 0.0
    values = instance.observations[::_EVERY, :, TOTAL]
    sides = _sides(params, TOTAL)
    observed_only = params.covariate[::_EVERY, :, TOTAL]

    table, rows = grid._log_grid_rows(sides, params, values, exposure, 32.0)
    codes, levels = grid._log_grid(exposure, 32.0)
    every, _ = grid._log_grid(observed_only, 32.0)

    unobserved = exposure == 0.0
    assert levels[0] == 0.0
    assert not codes[unobserved].any()
    assert np.array_equal(codes[~unobserved], every[~unobserved] + 1)
    assert not grid._log_grid_bound(sides, values, exposure, 32.0)[unobserved].any()
    assert not table[rows[unobserved].reshape(-1)].any()


@pytest.mark.smoke
def test_a_negative_exposure_is_refused_by_the_log_grid() -> None:
    instance = _instance()
    params = instance.params
    assert params.covariate is not None
    exposure = params.covariate[..., TOTAL].copy()
    exposure[0, 0] = -1.0

    with pytest.raises(ValueError, match="non-negative"):
        grid.log_grid_scale(
            _sides(params, TOTAL),
            instance.observations[..., TOTAL],
            exposure,
            1.0,
        )


@pytest.mark.experiment
@pytest.mark.parametrize("tolerance", [1.0, 0.25, 0.05])
def test_the_log_grid_codes_fewer_levels_than_the_linear_at_one_tolerance(
    tolerance: float,
) -> None:
    # The exposure spans 0.128 to 9.24, a ratio of 72: the log grid spends a
    # level per ratio and the linear per step. Measured levels, linear against
    # log, and the largest |Delta log p| each moved: at 1.0, 581 against 129,
    # 0.48 against 0.63; at 0.25, 2,324 against 516, 0.11 against 0.15; at
    # 0.05, 9,289 against 2,059, 0.034 against 0.037. The log grid codes 4.5x
    # fewer levels at every tolerance, and moves each score further within it.
    instance = _instance()
    params = instance.params
    assert params.covariate is not None
    values = instance.observations[::_EVERY, :, TOTAL]
    exposure = params.covariate[::_EVERY, :, TOTAL]
    sides = _sides(params, TOTAL)
    counts = torch.as_tensor(values.reshape(-1), dtype=torch.float64)
    exact = [
        side.log_density(counts, covariate=torch.as_tensor(exposure.reshape(-1, 1)))
        for side in sides
    ]
    n_levels, moved = {}, {}
    for kind, code in (("linear", grid._grid), ("log", grid._log_grid)):
        scale = GRIDS[kind][0](sides, values, exposure, tolerance)
        codes, levels = code(exposure, scale)
        at_grid = torch.as_tensor(levels[codes.reshape(-1)].reshape(-1, 1))
        n_levels[kind] = levels.size
        moved[kind] = max(
            float((side.log_density(counts, covariate=at_grid) - want).abs().max())
            for side, want in zip(sides, exact, strict=True)
        )

    assert max(moved.values()) <= tolerance
    assert 4 * n_levels["log"] < n_levels["linear"]
