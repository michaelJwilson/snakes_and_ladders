"""Deterministic annealing EM for the count-pair mixture (issue #903).

`opt.emission_mixture.expectation_maximization` takes `temperatures`: one E
and one M step at each, the responsibilities `softmax((log w + log p) / T)`,
then plain EM at one. Four referees, from the strongest: at `T = 1` the
tempered step is plain EM's step, so `None` and `[1.0]` reproduce the plain
fit bitwise; at `T -> inf` every responsibility is `1/K` and the M step is
the one-component fit to the pooled pairs; within one temperature the free
energy the tempered step ascends does not fall (Ueda & Nakano, 1998); and
from the `data` start on `emission_mixture/ci` the annealed fit is read
against the plain one from the same start.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from snakes_and_ladders.emissions import CountPairEmission
from snakes_and_ladders.opt.emission_mixture import (
    CountPairSeeding,
    EmissionMixtureFit,
    expectation_maximization,
    free_energy,
    uniform_start,
)
from snakes_and_ladders.opt.mixture import mixture_log_likelihood
from snakes_and_ladders.sample.schedule import ExponentialTempSchedule, ladder
from snakes_and_ladders.sim.emission_mixture import (
    EmissionMixtureParams,
    simulate_emission_mixture,
)
from snakes_and_ladders.sim.fixtures import fixture

#: The relative stopping rule `test_opt_emission_mixture.py` fits under.
EM_TOLERANCE = 1e-8

#: A temperature at which the tempered responsibilities are 1/K to 1e-12. The
#: joint log-densities at the data start span up to 51.2 nats across the
#: components of one pair of the ci draw, and the largest departure from 1/K
#: is about that span over 5T: measured 9.8e-6 at the ticket's 1e6, 9.8e-12 at
#: 1e12 and 9.8e-13 at 1e13.
HOT = 1e13

#: The schedule the notebook offers (issue #903): twenty steps from 8 to 1.
SCHEDULE = ladder(ExponentialTempSchedule(8.0, 1.0, 20))


def _instance() -> tuple[EmissionMixtureParams, np.ndarray, CountPairSeeding]:
    """The ci draw and the seam at the declared pooled shapes."""
    params = fixture("emission_mixture", "ci").params
    components = params.components
    assert isinstance(components, CountPairEmission)
    observations = simulate_emission_mixture(params).observations.astype(float)
    at = CountPairSeeding(
        float(components.total.dispersion.mean()),
        float(components.concentration.mean()),
        joint=components.joint,
    )
    return params, observations, at


def _recovery(posterior: torch.Tensor, labels: np.ndarray) -> float:
    """The fraction of pairs whose most responsible component generated them, under the best renaming."""
    assigned = posterior.argmax(dim=1).numpy()
    return max(
        float(np.mean(np.asarray(order)[assigned] == labels))
        for order in itertools.permutations(range(posterior.shape[1]))
    )


def _uniform(k: int) -> torch.Tensor:
    return torch.full((k,), 1.0 / k, dtype=torch.float64)


def _same(first: EmissionMixtureFit, second: EmissionMixtureFit) -> bool:
    """Every returned tensor equal bitwise, and the scalars equal."""
    tensors = [
        (first.weights, second.weights),
        (first.responsibilities, second.responsibilities),
        *zip(
            first.components.named_parameters().values(),
            second.components.named_parameters().values(),
            strict=True,
        ),
    ]
    return (
        all(torch.equal(a, b) for a, b in tensors)
        and first.log_likelihood == second.log_likelihood
        and first.iterations == second.iterations
        and first.termination == second.termination
    )


@pytest.fixture(scope="module")
def data_start() -> tuple[np.ndarray, CountPairEmission]:
    """The `data` start of the notebook on the ci draw: pairs drawn uniformly, seed 0."""
    params, observations, at = _instance()
    start = uniform_start(
        observations, params.n_components, at, np.random.default_rng(0)
    )
    assert isinstance(start, CountPairEmission)
    return observations, start


@pytest.fixture(scope="module")
def plain(data_start: tuple[np.ndarray, CountPairEmission]) -> EmissionMixtureFit:
    """Plain EM from the `data` start, read by two tests."""
    observations, start = data_start
    return expectation_maximization(
        observations, _uniform(start.n_states), start, tolerance=EM_TOLERANCE
    )


@pytest.mark.smoke
@pytest.mark.patch
def test_no_schedule_and_one_step_at_one_are_the_plain_fit_bitwise(
    data_start: tuple[np.ndarray, CountPairEmission], plain: EmissionMixtureFit
) -> None:
    observations, start = data_start
    k = start.n_states
    unset = expectation_maximization(
        observations, _uniform(k), start, tolerance=EM_TOLERANCE, temperatures=None
    )
    at_one = expectation_maximization(
        observations, _uniform(k), start, tolerance=EM_TOLERANCE, temperatures=[1.0]
    )
    assert _same(plain, unset)
    assert _same(plain, at_one)
    # The plain fit carries no tempered step; the one at one carries its
    # free energy, which at one is the log-likelihood at the start.
    assert plain.temperatures == plain.free_energies == ()
    values = torch.as_tensor(observations, dtype=torch.float64)
    assert at_one.free_energies == (
        float(mixture_log_likelihood(values, torch.log(_uniform(k)), start)),
    )


@pytest.mark.analytic
def test_a_hot_step_spreads_every_pair_evenly_and_fits_the_pooled_pairs(
    data_start: tuple[np.ndarray, CountPairEmission],
) -> None:
    # At T -> inf the tempered responsibilities are 1/K whatever the
    # components, so every component is re-estimated at one weight on every
    # pair: each is the one-component fit to the pooled pairs, and the
    # weights are 1/K.
    observations, start = data_start
    _, _, at = _instance()
    k = start.n_states
    hot = expectation_maximization(
        observations, _uniform(k), start, max_iterations=1, temperatures=[HOT]
    )
    assert hot.iterations == 1
    assert float((hot.responsibilities - 1.0 / k).abs().max()) < 1e-12
    assert float((hot.weights - 1.0 / k).abs().max()) < 1e-12
    values = torch.as_tensor(observations, dtype=torch.float64)
    one = at(observations[:1])
    pooled = one.reestimate(
        values, torch.ones((values.shape[0], 1), dtype=torch.float64)
    )
    assert pooled.converged
    for name, value in hot.components.named_parameters().items():
        reference = pooled.emissions.named_parameters()[name].expand_as(value)
        # Measured: the widest relative difference is 0 on this draw, the
        # inner solve reading weights 1/K where the pooled one reads ones.
        assert_allclose(value.numpy(), reference.numpy(), rtol=1e-9, err_msg=name)


@pytest.mark.analytic
def test_the_free_energy_does_not_fall_within_a_temperature(
    data_start: tuple[np.ndarray, CountPairEmission],
) -> None:
    # A tempered E step is the maximizer over q of the free energy's bound
    # and the M step raises the expected complete-data term, so at a fixed
    # temperature F_T is non-decreasing (Ueda & Nakano, 1998). Ten steps at
    # each of four temperatures; the M step's inner solve settles to its own
    # tolerance, so a fall is held to 1e-10 of the value.
    observations, start = data_start
    k = start.n_states
    steps = [8.0] * 10 + [4.0] * 10 + [2.0] * 10 + [1.0] * 10
    fit = expectation_maximization(
        observations,
        _uniform(k),
        start,
        max_iterations=len(steps),
        temperatures=steps,
    )
    assert fit.temperatures == tuple(steps)
    energies = np.asarray(fit.free_energies)
    for block in range(4):
        values = energies[10 * block : 10 * (block + 1)]
        falls = np.diff(values)
        assert bool((falls >= -1e-10 * np.abs(values[1:])).all()), (block, falls)
    # The free energy at T is the free_energy of the joint at the state each
    # step was handed; at the last step's state and temperature, recomputed.
    last = expectation_maximization(
        observations,
        _uniform(k),
        start,
        max_iterations=len(steps) - 1,
        temperatures=steps[:-1],
    )
    values_t = torch.as_tensor(observations, dtype=torch.float64)
    joint = torch.log(last.weights) + last.components.log_density(values_t)
    assert free_energy(joint, steps[-1]) == pytest.approx(energies[-1], rel=1e-12)


@pytest.mark.end2end
def test_annealing_from_the_data_start_is_read_against_plain_em(
    data_start: tuple[np.ndarray, CountPairEmission], plain: EmissionMixtureFit
) -> None:
    # The `data` start, the notebook's worst row, under both polishes, read
    # against the generating parameters' value on the draw (-7847.92) and
    # their recovery (0.956). Measured over data seeds 0 to 5: both reach
    # -7836.806 with recovery 0.953 from every seed, plain EM in 27 to 160
    # iterations and annealed EM in 56 to 58, the 20 tempered steps included.
    # On this draw annealing buys no higher maximum; it is not below plain EM
    # beyond EM's own tolerance, and both pass the generating parameters by
    # the maximum-likelihood excess.
    observations, start = data_start
    params, _, _ = _instance()
    k = start.n_states
    values = torch.as_tensor(observations, dtype=torch.float64)
    log_weight = torch.log(torch.as_tensor(params.weights, dtype=torch.float64))
    reference = float(mixture_log_likelihood(values, log_weight, params.components))
    labels = simulate_emission_mixture(params).labels
    annealed = expectation_maximization(
        observations,
        _uniform(k),
        start,
        tolerance=EM_TOLERANCE,
        temperatures=SCHEDULE,
    )
    assert annealed.temperatures == SCHEDULE
    assert annealed.log_likelihood >= plain.log_likelihood - EM_TOLERANCE * abs(
        plain.log_likelihood
    )
    for fit in (plain, annealed):
        assert reference < fit.log_likelihood < reference + 15.0
        assert _recovery(fit.responsibilities, labels) == pytest.approx(
            0.953, abs=0.001
        )


@pytest.mark.smoke
def test_a_temperature_is_positive_and_the_schedule_fits_the_budget(
    data_start: tuple[np.ndarray, CountPairEmission],
) -> None:
    observations, start = data_start
    k = start.n_states
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="positive and finite"):
            expectation_maximization(
                observations, _uniform(k), start, temperatures=[bad]
            )
    with pytest.raises(ValueError, match="do not fit"):
        expectation_maximization(
            observations, _uniform(k), start, max_iterations=1, temperatures=[2.0, 1.0]
        )
