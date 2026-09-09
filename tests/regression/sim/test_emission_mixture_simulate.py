"""The count-emission mixture simulator, against the closed forms it declares.

The Gaussian mixture's simulator is checked against the law of total variance
(``tests/regression/sim/test_mixture_simulate.py``); the same two checks apply
here, and the second is where a two-channel emission differs: the mixture's
moments are per channel, and the depth and the allele count each have their
own law of total variance over the component label.

The instances are built inline where the sample size is the variable a Monte
Carlo bound is checked over, and where the input is deliberately malformed.
The declared instance is read from the registry, never restated
(``sim/CLAUDE.md``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from snakes_and_ladders.emissions import CountPairEmission
from snakes_and_ladders.sim.emission_mixture import (
    EmissionMixtureParams,
    load_emission_mixture_params,
    simulate_emission_mixture,
)
from snakes_and_ladders.sim.fixtures import fixture

WEIGHTS = np.array([0.3, 0.7])
DISPERSION = np.array([5.0, 9.0])
MEAN = np.array([40.0, 120.0])
ALPHA = np.array([2.0, 7.0])
BETA = np.array([6.0, 3.0])
DRAWS = 200_000


def _params(n_samples: int = DRAWS, seed: int = 20260906) -> EmissionMixtureParams:
    """The two-component fixture, in the joint form."""
    return EmissionMixtureParams(
        weights=WEIGHTS,
        components=CountPairEmission(DISPERSION, MEAN, ALPHA, BETA, None, joint=True),
        n_samples=n_samples,
        seed=seed,
        tolerance=1e-12,
    )


@pytest.mark.structural
def test_the_component_labels_appear_at_their_declared_weights() -> None:
    # Standard error of a proportion at 200,000 draws is at most 0.0011, so
    # 0.005 is over four of them.
    dataset = simulate_emission_mixture(_params())

    frequencies = np.bincount(dataset.labels, minlength=2) / dataset.labels.size
    assert_allclose(frequencies, WEIGHTS, atol=0.005)


@pytest.mark.mathematical
def test_each_channel_matches_its_own_law_of_total_variance() -> None:
    # `E[Y] = sum_k w_k m_k` and
    # `Var[Y] = sum_k w_k (v_k + m_k**2) - E[Y]**2`, once per channel, with
    # `m_k` and `v_k` the component's own moments. The closed form is the
    # family's, so what this checks is the *mixing*: a simulator that drew the
    # right components in the wrong proportions would pass the frequency test
    # above and fail this one.
    dataset = simulate_emission_mixture(_params())
    components = _params().components
    assert isinstance(components, CountPairEmission)
    weights = WEIGHTS[:, None]

    expected_mean = (weights * components.mean.numpy()).sum(axis=0)
    expected_variance = (
        weights * (components.variance.numpy() + components.mean.numpy() ** 2)
    ).sum(axis=0) - expected_mean**2

    observed = dataset.observations.astype(float)
    assert_allclose(observed.mean(axis=0), expected_mean, rtol=0.01)
    assert_allclose(observed.var(axis=0), expected_variance, rtol=0.02)


@pytest.mark.structural
def test_a_seeded_instance_reproduces_its_draw_and_keeps_its_labels() -> None:
    # The generator rule (`sim/CLAUDE.md`): a fixture that names a seed and
    # passes no generator draws the same data every time, and the label that
    # produced each observation is retained, because a dataset without it
    # cannot referee a clustering.
    params = _params(n_samples=500)

    first = simulate_emission_mixture(params)
    second = simulate_emission_mixture(params)

    assert_allclose(first.observations, second.observations)
    assert_allclose(first.labels, second.labels)
    assert first.observations.shape == (500, CountPairEmission.N_CHANNELS)
    assert set(np.unique(first.labels)) <= {0, 1}


@pytest.mark.structural
def test_the_declared_instance_loads_and_draws_what_it_declares() -> None:
    # The registry's copy, not a restatement of it: the fixture file is the
    # instance, and this is the check that it reaches the simulator intact.
    declared = fixture("emission_mixture", "ci")
    params = declared.params

    dataset = simulate_emission_mixture(params)

    assert declared.model == "emission-mixture"
    assert params.n_components == 3
    assert dataset.observations.shape == (params.n_samples, 2)
    params.components.validate(dataset.observations)
    frequencies = np.bincount(dataset.labels, minlength=3) / params.n_samples
    # Standard error of a proportion at 900 draws is at most 0.017.
    assert_allclose(frequencies, params.weights, atol=0.05)


@pytest.mark.edge_case
def test_the_loader_refuses_a_family_or_a_form_it_cannot_build(tmp_path: Path) -> None:
    fields = (
        "seed: 1\nn_samples: 10\ntolerance: 0.1\nweights: [0.5, 0.5]\n"
        "dispersion: [1.0, 2.0]\nmean: [10.0, 20.0]\n"
        "alpha: [1.0, 2.0]\nbeta: [2.0, 1.0]\n"
    )
    cases = {
        "family: gaussian\n": "is not one of",
        "family: count-pair-joint\ntrials: [5, 5]\n": "trial count is the drawn total",
        "family: count-pair-independent\n": "needs 'trials'",
    }
    for tail, message in cases.items():
        path = tmp_path / "params.yaml"
        path.write_text(fields + tail)

        with pytest.raises(ValueError, match=message):
            load_emission_mixture_params(path)


@pytest.mark.edge_case
def test_weights_that_are_not_a_distribution_over_the_components_are_refused() -> None:
    components = CountPairEmission(DISPERSION, MEAN, ALPHA, BETA, None, joint=True)
    for weights, message in (
        (np.array([0.3, 0.3]), "weights sum to"),
        (np.array([0.0, 1.0]), "every weight must be positive"),
        (np.array([0.2, 0.3, 0.5]), "weights have shape"),
    ):
        with pytest.raises(ValueError, match=message):
            EmissionMixtureParams(
                weights=weights,
                components=components,
                n_samples=10,
                seed=1,
                tolerance=0.1,
            )
