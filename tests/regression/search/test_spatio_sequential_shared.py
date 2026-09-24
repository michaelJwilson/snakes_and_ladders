"""Emissions shared across classes, and count families seeded, in the coupled model (issue #933, R6).

`SpatioSequentialParams(shared_emissions=True)` holds one family for every
class, and `m_step` re-estimates it on every class's block pooled.
`seed_emissions(..., seed_counts=True)` seeds count families by
`Emission_Mixture++` in rate space. Referees: the pooled categorical M step
is the expected symbol counts summed over classes, by hand; one class shared
is one class unshared, bitwise; and a count-pair instance drawn under a
varying exposure and trial count with one family for both classes is
recovered.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from snakes_and_ladders.emissions import CategoricalEmission
from snakes_and_ladders.likelihood.spatio_sequential import (
    class_posteriors,
    labelled_log_likelihood,
)
from snakes_and_ladders.search.spatio_sequential import (
    fit_spatio_sequential,
    m_step,
    seed_emissions,
)
from snakes_and_ladders.sim.count_pairs import (
    IndependentCountPair,
    simulate_count_pairs,
)
from snakes_and_ladders.sim.fixtures import fixture
from snakes_and_ladders.sim.spatio_sequential import simulate_spatio_sequential


def _shared_categorical() -> tuple:  # type: ignore[type-arg]
    params = fixture("spatio_sequential", "ci").params
    shared = replace(
        params,
        emissions=(params.emissions[0],) * params.n_classes,
        shared_emissions=True,
    )
    data = simulate_spatio_sequential(shared, np.random.default_rng(933))
    return shared, data


@pytest.mark.critical
@pytest.mark.oracle
def test_the_pooled_m_step_is_the_symbol_counts_summed_over_classes() -> None:
    shared, data = _shared_categorical()
    labels = np.asarray(data.labels)
    posteriors = class_posteriors(shared, data.observations, labels)
    fitted = m_step(shared, data.observations, labels, posteriors)
    family = fitted.emissions[0]
    assert isinstance(family, CategoricalEmission)
    assert all(e is family for e in fitted.emissions)
    n_symbols = family.n_symbols
    counts = np.zeros((shared.n_states, n_symbols))
    for m in range(shared.n_classes):
        for node in np.flatnonzero(labels == m):
            for s in range(shared.n_positions):
                counts[:, int(data.observations[s, node])] += posteriors.posterior[m][s]
    np.testing.assert_allclose(
        family.matrix.numpy(), counts / counts.sum(axis=1, keepdims=True), rtol=1e-12
    )


@pytest.mark.critical
@pytest.mark.oracle
def test_one_class_shared_is_one_class_unshared() -> None:
    params = fixture("spatio_sequential", "ci").params
    single = replace(
        params,
        n_classes=1,
        initial=params.initial[:1],
        emissions=params.emissions[:1],
    )
    data = simulate_spatio_sequential(single, np.random.default_rng(1))
    labels = np.zeros(single.graph.n_nodes, dtype=np.int64)
    posteriors = class_posteriors(single, data.observations, labels)
    alone = m_step(single, data.observations, labels, posteriors).emissions[0]
    shared = m_step(
        replace(single, shared_emissions=True), data.observations, labels, posteriors
    ).emissions[0]
    assert torch.equal(
        alone.named_parameters()["log_emission"],
        shared.named_parameters()["log_emission"],
    )


@pytest.mark.smoke
def test_shared_emissions_that_differ_are_refused() -> None:
    params = fixture("spatio_sequential", "ci").params
    assert params.n_classes > 1
    with pytest.raises(ValueError, match="one family for every class"):
        replace(params, shared_emissions=True)


def _count_instance() -> tuple:  # type: ignore[type-arg]
    """The count-pair fixture at 200 positions, one family for both classes, a varying covariate."""
    declared = fixture("spatio_sequential_counts", "ci").params
    model = replace(declared.model, n_positions=200)
    rng = np.random.default_rng(933)
    shape = (model.n_positions, model.graph.n_nodes)
    covariate = np.stack(
        [rng.uniform(0.5, 2.0, shape), rng.integers(20, 60, shape).astype(float)],
        axis=-1,
    )
    shared = replace(
        model,
        emissions=(model.emissions[0],) * model.n_classes,
        shared_emissions=True,
        covariate=covariate,
    )
    return shared, simulate_count_pairs(replace(declared, model=shared))


@pytest.mark.oracle
def test_count_seeding_places_components_on_rate_space_rows() -> None:
    shared, instance = _count_instance()
    kept = seed_emissions(shared, instance.observations, np.random.default_rng(1))
    assert kept.emissions == shared.emissions
    seeded = seed_emissions(
        shared, instance.observations, np.random.default_rng(1), seed_counts=True
    )
    family = seeded.emissions[0]
    assert isinstance(family, IndependentCountPair)
    assert all(e is family for e in seeded.emissions)
    rates = instance.observations[..., 0] / shared.covariate[..., 0]
    assert bool(np.isin(family.total.mean.numpy(), rates.reshape(-1)).all())


@pytest.mark.end2end
def test_a_shared_count_pair_model_is_recovered_under_a_varying_covariate() -> None:
    # Planted: total means 24 and 72 per unit exposure, dispersions 6 and 12,
    # allele rates 0.22 and 0.57, one family for both classes, exposures in
    # [0.5, 2] and trials in [20, 60). Seeded in rate space and fitted for four
    # blocks from the planted labels, each state lands within 10% on its mean
    # and 15% on its dispersion, and the fit's joint is at least the truth's.
    shared, instance = _count_instance()
    seeded = seed_emissions(
        shared, instance.observations, np.random.default_rng(1), seed_counts=True
    )
    fit = fit_spatio_sequential(
        seeded,
        instance.observations,
        np.random.default_rng(2),
        n_blocks=4,
        labels=instance.labels,
    )
    family = fit.params.emissions[0]
    assert isinstance(family, IndependentCountPair)
    order = np.argsort(family.total.mean.numpy())
    np.testing.assert_allclose(family.total.mean.numpy()[order], [24.0, 72.0], rtol=0.1)
    np.testing.assert_allclose(
        family.total.dispersion.numpy()[order], [6.0, 12.0], rtol=0.15
    )
    np.testing.assert_allclose(
        (family.successes.alpha / family.successes.concentration).numpy()[order],
        [5.5 / 25.0, 14.25 / 25.0],
        atol=0.02,
    )
    truth = labelled_log_likelihood(shared, instance.observations, instance.labels)
    assert fit.log_likelihoods[-1] >= truth
