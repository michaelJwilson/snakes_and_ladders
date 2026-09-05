"""The mixture fit, and what k-means++ actually buys.

Two abstractions meet here (issue #262). The component M step is the emission
family's, called with responsibilities where an HMM passes state posteriors,
so what is pinned is that it is *the same step* rather than a similar one. And
the initializer is the first that reads its objective's data, which is the
case issue #251 built the protocol for and had nothing to exercise it with.

The k-means++ guarantee is the reason this can be tested rather than admired:
Arthur & Vassilvitskii (2007) bound the *expected* seeding cost at
``8 (ln k + 2)`` times optimal, and in one dimension the optimal clustering is
computable exactly, so the bound has something to be checked against.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from snakes_and_ladders.emissions import GaussianEmission
from snakes_and_ladders.opt.fit import fit
from snakes_and_ladders.opt.hmm import align_by_key
from snakes_and_ladders.opt.initialize import Initializer
from snakes_and_ladders.opt.mixture import (
    GaussianMixtureObjective,
    KMeansPlusPlus,
    clustering_cost,
    expectation_maximization,
    kmeans_plus_plus,
    mixture_log_likelihood,
    optimal_clustering_cost,
    responsibilities,
    seeding_guarantee,
    uniform_seeds,
)
from snakes_and_ladders.opt.potts import PottsObjective, PottsParams, simulate_chains
from snakes_and_ladders.sim.mixture import MixtureParams, simulate_mixture

from tests._objective_checks import assert_gradient_matches_finite_differences

WEIGHTS = np.array([0.35, 0.65])
MEAN = np.array([-3.0, 3.0])
SCALE = np.array([1.0, 1.5])

#: A three-component fixture whose components are six standard deviations
#: apart, where a clustering is unambiguous and the optimal cost is worth
#: comparing against.
SEPARATED_MEAN = np.array([-6.0, 0.0, 6.0])


def _dataset(
    mean: np.ndarray = MEAN,
    scale: np.ndarray = SCALE,
    weights: np.ndarray = WEIGHTS,
    n_samples: int = 800,
    seed: int = 20260905,
) -> np.ndarray:
    """Observations from a mixture fixture."""
    params = MixtureParams(
        weights=weights,
        components=GaussianEmission(mean, scale, 1e-12),
        n_samples=n_samples,
        seed=seed,
        tolerance=1e-12,
    )
    return simulate_mixture(params).observations


def test_the_gradient_matches_central_differences() -> None:
    objective = GaussianMixtureObjective(_dataset(), 2)

    realized = assert_gradient_matches_finite_differences(
        objective, objective.initial(), step=1e-5, rtol=1e-6
    )

    assert realized <= 1e-6


def test_the_gradient_fit_and_expectation_maximization_reach_the_same_optimum() -> None:
    # EM shares no optimizer, no parameterization and no constraint map with
    # `fit` -- only the model. On a well-separated mixture they agree to the
    # printed precision of the likelihood, not merely to a tolerance.
    observations = _dataset()
    objective = GaussianMixtureObjective(observations, 2)

    result = fit(objective)
    estimate = objective.constrain(result.theta)
    em = expectation_maximization(
        observations,
        torch.tensor([0.5, 0.5], dtype=torch.float64),
        GaussianEmission([-1.0, 1.0], [1.0, 1.0], objective.variance_floor),
    )

    assert_allclose(-float(result.value), em.log_likelihood, rtol=1e-10)
    assert_allclose(
        torch.exp(estimate["log_weight"]).numpy(), em.weights.numpy(), atol=1e-6
    )
    assert_allclose(estimate["mean"].numpy(), em.components.mean.numpy(), atol=1e-5)
    assert_allclose(estimate["scale"].numpy(), em.components.scale.numpy(), atol=1e-5)


def test_the_fit_recovers_the_generating_mixture_up_to_the_label_permutation() -> None:
    observations = _dataset()
    objective = GaussianMixtureObjective(observations, 2)

    estimate = objective.constrain(fit(objective).theta)

    order = list(
        align_by_key(
            estimate["mean"].reshape(-1, 1), torch.as_tensor(MEAN).reshape(-1, 1)
        )
    )
    assert_allclose(estimate["mean"].numpy()[order], MEAN, atol=0.15)
    assert_allclose(estimate["scale"].numpy()[order], SCALE, atol=0.15)
    assert_allclose(
        torch.exp(estimate["log_weight"]).numpy()[order], WEIGHTS, atol=0.05
    )


def test_the_component_m_step_is_the_emission_family_s_own() -> None:
    # Asserted rather than assumed: the mixture's EM and a direct call into
    # `GaussianEmission.reestimate` on the same responsibilities must produce
    # the same numbers, because they *are* the same call. If this ever
    # diverges, the seam has acquired a mixture-specific branch.
    observations = _dataset(n_samples=200)
    values = torch.as_tensor(observations, dtype=torch.float64)
    components = GaussianEmission([-1.0, 1.0], [2.0, 2.0], 1e-9)
    log_weight = torch.log(torch.tensor([0.4, 0.6], dtype=torch.float64))

    posterior = responsibilities(values, log_weight, components)
    direct = components.reestimate(
        values.reshape(1, -1), posterior.reshape(1, *posterior.shape)
    ).emissions
    one_step = expectation_maximization(
        observations, torch.exp(log_weight), components, max_iterations=1
    )

    assert_allclose(one_step.components.mean.numpy(), direct.mean.numpy(), rtol=1e-15)
    assert_allclose(one_step.components.scale.numpy(), direct.scale.numpy(), rtol=1e-15)
    assert_allclose(one_step.weights.numpy(), posterior.mean(dim=0).numpy(), rtol=1e-15)


def test_the_responsibilities_are_a_distribution_over_components() -> None:
    values = torch.as_tensor(_dataset(n_samples=100), dtype=torch.float64)
    components = GaussianEmission(MEAN, SCALE, 1e-12)

    posterior = responsibilities(
        values, torch.log(torch.as_tensor(WEIGHTS)), components
    )

    assert posterior.shape == (100, 2)
    assert_allclose(posterior.sum(dim=1).numpy(), np.ones(100), rtol=1e-14)
    assert bool((posterior >= 0.0).all())


def test_a_collapsing_component_is_refused_rather_than_returned() -> None:
    # The unbounded likelihood transfers from the Gaussian HMM unchanged,
    # because it is the same family: a component's mean on one observation
    # with its scale going to zero diverges, so EM is refused rather than
    # allowed to report a converged fit at a degenerate optimum.
    observations = _dataset(n_samples=200)
    objective = GaussianMixtureObjective(observations, 2)
    floor = objective.variance_floor

    with pytest.raises(ValueError, match="unbounded as a variance goes to zero"):
        expectation_maximization(
            observations,
            torch.tensor([0.5, 0.5], dtype=torch.float64),
            GaussianEmission(
                [float(observations[0]), 0.0],
                [float(np.sqrt(floor)) / 100.0, 2.0],
                floor,
            ),
        )


def test_the_mixture_evidence_is_a_density_and_may_exceed_one() -> None:
    # Inherited from the components, and worth pinning here too: a caller who
    # assumed a probability would read a positive log-likelihood as a bug.
    narrow = GaussianEmission([0.0, 5.0], [0.01, 1.0], 1e-12)

    scored = mixture_log_likelihood(
        torch.zeros(1, dtype=torch.float64),
        torch.log(torch.tensor([0.99, 0.01], dtype=torch.float64)),
        narrow,
    )

    assert float(scored) > 0.0


def test_the_optimal_clustering_is_exact_where_it_can_be_checked_by_hand() -> None:
    # Two obvious clusters of three points each: the optimum splits them, and
    # the cost is the within-run sum of squares, 2 * (1 + 0 + 1) / ... written
    # out below rather than taken from the function under test.
    values = np.array([0.0, 1.0, 2.0, 10.0, 11.0, 12.0])

    assert_allclose(optimal_clustering_cost(values, 2), 2.0 + 2.0, rtol=1e-14)
    assert_allclose(optimal_clustering_cost(values, 1), values.var() * 6.0, rtol=1e-12)
    assert_allclose(optimal_clustering_cost(values, 6), 0.0, atol=1e-12)
    # And it really is the minimum over the alternatives, not just a partition.
    for centres in ([0.5, 11.0], [1.0, 11.5], [5.0, 11.0]):
        assert clustering_cost(values, np.array(centres)) >= optimal_clustering_cost(
            values, 2
        )
    with pytest.raises(ValueError, match="n_centres must lie in"):
        optimal_clustering_cost(values, 7)


def test_kmeans_plus_plus_stays_inside_its_published_guarantee() -> None:
    # Arthur & Vassilvitskii (2007), theorem 1.1: the *expected* seeding cost
    # is within `8 (ln k + 2)` of optimal, so the mean over replicates is what
    # is checked and a single draw would not be a test of it. Realized on this
    # fixture: mean ratio 2.91 against a bound of 24.79, worst draw 14.41.
    observations = _dataset(
        mean=SEPARATED_MEAN,
        scale=np.ones(3),
        weights=np.full(3, 1.0 / 3.0),
        n_samples=300,
    )
    optimal = optimal_clustering_cost(observations, 3)
    rng = np.random.default_rng(11)

    ratios = np.array(
        [
            clustering_cost(observations, kmeans_plus_plus(observations, 3, rng))
            / optimal
            for _ in range(200)
        ]
    )

    assert ratios.mean() <= seeding_guarantee(3)
    assert ratios.mean() <= 4.0


def test_kmeans_plus_plus_beats_uniform_seeding_on_the_cost_it_optimizes() -> None:
    # The paired control, kept beside the strategy rather than written inside
    # this test: a comparison whose baseline lives only in the test that wins
    # it is not a comparison. Realized: mean ratio 2.91 against 11.03, and the
    # worst uniform draw (58.1x optimal) is outside the k-means++ guarantee
    # while the worst k-means++ draw (14.4x) is inside it.
    observations = _dataset(
        mean=SEPARATED_MEAN,
        scale=np.ones(3),
        weights=np.full(3, 1.0 / 3.0),
        n_samples=300,
    )
    optimal = optimal_clustering_cost(observations, 3)

    ratios = {}
    for name, seeder in (("seeded", kmeans_plus_plus), ("uniform", uniform_seeds)):
        rng = np.random.default_rng(11)
        ratios[name] = np.array(
            [
                clustering_cost(observations, seeder(observations, 3, rng)) / optimal
                for _ in range(200)
            ]
        )

    assert ratios["seeded"].mean() < 0.5 * ratios["uniform"].mean()
    assert ratios["uniform"].max() > seeding_guarantee(3)


def test_the_seeding_advantage_does_not_reach_the_mixture_likelihood() -> None:
    # **The negative result, and it is the one worth having.** k-means++ is
    # 3.8x better on the cost it optimizes, and on this mixture that buys
    # nothing downstream: EM reaches the same optimum from either seeding, and
    # from the objective's own quantile start too. Measured over 200
    # replicates: 200/200 for k-means++, 195/200 for uniform.
    #
    # Harder fixtures do not reverse it, they only make both fail: at five
    # components 1.5 standard deviations apart, neither seeding reached the
    # best optimum found in 200 draws, and at five with unequal weights
    # uniform reached it 9 times in 150 against k-means++'s 3 -- noise, in the
    # direction opposite to the one a default change would need.
    #
    # So no default moves. `GaussianMixtureObjective.initial()` stays the
    # quantile start, and k-means++ lands as a strategy a caller may choose.
    observations = _dataset(
        mean=SEPARATED_MEAN,
        scale=np.ones(3),
        weights=np.full(3, 1.0 / 3.0),
        n_samples=300,
    )
    objective = GaussianMixtureObjective(observations, 3)
    pooled = float(np.std(observations))
    uniform_weights = torch.full((3,), 1.0 / 3.0, dtype=torch.float64)

    def _from(centres: np.ndarray) -> float:
        return expectation_maximization(
            observations,
            uniform_weights.clone(),
            GaussianEmission(
                np.sort(centres), np.full(3, pooled), objective.variance_floor
            ),
        ).log_likelihood

    best = -float(fit(objective).value)
    reached = {}
    for name, seeder in (("seeded", kmeans_plus_plus), ("uniform", uniform_seeds)):
        rng = np.random.default_rng(11)
        reached[name] = sum(
            _from(seeder(observations, 3, rng)) >= best - 1e-6 for _ in range(20)
        )

    assert reached["seeded"] == 20
    assert reached["uniform"] >= 18


def test_the_initializer_satisfies_the_protocol_and_seeds_the_objective() -> None:
    observations = _dataset(
        mean=SEPARATED_MEAN,
        scale=np.ones(3),
        weights=np.full(3, 1.0 / 3.0),
        n_samples=300,
    )
    objective = GaussianMixtureObjective(observations, 3)
    initializer = KMeansPlusPlus(4, np.random.default_rng(5))

    assert isinstance(initializer, Initializer)
    starts = initializer.starts(objective)

    assert len(starts) == 4
    for start in starts:
        assert start.shape == (objective.n_parameters,)
        means = objective.constrain(start)["mean"].numpy()
        # Sorted, so the permutation is fixed and two starts differ only where
        # the seeding differed.
        assert list(means) == sorted(means)
        assert observations.min() <= means.min()
        assert means.max() <= observations.max()
    assert not torch.equal(starts[0], starts[1])


def test_an_initializer_that_reads_the_data_refuses_an_objective_it_cannot_read() -> (
    None
):
    # The finding #251's Open Question 4 turns into: the protocol needed no
    # change, because a data-dependent strategy is model-*specific* rather
    # than protocol-incompatible. It takes an `Objective` like every other
    # initializer, and refuses the ones whose parameter vector it cannot
    # interpret -- which is why it lives beside the mixture rather than in
    # `opt/initialize.py` with the model-free strategies.
    params = PottsParams(
        n_states=3,
        chain_length=6,
        n_chains=8,
        coupling=0.4,
        field=np.array([0.1, -0.2, 0.1]),
        seed=3,
    )
    unrelated = PottsObjective(simulate_chains(params), params.n_states)

    with pytest.raises(TypeError, match="does not know what"):
        KMeansPlusPlus(1, np.random.default_rng(0)).starts(unrelated)


def test_a_mixture_needs_at_least_two_components() -> None:
    with pytest.raises(ValueError, match="n_components must be >= 2"):
        GaussianMixtureObjective(_dataset(n_samples=20), 1)
    with pytest.raises(ValueError, match="n_centres must lie in"):
        kmeans_plus_plus(np.arange(5.0), 6, np.random.default_rng(0))


def test_a_known_truth_round_trips_through_the_unconstrained_coordinates() -> None:
    objective = GaussianMixtureObjective(_dataset(n_samples=100), 2)

    estimate = objective.constrain(objective.theta_from_truth(WEIGHTS, MEAN, SCALE))

    assert_allclose(torch.exp(estimate["log_weight"]).numpy(), WEIGHTS, rtol=1e-13)
    assert_allclose(estimate["mean"].numpy(), MEAN, rtol=1e-13)
    assert_allclose(estimate["scale"].numpy(), SCALE, rtol=1e-13)
