"""The mixture fit, and what k-means++ actually buys.

Issue #262. The component M step is the emission family's, called with
responsibilities, so it is pinned as the same step; the initializer is the
first to read its objective's data (#251). Arthur & Vassilvitskii (2007) bound
the expected seeding cost at ``8 (ln k + 2)`` times optimal, and the 1-D
optimum is computable exactly, so the bound is checked.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from numpy.typing import ArrayLike
from sal.backend import Backend
from sal.cost import Cost
from sal.emissions import GaussianEmission, Reestimate
from sal.likelihood.mixture_assignments import (
    enumerate_mixture_assignments,
)
from sal.opt.budget import Budget, Outcome, compare
from sal.opt.em import EM, EmConfig
from sal.opt.fit import fit
from sal.opt.hmm import align_by_key
from sal.opt.initialize import Initializer
from sal.opt.mixture import (
    GaussianMixtureObjective,
    KMeansPlusPlus,
    clustering_cost,
    expectation_maximization,
    kmeans_plus_plus,
    mixture_log_likelihood,
    optimal_clustering_cost,
    responsibilities_torch,
    seeding_guarantee,
    uniform_seeds,
)
from sal.opt.potts import PottsObjective
from sal.sim.mixture import MixtureParams, simulate_mixture
from sal.sim.potts_chain import PottsParams, simulate_chains

from tests._objective_checks import assert_gradient_matches_finite_differences
from tests._rows import every_value

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


@pytest.mark.analytic
def test_the_gradient_matches_central_differences() -> None:
    objective = GaussianMixtureObjective(_dataset(), 2)

    realized = assert_gradient_matches_finite_differences(
        objective, objective.initial(), step=1e-5, rtol=1e-6
    )

    assert realized <= 1e-6


@pytest.mark.analytic
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


@pytest.mark.end2end
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


@pytest.mark.oracle
def test_the_component_m_step_is_the_emission_family_s_own() -> None:
    # The same call, so the same numbers; the compiled route streams its own
    # moments, pinned within 1e-10 (issue #986).
    observations = _dataset(n_samples=200)
    values = torch.as_tensor(observations, dtype=torch.float64)
    components = GaussianEmission([-1.0, 1.0], [2.0, 2.0], 1e-9)
    log_weight = torch.log(torch.tensor([0.4, 0.6], dtype=torch.float64))

    posterior = responsibilities_torch(values, log_weight, components)
    direct = components.reestimate(
        values.reshape(1, -1), posterior.reshape(1, *posterior.shape)
    ).emissions
    one_step = expectation_maximization(
        observations,
        torch.exp(log_weight),
        components,
        backend=Backend.PYTHON,
        config=replace(EM, max_iterations=1),
    )

    assert_allclose(one_step.components.mean.numpy(), direct.mean.numpy(), rtol=1e-15)
    assert_allclose(one_step.components.scale.numpy(), direct.scale.numpy(), rtol=1e-15)
    assert_allclose(one_step.weights.numpy(), posterior.mean(dim=0).numpy(), rtol=1e-15)


@pytest.mark.smoke
def test_the_responsibilities_are_a_distribution_over_components() -> None:
    values = torch.as_tensor(_dataset(n_samples=100), dtype=torch.float64)
    components = GaussianEmission(MEAN, SCALE, 1e-12)

    posterior = responsibilities_torch(
        values, torch.log(torch.as_tensor(WEIGHTS)), components
    )

    assert posterior.shape == (100, 2)
    assert_allclose(posterior.sum(dim=1).numpy(), np.ones(100), rtol=1e-14)
    assert bool((posterior >= 0.0).all())


@pytest.mark.smoke
def test_a_collapsing_component_is_refused_rather_than_returned() -> None:
    # The unbounded likelihood transfers from the Gaussian HMM unchanged,
    # being the same family: a component's mean on one observation with its
    # scale going to zero diverges, so EM is refused rather than reporting a
    # converged fit at a degenerate optimum.
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


@pytest.mark.analytic
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


@pytest.mark.oracle
def test_the_optimal_clustering_is_exact_where_it_can_be_checked_by_hand() -> None:
    # Two clusters of three points: the optimum splits them, and the cost is
    # the within-run sum of squares, written out below rather than taken from
    # the function under test.
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


@pytest.mark.smoke
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


@pytest.mark.smoke
def test_kmeans_plus_plus_beats_uniform_seeding_on_the_cost_it_optimizes() -> None:
    # The paired control: mean ratio 2.91 against 11.03; worst uniform 58.1x
    # optimal (outside the guarantee), worst k-means++ 14.4x (inside).
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


@pytest.mark.smoke
def test_the_seeding_advantage_does_not_reach_the_mixture_likelihood() -> None:
    # The negative result: 3.8x better on its cost, nothing downstream (200/200
    # against 195/200 over 200 replicates). Harder fixtures make both fail (five
    # at 1.5 sd: neither; unequal weights: 9/150 uniform, 3/150 k-means++). The
    # quantile start stays default; k-means++ is a strategy a caller may choose.
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

    def seeded(data: np.ndarray, _budget: Budget, rng: np.random.Generator) -> Outcome:
        return Outcome(-_from(kmeans_plus_plus(data, 3, rng)), 1)

    def uniform(data: np.ndarray, _budget: Budget, rng: np.random.Generator) -> Outcome:
        return Outcome(-_from(uniform_seeds(data, 3, rng)), 1)

    # Twenty replicates of one dataset, each its own instance so each draws
    # its own stream; one EM fit per seeding is the budget (issue #281).
    result = compare(
        {"seeded": seeded, "uniform": uniform},
        [observations] * 20,
        Budget(Cost.FITS, 1),
        seeds=(11,),
        workers=1,
        known=[-best] * 20,
    )
    reached = result.hits(tolerance=1e-6)

    assert reached["seeded"] == 20, reached
    assert reached["uniform"] >= 18, reached


@pytest.mark.smoke
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


@pytest.mark.smoke
def test_an_initializer_that_reads_the_data_refuses_an_objective_it_cannot_read() -> (
    None
):
    # #251's Open Question 4: no protocol change; a data-dependent strategy is
    # model-specific, refuses objectives it cannot interpret, and lives here.
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


@pytest.mark.smoke
def test_a_mixture_needs_at_least_two_components() -> None:
    with pytest.raises(ValueError, match="n_components must be >= 2"):
        GaussianMixtureObjective(_dataset(n_samples=20), 1)
    with pytest.raises(ValueError, match="n_centres must lie in"):
        kmeans_plus_plus(np.arange(5.0), 6, np.random.default_rng(0))


@pytest.mark.end2end
def test_a_known_truth_round_trips_through_the_unconstrained_coordinates() -> None:
    objective = GaussianMixtureObjective(_dataset(n_samples=100), 2)

    estimate = objective.constrain(objective.theta_from_truth(WEIGHTS, MEAN, SCALE))

    assert_allclose(torch.exp(estimate["log_weight"]).numpy(), WEIGHTS, rtol=1e-13)
    assert_allclose(estimate["mean"].numpy(), MEAN, rtol=1e-13)
    assert_allclose(estimate["scale"].numpy(), SCALE, rtol=1e-13)


#: Two components at three sd, sixteen observations: ``2 ** 16 = 65,536``
#: assignments, inside ``MAX_ENUMERABLE_CONFIGURATIONS``; the declared fixture
#: is ``5 ** 500`` (issue #393).
ENUMERABLE_SAMPLES = 16
ENUMERABLE_WEIGHTS = np.array([0.4, 0.6])
ENUMERABLE_MEAN = np.array([-3.0, 3.0])
ENUMERABLE_SCALE = np.array([1.0, 1.0])


def _enumerable_mixture() -> tuple[np.ndarray, GaussianEmission]:
    """The enumerable instance's observations and its generating components."""
    components = GaussianEmission(ENUMERABLE_MEAN, ENUMERABLE_SCALE, 1e-12)
    return (
        _dataset(
            mean=ENUMERABLE_MEAN,
            scale=ENUMERABLE_SCALE,
            weights=ENUMERABLE_WEIGHTS,
            n_samples=ENUMERABLE_SAMPLES,
            seed=20260908,
        ),
        components,
    )


@pytest.mark.oracle
def test_the_evidence_and_the_e_step_match_the_enumerated_assignments() -> None:
    # The sum over 65,536 assignments uses no factorization, catching a wrong
    # normalization axis, broadcast or shared shift. Realized: evidence 1.2e-16
    # relative, responsibilities 4.4e-16, at truth and at the EM fixed point.
    observations, components = _enumerable_mixture()
    values = torch.as_tensor(observations, dtype=torch.float64)
    objective = GaussianMixtureObjective(observations, 2)
    fitted = expectation_maximization(
        observations, torch.as_tensor(ENUMERABLE_WEIGHTS), components
    )

    for weights, family in (
        (torch.as_tensor(ENUMERABLE_WEIGHTS), components),
        (fitted.weights, fitted.components),
    ):
        exact = enumerate_mixture_assignments(weights.numpy(), family, observations)
        log_weight = torch.log(weights)

        assert_allclose(
            float(mixture_log_likelihood(values, log_weight, family)),
            exact.log_evidence,
            rtol=1e-12,
        )
        assert_allclose(
            responsibilities_torch(values, log_weight, family).numpy(),
            exact.responsibilities,
            atol=1e-12,
        )

    # The objective the gradient fit descends is the same number negated, so
    # the enumeration referees it at whatever point it is asked about.
    theta = objective.theta_from_truth(
        ENUMERABLE_WEIGHTS, ENUMERABLE_MEAN, ENUMERABLE_SCALE
    )
    truth = enumerate_mixture_assignments(ENUMERABLE_WEIGHTS, components, observations)
    assert_allclose(float(objective(theta)), -truth.log_evidence, rtol=1e-12)
    # A density, not a probability: the sign is not an accident to assert past.
    assert truth.log_evidence < 0.0
    assert abs(truth.responsibilities.sum(axis=1) - 1.0).max() < 1e-12


@pytest.mark.oracle
def test_the_seeded_start_lands_in_the_enumerated_maximum_posterior_assignment() -> (
    None
):
    # The basin is exact here: k-means++ reaches the enumerated maximum-posterior
    # assignment on 20 of 20 seeds, uniform on 10 of 20; responsibilities agree
    # with the enumeration to 6.8e-15.
    observations, components = _enumerable_mixture()
    objective = GaussianMixtureObjective(observations, 2)
    fitted = expectation_maximization(
        observations, torch.as_tensor(ENUMERABLE_WEIGHTS), components
    )
    target = enumerate_mixture_assignments(
        fitted.weights.numpy(), fitted.components, observations
    ).assignment

    def enumerated_at(theta: torch.Tensor) -> tuple[np.ndarray, float]:
        named = objective.constrain(theta)
        family = objective.components(theta)
        exact = enumerate_mixture_assignments(
            torch.exp(named["log_weight"]).numpy(), family, observations
        )
        drift = np.abs(
            responsibilities_torch(
                torch.as_tensor(observations, dtype=torch.float64),
                named["log_weight"],
                family,
            ).numpy()
            - exact.responsibilities
        ).max()
        return exact.assignment, float(drift)

    seeded = uniform = 0
    for seed in range(20):
        start = KMeansPlusPlus(1, np.random.default_rng(seed)).starts(objective)[0]
        assignment, drift = enumerated_at(start)
        assert drift < 1e-12, drift
        seeded += int(np.array_equal(assignment, target))

        cold = objective.theta_from_centres(
            torch.as_tensor(
                np.sort(uniform_seeds(observations, 2, np.random.default_rng(seed)))
            )
        )
        uniform += int(np.array_equal(enumerated_at(cold)[0], target))

    assert seeded == 20, seeded
    assert uniform <= 15, uniform


#: (observations, clusters, seed): ``3 ** 10`` = 59,049 and ``2 ** 12`` = 4,096
#: assignments; the ``O(n ** 2 k)`` dynamic program does not care.
ENUMERATED_CLUSTERINGS = ((10, 3, 734), (12, 2, 11), (9, 3, 5))

#: The weights the equal-weight assumption is tilted by, and the instance of
#: :data:`ENUMERATED_CLUSTERINGS` whose maximum-posterior assignment moves
#: under them.
TILTED_WEIGHTS = ((0.05, 0.15, 0.8), (0.02, 0.49, 0.49))
TILTED_INSTANCE = (9, 3, 5)

#: The two-dimensional instance the contiguity argument is refuted on: eight
#: points, two clusters, sorted by their first coordinate.
NON_CONTIGUOUS_SEED = 1


def _assignment_matrix(n_samples: int, n_centres: int) -> np.ndarray:
    """Every assignment, as the digits of its index in base ``n_centres``."""
    place = n_centres ** np.arange(n_samples - 1, -1, -1, dtype=np.int64)
    return (
        np.arange(n_centres**n_samples, dtype=np.int64)[:, None] // place
    ) % n_centres


def _within_cluster_cost(values: np.ndarray, assignments: np.ndarray) -> np.ndarray:
    """Each assignment's sum of squares about its own clusters' means.

    The definition: empty clusters contribute nothing; any channel axis is summed.
    """
    n_centres = int(assignments.max()) + 1
    rows = values.reshape(values.shape[0], -1)
    membership = assignments[:, :, None] == np.arange(n_centres)
    counts = membership.sum(axis=1)
    totals = np.einsum("mnk,nc->mkc", membership.astype(np.float64), rows)
    squares = (membership * (rows**2).sum(axis=1)[None, :, None]).sum(axis=1)
    spread = squares - np.where(
        counts > 0, (totals**2).sum(axis=2) / np.maximum(counts, 1), 0.0
    )
    return np.asarray(spread.sum(axis=1))


def _enumerated_optimum(
    n_samples: int, n_centres: int, seed: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """One drawn instance, the assignment of least cost, and that cost."""
    values = np.sort(np.random.default_rng(seed).normal(scale=2.0, size=n_samples))
    assignments = _assignment_matrix(n_samples, n_centres)
    costs = _within_cluster_cost(values, assignments)
    return values, assignments[int(costs.argmin())], float(costs.min())


def _is_contiguous(partition: np.ndarray) -> bool:
    """Whether every cluster of ``partition`` is a run of consecutive indices."""
    return all(
        np.all(np.diff(np.flatnonzero(partition == cluster)) == 1)
        for cluster in range(int(partition.max()) + 1)
        if bool((partition == cluster).any())
    )


@pytest.mark.oracle
@pytest.mark.critical
def test_the_optimal_clustering_cost_is_the_minimum_over_the_enumerated_assignments() -> (
    None
):
    # The rung below (#734): enumeration over ``k ** n`` assignments uses no
    # ordering argument: 0.0, 2.8e-16, 1.6e-16 relative (1e-12). The minimizer
    # is contiguous on all three, asserted; it is the enumerated MAP of a
    # mixture at its cluster means at equal weights and shared scale. At
    # weights (0.05, 0.15, 0.8) and (0.02, 0.49, 0.49) the nine-observation
    # MAP departs from it, while the ten-observation one absorbs a ratio of
    # 16; both asserted.
    for n_samples, n_centres, seed in ENUMERATED_CLUSTERINGS:
        values, partition, cost = _enumerated_optimum(n_samples, n_centres, seed)

        assert_allclose(optimal_clustering_cost(values, n_centres), cost, rtol=1e-12)
        assert _is_contiguous(partition), partition

        means = np.array(
            [values[partition == cluster].mean() for cluster in range(n_centres)]
        )
        family = GaussianEmission(means, np.ones(n_centres), 1e-12)
        enumerated = enumerate_mixture_assignments(
            np.full(n_centres, 1.0 / n_centres), family, values
        )
        np.testing.assert_array_equal(enumerated.assignment, partition)

        for weights in TILTED_WEIGHTS:
            if n_centres != len(weights):
                continue
            tilted = enumerate_mixture_assignments(np.array(weights), family, values)
            moved = not np.array_equal(tilted.assignment, partition)
            assert moved == (n_samples == TILTED_INSTANCE[0]), (seed, weights)

    # "Only in one dimension": over 256 assignments of eight 2-D points the
    # optimum [0 0 0 0 1 0 0 0] costs 10.883, the best contiguous 14.422
    # (32.5% above); flattened, the callable returns 5.328, another question.
    points = np.random.default_rng(NON_CONTIGUOUS_SEED).normal(size=(8, 2)) * 2.0
    points = points[np.argsort(points[:, 0])]
    assignments = _assignment_matrix(8, 2)
    costs = _within_cluster_cost(points, assignments)
    contiguous = np.array(
        [
            cost
            for cost, one in zip(costs, assignments, strict=True)
            if _is_contiguous(one)
        ]
    )

    assert not _is_contiguous(assignments[int(costs.argmin())])
    assert_allclose(costs.min(), 10.883384005190335, rtol=1e-12)
    assert_allclose(contiguous.min(), 14.421633014203367, rtol=1e-12)
    assert_allclose(optimal_clustering_cost(points, 2), 5.327649356018618, rtol=1e-12)


class _ReportingGaussian(GaussianEmission):
    """`GaussianEmission` carrying the M-step report an iterative solve gives.

    Planted flags: the loop must read the report for any family (issue #856).
    """

    def __init__(
        self,
        mean: np.ndarray,
        scale: np.ndarray,
        variance_floor: float,
        *,
        converged: bool = True,
        at_boundary: bool = False,
    ) -> None:
        super().__init__(mean, scale, variance_floor)
        self._converged = converged
        self._at_boundary = at_boundary

    def reestimate(
        self,
        observations: ArrayLike,
        posterior: ArrayLike,
        covariate: ArrayLike | None = None,
    ) -> Reestimate[GaussianEmission]:
        """The closed-form step, reported as the planted flags say."""
        step = super().reestimate(observations, posterior, covariate)
        if not self._converged:
            return replace(step, converged=False, iterations=3, residual=0.25)
        return replace(step, at_boundary=self._at_boundary)


@pytest.mark.smoke
@pytest.mark.bug
def test_an_unconverged_component_m_step_is_refused() -> None:
    # The loop read `.emissions` off the report and dropped the rest, so a
    # number from an inner solve that never settled reached the outer
    # likelihood -- what `likelihood/CLAUDE.md` forbids and what the sibling
    # `opt.emission_mixture.expectation_maximization` refuses (issue #856).
    observations = _dataset(n_samples=100)
    components = _ReportingGaussian(MEAN, SCALE, 1e-9, converged=False)

    with pytest.raises(ValueError, match="did not settle at EM iteration 1"):
        expectation_maximization(
            observations, torch.as_tensor(WEIGHTS, dtype=torch.float64), components
        )


@pytest.mark.smoke
@pytest.mark.bug
def test_a_component_m_step_at_a_boundary_is_reported_on_the_fit() -> None:
    # `at_boundary` is not an error: the estimate is a bound rather than a
    # maximum, so a caller counts it (issue #122). It was computed and
    # discarded, and the fit carries the accumulation now (issue #856). The
    # flag moves no number: both fits run the same closed-form step.
    observations = _dataset(n_samples=100)
    weights = torch.as_tensor(WEIGHTS, dtype=torch.float64)

    flagged = expectation_maximization(
        observations,
        weights,
        _ReportingGaussian(MEAN, SCALE, 1e-9, at_boundary=True),
        config=replace(EM, max_iterations=3),
    )
    # The tensor route, which the flagging subclass also takes, so the flag
    # is the only difference the comparison can see (issue #986).
    clear = expectation_maximization(
        observations,
        weights,
        GaussianEmission(MEAN, SCALE, 1e-9),
        backend=Backend.PYTHON,
        config=replace(EM, max_iterations=3),
    )

    assert flagged.at_boundary
    assert not clear.at_boundary
    assert torch.equal(flagged.components.mean, clear.components.mean)
    assert torch.equal(flagged.components.scale, clear.components.scale)


@pytest.mark.oracle
def test_the_streamed_mixture_em_is_the_tensor_one() -> None:
    # Issue #986: the compiled route streams each draw into its components'
    # mass, mean and centred sum of squares; the tensor route is its oracle,
    # over ten iterations from a spread start on three well-separated modes.
    rng = np.random.default_rng(986)
    component = rng.choice(3, size=5_000, p=[0.3, 0.3, 0.4])
    draws = (
        np.array([-4.0, 0.0, 5.0])[component]
        + rng.normal(size=5_000) * np.array([1.0, 1.5, 1.0])[component]
    )
    start = GaussianEmission([-1.0, 0.5, 2.0], [2.0, 2.0, 2.0], 1e-12)
    weights = torch.full((3,), 1.0 / 3.0, dtype=torch.float64)
    oracle, streamed = (
        expectation_maximization(
            draws,
            weights,
            start,
            backend=backend,
            config=EmConfig(max_iterations=10, tolerance=-np.inf),
        )
        for backend in (Backend.PYTHON, Backend.RUST)
    )
    assert_allclose(
        streamed.weights.numpy(), oracle.weights.numpy(), rtol=0, atol=1e-12
    )
    assert_allclose(
        streamed.components.mean.numpy(), oracle.components.mean.numpy(), atol=1e-10
    )
    assert_allclose(
        streamed.components.scale.numpy(), oracle.components.scale.numpy(), atol=1e-10
    )
    assert_allclose(streamed.log_likelihood, oracle.log_likelihood, rtol=1e-12)
    assert streamed.iterations == oracle.iterations == 10


@pytest.mark.oracle
def test_the_mixture_gradient_is_autograd_s() -> None:
    # Issue #986: `GaussianMixtureObjective.gradient` streams responsibilities
    # into three sums per component; autograd through `__call__` pins it.
    rng = np.random.default_rng(9863)
    draws = np.concatenate(
        [
            rng.normal(-3.0, 1.0, 4_000),
            rng.normal(0.0, 1.5, 3_000),
            rng.normal(4.0, 1.0, 3_000),
        ]
    )
    objective = GaussianMixtureObjective(draws, 3)
    for _ in range(3):
        theta = torch.as_tensor(rng.normal(size=objective.n_parameters) * 0.5)
        point = theta.clone().requires_grad_(True)
        (autograd,) = torch.autograd.grad(objective(point), point)
        assert_allclose(
            objective.gradient(theta).numpy(), autograd.numpy(), rtol=1e-12, atol=1e-9
        )


@pytest.mark.oracle
def test_the_streamed_mixture_score_is_the_torch_one() -> None:
    # Issue #997: with no gradient to take, the log-likelihood streams
    # through the compiled kernel over chunks of 4,096; torch's logsumexp is
    # the oracle, and a tracked call still takes it.
    def check(n_samples: int) -> None:
        rng = np.random.default_rng(997)
        values = torch.as_tensor(rng.normal(0.0, 3.0, n_samples), dtype=torch.float64)
        family = GaussianEmission(
            torch.tensor([-3.0, 0.5, 4.0], dtype=torch.float64),
            torch.tensor([1.2, 1.0, 1.3], dtype=torch.float64),
            1e-12,
        )
        log_weight = torch.log(torch.tensor([0.3, 0.3, 0.4], dtype=torch.float64))
        streamed = mixture_log_likelihood(values, log_weight, family)
        oracle = mixture_log_likelihood(
            values, log_weight, family, backend=Backend.PYTHON
        )
        assert_allclose(float(streamed), float(oracle), rtol=1e-13)
        tracked = log_weight.clone().requires_grad_(True)
        assert mixture_log_likelihood(values, tracked, family).grad_fn is not None

    every_value([1, 4_097, 100_000], check)
