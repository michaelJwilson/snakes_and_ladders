"""The eight seedings of the emission parameters, in projection (issue #541).

At CI (the count fixture's four components) every candidate is held to the
simulated truth and its charged cost to its spend. At release the key model's
100 components carry the ordering `docs/nb/spatio_sequential.ipynb` states
and `docs/experiments/009-projection-emission-seedings.md` records, through
`opt.budget.compare`. Referee: the draw's truth. Exact beside it: `project`'s
draw is one closed-form mixture over ``M x K`` pairs; the Euclidean candidate
is `opt.mixture`'s 1-D k-means++ draw for draw; each other candidate's
divergence is written out and pinned to its draws (issue #734).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import numpy as np
import pytest
import torch
from scipy.optimize import minimize_scalar
from snakes_and_ladders.cost import Cost
from snakes_and_ladders.opt.budget import Budget, compare
from snakes_and_ladders.opt.initialize import (
    FromObjective,
    Initializer,
    quantile_locations,
)
from snakes_and_ladders.opt.mixture import (
    GaussianMixtureObjective,
    kmeans_plus_plus,
    uniform_seeds,
)
from snakes_and_ladders.sample.initialize import FromAnnealing, FromChain, FromTempering
from snakes_and_ladders.sample.schedule import ExponentialTempSchedule
from snakes_and_ladders.sample.statistics import chi_square_p_value
from snakes_and_ladders.search.projection import (
    PASSES_PER_GRADIENT,
    SEEDINGS,
    CountPairAt,
    ProjectedCounts,
    SeededFit,
    Seeding,
    data_seeding,
    emission_seeding,
    euclidean_seeding,
    fit_projection,
    flatten,
    project,
    seed_at_means,
    state_bytes,
)
from snakes_and_ladders.sim.count_pairs import (
    SUCCESSES,
    TOTAL,
    IndependentCountPair,
    binned_model,
    planted_labels,
)
from snakes_and_ladders.sim.fixtures import KEY, fixture
from snakes_and_ladders.track import MemoryRun, track

PROBLEM = "spatio_sequential_counts"

#: Observations the CI-size projection draws: 1,000 per component at four
#: components, where the fit reaches the truth inside the budget below.
CI_SAMPLES = 4000

#: Iterations the CI-size fit is held to.
CI_BUDGET = Budget(Cost.PASSES, 6)

#: The largest relative error in a component's negative-binomial mean that the
#: three chain-based candidates may leave at the CI size, and the recovery they
#: must reach. Measured on this draw: 0.06 to 0.08 and 0.75 to 0.76, against a
#: recovery of 0.78 under the generating parameters themselves.
SAMPLED_MEAN_ERROR = 0.15
SAMPLED_RECOVERY = 0.70

#: The recovery every candidate must reach, chance being 0.25 at four
#: components. The heuristics run 0.48 to 0.68 here at a six-iteration budget;
#: what separates them is the key model's business, at release.
CI_RECOVERY = 0.40

#: The tail probability below which a draw is judged not to be the closed-form
#: law, the repository's convention for a sampler (`search/test_gibbs.py`).
SIGNIFICANCE = 0.001


def _projection(tier: str, n_samples: int, seed: int) -> ProjectedCounts:
    """One projected draw of a declared model, at a bin factor of one."""
    declared = fixture(PROBLEM, tier).params
    params = binned_model(declared.model, 1)
    return project(params, n_samples, np.random.default_rng([541, seed]))


def _seam(tier: str) -> CountPairAt:
    """The `ComponentsAt` every candidate ends at, its shapes the model's own."""
    declared = fixture(PROBLEM, tier).params
    truth = flatten(binned_model(declared.model, 1))
    return CountPairAt(
        float(truth.total.dispersion.mean()),
        float(truth.successes.concentration.mean()),
        float(truth.successes.trials[0]),
    )


def _negative_binomial(
    support: Iterable[int], dispersion: float, mean: float
) -> np.ndarray:
    """``NegativeBinomial(r, mu)``'s mass function, written from the definition."""
    probability = dispersion / (dispersion + mean)
    return np.array(
        [
            math.exp(
                math.lgamma(count + dispersion)
                - math.lgamma(dispersion)
                - math.lgamma(count + 1.0)
                + dispersion * math.log(probability)
                + count * math.log1p(-probability)
            )
            for count in support
        ]
    )


def _beta_binomial(
    support: Iterable[int], trials: float, alpha: float, beta: float
) -> np.ndarray:
    """``BetaBinomial(n, a, b)``'s mass function, written from the definition."""

    def log_beta(first: float, second: float) -> float:
        return math.lgamma(first) + math.lgamma(second) - math.lgamma(first + second)

    return np.array(
        [
            math.exp(
                math.lgamma(trials + 1.0)
                - math.lgamma(count + 1.0)
                - math.lgamma(trials - count + 1.0)
                + log_beta(count + alpha, trials - count + beta)
                - log_beta(alpha, beta)
            )
            for count in support
        ]
    )


def _pooled(
    mass: np.ndarray, counts: np.ndarray, n_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """Cells merged left to right until each expects five, the tail in the last."""
    observed, expected, pending_o, pending_e = [], [], 0.0, 0.0
    for probability, count in zip(mass, counts, strict=True):
        pending_e += probability * n_samples
        pending_o += count
        if pending_e >= 5.0:
            expected.append(pending_e)
            observed.append(pending_o)
            pending_o = pending_e = 0.0
    expected[-1] += pending_e + (1.0 - float(mass.sum())) * n_samples
    observed[-1] += pending_o
    return np.array(observed), np.array(expected)


@pytest.mark.oracle
@pytest.mark.critical
def test_the_projected_draw_is_the_closed_form_mixture_of_the_flattened_families() -> (
    None
):
    # The rung's closed form (#734). Weights: class shares split over uniform
    # stationary states, 0.25 bitwise. Components: chi-square p 0.389, 0.171,
    # 0.874 over three draws (0.001). Channels per component: NB and
    # beta-binomial, 24 tests, smallest p 0.064. Channel independence is
    # unrefuted by marginals: the ticket's gap.
    params = binned_model(fixture(PROBLEM, "ci").params.model, 1)
    truth = flatten(params)
    trials = float(truth.successes.trials[0])
    share = np.bincount(
        planted_labels(params, params.n_classes), minlength=params.n_classes
    ) / float(params.graph.n_nodes)
    closed_form = np.repeat(share / params.n_states, params.n_states)

    for seed in range(3):
        instance = _projection("ci", CI_SAMPLES, seed)

        assert np.array_equal(instance.weights, closed_form)
        drawn = np.bincount(instance.components, minlength=len(closed_form))
        assert (
            chi_square_p_value(drawn.astype(float), closed_form * CI_SAMPLES)
            > SIGNIFICANCE
        )

        for component in range(len(closed_form)):
            rows = instance.observations[instance.components == component]
            n_samples = len(rows)
            for channel, mass in (
                (
                    0,
                    _negative_binomial(
                        range(int(rows[:, 0].max()) + 1),
                        float(truth.total.dispersion[component]),
                        float(truth.total.mean[component]),
                    ),
                ),
                (
                    1,
                    _beta_binomial(
                        range(int(trials) + 1),
                        trials,
                        float(truth.successes.alpha[component]),
                        float(truth.successes.beta[component]),
                    ),
                ),
            ):
                observed, expected = _pooled(
                    mass,
                    np.bincount(rows[:, channel], minlength=len(mass)).astype(float),
                    n_samples,
                )
                assert chi_square_p_value(observed, expected) > SIGNIFICANCE


@pytest.mark.end2end
@pytest.mark.parametrize("name", list(SEEDINGS))
def test_the_fit_improves_on_every_seeding_and_beats_chance(name: str) -> None:
    # The referee is the draw's own truth. Two claims that do not depend on
    # the budget the fit was given: it leaves every seeding better than it
    # found it, and the components it ends on assign the observations to their
    # generating component well above the 0.25 chance of four components.
    instance = _projection("ci", CI_SAMPLES, 0)

    fitted = fit_projection(
        instance, name, _seam("ci"), CI_BUDGET, np.random.default_rng([541, 0])
    )

    assert fitted.log_likelihoods[-1] > fitted.log_likelihoods[0], (
        f"{name}'s fit did not improve on its own seeding"
    )
    assert fitted.recovery >= CI_RECOVERY, (
        f"{name} assigned {fitted.recovery:.3f} of observations to their "
        f"generating component, below {CI_RECOVERY}"
    )


@pytest.mark.end2end
def test_the_sampled_seedings_recover_the_generating_means_at_the_ci_size() -> None:
    # What the notebook's table states: the three candidates that sample a
    # surface reach the truth inside the budget where the heuristics do not,
    # at 24 times the seeding cost. The claim here is on the truth alone; the
    # ordering against the heuristics is the release-tier comparison's.
    instance = _projection("ci", CI_SAMPLES, 0)

    for name in ("hmc", "anneal", "tempering"):
        fitted = fit_projection(
            instance, name, _seam("ci"), CI_BUDGET, np.random.default_rng([541, 0])
        )

        assert fitted.mean_error <= SAMPLED_MEAN_ERROR, (
            f"{name} left a relative error of {fitted.mean_error:.3f} in a "
            f"component mean, above {SAMPLED_MEAN_ERROR}"
        )
        assert fitted.recovery >= SAMPLED_RECOVERY, (
            f"{name} reached a recovery of {fitted.recovery:.3f}, below "
            f"{SAMPLED_RECOVERY}"
        )


@pytest.mark.analytic
@pytest.mark.parametrize("name", list(SEEDINGS))
def test_the_projected_likelihood_does_not_decrease_under_the_fit(name: str) -> None:
    # Expectation-maximization increases the likelihood at every step, so a
    # curve that dips is a broken M step and not a hard start.
    instance = _projection("ci", CI_SAMPLES, 1)

    fitted = fit_projection(
        instance, name, _seam("ci"), CI_BUDGET, np.random.default_rng([541, 1])
    )

    assert (np.diff(fitted.log_likelihoods) >= -1e-9).all(), (
        f"{name}'s curve is {fitted.log_likelihoods.tolist()}"
    )
    assert fitted.iterations <= CI_BUDGET.size


@pytest.mark.smoke
def test_a_seeding_charges_what_its_rule_spends() -> None:
    # The cost rule is what makes an unequal seeding budget reportable rather
    # than hidden, so the charge is asserted against the rule that incurs it:
    # nothing for a draw that reads no distance, one pass for a D-squared
    # sweep, two per gradient for a chain.
    instance = _projection("ci", CI_SAMPLES, 2)
    at = _seam("ci")

    charged = {
        name: SEEDINGS[name](instance, at, np.random.default_rng([541, 2])).passes
        for name in SEEDINGS
    }

    assert charged["data"] == 0.0
    assert charged["prior"] == 0.0
    assert charged["kmeans++"] == 1.0
    assert charged["emission++"] == 1.0
    assert 0.0 < charged["burn-in"] < 1.0
    for name in ("hmc", "anneal", "tempering"):
        assert charged[name] == PASSES_PER_GRADIENT * 72, (
            f"{name} charged {charged[name]} passes for 72 gradients"
        )


@pytest.mark.smoke
def test_the_chain_candidates_report_a_diagnostic() -> None:
    # A seed drawn from a chain that has not mixed is a random restart with a
    # longer bill, so a chain-based candidate that reported nothing would hide
    # exactly what the ticket asks to see.
    instance = _projection("ci", CI_SAMPLES, 3)
    at = _seam("ci")

    for name in ("hmc", "anneal", "tempering"):
        seeding = SEEDINGS[name](instance, at, np.random.default_rng([541, 3]))
        assert "acceptance" in seeding.diagnostics, name
    for name in ("data", "prior", "kmeans++", "emission++", "burn-in"):
        assert (
            SEEDINGS[name](instance, at, np.random.default_rng([541, 3])).diagnostics
            == ""
        )


@pytest.mark.oracle
def test_euclidean_seeding_is_one_dimensional_kmeans_plus_plus_on_a_flat_channel() -> (
    None
):
    # The reduction that says the Euclidean candidate is k-means++ as it
    # stands: with the second channel constant the squared distance between
    # pairs is the squared distance between totals, so the two schemes draw
    # the same seeds from the same generator.
    instance = _projection("ci", 600, 4)
    flat = np.stack(
        [instance.observations[:, 0], np.zeros(instance.observations.shape[0])], axis=1
    )
    flattened = ProjectedCounts(
        observations=flat.astype(np.int64),
        components=instance.components,
        truth=instance.truth,
        weights=instance.weights,
        trials=instance.trials,
    )

    seeded = euclidean_seeding(flattened, _seam("ci"), np.random.default_rng(4))
    expected = kmeans_plus_plus(
        np.asarray(flat[:, 0], dtype=np.float64),
        instance.n_components,
        np.random.default_rng(4),
    )

    assert np.allclose(np.sort(seeded.components.total.mean.numpy()), np.sort(expected))


@pytest.mark.smoke
def test_the_chain_starts_satisfy_the_initializer_seam() -> None:
    # The ticket's condition on a new initializer: the existing seam, not a
    # second shape.
    generator = torch.Generator().manual_seed(541)

    assert isinstance(FromChain(2, 1e-3, generator), Initializer)
    assert isinstance(
        FromAnnealing(ExponentialTempSchedule(1.0, 1.0, 1), 1e-3, generator),
        Initializer,
    )
    assert isinstance(FromTempering((1.0, 2.0), 2, 1e-3, generator), Initializer)


@pytest.mark.smoke
def test_a_projection_of_fewer_than_one_observation_is_refused() -> None:
    declared = fixture(PROBLEM, "ci").params

    with pytest.raises(ValueError, match="at least 1"):
        project(binned_model(declared.model, 1), 0, np.random.default_rng(0))


@pytest.mark.smoke
def test_a_bin_factor_that_does_not_divide_the_positions_is_refused() -> None:
    declared = fixture(PROBLEM, "ci").params

    with pytest.raises(ValueError, match="does not divide"):
        binned_model(declared.model, 7)


#: The release-tier sweep's shape, as `infra/seeding_sweep.py` runs it. The
#: test reproduces it over fewer instances: what it pins is the ordering, and
#: the spread over instances is the experiment's to report.
KEY_SAMPLES = 4000
KEY_INSTANCES = 3
KEY_BUDGET = Budget(Cost.PASSES, 6)

#: The two candidates the key model's finding is about. `tempering` samples a
#: surrogate surface for 144 passes; `prior` reads none of the data for none.
#: Experiment 009 records the rest of the eight.
KEY_SAMPLED = "tempering"
KEY_CONTROL = "prior"


@pytest.mark.release
@pytest.mark.end2end
def test_the_likelihood_and_the_truth_order_the_seedings_oppositely() -> None:
    """The key model's finding, and the reason no default moves.

    100 components, 4,000 observations: likelihood prefers the data-blind draw,
    truth the 144-pass surrogate sampler, on all six of experiment 009; three here.
    """
    declared = fixture(PROBLEM, KEY).params
    params = binned_model(declared.model, declared.key_factor)
    truth = flatten(params)
    at = CountPairAt(
        float(truth.total.dispersion.mean()),
        float(truth.successes.concentration.mean()),
        project(params, 1, np.random.default_rng([541, 0])).trials,
    )

    for index in range(KEY_INSTANCES):
        instance = project(params, KEY_SAMPLES, np.random.default_rng([541, index]))
        control, sampled = (
            fit_projection(
                instance, name, at, KEY_BUDGET, np.random.default_rng([0, index])
            )
            for name in (KEY_CONTROL, KEY_SAMPLED)
        )

        assert control.log_likelihood > sampled.log_likelihood, (
            f"on instance {index} the likelihood did not prefer {KEY_CONTROL}: "
            f"{control.log_likelihood:.1f} against {KEY_SAMPLED}'s "
            f"{sampled.log_likelihood:.1f}"
        )
        assert sampled.recovery > control.recovery, (
            f"on instance {index} the truth did not prefer {KEY_SAMPLED}: "
            f"{sampled.recovery:.3f} against {KEY_CONTROL}'s "
            f"{control.recovery:.3f}"
        )
        assert sampled.seeding.passes > KEY_BUDGET.size, (
            f"{KEY_SAMPLED} charged {sampled.seeding.passes:.0f} passes, which "
            f"is no longer the different proposition the finding turns on"
        )


#: Observations the seeding instance carries. Small enough that the rule is
#: written out over every pair in the test and large enough that the divergence
#: separates them: the D-squared law below runs from 2.1e-07 to 0.033 against a
#: uniform 0.0083, a total variation of 0.37 from uniform.
SEEDING_SAMPLES = 120

#: Generators each seeding is run from. Eight, because a rule that agrees on
#: one stream agrees by luck.
SEEDING_SEEDS = range(8)

#: Against `bregman_divergence`: the beta-binomial saturated rate is found by
#: 60 bisections there and bounded minimization here: 7.8e-14 absolute,
#: 4.7e-09 relative near zero.
DIVERGENCE_RTOL = 1e-6
DIVERGENCE_ATOL = 1e-12


def _beta_binomial_deviance(
    count: float, trials: float, concentration: float, rate: float
) -> float:
    """The log-density gap to the best rate at this concentration, written out.

    The unit deviance (no log-partition); zero at both ends of the support.
    """

    def negative(candidate: float) -> float:
        return -math.log(
            _beta_binomial(
                [int(count)],
                trials,
                concentration * candidate,
                concentration * (1.0 - candidate),
            )[0]
        )

    if count in (0.0, trials):
        saturated = 0.0
    else:
        saturated = -float(
            minimize_scalar(
                negative,
                bounds=(1e-12, 1.0 - 1e-12),
                method="bounded",
                options={"xatol": 1e-14},
            ).fun
        )
    return max(saturated + negative(rate), 0.0)


def _declared_divergence(
    seed_row: np.ndarray, rows: np.ndarray, at: CountPairAt
) -> np.ndarray:
    """``D_phi`` of every pair against the component the seam places on ``seed_row``.

    NB: ``r log((r + mu) / (r + y)) + y log(y (r + mu) / (mu (r + y)))``;
    plus the beta-binomial deviance.
    """
    mean = max(float(seed_row[TOTAL]), 1.0)
    rate = (float(seed_row[SUCCESSES]) + 0.5) / (at.trials + 1.0)
    counts = rows[:, TOTAL]
    negative_binomial = at.dispersion * np.log(
        (at.dispersion + mean) / (at.dispersion + counts)
    ) + np.where(
        counts > 0.0,
        counts
        * np.log(
            np.where(counts > 0.0, counts, 1.0)
            * (at.dispersion + mean)
            / (mean * (at.dispersion + counts))
        ),
        0.0,
    )
    beta_binomial = np.array(
        [
            _beta_binomial_deviance(float(count), at.trials, at.concentration, rate)
            for count in rows[:, SUCCESSES]
        ]
    )
    return np.asarray(negative_binomial + beta_binomial)


def _d_squared_draws(
    rows: np.ndarray,
    n_centres: int,
    rng: np.random.Generator,
    score: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Arthur & Vassilvitskii's D-squared rule, distance open: uniform, then by ``score``."""
    indices = np.arange(rows.shape[0], dtype=np.float64)
    chosen = [rng.choice(indices)]
    nearest = score(rows[int(chosen[0])], rows)
    for _ in range(1, n_centres):
        total = float(nearest.sum())
        chosen.append(
            rng.choice(indices)
            if total <= 0.0
            else rng.choice(indices, p=nearest / total)
        )
        nearest = np.minimum(nearest, score(rows[int(chosen[-1])], rows))
    return np.array(chosen, dtype=np.int64)


def _same_components(first: IndependentCountPair, second: IndependentCountPair) -> bool:
    """Whether two seeded families carry the same components, bit for bit."""
    return all(
        np.array_equal(left.numpy(), right.numpy())
        for left, right in (
            (first.total.mean, second.total.mean),
            (first.successes.alpha, second.successes.alpha),
            (first.successes.beta, second.successes.beta),
        )
    )


@pytest.mark.oracle
@pytest.mark.critical
def test_the_non_euclidean_seedings_draw_the_law_of_the_metric_they_declare() -> None:
    # The rung below (#734): `kmeans_plus_plus`. The written-out rule with the
    # first channel's squared distance is it, draw for draw, on 8 of 8
    # generators; with the pair's, `euclidean_seeding`; with the family
    # divergence, `emission_seeding`; all bitwise. The divergence matches the
    # family's to 7.8e-14 absolute (4.7e-09 relative; 1e-6). It picks
    # differently from Euclidean on 8 of 8, its law spanning 2.1e-07 to 0.033
    # against 0.0083 (TV 0.37). `data_seeding` is `uniform_seeds` bitwise.
    instance = _projection("ci", SEEDING_SAMPLES, 4)
    at = _seam("ci")
    rows = np.asarray(instance.observations, dtype=np.float64)

    def divergence(seed_row: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        return _declared_divergence(seed_row, candidates, at)

    def euclidean(seed_row: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        return np.asarray(((candidates - seed_row) ** 2).sum(axis=1))

    worst = 0.0
    for index in (5, 37, 96):
        written = _declared_divergence(rows[index], rows, at)
        shipped = (
            at(rows[[index]])
            .bregman_divergence(torch.as_tensor(rows, dtype=torch.float64))[:, 0]
            .numpy()
        )
        worst = max(
            worst,
            float(
                (np.abs(written - shipped) / np.maximum(np.abs(shipped), 1e-12)).max()
            ),
        )
        np.testing.assert_allclose(
            written, shipped, rtol=DIVERGENCE_RTOL, atol=DIVERGENCE_ATOL
        )
    assert worst < DIVERGENCE_RTOL, worst

    parted = 0
    flat = rows[:, TOTAL]
    for seed in SEEDING_SEEDS:
        drawn = _d_squared_draws(
            rows, instance.n_components, np.random.default_rng(seed), divergence
        )
        seeded = emission_seeding(instance, at, np.random.default_rng(seed))
        assert _same_components(seeded.components, at(rows[drawn])), seed

        euclid = _d_squared_draws(
            rows, instance.n_components, np.random.default_rng(seed), euclidean
        )
        assert _same_components(
            euclidean_seeding(instance, at, np.random.default_rng(seed)).components,
            at(rows[euclid]),
        ), seed
        np.testing.assert_array_equal(
            kmeans_plus_plus(flat, instance.n_components, np.random.default_rng(seed)),
            flat[
                _d_squared_draws(
                    flat.reshape(-1, 1),
                    instance.n_components,
                    np.random.default_rng(seed),
                    euclidean,
                )
            ],
        )
        parted += int(not np.array_equal(drawn, euclid))

        uniform = uniform_seeds(
            np.arange(rows.shape[0], dtype=np.float64),
            instance.n_components,
            np.random.default_rng(seed),
        ).astype(np.int64)
        assert _same_components(
            data_seeding(instance, at, np.random.default_rng(seed)).components,
            at(rows[uniform]),
        ), seed

    assert parted == len(SEEDING_SEEDS), parted

    law = _declared_divergence(rows[5], rows, at)
    law = law / law.sum()
    assert float(law.max()) > 3.0 / rows.shape[0], float(law.max())
    assert 0.5 * float(np.abs(law - 1.0 / rows.shape[0]).sum()) > 0.3


@pytest.mark.smoke
def test_the_tracked_curve_is_the_fit_s_own_log_likelihoods() -> None:
    # The run's series is the field's history, not a second definition of it
    # (`track.py`), so the notebook that plots one curve per candidate reads
    # what the fit already reports (issue #887).
    instance = _projection("ci", CI_SAMPLES, 0)

    with track(MemoryRun()) as tracked:
        fitted = fit_projection(
            instance, "emission++", _seam("ci"), CI_BUDGET, np.random.default_rng(0)
        )
    run = tracked.run
    assert isinstance(run, MemoryRun)

    recorded = run.series("log_likelihood")
    assert [step for step, _ in recorded] == list(range(len(fitted.log_likelihoods)))
    assert [value for _, value in recorded] == fitted.log_likelihoods.tolist()
    assert [value for _, value in run.series("objective")] == [
        -value for value in fitted.log_likelihoods
    ]
    assert run.last("state_bytes") == float(
        state_bytes(
            torch.full(
                (instance.n_components,),
                1.0 / instance.n_components,
                dtype=torch.float64,
            ),
            instance.truth,
        )
    )


@pytest.mark.smoke
@pytest.mark.patch
def test_an_untracked_fit_is_the_fit_before_the_hook() -> None:
    # The seam costs a returned call an iteration and moves no number: the
    # default binding is `track.NULL` (`track.py`).
    instance = _projection("ci", CI_SAMPLES, 0)
    arguments = (instance, "kmeans++", _seam("ci"), CI_BUDGET)

    outside = fit_projection(*arguments, np.random.default_rng(0))
    with track(MemoryRun()):
        inside = fit_projection(*arguments, np.random.default_rng(0))

    assert outside.log_likelihoods.tolist() == inside.log_likelihoods.tolist()


@pytest.mark.smoke
def test_a_seeding_made_elsewhere_is_fitted_through_the_same_loop() -> None:
    # The seam a start outside `SEEDINGS` reaches the fit by: the loop, the
    # budget and the referee are the candidates' own, so what separates it
    # from them is the seeding and nothing after it (issue #887).
    instance = _projection("ci", CI_SAMPLES, 0)
    at = _seam("ci")
    made = emission_seeding(instance, at, np.random.default_rng(0))

    named = fit_projection(
        instance, "emission++", at, CI_BUDGET, np.random.default_rng(0)
    )
    given = fit_projection(
        instance, "given", at, CI_BUDGET, np.random.default_rng(1), seeding=made
    )

    assert given.name == "given"
    assert given.log_likelihoods.tolist() == named.log_likelihoods.tolist()
    with pytest.raises(KeyError):
        fit_projection(instance, "given", at, CI_BUDGET, np.random.default_rng(0))


@pytest.mark.smoke
def test_locations_reach_the_seam_through_the_observations_nearest_them() -> None:
    # `seed_at_means` is what the chain-based candidates already went through,
    # exposed for a start that holds means rather than a surrogate's `theta`
    # (issue #887): the seeded family is the seam's on those observations.
    instance = _projection("ci", CI_SAMPLES, 0)
    at = _seam("ci")
    totals = np.asarray(instance.observations, dtype=np.float64)[:, TOTAL]
    means = torch.as_tensor(np.sort(totals[[3, 17, 91, 402]]))

    seeded = seed_at_means(instance, means, at)

    nearest = np.abs(totals[None, :] - means.numpy()[:, None]).argmin(axis=1)
    assert _same_components(seeded, at(instance.observations[nearest]))


@pytest.mark.smoke
def test_a_start_outside_the_candidate_set_compares_at_the_same_budget() -> None:
    # `compare` holds the budget equal; a start reaching the seam by another
    # rule enters the same comparison rather than a second one beside it.
    instance = _projection("ci", CI_SAMPLES, 0)
    at = _seam("ci")

    comparison = compare(
        {
            "emission++": SeededFit("emission++", at),
            "given": SeededFit("given", at, emission_seeding),
        },
        [instance],
        CI_BUDGET,
        [0],
        workers=1,
    )

    assert comparison.best[0, 0] == comparison.best[1, 0]
    assert comparison.spent.max() <= CI_BUDGET.size


@pytest.mark.end2end
def test_a_start_built_from_the_surrogate_s_locations_recovers_the_truth() -> None:
    # #887's claim for location starts (moved by #891): assignment recovery as
    # the candidates achieve, against the draw's truth and the same bound.
    instance = _projection("ci", CI_SAMPLES, 0)
    at = _seam("ci")
    objective = GaussianMixtureObjective(
        np.asarray(instance.observations, dtype=np.float64)[:, TOTAL],
        instance.n_components,
    )
    nominated = objective.components(FromObjective().starts(objective)[0]).mean
    quantiles = quantile_locations(objective.observations, instance.n_components)

    for label, means in (("objective", nominated), ("quantile", quantiles)):
        fitted = fit_projection(
            instance,
            label,
            at,
            CI_BUDGET,
            np.random.default_rng([887, 0]),
            seeding=Seeding(seed_at_means(instance, means, at), 0.0),
        )
        assert fitted.recovery >= CI_RECOVERY, (
            f"{label} reached {fitted.recovery:.3f}, below {CI_RECOVERY}"
        )

    # And the notebook's one structural finding there: the surrogate's own
    # nominated point *is* the quantile placement, so the two starts are one.
    assert _same_components(
        seed_at_means(instance, nominated, at), seed_at_means(instance, quantiles, at)
    )
