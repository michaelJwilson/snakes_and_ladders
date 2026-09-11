"""The eight seedings of the emission parameters, in projection (issue #541).

Two sizes and two claims. At the CI size --- the count fixture's own ``ci``
model, four components --- every candidate is held to the simulated truth it
was drawn from, and the cost each charges is checked against what it spent.
At release the key model's 100 components carry the ordering
`docs/nb/spatio_sequential.ipynb` states and
`docs/experiments/009-projection-emission-seedings.md` records, through
`opt.budget.compare` so no candidate wins by fitting longer.

**What refereed what.** The projection has no exact evidence at either size,
so the referee is the draw's own truth: the generating component of every
observation and the generating negative-binomial means. The one exact
statement available is a reduction --- squared Euclidean distance over a pair
whose second channel is constant is the one-dimensional k-means++ of
`opt.mixture`, draw for draw --- and it is asserted here because it is what
says the Euclidean candidate is k-means++ as it stands and not a second
algorithm.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from snakes_and_ladders.opt.budget import Budget, compare
from snakes_and_ladders.opt.initialize import (
    FromAnnealing,
    FromChain,
    FromTempering,
    Initializer,
)
from snakes_and_ladders.opt.mixture import kmeans_plus_plus
from snakes_and_ladders.opt.schedule import Exponential
from snakes_and_ladders.search.projection import (
    PASSES_PER_GRADIENT,
    SEEDINGS,
    CountPairAt,
    ProjectedCounts,
    SeededFit,
    euclidean_seeding,
    fit_projection,
    flatten,
    project,
)
from snakes_and_ladders.sim.count_pairs import binned_model
from snakes_and_ladders.sim.fixtures import KEY, fixture

PROBLEM = "spatio_sequential_counts"

#: Observations the CI-size projection draws: 1,000 per component at four
#: components, where the fit reaches the truth inside the budget below.
CI_SAMPLES = 4000

#: Iterations the CI-size fit is held to.
CI_BUDGET = Budget("passes", 6)

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


@pytest.mark.simulated_truth
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


@pytest.mark.simulated_truth
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


@pytest.mark.mathematical
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


@pytest.mark.structural
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


@pytest.mark.structural
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


@pytest.mark.structural
def test_the_chain_starts_satisfy_the_initializer_seam() -> None:
    # The ticket's condition on a new initializer: the existing seam, not a
    # second shape.
    generator = torch.Generator().manual_seed(541)

    assert isinstance(FromChain(2, 1e-3, generator), Initializer)
    assert isinstance(
        FromAnnealing(Exponential(1.0, 1.0, 1), 1e-3, generator), Initializer
    )
    assert isinstance(FromTempering((1.0, 2.0), 2, 1e-3, generator), Initializer)


@pytest.mark.edge_case
def test_a_projection_of_fewer_than_one_observation_is_refused() -> None:
    declared = fixture(PROBLEM, "ci").params

    with pytest.raises(ValueError, match="at least 1"):
        project(binned_model(declared.model, 1), 0, np.random.default_rng(0))


@pytest.mark.edge_case
def test_a_bin_factor_that_does_not_divide_the_positions_is_refused() -> None:
    declared = fixture(PROBLEM, "ci").params

    with pytest.raises(ValueError, match="does not divide"):
        binned_model(declared.model, 7)


#: The release-tier sweep's shape, as `infra/seeding_sweep.py` runs it. The
#: test reproduces it over fewer instances: what it pins is the ordering, and
#: the spread over instances is the experiment's to report.
KEY_SAMPLES = 4000
KEY_INSTANCES = 3
KEY_BUDGET = Budget("passes", 6)

#: The candidates whose ordering is claimed. `emission++` is the winner the
#: experiment records; `prior` is the control it must beat, and beating the
#: control is the claim a seeding that reads the data exists to support.
KEY_WINNER = "emission++"
KEY_CONTROL = "prior"


@pytest.mark.release
@pytest.mark.simulated_truth
def test_the_winning_seeding_beats_the_control_at_the_key_model() -> None:
    # The ordering the notebook states, at the key model's 100 components and
    # through the budgeted comparison, so no candidate reaches its optimum by
    # fitting longer than another.
    declared = fixture(PROBLEM, KEY).params
    params = binned_model(declared.model, declared.key_factor)
    instances = [
        project(params, KEY_SAMPLES, np.random.default_rng([541, index]))
        for index in range(KEY_INSTANCES)
    ]
    at = CountPairAt(
        float(flatten(params).total.dispersion.mean()),
        float(flatten(params).successes.concentration.mean()),
        instances[0].trials,
    )

    comparison = compare(
        {
            KEY_WINNER: SeededFit(KEY_WINNER, at),
            KEY_CONTROL: SeededFit(KEY_CONTROL, at),
        },
        instances,
        KEY_BUDGET,
        [0],
        workers=1,
    )

    gaps = comparison.mean_gap()
    assert gaps[KEY_WINNER] < gaps[KEY_CONTROL], (
        f"{KEY_WINNER} left a mean gap of {gaps[KEY_WINNER]:.1f} against "
        f"{KEY_CONTROL}'s {gaps[KEY_CONTROL]:.1f}"
    )
    assert (comparison.spent <= KEY_BUDGET.size).all()
