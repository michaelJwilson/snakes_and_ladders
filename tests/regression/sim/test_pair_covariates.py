"""Drawing the two-channel count pair: under a covariate, and from the general simulator.

Three seams could not previously draw a pair the way every seam above them
scores one. `sim.count_pairs.simulate_count_pairs` and its Rust twin ignored
the model's covariate (issue #671); `sim.spatio_sequential.simulate_spatio_
sequential` assumed a scalar observation and could not draw a pair at all
(issue #672). What referees the draws here is what refereed them before: the
families' own closed-form moments, never one simulator against the other
alone.

The covariate's two channels do different jobs and are asserted separately.
The exposure multiplies the negative binomial's mean, so a constant one moves
a mean that is known in closed form. The successes' covariate *replaces* the
declared trial count, so it bounds the draw outright --- a bound a mean-only
check would miss.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
import torch
from snakes_and_ladders.emissions import CovariateNotSupportedError
from snakes_and_ladders.sim.count_pairs import (
    SUCCESSES,
    TOTAL,
    CountPairInstance,
    IndependentCountPair,
    SpatioSequentialCountsParams,
    binned_model,
    coarsen,
    simulate_count_pairs,
    split_covariate,
)
from snakes_and_ladders.sim.count_pairs_rust import (
    _channels,
)
from snakes_and_ladders.sim.count_pairs_rust import (
    simulate_count_pairs as simulate_rust,
)
from snakes_and_ladders.sim.fixtures import fixture
from snakes_and_ladders.sim.graph import BoundaryCondition, lattice_graph
from snakes_and_ladders.sim.spatio_sequential import (
    SimulatedSpatioSequential,
    SpatioSequentialParams,
    simulate_spatio_sequential,
)

#: Either simulator of the declared count-pair instance.
Simulator = Callable[[SpatioSequentialCountsParams], CountPairInstance]

CI = "spatio_sequential_counts"

#: Relative tolerance on a sample mean over the ci instance's 64,000 draws.
#: The tightest the count families' own spread allows at this size; the Rust
#: simulator's existing moment tests use the same 3%.
MOMENT = 0.03


def _declared() -> SpatioSequentialCountsParams:
    """The ci count-pair fixture's declaration."""
    declared = fixture(CI, "ci").params
    assert isinstance(declared, SpatioSequentialCountsParams)
    return declared


def _covaried(exposure: float, trials: float) -> SpatioSequentialCountsParams:
    """The ci declaration with a constant covariate on both channels."""
    declared = _declared()
    params = declared.model
    shape = (params.n_positions, params.graph.n_nodes)
    covariate = np.stack([np.full(shape, exposure), np.full(shape, trials)], axis=-1)
    return replace(declared, model=replace(params, covariate=covariate))


def _per_state_mean(
    instance: CountPairInstance | SimulatedSpatioSequential,
    channel: int,
    class_index: int,
    state: int,
) -> float:
    """The sample mean of one channel over the positions one class sits in that state."""
    labels = np.asarray(instance.labels)
    states = np.asarray(instance.states)[class_index]
    columns = np.flatnonzero(labels == class_index)
    rows = np.flatnonzero(states == state)
    block = np.asarray(instance.observations)[np.ix_(rows, columns)][..., channel]
    return float(block.mean())


@pytest.mark.oracle
@pytest.mark.parametrize("simulate", [simulate_count_pairs, simulate_rust])
def test_a_constant_exposure_multiplies_the_negative_binomial_mean(
    simulate: Simulator,
) -> None:
    # Closed form, not a comparison between simulators: E[x] = e mu under an
    # exposure e, per (class, state), in both the NumPy and the Rust draw.
    exposure = 2.5
    declared = _covaried(exposure, 30.0)
    instance = simulate(declared)
    for m, family in enumerate(declared.model.emissions):
        assert isinstance(family, IndependentCountPair)
        for state in range(declared.model.n_states):
            expected = exposure * float(family.total.mean[state])
            drawn = _per_state_mean(instance, TOTAL, m, state)
            assert abs(drawn - expected) < expected * MOMENT


@pytest.mark.oracle
@pytest.mark.parametrize("simulate", [simulate_count_pairs, simulate_rust])
def test_the_successes_covariate_replaces_the_declared_trial_count(
    simulate: Simulator,
) -> None:
    # The bound is the point: a success count cannot exceed the trials it came
    # out of, so a covariate of 4 caps the channel at 4 whatever the file
    # declares. The mean n a / (a + b) is checked beside it, since a draw of
    # all zeros would also respect the bound.
    trials = 4.0
    declared = _covaried(1.0, trials)
    instance = simulate(declared)
    successes = np.asarray(instance.observations)[..., SUCCESSES]

    assert successes.max() <= trials
    for m, family in enumerate(declared.model.emissions):
        assert isinstance(family, IndependentCountPair)
        assert float(family.successes.trials.max()) > trials
        for state in range(declared.model.n_states):
            alpha = float(family.successes.alpha[state])
            beta = float(family.successes.beta[state])
            expected = trials * alpha / (alpha + beta)
            drawn = _per_state_mean(instance, SUCCESSES, m, state)
            assert abs(drawn - expected) < expected * 0.05


@pytest.mark.end2end
def test_the_two_simulators_agree_under_a_varying_covariate() -> None:
    # The existing NumPy-against-Rust comparison, made with a covariate that
    # varies per vertex and per position rather than a constant: the two
    # generators differ, so the claim is distributional, and what it pins is
    # that both read the same entry of the covariate for the same draw.
    declared = _declared()
    params = declared.model
    shape = (params.n_positions, params.graph.n_nodes)
    rng = np.random.default_rng(670)
    covariate = np.stack([rng.uniform(0.5, 3.0, shape), np.full(shape, 12.0)], axis=-1)
    covaried = replace(declared, model=replace(params, covariate=covariate))

    numpy_drawn = simulate_count_pairs(covaried)
    rust_drawn = simulate_rust(covaried)
    for channel in (TOTAL, SUCCESSES):
        for m in range(params.n_classes):
            for state in range(params.n_states):
                first = _per_state_mean(numpy_drawn, channel, m, state)
                second = _per_state_mean(rust_drawn, channel, m, state)
                assert abs(first - second) < max(first, second) * MOMENT


@pytest.mark.smoke
def test_a_covariate_that_names_no_channel_is_refused() -> None:
    declared = _declared()
    params = declared.model
    flat = np.ones((params.n_positions, params.graph.n_nodes))
    covaried = replace(declared, model=replace(params, covariate=flat))

    # One type for one refusal: a shape the NumPy draw rejects must not be a
    # shape the Rust draw accepts, and the caller catches one exception.
    for simulate in (simulate_rust, simulate_count_pairs):
        with pytest.raises(
            CovariateNotSupportedError, match="one covariate per channel"
        ):
            simulate(covaried)


def _numpy_channels(covariate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The NumPy draw's check: the family's own split, flattened per channel."""
    family = _declared().model.emissions[0]
    exposure, trials = split_covariate(family, torch.as_tensor(covariate))
    assert exposure is not None
    assert trials is not None
    return exposure.numpy().reshape(-1), trials.numpy().reshape(-1)


def _rust_channels(covariate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The Rust draw's check, which is now the same call."""
    exposure, trials = _channels(covariate)
    assert exposure is not None
    assert trials is not None
    return exposure, trials


@pytest.mark.smoke
@pytest.mark.bug
@pytest.mark.parametrize(
    "twin", [_numpy_channels, _rust_channels], ids=["numpy", "rust"]
)
@pytest.mark.parametrize(
    ("shape", "refused"),
    [((6, 2), False), ((3, 4, 2), False), ((4, 3), True)],
)
def test_both_twins_read_one_rank_rule(
    twin: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    shape: tuple[int, ...],
    refused: bool,
) -> None:
    # The NumPy check accepted a bare `(2,)` and the Rust one refused every
    # rank but three, so a covariate one simulator drew under was a refusal in
    # the other (issue #856). One check now: the trailing axis is the channel
    # pair, the leading axes are the caller's layout, and each twin gives the
    # same answer on the same shape.
    covariate = np.arange(float(np.prod(shape))).reshape(shape)

    if refused:
        with pytest.raises(
            CovariateNotSupportedError, match="one covariate per channel"
        ):
            twin(covariate)
        return

    exposure, trials = twin(covariate)

    np.testing.assert_array_equal(exposure, covariate[..., TOTAL].reshape(-1))
    np.testing.assert_array_equal(trials, covariate[..., SUCCESSES].reshape(-1))


@pytest.mark.analytic
def test_binning_sums_the_covariate_with_the_counts() -> None:
    # A binned count was drawn at the sum of its bin's exposures and out of
    # the sum of its trial counts, so the coarse covariate is the same
    # reduction as the coarse observation. Asserted against the sum written
    # out, and asserted equal in the two places a coarse model is built.
    factor = 10
    declared = _declared()
    params = declared.model
    shape = (params.n_positions, params.graph.n_nodes)
    rng = np.random.default_rng(671)
    covariate = np.stack(
        [rng.uniform(0.5, 3.0, shape), rng.integers(4, 10, shape).astype(float)],
        axis=-1,
    )
    covaried = replace(params, covariate=covariate)
    expected = covariate.reshape(
        params.n_positions // factor, factor, *covariate.shape[1:]
    ).sum(axis=1)

    binned = coarsen(simulate_count_pairs(replace(declared, model=covaried)), factor)
    for coarse in (binned.params.covariate, binned_model(covaried, factor).covariate):
        assert coarse is not None
        np.testing.assert_allclose(coarse, expected)


def _pair_model(covariate: np.ndarray | None) -> SpatioSequentialParams:
    """A two-class coupled model whose emissions are count pairs, on four vertices."""
    declared = _declared()
    return replace(
        declared.model,
        graph=lattice_graph((2, 2), BoundaryCondition.OPEN, 1.0),
        n_positions=300,
        covariate=covariate,
    )


@pytest.mark.end2end
def test_the_coupled_simulator_draws_a_pair_family() -> None:
    # The draw it refused outright (issue #672). The observation carries the
    # family's channel axis, and the truth it is held to is the family's own
    # mean per (class, state) --- the same referee the count-pair simulator
    # answers to, now reached through the general one.
    params = _pair_model(None)
    labels = np.array([0, 0, 1, 1])
    drawn = simulate_spatio_sequential(
        params, np.random.default_rng(672), labels=labels
    )

    assert drawn.observations.shape == (params.n_positions, 4, 2)
    for m, family in enumerate(params.emissions):
        assert isinstance(family, IndependentCountPair)
        for state in range(params.n_states):
            expected = float(family.total.mean[state])
            got = _per_state_mean(drawn, TOTAL, m, state)
            assert abs(got - expected) < expected * 0.06


@pytest.mark.end2end
def test_the_coupled_simulator_draws_a_pair_under_a_per_channel_covariate() -> None:
    # `_drawing_covariate` appended a singleton to a covariate that already
    # carried the channel axis, so `split_covariate` refused it; the draw is
    # held to the moments the covariate moves rather than to its shape.
    exposure, trials = 3.0, 5.0
    covariate = np.stack(
        [np.full((300, 4), exposure), np.full((300, 4), trials)], axis=-1
    )
    params = _pair_model(covariate)
    labels = np.array([0, 0, 1, 1])
    drawn = simulate_spatio_sequential(
        params, np.random.default_rng(673), labels=labels
    )

    assert drawn.observations[..., SUCCESSES].max() <= trials
    for m, family in enumerate(params.emissions):
        assert isinstance(family, IndependentCountPair)
        for state in range(params.n_states):
            expected = exposure * float(family.total.mean[state])
            got = _per_state_mean(drawn, TOTAL, m, state)
            assert abs(got - expected) < expected * 0.06


@pytest.mark.smoke
def test_a_scalar_family_still_draws_the_array_it_always_did() -> None:
    # The general simulator now reads the family's trailing axes; a family
    # with none must be unchanged, bitwise, or every committed coupled number
    # moved silently.
    entry = fixture("spatio_sequential", "ci")
    first = simulate_spatio_sequential(entry.params, np.random.default_rng(11))
    second = simulate_spatio_sequential(entry.params, np.random.default_rng(11))

    assert first.observations.shape == (entry.params.n_positions, 4)
    np.testing.assert_array_equal(first.observations, second.observations)
