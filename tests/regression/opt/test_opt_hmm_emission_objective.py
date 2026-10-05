"""`opt.hmm.EmissionHmmObjective`: one HMM objective over any family, on segments (issue #1169).

Referees, cheapest first: each per-family ``*HmmObjective`` where the two
overlap, its value and its autograd gradient at the same constrained point;
autograd through ``__call__``, which the declared Fisher-identity gradient is
pinned to; two channel identities of the independent count pair; and the
planted truth of a simulated segmented HMM, recovered within four standard
errors for a count, a joint and a Gaussian family.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from sal.backend import Backend
from sal.emissions import (
    BetaBinomialEmission,
    BinomialEmission,
    CategoricalEmission,
    CountPairEmission,
    EmissionFamily,
    GaussianEmission,
    NegativeBinomialEmission,
    PoissonEmission,
)
from sal.opt.fit import fit
from sal.opt.hmm import (
    BetaBinomialHmmObjective,
    BinomialHmmObjective,
    EmissionHmmObjective,
    GaussianHmmObjective,
    HmmObjective,
    NegativeBinomialHmmObjective,
    PoissonHmmObjective,
    align_families,
    forward_log_likelihood_ragged,
)
from sal.opt.hmm.objectives import _HmmObjective
from sal.opt.initialize import quantile_locations
from sal.opt.objective import (
    DeclaredBlocks,
    Restricted,
    autograd_value_and_gradient,
    coordinates,
    value_and_gradient,
)
from sal.ragged import Ragged
from sal.sim.hmm import HmmParams, simulate_sequences

SEED = 1169

#: Segment lengths of the recovery fixture: twenty segments of 50-249 positions.
LENGTHS = tuple(
    int(length) for length in np.random.default_rng(SEED).integers(50, 250, 20)
)
INITIAL = np.array([0.5, 0.5])
TRANSITION = np.array([[0.95, 0.05], [0.08, 0.92]])

#: Recovery bound, in standard errors of the fitted parameter. Fourteen
#: parameters at most, so a correct fit fails it with probability below 1e-3.
Z_BOUND = 4.0


def _overlaps() -> dict[str, tuple[_HmmObjective, EmissionHmmObjective]]:
    """Each per-family objective and this one on the same rectangular data and family.

    Three sequences of 40, two states; the start family is the per-family
    objective's own at its initial point.
    """
    rng = np.random.default_rng([SEED, 0])
    counts = rng.poisson(5.0, size=(3, 40))
    trials = np.array([12, 12])
    successes = rng.binomial(12, 0.3, size=(3, 40))
    reals = rng.normal(size=(3, 40))
    symbols = rng.integers(0, 3, size=(3, 40))
    exposure = rng.uniform(0.5, 2.0, size=(3, 40))
    references: dict[str, tuple[_HmmObjective, np.ndarray, np.ndarray | None]] = {
        "categorical": (
            HmmObjective(symbols, 2, 3, backend=Backend.TORCH),
            symbols,
            None,
        ),
        "gaussian": (
            GaussianHmmObjective(reals, 2, backend=Backend.TORCH),
            reals,
            None,
        ),
        "poisson": (
            PoissonHmmObjective(counts, 2, backend=Backend.TORCH),
            counts,
            None,
        ),
        "binomial": (
            BinomialHmmObjective(successes, 2, trials, backend=Backend.TORCH),
            successes,
            None,
        ),
        "beta_binomial": (
            BetaBinomialHmmObjective(successes, 2, trials, backend=Backend.TORCH),
            successes,
            None,
        ),
        "negative_binomial": (
            NegativeBinomialHmmObjective(counts, 2, backend=Backend.TORCH),
            counts,
            None,
        ),
        "negative_binomial_exposure": (
            NegativeBinomialHmmObjective(
                counts, 2, covariate=exposure, backend=Backend.TORCH
            ),
            counts,
            exposure[..., None],
        ),
    }
    return {
        name: (
            reference,
            EmissionHmmObjective(
                data,
                reference.components(reference.initial()),
                covariate=covariate,
            ),
        )
        for name, (reference, data, covariate) in references.items()
    }


def _off_start(objective: EmissionHmmObjective) -> torch.Tensor:
    """A point off the start, so no coordinate sits at a symmetric value."""
    start = objective.initial()
    return start + torch.linspace(-0.3, 0.2, start.numel(), dtype=torch.float64)


@pytest.mark.parametrize("name", list(_overlaps()))
@pytest.mark.oracle
def test_value_and_gradient_equal_the_per_family_objective_s(name: str) -> None:
    # The same constrained point in both layouts, mapped by `theta_from`; the
    # gradient compared block by block under `constrain`'s keys, since the
    # per-family layout orders a family's parameters its own way. Every map
    # is the same per block, so the free coordinates agree entry for entry.
    # Measured: value 0.0 relative; gradient 5.7e-14 absolute at worst, on
    # gradients up to 1.0e2.
    reference, objective = _overlaps()[name]
    theta = _off_start(objective)
    mapped = reference.theta_from(objective.constrain(theta))

    value, gradient = autograd_value_and_gradient(objective, theta)
    want_value, want_gradient = autograd_value_and_gradient(reference, mapped)

    assert float(value) == pytest.approx(float(want_value), rel=1e-12, abs=0.0)
    for block, where in objective.blocks.items():
        np.testing.assert_allclose(
            gradient[where].numpy(),
            want_gradient[reference.blocks[block]].numpy(),
            rtol=0.0,
            atol=1e-12,
        )


@pytest.mark.parametrize("name", list(_overlaps()))
@pytest.mark.oracle
def test_the_declared_gradient_is_autograd_s_through_the_value(name: str) -> None:
    # Fisher's identity: the compiled posteriors, then one backward pass of
    # the complete-data surrogate. Measured: value 2.1e-16 relative at worst,
    # gradient 3.8e-12 absolute at worst, on gradients up to 1.0e2.
    _, objective = _overlaps()[name]
    theta = _off_start(objective)

    value, gradient = value_and_gradient(objective, theta)
    want_value, want_gradient = autograd_value_and_gradient(objective, theta)

    assert float(value) == pytest.approx(float(want_value), rel=1e-10, abs=0.0)
    np.testing.assert_allclose(
        gradient.numpy(), want_gradient.numpy(), rtol=0.0, atol=1e-10
    )


def _independent_pair(
    beta_binomial: tuple[list[float], list[float]],
) -> tuple[Ragged, CountPairEmission]:
    """Independent count pairs over three segments, and the family that drew them."""
    rng = np.random.default_rng([SEED, 1])
    alpha, beta = beta_binomial
    family = CountPairEmission(
        [5.0, 12.0], [6.0, 30.0], alpha, beta, [20, 20], joint=False
    )
    states = rng.integers(0, 2, 150)
    values = np.asarray(family.sample(states, rng), dtype=np.float64)
    return Ragged(values, (30, 50, 70)), family


@pytest.mark.oracle
def test_independent_channels_score_as_one_forward_on_the_summed_densities() -> None:
    # The identity under test: with p(n, y | z) = NB(n | z) BB(y | z) under one
    # state path, the pair HMM's log-likelihood is the forward recursion run
    # on log NB(n | z) + log BB(y | z), each channel scored by its own family
    # alone. Not the sum of two HMMs: the channels share the path.
    # Measured: 0.0 relative.
    batch, family = _independent_pair(([2.0, 9.0], [8.0, 3.0]))
    objective = EmissionHmmObjective(batch, family)
    theta = _off_start(objective)
    named = objective.constrain(theta)
    totals = NegativeBinomialEmission(named["dispersion"], named["mean"])
    successes = BetaBinomialEmission([20, 20], named["alpha"], named["beta"])
    values = torch.as_tensor(batch.values)

    expected = forward_log_likelihood_ragged(
        totals.log_density(values[:, 0]) + successes.log_density(values[:, 1]),
        batch.lengths,
        named["log_initial"],
        named["log_transition"],
    )

    assert float(objective(theta)) == pytest.approx(-float(expected), rel=1e-12)


@pytest.mark.oracle
def test_a_state_blind_channel_adds_its_own_hmm_s_likelihood() -> None:
    # Where the success channel emits the same beta-binomial in every state,
    # sum_z p(path) prod_t NB_t(z) BB_t = prod_t BB_t sum_z p(path) prod_t
    # NB_t(z): the pair HMM is the sum of the two marginal HMMs, each a
    # per-family objective on its own channel. The beta-binomial HMM's
    # emissions are state-blind, so its value does not depend on the chain.
    # Measured: 9.4e-16 relative; at the same point with the success channel
    # left state-dependent the two differ by 5.0e-5 relative, the control.
    batch, family = _independent_pair(([4.0, 4.0], [6.0, 6.0]))
    rectangular = Ragged(batch.values[:150], (50, 50, 50))
    objective = EmissionHmmObjective(rectangular, family)
    # Off the start everywhere but the success channel, put back state-blind.
    pair = objective.theta_from(
        {
            **objective.constrain(_off_start(objective)),
            "alpha": torch.full((2,), 4.0, dtype=torch.float64),
            "beta": torch.full((2,), 6.0, dtype=torch.float64),
        }
    )
    counts = rectangular.values.reshape(3, 50, 2)
    totals = NegativeBinomialHmmObjective(counts[..., 0], 2, backend=Backend.TORCH)
    successes = BetaBinomialHmmObjective(
        counts[..., 1], 2, np.array([20, 20]), backend=Backend.TORCH
    )

    def marginals(theta: torch.Tensor) -> float:
        """The two per-family HMMs' values summed, at ``theta``'s parameters."""
        named = objective.constrain(theta)
        chain = {key: named[key] for key in ("log_initial", "log_transition")}
        first = {key: named[key] for key in ("dispersion", "mean")}
        second = {key: named[key] for key in ("alpha", "beta")}
        return float(
            totals(totals.theta_from({**chain, **first}))
            + successes(successes.theta_from({**chain, **second}))
        )

    assert float(objective(pair)) == pytest.approx(marginals(pair), rel=1e-12)
    control = _off_start(objective)
    assert float(objective(control)) != pytest.approx(marginals(control), rel=1e-6)


@pytest.mark.smoke
def test_the_blocks_partition_theta_and_the_start_is_uniform() -> None:
    _, objective = _overlaps()["negative_binomial"]
    assert isinstance(objective, DeclaredBlocks)
    blocks = objective.blocks
    stops = [0, *(block.stop for block in blocks.values())]
    assert list(blocks) == list(objective.constrain(objective.initial()))
    assert [block.start for block in blocks.values()] == stops[:-1]
    assert stops[-1] == objective.n_parameters
    start = objective.constrain(objective.initial())
    assert torch.equal(
        start["log_initial"], torch.full((2,), -math.log(2.0), dtype=torch.float64)
    )
    assert torch.equal(
        start["log_transition"], torch.full((2, 2), -math.log(2.0), dtype=torch.float64)
    )


@pytest.mark.smoke
def test_misaligned_inputs_are_refused() -> None:
    family = PoissonEmission([2.0, 6.0])
    with pytest.raises(ValueError, match="at least two states"):
        EmissionHmmObjective(np.zeros((2, 5)), PoissonEmission([2.0]))
    with pytest.raises(ValueError, match="does not align"):
        EmissionHmmObjective(np.zeros((2, 5)), family, covariate=np.ones((2, 4)))


@pytest.mark.oracle
def test_restricted_holds_the_transitions_and_keeps_the_compiled_gradient() -> None:
    # Bitwise against the full objective's declared route at the embedded
    # point, gathered onto `varied`; then a fit of the restriction leaves the
    # held transitions exactly where they were put.
    _, objective = _overlaps()["negative_binomial"]
    at = _off_start(objective)
    varied = coordinates(
        objective, [name for name in objective.blocks if name != "log_transition"]
    )
    restricted = Restricted(objective, at, varied)
    theta = restricted.initial() + 0.05

    value, gradient = value_and_gradient(restricted, theta)
    want_value, want_gradient = objective.value_and_gradient(restricted.embed(theta))
    result = fit(restricted)

    assert torch.equal(value, want_value)
    assert torch.equal(gradient, want_gradient[varied])
    assert result.converged
    assert torch.equal(
        restricted.constrain(result.theta)["log_transition"],
        objective.constrain(at)["log_transition"],
    )


def _quantile_start(name: str, values: torch.Tensor) -> EmissionFamily:
    """A start on the data: locations at quantiles, shapes at a pooled value."""
    if name == "negative_binomial":
        return NegativeBinomialEmission(
            [2.0, 2.0], quantile_locations(values, 2).clamp_min(0.5)
        )
    if name == "joint_pair":
        rate = float(values[:, 1].sum() / values[:, 0].sum())
        return CountPairEmission(
            [2.0, 2.0],
            quantile_locations(values[:, 0], 2).clamp_min(0.5),
            [rate, rate],
            [1.0 - rate, 1.0 - rate],
            joint=True,
        )
    return GaussianEmission(
        quantile_locations(values, 2), [float(values.std())] * 2, 1e-6
    )


TRUTHS: dict[str, EmissionFamily] = {
    "negative_binomial": NegativeBinomialEmission([4.0, 10.0], [3.0, 25.0]),
    "joint_pair": CountPairEmission(
        [6.0, 12.0], [10.0, 40.0], [2.0, 9.0], [8.0, 3.0], joint=True
    ),
    "gaussian": GaussianEmission([-1.0, 1.5], [0.7, 1.0], 1e-6),
}


@pytest.mark.parametrize("name", list(TRUTHS))
@pytest.mark.end2end
def test_the_planted_hmm_is_recovered_from_a_quantile_start(name: str) -> None:
    # Twenty segments of 50-249 positions drawn from the truth; the fit
    # starts at quantiles of the pooled data with uniform chain
    # probabilities. Every constrained parameter, aligned to the truth's
    # state order, lies within `Z_BOUND` delta-method standard errors.
    # Measured worst |z|: 1.45 (negative binomial), 2.19 (joint pair, a
    # log transition), 1.70 (Gaussian, a scale).
    truth = TRUTHS[name]
    data = simulate_sequences(
        HmmParams(2, LENGTHS, INITIAL, TRANSITION, truth, seed=SEED, tolerance=0.0)
    )
    batch = data.batch
    objective = EmissionHmmObjective(
        batch, _quantile_start(name, torch.as_tensor(batch.values, dtype=torch.float64))
    )

    result = fit(objective, include_intervals=True)

    assert result.converged
    assert result.standard_errors is not None
    fitted = objective.constrain(result.theta)
    order = list(align_families(objective.components(result.theta), truth))
    want = {
        "log_initial": torch.log(torch.as_tensor(INITIAL)),
        "log_transition": torch.log(torch.as_tensor(TRANSITION)),
        **truth.named_parameters(),
    }
    for key, value in want.items():
        estimate = fitted[key].detach()[order]
        error = result.standard_errors[key][order]
        if key == "log_transition":
            estimate, error = estimate[:, order], error[:, order]
        z = ((estimate - value) / error).abs()
        assert float(z.max()) <= Z_BOUND, (key, z)


@pytest.mark.smoke
def test_a_categorical_family_is_scored_on_integer_symbols() -> None:
    family = CategoricalEmission.from_log(
        torch.log(torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]], dtype=torch.float64))
    )
    objective = EmissionHmmObjective(Ragged(np.array([0, 2, 1, 2, 0]), (2, 3)), family)
    assert objective.observations.dtype == torch.long
    assert math.isfinite(float(objective(objective.initial())))


@pytest.mark.smoke
def test_a_binomial_family_keeps_its_trials() -> None:
    family = BinomialEmission([10, 10], [0.2, 0.7])
    objective = EmissionHmmObjective(np.array([[1, 8, 7], [2, 3, 9]]), family)
    rebuilt = objective.components(objective.initial())
    assert isinstance(rebuilt, BinomialEmission)
    assert torch.equal(rebuilt.trials, family.trials)
