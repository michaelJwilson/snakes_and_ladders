"""`opt.objective.Restricted` and the blocks it is built from (issue #1168).

Referees: the full objective at the embedded point, whose value and gradient
the restriction must reproduce bitwise with the gradient gathered onto
``varied``; autograd through the full objective, which a declared route is
pinned to; and the count mixture's compiled kernel, counted, which must stay
the route inside the restriction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from sal.backend import Backend
from sal.emissions import (
    BetaBinomialEmission,
    BinomialEmission,
    CategoricalEmission,
    CountPairEmission,
    GaussianEmission,
    NegativeBinomialEmission,
    PoissonEmission,
)
from sal.opt.emission_mixture import EmissionMixtureObjective
from sal.opt.hmm import (
    EmissionHmmObjective,
    family_start,
)
from sal.opt.objective import (
    DeclaredBlocks,
    Objective,
    Restricted,
    autograd_value_and_gradient,
    coordinates,
    declares_gradient,
    energy_of,
    value_and_gradient,
)
from sal.sample.declared import declared_energy, declared_jax_energy

PAIRS = CountPairEmission([6.0, 12.0], [20.0, 60.0], [2.0, 9.0], [8.0, 3.0], joint=True)


def _pairs() -> np.ndarray:
    rng = np.random.default_rng([1168, 0])
    return np.asarray(PAIRS.sample(rng.choice(2, size=200, p=[0.4, 0.6]), rng))


def _hmms(backend: Backend) -> dict[str, Objective]:
    """One objective per HMM family, two states, on small seeded data."""
    rng = np.random.default_rng([1168, 1])
    counts = rng.poisson(4.0, size=(2, 30))
    trials = np.array([12, 12])
    successes = rng.binomial(12, 0.3, size=(2, 30))
    symbols = rng.integers(0, 3, size=(2, 30))
    reals = rng.normal(size=(2, 30))
    return {
        "categorical": EmissionHmmObjective(
            symbols,
            family_start(CategoricalEmission, symbols, 2, n_symbols=3),
            backend=backend,
        ),
        "gaussian": EmissionHmmObjective(
            reals, family_start(GaussianEmission, reals, 2), backend=backend
        ),
        "poisson": EmissionHmmObjective(
            counts, family_start(PoissonEmission, counts, 2), backend=backend
        ),
        "binomial": EmissionHmmObjective(
            successes,
            family_start(BinomialEmission, successes, 2, trials=trials),
            backend=backend,
        ),
        "beta_binomial": EmissionHmmObjective(
            successes,
            family_start(BetaBinomialEmission, successes, 2, trials=trials),
            backend=backend,
        ),
        "negative_binomial": EmissionHmmObjective(
            counts, family_start(NegativeBinomialEmission, counts, 2), backend=backend
        ),
    }


def _blocked() -> dict[str, Objective]:
    return {
        **_hmms(Backend.TORCH),
        "emission_mixture": EmissionMixtureObjective(_pairs(), PAIRS),
    }


def _point(objective: Objective) -> torch.Tensor:
    """A point off the start, so no coordinate sits at a symmetric value."""
    start = objective.initial()
    return start + torch.linspace(-0.3, 0.2, start.numel(), dtype=torch.float64)


@pytest.mark.parametrize("name", list(_blocked()))
@pytest.mark.smoke
def test_the_blocks_partition_theta_under_constrain_s_keys(name: str) -> None:
    # The contract `coordinates` reads: one block per named parameter,
    # disjoint, covering `theta` in order, and moving one block moves that
    # parameter alone.
    objective = _blocked()[name]
    assert isinstance(objective, DeclaredBlocks)
    blocks = objective.blocks
    at = _point(objective)
    before = objective.constrain(at)
    assert list(blocks) == list(before)
    stops = [0, *(block.stop for block in blocks.values())]
    assert [block.start for block in blocks.values()] == stops[:-1]
    assert stops[-1] == at.numel()
    for block in blocks:
        moved = at.clone()
        moved[coordinates(objective, block)] += 0.1
        after = objective.constrain(moved)
        changed = [key for key in before if not torch.equal(before[key], after[key])]
        assert changed == [block]


@pytest.mark.parametrize("name", list(_blocked()))
@pytest.mark.oracle
def test_value_and_gradient_are_the_full_objective_s_at_the_embedded_point(
    name: str,
) -> None:
    # Bitwise against the full objective at `at` with the varied block set:
    # its value, and its gradient gathered onto `varied`. The full gradient
    # is autograd's through the full objective --- every HMM here is on the
    # torch backend --- and the mixture's is its compiled kernel's, which
    # its own module pins to autograd.
    objective = _blocked()[name]
    at = _point(objective)
    held = next(iter(objective.blocks))  # type: ignore[attr-defined]
    varied = coordinates(
        objective,
        [block for block in objective.blocks if block != held],  # type: ignore[attr-defined]
    )
    restricted = Restricted(objective, at, varied)
    theta = restricted.initial() + 0.05
    full = at.clone()
    full[varied] = theta

    value, gradient = value_and_gradient(restricted, theta)
    expected_value, expected_gradient = value_and_gradient(objective, full)

    assert torch.equal(value, expected_value)
    assert torch.equal(gradient, expected_gradient[varied])
    # `gradient` alone takes the declared gradient where there is one, as
    # `sample.chain.gradient_at` does: the Gaussian HMM's closed form.
    alone = (
        objective.gradient(full)  # type: ignore[attr-defined]
        if declares_gradient(objective)
        else expected_gradient
    )
    assert torch.equal(restricted.gradient(theta), alone[varied])
    assert energy_of(restricted, theta.numpy()) == energy_of(objective, full.numpy())
    if name != "emission_mixture":
        autograd = autograd_value_and_gradient(objective, full)
        assert torch.equal(gradient, autograd[1][varied])


@pytest.mark.oracle
def test_autograd_through_the_restriction_is_the_gathered_full_gradient() -> None:
    # The restriction's own graph, through `index_copy`, against the full
    # objective's autograd gathered: what a consumer that differentiates
    # `__call__` reads, bitwise.
    objective = _hmms(Backend.TORCH)["negative_binomial"]
    at = _point(objective)
    varied = coordinates(objective, ["log_transition", "mean"])
    restricted = Restricted(objective, at, varied)
    theta = restricted.initial() - 0.1
    full = at.index_copy(0, varied, theta)

    value, gradient = autograd_value_and_gradient(restricted, theta)
    expected_value, expected_gradient = autograd_value_and_gradient(objective, full)

    assert torch.equal(value, expected_value)
    assert torch.equal(gradient, expected_gradient[varied])


@pytest.mark.smoke
@pytest.mark.patch
def test_a_compiled_gradient_stays_compiled_inside_the_restriction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The count mixture's kernel, counted: one call per value and gradient
    # and one per energy, none falling back to autograd, and its output
    # gathered bitwise.
    import sal.opt.emission_mixture as module
    from sal import oxisal

    calls = 0
    kernel = oxisal.count_mixture_value_and_gradient

    def counted(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return kernel(*args, **kwargs)

    def refused(*_args: object, **_kwargs: object) -> object:
        msg = "autograd was taken where the kernel applies"
        raise AssertionError(msg)

    monkeypatch.setattr(oxisal, "count_mixture_value_and_gradient", counted)
    monkeypatch.setattr(module, "autograd_value_and_gradient", refused)
    objective = EmissionMixtureObjective(_pairs(), PAIRS)
    at = _point(objective)
    varied = coordinates(objective, ["log_weight", "mean"])
    restricted = Restricted(objective, at, varied)
    theta = restricted.initial()

    value, gradient = value_and_gradient(restricted, theta)
    restricted.gradient(theta)
    energy = energy_of(restricted, theta.numpy())
    assert calls == 3

    expected_value, expected_gradient = objective.value_and_gradient(at)
    assert torch.equal(value, expected_value)
    assert torch.equal(gradient, expected_gradient[varied])
    assert energy == float(expected_value)


@pytest.mark.smoke
def test_no_full_dimensional_declaration_reaches_a_compiling_consumer() -> None:
    # `sample.declared` compiles an energy from these by attribute; on the
    # restriction it would be the full objective's, of the wrong dimension.
    observations = np.random.default_rng([1168, 2]).normal(size=(2, 30))
    objective = EmissionHmmObjective(
        observations, family_start(GaussianEmission, observations, 2)
    )
    assert declared_jax_energy(objective) is not None
    restricted = Restricted(
        objective, objective.initial(), coordinates(objective, "mean")
    )
    assert declared_jax_energy(restricted) is None
    assert declared_energy(restricted) is None
    assert not hasattr(restricted, "gaussian_hmm_declaration")


@pytest.mark.smoke
def test_the_restriction_starts_where_it_was_taken_and_inverts() -> None:
    objective = _hmms(Backend.TORCH)["poisson"]
    at = _point(objective)
    varied = coordinates(objective, ["mean", "log_initial"])
    restricted = Restricted(objective, at, varied)
    assert varied.tolist() == [0, 3, 4]
    assert torch.equal(restricted.initial(), at[varied])
    named = restricted.constrain(restricted.initial())
    assert torch.equal(
        restricted.theta_from(named), objective.theta_from(named)[varied]
    )


@pytest.mark.smoke
def test_malformed_arguments_are_refused() -> None:
    objective = _hmms(Backend.TORCH)["poisson"]
    at = objective.initial()
    with pytest.raises(ValueError, match="1-D"):
        Restricted(objective, at[None, :], torch.tensor([0]))
    with pytest.raises(ValueError, match="integer"):
        Restricted(objective, at, torch.tensor([0.0]))
    with pytest.raises(ValueError, match="outside"):
        Restricted(objective, at, torch.tensor([at.numel()]))
    with pytest.raises(ValueError, match="repeats"):
        Restricted(objective, at, torch.tensor([1, 1]))
    with pytest.raises(ValueError, match="unknown block"):
        coordinates(objective, ["rate"])
    with pytest.raises(TypeError, match="declares no blocks"):
        coordinates(object(), ["mean"])
