"""Every family declares its parameters' domains and rebuilds itself from them (issue #1164).

Referees: the family itself --- ``with_parameters(named_parameters())`` scores
every observation of a grid bit for bit as the family it was read from, and
keeps its constants; and the inverse map of each
:class:`~sal.emissions.Domain` --- ``free_from(constrained(theta))`` returns
``theta`` to 1e-15, scaled for the logit by the conditioning of ``1 - p``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch
from sal.emissions import (
    BetaBinomialEmission,
    BinomialEmission,
    CategoricalEmission,
    CountPairEmission,
    Domain,
    EmissionFamily,
    GaussianEmission,
    NegativeBinomialEmission,
    PoissonEmission,
)
from sal.opt.constrain import constrained, free_from, free_shape
from sal.sim.count_pairs import IndependentCountPair, ReflectedEmission

SEED = 1164

#: Integer counts up to 60, for every one-channel count family.
COUNTS = torch.arange(61, dtype=torch.float64)

#: Every pair of a total up to 30 and successes up to 30, impossible ones
#: included: a rebuilt family must score the support's edge as the original.
PAIRS = torch.cartesian_prod(
    torch.arange(31, dtype=torch.float64), torch.arange(31, dtype=torch.float64)
)


def _families() -> dict[str, tuple[EmissionFamily, torch.Tensor]]:
    """Each family, every constant it carries set away from its default, and a grid it scores."""
    independent = IndependentCountPair(
        NegativeBinomialEmission([4.0, 9.0], [10.0, 25.0]),
        BetaBinomialEmission([30.0, 30.0], [2.0, 7.0], [6.0, 2.0]),
    )
    return {
        "categorical": (
            CategoricalEmission(np.array([[0.2, 0.5, 0.3], [0.6, 0.1, 0.3]])),
            torch.arange(3),
        ),
        "gaussian": (
            GaussianEmission([-1.5, 2.0], [0.5, 1.5], variance_floor=1e-3),
            torch.linspace(-6.0, 6.0, 121, dtype=torch.float64),
        ),
        "gaussian, two channels": (
            GaussianEmission(
                np.array([[-1.0, 0.5], [2.0, -3.0]]),
                np.array([[0.5, 1.0], [1.5, 0.25]]),
                1e-3,
            ),
            torch.cartesian_prod(
                torch.linspace(-4.0, 4.0, 17, dtype=torch.float64),
                torch.linspace(-4.0, 4.0, 17, dtype=torch.float64),
            ),
        ),
        "negative binomial": (
            NegativeBinomialEmission([3.0, 8.0, 20.0], [5.0, 30.0, 45.0]),
            COUNTS,
        ),
        "negative binomial, tied": (
            NegativeBinomialEmission([6.0, 6.0], [5.0, 40.0], tied=True),
            COUNTS,
        ),
        "poisson": (PoissonEmission([2.0, 15.0]), COUNTS),
        "binomial": (BinomialEmission([60.0, 40.0], [0.2, 0.7]), COUNTS),
        "beta-binomial": (
            BetaBinomialEmission([60.0, 40.0], [2.0, 7.0], [6.0, 2.0]),
            COUNTS,
        ),
        "beta-binomial, tied": (
            BetaBinomialEmission([60.0, 60.0], [2.0, 6.0], [6.0, 2.0], tied=True),
            COUNTS,
        ),
        "count pair, joint": (
            CountPairEmission(
                [6.0, 12.0], [20.0, 60.0], [2.0, 9.0], [8.0, 3.0], joint=True
            ),
            PAIRS,
        ),
        "count pair, independent": (
            CountPairEmission(
                [6.0, 12.0],
                [20.0, 60.0],
                [2.0, 9.0],
                [8.0, 3.0],
                [25.0, 30.0],
                joint=False,
            ),
            PAIRS,
        ),
        "independent count pair": (independent, PAIRS),
        "reflected beta-binomial": (
            ReflectedEmission(
                BetaBinomialEmission([30.0, 30.0], [2.0, 7.0], [6.0, 2.0])
            ),
            COUNTS[:31],
        ),
        "reflected count pair": (ReflectedEmission(independent), PAIRS),
    }


#: Each domain's membership test, so a declared domain is checked against the
#: values the family holds rather than restated.
MEMBER: dict[Domain, Callable[[torch.Tensor], bool]] = {
    Domain.REAL: lambda v: bool(torch.isfinite(v).all()),
    Domain.POSITIVE: lambda v: bool((v > 0.0).all()),
    Domain.PROBABILITY: lambda v: bool(((v > 0.0) & (v < 1.0)).all()),
    Domain.LOG_SIMPLEX: lambda v: bool(
        torch.allclose(torch.logsumexp(v, dim=-1), torch.zeros(v.shape[:-1]).double())
    ),
}


@pytest.mark.oracle
@pytest.mark.parametrize("name", list(_families()))
def test_a_family_rebuilt_at_its_own_parameters_scores_bitwise(name: str) -> None:
    family, grid = _families()[name]
    named = family.named_parameters()

    rebuilt = family.with_parameters(named)

    assert type(rebuilt) is type(family)
    assert torch.equal(rebuilt.log_density(grid), family.log_density(grid))
    # The constants are kept: a draw reads the trial count, the form and the
    # stored probabilities, so it is the original's bit for bit.
    rng, again = np.random.default_rng(SEED), np.random.default_rng(SEED)
    states = np.arange(family.n_states).repeat(50)
    assert np.array_equal(
        np.asarray(rebuilt.sample(states, rng)),
        np.asarray(family.sample(states, again)),
    )
    for key in ("tied", "joint", "trials", "variance_floor"):
        kept, held = getattr(rebuilt, key, None), getattr(family, key, None)
        assert (kept is None and held is None) or torch.equal(
            torch.as_tensor(kept), torch.as_tensor(held)
        ), key


@pytest.mark.oracle
@pytest.mark.parametrize("name", list(_families()))
def test_each_declared_domain_holds_the_family_s_values(name: str) -> None:
    family, _ = _families()[name]
    named, domains = family.named_parameters(), family.parameter_domains()

    assert list(domains) == list(named)
    for key, domain in domains.items():
        value = torch.as_tensor(named[key], dtype=torch.float64)
        assert MEMBER[domain](value), key
        # The free coordinates map back onto the family's own value.
        np.testing.assert_allclose(
            constrained(domain, free_from(domain, value)).numpy(),
            value.numpy(),
            rtol=1e-14,
            atol=0,
        )


@pytest.mark.oracle
@pytest.mark.parametrize("domain", list(Domain))
def test_each_domain_map_round_trips_theta(domain: Domain) -> None:
    generator = torch.Generator().manual_seed(SEED)
    theta = torch.empty(free_shape(domain, (64, 5)), dtype=torch.float64).uniform_(
        -3.0, 3.0, generator=generator
    )

    back = free_from(domain, constrained(domain, theta))

    error = (back - theta).abs()
    if domain is Domain.PROBABILITY:
        # ``1 - p`` cancels as ``|theta|`` grows, by a factor ``1 + e^|theta|``:
        # measured, the error is at most 2.4e-16 of it on [-5, 5].
        assert bool((error <= 1e-15 * (1.0 + theta.abs().exp())).all())
    else:
        # Measured: 0 for REAL, 5.6e-17 for POSITIVE, 6.7e-16 for LOG_SIMPLEX.
        assert float(error.max()) <= 1e-15


@pytest.mark.smoke
def test_a_rebuild_refuses_a_name_the_family_does_not_carry() -> None:
    family = NegativeBinomialEmission([3.0, 8.0], [5.0, 30.0])
    with pytest.raises(ValueError, match="parameterized by"):
        family.with_parameters({"mean": torch.ones(2)})
    with pytest.raises(ValueError, match="parameterized by"):
        family.with_parameters({**family.named_parameters(), "scale": torch.ones(2)})
