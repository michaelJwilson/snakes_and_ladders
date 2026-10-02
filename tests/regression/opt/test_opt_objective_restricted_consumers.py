"""`opt.objective.Restricted` handed to the consumers of an `Objective` (issue #1168).

Shared machinery, so the target is a bowl written here and no problem: a
consumer that read ``theta``'s length from anywhere but the objective, or
dropped a declared route, fails here whatever model is restricted. Referees:
the bowl's minimum, which is known; the held coordinates, which come back
exactly as they went in; and the bowl's own call counts, which say the
declared gradient was the route taken.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest
import torch
from sal.opt.fit import fit
from sal.opt.objective import Objective, Restricted
from sal.sample.hmc import anneal, parallel_tempering, sample
from sal.sample.schedule import ConstantTempSchedule

CENTRE = torch.tensor([1.0, -2.0, 3.0, 0.5], dtype=torch.float64)
WEIGHT = torch.tensor([1.0, 2.0, 4.0, 0.5], dtype=torch.float64)
AT = torch.tensor([0.25, 0.0, -1.0, 0.0], dtype=torch.float64)
VARIED = torch.tensor([1, 3])
HELD = [0, 2]


class _Bowl(Objective):
    """``sum(w (theta - c)^2) / 2``, declaring its gradient and its energy, and counting both."""

    def __init__(self) -> None:
        self.gradients = 0
        self.energies = 0

    def initial(self) -> torch.Tensor:
        return torch.zeros(4, dtype=torch.float64)

    def constrain(self, theta: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return {"x": theta}

    def theta_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return named["x"]

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        return 0.5 * (WEIGHT * (theta - CENTRE) ** 2).sum()

    def gradient(self, theta: torch.Tensor) -> torch.Tensor:
        self.gradients += 1
        return WEIGHT * (theta - CENTRE)

    def energy(self, x: np.ndarray) -> float:
        self.energies += 1
        return float(0.5 * (WEIGHT.numpy() * (x - CENTRE.numpy()) ** 2).sum())


def _held_exactly(restricted: Restricted, theta: torch.Tensor) -> bool:
    full = restricted.constrain(theta)["x"]
    return bool(torch.equal(full[HELD], AT[HELD]))


@pytest.mark.infra
def test_fit_reaches_the_minimum_on_the_varied_coordinates_alone() -> None:
    # L-BFGS on the restriction: the minimum on the varied coordinates is the
    # centre's, the bowl being separable, and the declared gradient is the
    # route its closure takes.
    bowl = _Bowl()
    restricted = Restricted(bowl, AT, VARIED)

    result = fit(restricted)

    assert result.converged
    assert result.theta.shape == (2,)
    torch.testing.assert_close(result.theta, CENTRE[VARIED], rtol=0.0, atol=1e-8)
    assert _held_exactly(restricted, result.theta)
    assert bowl.gradients > 0


@pytest.mark.infra
def test_hamiltonian_sampling_draws_in_the_varied_dimension() -> None:
    # The chain's dimension is the restriction's, its kicks the bowl's
    # declared gradient, and the held coordinates never move. The draws'
    # mean is the centre's within five independent-draw standard errors;
    # seed 1168 realizes 0.07 and 0.01 of one.
    bowl = _Bowl()
    restricted = Restricted(bowl, AT, VARIED)

    chain = sample(
        restricted,
        torch.Generator().manual_seed(1168),
        2000,
        step_size=0.3,
        n_steps=8,
        burn_in=200,
    )

    assert chain.draws.shape == (2000, 2)
    assert bowl.gradients > 0
    standard_error = 1.0 / torch.sqrt(WEIGHT[VARIED] * 2000)
    deviation = (chain.draws.mean(0) - CENTRE[VARIED]).abs()
    assert bool((deviation < 5.0 * standard_error).all())
    assert _held_exactly(restricted, chain.draws[-1])


@pytest.mark.infra
def test_annealing_and_tempering_run_on_the_restriction() -> None:
    bowl = _Bowl()
    restricted = Restricted(bowl, AT, VARIED)

    annealed = anneal(
        restricted,
        ConstantTempSchedule(0.01, 200),
        rng=torch.Generator().manual_seed(7),
        step_size=0.1,
        n_steps=8,
    )
    tempered = parallel_tempering(
        restricted,
        [1.0, 2.0, 4.0],
        rng=torch.Generator().manual_seed(8),
        n_rounds=50,
        step_size=0.2,
        n_steps=8,
    )

    for best in (annealed.best, tempered.best):
        assert best.shape == (2,)
        assert _held_exactly(restricted, best)
    # The lowest point visited at temperature 0.01, where one standard
    # deviation `sqrt(T / w)` is 0.14 on the flattest varied axis; seed 7
    # realizes 0.0099.
    assert float((annealed.best - CENTRE[VARIED]).abs().max()) < 0.1
    assert bowl.gradients > 0
