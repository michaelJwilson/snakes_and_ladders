"""The Hamiltonian chain evaluates ``U`` nowhere it has already evaluated it (issue #1217).

A transition reads ``U`` at the current point and at the proposal. The first
is carried from the transition that landed there and the second is the
integrator's last force evaluation's, so a proposal costs its force
evaluations and nothing beside them. Each carried value is pinned to the
objective's forward at the point it names: bitwise where both come from one
arithmetic, and at :data:`ROUND_OFF` where the value is a compiled E step's
and the forward is torch's recursion.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch
from sal.emissions import GaussianEmission
from sal.opt.hmm import EmissionHmmObjective
from sal.opt.mixture import GaussianMixtureObjective
from sal.opt.objective import Objective
from sal.opt.testfunctions import Rosenbrock
from sal.sample.hmc import _HamiltonianKernel, _transition, leapfrog, sample
from sal.sample.langevin import mala
from sal.search.hmm_starts import gaussian_quantile_start
from sal.sim.hmm import HmmParams, simulate_sequences

from tests._objective_checks import Counted
from tests._posteriors import GAUSSIAN

#: Relative agreement of the HMM's compiled value with its torch forward:
#: the two sum the same terms in a different order. Measured at 4.0e-16 over
#: 30 transitions on the 4-state, 12-segment instance of `test_hmm_starts`.
ROUND_OFF = 1e-10

#: Transitions each carried value is checked over.
TRANSITIONS = 25


def _hmm() -> EmissionHmmObjective:
    """A 3-state Gaussian HMM on 6 segments, started at its quantiles."""
    n_states = 3
    lengths = (40, 55, 30, 70, 45, 60)
    stay = 0.9
    off = (1.0 - stay) / (n_states - 1)
    transition = np.full((n_states, n_states), off) + np.eye(n_states) * (stay - off)
    emission = GaussianEmission(
        np.arange(n_states, dtype=float), np.full(n_states, 0.7), 1e-6
    )
    data = simulate_sequences(
        HmmParams(
            n_states,
            lengths,
            np.full(n_states, 1.0 / n_states),
            transition,
            emission,
            1217,
            0.0,
        )
    )
    return EmissionHmmObjective(
        data.batch, gaussian_quantile_start(data.batch.values, n_states, 1e-6)
    )


def _mixture() -> GaussianMixtureObjective:
    """Two components 4 apart, 150 observations each."""
    rng = np.random.default_rng(1217)
    return GaussianMixtureObjective(
        np.concatenate([rng.normal(-2.0, 1.0, 150), rng.normal(2.0, 1.0, 150)]), 2
    )


#: ``(objective, step size, relative tolerance)``: 0 is bitwise.
CASES: dict[str, tuple[Callable[[], Objective], float, float]] = {
    "gaussian": (lambda: GAUSSIAN, 1.0, 0.0),
    "rosenbrock": (lambda: Rosenbrock(2), 0.04, 0.0),
    "mixture": (_mixture, 0.05, 0.0),
    "hmm": (_hmm, 0.08, ROUND_OFF),
}


@pytest.mark.analytic
@pytest.mark.parametrize("name", sorted(CASES))
def test_the_carried_energy_is_the_objective_where_the_step_landed(name: str) -> None:
    # The value handed from one transition to the next is `U` at the
    # position it hands over, accepted or not, against the forward there.
    build, step_size, tolerance = CASES[name]
    objective = build()
    generator = torch.Generator().manual_seed(3)
    position = objective.initial()
    potential = None
    accepted = 0
    for _ in range(TRANSITIONS):
        step, potential = _transition(
            objective,
            position,
            1.0,
            generator,
            step_size,
            4,
            leapfrog,
            potential=potential,
        )
        position = step.position
        accepted += step.accepted
        expected = float(objective(position))
        if tolerance == 0.0:
            assert potential == expected
        else:
            assert potential == pytest.approx(expected, rel=tolerance, abs=0.0)
    # Both branches of the carry are exercised: an accepted end point, and a
    # rejected one that keeps the current point's value.
    assert 0 < accepted < TRANSITIONS


@pytest.mark.smoke
def test_a_chain_evaluates_the_objective_only_for_its_forces() -> None:
    # One value at the start; then per proposal the trajectory's force
    # evaluations alone. Two further values a proposal were taken before.
    counted = Counted(GAUSSIAN)

    chain = sample(
        counted, torch.Generator().manual_seed(4), 30, step_size=0.3, n_steps=5
    )

    assert counted.calls == 1 + chain.spent


@pytest.mark.smoke
def test_mala_evaluates_the_objective_only_for_its_gradients() -> None:
    # The ratio's two values come with the two gradients a proposal costs.
    counted = Counted(GAUSSIAN)

    chain = mala(counted, torch.Generator().manual_seed(4), 30, step_size=0.5)

    assert counted.calls == chain.spent


@pytest.mark.smoke
def test_a_change_of_objective_is_evaluated_afresh() -> None:
    # The carry is keyed on the objective's and the tensor's identity, so a
    # second objective at the same tensor is not handed the first one's value.
    kernel = _HamiltonianKernel(n_steps=3, integrator=leapfrog)
    generator = torch.Generator().manual_seed(5)
    position = GAUSSIAN.initial()
    step = kernel(GAUSSIAN, position, 1.0, generator, 0.3)
    counted = Counted(GAUSSIAN)

    kernel(counted, step.position, 1.0, generator, 0.3)

    assert counted.calls == 1 + leapfrog.force_evaluations(3)
