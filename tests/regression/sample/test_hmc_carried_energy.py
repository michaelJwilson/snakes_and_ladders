"""The Hamiltonian chain evaluates ``U`` and ``grad U`` nowhere it has already evaluated them (issues #1217, #1222).

A transition reads ``(U, grad U)`` at the current point and at the proposal.
The first pair is carried from the transition that landed there and the
second is the integrator's last force evaluation's, so a proposal costs the
trajectory's force evaluations after its first and nothing beside them. Each
carried value is pinned to the objective's forward at the point it names:
bitwise where both come from one arithmetic, and at :data:`ROUND_OFF` where
the value is a compiled E step's and the forward is torch's recursion. Each
carried gradient is pinned bitwise to :func:`~sal.sample.chain.gradient_at`
there, which is what the trajectory's first kick read before the carry.
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
from sal.sample.chain import gradient_at
from sal.sample.hmc import (
    Adaptation,
    _HamiltonianKernel,
    _transition,
    anneal,
    leapfrog,
    parallel_tempering,
    sample,
)
from sal.sample.langevin import mala
from sal.sample.schedule import ConstantTempSchedule
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


def _assert_carried(objective: Objective, step_size: float, tolerance: float) -> None:
    """The pair handed from one transition to the next is ``(U, grad U)`` where it lands.

    Accepted or not: ``U`` against the forward there, bitwise at
    ``tolerance`` 0; ``grad U`` against :func:`gradient_at` there, bitwise.
    """
    generator = torch.Generator().manual_seed(3)
    position = objective.initial()
    potential = None
    force = None
    accepted = 0
    for _ in range(TRANSITIONS):
        step, potential, force = _transition(
            objective,
            position,
            1.0,
            generator,
            step_size,
            4,
            leapfrog,
            potential=potential,
            force=force,
        )
        position = step.position
        accepted += step.accepted
        expected = float(objective(position))
        if tolerance == 0.0:
            assert potential == expected
        else:
            assert potential == pytest.approx(expected, rel=tolerance, abs=0.0)
        assert force is not None
        assert torch.equal(force, gradient_at(objective, position))
    # Both branches of the carry are exercised: an accepted end point, and a
    # rejected one that keeps the current point's value.
    assert 0 < accepted < TRANSITIONS


@pytest.mark.analytic
@pytest.mark.parametrize(
    ("build", "step_size"),
    [(lambda: GAUSSIAN, 1.0), (lambda: Rosenbrock(2), 0.04)],
    ids=["gaussian", "rosenbrock"],
)
def test_the_carried_energy_is_the_objective_bitwise_on_one_arithmetic(
    build: Callable[[], Objective], step_size: float
) -> None:
    _assert_carried(build(), step_size, 0.0)


@pytest.mark.analytic
@pytest.mark.mixture
def test_the_carried_energy_is_the_mixture_objective_bitwise() -> None:
    # No declared gradient: value and gradient come from one autograd pass.
    _assert_carried(_mixture(), 0.05, 0.0)


@pytest.mark.analytic
@pytest.mark.hmm
def test_the_carried_energy_is_the_hmm_objective_at_round_off() -> None:
    # The compiled E step's evidence against the torch forward recursion.
    _assert_carried(_hmm(), 0.08, ROUND_OFF)


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
    # The ratio's two values come with the gradients: one at the start, then
    # the proposal's alone (issue #1222).
    counted = Counted(GAUSSIAN)

    chain = mala(counted, torch.Generator().manual_seed(4), 30, step_size=0.5)

    assert counted.calls == chain.spent == 1 + 30


@pytest.mark.smoke
def test_an_adapted_chain_evaluates_the_gradient_once_at_each_start() -> None:
    # Issue #1222: the warm-up's two windows and the chain each start the
    # kernel on a new objective, which evaluates `(U, grad U)` there once;
    # every other gradient is a trajectory's own after its first kick.
    counted = Counted(GAUSSIAN)
    adaptation = Adaptation(warmup=16, target_acceptance=0.65, step_jitter=0.2)

    chain = sample(
        counted,
        torch.Generator().manual_seed(4),
        20,
        step_size=0.3,
        n_steps=5,
        burn_in=4,
        adaptation=adaptation,
    )

    assert chain.spent == 3 + (16 + 4 + 20) * leapfrog.force_evaluations(
        5, carried=True
    )
    assert counted.calls == 3 + chain.spent

    counted = Counted(GAUSSIAN)
    mala(
        counted,
        torch.Generator().manual_seed(4),
        20,
        step_size=0.5,
        burn_in=4,
        adaptation=adaptation,
    )
    assert counted.calls == 3 + 16 + 4 + 20


@pytest.mark.smoke
def test_annealing_evaluates_the_gradient_once_at_its_start() -> None:
    # One value at the start; one gradient there; then the trajectory's own.
    counted = Counted(GAUSSIAN)

    run = anneal(
        counted,
        ConstantTempSchedule(2.0, 30),
        torch.Generator().manual_seed(4),
        step_size=0.3,
        n_steps=5,
    )

    assert run.spent == 1 + 30 * leapfrog.force_evaluations(5, carried=True)
    assert counted.calls == 1 + run.spent


@pytest.mark.smoke
def test_tempering_moves_the_carried_gradient_with_the_position() -> None:
    # Issue #1222: an exchange swaps the tensors, and their gradients with
    # them, so a replica costs one gradient at its start beside the
    # trajectories' own; swaps are accepted, so the carry crossed rungs.
    counted = Counted(GAUSSIAN)
    ladder = (1.0, 2.0, 4.0)

    run = parallel_tempering(
        counted, ladder, torch.Generator().manual_seed(6), 20, step_size=0.3, n_steps=4
    )

    assert run.spent == 3 + 20 * 3 * leapfrog.force_evaluations(4, carried=True)
    assert counted.calls == 1 + run.spent
    assert bool((run.swap_acceptance > 0.0).all()), run.swap_acceptance


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
