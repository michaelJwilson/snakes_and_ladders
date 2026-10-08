"""A step chosen by a pilot over a log grid (issue #1219).

Referees: on an isotropic and an anisotropic Gaussian the step the ESJD per
gradient chooses is within :data:`WITHIN` of the brute-force optimum, read
off fixed-step chains of :func:`~sal.sample.hmc.sample` on a finer grid
(`oracle`; the full-size grid and chains at release); a one-candidate pilot
is :func:`~sal.sample.hmc.anneal` at that step, bitwise (`smoke`, `patch`); every
sampler given ``step_size="auto"`` charges its pilot in ``spent``, to the
gradient, and its polishes in ``polish_iterations`` (`analytic`); on a 1-D
double well the polished gap ranks a step that stays in the deep basin above
one that crosses into the shallow one, which the lowest energy ranks first
(`analytic`, issue #1251); and the refusals (`smoke`).
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import numpy as np
import pytest
import torch
from sal.cost import Cost
from sal.opt.budget import Budget
from sal.opt.objective import Objective
from sal.opt.starts import PolishedPoint
from sal.opt.termination import Termination
from sal.sample.hmc import (
    Adaptation,
    anneal,
    hamiltonian_pilot,
    leapfrog,
    parallel_tempering,
    sample,
)
from sal.sample.initialize import (
    FromAnnealing,
    FromChain,
    FromTempering,
    LadderCalibration,
    LadderRule,
)
from sal.sample.loop import Moved
from sal.sample.schedule import ConstantTempSchedule, ExponentialTempSchedule
from sal.sample.tune import (
    AUTO,
    Criterion,
    Pilot,
    StepTuning,
    compress,
    tune_step,
)

from tests._objective_checks import AnalyticGaussian

#: Dimension of both Gaussians and leapfrog steps per proposal.
DIMENSION = 10
N_STEPS = 5

#: Unit scales, and scales log-spaced from 0.1 to 1: the stiff direction
#: sets the stability limit, the soft one the distance worth travelling.
ISOTROPIC = (1.0,) * DIMENSION
ANISOTROPIC = tuple(float(scale) for scale in np.geomspace(0.1, 1.0, DIMENSION))

#: The candidates: quarter-decade steps from 1e-2 to 10 ** 0.25, so the
#: grid's own spacing puts its nearest point within 10 ** 0.125 = 1.33 of
#: any optimum inside it.
CANDIDATES = tuple(float(10.0**k) for k in np.arange(-2.0, 0.5, 0.25))

#: The pilot's budget in gradients: 39 proposals per candidate.
PILOT = Budget(Cost.GRADIENTS, 2000)

#: How far the chosen step may sit from the brute-force optimum, as a ratio
#: either way: the issue's 1.5.
WITHIN = 1.5

#: The per-PR reference: eighth-decade steps over the candidates' range, one
#: chain of 300 proposals each. The release reference: sixteenth-decade, two
#: chains of 2,000 each, 100 s per target where this is 5 s.
REFERENCE_GRID = tuple(float(10.0**k) for k in np.arange(-1.5, 0.375, 0.125))
REFERENCE_PROPOSALS = 300
RELEASE_GRID = tuple(float(10.0**k) for k in np.arange(-2.0, 0.5, 1 / 16))
RELEASE_PROPOSALS = 2000


def _target(scales: tuple[float, ...]) -> AnalyticGaussian:
    return AnalyticGaussian([0.0] * DIMENSION, np.diag(np.square(scales)).tolist())


def _stationary_start(scales: tuple[float, ...]) -> torch.Tensor:
    """One exact draw from the target, so no pilot or chain spends a transient."""
    generator = torch.Generator().manual_seed(7)
    noise = torch.randn(DIMENSION, generator=generator, dtype=torch.float64)
    return noise * torch.tensor(scales, dtype=torch.float64)


def _esjd_per_gradient(
    objective: AnalyticGaussian, start: torch.Tensor, step: float, n: int, seed: int
) -> float:
    """A fixed-step chain's squared jumps over its gradients: the brute force's one reading."""
    chain = sample(
        objective,
        torch.Generator().manual_seed(seed),
        n,
        step_size=step,
        n_steps=N_STEPS,
        start=start,
    )
    path = torch.cat([start[None], chain.draws])
    return float(((path[1:] - path[:-1]) ** 2).sum()) / chain.spent


def _optimum(
    scales: tuple[float, ...], grid: tuple[float, ...], n: int, seeds: range
) -> float:
    objective, start = _target(scales), _stationary_start(scales)
    readings = [
        np.mean([_esjd_per_gradient(objective, start, s, n, seed) for seed in seeds])
        for s in grid
    ]
    return grid[int(np.argmax(readings))]


def _tuned(scales: tuple[float, ...], seed: int) -> float:
    objective, start = _target(scales), _stationary_start(scales)
    tuned = tune_step(
        hamiltonian_pilot(
            objective,
            functools.partial(ConstantTempSchedule, 1.0),
            start=start,
            n_steps=N_STEPS,
        ),
        budget=PILOT,
        criterion=Criterion.ESJD,
        rng=torch.Generator().manual_seed(seed),
        grid=CANDIDATES,
    )
    assert tuned.spent <= PILOT.size
    return tuned.step_size


def _ratio(a: float, b: float) -> float:
    return max(a / b, b / a)


@pytest.mark.oracle
@pytest.mark.parametrize("scales", [ISOTROPIC, ANISOTROPIC], ids=["iso", "aniso"])
def test_the_esjd_step_is_near_the_brute_force_optimum(
    scales: tuple[float, ...],
) -> None:
    # Measured: optimum 0.562 and 0.133 at this size (0.649 and 0.154 on the
    # release grid); tuned 0.562 at all three seeds, and 0.100, 0.178 and
    # 0.178: ratios 1.00 and 1.33 against the tolerance's 1.5.
    optimum = _optimum(scales, REFERENCE_GRID, REFERENCE_PROPOSALS, range(1))
    for seed in range(3):
        assert _ratio(_tuned(scales, seed), optimum) <= WITHIN, seed


@pytest.mark.release
@pytest.mark.oracle
@pytest.mark.parametrize("scales", [ISOTROPIC, ANISOTROPIC], ids=["iso", "aniso"])
def test_the_esjd_step_is_near_the_optimum_of_long_chains_on_a_fine_grid(
    scales: tuple[float, ...],
) -> None:
    optimum = _optimum(scales, RELEASE_GRID, RELEASE_PROPOSALS, range(2))
    for seed in range(10):
        assert _ratio(_tuned(scales, seed), optimum) <= WITHIN, seed


@pytest.mark.smoke
@pytest.mark.patch
def test_a_one_candidate_pilot_is_annealing_at_that_step_bitwise() -> None:
    objective, start = _target(ANISOTROPIC), _stationary_start(ANISOTROPIC)
    schedule = ExponentialTempSchedule(8.0, 1.0, 12)
    tuned = tune_step(
        hamiltonian_pilot(
            objective, functools.partial(compress, schedule), start=start, n_steps=3
        ),
        budget=Budget(Cost.GRADIENTS, 1 + 12 * 3),
        criterion=Criterion.LOWEST_ENERGY,
        rng=torch.Generator().manual_seed(3),
        grid=(0.1,),
    )
    annealed = anneal(
        objective,
        schedule,
        torch.Generator().manual_seed(3),
        step_size=0.1,
        n_steps=3,
        start=start,
    )
    assert tuned.proposals == 12
    assert torch.equal(tuned.chosen.best, annealed.best)
    assert tuned.chosen.lowest_energy == annealed.value
    assert tuned.spent == annealed.spent == 1 + 12 * 3


@pytest.mark.analytic
def test_the_criterion_ranks_the_candidates_it_names() -> None:
    # From a start far in the tails, the lowest energy and the ESJD rank the
    # candidates differently, and each picks its own first.
    objective = _target(ANISOTROPIC)
    start = torch.full((DIMENSION,), 3.0, dtype=torch.float64)
    for criterion in (Criterion.ESJD, Criterion.LOWEST_ENERGY):
        tuned = tune_step(
            hamiltonian_pilot(
                objective,
                functools.partial(ConstantTempSchedule, 1.0),
                start=start,
                n_steps=N_STEPS,
            ),
            budget=Budget(Cost.GRADIENTS, 800),
            criterion=criterion,
            rng=torch.Generator().manual_seed(0),
            grid=CANDIDATES,
        )
        if criterion is Criterion.ESJD:
            best = max(c.esjd for c in tuned.candidates)
            assert tuned.chosen.esjd == best
        else:
            best = min(c.lowest_energy for c in tuned.candidates)
            assert tuned.chosen.lowest_energy == best
        assert tuned.spent == 1 + len(CANDIDATES) * tuned.proposals * N_STEPS
        assert tuned.termination.iterations == len(CANDIDATES)
        assert "*" in tuned.table()


#: A tilted double well, ``(x**2 - 1)**2 + TILT * x``: the deep basin's
#: minimum near -1, the shallow one's near 1, the barrier between near 0.075.
TILT = 0.3


def _well(x: float) -> float:
    return (x * x - 1.0) ** 2 + TILT * x


def _well_slope(x: float) -> float:
    return 4.0 * x * (x * x - 1.0) + TILT


def _barrier() -> float:
    # The local maximum of the well: the root of its slope between the minima.
    roots = np.roots([4.0, 0.0, -4.0, TILT])
    return float(sorted(roots.real)[1])


def _basin_minimum(x: float) -> PolishedPoint:
    # A polish run to convergence, read off analytically: the minimum of the
    # basin `x` lies in, at one iteration.
    roots = sorted(np.roots([4.0, 0.0, -4.0, TILT]).real)
    minimum = roots[0] if x < roots[1] else roots[2]
    return PolishedPoint(
        value=float(_well(minimum)),
        termination=Termination.after(1, converged=True),
        theta=torch.tensor([minimum]),
    )


def _descent(size: float) -> Callable[..., Moved[float, None]]:
    # A step of gradient descent, accepted always: what a step of `size` does
    # with no noise, so where it lands is arithmetic.
    def step(
        x: float, _energy: float, _carried: None, _t: float, _rng: object
    ) -> Moved[float, None]:
        moved = x - size * _well_slope(x)
        return Moved(moved, _well(moved), None, 1)

    return step


@pytest.mark.analytic
def test_the_polished_gap_ranks_a_step_that_stays_in_its_basin_first() -> None:
    # From x = -2 in the deep basin, three steps of descent: at 0.01 the
    # walk stays left of the barrier, at 0.1 its first step lands right of
    # it. The far step reaches the lower energy in three steps, so the lowest
    # energy ranks it first; its basin's minimum is 0.59 higher, so the
    # polished gap ranks the near step first.
    start = Moved(-2.0, _well(-2.0), None, 0)
    pilot: Pilot[float, None, object] = Pilot(
        step=_descent,
        start=start,
        schedule=functools.partial(ConstantTempSchedule, 1.0),
        per_proposal=1,
        unit=Cost.GRADIENTS,
        keep=float,
        squared_jump=lambda a, b: (b - a) ** 2,
    )
    near, far = 0.01, 0.1
    barrier = _barrier()
    by = {
        criterion: tune_step(
            pilot,
            budget=Budget(Cost.GRADIENTS, 6),
            criterion=criterion,
            rng=None,
            grid=(near, far),
            polish=_basin_minimum if criterion is Criterion.POLISHED_GAP else None,
        )
        for criterion in (Criterion.LOWEST_ENERGY, Criterion.POLISHED_GAP)
    }
    stays, leaves = by[Criterion.POLISHED_GAP].candidates
    assert stays.best < barrier < leaves.best
    assert leaves.lowest_energy < stays.lowest_energy
    assert by[Criterion.LOWEST_ENERGY].step_size == far
    assert by[Criterion.POLISHED_GAP].step_size == near
    assert stays.polished is not None
    assert leaves.polished is not None
    # The minima are 0.5996 apart in energy, near 2 * TILT where the tilt is small.
    assert leaves.polished - stays.polished == pytest.approx(0.5996, abs=1e-4)
    assert by[Criterion.POLISHED_GAP].polish_iterations == 2
    assert by[Criterion.LOWEST_ENERGY].candidates[0].polished is None


def _polish_at_energy(
    objective: Objective, theta: torch.Tensor, budget: Budget
) -> PolishedPoint:
    # A polish that moves nothing and charges its whole budget: the ranking
    # is then the lowest energy's, and the charge is the cap.
    return PolishedPoint(
        value=float(objective(theta)),
        termination=Termination.after(budget.size, converged=False),
        theta=theta,
    )


@pytest.mark.analytic
def test_every_auto_sampler_charges_its_polishes_apart_from_its_gradients() -> None:
    # Two candidates at a polish of 6 iterations: 12 for one pilot, and a
    # ladder of two splits it as it splits the budget, 3 per candidate per
    # rung. The gradients are the lowest energy's tuning's, unchanged.
    objective = _target(ISOTROPIC)
    polishing = StepTuning(
        TUNING.budget,
        Criterion.POLISHED_GAP,
        TUNING.grid,
        _polish_at_energy,
        Budget(Cost.ITERATIONS, 6),
    )
    chain = sample(
        objective,
        torch.Generator().manual_seed(0),
        4,
        step_size=AUTO,
        n_steps=N_STEPS,
        tuning=polishing,
    )
    assert chain.tuned is not None
    assert chain.tuned.spent == 241
    assert chain.tuned.polish_iterations == 12
    assert chain.tuned.chosen.polished == min(
        c.polished for c in chain.tuned.candidates if c.polished is not None
    )
    tempered = parallel_tempering(
        objective,
        (1.0, 4.0),
        torch.Generator().manual_seed(0),
        3,
        step_size=AUTO,
        n_steps=N_STEPS,
        tuning=polishing,
    )
    assert tempered.tuned is not None
    assert [pilot.polish_iterations for pilot in tempered.tuned] == [6, 6]
    assert [pilot.spent for pilot in tempered.tuned] == [111, 111]


TUNING = StepTuning(Budget(Cost.GRADIENTS, 241), Criterion.LOWEST_ENERGY, (0.03, 0.3))


@pytest.mark.analytic
def test_every_auto_sampler_charges_its_pilot_to_the_gradient() -> None:
    # 241 gradients over two candidates at five leapfrog steps: one at the
    # start and 24 proposals each, 241 exactly.
    objective = _target(ISOTROPIC)
    per_proposal = leapfrog.force_evaluations(N_STEPS, carried=True)
    chain = sample(
        objective,
        torch.Generator().manual_seed(0),
        4,
        step_size=AUTO,
        n_steps=N_STEPS,
        tuning=TUNING,
    )
    assert chain.tuned is not None
    assert chain.tuned.spent == 241
    assert chain.spent == 241 + 1 + 4 * per_proposal

    annealed = anneal(
        objective,
        ExponentialTempSchedule(4.0, 1.0, 6),
        torch.Generator().manual_seed(0),
        step_size=AUTO,
        n_steps=N_STEPS,
        tuning=TUNING,
    )
    assert annealed.tuned is not None
    assert annealed.spent == 241 + 1 + 6 * per_proposal

    # A ladder of two splits the budget: 120 each, 11 proposals per candidate.
    tempered = parallel_tempering(
        objective,
        (1.0, 4.0),
        torch.Generator().manual_seed(0),
        3,
        step_size=AUTO,
        n_steps=N_STEPS,
        tuning=TUNING,
    )
    assert tempered.tuned is not None
    assert [pilot.spent for pilot in tempered.tuned] == [111, 111]
    assert tempered.spent == 2 * 111 + 2 * (1 + 3 * per_proposal)

    generator = torch.Generator().manual_seed(0)
    for start in (
        FromChain(1, AUTO, generator, N_STEPS, adaptation=None, tuning=TUNING),
        FromAnnealing(
            ConstantTempSchedule(1.0, 2), AUTO, generator, N_STEPS, tuning=TUNING
        ),
        FromTempering((1.0, 4.0), 2, AUTO, generator, N_STEPS, tuning=TUNING),
    ):
        assert len(start.starts(objective)) == 1


@pytest.mark.smoke
def test_a_tuning_and_its_step_must_agree() -> None:
    objective = _target(ISOTROPIC)
    generator = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError, match="needs a StepTuning"):
        sample(objective, generator, 2, step_size=AUTO)
    with pytest.raises(ValueError, match="step is given"):
        sample(objective, generator, 2, step_size=0.1, tuning=TUNING)
    with pytest.raises(ValueError, match="adaptation"):
        sample(
            objective,
            generator,
            2,
            step_size=AUTO,
            tuning=TUNING,
            adaptation=Adaptation(warmup=8, target_acceptance=0.65, step_jitter=0.0),
        )
    pilot = hamiltonian_pilot(objective, functools.partial(ConstantTempSchedule, 1.0))
    with pytest.raises(ValueError, match="budget is in"):
        tune_step(
            pilot,
            budget=Budget(Cost.EVALUATIONS, 100),
            criterion=Criterion.ESJD,
            rng=generator,
        )
    with pytest.raises(ValueError, match="a pilot needs two"):
        tune_step(
            pilot,
            budget=Budget(Cost.GRADIENTS, 100),
            criterion=Criterion.ESJD,
            rng=generator,
        )
    with pytest.raises(ValueError, match="only with it"):
        tune_step(
            pilot,
            budget=Budget(Cost.GRADIENTS, 100),
            criterion=Criterion.POLISHED_GAP,
            rng=generator,
            grid=(0.1,),
        )
    with pytest.raises(ValueError, match="only with it"):
        StepTuning(TUNING.budget, Criterion.POLISHED_GAP, TUNING.grid)
    with pytest.raises(ValueError, match="only with it"):
        StepTuning(
            TUNING.budget,
            Criterion.ESJD,
            TUNING.grid,
            _polish_at_energy,
            Budget(Cost.ITERATIONS, 2),
        )
    calibration = LadderCalibration(LadderRule.ROUND_TRIPS, 2, 1, tolerance=0.1)
    with pytest.raises(ValueError, match="calibrated"):
        FromTempering(
            (1.0, 4.0),
            2,
            AUTO,
            generator,
            calibration=calibration,
            tuning=TUNING,
        )
