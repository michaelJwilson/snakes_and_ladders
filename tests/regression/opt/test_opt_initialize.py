"""Where an optimization starts, and what choosing it is worth.

Issue #251. The seam existed -- `Objective.initial()` and `fit`'s `theta0`
override, both already pinned -- and every implementation put one fixed
constant through it. These tests pin the initializers and then *measure*
whether multiple starts buy anything, on the two surfaces where the answer is
known independently of this repository.

The numbers below say multi-start is true of one surface and close to false
of the other, at the same cost. Root `CLAUDE.md`'s demand for a benchmark
before a change, applied to a search strategy.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from snakes_and_ladders.cost import Cost
from snakes_and_ladders.opt.budget import Budget, Outcome, compare, restarts
from snakes_and_ladders.opt.fit import fit, fit_from
from snakes_and_ladders.opt.hmm import HmmObjective
from snakes_and_ladders.opt.initialize import FromObjective, Perturbed, RandomRestart
from snakes_and_ladders.opt.testfunctions import (
    HIMMELBLAU_MINIMA,
    Himmelblau,
    Rastrigin,
    Rosenbrock,
)
from snakes_and_ladders.sample import hmc
from snakes_and_ladders.sample.initialize import (
    CHAIN_ADAPTATION,
    FromAnnealing,
    FromChain,
    FromTempering,
)
from snakes_and_ladders.sample.schedule import ExponentialTempSchedule

from tests._objective_checks import AnalyticGaussian

#: Distance within which a fit counts as having reached a named minimum.
TOLERANCE = 0.1


@pytest.mark.smoke
def test_the_default_initializer_is_the_objective_s_own_start() -> None:
    """`FromObjective` is today's behaviour exactly, not an approximation.

    Every number `STATUS.md` pins was produced from the objective's own
    `initial()`, so it must stay reachable and identical; otherwise the
    abstraction silently moves results.
    """
    objective = Rosenbrock()

    starts = FromObjective().starts(objective)

    assert len(starts) == 1
    assert torch.equal(starts[0], objective.initial())


@pytest.mark.smoke
def test_a_single_start_makes_the_multi_start_fit_the_ordinary_one() -> None:
    """One start is the degenerate case of many, and the code agrees.

    `fit_from` with `FromObjective` differing from `fit` would make the
    abstraction a second code path rather than a generalization.
    """
    objective = Rosenbrock()

    single = fit(objective)
    through_initializer = fit_from(objective, FromObjective(), workers=1)

    assert through_initializer.best.value == pytest.approx(single.value, rel=1e-12)
    assert torch.allclose(through_initializer.best.theta, single.theta)
    assert through_initializer.spread == 0.0


@pytest.mark.analytic
def test_the_perturbed_start_leaves_the_stationary_point_the_uniform_hmm_sits_on() -> (
    None
):
    """The defect this initializer generalizes, asserted rather than described.

    `opt/hmm.py` records that a uniform HMM is a *stationary point*: with every
    hidden state identical, the gradient with respect to the initial and
    transition parameters is exactly zero and an optimizer never leaves. Fixed
    there by hand; checked here of the model-free perturbation, whose nudged
    start has a gradient the symmetric one does not.
    """
    observations = np.array([[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]], dtype=np.int64)
    objective = HmmObjective(observations=observations, n_states=2, n_symbols=2)

    symmetric = torch.zeros(objective.n_parameters, dtype=torch.float64)
    symmetric.requires_grad_(True)
    objective(symmetric).backward()  # type: ignore[no-untyped-call]
    assert symmetric.grad is not None
    symmetric_norm = float(torch.linalg.vector_norm(symmetric.grad))

    tilted = Perturbed(magnitude=0.1).starts(objective)[0].detach().clone()
    tilted.requires_grad_(True)
    objective(tilted).backward()  # type: ignore[no-untyped-call]
    assert tilted.grad is not None
    tilted_norm = float(torch.linalg.vector_norm(tilted.grad))

    assert tilted_norm > symmetric_norm, (
        f"the tilt did not leave the symmetry: {tilted_norm} against "
        f"{symmetric_norm} at the uniform point"
    )


@pytest.mark.end2end
def test_restarts_reach_every_himmelblau_basin_and_one_start_reaches_one() -> None:
    """The case multi-start is for, measured against four analytic minima.

    Himmelblau has four equal global minima, so the start alone decides which
    comes back. A single fixed start reaches one basin however often it is
    run; four random restarts reach all four.
    """
    objective = Himmelblau()

    from_one = {
        Himmelblau.nearest_minimum(
            fit_from(objective, FromObjective(), workers=1).best.theta
        ).index
        for _ in range(8)
    }
    assert len(from_one) == 1, "a deterministic start reached more than one basin"

    from_many = {
        Himmelblau.nearest_minimum(result.theta).index
        for trial in range(8)
        for result in fit_from(
            objective, RandomRestart(4, 3.0, np.random.default_rng(trial)), workers=1
        ).all_fits
    }
    assert from_many == set(range(len(HIMMELBLAU_MINIMA)))


@pytest.mark.end2end
def test_restarts_barely_help_on_rastrigin_and_the_number_says_so() -> None:
    """The negative result, kept because it is the more useful one.

    Rastrigin has roughly `10 ** n` local minima, one per lattice cell, and
    every one satisfies the first-order condition. Restarts drawn around a
    fixed centre land in *some* cell and stay there, so the global minimum is
    reached 2 times in 30 at sixteen starts against 0 in 30 at one -- sixteen
    times the cost for a success rate still near zero. Widening the draw does
    not fix it: the same 2 in 30 at scale 4.0 as at 2.0, because the obstacle
    is the density of minima and not the reach of the proposal.

    So the defaults do not change: multi-start is a tool for a few
    well-separated basins, and a general improvement is what the measurement
    contradicts. Through `opt.budget.compare` at eight fits it is 0 of 10
    either way.
    """
    objective = Rastrigin()

    def one_start(
        instance: Rastrigin, _budget: Budget, _rng: np.random.Generator
    ) -> Outcome:
        return Outcome(
            float(fit_from(instance, FromObjective(), workers=1).best.value), 1
        )

    def random_start(
        instance: Rastrigin, _budget: Budget, rng: np.random.Generator
    ) -> Outcome:
        return Outcome(
            float(fit_from(instance, RandomRestart(1, 2.0, rng), workers=1).best.value),
            1,
        )

    # Ten trials, each an instance of the same surface with its own stream;
    # eight fits is the budget, and the fixed start spends one of them
    # (issue #281). A hit is the global minimum's value, zero, within the
    # optimizer's tolerance; every other minimum sits at least one above it.
    result = compare(
        {"one start": one_start, "restarts": restarts(random_start, 1)},
        [objective] * 10,
        Budget(Cost.FITS, 8),
        seeds=(0,),
        workers=1,
        known=[0.0] * 10,
    )
    hits = result.hits(tolerance=1e-6)
    single, many = hits["one start"], hits["restarts"]

    assert single == 0, "the fixed start unexpectedly found the global minimum"
    assert many <= 2, (
        f"restarts reached the global minimum {many}/10 times; the recorded "
        "measurement is near zero, and a change this large means the fixture "
        "or the optimizer moved"
    )


@pytest.mark.smoke
def test_the_spread_reports_that_the_starts_disagreed() -> None:
    """A multi-start fit that returned only the best would hide the multimodality.

    On Himmelblau every basin has value 0, so the spread is ~0 even though the
    *answers* differ, which is why the fits are returned as well as the spread.
    On Rastrigin the values differ, and the spread says how wrong a single fit
    could have been.
    """
    flat = fit_from(
        Himmelblau(), RandomRestart(4, 3.0, np.random.default_rng(0)), workers=1
    )
    assert flat.spread == pytest.approx(0.0, abs=1e-6)
    assert len({Himmelblau.nearest_minimum(f.theta).index for f in flat.all_fits}) > 1

    rugged = fit_from(
        Rastrigin(), RandomRestart(6, 2.0, np.random.default_rng(0)), workers=1
    )
    assert rugged.spread > 0.0
    assert rugged.best.value == min(f.value for f in rugged.all_fits)


@pytest.mark.smoke
def test_two_generators_seeded_alike_give_the_same_restarts() -> None:
    """A declared seed still determines the run.

    `opt/hmm.py` rejected a jitter because it "would make the fit depend on a
    second seed nobody declared". Taking the generator removes that objection:
    the seed is the caller's, and declared.
    """
    objective = Himmelblau()

    first = RandomRestart(4, 1.0, np.random.default_rng(3)).starts(objective)
    second = RandomRestart(4, 1.0, np.random.default_rng(3)).starts(objective)

    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))


@pytest.mark.smoke
def test_independent_restart_sets_come_from_one_generator() -> None:
    """The property `sim/CLAUDE.md`'s rule exists for, on this module.

    Seeding inside would make every restart set identical, which looks like an
    ensemble and is one draw.
    """
    objective = Himmelblau()
    rng = np.random.default_rng(9)
    initializer = RandomRestart(2, 1.0, rng, include_nominal=False)

    drawn = [
        tuple(float(v) for v in initializer.starts(objective)[0]) for _ in range(4)
    ]

    assert len(set(drawn)) > 1


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("n_starts", "scale", "match"),
    [
        (0, 1.0, "at least 1"),
        (2, 0.0, "must be positive"),
        (2, -1.0, "must be positive"),
    ],
)
def test_an_unusable_restart_specification_is_refused(
    n_starts: int, scale: float, match: str
) -> None:
    """Zero starts and a non-positive scale are refused where they are stated.

    A zero-start initializer surfaces as an empty `min` inside `fit_from`, and
    a zero scale is a restart set that is not one.
    """
    with pytest.raises(ValueError, match=match):
        RandomRestart(n_starts, scale, np.random.default_rng(0))


@pytest.mark.smoke
def test_a_non_positive_perturbation_is_refused() -> None:
    """A zero tilt does not leave a stationary point, which is the whole job."""
    with pytest.raises(ValueError, match="must be positive"):
        Perturbed(magnitude=0.0)


@pytest.mark.smoke
def test_four_workers_fit_the_starts_one_worker_fits() -> None:
    """A multi-start fit on a process pool is the serial one, bitwise (issue #344).

    A start draws nothing once the initializer has produced it, and every
    worker runs at the intra-op thread count the serial path runs at, so the
    same kernels reduce in the same order: the fitted parameters are equal
    with ``torch.equal``, not ``allclose``.
    """
    objective = Himmelblau()

    serial = fit_from(
        objective, RandomRestart(4, 3.0, np.random.default_rng(0)), workers=1
    )
    pooled = fit_from(
        objective, RandomRestart(4, 3.0, np.random.default_rng(0)), workers=4
    )

    assert len(serial.all_fits) == 4
    assert pooled.spread == serial.spread
    for one, four in zip(serial.all_fits, pooled.all_fits, strict=True):
        assert torch.equal(four.theta, one.theta)
        assert four.value == one.value
        assert four.iterations == one.iterations
        assert four.converged == one.converged


@pytest.mark.smoke
def test_a_multi_start_fit_refuses_no_workers() -> None:
    with pytest.raises(ValueError, match="at least one"):
        fit_from(Himmelblau(), FromObjective(), workers=0)


#: Distance within which a fit counts as having *reached* a published
#: minimizer rather than merely landed in its basin. Set from the precision
#: the constants are quoted to --- six decimals
#: (:data:`snakes_and_ladders.opt.testfunctions.HIMMELBLAU_MINIMA`) --- so a
#: tighter bound would be checking the transcription and not the fit.
PUBLISHED_TOLERANCE = 1e-5


@pytest.mark.oracle
def test_every_restart_lands_on_a_published_himmelblau_minimizer() -> None:
    # The multi-start initializer was refereed by basin *coverage*: four
    # restarts reach four distinct basins. Coverage passes whatever the fit
    # converged to and says nothing about where in the basin it stopped.
    # Himmelblau's minimizers are published to six decimals, which makes the
    # stronger statement exact: every fit from every restart is within 6.2e-07
    # of one of them, against a tolerance of 1e-5, and its value is 7.9e-31
    # against an exact 0. The four together are still covered.
    objective = Himmelblau()

    reached, worst_distance, worst_value = set(), 0.0, 0.0
    for trial in range(8):
        result = fit_from(
            objective, RandomRestart(4, 3.0, np.random.default_rng(trial)), workers=1
        )
        for one in result.all_fits:
            index, distance = Himmelblau.nearest_minimum(one.theta)
            reached.add(index)
            worst_distance = max(worst_distance, distance)
            worst_value = max(worst_value, abs(float(one.value)))

    assert worst_distance < PUBLISHED_TOLERANCE, worst_distance
    assert worst_value < 1e-12, worst_value
    assert reached == set(range(len(HIMMELBLAU_MINIMA))), reached


#: The generators the three sampled starts are run from. Eight, because what
#: they buy is a rate and one seed is an anecdote.
SAMPLED_SEEDS = range(8)

#: Where Rastrigin's descent-only fit stops from the objective's own start:
#: the cell at (3.9798, 3.9798), whose value is a closed-form property of the
#: surface and not a measured one.
TRAPPED_VALUE = 31.83848758492307

#: How far a Rastrigin local minimizer sits from the lattice point it belongs
#: to: ``2 x + 20 pi sin(2 pi x) = 0`` puts it inside a thirtieth of a cell,
#: and every fit below is checked to land on one rather than between them.
#: Realized over the 80 fits: 0.0253.
LATTICE_TOLERANCE = 0.03

#: The largest value a fit from each sampled start reaches on any of the eight
#: seeds. All three are below :data:`TRAPPED_VALUE`; none is 0.
SAMPLED_CEILING = {"chain": 7.9597, "anneal": 25.8687, "temper": 7.9597}


def _on_the_lattice(theta: torch.Tensor) -> float:
    """How far ``theta`` sits from the integer lattice Rastrigin's minima lie on."""
    return float((theta - torch.round(theta)).abs().max())


def _lowest(objective: Rastrigin, points: list[torch.Tensor]) -> float:
    """The least value over ``points``, each checked to be a lattice cell."""
    for point in points:
        assert _on_the_lattice(point) < LATTICE_TOLERANCE, point
    return min(float(objective(point)) for point in points)


def _chain_start(seed: int) -> FromChain:
    """Six draws of a short chain, at the step the surface accepts.

    Fixed-parameter, ``adaptation=None``: the comparison below is at the
    chain's 286 gradients, and FromChain's default warm-up (issue #898) would
    add 300 proposals of 11 gradients each to it alone.
    """
    return FromChain(
        6,
        0.05,
        torch.Generator().manual_seed(seed),
        n_steps=10,
        burn_in=20,
        adaptation=None,
    )


@pytest.mark.patch
def test_a_chain_start_warms_up_by_default_and_none_is_the_chain_it_drew_before() -> (
    None
):
    # Issue #898 made the warm-up FromChain's default. The referee is
    # `hmc.sample` called directly: the default is that call with
    # CHAIN_ADAPTATION, bitwise, charging the warm-up's 300 proposals; and
    # `adaptation=None` is that call without one, bitwise the pre-#898 chain.
    # The acceptance these values reach is pinned where they were measured,
    # `tests/regression/sample/test_opt_hmc_adaptive.py`'s ADAPTATION.
    objective = AnalyticGaussian([1.0, -2.0], [[2.0, 0.6], [0.6, 0.5]])
    draws, step, steps, burn_in = 4, 0.1, 5, 3
    per_proposal = hmc.leapfrog.force_evaluations(steps)

    def chain(adaptation: hmc.Adaptation | None) -> hmc.HmcChain:
        return hmc.sample(
            objective,
            torch.Generator().manual_seed(898),
            draws,
            step_size=step,
            n_steps=steps,
            burn_in=burn_in,
            adaptation=adaptation,
        )

    warmed = FromChain(
        draws, step, torch.Generator().manual_seed(898), steps, burn_in
    ).chain(objective)
    assert torch.equal(warmed.theta, chain(CHAIN_ADAPTATION).theta)
    assert warmed.adapted is not None
    assert (
        warmed.force_evaluations
        == (CHAIN_ADAPTATION.warmup + burn_in + draws) * per_proposal
    )

    fixed = FromChain(
        draws, step, torch.Generator().manual_seed(898), steps, burn_in, None
    ).chain(objective)
    assert torch.equal(fixed.theta, chain(None).theta)
    assert fixed.adapted is None
    assert fixed.force_evaluations == (burn_in + draws) * per_proposal


def _annealed_start(seed: int) -> FromAnnealing:
    """A falling temperature over 120 proposals: 1,320 gradients."""
    return FromAnnealing(
        ExponentialTempSchedule(20.0, 0.05, 120),
        0.05,
        torch.Generator().manual_seed(seed),
        n_steps=10,
    )


def _tempered_start(seed: int) -> FromTempering:
    """Four replicas over 30 rounds: the same 1,320 gradients as annealing."""
    return FromTempering(
        (1.0, 3.0, 9.0, 27.0), 30, 0.05, torch.Generator().manual_seed(seed), n_steps=10
    )


@pytest.mark.oracle
@pytest.mark.release
def test_the_sampled_starts_are_their_runs_own_records_and_leave_the_cell_descent_cannot() -> (
    None
):
    # The referee is outside the ladder (issue #734): each run's own record,
    # bitwise, and Rastrigin's closed form -- roughly 10 ** n local minima on
    # the integer lattice, one global minimum of 0 at the origin, and a fit
    # that stops at whichever cell it was started in.
    #
    # What a start *is*, asserted rather than described. From generators
    # seeded alike, `FromChain.starts` is the chain's own draws in order,
    # `FromAnnealing.starts` is `[run.theta]` and `FromTempering.starts` is
    # `[run.theta]`, every one of them equal bit for bit; both runs report the
    # value at the point they return, exactly; annealing's best is at or below
    # where its chain ended; and tempering's best is the lowest value in the
    # positions it recorded, with a gap of exactly 0.0.
    #
    # What the sampling buys, over eight seeds. Descent from the objective's
    # own start lands at 31.8385 every time -- the cell at (3.9798, 3.9798) --
    # and every one of the three sampled starts lands strictly below it on all
    # eight: at most 7.9597 for the chain and for tempering, 25.8687 for
    # annealing, every landing point on the lattice to 0.0253 against the
    # 0.03 declared.
    #
    # One wording the ticket offers does not hold. The tempered start is *not*
    # the coldest replica's: over the eight seeds the lowest value recorded
    # sits at the coldest replica on 7 and at a hotter one on the eighth,
    # which is what a ladder is for, so the count is what is asserted.
    #
    # Where it stops, and it stops short of the ticket's wording: none of the
    # three *reaches* the closed-form optimum at this budget. Over 24 runs the
    # global minimum is found 0 times, so what is pinned is escape from the
    # start's cell and not a solution of Rastrigin. The cheapest of the three
    # is no worse here: 1,320 gradients of annealing buy a ceiling three times
    # the 286-gradient chain's.
    objective = Rastrigin()

    drawn = _chain_start(0).starts(objective)
    recorded = _chain_start(0).chain(objective).theta
    assert len(drawn) == recorded.shape[0] == 6
    for start, draw in zip(drawn, recorded, strict=True):
        assert torch.equal(start, draw)

    annealed = _annealed_start(0).run(objective)
    assert torch.equal(_annealed_start(0).starts(objective)[0], annealed.theta)
    assert annealed.value == float(objective(annealed.theta))
    assert annealed.value <= float(objective(annealed.final))

    trapped = fit(objective)
    assert _on_the_lattice(trapped.theta) < LATTICE_TOLERANCE, trapped.theta
    np.testing.assert_allclose(float(trapped.value), TRAPPED_VALUE, rtol=1e-12)

    reached: dict[str, list[float]] = {"chain": [], "anneal": [], "temper": []}
    coldest = 0
    for seed in SAMPLED_SEEDS:
        fitted = [
            fit(objective, theta0=start).theta
            for start in _chain_start(seed).starts(objective)
        ]
        reached["chain"].append(_lowest(objective, fitted))

        annealed = _annealed_start(seed).run(objective)
        reached["anneal"].append(
            _lowest(objective, [fit(objective, theta0=annealed.theta).theta])
        )

        tempered = _tempered_start(seed).run(objective)
        assert torch.equal(_tempered_start(seed).starts(objective)[0], tempered.theta)
        assert tempered.value == float(objective(tempered.theta))
        visited = np.array(
            [
                [float(objective(position)) for position in exchange]
                for exchange in tempered.positions
            ]
        )
        assert visited.min() - tempered.value == 0.0
        coldest += int(np.unravel_index(int(visited.argmin()), visited.shape)[1] == 0)
        reached["temper"].append(
            _lowest(objective, [fit(objective, theta0=tempered.theta).theta])
        )

    for name, values in reached.items():
        assert max(values) <= SAMPLED_CEILING[name] + 1e-9, (name, values)
        assert max(values) < TRAPPED_VALUE, (name, values)
        assert min(values) > 1e-9, (name, values)

    assert coldest == 7, coldest
