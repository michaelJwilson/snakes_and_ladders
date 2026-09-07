"""Where an optimization starts, and what choosing it is worth.

Issue #251. The seam existed -- `Objective.initial()` and `fit`'s `theta0`
override, both already pinned -- and every implementation put one fixed
constant through it. These tests pin the initializers and then *measure*
whether multiple starts buy anything, on the two surfaces where the answer is
known independently of this repository.

The measurement is the part that matters. "Multi-start helps" is plausible
enough to be assumed, and the numbers below say it is true of one surface and
close to false of the other, at the same cost. Root `CLAUDE.md` asks for a
benchmark against the reference before committing to a change; this is the
same demand applied to a search strategy.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from snakes_and_ladders.opt.fit import fit, fit_from
from snakes_and_ladders.opt.hmm import HmmObjective
from snakes_and_ladders.opt.initialize import (
    FromObjective,
    Initializer,
    Perturbed,
    RandomRestart,
)
from snakes_and_ladders.opt.testfunctions import (
    HIMMELBLAU_MINIMA,
    Himmelblau,
    Rastrigin,
    Rosenbrock,
)

#: Distance within which a fit counts as having reached a named minimum.
TOLERANCE = 0.1


@pytest.mark.structural
def test_the_default_initializer_is_the_objective_s_own_start() -> None:
    """`FromObjective` is today's behaviour exactly, not an approximation.

    Every number `STATUS.md` pins was produced from the objective's own
    `initial()`, so this has to stay reachable and stay identical -- otherwise
    adopting the abstraction would silently move results.
    """
    objective = Rosenbrock()

    starts = FromObjective().starts(objective)

    assert len(starts) == 1
    assert torch.equal(starts[0], objective.initial())


@pytest.mark.structural
def test_a_single_start_makes_the_multi_start_fit_the_ordinary_one() -> None:
    """One start is the degenerate case of many, and the code agrees.

    If `fit_from` with `FromObjective` differed from `fit`, the abstraction
    would be a second code path rather than a generalization of the first.
    """
    objective = Rosenbrock()

    single = fit(objective)
    through_initializer = fit_from(objective, FromObjective())

    assert through_initializer.best.value == pytest.approx(single.value, rel=1e-12)
    assert torch.allclose(through_initializer.best.theta, single.theta)
    assert through_initializer.spread == 0.0


@pytest.mark.mathematical
def test_the_perturbed_start_leaves_the_stationary_point_the_uniform_hmm_sits_on() -> (
    None
):
    """The defect this initializer generalizes, asserted rather than described.

    `opt/hmm.py` records that a uniform HMM is a *stationary point*: with every
    hidden state identical, the gradient with respect to the initial and
    transition parameters is exactly zero and an optimizer never leaves. That
    was fixed once, by hand, inside that file. Here the same property is
    checked of the model-free perturbation: the nudged start has a gradient
    the symmetric one does not.
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


@pytest.mark.simulated_truth
def test_restarts_reach_every_himmelblau_basin_and_one_start_reaches_one() -> None:
    """The case multi-start is for, measured against four analytic minima.

    Himmelblau has four equal global minima, so "the" optimum is not a
    well-formed question and the start alone decides which one comes back. A
    single fixed start reaches exactly one basin however many times it is run;
    four random restarts reach all four.
    """
    objective = Himmelblau()

    from_one = {
        Himmelblau.nearest_minimum(fit_from(objective, FromObjective()).best.theta)[0]
        for _ in range(8)
    }
    assert len(from_one) == 1, "a deterministic start reached more than one basin"

    from_many = {
        Himmelblau.nearest_minimum(result.theta)[0]
        for trial in range(8)
        for result in fit_from(
            objective, RandomRestart(4, 3.0, np.random.default_rng(trial))
        ).all_fits
    }
    assert from_many == set(range(len(HIMMELBLAU_MINIMA)))


@pytest.mark.simulated_truth
def test_restarts_barely_help_on_rastrigin_and_the_number_says_so() -> None:
    """The negative result, kept because it is the more useful one.

    Rastrigin has roughly `10 ** n` local minima, one per lattice cell, and
    every one satisfies the first-order condition. Restarts drawn around a
    fixed centre land in *some* cell and stay there, so the global minimum is
    reached 2 times in 30 at sixteen starts against 0 in 30 at one -- sixteen
    times the cost for a success rate still near zero. Widening the draw does
    not fix it: the same 2 in 30 at scale 4.0 as at 2.0, because the obstacle
    is the density of minima and not the reach of the proposal.

    This is why the defaults do not change in this pull request. Multi-start is
    a tool for a few well-separated basins, and asserting it as a general
    improvement would be asserting something the measurement contradicts.
    """
    objective = Rastrigin()
    target = objective.minimizer()

    def reaches_global(initializer: Initializer) -> bool:
        best = fit_from(objective, initializer).best
        return float(torch.linalg.vector_norm(best.theta - target)) < TOLERANCE

    single = sum(reaches_global(FromObjective()) for _ in range(10))
    many = sum(
        reaches_global(RandomRestart(8, 2.0, np.random.default_rng(trial)))
        for trial in range(10)
    )

    assert single == 0, "the fixed start unexpectedly found the global minimum"
    assert many <= 2, (
        f"restarts reached the global minimum {many}/10 times; the recorded "
        "measurement is near zero, and a change this large means the fixture "
        "or the optimizer moved"
    )


@pytest.mark.structural
def test_the_spread_reports_that_the_starts_disagreed() -> None:
    """A multi-start fit that returned only the best would hide the multimodality.

    On Himmelblau every basin has value 0, so the spread is ~0 even though the
    *answers* differ -- which is itself worth knowing, and why the fits are
    returned as well as the spread. On Rastrigin the values differ, and the
    spread is what says a single fit could have been wrong by that much.
    """
    flat = fit_from(Himmelblau(), RandomRestart(4, 3.0, np.random.default_rng(0)))
    assert flat.spread == pytest.approx(0.0, abs=1e-6)
    assert len({Himmelblau.nearest_minimum(f.theta)[0] for f in flat.all_fits}) > 1

    rugged = fit_from(Rastrigin(), RandomRestart(6, 2.0, np.random.default_rng(0)))
    assert rugged.spread > 0.0
    assert rugged.best.value == min(f.value for f in rugged.all_fits)


@pytest.mark.structural
def test_two_generators_seeded_alike_give_the_same_restarts() -> None:
    """A declared seed still determines the run.

    `opt/hmm.py` rejected a jitter because it "would make the fit depend on a
    second seed nobody declared". Taking the generator removes that objection
    rather than ignoring it: the seed is the caller's and it is declared.
    """
    objective = Himmelblau()

    first = RandomRestart(4, 1.0, np.random.default_rng(3)).starts(objective)
    second = RandomRestart(4, 1.0, np.random.default_rng(3)).starts(objective)

    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))


@pytest.mark.structural
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


@pytest.mark.edge_case
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

    A zero-start initializer would surface as an empty `min` inside `fit_from`,
    and a zero scale is a restart set that is not one.
    """
    with pytest.raises(ValueError, match=match):
        RandomRestart(n_starts, scale, np.random.default_rng(0))


@pytest.mark.edge_case
def test_a_non_positive_perturbation_is_refused() -> None:
    """A zero tilt does not leave a stationary point, which is the whole job."""
    with pytest.raises(ValueError, match="must be positive"):
        Perturbed(magnitude=0.0)
