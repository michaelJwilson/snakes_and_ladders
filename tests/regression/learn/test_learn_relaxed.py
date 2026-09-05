"""The Gumbel-softmax relaxation, against enumeration at every step.

The claim under test is that gradient ascent on a relaxed discrete objective
finds what discrete search finds. It is falsifiable here and only here,
because both spaces relaxed are enumerable: the exact optimum, the exact
expected score, and the exact gradient are all computable, so nothing has to
be assumed small.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from phylo.learn.potts import Configuration, PottsLandscape, optimum
from phylo.learn.relaxed import (
    MINIMUM_TEMPERATURE,
    RelaxationMode,
    RelaxedHmmPath,
    RelaxedObjective,
    RelaxedPotts,
    anneal,
    enumerate_optimum,
    estimate_gradient,
    exact_expected_gradient,
    exact_expected_score,
    gumbel_softmax,
    one_hot,
    optimize,
)
from phylo.learn.rollout import greedy_rollout
from phylo.likelihood.hmm_paths import enumerate_hidden_paths, path_log_probability
from phylo.sim.hmm import HmmParams, load_hmm_params, simulate_sequences

FIXTURES = Path(__file__).parent.parent / "fixtures"

# An antiferromagnetic chain with two nearly-degenerate states. The
# repository's own `potts_params.yaml` has `J = 0.75 > 0`, so its optimum is
# `argmax(h)` repeated and *every* method finds it -- a fixture that cannot
# separate anything, which is the recurring lesson of #177 and #198.
HARD = (-0.9, np.array([0.4, 0.35, -0.6]), 7)


def _landscape() -> PottsLandscape:
    return PottsLandscape(*HARD)


def _hmm(length: int = 8) -> tuple[RelaxedHmmPath, HmmParams, np.ndarray]:
    params = load_hmm_params(FIXTURES / "hmm_params.yaml")
    observations = np.asarray(simulate_sequences(params).observations[0][:length])
    objective = RelaxedHmmPath(
        log_initial=torch.log(torch.from_numpy(params.initial)),
        log_transition=torch.log(torch.from_numpy(params.transition)),
        log_emission=torch.log(torch.from_numpy(params.emission)),
        observations=torch.from_numpy(observations),
    )
    return objective, params, observations


def _mcnemar(first: np.ndarray, second: np.ndarray) -> tuple[int, int, float]:
    """Exact two-sided McNemar on paired successes.

    Written out because the repository carries no `scipy`, and adding one for
    a binomial tail would not survive root `CLAUDE.md`'s dependency rule.
    """
    only_first = int((first & ~second).sum())
    only_second = int((~first & second).sum())
    discordant = only_first + only_second
    if discordant == 0:
        return only_first, only_second, 1.0
    lower = min(only_first, only_second)
    tail = sum(math.comb(discordant, count) for count in range(lower + 1))
    return only_first, only_second, min(1.0, 2.0 * tail / 2**discordant)


# --- 1. The relaxation is an extension ------------------------------------


def test_the_potts_relaxation_is_exact_at_every_corner() -> None:
    # The first thing that must be true. A relaxation that disagrees with the
    # discrete score at a one-hot is a different model, and every measurement
    # made against it transfers to nothing.
    landscape = PottsLandscape(-0.9, np.array([0.4, 0.35, -0.6]), 5)
    objective = RelaxedPotts(landscape)

    for candidate in itertools.product(range(3), repeat=5):
        relaxed = float(objective.relaxed(one_hot(candidate, 3)))
        assert relaxed == pytest.approx(landscape.energy(candidate), rel=1e-11)


def test_the_hmm_relaxation_is_exact_at_every_corner() -> None:
    # The same check across a module boundary, which makes it stronger than
    # the one above: `phylo.learn` may not import `phylo.likelihood`, so
    # `RelaxedHmmPath.discrete` and `path_log_probability` are genuinely
    # independent implementations of `log P(path, observations)`.
    objective, params, observations = _hmm(length=5)

    for candidate in itertools.product(range(3), repeat=5):
        relaxed = float(objective.relaxed(one_hot(candidate, 3)))
        reference = path_log_probability(
            params, np.array(candidate, dtype=np.int64), observations
        )
        assert relaxed == pytest.approx(reference, rel=1e-11)
        assert objective.discrete(candidate) == pytest.approx(reference, rel=1e-11)


def test_the_relaxed_optimum_of_the_hmm_is_the_viterbi_path() -> None:
    # The relaxed objective's discrete optimum must be the answer another
    # module computes for the same question, or the relaxation is optimizing
    # something else.
    objective, params, observations = _hmm(length=8)

    best, score = enumerate_optimum(objective)
    reference = enumerate_hidden_paths(params, observations)

    assert list(best) == list(reference.viterbi)
    assert score == pytest.approx(reference.viterbi_log_probability, rel=1e-11)


# --- 2. The identity, and where it actually breaks ------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_expected_discrete_score_equals_the_score_at_the_marginals(
    seed: int,
) -> None:
    # `E_q[score] = score(q)` for a multilinear objective under a factorized
    # `q`. Checked against enumeration over every configuration, which shares
    # no algebra with the closed form. This is what licenses the deterministic
    # relaxation: it approximates nothing.
    torch.manual_seed(seed)
    for objective in (RelaxedPotts(_landscape()), _hmm(length=7)[0]):
        logits = torch.randn(
            (objective.n_sites, objective.n_states), dtype=torch.float64
        )

        enumerated = exact_expected_score(objective, logits)
        closed_form = float(objective.relaxed(torch.softmax(logits, dim=1)))

        assert enumerated == pytest.approx(closed_form, rel=1e-11)


def test_a_term_over_three_distinct_sites_does_not_break_the_identity() -> None:
    # The boundary is easy to state wrongly, and "the terms must be pairwise"
    # is one of the wrong statements. Three distinct sites is still one factor
    # per site, so the expectation still factorizes.
    @dataclass(frozen=True)
    class Triple:
        n_sites: int = 3
        n_states: int = 2

        def relaxed(self, probabilities: torch.Tensor) -> torch.Tensor:
            return (probabilities[0] * probabilities[1] * probabilities[2]).sum()

        def discrete(self, configuration: Configuration) -> float:
            return float(configuration[0] == configuration[1] == configuration[2])

    objective = Triple()
    logits = torch.tensor([[0.4, -0.3], [0.1, 0.2], [-0.5, 0.6]], dtype=torch.float64)

    assert exact_expected_score(objective, logits) == pytest.approx(
        float(objective.relaxed(torch.softmax(logits, dim=1))), rel=1e-11
    )


def test_a_term_using_one_site_twice_does_break_the_identity() -> None:
    # What actually breaks it: `E[X**2]` is `E[X]` for an indicator and not
    # `E[X]**2`. Not hypothetical -- `PottsGraph` permits a doubled bond,
    # because a periodic lattice of extent 2 produces one, and after wrapping
    # such an edge joins a node to itself. Measured: 1.000 against 0.557.
    @dataclass(frozen=True)
    class SelfPair:
        n_sites: int = 2
        n_states: int = 2

        def relaxed(self, probabilities: torch.Tensor) -> torch.Tensor:
            return (probabilities[0] * probabilities[0]).sum()

        def discrete(self, configuration: Configuration) -> float:
            # A one-hot always agrees with itself, whatever state it names.
            return float(configuration[0] == configuration[0])

    objective = SelfPair()
    logits = torch.tensor([[0.4, -0.3], [0.1, 0.2]], dtype=torch.float64)

    assert exact_expected_score(objective, logits) == pytest.approx(1.0)
    assert float(objective.relaxed(torch.softmax(logits, dim=1))) == pytest.approx(
        0.5565742534, abs=1e-9
    )


def test_the_relaxation_introduces_no_optimum_the_discrete_problem_lacks() -> None:
    # A multilinear function on a product of simplices attains its maximum at
    # a vertex, so the relaxed optimum cannot exceed the discrete one. It
    # follows that everything a relaxed search loses is lost to local optima
    # of the ascent, never to the relaxation.
    objective = RelaxedPotts(_landscape())
    _, best = enumerate_optimum(objective)
    generator = torch.Generator().manual_seed(4)

    for _ in range(200):
        logits = torch.randn(
            (objective.n_sites, objective.n_states),
            generator=generator,
            dtype=torch.float64,
        )
        value = float(objective.relaxed(torch.softmax(logits, dim=1)))
        assert value <= best + 1e-12


def test_the_two_objectives_satisfy_the_protocol() -> None:
    # The seam this module is built on. Everything -- estimators, exact
    # gradient, optimizer -- is written against `RelaxedObjective` and never
    # against a Potts chain or an HMM.
    assert isinstance(RelaxedPotts(_landscape()), RelaxedObjective)
    assert isinstance(_hmm(length=4)[0], RelaxedObjective)


# --- 3. The estimators, against the exact gradient ------------------------


@pytest.mark.parametrize("mode", list(RelaxationMode))
def test_the_estimator_bias_falls_and_its_variance_rises_as_temperature_falls(
    mode: RelaxationMode,
) -> None:
    # The trade the method rests on, measured rather than asserted small.
    # Against the exact gradient at 20000 draws, scaled by the largest exact
    # component:
    #
    #   tau    soft bias (SEM)    soft sd    ST bias (SEM)     ST sd
    #   2.00   0.5975 (0.0012)     0.165     0.5620 (0.0027)   0.382
    #   1.00   0.3373 (0.0034)     0.487     0.3233 (0.0051)   0.723
    #   0.50   0.1400 (0.0076)     1.077     0.1418 (0.0090)   1.272
    #   0.20   0.0475 (0.0157)     2.220     0.0502 (0.0165)   2.340
    #   0.10   0.0356 (0.0240)     3.392     0.0380 (0.0246)   3.472
    #
    # Two readings. The bias falls by a factor of 17 while the standard
    # deviation rises by a factor of 21, so no temperature is good at both.
    # And straight-through's bias matches the soft estimator's within error
    # while its variance is higher at every temperature -- it buys nothing
    # here, which is a result about this problem and not a general claim.
    #
    # The run below uses 2000 draws to stay inside the CI budget, so it
    # asserts the *ordering*, which is stable, rather than the numbers above.
    objective = RelaxedPotts(_landscape())
    torch.manual_seed(0)
    logits = 0.5 * torch.randn(
        (objective.n_sites, objective.n_states), dtype=torch.float64
    )
    exact = exact_expected_gradient(objective, logits)
    scale = float(np.abs(exact).max())

    biases, deviations = [], []
    for temperature in (2.0, 1.0, 0.5):
        generator = torch.Generator().manual_seed(11)
        draws = np.array(
            [
                estimate_gradient(objective, logits, temperature, generator, mode=mode)
                for _ in range(2000)
            ]
        )
        biases.append(float(np.abs(draws.mean(axis=0) - exact).max()) / scale)
        deviations.append(float(draws.std(axis=0).max()) / scale)

    assert biases[0] > biases[1] > biases[2]
    assert deviations[0] < deviations[1] < deviations[2]
    # And the estimator is genuinely biased at every temperature tried, which
    # is the reason the exact gradient has to be the reference.
    assert min(biases) > 0.05


def test_more_samples_cut_the_variance_and_leave_the_bias() -> None:
    # The distinction the two are reported separately for: averaging is a
    # variance reduction and not a bias reduction, so a method that fails
    # because of bias cannot be fixed by drawing more.
    objective = RelaxedPotts(_landscape())
    torch.manual_seed(0)
    logits = 0.5 * torch.randn(
        (objective.n_sites, objective.n_states), dtype=torch.float64
    )
    exact = exact_expected_gradient(objective, logits)

    means, deviations = [], []
    for n_samples in (1, 16):
        generator = torch.Generator().manual_seed(11)
        draws = np.array(
            [
                estimate_gradient(
                    objective, logits, 1.0, generator, n_samples=n_samples
                )
                for _ in range(400)
            ]
        )
        means.append(np.abs(draws.mean(axis=0) - exact).max())
        deviations.append(float(draws.std(axis=0).max()))

    assert deviations[1] < deviations[0] / 3.0
    assert means[1] == pytest.approx(means[0], rel=0.35)


def test_the_exact_gradient_matches_a_finite_difference() -> None:
    # The reference every estimator is measured against needs its own check,
    # or a bias measurement is only evidence that two wrong things differ.
    objective = RelaxedPotts(_landscape())
    torch.manual_seed(1)
    logits = 0.5 * torch.randn(
        (objective.n_sites, objective.n_states), dtype=torch.float64
    )

    exact = exact_expected_gradient(objective, logits)

    step = 1e-6
    for site, state in [(0, 0), (3, 1), (6, 2)]:
        shifted = logits.clone()
        shifted[site, state] += step
        forward = exact_expected_score(objective, shifted)
        shifted[site, state] -= 2 * step
        backward = exact_expected_score(objective, shifted)
        assert exact[site, state] == pytest.approx(
            (forward - backward) / (2 * step), rel=1e-5
        )


# --- 4. Against the baseline, at matched restarts -------------------------


def test_the_deterministic_relaxation_beats_single_flip_hill_climbing() -> None:
    # The comparison that decides whether this is worth having, on shared
    # seeds with the exact optimum as the target. Measured over 40 restarts:
    #
    #   greedy hill climbing        5/40
    #   deterministic relaxation   18/40   McNemar p = 0.00098
    #   soft Gumbel-softmax        11/40   McNemar p = 0.180
    #   straight-through           11/40   McNemar p = 0.180
    #   annealed soft (0.5 -> 0.05) 11/40  McNemar p = 0.180
    #
    # at 100 gradient steps, which is where the deterministic run has
    # converged: at 400 steps it is unchanged at 18/40 and the sampled runs
    # move by one instance.
    #
    # The headline is that the *sampling* costs, not the relaxation. The
    # deterministic ascent -- which the identity above licenses -- is
    # significantly better than the baseline; adding Gumbel noise gives up
    # that advantage and lands at a tie, and annealing does not recover it.
    #
    # The budgets are not the same unit and are not claimed to be: greedy
    # terminates at a local maximum after 3.5 decisions on average, at 14
    # discrete evaluations each, while the relaxation takes gradient steps and
    # evaluates no discrete configuration until the end. What is matched is
    # the restart count and the seeds. Measured separately, the relaxation
    # already wins at 25 gradient steps (15/40, p = 0.0064), so the advantage
    # is not bought with the larger budget.
    landscape = _landscape()
    objective = RelaxedPotts(landscape)
    _, best = optimum(landscape)

    greedy = np.array(
        [
            landscape.energy(
                greedy_rollout(
                    landscape, landscape.reset(np.random.default_rng(seed)), 200
                ).states[-1]
            )
            > best - 1e-9
            for seed in range(40)
        ]
    )
    relaxed = np.array(
        [
            optimize(
                objective, seed=seed, temperature=0.5, steps=100, stochastic=False
            ).score
            > best - 1e-9
            for seed in range(40)
        ]
    )

    assert int(greedy.sum()) == 5
    assert int(relaxed.sum()) == 18
    _, _, p_value = _mcnemar(greedy, relaxed)
    assert p_value < 0.01


def test_the_sampled_estimators_only_tie_with_the_baseline() -> None:
    # Reported as a tie because it is one. #193 established that precedent for
    # the tree policy, and a tie stated as a tie is worth more than a variant
    # promoted on an unpaired difference.
    landscape = _landscape()
    objective = RelaxedPotts(landscape)
    _, best = optimum(landscape)

    greedy = np.array(
        [
            landscape.energy(
                greedy_rollout(
                    landscape, landscape.reset(np.random.default_rng(seed)), 200
                ).states[-1]
            )
            > best - 1e-9
            for seed in range(40)
        ]
    )
    for mode in RelaxationMode:
        sampled = np.array(
            [
                optimize(
                    objective, seed=seed, temperature=0.5, steps=100, mode=mode
                ).score
                > best - 1e-9
                for seed in range(40)
            ]
        )
        assert int(sampled.sum()) == 11
        _, _, p_value = _mcnemar(greedy, sampled)
        assert p_value > 0.05


def test_the_hmm_path_is_recovered_from_every_restart() -> None:
    # The HMM half validates correctness, not difficulty: Viterbi is exact in
    # `O(T k**2)` and nothing here is hard. Recovering it from 20 of 20
    # restarts is the check that the relaxation optimizes the right objective,
    # and it would be a mistake to read it as a search result.
    objective, _, _ = _hmm(length=8)
    _, best = enumerate_optimum(objective)

    reached = sum(
        optimize(
            objective, seed=seed, temperature=0.5, steps=150, stochastic=False
        ).score
        > best - 1e-9
        for seed in range(20)
    )

    assert reached == 20


# --- 5. The pieces --------------------------------------------------------


@pytest.mark.parametrize("mode", list(RelaxationMode))
def test_a_gumbel_softmax_sample_is_row_stochastic(mode: RelaxationMode) -> None:
    generator = torch.Generator().manual_seed(2)
    logits = torch.zeros((5, 4), dtype=torch.float64)

    sample = gumbel_softmax(logits, 0.5, generator, mode=mode)

    assert sample.shape == (5, 4)
    assert sample.sum(dim=1).numpy() == pytest.approx(np.ones(5))
    assert float(sample.min()) >= 0.0


def test_straight_through_is_one_hot_forward_and_soft_backward() -> None:
    # The identity the mode is built on, checked on both halves: the value is
    # a corner, and the gradient is not the corner's (which would be zero
    # everywhere).
    generator = torch.Generator().manual_seed(3)
    logits = torch.randn((4, 3), dtype=torch.float64, requires_grad=True)

    sample = gumbel_softmax(
        logits, 0.5, generator, mode=RelaxationMode.STRAIGHT_THROUGH
    )
    sample.sum().backward()  # type: ignore[no-untyped-call]

    assert set(np.unique(sample.detach().numpy())) == {0.0, 1.0}
    assert logits.grad is not None
    assert float(np.abs(logits.grad.numpy()).max()) > 0.0


def test_the_gumbel_draws_are_independent_across_calls() -> None:
    # The generator is passed in rather than seeded inside, so a batch is
    # independent. Seeding per call silently makes every draw identical, and
    # that mistake has been made in this repository before.
    generator = torch.Generator().manual_seed(5)
    logits = torch.zeros((6, 3), dtype=torch.float64)

    drawn = {
        tuple(gumbel_softmax(logits, 1.0, generator).argmax(dim=1).tolist())
        for _ in range(20)
    }

    assert len(drawn) > 1


def test_a_temperature_below_the_floor_is_refused() -> None:
    with pytest.raises(ValueError, match="temperature must be >="):
        gumbel_softmax(
            torch.zeros((2, 2), dtype=torch.float64),
            MINIMUM_TEMPERATURE / 10,
            torch.Generator(),
        )


def test_the_anneal_schedule_is_geometric_and_hits_both_endpoints() -> None:
    # Geometric because the relaxation's behaviour is set by the ratio of
    # logit gaps to `tau`, so equal multiplicative steps are equal steps in
    # the thing that matters.
    values = [anneal(1.0, 0.01, 5, step) for step in range(5)]

    assert values[0] == pytest.approx(1.0)
    assert values[-1] == pytest.approx(0.01)
    ratios = [second / first for first, second in itertools.pairwise(values)]
    assert ratios == pytest.approx([ratios[0]] * len(ratios))


def test_a_single_step_schedule_stays_at_the_start() -> None:
    assert anneal(0.5, 0.1, 1, 0) == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("start", "end", "steps", "message"),
    [
        (1.0, 1e-9, 10, "endpoints must be >="),
        (0.1, 0.5, 10, "end must not exceed start"),
        (1.0, 0.1, 0, "steps must be at least 1"),
    ],
)
def test_an_invalid_schedule_is_refused(
    start: float, end: float, steps: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        anneal(start, end, steps, 0)


def test_annealing_reaches_the_final_temperature_during_optimization() -> None:
    # Both schedules are supported because the fixed-`tau` sweep is the
    # measurement and annealing is the practice. This checks the annealed run
    # actually anneals rather than silently holding `temperature`.
    objective = RelaxedPotts(_landscape())

    fixed = optimize(objective, seed=1, temperature=0.5, steps=40, stochastic=False)
    annealed = optimize(
        objective,
        seed=1,
        temperature=0.5,
        final_temperature=0.01,
        steps=40,
        stochastic=False,
    )

    # At `tau = 0.01` the softmax is far closer to a corner, so the relaxed
    # value at the end sits closer to a discrete score.
    assert abs(annealed.relaxed_score - annealed.score) < abs(
        fixed.relaxed_score - fixed.score
    )
