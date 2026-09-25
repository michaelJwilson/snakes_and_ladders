"""Five-component mixture at equal evaluations: tempering against restarts (issue #332).

Deferred by #284 and #303. Multi-start EM, annealing with Hamiltonian
proposals and parallel tempering spend equal evaluations per start
(`opt.budget.compare`), each ending in a charged L-BFGS polish: raw EM at 500
iterations is still 4 to 6 nats above its basin's optimum. The referee is the
best known optimum (polished truth and 1,000 polished restarts), each method
checked against it. 8 starts pin the direction per PR; 40 are the release
measurement in `docs/experiments/004`, McNemar included.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from sal.cost import Cost
from sal.emissions import GaussianEmission
from sal.opt.budget import (
    Budget,
    Comparison,
    Method,
    Outcome,
    compare,
    restarts,
)
from sal.opt.fit import fit
from sal.opt.mixture import (
    GaussianMixtureObjective,
    MixtureFit,
    expectation_maximization,
    uniform_seeds,
)
from sal.sample.hmc import anneal, leapfrog, parallel_tempering
from sal.sample.schedule import ExponentialTempSchedule
from sal.sim import fixtures
from sal.sim.mixture import simulate_mixture

from tests._objective_checks import Counted

#: Five components 1.5 sd apart, unequal weights (#262), declared in
#: `mixture/ci.yaml` (issue #622).
PARAMS = fixtures.fixture("mixture", "ci").params
WEIGHTS = np.asarray(PARAMS.weights)
MEAN = np.asarray(PARAMS.components.mean)
SCALE = np.asarray(PARAMS.components.scale)

#: Held equal across methods. One evaluation is one pass over the
#: observations' per-component log densities: an EM iteration, an objective
#: value, or an objective gradient each count one (measured single-threaded:
#: 0.6 ms, 0.4 ms and 1.5 ms).
BUDGET = Budget(Cost.EVALUATIONS, 3000)
#: EM iterations per restart before its polish.
EM_COST = 200
#: The polish: L-BFGS outer steps, and the evaluations that bounds. Each
#: outer step is one closure call, at most 25 in the line search and one
#: relative gradient norm; the result costs two more.
POLISH_STEPS = 8
POLISH_RESERVE = POLISH_STEPS * 27 + 2
#: The tempered methods. Step 0.02 is the largest at which every temperature
#: tried accepted at least 0.85 of proposals; the ladder is ratio 2 with four
#: replicas, as the Potts study's, and annealing spans the same range.
STEP_SIZE = 0.02
N_STEPS = 10
LADDER = (1.0, 2.0, 4.0, 8.0)
#: Objective calls per Hamiltonian proposal: two Hamiltonians, the trajectory,
#: and the value where the chain landed. Pinned by
#: `test_tempering_costs_what_its_accounting_says_and_is_reproducible`.
PER_PROPOSAL = leapfrog.force_evaluations(N_STEPS) + 3

#: Reaching the optimum: within this of the referee, relative to it.
TOLERANCE = 1e-6
#: Restarts behind the release referee, on the stream ``[REFEREE_SEED, i]``.
REFEREE_RESTARTS = 1000
REFEREE_SEED = 20260908
#: The best-known negative log-likelihood: 16 of 1,000 polished restarts reach
#: it and every method from at least one of 40 starts; beating it means stale.
BEST_KNOWN = 1111.596410

METHODS = ("restarts", "anneal", "tempering")


@pytest.fixture(autouse=True)
def _single_thread() -> Iterator[None]:
    # On the four-core CI runner torch's intra-op pool thrashes on these
    # 500-by-5 tensors: 164 ms per EM iteration against 0.65 ms on one
    # thread. Pinned for this module and restored after.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@dataclass(frozen=True)
class Fixture:
    """The dataset, its objective, and the seeds a start draws from."""

    observations: np.ndarray
    objective: GaussianMixtureObjective
    pooled_scale: float


def _fixture() -> Fixture:
    observations = simulate_mixture(PARAMS).observations
    return Fixture(
        observations,
        GaussianMixtureObjective(observations, len(WEIGHTS)),
        float(np.std(observations)),
    )


def _random_start(fixture: Fixture, rng: np.random.Generator) -> torch.Tensor:
    """Uniform weights, means at five distinct observations, pooled scales."""
    centres = np.sort(uniform_seeds(fixture.observations, len(WEIGHTS), rng))
    return fixture.objective.theta_from_centres(torch.as_tensor(centres))


def _theta_of(fixture: Fixture, em: MixtureFit) -> torch.Tensor:
    return fixture.objective.theta_from(
        {
            "log_weight": torch.log(em.weights),
            "mean": em.components.mean,
            "scale": em.components.scale,
        }
    )


def _polish(fixture: Fixture, theta: torch.Tensor, value: float) -> tuple[float, int]:
    """L-BFGS from ``theta``: the lower of its value and ``value``, and the evaluations.

    A zero-scale probe (~1 start in 50) is refused, keeping the unpolished value.
    """
    counted = Counted(fixture.objective)
    try:
        polished = fit(counted, theta, max_iterations=POLISH_STEPS).value
    except ValueError:
        polished = value
    assert counted.calls <= POLISH_RESERVE, counted.calls
    return min(value, polished), counted.calls


def _em_then_polish(
    fixture: Fixture, budget: Budget, rng: np.random.Generator
) -> Outcome:
    """One restart: EM from a random start for the budget less the polish, then the polish."""
    theta = _random_start(fixture, rng)
    named = fixture.objective.constrain(theta)
    try:
        em = expectation_maximization(
            fixture.observations,
            torch.exp(named["log_weight"]),
            GaussianEmission(
                named["mean"], named["scale"], fixture.objective.variance_floor
            ),
            max_iterations=budget.size - POLISH_RESERVE,
        )
    except ValueError:
        # A component collapsed: refused, and charged the iterations it had.
        return Outcome(np.inf, budget.size - POLISH_RESERVE)
    value, polish = _polish(fixture, _theta_of(fixture, em), -em.log_likelihood)
    return Outcome(value, em.iterations + polish)


def _anneal(fixture: Fixture, budget: Budget, rng: np.random.Generator) -> Outcome:
    """One chain from a random start, hot to cold, then the polish from its best point."""
    theta = _random_start(fixture, rng)
    counted = Counted(fixture.objective)
    n_proposals = (budget.size - POLISH_RESERVE - 1) // PER_PROPOSAL
    run = anneal(
        counted,
        ExponentialTempSchedule(LADDER[-1], LADDER[0], n_proposals),
        torch.Generator().manual_seed(int(rng.integers(2**31 - 1))),
        step_size=STEP_SIZE,
        n_steps=N_STEPS,
        theta0=theta,
    )
    value, polish = _polish(fixture, run.theta, run.value)
    return Outcome(value, counted.calls + polish)


def _tempering(fixture: Fixture, budget: Budget, rng: np.random.Generator) -> Outcome:
    """Four replicas from one random start, exchanging, then the polish from the best."""
    theta = _random_start(fixture, rng)
    counted = Counted(fixture.objective)
    n_rounds = (budget.size - POLISH_RESERVE - 1) // (PER_PROPOSAL * len(LADDER))
    run = parallel_tempering(
        counted,
        LADDER,
        torch.Generator().manual_seed(int(rng.integers(2**31 - 1))),
        n_rounds,
        step_size=STEP_SIZE,
        n_steps=N_STEPS,
        theta0=theta,
    )
    value, polish = _polish(fixture, run.theta, run.value)
    return Outcome(value, counted.calls + polish)


def _referee(fixture: Fixture, n_restarts: int) -> tuple[float, float, float]:
    """The polished truth, the best of ``n_restarts`` polished restarts, and the truth's own value."""
    truth = fixture.objective.theta_from_truth(WEIGHTS, MEAN, SCALE)
    at_truth = float(fixture.objective(truth))
    em = expectation_maximization(
        fixture.observations,
        torch.as_tensor(WEIGHTS),
        GaussianEmission(MEAN, SCALE, fixture.objective.variance_floor),
    )
    from_truth, _ = _polish(fixture, _theta_of(fixture, em), -em.log_likelihood)
    best = np.inf
    for index in range(n_restarts):
        outcome = _em_then_polish(
            fixture,
            Budget(Cost.EVALUATIONS, 500 + POLISH_RESERVE),
            np.random.default_rng([REFEREE_SEED, index]),
        )
        best = min(best, outcome.value)
    return from_truth, best, at_truth


@dataclass(frozen=True)
class Measurement:
    """One run of the comparison, and what it is scored against."""

    comparison: Comparison
    reference: float
    seconds: dict[str, float]

    def hits(self) -> dict[str, int]:
        return self.comparison.hits(TOLERANCE, relative=True)

    def paired_p(self, method: str) -> float:
        return self.comparison.paired_p(method, "restarts", TOLERANCE, relative=True)

    def table(self) -> str:
        """The rows `docs/experiments/004` carries: hits, p against restarts, gap, spend, time."""
        n_starts = self.comparison.reference.shape[0]
        hits = self.hits()
        gaps = self.comparison.mean_gap()
        lines = [
            "| method | starts reaching the referee | McNemar p against restarts "
            "| mean gap (nats) | evaluations spent | seconds per start |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for row, name in enumerate(self.comparison.methods):
            p_value = "—" if name == "restarts" else f"{self.paired_p(name):.3f}"
            lines.append(
                f"| {name} | {hits[name]}/{n_starts} | {p_value} | {gaps[name]:.2f} | "
                f"{int(self.comparison.spent[row].max())} of {BUDGET.size} | "
                f"{self.seconds[name] / n_starts:.1f} |"
            )
        return "\n".join(lines)


def measure(n_starts: int, reference: float) -> Measurement:
    """Every method on ``n_starts`` starts at the shared budget, scored against ``reference``.

    Start ``i`` uses ``np.random.default_rng([0, i])`` for every method.
    """
    fixture = _fixture()
    seconds = dict.fromkeys(METHODS, 0.0)

    def timed(name: str, method: Method[Fixture]) -> Method[Fixture]:
        def run(instance: Fixture, budget: Budget, rng: np.random.Generator) -> Outcome:
            started = time.perf_counter()
            outcome = method(instance, budget, rng)
            seconds[name] += time.perf_counter() - started
            return outcome

        return run

    methods: dict[str, Method[Fixture]] = {
        "restarts": restarts(_em_then_polish, EM_COST + POLISH_RESERVE),
        "anneal": _anneal,
        "tempering": _tempering,
    }
    comparison = compare(
        {name: timed(name, method) for name, method in methods.items()},
        [fixture] * n_starts,
        BUDGET,
        seeds=(0,),
        workers=1,
        known=[reference] * n_starts,
    )
    return Measurement(comparison, reference, seconds)


def _assert_nothing_beats_the_referee(measurement: Measurement) -> None:
    margin = TOLERANCE * abs(measurement.reference)
    assert float(measurement.comparison.best.min()) >= measurement.reference - margin, (
        "a method found a lower optimum than the referee; update BEST_KNOWN",
        float(measurement.comparison.best.min()),
    )


@pytest.mark.end2end
def test_at_eight_starts_restarts_reach_the_optimum_from_the_most_starts() -> None:
    # The per-pull-request tier of the release measurement below: the same code
    # on its first 8 starts. Realized: restarts 2/8, annealing 1/8, tempering
    # 0/8, and the simulated parameters' basin (1116.43 polished) is not the
    # best-known optimum.
    fixture = _fixture()
    from_truth, _, at_truth = _referee(fixture, 0)
    assert at_truth >= BEST_KNOWN
    assert from_truth >= BEST_KNOWN - TOLERANCE * BEST_KNOWN

    measurement = measure(8, BEST_KNOWN)
    hits = measurement.hits()

    _assert_nothing_beats_the_referee(measurement)
    assert measurement.comparison.spent.max() <= BUDGET.size
    assert hits == {"restarts": 2, "anneal": 1, "tempering": 0}, hits


@pytest.mark.release
@pytest.mark.end2end
def test_at_forty_starts_restarts_reach_the_optimum_from_the_most_starts() -> None:
    # `docs/experiments/004`: referee rebuilt, must match the constant.
    # Realized: restarts 7/40, tempering 4/40 (McNemar p = 0.549), annealing
    # 1/40 (p = 0.031); asserted: the ordering and which p cross 0.05.
    fixture = _fixture()
    from_truth, best_of_restarts, at_truth = _referee(fixture, REFEREE_RESTARTS)
    reference = min(from_truth, best_of_restarts)
    assert reference == pytest.approx(BEST_KNOWN, rel=TOLERANCE), reference
    assert at_truth > reference

    measurement = measure(40, reference)
    hits = measurement.hits()
    print()
    print(
        f"referee {reference:.6f}: polished truth {from_truth:.6f}, "
        f"best of {REFEREE_RESTARTS} restarts {best_of_restarts:.6f}, "
        f"truth itself {at_truth:.6f}"
    )
    print(measurement.table())

    _assert_nothing_beats_the_referee(measurement)
    assert hits == {"restarts": 7, "anneal": 1, "tempering": 4}, hits
    assert measurement.paired_p("anneal") < 0.05
    assert measurement.paired_p("tempering") > 0.05
