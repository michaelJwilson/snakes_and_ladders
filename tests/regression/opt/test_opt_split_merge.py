"""Split-and-merge EM repairs what plain EM converges to (issue #904).

The referee is the simulated truth of `emission_mixture/ci`: its generating
parameters' log-likelihood on the draw, and the generating component of every
pair. A planted start puts one component across the first two generating
components and two on the third, a fixed point plain EM converges to and
cannot leave; split-and-merge must merge the two and split the one, and end
where a fit started at the truth ends.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch
from snakes_and_ladders.emissions import CountPairEmission
from snakes_and_ladders.opt.emission_mixture import (
    CountPairSeeding,
    EmissionMixtureFit,
    expectation_maximization,
    uniform_start,
)
from snakes_and_ladders.opt.mixture import mixture_log_likelihood, responsibilities
from snakes_and_ladders.opt.split_merge import (
    SplitMerge,
    candidate_moves,
    merge_criterion,
    split_and_merge,
    split_criterion,
)
from snakes_and_ladders.sim.emission_mixture import (
    SimulatedEmissionMixtureDataset,
    simulate_emission_mixture,
)
from snakes_and_ladders.sim.fixtures import fixture

#: The relative stopping rule `test_opt_emission_mixture.py` fits under.
EM_TOLERANCE = 1e-8

#: Moves tried per round, the ticket's five; at K = 3 there are three pairs.
CANDIDATES = 5


def _draw() -> tuple[SimulatedEmissionMixtureDataset, CountPairSeeding]:
    """The ci draw and the seam at the declared pooled shapes."""
    params = fixture("emission_mixture", "ci").params
    truth = params.components
    assert isinstance(truth, CountPairEmission)
    return simulate_emission_mixture(params), CountPairSeeding(
        float(truth.total.dispersion.mean()),
        float(truth.concentration.mean()),
        joint=True,
    )


def _values(draw: SimulatedEmissionMixtureDataset) -> torch.Tensor:
    return torch.as_tensor(draw.observations, dtype=torch.float64)


def _uniform(k: int) -> torch.Tensor:
    return torch.full((k,), 1.0 / k, dtype=torch.float64)


def _recovery(posterior: torch.Tensor, labels: np.ndarray) -> float:
    """The fraction of pairs whose most responsible component is their generating one, under the best renaming."""
    assigned = posterior.argmax(dim=1).numpy()
    k = posterior.shape[1]
    return max(
        float(np.mean(np.asarray(order)[assigned] == labels))
        for order in itertools.permutations(range(k))
    )


def _reference(draw: SimulatedEmissionMixtureDataset) -> tuple[float, float]:
    """The generating parameters' log-likelihood on the draw, and their recovery."""
    values = _values(draw)
    log_weight = torch.log(torch.as_tensor(draw.weights, dtype=torch.float64))
    truth = draw.components
    return (
        float(mixture_log_likelihood(values, log_weight, truth)),
        _recovery(responsibilities(values, log_weight, truth), draw.labels),
    )


@pytest.fixture(scope="module")
def planted() -> tuple[
    SimulatedEmissionMixtureDataset, CountPairSeeding, EmissionMixtureFit
]:
    """Plain EM from one component across generating components 0 and 1 and two identical ones on 2.

    Identical components stay identical under EM; seeded apart, EM reaches the truth.
    """
    draw, at = _draw()
    pairs = draw.observations.astype(float)
    labels = draw.labels
    shared = np.round(pairs[labels <= 1].mean(axis=0))
    third = pairs[labels == 2]
    median = third[np.argsort(third[:, 0])[len(third) // 2]]
    rows = np.stack([shared, median, median])
    fit = expectation_maximization(pairs, _uniform(3), at(rows), tolerance=EM_TOLERANCE)
    return draw, at, fit


@pytest.fixture(scope="module")
def repaired(
    planted: tuple[
        SimulatedEmissionMixtureDataset, CountPairSeeding, EmissionMixtureFit
    ],
) -> SplitMerge:
    """Split-and-merge from the planted fixed point, once: about 16 s, read by two tests."""
    draw, at, plain = planted
    return split_and_merge(
        plain, draw.observations, at, candidates=CANDIDATES, tolerance=EM_TOLERANCE
    )


@pytest.mark.end2end
def test_split_and_merge_repairs_the_planted_fixed_point(
    planted: tuple[
        SimulatedEmissionMixtureDataset, CountPairSeeding, EmissionMixtureFit
    ],
    repaired: SplitMerge,
) -> None:
    draw, _, plain = planted
    reference, reference_recovery = _reference(draw)
    # Plain EM converged, and short of the generating parameters' value by
    # more than 100 nats: it is the fixed point the planted start was built
    # to reach.
    assert plain.termination is not None
    assert plain.termination.converged
    assert plain.log_likelihood < reference - 100.0
    assert repaired.accepted >= 1
    # The maximum a fit of this draw reaches is EM's from the generating
    # parameters themselves, -7836.806, 11.1 nats above their own value: the
    # maximum-likelihood excess. The repair ends there, to EM's tolerance.
    truth = expectation_maximization(
        draw.observations,
        torch.as_tensor(draw.weights, dtype=torch.float64),
        draw.components,
        tolerance=EM_TOLERANCE,
    )
    assert truth.log_likelihood > reference
    assert repaired.fit.log_likelihood == pytest.approx(
        truth.log_likelihood, rel=10 * EM_TOLERANCE
    )
    recovery = _recovery(repaired.fit.responsibilities, draw.labels)
    assert abs(recovery - reference_recovery) <= 0.02


@pytest.mark.analytic
def test_the_likelihood_never_falls_and_only_a_rise_is_kept(
    planted: tuple[
        SimulatedEmissionMixtureDataset, CountPairSeeding, EmissionMixtureFit
    ],
    repaired: SplitMerge,
) -> None:
    _, _, plain = planted
    kept = [plain.log_likelihood]
    for step in repaired.steps:
        # Every step starts from the fit kept last, and is kept only above it.
        assert step.before == kept[-1]
        assert step.accepted == (step.after > step.before)
        if step.accepted:
            kept.append(step.after)
    assert kept == sorted(kept)
    assert repaired.fit.log_likelihood == kept[-1]
    # The last round tried every candidate and kept none.
    last = [s for s in repaired.steps if s.before == kept[-1]]
    assert last
    assert not any(s.accepted for s in last)


@pytest.mark.release
@pytest.mark.end2end
def test_from_the_data_start_split_and_merge_is_read_against_plain_em() -> None:
    # The notebook's `data` start, seed 0: pairs drawn uniformly. Plain EM
    # from it and split-and-merge from plain EM's fit, both against the
    # truth. Split-and-merge cannot end below plain EM.
    draw, at = _draw()
    pairs = draw.observations.astype(float)
    start = uniform_start(pairs, 3, at, np.random.default_rng(0))
    plain = expectation_maximization(pairs, _uniform(3), start, tolerance=EM_TOLERANCE)
    repaired = split_and_merge(
        plain, pairs, at, candidates=CANDIDATES, tolerance=EM_TOLERANCE
    )
    assert repaired.fit.log_likelihood >= plain.log_likelihood
    reference, reference_recovery = _reference(draw)
    assert repaired.fit.log_likelihood >= reference
    assert (
        abs(_recovery(repaired.fit.responsibilities, draw.labels) - reference_recovery)
        <= 0.02
    )


@pytest.mark.analytic
def test_the_criteria_read_shared_and_misfit_components() -> None:
    # Two identical columns have cosine one and orthogonal ones zero; a
    # component whose density is the data it owns has divergence zero, and
    # one that spreads mass where it owns none has a positive one.
    posterior = torch.tensor(
        [[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    merge = merge_criterion(posterior)
    assert float(merge[0, 1]) == pytest.approx(1.0, abs=1e-15)
    assert float(merge[0, 2]) == 0.0
    assert bool(torch.isinf(merge.diagonal()).all())
    exact = torch.log(
        torch.tensor(
            [[0.5, 0.5, 0.25]] * 2 + [[0.25, 0.25, 0.5]] * 2, dtype=torch.float64
        )
    )
    divergence = split_criterion(posterior, exact)
    assert divergence.tolist() == pytest.approx([0.0, 0.0, 0.0], abs=1e-15)
    spread = exact.clone()
    spread[:, 2] = torch.log(torch.tensor(0.25, dtype=torch.float64))
    assert float(split_criterion(posterior, spread)[2]) == pytest.approx(np.log(2.0))
    # The best pair is (0, 1) and the split the one outside it.
    assert candidate_moves(posterior, spread, 1) == [(0, 1, 2)]
    assert candidate_moves(posterior[:, :2], spread[:, :2], 5) == []


@pytest.mark.smoke
def test_split_and_merge_refuses_no_candidates(
    planted: tuple[
        SimulatedEmissionMixtureDataset, CountPairSeeding, EmissionMixtureFit
    ],
) -> None:
    draw, at, plain = planted
    with pytest.raises(ValueError, match="at least 1"):
        split_and_merge(plain, draw.observations, at, candidates=0)
