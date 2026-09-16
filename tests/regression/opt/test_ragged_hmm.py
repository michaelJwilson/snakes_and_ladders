"""A fit over segments of unequal length, refereed three ways.

Issue #666. The claims are ordered by how little they cost: the conserved
rectangular route is bitwise, the concatenation identity is arithmetic, and
only the recovery spends a tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from snakes_and_ladders.emissions import PoissonEmission
from snakes_and_ladders.likelihood.forward_backward import forward_backward
from snakes_and_ladders.opt.hmm import baum_welch_family
from snakes_and_ladders.ragged import Ragged
from snakes_and_ladders.sandbox.rectangular_hmm import baum_welch_rectangular

#: One chain, two states, a sticky kernel and well-separated rates.
STATES = 2
STICKY = np.array([[0.9, 0.1], [0.15, 0.85]])
RATES = np.array([2.0, 12.0])


def _model() -> tuple[torch.Tensor, torch.Tensor, PoissonEmission]:
    """The parameters a fit starts from, shared by every test here."""
    return (
        torch.log(torch.full((STATES,), 1.0 / STATES, dtype=torch.float64)),
        torch.log(torch.as_tensor(STICKY, dtype=torch.float64)),
        PoissonEmission(mean=np.array([3.0, 9.0])),
    )


def _draw(lengths: tuple[int, ...], seed: int) -> np.ndarray:
    """Counts from the planted chain, concatenated over the segments."""
    rng = np.random.default_rng(seed)
    drawn = []
    for length in lengths:
        state = int(rng.integers(STATES))
        for _ in range(length):
            drawn.append(rng.poisson(RATES[state]))
            state = int(rng.choice(STATES, p=STICKY[state]))
    return np.asarray(drawn, dtype=np.float64)


@pytest.mark.critical
@pytest.mark.oracle
def test_equal_lengths_reproduce_the_conserved_route_bitwise() -> None:
    """The sandbox rule's own standard, and the guard on the whole refactor.

    At equal lengths the ragged path's mask is everywhere true, so every
    `torch.where` must fall through to exactly the conserved arithmetic in the
    same order. A masked form that reassociates a sum, or gathers the evidence
    from the wrong column, differs here in the last bits and nowhere else.
    """
    observations = _draw((40, 40, 40), seed=11).reshape(3, 40)
    initial, transition, family = _model()
    ragged = baum_welch_family(observations, initial, transition, family)
    conserved = baum_welch_rectangular(observations, initial, transition, family)

    assert ragged.log_likelihood == conserved.log_likelihood
    assert torch.equal(ragged.log_initial, conserved.log_initial)
    assert torch.equal(ragged.log_transition, conserved.log_transition)
    assert isinstance(ragged.emissions, PoissonEmission)
    assert isinstance(conserved.emissions, PoissonEmission)
    np.testing.assert_array_equal(ragged.emissions.mean, conserved.emissions.mean)


@pytest.mark.critical
@pytest.mark.mathematical
def test_the_evidence_is_the_sum_over_segments() -> None:
    """A batch's evidence is its segments', added --- the free referee.

    Nothing is fitted here: the claim is about the E step alone, so it rests on
    arithmetic rather than on a second implementation agreeing.
    """
    lengths = (7, 13, 5)
    batch = Ragged(_draw(lengths, seed=3), lengths)
    initial, transition, family = _model()

    fit = baum_welch_family(batch, initial, transition, family, max_iterations=1)

    per_segment = 0.0
    for segment in batch.segments():
        emit = family.log_density(torch.as_tensor(segment)).numpy()
        per_segment += forward_backward(
            emit, initial.numpy(), transition.numpy()
        ).log_evidence
    np.testing.assert_allclose(fit.log_likelihood, per_segment, rtol=1e-12)


@pytest.mark.critical
@pytest.mark.mathematical
def test_a_boundary_is_not_a_transition() -> None:
    """The closed form that says the segmentation was honoured.

    The same observations, cut once in the middle or left whole. The whole
    chain pays one transition for the step across the cut; the split pays an
    initial distribution instead. The difference between the two evidences is
    therefore exactly those two terms, and nothing else.
    """
    initial, transition, family = _model()
    counts = _draw((12,), seed=5)
    emit = family.log_density(torch.as_tensor(counts)).numpy()

    whole = forward_backward(emit, initial.numpy(), transition.numpy()).log_evidence
    split = sum(
        forward_backward(part, initial.numpy(), transition.numpy()).log_evidence
        for part in (emit[:6], emit[6:])
    )
    assert whole != split, "a cut that changed nothing is not a boundary"

    batch = Ragged(counts, (6, 6))
    fit = baum_welch_family(batch, initial, transition, family, max_iterations=1)
    np.testing.assert_allclose(fit.log_likelihood, split, rtol=1e-12)


@pytest.mark.critical
@pytest.mark.simulated_truth
def test_the_planted_rates_are_recovered_from_ragged_segments() -> None:
    """Unequal lengths, one far longer than the rest, and the truth comes back."""
    lengths = (30, 200, 45, 15)
    batch = Ragged(_draw(lengths, seed=17), lengths)
    initial, transition, family = _model()

    fit = baum_welch_family(batch, initial, transition, family)

    assert isinstance(fit.emissions, PoissonEmission)
    recovered = np.sort(fit.emissions.mean)
    np.testing.assert_allclose(recovered, np.sort(RATES), rtol=0.15)


@pytest.mark.critical
@pytest.mark.edge_case
def test_a_segment_of_one_position_never_reaches_the_fit() -> None:
    """Refused at the carrier, which is where the shape is declared."""
    with pytest.raises(ValueError, match="at least 2 positions"):
        Ragged(_draw((2, 1), seed=1)[:3], (2, 1))
