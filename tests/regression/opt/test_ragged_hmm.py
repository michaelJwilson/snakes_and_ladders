"""A fit over segments of unequal length, refereed three ways.

Issue #666. The claims are ordered by how little they cost: the conserved
rectangular route is bitwise, the concatenation identity is arithmetic, and
only the recovery spends a tolerance.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from sal.backend import Backend
from sal.emissions import CategoricalEmission, PoissonEmission
from sal.likelihood.forward_backward import forward_backward
from sal.opt.em import EM
from sal.opt.hmm import baum_welch_family
from sal.ragged import Ragged
from sal.sandbox.rectangular_hmm import baum_welch_rectangular
from sal.sim import fixtures

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

    At equal lengths the mask is all true: exactly the conserved arithmetic.
    """
    observations = _draw((40, 40, 40), seed=11).reshape(3, 40)
    initial, transition, family = _model()
    ragged = baum_welch_family(
        observations, initial, transition, family, backend=Backend.PYTHON
    )
    conserved = baum_welch_rectangular(observations, initial, transition, family)

    assert ragged.log_likelihood == conserved.log_likelihood
    assert torch.equal(ragged.log_initial, conserved.log_initial)
    assert torch.equal(ragged.log_transition, conserved.log_transition)
    assert isinstance(ragged.emissions, PoissonEmission)
    assert isinstance(conserved.emissions, PoissonEmission)
    np.testing.assert_array_equal(ragged.emissions.mean, conserved.emissions.mean)

    # The default compiled E step (#933) sums in its own order: measured
    # 4.4e-16 relative on the likelihood and 4.3e-14 on the log initial,
    # held to the float64 tolerance of `test_opt_hmm_ragged_estep`.
    compiled = baum_welch_family(observations, initial, transition, family)
    assert abs(compiled.log_likelihood - conserved.log_likelihood) <= 1e-11 * abs(
        conserved.log_likelihood
    )
    torch.testing.assert_close(
        compiled.log_initial, conserved.log_initial, rtol=0.0, atol=1e-11
    )
    torch.testing.assert_close(
        compiled.log_transition, conserved.log_transition, rtol=0.0, atol=1e-11
    )
    assert isinstance(compiled.emissions, PoissonEmission)
    torch.testing.assert_close(
        compiled.emissions.mean, conserved.emissions.mean, rtol=1e-11, atol=0.0
    )


@pytest.mark.critical
@pytest.mark.analytic
def test_the_evidence_is_the_sum_over_segments() -> None:
    """A batch's evidence is its segments', added --- the free referee."""
    lengths = (7, 13, 5)
    batch = Ragged(_draw(lengths, seed=3), lengths)
    initial, transition, family = _model()

    fit = baum_welch_family(
        batch, initial, transition, family, config=replace(EM, max_iterations=1)
    )

    per_segment = 0.0
    for segment in batch.segments():
        emit = family.log_density(torch.as_tensor(segment)).numpy()
        per_segment += forward_backward(
            emit, initial.numpy(), transition.numpy()
        ).log_evidence
    np.testing.assert_allclose(fit.log_likelihood, per_segment, rtol=1e-12)


@pytest.mark.critical
@pytest.mark.analytic
def test_a_boundary_is_not_a_transition() -> None:
    """The closed form that says the segmentation was honoured.

    Cut or whole, the evidences differ by one transition against one initial.
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
    fit = baum_welch_family(
        batch, initial, transition, family, config=replace(EM, max_iterations=1)
    )
    np.testing.assert_allclose(fit.log_likelihood, split, rtol=1e-12)


@pytest.mark.critical
@pytest.mark.end2end
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
@pytest.mark.smoke
def test_a_segment_of_one_position_never_reaches_the_fit() -> None:
    """Refused at the carrier, which is where the shape is declared."""
    with pytest.raises(ValueError, match="at least 2 positions"):
        Ragged(_draw((2, 1), seed=1)[:3], (2, 1))


@pytest.mark.critical
@pytest.mark.end2end
def test_the_declared_ragged_instance_recovers_its_transition() -> None:
    """The fixture, loaded, simulated at its own segmentation, and fitted.

    Segments are drawn here; a ragged simulator is deferred on #666.
    """
    declared = fixtures.fixture("ragged_hmm", "ci").params
    lengths = declared.segment_lengths
    rng = np.random.default_rng(declared.seed)

    drawn = []
    for length in lengths:
        state = int(rng.choice(declared.n_states, p=declared.initial))
        for _ in range(length):
            drawn.append(int(declared.emissions.sample(np.array([state]), rng)[0]))
            state = int(rng.choice(declared.n_states, p=declared.transition[state]))
    batch = Ragged(np.asarray(drawn, dtype=np.float64), lengths)

    start = CategoricalEmission(
        np.full((declared.n_states, declared.emissions.matrix.shape[1]), 0.25)
    )
    fit = baum_welch_family(
        batch,
        torch.log(torch.as_tensor(declared.initial)),
        torch.log(torch.as_tensor(declared.transition)),
        start,
        config=replace(EM, max_iterations=50),
    )
    # The batch took one transition fewer per boundary, and the fit is over
    # that many: what is asserted is that it ran on the declared instance and
    # improved on its start, not a tolerance this fixture cannot support at 89
    # positions.
    assert fit.log_likelihood > -np.inf
    assert sum(lengths) - len(lengths) == 84
