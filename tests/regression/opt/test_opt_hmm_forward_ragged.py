"""The differentiable forward log-likelihood over ragged segments, `forward_log_likelihood_ragged` (issue #1167).

Referees: on equal lengths it is `forward_log_likelihood_from_density`
bitwise; on unequal lengths it is the summed per-segment evidence of
`sal.likelihood.ragged.posteriors` (the compiled kernel) within 1e-12, and
each segment run alone through the rectangular path; its gradient with respect
to the scores is the posterior marginal of that kernel within 1e-12 (the
Fisher identity); and its gradient with respect to the transition matrix is
the central difference within the measured truncation error.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from sal.likelihood.ragged import posteriors
from sal.opt.hmm import (
    forward_log_likelihood_from_density,
    forward_log_likelihood_ragged,
)
from sal.ragged import Ragged

#: Absolute tolerance against the compiled ragged kernel (issue #1167).
ORACLE_TOLERANCE = 1e-12

#: Central-difference step, and the tolerance on the derivative it gives:
#: truncation ``h^2 f'''/6`` and rounding ``eps |f| / h`` both sit near 1e-10
#: here; the realized maximum error is 2.4e-10, against 4.9e-9 at h = 1e-4
#: and 2.1e-9 at h = 1e-6 (issue #1167).
STEP = 1e-5
DIFFERENCE_TOLERANCE = 1e-9

M = 3
#: Unequal segments, each at least `MINIMUM_LENGTH` so `Ragged` admits them.
LENGTHS = (7, 2, 12, 5, 9, 3)


def _parameters(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A seeded log initial distribution and log transition matrix."""
    generator = torch.Generator().manual_seed(seed)
    initial = torch.log_softmax(
        torch.randn(M, generator=generator, dtype=torch.float64), dim=0
    )
    transition = torch.log_softmax(
        torch.randn(M, M, generator=generator, dtype=torch.float64), dim=1
    )
    return initial, transition


def _density(total: int, seed: int) -> torch.Tensor:
    """Seeded scores in `Ragged.values` layout."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(total, M, generator=generator, dtype=torch.float64)


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.backend
def test_equal_lengths_are_the_rectangular_path_bitwise() -> None:
    # Five chains of eight positions: the ragged entry reshapes and runs the
    # same recursion, so the sum must agree bit for bit.
    initial, transition = _parameters(1167)
    density = _density(5 * 8, 1167)
    ragged = forward_log_likelihood_ragged(density, [8] * 5, initial, transition)
    rectangular = forward_log_likelihood_from_density(
        density.reshape(5, 8, M), initial, transition
    )
    assert torch.equal(ragged, rectangular)


@pytest.mark.critical
@pytest.mark.oracle
def test_unequal_lengths_match_the_compiled_ragged_evidence() -> None:
    # The compiled kernel restarts each segment at the initial distribution;
    # the padded torch recursion must sum to the same total.
    initial, transition = _parameters(1168)
    density = _density(sum(LENGTHS), 1168)
    total = forward_log_likelihood_ragged(density, LENGTHS, initial, transition)
    expected = posteriors(
        Ragged(values=density.numpy(), lengths=LENGTHS),
        initial.numpy(),
        transition.numpy(),
    ).log_evidence.sum()
    assert abs(float(total) - float(expected)) <= ORACLE_TOLERANCE


@pytest.mark.oracle
@pytest.mark.backend
def test_unequal_lengths_are_each_segment_run_alone() -> None:
    # Segments run one at a time through the rectangular path, a length-one
    # segment among them, which `Ragged` refuses and the recursion admits.
    lengths = [*LENGTHS, 1]
    initial, transition = _parameters(1169)
    density = _density(sum(lengths), 1169)
    total = forward_log_likelihood_ragged(density, lengths, initial, transition)
    alone = math.fsum(
        float(
            forward_log_likelihood_from_density(piece.unsqueeze(0), initial, transition)
        )
        for piece in torch.split(density, lengths)
    )
    assert abs(float(total) - alone) <= ORACLE_TOLERANCE


@pytest.mark.critical
@pytest.mark.oracle
def test_score_gradient_is_the_posterior_marginal() -> None:
    # Fisher identity: d log Z / d log_density[t, i] = P(z_t = i | x). The
    # gradient flows through the padded scatter back to the live rows only.
    initial, transition = _parameters(1170)
    density = _density(sum(LENGTHS), 1170).requires_grad_(True)
    forward_log_likelihood_ragged(density, LENGTHS, initial, transition).backward()  # type: ignore[no-untyped-call]
    assert density.grad is not None
    marginal = np.exp(
        posteriors(
            Ragged(values=density.detach().numpy(), lengths=LENGTHS),
            initial.numpy(),
            transition.numpy(),
        ).log_posterior
    )
    np.testing.assert_allclose(
        density.grad.numpy(), marginal, rtol=0.0, atol=ORACLE_TOLERANCE
    )


@pytest.mark.analytic
def test_transition_gradient_matches_central_differences() -> None:
    # Every entry of the transition matrix is perturbed by +-h, unconstrained:
    # the rows no longer sum to one, which the recursion must not assume.
    initial, transition = _parameters(1171)
    density = _density(sum(LENGTHS), 1171)
    variable = transition.clone().requires_grad_(True)
    forward_log_likelihood_ragged(density, LENGTHS, initial, variable).backward()  # type: ignore[no-untyped-call]
    assert variable.grad is not None
    numeric = torch.zeros_like(transition)
    for i in range(M):
        for j in range(M):
            shift = torch.zeros_like(transition)
            shift[i, j] = STEP
            up = forward_log_likelihood_ragged(
                density, LENGTHS, initial, transition + shift
            )
            down = forward_log_likelihood_ragged(
                density, LENGTHS, initial, transition - shift
            )
            numeric[i, j] = (up - down) / (2 * STEP)
    error = float((variable.grad - numeric).abs().max())
    assert error <= DIFFERENCE_TOLERANCE


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("lengths", "shape"),
    [((), (0, M)), ((4, 0), (4, M)), ((3, 3), (5, M)), ((5,), (5,))],
    ids=["none", "zero", "sum", "rank"],
)
def test_refuses_lengths_that_do_not_tile_the_scores(
    lengths: tuple[int, ...], shape: tuple[int, ...]
) -> None:
    initial, transition = _parameters(1172)
    with pytest.raises(ValueError, match="log_density|lengths"):
        forward_log_likelihood_ragged(
            torch.zeros(shape, dtype=torch.float64), lengths, initial, transition
        )
