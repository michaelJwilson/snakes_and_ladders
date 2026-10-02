"""The one torch forward recursion, `forward_messages`, on padded and rectangular batches (issue #1162).

`forward_log_likelihood_from_density` and the Baum-Welch Python E step both
call it. Referees: on equal lengths the padded form (the table kept, the
evidence gathered at each chain's last position) is the rectangular form
bitwise, for each of the three kernel ranks; and on a ragged batch each
chain's evidence is the sum over every hidden path of that chain alone, by
enumeration, at the float64 tolerance.
"""

from __future__ import annotations

import itertools

import pytest
import torch
from sal.opt.hmm.forward import (
    forward_log_likelihood_from_density,
    forward_messages,
)

#: The float64 tolerance for a reordered recursion (DEV.md; issue #649).
TOLERANCE = 1e-11

N, LENGTH, M = 5, 9, 3


def _problem(rank: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A seeded density, initial distribution and kernel of the given rank."""
    generator = torch.Generator().manual_seed(1162 + rank)
    density = torch.randn(N, LENGTH, M, generator=generator, dtype=torch.float64)
    initial = torch.log_softmax(
        torch.randn(M, generator=generator, dtype=torch.float64), dim=0
    )
    shape = {2: (M, M), 3: (LENGTH - 1, M, M), 4: (N, LENGTH - 1, M, M)}[rank]
    kernels = torch.log_softmax(
        torch.randn(shape, generator=generator, dtype=torch.float64), dim=-1
    )
    return density, initial, kernels


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.parametrize("rank", [2, 3, 4])
def test_padded_form_on_equal_lengths_is_the_rectangular_form_bitwise(
    rank: int,
) -> None:
    # Every chain ends at the last column, so the gathered evidence and the
    # running column's must agree bit for bit, and the table's last column is
    # the running column.
    density, initial, kernels = _problem(rank)
    final = torch.full((N,), LENGTH - 1, dtype=torch.long)
    rectangular, none = forward_messages(density, initial, kernels)
    padded, table = forward_messages(density, initial, kernels, final=final)
    assert none is None
    assert table is not None
    assert table.shape == (N, LENGTH, M)
    assert torch.equal(padded, rectangular)
    assert torch.equal(torch.logsumexp(table[:, -1], dim=1), rectangular)
    if rank < 4:
        # The objective admits the two shared forms and sums the same evidence.
        total = forward_log_likelihood_from_density(density, initial, kernels)
        assert torch.equal(total, rectangular.sum())


def _enumerated(
    density: torch.Tensor, initial: torch.Tensor, kernels: torch.Tensor
) -> float:
    """Log evidence of one chain by summing over every hidden path."""
    length = density.shape[0]
    terms = []
    for path in itertools.product(range(M), repeat=length):
        score = initial[path[0]] + density[0, path[0]]
        for t in range(1, length):
            score = score + kernels[t - 1, path[t - 1], path[t]] + density[t, path[t]]
        terms.append(score)
    return float(torch.logsumexp(torch.stack(terms), dim=0))


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.parametrize("rank", [2, 3, 4])
def test_ragged_evidence_is_the_enumerated_sum_over_paths(rank: int) -> None:
    # Chains of 1-6 positions padded to 6. Padding scores log one, as the E
    # step masks it; each chain's evidence must be its own path sum, which
    # enumeration gives exactly (3^6 = 729 paths at most).
    lengths = [6, 1, 4, 2, 5]
    longest = max(lengths)
    density, initial, kernels = _problem(rank)
    density = density[:, :longest].clone()
    kernels = kernels[..., : longest - 1, :, :] if rank > 2 else kernels
    for row, length in enumerate(lengths):
        density[row, length:] = 0.0
    final = torch.as_tensor(lengths, dtype=torch.long) - 1
    evidence, _ = forward_messages(density, initial, kernels, final=final)
    for row, length in enumerate(lengths):
        # The chain's own per-step kernels, whichever rank carried them.
        own = kernels.expand(longest - 1, M, M) if rank == 2 else kernels
        own = own[row] if rank == 4 else own
        expected = _enumerated(density[row, :length], initial, own)
        assert abs(float(evidence[row]) - expected) <= TOLERANCE * abs(expected)
