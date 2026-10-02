"""Forward plus backward over ragged segments: `forward_log_likelihood_ragged` against padding and masking by hand (issue #1167).

The stress size of issue #933: 200 chains of 100-3,000 positions over four
states. `ragged` is the entry point, which scatters the segments into a
block scoring log one at padding and gathers each segment's evidence from the
kept message table. `hand_masked` is what a caller wrote before it: the same
block, padded by hand, and a recursion that freezes each chain's message once
its segment ends, so no table is kept and the evidence is read off the last
column. Both are differentiated with respect to the scores and the transition
matrix; the regression file pins the entry point's value and gradients.

Measured at that size (min of three rounds, 4 cores): `ragged` 0.56 s against
`hand_masked` 9.99 s. Before `forward_messages` stacked its columns, a slice
assignment per step made the backward pass copy the whole table per step, and
`ragged` took 15.80 s against 10.75 s (issue #1167); before it split its
inputs once with `unbind`, a slice read per step did the same to the scores,
and `ragged` took 9.22 s against 11.32 s (issue #1199).
"""

from __future__ import annotations

import math

import pytest
import torch
from sal.opt.hmm import forward_log_likelihood_ragged

M = 4


def _problem() -> tuple[torch.Tensor, list[int], torch.Tensor, torch.Tensor]:
    """Seeded scores in `Ragged.values` layout, lengths, initial and transition."""
    generator = torch.Generator().manual_seed(1167)
    lengths = torch.randint(100, 3000, (200,), generator=generator).tolist()
    density = torch.randn(sum(lengths), M, generator=generator, dtype=torch.float64)
    initial = torch.full((M,), -math.log(M), dtype=torch.float64)
    transition = torch.log_softmax(
        torch.randn(M, M, generator=generator, dtype=torch.float64), dim=1
    )
    return density, lengths, initial, transition


def _hand_masked(
    density: torch.Tensor,
    lengths: list[int],
    initial: torch.Tensor,
    transition: torch.Tensor,
) -> torch.Tensor:
    """Pad with ``torch.split`` and ``pad_sequence``, then mask the recursion step by step."""
    block = torch.nn.utils.rnn.pad_sequence(
        list(torch.split(density, lengths)), batch_first=True
    )
    live = torch.arange(block.shape[1]).unsqueeze(0) < torch.as_tensor(
        lengths
    ).unsqueeze(1)
    alpha = initial.unsqueeze(0) + block[:, 0]
    for t in range(1, block.shape[1]):
        step = torch.logsumexp(alpha.unsqueeze(2) + transition, dim=1) + block[:, t]
        alpha = torch.where(live[:, t, None], step, alpha)
    return torch.logsumexp(alpha, dim=1).sum()


@pytest.mark.benchmark
@pytest.mark.release
@pytest.mark.parametrize("route", ["ragged", "hand_masked"])
def test_forward_backward_at_stress_size(benchmark: object, route: str) -> None:
    density, lengths, initial, transition = _problem()
    run = forward_log_likelihood_ragged if route == "ragged" else _hand_masked

    def once() -> float:
        scores = density.clone().requires_grad_(True)
        kernel = transition.clone().requires_grad_(True)
        total = run(scores, lengths, initial, kernel)
        total.backward()  # type: ignore[no-untyped-call]
        return float(total.detach())

    total = benchmark.pedantic(once, rounds=3, iterations=1)  # type: ignore[attr-defined]
    # The two routes are the same sum; the benchmark refuses a fast wrong one.
    reference = float(
        forward_log_likelihood_ragged(density, lengths, initial, transition)
    )
    assert abs(total - reference) <= 1e-9 * abs(reference)
