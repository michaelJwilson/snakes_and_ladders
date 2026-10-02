"""Forward plus backward under a Kronecker switch: the factored step against a dense per-step kernel (issue #1186).

The stress size of issue #933, 200 chains of 100-3,000 positions, at ``K`` slow
states, so ``2 K`` states under the fast layer. `factored` is
`forward_log_likelihood_ragged(..., switch, switch_kind=KRONECKER)`, which
applies ``S_t`` to the layer axis and ``A`` to the slow axis, ``2 K^2 + 4 K``
terms per row. `dense` builds each step's ``2 K x 2 K`` log kernel
``log A ⊗ log S_t`` from the same tensors and takes the log-domain step
`forward_messages` takes, ``4 K^2`` terms per row, with the chains masked by
hand as `test_forward_ragged_bench.py` masks them. Both are differentiated
with respect to the scores, the transition and the switch; the regression
file pins the factored value and gradients against the compiled kernel.

Measured at that size (min of three rounds, 4 cores): `factored` 1.40 s
against `dense` 3.99 s at K = 4, 2.9x, and 2.37 s against 53.34 s at K = 16,
22.5x. Before both split their per-step inputs with `unbind`, a slice per step
made the backward pass write a zero block of the whole input per step, and
`factored` took 76.76 s and 301.29 s (issue #1186).
"""

from __future__ import annotations

import pytest
import torch
from sal.opt.hmm import SwitchKind, forward_log_likelihood_ragged

Problem = tuple[torch.Tensor, list[int], torch.Tensor, torch.Tensor, torch.Tensor]


def _problem(slow: int) -> Problem:
    """Seeded scores, lengths, initial, slow transition and switch at ``slow`` states."""
    generator = torch.Generator().manual_seed(1186)
    lengths = torch.randint(100, 3000, (200,), generator=generator).tolist()
    total, n_states = sum(lengths), 2 * slow
    density = torch.randn(total, n_states, generator=generator, dtype=torch.float64)
    initial = torch.log_softmax(
        torch.randn(n_states, generator=generator, dtype=torch.float64), dim=0
    )
    transition = torch.log_softmax(
        torch.randn(slow, slow, generator=generator, dtype=torch.float64), dim=1
    )
    switch = torch.rand(total, generator=generator, dtype=torch.float64)
    return density, lengths, initial, transition, switch


def _factored(
    density: torch.Tensor,
    lengths: list[int],
    initial: torch.Tensor,
    transition: torch.Tensor,
    switch: torch.Tensor,
) -> torch.Tensor:
    """The entry point under the Kronecker switch."""
    return forward_log_likelihood_ragged(
        density,
        lengths,
        initial,
        transition,
        switch=switch,
        switch_kind=SwitchKind.KRONECKER,
    )


def _dense(
    density: torch.Tensor,
    lengths: list[int],
    initial: torch.Tensor,
    transition: torch.Tensor,
    switch: torch.Tensor,
) -> torch.Tensor:
    """Each step's ``2 K x 2 K`` log kernel built, then the log-domain step."""
    slow = transition.shape[0]
    pad = torch.nn.utils.rnn.pad_sequence
    block = pad(list(torch.split(density, lengths)), batch_first=True)
    steps = pad(list(torch.split(switch, lengths)), batch_first=True)
    n = block.shape[0]
    live = torch.arange(block.shape[1]).unsqueeze(0) < torch.as_tensor(
        lengths
    ).unsqueeze(1)
    flip = 1.0 - torch.eye(2, dtype=torch.float64)
    # Split once, as the entry point splits: a slice per step would make the
    # backward pass write a zero block of the whole input per step.
    scores, switches = block.unbind(dim=1), steps.unbind(dim=1)
    alpha = initial.unsqueeze(0) + scores[0]
    for t in range(1, block.shape[1]):
        s = switches[t][:, None, None]
        # `S_t` per chain, then `log A[i, j] + log S_t[a, b]` at `(2 i + a, 2 j + b)`.
        layer = torch.log((1.0 - s) * (1.0 - flip) + s * flip)
        kernel = (
            transition[None, :, None, :, None] + layer[:, None, :, None, :]
        ).reshape(n, 2 * slow, 2 * slow)
        step = torch.logsumexp(alpha.unsqueeze(2) + kernel, dim=1) + scores[t]
        alpha = torch.where(live[:, t, None], step, alpha)
    return torch.logsumexp(alpha, dim=1).sum()


@pytest.mark.benchmark
@pytest.mark.release
@pytest.mark.parametrize("slow", [4, 16], ids=lambda k: f"K={k}")
@pytest.mark.parametrize("route", ["factored", "dense"])
def test_switched_forward_backward_at_stress_size(
    benchmark: object, route: str, slow: int
) -> None:
    density, lengths, initial, transition, switch = _problem(slow)
    run = _factored if route == "factored" else _dense

    def once() -> float:
        scores = density.clone().requires_grad_(True)
        kernel = transition.clone().requires_grad_(True)
        probability = switch.clone().requires_grad_(True)
        total = run(scores, lengths, initial, kernel, probability)
        total.backward()  # type: ignore[no-untyped-call]
        return float(total.detach())

    total = benchmark.pedantic(once, rounds=3, iterations=1)  # type: ignore[attr-defined]
    # The two routes are the same sum; the benchmark refuses a fast wrong one.
    reference = float(_factored(density, lengths, initial, transition, switch))
    assert abs(total - reference) <= 1e-9 * abs(reference)
