"""Forward plus backward through `forward_messages` on a per-step kernel stack (issue #1199).

50 chains of 1,000 positions over K = 16 states, the size issue #1199 was
found at, with a kernel per step: shared by the chains, ``(length - 1, K, K)``,
or each chain's own, ``(n, length - 1, K, K)``. Differentiated with respect to
the scores, the initial distribution and the kernels. The time-constant
kernel at 200 chains of 100-3,000 positions is `test_forward_ragged_bench.py`'s
`ragged` route. The regression files pin values and gradients
(`tests/regression/opt/test_opt_hmm_forward_messages.py`).

Measured at this size (min of three rounds, 4 cores): `shared` 0.18 s and
`per_chain` 0.34 s. Before the inputs were split once with `unbind`, a slice
of ``log_density[:, t]`` and of the kernel per step made the backward pass
write a zero block of the whole input per step, and they took 1.57 s and
67.47 s (issue #1199).
"""

from __future__ import annotations

import pytest
import torch
from sal.opt.hmm.forward import forward_messages

N, LENGTH, K = 50, 1000, 16


def _problem(
    per_chain: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Seeded scores, initial distribution and a per-step kernel stack."""
    generator = torch.Generator().manual_seed(1199)
    density = torch.randn(N, LENGTH, K, generator=generator, dtype=torch.float64)
    initial = torch.log_softmax(
        torch.randn(K, generator=generator, dtype=torch.float64), dim=0
    )
    shape = (N, LENGTH - 1, K, K) if per_chain else (LENGTH - 1, K, K)
    kernels = torch.log_softmax(
        torch.randn(*shape, generator=generator, dtype=torch.float64), dim=-1
    )
    return density, initial, kernels


@pytest.mark.benchmark
@pytest.mark.release
@pytest.mark.parametrize("per_chain", [False, True], ids=["shared", "per_chain"])
def test_forward_messages_per_step_forward_backward(
    benchmark: object, per_chain: bool
) -> None:
    density, initial, kernels = _problem(per_chain)

    def once() -> float:
        scores = density.clone().requires_grad_(True)
        start = initial.clone().requires_grad_(True)
        steps = kernels.clone().requires_grad_(True)
        evidence, _ = forward_messages(scores, start, steps)
        total = evidence.sum()
        total.backward()  # type: ignore[no-untyped-call]
        return float(total.detach())

    total = benchmark.pedantic(once, rounds=3, iterations=1)  # type: ignore[attr-defined]
    # The benchmark refuses a fast wrong answer: the same sum without a graph.
    with torch.no_grad():
        reference = float(forward_messages(density, initial, kernels)[0].sum())
    assert total == reference
