"""Issue #1000's decision table: PyTorch autograd against JAX, per objective and size.

Each cell runs in a fresh interpreter: the objective is built, one warm-up
gradient is taken (JAX compiles there), and then the median seconds of
``repeats`` gradients is read, with the peak resident memory the gradients
added (``/proc/self/clear_refs`` then ``VmHWM`` over ``VmRSS``, as issue #987
reads a framework's). The rule: JAX replaces PyTorch where it takes at most
half the runtime at no more than 1.25x the memory, or the converse.

Usage: ``python infra/jax_decision.py`` prints a Markdown table.
"""

from __future__ import annotations

import json
import subprocess
import sys

FAMILIES = (
    "categorical",
    "gaussian",
    "poisson",
    "binomial",
    "negative_binomial",
    "beta_binomial",
)
SIZES = (100, 1_000)
REPEATS = 7

CELL = r"""
import json, statistics, sys, time
from pathlib import Path
import numpy as np, torch

family, n_sequences, backend, repeats = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
from snakes_and_ladders.opt.hmm import (BetaBinomialHmmObjective, BinomialHmmObjective, GaussianHmmObjective,
    HmmObjective, NegativeBinomialHmmObjective, PoissonHmmObjective)
rng = np.random.default_rng(1000)
shape = (n_sequences, 100)
objective = {
    "categorical": lambda: HmmObjective(rng.integers(0, 4, size=shape), 3, 4),
    "gaussian": lambda: GaussianHmmObjective(rng.normal(size=shape), 3),
    "poisson": lambda: PoissonHmmObjective(rng.poisson(6.0, size=shape), 3),
    "binomial": lambda: BinomialHmmObjective(rng.binomial(20, 0.4, size=shape), 3, np.full(3, 20.0)),
    "negative_binomial": lambda: NegativeBinomialHmmObjective(rng.poisson(6.0, size=shape), 3),
    "beta_binomial": lambda: BetaBinomialHmmObjective(rng.binomial(20, 0.4, size=shape), 3, np.full(3, 20.0)),
}[family]()
theta = objective.initial()
if backend == "jax":
    from snakes_and_ladders.opt.hmm_jax import value_and_grad
    twin = value_and_grad(objective)
    point = theta.numpy()
    one = lambda: twin(point)
else:
    def one():
        p = theta.clone().requires_grad_(True)
        return torch.autograd.grad(objective(p), p)
one()

def status(field):
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1]) * 1024
Path("/proc/self/clear_refs").write_text("5")
before = status("VmRSS")
seconds = []
for _ in range(repeats):
    start = time.perf_counter(); one(); seconds.append(time.perf_counter() - start)
print(json.dumps({"seconds": statistics.median(seconds), "peak": max(status("VmHWM") - before, 0)}))
"""


def cell(family: str, n_sequences: int, backend: str) -> dict[str, float]:
    """One measurement in a fresh interpreter."""
    out = subprocess.run(
        [sys.executable, "-c", CELL, family, str(n_sequences), backend, str(REPEATS)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(out.strip().splitlines()[-1])  # type: ignore[no-any-return]


def main() -> None:
    """Print the table, one row per objective and size, with the rule's verdict."""
    print(
        "| objective | positions | torch ms | jax ms | runtime ratio | torch MB | jax MB | memory ratio | verdict |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for family in FAMILIES:
        for n_sequences in SIZES:
            torch_cell = cell(family, n_sequences, "torch")
            jax_cell = cell(family, n_sequences, "jax")
            runtime = jax_cell["seconds"] / torch_cell["seconds"]
            memory = (jax_cell["peak"] + 1) / (torch_cell["peak"] + 1)
            replace = (runtime <= 0.5 and memory <= 1.25) or (
                memory <= 0.5 and runtime <= 1.25
            )
            print(
                f"| {family} | {100 * n_sequences:,} | {1e3 * torch_cell['seconds']:.2f} | "
                f"{1e3 * jax_cell['seconds']:.2f} | {runtime:.2f} | {torch_cell['peak'] / 1e6:.1f} | "
                f"{jax_cell['peak'] / 1e6:.1f} | {memory:.2f} | {'replace' if replace else 'sandbox'} |",
                flush=True,
            )


if __name__ == "__main__":
    main()
