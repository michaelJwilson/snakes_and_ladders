"""Issue #1000's decision table: PyTorch autograd against JAX, per objective and size.

Each cell runs in a fresh interpreter: the objective is built, one warm-up
gradient is taken (JAX compiles there), and then the median seconds of
``repeats`` gradients is read, with the peak resident memory the gradients
added (``/proc/self/clear_refs`` then ``VmHWM`` over ``VmRSS``, as issue #987
reads a framework's). The rule: JAX replaces PyTorch where it takes at most
half the runtime at no more than 1.25x the memory, or the converse.

Issue #1005 adds the phylogenetic cells (:data:`PRUNING`): the JC
branch-length and GTR objectives on a balanced tree, taxa and sites crossed,
the same rule deciding whether ``Backend.JAX`` becomes their default.

Usage: ``python infra/jax_decision.py [hmm|pruning]`` prints a Markdown table.
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
from sal.opt.hmm import (BetaBinomialHmmObjective, BinomialHmmObjective, GaussianHmmObjective,
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
    from sal.opt.hmm.jax import value_and_grad
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


PRUNING_CELL = r"""
import json, statistics, sys, time
from pathlib import Path
import numpy as np

family, n_taxa, n_sites, backend, repeats = (sys.argv[1], int(sys.argv[2]), int(sys.argv[3]),
    sys.argv[4], int(sys.argv[5]))
from sal.backend import Backend
from sal.likelihood.objective import BranchLengthObjective, SubstitutionModelObjective
from sal.sim.simulate import simulate_alignment
from sal.sim.tree import balanced_tree
tau, k = balanced_tree(n_taxa, 0.5), 4
pi = np.full(k, 1.0 / k)
alignment = dict(simulate_alignment(tau, k, pi, np.random.default_rng(1005), n_sites).alignment)
route = {"torch": {}, "analytic": {"gradient": "analytic"}, "jax": {"backend": Backend.JAX}}[backend]
if family == "jc":
    objective = BranchLengthObjective(tau, k, pi, alignment, **route)
else:
    objective = SubstitutionModelObjective(tau, k, alignment, **route)
theta = objective.initial()
one = lambda: objective.value_and_gradient(theta)
one()
""" + CELL[CELL.index("\ndef status") :]

#: The phylogenetic cells (issue #1005): the objective, then taxa and sites
#: crossed; ``analytic`` is the JC objective's closed-form route beside them.
PRUNING = (("jc", "gtr"), (4, 8, 16, 32), (20_000, 100_000))


def cell(program: str, *arguments: object) -> dict[str, float]:
    """One measurement of ``program`` in a fresh interpreter."""
    out = subprocess.run(
        [sys.executable, "-c", program, *map(str, arguments), str(REPEATS)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(out.strip().splitlines()[-1])  # type: ignore[no-any-return]


def verdict(torch_cell: dict[str, float], jax_cell: dict[str, float]) -> str:
    """The row's ratios and the rule's verdict, as table cells."""
    runtime = jax_cell["seconds"] / torch_cell["seconds"]
    memory = (jax_cell["peak"] + 1) / (torch_cell["peak"] + 1)
    replace = (runtime <= 0.5 and memory <= 1.25) or (memory <= 0.5 and runtime <= 1.25)
    return (
        f"{1e3 * torch_cell['seconds']:.2f} | {1e3 * jax_cell['seconds']:.2f} | "
        f"{runtime:.2f} | {torch_cell['peak'] / 1e6:.1f} | {jax_cell['peak'] / 1e6:.1f} | "
        f"{memory:.2f} | {'replace' if replace else 'sandbox'}"
    )


def hmm() -> None:
    """The HMM table (issue #1000), one row per objective and size."""
    print(
        "| objective | positions | torch ms | jax ms | runtime ratio | torch MB | jax MB | memory ratio | verdict |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for family in FAMILIES:
        for n_sequences in SIZES:
            torch_cell = cell(CELL, family, n_sequences, "torch")
            jax_cell = cell(CELL, family, n_sequences, "jax")
            print(
                f"| {family} | {100 * n_sequences:,} | {verdict(torch_cell, jax_cell)} |",
                flush=True,
            )


def pruning() -> None:
    """The phylogenetic table (issue #1005), with the JC analytic route beside it."""
    print(
        "| objective | taxa | sites | analytic ms | torch ms | jax ms | runtime ratio | torch MB | jax MB | memory ratio | verdict |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    families, taxa, sites = PRUNING
    for family in families:
        for n_taxa in taxa:
            for n_sites in sites:
                torch_cell = cell(PRUNING_CELL, family, n_taxa, n_sites, "torch")
                jax_cell = cell(PRUNING_CELL, family, n_taxa, n_sites, "jax")
                analytic = (
                    f"{1e3 * cell(PRUNING_CELL, family, n_taxa, n_sites, 'analytic')['seconds']:.2f}"
                    if family == "jc"
                    else "-"
                )
                print(
                    f"| {family} | {n_taxa} | {n_sites:,} | {analytic} | {verdict(torch_cell, jax_cell)} |",
                    flush=True,
                )


def main() -> None:
    """Print the table the argument names: ``hmm`` (the default) or ``pruning``."""
    {"hmm": hmm, "pruning": pruning}[sys.argv[1] if len(sys.argv) > 1 else "hmm"]()


if __name__ == "__main__":
    main()
