"""`pruning_torch`'s recursion through `torch.compile`, and the facts that declined it (issues #322, #443).

Issue #443 proposed that compiling the pruning forward closes the per-fit gap
on its own. `docs/experiments/007-per-fit-cost-of-the-tree-likelihood.md`
measured it at 1.21x eager at 8 taxa and 1.09x at 20 and declined it, so
`snakes_and_ladders.sandbox.compiled_pruning` keeps the front
(`sandbox/CLAUDE.md`) and this module keeps the three facts the decline rests
on:

- the compiled value and its gradient are the eager ones, so the two sides
  timed against each other computed the same thing;
- Dynamo compiles the recursion with **no graph break**, so the slower side
  is a compiled recursion and not a partial fallback to the faster one --
  without this the whole comparison is eager against eager plus overhead;
- one artifact serves distinct random topologies without recompiling, which
  is what closes the amortization escape hatch the ticket left open: the
  13.6 s cold compile is paid once, and there is still no steady-state win.

The timings are not re-run here. Experiment 007 took them under
`with_lock measure` on the reference host, and a wall clock asserted in the
suite would be a different claim on different hardware (`DEV.md`); what this
module holds is the correctness the ratio was read against.

Not `critical`, and measured rather than assumed: under `with_lock measure`
on the reference host the module costs 9.2 s with a warm
`TORCHINDUCTOR_CACHE_DIR` and 20.5 s with a fresh one, of which every call is
0.01--0.04 s and the rest is the one compile, in a module-scoped fixture the
three tests share. It stays in the
per-PR tier rather than moving to `release` because what it catches is a
torch upgrade that breaks the graph, and an upgrade arrives in a pull
request's lockfile; `release` would report it a version late. One compile
rather than three is what makes that affordable.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch
from snakes_and_ladders.likelihood import pruning_torch
from snakes_and_ladders.likelihood.device import CROSS_DEVICE_RTOL_FLOAT64
from snakes_and_ladders.sandbox import compiled_pruning
from snakes_and_ladders.search.rl import with_uniform_branch_lengths
from snakes_and_ladders.search.topology import random_topology
from snakes_and_ladders.sim.simulate import simulate_alignment
from snakes_and_ladders.sim.tree import Node

# The two sizes experiment 007 reports the ratio at, and its site count.
TAXA = (8, 20)
N_SITES = 1_000

type Problem = tuple[Node, np.ndarray, dict[str, np.ndarray], torch.Tensor]


def _problem(n_taxa: int, seed: int) -> Problem:
    """One random topology, its alignment, and branch lengths to evaluate at.

    The lengths are drawn on [0.05, 0.3], experiment 007's range, rather than
    read back from the tree: a fit evaluates away from the generating
    lengths, and the recursion is the same either way.
    """
    rng = np.random.default_rng(seed)
    names = [f"t{index}" for index in range(n_taxa)]
    tau = with_uniform_branch_lengths(random_topology(names, rng), 0.1)
    pi = np.full(4, 0.25)
    dataset = simulate_alignment(
        tau=tau, k=4, pi=pi, rng=np.random.default_rng(1), n_sites=N_SITES
    )
    lengths = rng.uniform(0.05, 0.3, size=len(pruning_torch.branch_order(tau)))
    return tau, pi, dict(dataset.alignment), torch.tensor(lengths, dtype=torch.float64)


@pytest.fixture(scope="module")
def front() -> Callable[..., torch.Tensor]:
    """One compiled artifact, warmed at both taxon counts.

    Warming at two sizes rather than one is deliberate: the second distinct
    branch count is where Dynamo generalizes the shape, and every reading
    below is of the artifact after that has happened. Compilation state is
    process-wide, so the reset is what makes the counts this module's own.
    """
    compiled_pruning.reset_compilation()
    compiled = compiled_pruning.compiled_log_likelihood()
    for n_taxa in TAXA:
        tau, pi, alignment, lengths = _problem(n_taxa, seed=n_taxa)
        lengths.requires_grad_(True)
        torch.autograd.grad(compiled(tau, 4, pi, alignment, lengths), lengths)
    return compiled


@pytest.mark.oracle
@pytest.mark.parametrize("n_taxa", TAXA)
def test_the_compiled_recursion_is_the_eager_one_to_the_float64_bound(
    front: Callable[..., torch.Tensor], n_taxa: int
) -> None:
    # Pinned at `likelihood/CLAUDE.md`'s float64 backend bound rather than at
    # the 1.8e-16 experiment 007 measured: the bound is the rule this
    # repository applies to one recursion evaluated two ways, and the
    # measurement is one instance of it. The gradient is pinned too, because
    # the ratio that declined the route is a forward *and* a backward.
    tau, pi, alignment, lengths = _problem(n_taxa, seed=100 + n_taxa)
    eager_lengths = lengths.clone().requires_grad_(True)
    eager = pruning_torch.log_likelihood(tau, 4, pi, alignment, eager_lengths)
    (eager_gradient,) = torch.autograd.grad(eager, eager_lengths)

    compiled_lengths = lengths.clone().requires_grad_(True)
    compiled = front(tau, 4, pi, alignment, compiled_lengths)
    (compiled_gradient,) = torch.autograd.grad(compiled, compiled_lengths)

    torch.testing.assert_close(
        compiled, eager, rtol=CROSS_DEVICE_RTOL_FLOAT64, atol=0.0
    )
    torch.testing.assert_close(
        compiled_gradient, eager_gradient, rtol=CROSS_DEVICE_RTOL_FLOAT64, atol=0.0
    )


@pytest.mark.structural
@pytest.mark.usefixtures("front")
def test_the_front_compiles_the_recursion_with_no_graph_break() -> None:
    # The assertion the finding depends on. A graph break runs the broken
    # span eager, so a compiled side that broke would be the eager side plus
    # Dynamo's overhead, and "1.21x eager" would be measuring the guard
    # machinery rather than the compiler.
    state = compiled_pruning.compile_state()

    assert state.graph_breaks == 0
    assert state.frames > 0
    assert state.graphs > 0


@pytest.mark.structural
def test_one_artifact_serves_distinct_topologies_without_recompiling(
    front: Callable[..., torch.Tensor],
) -> None:
    # The amortization escape hatch, closed. If each candidate topology cost
    # its own compile, the 13.6 s would be a per-candidate tax and the
    # decline would be about the compile rather than about the steady state.
    # It is not: the compile is paid once and the ratio still does not pay.
    # Sizes off the warmed pair as well, since a search's candidates are not
    # all one shape. Taped, as the fixture warmed it and as a fit evaluates
    # it: whether the forward is recorded for a backward is itself a guard,
    # so an untaped call here would report a recompile that no fit pays.
    before = compiled_pruning.compile_state()

    for seed, n_taxa in enumerate((8, 8, 12, 16, 20, 20)):
        tau, pi, alignment, lengths = _problem(n_taxa, seed=200 + seed)
        value = front(tau, 4, pi, alignment, lengths.requires_grad_(True))
        assert torch.isfinite(value.detach())

    assert compiled_pruning.compile_state() == before
