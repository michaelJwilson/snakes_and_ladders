"""``torch.compile`` fronting :func:`~snakes_and_ladders.likelihood.pruning_torch.log_likelihood` (issues #322, #443).

The front that was measured and declined. Issue #443 proposed that compiling
the pruning forward closes the per-fit gap on its own, at a compile cost a
search of 300 candidates amortizes, and
`docs/experiments/007-per-fit-cost-of-the-tree-likelihood.md` measured it:
forward and backward at 1,000 sites, one thread on a shared four-core Linux
x86-64 host under the exclusive lock, torch 2.13.0, the compiled recursion
took **1.21x** eager at 8 taxa and **1.09x** at 20. The compile is 13.6 s on a
fresh ``TORCHINDUCTOR_CACHE_DIR`` and 3.7--3.8 s on a warm one, and it is
amortized -- 12 distinct random topologies at both sizes were served without
recompiling -- so amortization is not what stops it. There is no steady-state
win to amortize.

**What this module is, stated first.** ``torch.compile`` is a compiler over
the recursion ``pruning_torch`` already states, not a second implementation of
it: :func:`compiled_log_likelihood` is one call, and no equation is restated
here. What the one call cannot say is whether the artifact it returns is a
compiled recursion or a silent fallback to eager, and that distinction is the
whole finding -- a decline measured against a fallback would be a decline
against eager twice. So the module adds the one thing a caller cannot write
beside the call: :func:`compile_state`, which reads Dynamo's compile counters,
turning "0 graph breaks" and "one artifact, many topologies" into assertions.
The solution conserved here is `tests/regression/likelihood/test_pruning_torch_compile.py`;
this module is the surface it needs, and it is deliberately no larger.

Reading ``torch._dynamo.utils.counters`` is reading a private API, which is
why it is here and in one place rather than spelled out in the test: a torch
release that moves the counters breaks one function.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch._dynamo

from snakes_and_ladders.likelihood import pruning_torch


def compiled_log_likelihood() -> Callable[..., torch.Tensor]:
    """A fresh ``torch.compile`` artifact over ``pruning_torch.log_likelihood``.

    Default mode, which is the mode experiment 007 measured; ``fullgraph``
    and ``mode="reduce-overhead"`` were not measured and the experiment
    claims nothing about them.

    A factory rather than a module-level artifact, because
    :func:`compile_state` counts compilations process-wide: a caller that
    wants to know what its own calls compiled needs its own artifact, and a
    shared one would report every other caller's frames as well.

    Returns
    -------
    Callable[..., torch.Tensor]
        Argument for argument what
        :func:`~snakes_and_ladders.likelihood.pruning_torch.log_likelihood`
        takes and returns, so the two can be pinned against each other and
        timed against each other without an adapter in between. Compilation
        is lazy: it happens on the first call, not here.
    """
    compiled: Callable[..., torch.Tensor] = torch.compile(pruning_torch.log_likelihood)
    return compiled


@dataclass(frozen=True)
class CompileState:
    """What Dynamo has compiled in this process, as experiment 007 reports it.

    Attributes
    ----------
    frames : int
        Python frames compiled without falling back -- the "4 frames" the
        experiment quotes.
    graphs : int
        Distinct FX graphs produced. One artifact serving a second topology
        leaves this unchanged, which is the amortization claim.
    graph_breaks : int
        Points where Dynamo gave up and ran eager. Zero, or the recursion
        that was timed is partly the one it was timed against.
    """

    frames: int
    graphs: int
    graph_breaks: int


def reset_compilation() -> None:
    """Drop every compiled artifact and zero the counters :func:`compile_state` reads.

    Compilation state is process-wide, so a caller that reads counts has to
    start from a known zero rather than from whatever ran before it.
    """
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()


def compile_state() -> CompileState:
    """Dynamo's compile counters, since the last :func:`reset_compilation`.

    Returns
    -------
    CompileState
        Process-wide, not per artifact: torch counts compilations globally
        and offers no per-artifact view, so a reading is only about one
        artifact if the caller made only one.
    """
    counters = torch._dynamo.utils.counters
    return CompileState(
        frames=int(counters["frames"]["ok"]),
        graphs=int(counters["stats"]["unique_graphs"]),
        graph_breaks=int(sum(counters["graph_break"].values())),
    )


__all__ = [
    "CompileState",
    "compile_state",
    "compiled_log_likelihood",
    "reset_compilation",
]
