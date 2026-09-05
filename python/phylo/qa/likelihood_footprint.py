"""QA table: the memory footprint of a likelihood evaluation, against the
declared scale.

``ROADMAP.md`` §1.2 bounds the footprint to ``O(n x L x k)`` inside 16 GB
unified memory or 24 GB VRAM, and ``STATUS.md`` recorded that row as **Not
measured**. This table closes it.

**What is published, and what is measured.** The table is the footprint the
arrays occupy, computed from their shapes: exact arithmetic, identical on every
machine, which is what ``docs/CLAUDE.md`` requires of a published number since
CI byte-compares the rebuilt artifact. A ``tracemalloc`` peak is *not* that --
an earlier draft published one and the continuous-integration runner
regenerated a different table, which is the failure mode the rule exists to
prevent. The measurement is where a tolerance is allowed:
``tests/regression/qa/test_qa_likelihood_footprint.py`` pins every figure here
against the allocator's own count, and the realized agreement is 2.5% at the
smallest cell and 0.3% at the two larger ones.

**Why a caterpillar.** Pruning holds a partial likelihood per *open* node, so
its footprint is set by the topology's depth rather than by its taxa. A
caterpillar is the deepest tree on ``n`` leaves, so it is the worst case, the
one a *requirement* is about, and the shape at which ``O(n x L x k)`` is tight
rather than loose. A balanced tree of the same size costs strictly less, which
is what makes the number a bound.

Wall clock is deliberately absent: ``DEV.md`` forbids ranking performance on CI
hardware, so timings live in ``tests/benchmarks`` instead.
"""

from __future__ import annotations

import tracemalloc

import numpy as np

from phylo.likelihood import pruning
from phylo.qa.figure import QATable, latex_integer
from phylo.qa.runner import table_main
from phylo.search.rl import with_uniform_branch_lengths
from phylo.search.topology import Topology
from phylo.sim.simulate import simulate_alignment
from phylo.sim.tree import Node

#: The declared scale's lower and middle reaches, the sizes the regression
#: suite pins the model against.
MEASURED_SIZES: tuple[tuple[int, int], ...] = ((20, 2_000), (20, 11_000), (100, 11_000))

#: The upper corner of ``ROADMAP.md`` §1.2's declared scale.
DECLARED_MAXIMUM: tuple[int, int] = (1_000, 11_000)

#: The tighter of the two hardware bounds in ``ROADMAP.md`` §1.2 (16 GB unified
#: memory against 24 GB VRAM), so the headroom reported is the one that binds.
MEMORY_BUDGET_BYTES: int = 16 * 1024**3

#: States in the alphabet every figure here is computed at.
N_STATES: int = 4

#: Bytes per entry. Both arrays are ``int64``/``float64``; a backend that moved
#: to ``float32`` would halve the second term and is a different table.
BYTES_PER_ENTRY: int = 8


def caterpillar(n_taxa: int) -> Topology:
    """The deepest unrooted topology on ``n_taxa`` leaves, branch lengths unset.

    Every internal node has one leaf and one internal child, so the post-order
    depth is ``n_taxa - 2`` and pruning holds that many partials at once.

    Parameters
    ----------
    n_taxa : int
        Leaf count, at least 3.

    Returns
    -------
    Topology
        A caterpillar rooted at a trifurcation, as `phylo.search.topology`
        represents an unrooted tree.
    """
    leaves = [Node(name=f"t{index}", branch_length=None) for index in range(n_taxa)]
    tail: Node = leaves[-1]
    for leaf in reversed(leaves[2:-1]):
        tail = Node(name="i", branch_length=None, children=(leaf, tail))
    return Node(name="root", branch_length=None, children=(leaves[0], leaves[1], tail))


def simulation_bytes(n_taxa: int, n_sites: int) -> int:
    """What the simulator holds: every node's states, leaf and internal alike.

    An unrooted tree on ``n_taxa`` leaves has ``2 * n_taxa - 1`` nodes in the
    representation this repository uses, and ``simulate_alignment`` retains all
    of them -- the ancestral states are the truth the validation tests need, so
    keeping them is the point rather than an oversight.

    Returns
    -------
    int
        Bytes.
    """
    return (2 * n_taxa - 1) * n_sites * BYTES_PER_ENTRY


def evaluation_bytes(n_taxa: int, n_sites: int, n_states: int = N_STATES) -> int:
    """What pruning holds on a caterpillar: one ``(n_sites, k)`` partial per node.

    Every node but the root is open at once at the deepest point of a
    caterpillar's post-order, which is what makes this the worst case.

    Returns
    -------
    int
        Bytes.
    """
    return (2 * n_taxa - 2) * n_sites * n_states * BYTES_PER_ENTRY


def measure(n_taxa: int, n_sites: int) -> tuple[float, float]:
    """Peak bytes the allocator counts, simulating and then evaluating.

    Not published: this is what `simulation_bytes` and `evaluation_bytes` are
    checked against in `tests/regression/qa/test_qa_likelihood_footprint.py`.
    A ``tracemalloc`` peak does not survive a change of machine, and the table
    must.

    Returns
    -------
    tuple[float, float]
        Peak simulation bytes and peak evaluation bytes.
    """
    tau = with_uniform_branch_lengths(caterpillar(n_taxa), 0.1)
    pi = np.full(N_STATES, 1.0 / N_STATES)

    tracemalloc.start()
    dataset = simulate_alignment(tau=tau, k=N_STATES, pi=pi, seed=1, n_sites=n_sites)
    _, simulate_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    alignment = dict(dataset.alignment)
    tracemalloc.start()
    pruning.log_likelihood(tau, N_STATES, pi, alignment)
    _, evaluate_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return float(simulate_peak), float(evaluate_peak)


def warm_up() -> None:
    """Discard one measurement, so the next is a property of the problem size.

    The first simulation in a process pays NumPy's own one-time allocations and
    ``tracemalloc`` counts them: measured cold, the smallest cell reads twice
    what it reads warm. Only the regression tests call this, since only they
    measure.
    """
    measure(*MEASURED_SIZES[0])


def _megabytes(value: int) -> str:
    """Format bytes as megabytes, three significant digits."""
    return f"{float(f'{value / 1e6:.3g}'):g}"


def render_footprint() -> str:
    """Build the LaTeX ``tabular``: the measured sizes, then the declared maximum.

    Returns
    -------
    str
        A complete ``tabular`` environment.
    """
    rows = [
        " & ".join(
            [
                latex_integer(n_taxa),
                latex_integer(n_sites),
                _megabytes(simulation_bytes(n_taxa, n_sites)),
                _megabytes(evaluation_bytes(n_taxa, n_sites)),
                _megabytes(
                    simulation_bytes(n_taxa, n_sites)
                    + evaluation_bytes(n_taxa, n_sites)
                ),
            ]
        )
        + r" \\"
        for n_taxa, n_sites in (*MEASURED_SIZES, DECLARED_MAXIMUM)
    ]
    return "\n".join(
        [
            r"\begin{tabular}{rrrrr}",
            r"  \toprule",
            r"  Taxa & Sites & Simulate (MB) & Evaluate (MB) & Total (MB) \\",
            r"  \midrule",
            *(f"  {row}" for row in rows[:-1]),
            r"  \midrule",
            f"  {rows[-1]}",
            r"  \bottomrule",
            r"\end{tabular}",
        ]
    )


def build_caption() -> str:
    """Caption text for the footprint table.

    Returns
    -------
    str
        Plain-text caption, safe to ``\\input`` into LaTeX verbatim.
    """
    taxa, sites = DECLARED_MAXIMUM
    total = simulation_bytes(taxa, sites) + evaluation_bytes(taxa, sites)
    headroom = f"{float(f'{MEMORY_BUDGET_BYTES / total:.2g}'):g}"
    return (
        "Memory held while simulating a Jukes-Cantor alignment and while "
        f"evaluating its likelihood by pruning, at {N_STATES} states, across "
        "the declared scale. The simulator retains every node's states and "
        "pruning retains one partial likelihood per open node, so both are "
        "computed from the arrays' own shapes and are exact rather than "
        "sampled; the regression suite pins each figure against the "
        "allocator's count. The last row is the declared maximum of "
        f"{latex_integer(taxa)} taxa and {latex_integer(sites)} sites, which "
        f"sits a factor of {headroom} inside the 16 GB unified-memory "
        "requirement. The topology is a caterpillar at every size, the deepest "
        "tree on its leaves and so the worst case: a balanced tree of the same "
        "taxon count costs strictly less. Timings are reported in the benchmark "
        "suite instead, because a wall clock ranks the machine rather than the "
        "code."
    )


def build_table() -> tuple[str, str]:
    """Assemble the ``tabular`` body and its caption.

    Returns
    -------
    tuple[str, str]
        The ``tabular`` body and the caption.
    """
    return render_footprint(), build_caption()


def main(argv: list[str] | None = None) -> QATable:
    """Render the footprint table.

    Parameters
    ----------
    argv : list[str] | None
        Argument vector; ``None`` reads ``sys.argv``.

    Returns
    -------
    QATable
        Paths written, and the caption.
    """
    return table_main(
        stem="likelihood_footprint",
        description=__doc__,
        params=(),
        build=build_table,
        argv=argv,
    )


if __name__ == "__main__":
    main()
