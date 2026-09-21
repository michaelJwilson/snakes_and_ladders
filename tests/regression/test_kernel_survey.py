"""The compiled-kernel survey, asserted against the tree it reads.

Issue #678. `infra/appraise_kernels.py` derives three relations --- which
kernels take the thread pool, what crosses the boundary, and what referees each
one --- and a survey is only worth reading if its derivation is checked. What
is asserted here is the derivation, never a profile: a ratio belongs in
`STATUS.md` beside the measurement that produced it, and a test that pinned one
would fail on a faster host.
"""

from __future__ import annotations

import appraise_kernels
import pytest


@pytest.fixture(scope="module")
def found() -> list[appraise_kernels.Kernel]:
    return appraise_kernels.kernels()


@pytest.mark.critical
@pytest.mark.infra
def test_the_pool_is_read_from_the_source_and_not_from_a_list(
    found: list[appraise_kernels.Kernel],
) -> None:
    # The fact the ticket opens with: two kernels landed after #612 decided the
    # thread pool and neither takes it. Read from each module's own parallel
    # iterators, so a port that lands makes this true without anyone editing a
    # list -- which is the failure #586 deleted the hand inventories over.
    #
    # Stated as containment in both directions and not as equality (issue
    # #745). The equality was the set the day #678 was written, so #715's two
    # max-flow kernels took the pool and turned the guard red on `main`
    # without a defect: an inventory pinned inside an assertion is the hand
    # list one level up, and every later port breaks it. What is asserted is
    # what the guard exists for -- the three that take the pool still do, and
    # the two that were found bare still are.
    pool = {kernel.module for kernel in found if kernel.parallel}

    assert {"coupled", "pruning", "sampling"} <= pool
    assert {"count_pairs", "ragged"} & pool == set()


@pytest.mark.infra
def test_every_kernel_is_pinned_against_a_referee_its_tests_import(
    found: list[appraise_kernels.Kernel],
) -> None:
    # `likelihood/CLAUDE.md`'s oracle rule, asserted over the boundary rather
    # than trusted: every module that exports a `#[pyfunction]` has a test
    # importing a second implementation of what it computes. The build probe is
    # excluded by name and with its reason, because arithmetic is not an oracle.
    unpinned = {
        kernel.module
        for kernel in found
        if kernel.entry_points
        and not kernel.pinned
        and kernel.module not in appraise_kernels.PROBES
    }

    assert unpinned == set()


@pytest.mark.infra
def test_an_oracle_inside_the_adapter_is_a_referee(
    found: list[appraise_kernels.Kernel],
) -> None:
    # The case the first draft of this tool got wrong, which is why the
    # placement is in the vocabulary: `test_ragged_rust.py` imports
    # `posteriors` and `posteriors_oracle` from one module, so the referee is a
    # symbol and not a module. A tool that only looked at modules called a
    # pinned kernel bare, and a survey that reports a false gap is worse than
    # one that reports none.
    ragged = next(kernel for kernel in found if kernel.module == "ragged")

    assert ragged.referees == ("likelihood.ragged_rust.posteriors_oracle",)


@pytest.mark.infra
def test_the_probe_is_excluded_with_its_reason(
    found: list[appraise_kernels.Kernel],
) -> None:
    # `lib.double` crosses the boundary and is not a kernel. An exclusion
    # carries a reason for the same purpose `MEASURED` serves in the structure
    # survey: the next reader re-reads the argument rather than re-making it.
    probe = next(kernel for kernel in found if kernel.module == "lib")

    assert probe.entry_points == ("double",)
    assert "lib" in appraise_kernels.PROBES
    assert "extension-loads probe" in appraise_kernels.PROBES["lib"]


@pytest.mark.infra
def test_the_boundary_names_the_adapter_each_kernel_is_called_through(
    found: list[appraise_kernels.Kernel],
) -> None:
    # The relation a reader actually needs: from a Rust module to the Python it
    # is reached from. Spot-checked on the two the ticket is about, so a kernel
    # that loses its caller -- reachable from a test alone -- shows here.
    adapters = {kernel.module: kernel.adapters for kernel in found}

    assert adapters["count_pairs"] == ("sim.count_pairs_rust",)
    assert adapters["ragged"] == ("likelihood.ragged_rust",)
    assert adapters["coupled"] == ("likelihood.spatio_sequential_rust",)


@pytest.mark.infra
def test_a_use_of_rayon_without_a_parallel_iterator_is_not_parallel(
    found: list[appraise_kernels.Kernel],
) -> None:
    # Importing `rayon` and iterating serially is possible, so the markers are
    # what decide and the import is reported beside them. Every module this
    # calls parallel carries at least one parallel iterator, not just the use
    # line.
    for kernel in found:
        iterators = [m for m in kernel.markers if m != "use rayon"]
        assert kernel.parallel == bool(iterators), kernel.module
