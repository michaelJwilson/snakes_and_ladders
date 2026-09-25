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
    # Two kernels landed after #612 without the thread pool, read from each
    # module's parallel iterators. Containment both ways, not equality: an
    # inventory in an assertion broke on #715's ports without a defect (#745).
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
def test_the_gateway_beside_the_twin_is_a_referee(
    found: list[appraise_kernels.Kernel],
) -> None:
    # The referee was a symbol until #1059: `test_ragged_rust.py` imported
    # `posteriors` and `posteriors_oracle` from the adapter itself, and a
    # module-only survey reported a false gap. With the twin at
    # `likelihood.ragged.rust`, the oracle sits in the gateway beside it, the
    # sibling the survey reads for every other twin.
    ragged = next(kernel for kernel in found if kernel.module == "ragged")

    # `opt.hmm` calls the kernel too (#933, R5), and its torch recursion is its
    # oracle, so the adapter `opt.hmm.estimation` (#1010) is itself a referee.
    assert "likelihood.ragged" in ragged.referees
    assert ragged.referees == (
        "likelihood.message_passing_reference",
        "likelihood.ragged",
        "opt.hmm.estimation",
        "sandbox.rectangular_hmm",
    )


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

    assert adapters["count_pairs"] == ("sim.count_pairs.rust",)
    # Two callers since #933 (R5): `opt.hmm` may not import `likelihood`, so
    # Baum-Welch's compiled E step reaches the kernel through the extension.
    assert adapters["ragged"] == ("likelihood.ragged.rust", "opt.hmm.estimation")
    assert adapters["coupled"] == ("likelihood.spatio_sequential.rust",)


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
