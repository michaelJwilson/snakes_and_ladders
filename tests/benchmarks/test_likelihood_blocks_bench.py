"""What the block-frequency interval costs against the evaluation it stands in for (issue #408).

Correctness is pinned in
tests/regression/likelihood/test_likelihood_blocks.py. What is measured here
is the trade the frequency cutoff sets: at ``min_count = 1`` the interval is
the exact value over the site-pattern table, and each larger cutoff drops
blocks from the exact half, so the width rises. The uncompressed evaluation
is the denominator.

**The cost does not fall with the cutoff, and that is the result.** One
thread on the reference host, mean over 50 calls:

    tree_search/ci, 5 taxa, 2 000 sites, 427 patterns
      uncompressed evaluation   0.908 ms      interval, cutoff 1   3.650 ms
      pattern-compressed        0.660 ms      interval, cutoff 8   3.314 ms
      per-tree extremes         0.384 ms      interval, cutoff 32  3.047 ms
    tree_jc/ci, 4 taxa, 20 000 sites, 256 patterns
      uncompressed evaluation   3.974 ms      interval, cutoff 1  23.183 ms
      pattern-compressed        0.511 ms      interval, cutoff 8  22.699 ms
      per-tree extremes         0.257 ms      interval, cutoff 32 22.588 ms

The interval costs 4.0x the uncompressed evaluation at 2 000 sites and 5.8x
at 20 000, and raising the cutoff from 1 to 32 buys 17% and 2.5%. Subtracting
the two terms that scale with the cutoff --- the exact evaluation, at most
0.66 ms, and the extremes, 0.26 ms and independent of the alignment --- leaves
the block partition and its ``np.unique`` as everything else, and that sorts
every block whatever the cutoff. So this is not a cheaper forward pass and
must not be used as one.

What it is for is a cheaper *fit*: a branch-length fit costs 254 ms at this
fixture (`STATUS.md`, the plug-in bound), so replacing one fit with one
interval wins by 70x, and the ranked search's 2 fits against 13 is where the
saving is measured. Making the interval itself cheap is a separate decision
against a profile, and issue #408 does not take it.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.likelihood.blocks import block_frequency_interval
from snakes_and_ladders.likelihood.pruning_torch import (
    branch_lengths_from_tree,
    log_likelihood,
)
from snakes_and_ladders.sim.simulate import simulate_alignment

from tests._fixtures import load_fixture

FIXTURE = "tree_search/ci.yaml"  # 5 taxa
N_SITES = 2000


@pytest.mark.parametrize("cutoff", [None, 1, 8, 32])
def test_block_frequency_interval_benchmark(
    benchmark: BenchmarkFixture, cutoff: int | None
) -> None:
    params = load_fixture(FIXTURE)
    alignment = dict(
        simulate_alignment(
            tau=params.tau,
            k=params.k,
            pi=params.pi,
            rng=np.random.default_rng(params.seed),
            n_sites=N_SITES,
        ).alignment
    )
    pi = np.asarray(params.pi)
    lengths = branch_lengths_from_tree(params.tau)

    # cutoff None is the uncompressed evaluation this is measured against.
    call: Callable[[], float] = (
        (lambda: float(log_likelihood(params.tau, params.k, pi, alignment, lengths)))
        if cutoff is None
        else (
            lambda: float(
                block_frequency_interval(
                    params.tau,
                    params.k,
                    pi,
                    alignment,
                    lengths,
                    block_size=1,
                    min_count=cutoff,
                ).lower
            )
        )
    )

    result = benchmark(call)

    assert math.isfinite(result)
    assert result < 0.0
