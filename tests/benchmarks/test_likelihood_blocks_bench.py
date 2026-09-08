"""What the block-frequency interval costs against the evaluation it stands in for (issue #408).

Correctness is pinned in
tests/regression/likelihood/test_likelihood_blocks.py. What is measured here
is the trade the frequency cutoff sets: at ``min_count = 1`` the interval is
the exact value over the site-pattern table, and each larger cutoff drops
blocks from the exact half, so the cost falls and the width rises. The
uncompressed evaluation is the denominator.
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
