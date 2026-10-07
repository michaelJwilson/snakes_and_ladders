"""Per-call overhead of an external call, one-shot against a session per transport (issue #1282, steps 2 and 2b).

Each row times `CALLS` calls of `scripts/selftest.py`, which echoes its
inputs and doubles `values`, so the time is the transport and the process
and not a framework's work: `runner.run` starts an interpreter per call, and
a session starts one worker and serves every call from it. The bitwise
agreement of the outputs is pinned in
`tests/regression/test_external_session.py`.

Sizes: 0.2 MB is the gate size. The stress sizes, 10 and 128 MB at 1, 10 and
100 calls, and PyMaxflow's lattice cuts, are reported in the pull request that
introduced the session, read on the 4-core reference host.
"""

from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from sal.external import Session, Transport, runner

#: Calls per timed round.
CALLS = 10

#: The gate size, 0.2 MB of `float64`.
INPUTS = {"values": np.ones(25_000)}


def test_one_shot_calls_benchmark(benchmark: BenchmarkFixture) -> None:
    runs = benchmark.pedantic(  # type: ignore[no-untyped-call]
        lambda: [runner.run("selftest", INPUTS) for _ in range(CALLS)],
        rounds=3,
        iterations=1,
    )
    assert np.array_equal(runs[-1].outputs["doubled"], 2.0 * INPUTS["values"])


@pytest.mark.parametrize("transport", list(Transport), ids=str)
def test_session_calls_benchmark(
    benchmark: BenchmarkFixture, transport: Transport
) -> None:
    def calls() -> list[runner.Run]:
        with Session("selftest", transport=transport) as opened:
            return [opened.run(INPUTS) for _ in range(CALLS)]

    runs = benchmark.pedantic(calls, rounds=3, iterations=1)  # type: ignore[no-untyped-call]
    assert np.array_equal(runs[-1].outputs["doubled"], 2.0 * INPUTS["values"])
