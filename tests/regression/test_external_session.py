"""`external.session` serves many calls from one worker and returns the one-shot call's bytes (issue #1282, steps 2 and 2b).

Read against the one-shot `runner.run` and `invoke`: every transport returns
the same outputs, bitwise in value, dtype and shape, on `scripts/selftest.py`
(NumPy alone); PyMaxflow's session is pinned in
`tests/validation/test_pymaxflow.py`. Ownership is read from
`/dev/shm` and the session's directory: no block or file outlives a call,
including one whose worker died or was killed.
"""

from __future__ import annotations

import os
import signal
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sal.external import (
    Capability,
    CapabilityRefused,
    ExternalUnavailable,
    ScriptError,
    Session,
    Solver,
    Transport,
    runner,
    session,
    sessions,
    solvers,
)

#: Every dtype a session admits, an empty array and a 0-d array among them.
INPUTS = {
    "values": np.linspace(-1.0, 1.0, 7),
    "labels": np.arange(6, dtype=np.int64).reshape(2, 3),
    "flags": np.array([True, False, True]),
    "empty": np.zeros((0, 3)),
    "scalar": np.asarray(3, dtype=np.int64),
}


def _same(ours: runner.Run, reference: runner.Run) -> None:
    """Every output equal in bytes, dtype and shape, the scripts' clocks aside."""
    assert ours.outputs.keys() == reference.outputs.keys()
    for name, array in reference.outputs.items():
        if name.endswith("seconds"):
            continue
        got = ours.outputs[name]
        assert (got.dtype, got.shape) == (array.dtype, array.shape), name
        assert got.tobytes() == array.tobytes(), name


def _orphans(opened: Session) -> list[str]:
    """Files of ``opened`` still on disk, inputs and outputs alike."""
    files = [p.name for p in Path(opened._directory.name).iterdir()]
    return [f for f in files if f != "stderr"]


@pytest.fixture(params=list(Transport), ids=str)
def echo(request: pytest.FixtureRequest) -> Iterator[Session]:
    with Session("selftest", transport=request.param) as opened:
        yield opened


@pytest.mark.critical
@pytest.mark.infra
@pytest.mark.patch
def test_every_transport_returns_the_one_shot_bytes(echo: Session) -> None:
    reference = runner.run("selftest", INPUTS)
    for _ in range(3):
        _same(echo.run(INPUTS), reference)
    assert _orphans(echo) == []


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("inputs", "named"),
    [
        ({"values": np.zeros(3, dtype=np.float32)}, "values is float32"),
        ({"values": np.zeros((3, 2)).T}, "values is not C-contiguous"),
        ({"values": [0.0, 1.0]}, "values is a list, not an ndarray"),
        ({"call": np.asarray("cut")}, "call is <U3"),
    ],
)
def test_an_inadmissible_input_is_refused_before_the_call(
    echo: Session, inputs: dict[str, Any], named: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[Any] = []
    monkeypatch.setattr(echo, "_send", sent.append)
    with pytest.raises(TypeError, match=named):
        echo.run(inputs)
    assert sent == []
    assert _orphans(echo) == []


@pytest.mark.smoke
def test_a_capability_refusal_reaches_no_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[Any] = []
    with Session("selftest", solver=Solver.PYMAXFLOW_EXACT) as opened:
        monkeypatch.setattr(opened, "_send", sent.append)
        with pytest.raises(CapabilityRefused, match="multi_label"):
            opened.invoke({Capability.MULTI_LABEL}, INPUTS)
    assert sent == []


@pytest.mark.smoke
def test_an_absent_framework_starts_no_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_: object) -> None:
        message = "no worker may start"
        raise AssertionError(message)

    monkeypatch.setattr(solvers, "installed", lambda _: False)
    monkeypatch.setattr(sessions, "worker", refuse)
    with (
        pytest.raises(ExternalUnavailable, match="validation-pymaxflow"),
        session(Solver.PYMAXFLOW_EXACT),
    ):
        pass


@pytest.mark.infra
def test_the_worker_ends_with_the_block_normally_and_on_an_exception() -> None:
    with Session("selftest") as opened:
        pid = opened.pid
        assert Path(f"/proc/{pid}").exists()
    assert opened.pid is None
    assert not Path(f"/proc/{pid}").exists()
    directory = Path(opened._directory.name)
    assert not directory.exists()

    def raises_inside() -> None:
        # The public entry point, on HiGHS, which SciPy always carries.
        nonlocal pid
        with session(Solver.HIGHS_LP) as raising:
            pid = raising.pid
            message = "raised in the block"
            raise KeyError(message)

    with pytest.raises(KeyError, match="raised in the block"):
        raises_inside()
    assert not Path(f"/proc/{pid}").exists()


@pytest.mark.infra
def test_a_script_failure_is_a_script_error_with_its_stderr(echo: Session) -> None:
    # The worker dies mid-call, its input blocks mapped: `fail` exits after
    # `received`, as the one-shot script does.
    failing = {**INPUTS, "fail": np.asarray(3, dtype=np.int64)}
    with pytest.raises(ScriptError, match="selftest exited 3") as died:
        echo.run(failing)
    assert "selftest asked to fail with 3" in str(died.value)
    assert _orphans(echo) == []
    with pytest.raises(ScriptError, match="session is closed"):
        echo.run(INPUTS)


@pytest.mark.critical
@pytest.mark.infra
def test_a_worker_killed_between_calls_leaves_no_block(echo: Session) -> None:
    echo.run(INPUTS)
    assert echo.pid is not None
    os.kill(echo.pid, signal.SIGKILL)
    with pytest.raises(ScriptError, match="selftest exited -9"):
        echo.run(INPUTS)
    assert _orphans(echo) == []


@pytest.mark.critical
@pytest.mark.infra
@pytest.mark.parametrize("transport", [Transport.MMAP], ids=str)
def test_a_worker_killed_mid_call_leaves_no_block(
    transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Killed after its first reply: its input blocks are mapped, its outputs
    # declared and the parent's output blocks about to be created.
    with Session("selftest", transport=transport) as opened:
        read = opened._read

        def killing(expected: set[str], timeout: float) -> dict[str, Any]:
            reply = read(expected, timeout)
            if "outputs" in reply:
                assert opened._process is not None
                opened._process.send_signal(signal.SIGKILL)
                opened._process.wait()
            return reply

        monkeypatch.setattr(opened, "_read", killing)
        with pytest.raises(ScriptError, match="selftest exited -9"):
            opened.run(INPUTS)
        assert _orphans(opened) == []


@pytest.mark.infra
def test_a_timeout_ends_the_worker() -> None:
    with Session("selftest") as opened:
        large = {"values": np.zeros(1_250_000)}
        with pytest.raises(subprocess.TimeoutExpired):
            opened.run(large, timeout=1e-6)
        assert opened.pid is None
        assert _orphans(opened) == []
