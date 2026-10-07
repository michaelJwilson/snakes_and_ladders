"""One worker process for many calls to one solver (issue #1282, steps 2 and 2b).

A one-shot :func:`~sal.external.solvers.invoke` starts an interpreter, imports
NumPy and the framework, and makes the call: about 95 ms before the
framework's own work on the reference host, and over a second for BlackJAX
(#1282's baseline). :func:`session` pays that once::

    with external.session(Solver.PYMAXFLOW_EXACT) as s:
        for inputs in problems:
            run = s.invoke({Capability.GROUND_STATE}, inputs)

The worker is :mod:`sal.external.worker`, started by
:func:`sal.external.runner.worker`, and runs the same script a one-shot call
runs, so a call's protocol is unchanged and its outputs are bitwise those of
:func:`~sal.external.solvers.invoke`. ``transport`` chooses how the arrays
cross (:class:`~sal.external.transport.Transport`); every choice returns the
same bytes.

The parent owns every file a call uses: it creates
each, and releases each when the call returns or fails. The worker is shut
down when the ``with`` block exits, normally or by an exception; a worker
that dies surfaces as :class:`~sal.external.runner.ScriptError` carrying its
standard error, and the session is closed from then on.
"""

from __future__ import annotations

import json
import os
import secrets
import selectors
import subprocess
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import IO, Any

import numpy as np

from sal.external.protocol import load, save
from sal.external.runner import Run, ScriptError, as_run, worker
from sal.external.solvers import (
    Capability,
    ExternalUnavailable,
    Solver,
    available,
    require,
)
from sal.external.transport import Block, Transport, admit, create

#: Seconds the worker is given to exit once its input closes.
SHUTDOWN = 10.0


class Session:
    """A worker serving one script; open it with :func:`session`."""

    def __init__(
        self,
        script: str,
        *,
        framework: str | None = None,
        solver: Solver | None = None,
        transport: Transport = Transport.MMAP,
        timeout: float = 600.0,
    ) -> None:
        self.script = script
        self.solver = solver
        self.transport = Transport(transport)
        self._directory = tempfile.TemporaryDirectory(prefix="sal-session-")
        #: The prefix of every block file this session creates.
        self.prefix = f"sal{os.getpid()}{secrets.token_hex(3)}_"
        self._count = 0
        self._mark = 0
        self._stderr: IO[bytes] = (Path(self._directory.name) / "stderr").open("w+b")
        self._process: subprocess.Popen[str] | None = worker(
            script, framework, self._stderr
        )
        try:
            self._read({"ready"}, timeout)
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def pid(self) -> int | None:
        """The worker's process id, ``None`` once it has exited."""
        return None if self._process is None else self._process.pid

    def invoke(
        self,
        needs: Iterable[Capability],
        inputs: Mapping[str, np.ndarray],
        *,
        timeout: float = 600.0,
    ) -> Run:
        """:func:`~sal.external.solvers.invoke`, served by this session's worker.

        The capability check runs here, before anything reaches the worker.
        """
        if self.solver is None:
            message = f"a session on {self.script} serves no solver; call run"
            raise TypeError(message)
        require(self.solver, needs)
        return self.run(inputs, timeout=timeout)

    def run(self, inputs: Mapping[str, np.ndarray], *, timeout: float = 600.0) -> Run:
        """:func:`~sal.external.runner.run` on this session's script, in its worker.

        Raises :class:`TypeError` for an input :func:`admit` refuses, before
        the call; :class:`ScriptError` when the worker has died or dies
        during the call; ``subprocess.TimeoutExpired`` when a reply takes
        longer than ``timeout`` seconds. Any failure during the call ends the
        worker and closes the session.
        """
        admit(inputs)
        if self._process is None:
            message = f"{self.script} session is closed"
            raise ScriptError(message)
        self._mark = self._stderr.seek(0, os.SEEK_END)
        try:
            if self.transport is Transport.NPZ:
                return self._run_npz(inputs, timeout)
            return self._run_blocks(inputs, timeout)
        except BaseException:
            # A call cut short leaves the worker mid-request; it serves no more.
            self._abandon()
            raise

    def _abandon(self) -> None:
        """Kill the worker, if it is still alive, and close the session."""
        process, self._process = self._process, None
        if process is None:
            return
        if process.poll() is None:
            process.kill()
            process.wait()
        for pipe in (process.stdin, process.stdout):
            if pipe is not None:
                with suppress(BrokenPipeError):
                    pipe.close()

    def _handle(self) -> str:
        """A fresh block file in the session's directory."""
        self._count += 1
        return str(Path(self._directory.name) / f"{self.prefix}{self._count}")

    def _run_npz(self, inputs: Mapping[str, np.ndarray], timeout: float) -> Run:
        self._count += 1
        given = Path(self._directory.name) / f"{self._count}-inputs.npz"
        returned = Path(self._directory.name) / f"{self._count}-outputs.npz"
        try:
            save(given, inputs)
            self._send(
                {"transport": "npz", "inputs": str(given), "outputs": str(returned)}
            )
            self._read({"done"}, timeout)
            return as_run(load(returned))
        finally:
            given.unlink(missing_ok=True)
            returned.unlink(missing_ok=True)

    def _run_blocks(self, inputs: Mapping[str, np.ndarray], timeout: float) -> Run:
        blocks: list[Block] = []
        try:
            specs = {}
            for name, array in inputs.items():
                block = create(self.transport, self._handle(), array.dtype, array.shape)
                blocks.append(block)
                assert block.array is not None
                block.array[...] = array
                specs[name] = block.spec
            self._send({"transport": str(self.transport), "inputs": specs})
            declared = self._read({"outputs"}, timeout)["outputs"]
            outputs: dict[str, Block] = {}
            for name, spec in declared.items():
                block = create(
                    self.transport,
                    self._handle(),
                    np.dtype(spec["dtype"]),
                    tuple(spec["shape"]),
                )
                blocks.append(block)
                outputs[name] = block
            self._send({"blocks": {name: b.spec for name, b in outputs.items()}})
            self._read({"done"}, timeout)
            return as_run({name: np.array(b.array) for name, b in outputs.items()})
        finally:
            for block in blocks:
                block.release()

    def _send(self, message: Mapping[str, Any]) -> None:
        assert self._process is not None
        assert self._process.stdin is not None
        try:
            self._process.stdin.write(json.dumps(message) + "\n")
            self._process.stdin.flush()
        except BrokenPipeError:
            self._died()

    def _read(self, expected: set[str], timeout: float) -> dict[str, Any]:
        """The worker's next reply, which must carry one of ``expected``."""
        assert self._process is not None
        stdout = self._process.stdout
        assert stdout is not None
        with selectors.DefaultSelector() as selector:
            selector.register(stdout, selectors.EVENT_READ)
            if not selector.select(timeout):
                raise subprocess.TimeoutExpired(self.script, timeout)
        line = stdout.readline()
        if not line:
            self._died()
        reply: dict[str, Any] = json.loads(line)
        if not expected & reply.keys():
            message = f"{self.script} worker replied {reply}, expected {expected}"
            raise ScriptError(message)
        return reply

    def _died(self) -> None:
        """Raise the worker's exit as a one-shot script's: its code and its standard error."""
        assert self._process is not None
        code = self._process.wait()
        self._abandon()
        self._stderr.seek(self._mark)
        stderr = self._stderr.read().decode(errors="replace").strip()
        message = f"{self.script} exited {code}:\n{stderr}"
        raise ScriptError(message)

    def close(self) -> None:
        """End the worker, then remove the session's directory and every file in it."""
        process, self._process = self._process, None
        if process is not None:
            assert process.stdin is not None
            with suppress(BrokenPipeError):
                process.stdin.close()
            try:
                process.wait(SHUTDOWN)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            if process.stdout is not None:
                process.stdout.close()
        self._stderr.close()
        self._directory.cleanup()


def served_by(session: Session | None, solver: Solver) -> None:
    """Refuse a ``session`` opened on another solver than ``solver``.

    Every problem family's call reads it before it sends anything, so the
    refusal is one wording; raises :class:`ValueError`.
    """
    if session is not None and session.solver is not solver:
        msg = f"the session serves {session.solver}, not {solver}"
        raise ValueError(msg)


@contextmanager
def session(
    solver: Solver,
    *,
    transport: Transport = Transport.MMAP,
    timeout: float = 600.0,
) -> Iterator[Session]:
    """A worker for ``solver``, alive for the ``with`` block and shut down on its exit.

    ``timeout`` bounds the worker's start-up, its imports included. Raises
    :class:`~sal.external.solvers.ExternalUnavailable` before any process
    starts where the framework is absent.
    """
    if not available(solver):
        raise ExternalUnavailable(solver)
    framework = solver.framework
    opened = Session(
        framework.name,
        framework=framework.module,
        solver=solver,
        transport=transport,
        timeout=timeout,
    )
    try:
        yield opened
    finally:
        opened.close()
