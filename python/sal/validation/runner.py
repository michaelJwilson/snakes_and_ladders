"""Run a validation script in a subprocess and read its answer back (issue #972).

:func:`run` is the one path every adapter takes to a framework. It writes the
inputs to an ``.npz`` in a temporary directory, runs the script as a module
under ``sys.executable`` so the subprocess sees the environment the caller
sees, and reads the outputs and the seconds the script measured around the
framework's own call. Interpreter start-up and the file round trip are not
in that figure, which is what a benchmark pairs against the package's time.

:func:`available` answers whether a framework is installed without importing
it: ``importlib.util.find_spec`` locates a module and executes nothing, so
the package process stays free of the framework even while asking.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sal.validation.protocol import PEAK_BYTES, SECONDS, load, save

#: Where the scripts live, as a module path.
SCRIPTS = "sal.validation.scripts"


class ScriptError(RuntimeError):
    """A script exited non-zero; the message carries its standard error."""


@dataclass(frozen=True)
class Run:
    """What a script returned: its outputs and the seconds it measured."""

    outputs: dict[str, np.ndarray]
    #: Wall seconds around the framework's own call, measured in the script.
    seconds: float
    #: Resident bytes the call added at its peak; zero where the script
    #: measured none, which every reader substituted by hand (issue #1010).
    peak_bytes: int = 0


def available(module: str) -> bool:
    """Whether ``module`` is importable here, found without being imported."""
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def run(
    script: str,
    inputs: Mapping[str, np.ndarray],
    *,
    timeout: float = 600.0,
) -> Run:
    """Run ``scripts/<script>.py`` on ``inputs`` and return what it wrote.

    Raises :class:`ScriptError` when the script exits non-zero and
    ``subprocess.TimeoutExpired`` past ``timeout`` seconds.
    """
    with tempfile.TemporaryDirectory(prefix="sal-validation-") as directory:
        given = Path(directory) / "inputs.npz"
        returned = Path(directory) / "outputs.npz"
        save(given, inputs)
        completed = subprocess.run(
            [sys.executable, "-m", f"{SCRIPTS}.{script}", str(given), str(returned)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if completed.returncode != 0:
            message = (
                f"{script} exited {completed.returncode}:\n{completed.stderr.strip()}"
            )
            raise ScriptError(message)
        outputs = load(returned)
    seconds = float(outputs.pop(SECONDS))
    peak = outputs.pop(PEAK_BYTES, None)
    return Run(
        outputs=outputs,
        seconds=seconds,
        peak_bytes=0 if peak is None else int(peak),
    )


def package(
    call: str, inputs: Mapping[str, np.ndarray], *, timeout: float = 600.0
) -> Run:
    """Run the package's ``call`` in ``scripts/package.py``, measured as a framework is."""
    return run("package", {**inputs, "call": np.asarray(call)}, timeout=timeout)
