"""A persistent worker: one script, one framework import, many calls (issue #1282, step 2).

Run as ``python -m sal.external.worker <script module> [<framework module>]``
by :func:`sal.external.runner.worker`, and only there. It imports the script
and the framework once, writes ``{"ready": true}``, and serves one request
per line on its standard input until that closes. The script is the one a
one-shot call runs: its ``main`` reads through :func:`~sal.external.protocol.received`
and writes through :func:`~sal.external.protocol.dump` as before, and the
worker chooses what those two read and write.

- ``NPZ``: a request names the ``.npz`` pair; the script reads and writes
  the files, and the worker replies ``{"done": true}``.
- ``SHARED`` and ``MMAP``: a request carries each input's block spec. The
  worker maps the blocks, runs the script on views over them, and replies
  each output's dtype and shape; the parent creates a block per output and
  names them; the worker copies each output into its block, unmaps every
  block and replies ``{"done": true}``. The worker creates and unlinks no
  block.

A script that raises ends the worker as it ends a one-shot script: the
traceback, or what the script wrote, goes to standard error and the process
exits non-zero. The parent reads that as :class:`~sal.external.runner.ScriptError`.

The reply channel is a duplicate of the original standard output; standard
output itself is pointed at standard error, so what a framework prints
cannot corrupt a reply.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import IO, Any

import numpy as np

from sal.external.protocol import Codec, through
from sal.external.transport import Block, attach


def _reply(channel: IO[str], message: Mapping[str, Any]) -> None:
    channel.write(json.dumps(message) + "\n")
    channel.flush()


def _call(script: ModuleType, given: str, returned: str) -> None:
    """Run the script's ``main`` as a one-shot call would, on these two handles."""
    sys.argv = [script.__name__, given, returned]
    script.main()


def serve(
    script: ModuleType, request: Mapping[str, Any], requests: IO[str], channel: IO[str]
) -> None:
    """Answer one request; a two-step transport reads its second line from ``requests``."""
    if request["transport"] == "npz":
        _call(script, request["inputs"], request["outputs"])
        _reply(channel, {"done": True})
        return
    blocks: list[Block] = []
    try:
        inputs: dict[str, np.ndarray] = {}
        for name, spec in request["inputs"].items():
            block = attach(spec)
            blocks.append(block)
            assert block.array is not None
            inputs[name] = block.array
        outputs: dict[str, np.ndarray] = {}

        def write(_: str | Path, arrays: Mapping[str, np.ndarray]) -> None:
            # `np.savez` stores each value as `np.asanyarray` of it, so the
            # bytes match the `.npz` the one-shot call writes.
            outputs.update({name: np.asanyarray(a) for name, a in arrays.items()})

        with through(Codec(read=lambda _: dict(inputs), write=write)):
            _call(script, "-", "-")
        inputs.clear()
        _reply(
            channel,
            {
                "outputs": {
                    name: {"dtype": array.dtype.str, "shape": list(array.shape)}
                    for name, array in outputs.items()
                }
            },
        )
        named = json.loads(requests.readline())["blocks"]
        for name, array in outputs.items():
            block = attach(named[name])
            blocks.append(block)
            assert block.array is not None
            block.array[...] = array
        outputs.clear()
    finally:
        for block in blocks:
            block.release()
    _reply(channel, {"done": True})


def main(argv: list[str] | None = None) -> None:
    """Import the script and its framework, then serve requests until standard input closes."""
    arguments = sys.argv[1:] if argv is None else argv
    script = importlib.import_module(arguments[0])
    for framework in arguments[1:]:
        importlib.import_module(framework)
    channel = os.fdopen(os.dup(sys.stdout.fileno()), "w")
    sys.stdout.flush()
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    _reply(channel, {"ready": True})
    requests = sys.stdin
    for line in requests:
        serve(script, json.loads(line), requests, channel)


if __name__ == "__main__":
    main()
