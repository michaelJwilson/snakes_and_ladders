"""The package's own calls, measured the way a framework's script measures its own (issue #987).

A benchmark pair reads a framework's time and peak resident memory in a
fresh interpreter; this script reads the package's the same way, so the two
figures come from one method in two interpreters of the same kind and the
process under test carries no measurement. ``call`` names an entry of
:data:`CALLS`, which builds the call from the inputs outside the measured
region; the call itself runs under :func:`~snakes_and_ladders.validation.protocol.timed`
inside :func:`~snakes_and_ladders.validation.protocol.peaked`.

``allocate`` is the measure's own referee: it fills ``n`` float64 values,
``8 n`` bytes the peak must read back.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

from snakes_and_ladders.validation.protocol import dump, load, paths, peaked, timed

#: One output mapping from one measured call.
Outputs = dict[str, np.ndarray]

#: A package call: the inputs in, a zero-argument call out whose run is measured.
Build = Callable[[Mapping[str, np.ndarray]], Callable[[], Outputs]]


def _allocate(inputs: Mapping[str, np.ndarray]) -> Callable[[], Outputs]:
    size = int(inputs["n"])

    def call() -> Outputs:
        block = np.empty(size)
        block.fill(0.0)
        return {"total": np.asarray(block.sum())}

    return call


#: The calls this script measures, by name.
CALLS: dict[str, Build] = {"allocate": _allocate}


def main() -> None:
    """Build the named call, measure it, and write its outputs back."""
    given, returned = paths()
    inputs = load(given)
    call = CALLS[str(inputs["call"])](inputs)
    (outputs, seconds), peak_bytes = peaked(lambda: timed(call))
    dump(returned, outputs, seconds, peak_bytes)


if __name__ == "__main__":
    main()
