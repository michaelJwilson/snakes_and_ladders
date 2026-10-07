"""One OpenGM algorithm on a Potts model the caller posed (issue #1279).

OpenGM is a C++ template library on no package index; ``infra/build_opengm.sh``
compiles its header-only inference with ``infra/opengm/sal_opengm.cxx`` into
``libsal_opengm.so``, which this script loads with ``ctypes``. The library
is loaded here alone, so OpenGM's native code stays out of the package
process, as an imported framework does.

Inputs: ``unary``, the ``(n_nodes, n_states)`` cost ``-h``, finite;
``first`` and ``second``, each edge's ends with ``first < second``;
``coupling``, ``J`` per edge, read as ``-J [a == b]``; ``algorithm``, the
index :data:`sal.external.potts_inputs.OPENGM_ALGORITHMS` assigns;
``max_iterations`` and ``tolerance``, the algorithm's own loop; optionally
``start``, an initial labelling (ICM, expansion, swap). Outputs: ``labels``;
``value``, OpenGM's energy of them; ``bound``, its lower bound, ``-inf``
where the algorithm gives none; ``trace``, the bound after each iteration
the algorithm reports (TRW-S, dual decomposition; loopy BP's entries count
its iterations and are ``-inf``); ``build_seconds``, the ``ctypes`` call less
the inference. The measured seconds are the library's own clock around
``infer()``; the peak resident memory is the whole call's.
"""

from __future__ import annotations

import ctypes

import numpy as np

from sal.external.frameworks import FRAMEWORKS, built
from sal.external.protocol import dump, measured, received

#: Bytes the library may write as an error message.
ERROR_BYTES = 1024


def _pointer(array: np.ndarray) -> ctypes.c_void_p:
    return array.ctypes.data_as(ctypes.c_void_p)


def main() -> None:
    """Load the library, solve, and write the labelling and bounds back."""
    library = built(FRAMEWORKS["opengm"])
    if library is None:
        message = "OpenGM is not built: run `infra/build_opengm.sh`"
        raise RuntimeError(message)
    solve = ctypes.CDLL(str(library)).sal_opengm_solve
    solve.restype = ctypes.c_int

    inputs, returned = received()
    unary = np.ascontiguousarray(inputs["unary"], dtype=np.float64)
    first = np.ascontiguousarray(inputs["first"], dtype=np.int64)
    second = np.ascontiguousarray(inputs["second"], dtype=np.int64)
    coupling = np.ascontiguousarray(inputs["coupling"], dtype=np.float64)
    start = (
        np.ascontiguousarray(inputs["start"], dtype=np.int64)
        if "start" in inputs
        else None
    )
    n_nodes, n_states = unary.shape
    steps = int(inputs["max_iterations"])
    labels = np.zeros(n_nodes, dtype=np.int64)
    trace = np.zeros(max(steps, 1), dtype=np.float64)
    value, bound, seconds = ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
    taken = ctypes.c_int64()
    error = ctypes.create_string_buffer(ERROR_BYTES)

    def call() -> int:
        return int(
            solve(
                ctypes.c_int(int(inputs["algorithm"])),
                ctypes.c_int64(n_nodes),
                ctypes.c_int64(n_states),
                _pointer(unary),
                ctypes.c_int64(first.shape[0]),
                _pointer(first),
                _pointer(second),
                _pointer(coupling),
                None if start is None else _pointer(start),
                ctypes.c_int64(steps),
                ctypes.c_double(float(inputs["tolerance"])),
                _pointer(labels),
                ctypes.byref(value),
                ctypes.byref(bound),
                _pointer(trace),
                ctypes.byref(taken),
                ctypes.byref(seconds),
                error,
                ctypes.c_int64(ERROR_BYTES),
            )
        )

    status, total, peak_bytes = measured(call)
    if status != 0:
        raise RuntimeError(error.value.decode(errors="replace"))
    dump(
        returned,
        {
            "labels": labels,
            "value": np.asarray(value.value, dtype=np.float64),
            "bound": np.asarray(bound.value, dtype=np.float64),
            "trace": trace[: taken.value].copy(),
            "build_seconds": np.asarray(total - seconds.value, dtype=np.float64),
        },
        seconds.value,
        peak_bytes,
    )


if __name__ == "__main__":
    main()
