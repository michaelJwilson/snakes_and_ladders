"""One realization of a downstream package's problem stream, built by its own stage functions (issue #1414).

Run by :func:`sal.external.stream.capture` through
:func:`sal.external.runner.run`. The inputs name the downstream's stage
module (``stage``), the module holding its inference entry (``entry``), a
manifest, a realization index and a scratch directory. The script draws that
realization with the stage module's ``members``, runs the downstream to its
copy-state fit at the planted clones (``at_oracle_clones``) and on to the
clone-assignment field (its ``then``), and returns both problems as arrays.
The scratch directory, the draw and the run's outputs included, is deleted
before the script returns, so only the arrays leave it.

The downstream package is imported here by the names the parent passes; the
parent names it once (:data:`sal.external.stream.STAGE`).
"""

from __future__ import annotations

import importlib
import inspect
import shutil
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from sal.external.protocol import dump, received

#: The fit's arguments kept by name, each an array or a scalar.
FIT_ARGUMENTS = ("t", "init_log_mu", "init_p_binom", "normal_lambda", "clone_lengths")


@contextmanager
def bound_to(original: Callable[..., Any]) -> Iterator[None]:
    """Read the downstream's wrapped inference entry's signature as ``original``'s while the block runs.

    A workaround for a downstream defect, not a change of what it computes:
    its stage driver replaces the inference entry with a keyword-only
    wrapper, and its run recorder binds the call's positional arguments to
    the entry it read at install, that wrapper, so the bind raises
    ``TypeError: too many positional arguments``. Within the block a
    wrapper whose qualified name is the driver's local ``inference`` reports
    ``original``'s signature; every other object is unchanged.
    """
    signature = inspect.signature

    def patched(target: Any, *args: Any, **kwargs: Any) -> inspect.Signature:
        if getattr(target, "__qualname__", "").endswith("<locals>.inference"):
            return signature(original)
        return signature(target, *args, **kwargs)

    inspect.signature = patched  # type: ignore[assignment]
    try:
        yield
    finally:
        inspect.signature = signature


def _max_rdr(model: type) -> float:
    """The model's depth-ratio ceiling, its optimizer's declared default: a row above it is masked."""
    optimizer = getattr(model, "_run_optimization_pipeline", None)
    if optimizer is None:
        return float("inf")
    found = inspect.signature(optimizer).parameters.get("max_rdr")
    if found is None or found.default is None:
        return float("inf")
    return float(found.default)


def capture(inputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Both problems of one realization, as arrays; the scratch directory removed."""
    stage = importlib.import_module(str(inputs["stage"]))
    entry = importlib.import_module(str(inputs["entry"]))
    scratch = Path(str(inputs["scratch"]))
    manifest = Path(str(inputs["manifest"]))
    realization = int(inputs["realization"])
    held: dict[str, Any] = {}
    opened = time.perf_counter()
    try:
        member = next(
            stage.members(manifest, scratch / "draws", n=1, first=realization)
        )
        drawn = time.perf_counter() - opened

        def fit(found: Any) -> None:
            held["fit"] = found
            held["fit_seconds"] = time.perf_counter() - opened

        def field(found: Any) -> None:
            held["field"] = found

        with bound_to(entry.UPSTREAM):
            stage.at_oracle_clones(member.sample, fit, root=scratch / "run", then=field)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    found, potts = held["fit"], held["field"]
    X = np.asarray(found.X)
    arguments = found.arguments
    _, truth = np.unique(np.asarray(found.planted), axis=0, return_inverse=True)
    hmm = {
        "hmm_counts": X[:, 0, 0].astype(np.int64),
        "hmm_successes": X[:, 1, 0].astype(np.int64),
        "hmm_exposure": np.asarray(found.base_nb_mean, dtype=np.float64)[:, 0],
        "hmm_trials": np.asarray(found.total_bb_RD, dtype=np.float64)[:, 0],
        "hmm_lengths": np.asarray(found.lengths, dtype=np.int64),
        "hmm_log_switch": np.asarray(found.args[6], dtype=np.float64),
        "hmm_n_states": np.asarray(found.n_states),
        "hmm_truth": np.asarray(truth, dtype=np.int64).ravel(),
        "hmm_phased": np.asarray("phas" in arguments["hmmclass"].__name__),
        "hmm_max_rdr": np.asarray(_max_rdr(arguments["hmmclass"])),
    }
    hmm.update(
        {
            f"hmm_{name}": np.asarray(arguments[name], dtype=np.float64)
            for name in FIT_ARGUMENTS
        }
    )
    return {
        **hmm,
        "potts_field": np.asarray(potts.field, dtype=np.float64),
        "potts_planted": np.asarray(potts.planted, dtype=np.int64),
        "potts_indptr": np.asarray(potts.indptr, dtype=np.int64),
        "potts_indices": np.asarray(potts.indices, dtype=np.int64),
        "potts_weights": np.asarray(potts.weights, dtype=np.float64),
        "potts_spatial_weight": np.asarray(potts.spatial_weight),
        "realization": np.asarray(realization),
        "hash": np.asarray(str(member.hash)),
        "draw_seconds": np.asarray(drawn),
        "fit_seconds": np.asarray(held["fit_seconds"]),
    }


def main() -> None:
    """Capture the realization the inputs name and write its arrays back."""
    inputs, returned = received()
    opened = time.perf_counter()
    outputs = capture(inputs)
    dump(returned, outputs, time.perf_counter() - opened, 0)


if __name__ == "__main__":
    main()
