"""A downstream package's problem stream, captured into sal's arrays (issue #1414).

A downstream package builds two problems per realization of its simulated
stream with its own stage functions: a count-pair HMM fit at the planted
clones, and the Potts labelling problem its clone assignment is handed after
that fit. :func:`capture` runs those functions on one realization and keeps
both problems as compact arrays (:class:`Captured`); the solvers that run on
them are sal's own, in :mod:`sal.qa.combined_solvers`.

**A runtime import, not an extra.** The downstream package depends on sal, so
locking it as an extra of sal is circular, and ``uv`` refuses it (issue
#1414). It is installed beside sal by hand, and :func:`available` asks whether
it is there without importing it. Where it is absent, nothing here runs and
the figure falls back to sal's seeded fixtures.

**In a subprocess.** As every framework does (``external/CLAUDE.md``), the
package runs in its own interpreter, through :func:`sal.external.runner.run`
and the script ``sal.validation.scripts.stream``: its patches of its own
modules, its threads and its scratch files stay out of this process. The
script removes the realization's draw and run directory before it returns.
:data:`STAGE` and :data:`ENTRY` are the only names of the package in sal, and
the script imports what they name.

**Cache.** :func:`load_or_capture` keeps each realization's arrays as one
``.npz`` under a cache directory, keyed by the manifest's name and the
realization, and reads it back rather than run the package again.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sal.external import runner
from sal.external.protocol import load, save

#: The downstream's stage module: ``members``, ``at_oracle_clones``.
STAGE = "port.studies.stage"
#: The module whose ``UPSTREAM`` is the inference entry the stage driver wraps.
ENTRY = "port.patch.hmrf.core_inference"
#: The manifest of the stream the study reads, in the downstream's tree.
MANIFEST = Path("sim/manifests/study15.toml")
#: Seconds one realization may take: its draw, its run to the field.
TIMEOUT = 1800.0


def available() -> bool:
    """Whether the downstream package is importable here, found without importing it."""
    return runner.installed(STAGE)


@dataclass(frozen=True)
class Captured:
    """One realization's two problems, as arrays.

    The HMM (``hmm_*``): ``counts`` and ``successes`` per row, the row's
    ``exposure`` and ``trials``, the segment ``lengths``, the log phase-switch
    probability per row, ``n_states`` copy states, the planted state per row
    (``truth``), whether the run's model is ``phased``, its depth-ratio ceiling
    ``max_rdr``, and the fit's own arguments: the chain's stay probability
    ``t``, its starting ``init_log_mu`` and ``init_p_binom``, and the per-row
    ``normal_lambda`` and ``clone_lengths``.

    The Potts problem (``potts_*``): the ``(sites, labels)`` field as a
    log-weight, the planted labelling, the directed adjacency as CSR and the
    coupling scale ``spatial_weight``.
    """

    arrays: dict[str, np.ndarray]

    @property
    def realization(self) -> int:
        """The manifest's realization index."""
        return int(self.arrays["realization"])

    @property
    def hash(self) -> str:
        """The downstream's hash of the realization as drawn."""
        return str(self.arrays["hash"])

    @property
    def seconds(self) -> float:
        """Wall seconds of the capture in the subprocess: draw, run to the field."""
        return float(self.arrays["seconds"])

    def __getitem__(self, name: str) -> np.ndarray:
        return self.arrays[name]


def capture(manifest: Path, realization: int, scratch: Path) -> Captured:
    """``realization`` of ``manifest``, both problems built by the downstream's stage functions.

    Raises
    ------
    ModuleNotFoundError
        If the downstream package is not installed (:func:`available`).
    sal.external.runner.ScriptError
        If the script exits non-zero; the message carries its standard error.
    """
    if not available():
        msg = f"{STAGE} is not installed; it is a runtime import, installed by hand"
        raise ModuleNotFoundError(msg)
    ran = runner.run(
        "stream",
        {
            "stage": np.asarray(STAGE),
            "entry": np.asarray(ENTRY),
            "manifest": np.asarray(str(manifest)),
            "realization": np.asarray(realization),
            "scratch": np.asarray(str(scratch)),
        },
        timeout=TIMEOUT,
    )
    return Captured({**ran.outputs, "seconds": np.asarray(ran.seconds)})


def cached(cache: Path, manifest: Path, realization: int) -> Path:
    """Where :func:`load_or_capture` keeps ``realization`` of ``manifest``."""
    return cache / f"{Path(manifest).stem}-r{realization:03d}.npz"


def load_or_capture(cache: Path, manifest: Path, realization: int) -> Captured:
    """The cached arrays of ``realization``, captured and kept first if absent."""
    path = cached(cache, manifest, realization)
    if path.is_file():
        return Captured(load(path))
    found = capture(manifest, realization, cache / f".scratch-r{realization}")
    cache.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}")
    save(partial, found.arrays)
    partial.replace(path)
    return found
