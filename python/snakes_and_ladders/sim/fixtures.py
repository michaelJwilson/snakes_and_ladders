"""The registry of supported problem instances (issue #382).

`PROBLEMS.md` names the problems this repository supports; this module is
where an *instance* of one is obtained. A fixture is a file under
``tests/regression/fixtures/<problem>/<tier>.yaml`` stating a size, a seed,
the model that reads it and the oracle that referees a result at that size,
and :func:`fixture` returns it loaded through that model's own loader.

**A supported instance is a fixture, never a literal.** An instance typed into
a test, a notebook or a figure script is one nothing else can find: the next
caller writes its own, the two drift, and a claim about "the problem" is a
claim about whichever copy the reader opened. Naming the instance in one file
makes "every method is applied to every problem" a question the tools can
answer rather than a reviewer.

**The files live under ``tests/`` and this module lives in the package**
because the consumers are not only the suite: ``snakes_and_ladders.qa``
renders figures from fixtures and the notebooks run on them, and neither may
import from ``tests/``. The directory is a constant here and a parameter of
every function, so a caller working from a different tree passes its own.

**A baseline record is a second file, never a block in the fixture** (issue
#401). ``<tier>.baseline.json`` beside ``<tier>.yaml`` holds what reference
algorithms achieve on the instance --- an enumerated maximum, the rate at
which random-restart hill climbing reaches it, the rate an untrained policy
reaches it. The fixture declares an instance and is written by hand; the
record states a measurement and is written by ``infra/baselines.py``, so a
regenerated measurement never rewrites a declaration and a reviewer reading
a diff can tell which of the two moved.

**A record carries no digest of the tree it was written from** (issue #460).
It did until then: one hash over the fixture bytes, the transitive import
closure of the computing modules and three library versions, which
:func:`baseline` compared with the current tree's and raised on. Over the whole
history of the five records that field moved 54 times and carried a moved
number 0 times --- every re-key was a source edit somewhere in a closure
spanning 66 of the 145 tracked source files, and the recomputation that
followed reproduced the values byte for byte. It also made 8 of the 38
conflicts across the eight open branches. ``infra/baselines.py`` now recomputes
the records a change could have moved, per pull request, and every record at
the release gate.

What survives here is the check a recomputation cannot make, being a fact about
the machine rather than the tree: :func:`baseline` refuses a record whose
``numpy``, ``scipy`` or ``torch`` version is not the installed one, and that
refusal is still a :class:`StaleBaselineError`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from snakes_and_ladders.fixtures import Scale
from snakes_and_ladders.inputs import library_versions
from snakes_and_ladders.opt.potts import load_potts_params
from snakes_and_ladders.opt.testfunctions import load_test_function_params
from snakes_and_ladders.sim.canonical import load_frustrated_lattice_params
from snakes_and_ladders.sim.convolutional import load_turbo_params
from snakes_and_ladders.sim.count_pairs import load_spatio_sequential_counts_params
from snakes_and_ladders.sim.emission_mixture import load_emission_mixture_params
from snakes_and_ladders.sim.hmm import load_hmm_params
from snakes_and_ladders.sim.ldpc import load_ldpc_params
from snakes_and_ladders.sim.mixture import load_mixture_params
from snakes_and_ladders.sim.params import load_simulation_params
from snakes_and_ladders.sim.potts import (
    load_potts_lattice_params,
    load_spatio_only_params,
)
from snakes_and_ladders.sim.spatio_sequential import load_spatio_sequential_params

#: The repository root, from this file rather than a working directory: the
#: fixture directory is named from it, and a caller in another tree passes its
#: own.
REPO_ROOT = Path(__file__).resolve().parents[3]

#: Where the fixture files live, as a path from this file rather than from a
#: working directory: a figure is rendered from the repository root and a
#: test from wherever pytest was started.
FIXTURES_DIR = REPO_ROOT / "tests" / "regression" / "fixtures"

#: Which loader reads each declared model. The model is stated in the file
#: rather than derived from the directory, so two problems may share a model
#: --- as the tree fixtures do --- without either naming the other.
LOADERS: dict[str, Callable[[Path], Any]] = {
    "jukes-cantor": load_simulation_params,
    "potts-chain": load_potts_params,
    "potts-lattice": load_potts_lattice_params,
    "spatio-only": load_spatio_only_params,
    "hidden-markov": load_hmm_params,
    "gaussian-mixture": load_mixture_params,
    "emission-mixture": load_emission_mixture_params,
    "spatio-sequential": load_spatio_sequential_params,
    "spatio-sequential-counts": load_spatio_sequential_counts_params,
    "ldpc": load_ldpc_params,
    "frustrated-lattice": load_frustrated_lattice_params,
    "test-functions": load_test_function_params,
    "turbo": load_turbo_params,
}

#: The oracles a fixture may state: an exhaustive sum over the instance, the
#: exact normalizer of a chain, a formula or published value, or nothing
#: independent at this size. ``none`` is a statement and not an omission: the
#: size is past every oracle, and a result there is refereed by a symmetry or
#: by the simulated truth.
ORACLES = ("enumeration", "transfer-matrix", "closed-form", "none")

_REQUIRED_FIELDS = frozenset({"model", "oracle"})

#: The alias :func:`fixture` accepts beside a tier. Not a fourth size: it names
#: *which declared instance a study defaults to*, and the file says which by
#: marking one of its instances ``key`` (issue #399, fifth amendment). A
#: problem whose instances all fit the per-pull-request budget needs none; one
#: whose largest useful instance takes two minutes needs a name for it.
KEY = "key"


def _declares_key(raw: Mapping[str, Any]) -> bool:
    """Whether a fixture file marks one of its instances the key instance."""
    declared = raw.get("bin")
    if not isinstance(declared, list):
        return False
    return any(
        isinstance(entry, Mapping) and entry.get("marker") == KEY for entry in declared
    )


@dataclass(frozen=True)
class Fixture:
    """One loaded fixture: the instance, and what is known about it.

    Parameters
    ----------
    problem : str
        The directory name, the problem the registry knows the instance as.
        `PROBLEMS.md` names it against a catalogue row.
    tier : Scale
        The size tier, which is the file name: the time budget the size was
        chosen for, per ``DEV.md``.
    model : str
        The declared model, and so the loader that read the file.
    oracle : str
        One of :data:`ORACLES`, the independent answer available at this size.
    path : Path
        The file, reported in errors and by any caption rendered from it.
    params : Any
        The loaded instance, of whatever type the model's loader returns.
    """

    problem: str
    tier: Scale
    model: str
    oracle: str
    path: Path
    params: Any


def _key_path(problem: str, directory: Path) -> Path:
    """The file declaring ``problem``'s key instance.

    Returns
    -------
    Path

    Raises
    ------
    FileNotFoundError
        If no file of the problem marks an instance ``key``. Refused rather than
        answered with the largest tier: "the instance a study defaults to" is a
        claim the fixture makes, not one the registry may make for it.
    """
    found = [
        path
        for path in sorted((directory / problem).glob("*.yaml"))
        if _declares_key(yaml.safe_load(path.read_text()))
    ]
    if len(found) == 1:
        return found[0]
    if not found:
        msg = (
            f"no key instance for {problem!r} in {directory}; a fixture file "
            f"marks one by giving a bin entry marker: {KEY}"
        )
        raise FileNotFoundError(msg)
    msg = (
        f"{problem!r} marks a key instance in {[path.name for path in found]}; "
        f"one problem has one default instance"
    )
    raise FileNotFoundError(msg)


def path_of(problem: str, tier: str | Scale, directory: Path = FIXTURES_DIR) -> Path:
    """The file a problem's tier is declared in.

    Parameters
    ----------
    problem : str
        Directory name under ``directory``.
    tier : str | Scale
        ``ci``, ``stress``, ``release``, or :data:`KEY`.
    directory : Path
        Where the fixtures live.

    Returns
    -------
    Path
        The file.

    Raises
    ------
    FileNotFoundError
        If no such fixture exists, or --- asked for :data:`KEY` --- if the
        problem marks no key instance or marks one in more than one file. The
        error lists what the problem does carry, because the usual cause is
        asking for a tier a problem has no instance at rather than a
        mis-spelled problem.
    """
    if tier == KEY:
        return _key_path(problem, directory)
    path = directory / problem / f"{Scale(tier)}.yaml"
    if not path.is_file():
        available = sorted(p.stem for p in (directory / problem).glob("*.yaml"))
        msg = (
            f"no {tier} fixture for {problem!r} in {directory}; "
            f"{problem!r} carries {available}"
        )
        raise FileNotFoundError(msg)
    return path


def fixture(problem: str, tier: str | Scale, directory: Path = FIXTURES_DIR) -> Fixture:
    """Load the instance a problem declares at one tier.

    Parameters
    ----------
    problem : str
        Directory name under ``directory``.
    tier : str | Scale
        ``ci``, ``stress``, ``release``, or :data:`KEY`, which resolves to
        whichever file declares the key instance.
    directory : Path
        Where the fixtures live.

    Returns
    -------
    Fixture
        The loaded instance, its model and its oracle. Asked for by
        :data:`KEY`, its ``tier`` is the file's own: ``key`` says which
        instance a study defaults to, never which budget the file was sized
        for.

    Raises
    ------
    ValueError
        If the file declares a model no loader reads, or an oracle outside
        :data:`ORACLES`. Both are refused here rather than by the loader,
        because a fixture nothing can read and a fixture whose oracle nothing
        knows are registry errors and not model errors.
    """
    path = path_of(problem, tier, directory)
    raw = yaml.safe_load(path.read_text())
    resolved = Scale(path.stem) if tier == KEY else Scale(tier)
    missing = _REQUIRED_FIELDS - raw.keys()
    if missing:
        msg = f"{path}: missing required field(s) {sorted(missing)}"
        raise ValueError(msg)
    model = str(raw["model"])
    if model not in LOADERS:
        msg = f"{path}: model {model!r} has no loader; known: {sorted(LOADERS)}"
        raise ValueError(msg)
    oracle = str(raw["oracle"])
    if oracle not in ORACLES:
        msg = f"{path}: oracle {oracle!r} is not one of {list(ORACLES)}"
        raise ValueError(msg)
    return Fixture(
        problem=problem,
        tier=resolved,
        model=model,
        oracle=oracle,
        path=path,
        params=LOADERS[model](path),
    )


def problems(directory: Path = FIXTURES_DIR) -> tuple[str, ...]:
    """Every problem the registry carries, by name.

    Returns
    -------
    tuple[str, ...]
        Sorted, so a parametrized test runs in the same order everywhere.
    """
    return tuple(
        sorted(
            child.name
            for child in directory.iterdir()
            if child.is_dir() and any(child.glob("*.yaml"))
        )
    )


def tiers(problem: str, directory: Path = FIXTURES_DIR) -> tuple[Scale, ...]:
    """The tiers one problem declares an instance at, smallest first.

    Returns
    -------
    tuple[Scale, ...]
    """
    present = {path.stem for path in (directory / problem).glob("*.yaml")}
    return tuple(scale for scale in Scale if scale in present)


def fixtures(tier: str | Scale, directory: Path = FIXTURES_DIR) -> tuple[Fixture, ...]:
    """Every problem's instance at one tier, in problem order.

    The iteration a parametrized test runs over: a method that applies to a
    class of problems is applied to each of them at the tier the suite is
    running under, rather than to whichever one its author had at hand.

    Parameters
    ----------
    tier : str | Scale
        ``ci``, ``stress`` or ``release``.
    directory : Path
        Where the fixtures live.

    Returns
    -------
    tuple[Fixture, ...]
        The loaded instances of the problems declaring that tier; a problem
        that declares none is absent rather than an error.
    """
    return tuple(
        fixture(problem, tier, directory)
        for problem in problems(directory)
        if Scale(tier) in tiers(problem, directory)
    )


def declared(
    problem: str, tier: str | Scale, directory: Path = FIXTURES_DIR
) -> Mapping[str, Any]:
    """The fixture file's fields as written, for a caller reading one field.

    Used where a caller needs a size or a seed the model's own type does not
    carry --- a scan's extent, a study's replicate count --- rather than the
    loaded instance.

    Returns
    -------
    Mapping[str, Any]
        The parsed mapping, unmodified.
    """
    loaded = yaml.safe_load(path_of(problem, tier, directory).read_text())
    return dict(loaded)


# --- baseline records (issue #401) ------------------------------------------

#: The libraries a baseline's numbers are a function of, and the only list
#: :func:`snakes_and_ladders.inputs.library_versions` is called with: a
#: search rate is a number, not a rendering, so the drawing libraries do not
#: change it and a matplotlib release must not invalidate every record. That
#: wider list was the *figures'*, and it went with them (issue #490).
BASELINE_LIBRARIES: tuple[str, ...] = ("numpy", "scipy", "torch")

#: What a baseline file is called, beside ``<tier>.yaml``.
BASELINE_SUFFIX = ".baseline.json"


class StaleBaselineError(RuntimeError):
    """Raised when a record's numbers are not the ones this tree produces.

    Two callers raise it. :func:`baseline` raises on the cheap read, when the
    installed ``numpy``, ``scipy`` or ``torch`` is not the one the record was
    computed against --- a difference no recomputation elsewhere can see,
    because it is a fact about this machine. ``infra/baselines.py`` raises it
    on the recomputation itself, when a recomputed value or a recorded budget
    disagrees with what is committed. Regenerate with
    ``infra/baselines.py --write``, which recomputes rather than restamps.
    """


@dataclass(frozen=True)
class Measurement:
    """One number a reference algorithm achieved, with what produced it.

    Parameters
    ----------
    algorithm : str
        What computed the value, in words: the reference the number is the
        result of, not the code path.
    value : float | tuple[float, ...]
        The number, or the vector of them where the reference produces one
        per instance --- the maximized log-likelihood of every topology of
        an alignment, say.
    seed : int | None
        The seed the value depends on, or None where it depends on none.
        A rate over seeded restarts has one; an enumerated maximum has not.
    budget : Mapping[str, Any]
        Every other quantity the value depends on: the restart count, the
        horizon, the sizes the reference was run at. Named here rather than
        left in the computing code, because a rate at a different budget is
        a different number under the same name.
    """

    algorithm: str
    value: float | tuple[float, ...]
    seed: int | None
    budget: Mapping[str, Any]


@dataclass(frozen=True)
class Baseline:
    """One fixture's baseline record: the numbers, and what they were computed from.

    Parameters
    ----------
    problem, tier : str, Scale
        The instance the record belongs to.
    path : Path
        The record file, reported in every error.
    modules : tuple[str, ...]
        The dotted names of the modules that computed the values. They are
        what ``infra/baselines.py`` selects on: a pull request touching one
        module's import closure recomputes the records that closure reaches,
        and no others.
    libraries : tuple[str, ...]
        ``name==version`` at the computation, for :data:`BASELINE_LIBRARIES`.
        :func:`baseline` compares them with the versions installed *now*,
        which is what makes a library change visible on the cheap read.
    measurements : Mapping[str, Measurement]
        The numbers, by name.
    """

    problem: str
    tier: Scale
    path: Path
    modules: tuple[str, ...]
    libraries: tuple[str, ...]
    measurements: Mapping[str, Measurement]

    def measurement(self, name: str) -> Measurement:
        """The measurement named ``name``.

        Returns
        -------
        Measurement

        Raises
        ------
        KeyError
            If the record carries no such measurement. The message lists
            what it does carry, because the usual cause is a name that moved
            with the code rather than a record that never held it.
        """
        if name not in self.measurements:
            msg = (
                f"{self.path}: no measurement {name!r}; "
                f"carries {sorted(self.measurements)}"
            )
            raise KeyError(msg)
        return self.measurements[name]

    def value(self, name: str) -> float:
        """The scalar named ``name``.

        Returns
        -------
        float

        Raises
        ------
        TypeError
            If the measurement is a vector, which a caller asking for a
            scalar has mistaken for one.
        """
        found = self.measurement(name).value
        if not isinstance(found, float):
            msg = f"{self.path}: {name!r} is a vector of {len(found)}; use values()"
            raise TypeError(msg)
        return found

    def values(self, name: str) -> tuple[float, ...]:
        """The vector named ``name``.

        Returns
        -------
        tuple[float, ...]

        Raises
        ------
        TypeError
            If the measurement is a scalar.
        """
        found = self.measurement(name).value
        if isinstance(found, float):
            msg = f"{self.path}: {name!r} is a scalar; use value()"
            raise TypeError(msg)
        return found


def baseline_path(
    problem: str, tier: str | Scale, directory: Path = FIXTURES_DIR
) -> Path:
    """Where a problem's tier records its baseline.

    Returns
    -------
    Path
        ``<directory>/<problem>/<tier>.baseline.json``, whether or not it
        exists --- the writer needs the name of a file it is about to make.
    """
    return directory / problem / f"{Scale(tier)}{BASELINE_SUFFIX}"


def _as_measurement(name: str, raw: Mapping[str, Any], path: Path) -> Measurement:
    """One record entry, with a list value read as a vector.

    Raises
    ------
    ValueError
        If the entry is missing a field, which a hand-edited record is the
        usual cause of.
    """
    missing = {"algorithm", "value", "seed", "budget"} - raw.keys()
    if missing:
        msg = f"{path}: measurement {name!r} is missing {sorted(missing)}"
        raise ValueError(msg)
    value = raw["value"]
    return Measurement(
        algorithm=str(raw["algorithm"]),
        value=tuple(float(item) for item in value)
        if isinstance(value, list)
        else float(value),
        seed=None if raw["seed"] is None else int(raw["seed"]),
        budget=dict(raw["budget"]),
    )


def read_baseline(path: Path) -> Baseline:
    """Parse a baseline record without checking it against this machine.

    The unchecked read, for the writer that is about to replace the file and
    for the recomputation that is about to compare with it. Callers wanting a
    number want :func:`baseline`.

    Returns
    -------
    Baseline

    Raises
    ------
    FileNotFoundError
        If there is no record at ``path``.
    ValueError
        If the file is missing a top-level field.
    """
    if not path.is_file():
        msg = f"no baseline record at {path}; write one with infra/baselines.py --write"
        raise FileNotFoundError(msg)
    raw = json.loads(path.read_text())
    missing = {
        "problem",
        "tier",
        "modules",
        "libraries",
        "measurements",
    } - raw.keys()
    if missing:
        msg = f"{path}: missing required field(s) {sorted(missing)}"
        raise ValueError(msg)
    return Baseline(
        problem=str(raw["problem"]),
        tier=Scale(raw["tier"]),
        path=path,
        modules=tuple(str(name) for name in raw["modules"]),
        libraries=tuple(str(name) for name in raw["libraries"]),
        measurements={
            name: _as_measurement(name, entry, path)
            for name, entry in raw["measurements"].items()
        },
    )


def write_baseline(record: Baseline) -> None:
    """Write ``record`` to its own path, as the JSON :func:`read_baseline` reads.

    Sorted keys and a trailing newline, so a regeneration that changed no
    number produces no diff.
    """
    payload = {
        "problem": record.problem,
        "tier": str(record.tier),
        "modules": list(record.modules),
        "libraries": list(record.libraries),
        "measurements": {
            name: {
                "algorithm": measurement.algorithm,
                "value": list(measurement.value)
                if isinstance(measurement.value, tuple)
                else measurement.value,
                "seed": measurement.seed,
                "budget": dict(measurement.budget),
            }
            for name, measurement in sorted(record.measurements.items())
        },
    }
    record.path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def baseline(
    problem: str,
    tier: str | Scale,
    directory: Path = FIXTURES_DIR,
) -> Baseline:
    """The baseline record for a problem's tier, refused if it has gone stale.

    Parameters
    ----------
    problem : str
        Directory name under ``directory``.
    tier : str | Scale
        ``ci``, ``stress`` or ``release``.
    directory : Path
        Where the fixtures live.

    Returns
    -------
    Baseline
        The record, computed against the libraries installed here.

    Raises
    ------
    FileNotFoundError
        If the problem declares no record at that tier.
    StaleBaselineError
        If ``numpy``, ``scipy`` or ``torch`` is not the version the record
        was computed against. That is the one input to the numbers this read
        can check for nothing, and it is the one a recomputation on another
        host cannot check at all: a test reading a number produced under a
        different linear algebra would assert against history. Whether the
        *code* still produces the numbers is refereed by recomputing them ---
        ``infra/baselines.py``, per pull request over the records the change
        could have moved and over every record at the release gate (issue
        #460).
    """
    record = read_baseline(baseline_path(problem, tier, directory))
    installed = tuple(library_versions(BASELINE_LIBRARIES))
    if installed != record.libraries:
        msg = (
            f"{record.path} was computed against {list(record.libraries)}, "
            f"and {list(installed)} is installed. Recompute with: "
            f"uv run python infra/baselines.py --write"
        )
        raise StaleBaselineError(msg)
    return record


def baselines(directory: Path = FIXTURES_DIR) -> tuple[tuple[str, Scale], ...]:
    """Every ``(problem, tier)`` that carries a record, in problem order.

    Returns
    -------
    tuple[tuple[str, Scale], ...]
    """
    found = [
        (path.parent.name, Scale(path.name[: -len(BASELINE_SUFFIX)]))
        for path in sorted(directory.rglob(f"*{BASELINE_SUFFIX}"))
    ]
    return tuple(found)
