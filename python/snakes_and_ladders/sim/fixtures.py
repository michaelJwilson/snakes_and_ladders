"""The registry of supported problem instances (issue #382).

`PROBLEMS.md` names the problems this repository supports; this module is
where an *instance* of one is obtained. A fixture is a file under
``tests/regression/fixtures/<problem>/<tier>.yaml`` stating a size, a seed,
the model that reads it and the oracle that referees a result at that size,
and :func:`fixture` returns it loaded through that model's own loader.

**A supported instance is a fixture, never a literal.** An instance typed
into a test, a notebook or a figure script is one nothing else can find: the
next caller writes its own, the two drift, and a claim about "the problem"
is a claim about whichever copy the reader happened to open. Naming the
instance in one file makes "every method is applied to every problem" a
question the tools can answer rather than one a reviewer has to.

**The files live under ``tests/`` and this module lives in the package**
because the consumers are not only the suite: ``snakes_and_ladders.qa``
renders figures from fixtures and the notebooks run on them, and neither may
import from ``tests/``. The directory is a constant here and a parameter of
every function, so a caller working from a different tree passes its own.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from snakes_and_ladders.fixtures import Scale
from snakes_and_ladders.opt.potts import load_potts_params
from snakes_and_ladders.opt.testfunctions import load_test_function_params
from snakes_and_ladders.sim.canonical import load_frustrated_lattice_params
from snakes_and_ladders.sim.emission_mixture import load_emission_mixture_params
from snakes_and_ladders.sim.hmm import load_hmm_params
from snakes_and_ladders.sim.ldpc import load_ldpc_params
from snakes_and_ladders.sim.mixture import load_mixture_params
from snakes_and_ladders.sim.params import load_simulation_params
from snakes_and_ladders.sim.potts import load_potts_lattice_params
from snakes_and_ladders.sim.spatio_sequential import load_spatio_sequential_params

#: Where the fixture files live, as a path from this file rather than from a
#: working directory: a figure is rendered from the repository root and a
#: test from wherever pytest was started.
FIXTURES_DIR = Path(__file__).resolve().parents[3] / "tests" / "regression" / "fixtures"

#: Which loader reads each declared model. The model is stated in the file
#: rather than derived from the directory, so two problems may share a model
#: --- as the tree fixtures do --- without either naming the other.
LOADERS: dict[str, Callable[[Path], Any]] = {
    "jukes-cantor": load_simulation_params,
    "potts-chain": load_potts_params,
    "potts-lattice": load_potts_lattice_params,
    "hidden-markov": load_hmm_params,
    "gaussian-mixture": load_mixture_params,
    "emission-mixture": load_emission_mixture_params,
    "spatio-sequential": load_spatio_sequential_params,
    "ldpc": load_ldpc_params,
    "frustrated-lattice": load_frustrated_lattice_params,
    "test-functions": load_test_function_params,
}

#: The oracles a fixture may state: an exhaustive sum over the instance, the
#: exact normalizer of a chain, a formula or published value, or nothing
#: independent at this size. ``none`` is a statement, not an omission --- it
#: says the size is past every oracle, and a result there is refereed by a
#: symmetry or by the simulated truth.
ORACLES = ("enumeration", "transfer-matrix", "closed-form", "none")

_REQUIRED_FIELDS = frozenset({"model", "oracle"})


@dataclass(frozen=True)
class Fixture:
    """One loaded fixture: the instance, and what is known about it.

    Parameters
    ----------
    problem : str
        The directory name, which is the problem the registry knows the
        instance as. `PROBLEMS.md` names it against a catalogue row.
    tier : Scale
        The size tier, which is the file name: which time budget the size was
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


def path_of(problem: str, tier: str | Scale, directory: Path = FIXTURES_DIR) -> Path:
    """The file a problem's tier is declared in.

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
    Path
        The file.

    Raises
    ------
    FileNotFoundError
        If no such fixture exists. The error lists what the problem does
        carry, because the usual cause is asking for a tier a problem has no
        instance at rather than a mis-spelled problem.
    """
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
        ``ci``, ``stress`` or ``release``.
    directory : Path
        Where the fixtures live.

    Returns
    -------
    Fixture
        The loaded instance, its model and its oracle.

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
        tier=Scale(tier),
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
