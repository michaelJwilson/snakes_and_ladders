#!/usr/bin/env python
"""The compiled kernels, what pins each one, and which take the thread pool.

Issue #678. The repository has nine Rust modules and three of them use
``rayon``; which three was a fact a reader had to grep for, and *what referees
each kernel* was a fact nobody had written down at all. The last Rust audit
carried a hand-written list, which is the shape root ``CLAUDE.md`` and issue
#586 both refuse: a list re-counts what is known and discovers nothing.

This derives three relations instead, each from the tree:

**Parallel or serial** --- from the ``rayon`` markers in the module's own
source, not from a list of ports.

**The boundary** --- every ``#[pyfunction]`` a module exports, and the Python
adapters that name it. A kernel nothing calls is reachable only from a test,
which is a different claim from being in use.

**The oracle** --- read from the *tests*, not from a naming convention. The
convention is real (``pruning_rust.py`` beside ``pruning.py``) and it does not
hold everywhere: the ragged kernel is pinned against a conserved route under
``sandbox``, which no sibling rule would find. So a kernel's referees are the
implementations its own test modules import beside it, narrowed to the
vocabulary this repository actually referees with --- the adapter's
pure-Python sibling, a brute-force or enumerating module, a conserved route
under ``sandbox``, and an oracle **inside the adapter itself** --- because a
test imports thirty modules and three of them are the referee. That last
placement is in this list because the first draft of this tool reported the
ragged kernel as unpinned and the tree said otherwise:
``test_ragged_rust.py`` imports ``posteriors`` and ``posteriors_oracle`` from
one module, so the referee is a symbol rather than a module and a tool that
only looks at modules calls a pinned kernel bare (issue #678). A kernel whose
tests import none of the four reads as ``NONE``, which is the gap
``likelihood/CLAUDE.md`` names.

What it does **not** derive is a profile. A ratio is half a claim and the
effect size is the other half (root ``CLAUDE.md``), and neither can be read
from source: the ranking lives in ``STATUS.md`` beside the measurement that
produced it. This reports the structure a profile is then spent on.

Run with ``uv run python infra/appraise_kernels.py``; ``--json`` writes the
inventory for a test to read rather than re-deriving it.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from _paths import REPO_ROOT

CRATE_ROOT = REPO_ROOT / "src"
PACKAGE_ROOT = REPO_ROOT / "python" / "sal"
TESTS_ROOT = REPO_ROOT / "tests"

#: The crate the bindings are imported from, as every adapter spells it.
EXTENSION = "oxisal"

#: What a referee is called here. Not a guess at semantics: these are the three
#: forms the oracle rule takes in this tree --- the implementation an adapter
#: fronts, an exhaustive one, and a route conserved because it still answers a
#: case (`sandbox/CLAUDE.md`).
_ORACLE_MARKS = ("oracle", "brute_force", "enumerate", "reference", "exact")
_CONSERVED = "sandbox"

#: Modules that cross the boundary and are not kernels, with the reason. The
#: unpinned list is for gaps, and `lib`'s `double` is the build probe ---
#: `double(21) == 42` in `tests/test_oxisal_bindings.py`
#: asserts that the extension imports at all, and arithmetic is not an oracle
#: a reference implementation could referee.
PROBES: dict[str, str] = {
    "lib": (
        "the extension-loads probe: `double(21) == 42` is checked by "
        "arithmetic in tests/test_oxisal_bindings.py, which "
        "is what it is for"
    ),
}

#: What taking the thread pool looks like in this crate. `rayon` in a `use`
#: line is not enough on its own --- a module could import it and iterate
#: serially --- so the parallel iterators are what decide it, and the import
#: is reported beside them so a mismatch is visible rather than silent.
_PARALLEL_MARKERS = ("par_iter", "par_chunks", "into_par", "par_bridge", "par_extend")


@dataclass(frozen=True)
class Kernel:
    """One Rust module, and everything the tree says about it."""

    module: str
    lines: int
    parallel: bool
    markers: tuple[str, ...]
    entry_points: tuple[str, ...]
    adapters: tuple[str, ...]
    tests: tuple[str, ...]
    referees: tuple[str, ...]

    @property
    def pinned(self) -> bool:
        """Whether a test imports a second implementation beside this one."""
        return bool(self.referees)


def _rust_modules(root: Path = CRATE_ROOT) -> dict[str, str]:
    return {path.stem: path.read_text() for path in sorted(root.glob("*.rs"))}


def _entry_points(source: str) -> tuple[str, ...]:
    """Every ``#[pyfunction]`` the module exports, in declaration order."""
    names: list[str] = []
    lines = source.splitlines()
    for index, line in enumerate(lines):
        if "#[pyfunction]" not in line:
            continue
        for follower in lines[index + 1 : index + 6]:
            match = re.match(r"\s*(?:pub\s+)?fn\s+(\w+)", follower)
            if match:
                names.append(match.group(1))
                break
    return tuple(names)


def _python_modules(root: Path = PACKAGE_ROOT) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*.py")):
        parts = list(path.relative_to(root).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        out[".".join(parts) or "__init__"] = path.read_text()
    return out


def _test_modules(root: Path = TESTS_ROOT) -> dict[str, str]:
    return {
        str(path.relative_to(REPO_ROOT)): path.read_text()
        for path in sorted(root.rglob("test_*.py"))
    }


def _package_imports(source: str) -> set[str]:
    """Every ``sal`` module a source file imports, dotted."""
    found: set[str] = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:  # pragma: no cover - a parse failure is a lint failure
        return found
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("sal"):
            module = (node.module or "").removeprefix("sal").lstrip(".")
            found.add(module)
            found.update(f"{module}.{alias.name}".lstrip(".") for alias in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("sal"):
                    found.add(alias.name.removeprefix("sal").lstrip("."))
    return found


def _reexports(python: dict[str, str]) -> dict[str, set[str]]:
    """``package -> submodules`` whose names the package's ``__init__`` re-exports.

    A test importing ``baum_welch_family`` from ``opt.hmm`` runs the adapter in
    ``opt.hmm.estimation`` the package re-exports it from (issue #1010), so an
    import of the package is an import of each submodule it imports from.
    """
    return {
        name: {
            module
            for module in _package_imports(source)
            if module.startswith(f"{name}.") and module in python
        }
        for name, source in python.items()
    }


def kernels() -> list[Kernel]:
    """Every Rust module, with its boundary, its callers and its referees."""
    python = _python_modules()
    tests = _test_modules()
    reexports = _reexports(python)

    def imported(source: str) -> set[str]:
        names = _package_imports(source)
        return names | {sub for name in names for sub in reexports.get(name, ())}

    adapters_by_module = {
        name: source for name, source in python.items() if EXTENSION in source
    }
    out: list[Kernel] = []
    for module, source in _rust_modules().items():
        entries = _entry_points(source)
        markers = tuple(marker for marker in _PARALLEL_MARKERS if marker in source) + (
            ("use rayon",) if "use rayon" in source else ()
        )
        adapters = tuple(
            sorted(
                name
                for name, text in adapters_by_module.items()
                if any(re.search(rf"\b{entry}\b", text) for entry in entries)
            )
        )
        exercising = tuple(
            sorted(
                path
                for path, text in tests.items()
                if any(adapter in imported(text) for adapter in adapters)
            )
        )
        # The referees: what those tests import besides the adapter, narrowed
        # to modules that exist and to the oracle vocabulary above. A kernel
        # pinned only against itself reads as an empty tuple rather than as
        # silence, which is the gap `likelihood/CLAUDE.md` names.
        siblings = {adapter.removesuffix("_rust") for adapter in adapters}
        referees: set[str] = set()
        for path in exercising:
            for name in imported(tests[path]):
                symbol = name.rsplit(".", 1)[-1]
                # An oracle beside the kernel in the adapter's own module: the
                # name is `<adapter>.<symbol>`, so it is not a module and the
                # module filter below would drop it.
                if any(name.startswith(f"{adapter}.") for adapter in adapters) and any(
                    mark in symbol for mark in _ORACLE_MARKS
                ):
                    referees.add(name)
                    continue
                if name not in python or name.endswith("_rust"):
                    continue
                if (
                    name in siblings
                    or name.startswith(f"{_CONSERVED}.")
                    or any(mark in name for mark in _ORACLE_MARKS)
                ):
                    referees.add(name)
        out.append(
            Kernel(
                module=module,
                lines=len(source.splitlines()),
                parallel=any(marker in source for marker in _PARALLEL_MARKERS),
                markers=markers,
                entry_points=entries,
                adapters=adapters,
                tests=exercising,
                referees=tuple(sorted(referees)),
            )
        )
    return out


def render(found: list[Kernel]) -> str:
    parallel = [k for k in found if k.parallel]
    lines = [
        f"{len(found)} Rust modules, {sum(k.lines for k in found)} lines; "
        f"{len(parallel)} take the thread pool",
        "",
        f"{'module':16s} {'lines':>6s} {'pool':>6s}  entry points",
    ]
    for kernel in found:
        lines.append(
            f"{kernel.module:16s} {kernel.lines:6d} "
            f"{'rayon' if kernel.parallel else 'serial':>6s}  "
            f"{', '.join(kernel.entry_points) or '--'}"
        )
    lines.append("")
    for kernel in found:
        if not kernel.entry_points:
            continue
        lines.append(f"## {kernel.module}")
        lines.append(f"  adapters  {', '.join(kernel.adapters) or '-- none'}")
        lines.append(f"  tests     {len(kernel.tests)}")
        lines.append(f"  referees  {', '.join(kernel.referees) or '-- NONE'}")
        lines.append("")
    for module, reason in sorted(PROBES.items()):
        lines.append(f"  {module}: not a kernel --- {reason}")
    unpinned = [
        k.module
        for k in found
        if k.entry_points and not k.pinned and k.module not in PROBES
    ]
    if unpinned:
        lines.append(f"! crossed the boundary with no referee in its tests: {unpinned}")
    else:
        lines.append("  every kernel is pinned against a referee its own tests import")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="write the inventory here")
    args = parser.parse_args(argv)

    found = kernels()
    if args.json:
        args.json.write_text(json.dumps([asdict(k) for k in found], indent=1))
        print(f"wrote {args.json}")
        return 0
    print(render(found))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
