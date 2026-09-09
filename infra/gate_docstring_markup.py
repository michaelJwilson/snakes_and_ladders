"""The docstrings the branch changed parse under the documentation build.

`infra/review_gates.sh` calls this. The class it catches is a docstring whose
reStructuredText the `docs` CI job rejects: an undefined role, an undefined
substitution, a malformed table. Two of those reached CI on one day (issue
#451), both invisible to every local check, because the Sphinx build was not
one of the gates.

The full build is not the gate, and the number is why. `docs/source/index.rst`
is a *single* document that autodocuments every module, so changing any one
docstring invalidates it and Sphinx re-reads all of them: 26.5 s incremental
against a 30 s budget for the whole table (`DEV.md`). This builds the same
configuration over the changed modules alone -- 2.8 s for one, 3.3 s for five
-- and is bounded above by 20.7 s, the whole index, which is the answer when
`docs/source/` itself changed and the module list is what moved.

It is Sphinx that parses, not a substitute for it: the real `conf.py`, the real
`automodule` directives, `napoleon` converting the same NumPy docstrings. A
grep for the roles the configuration does not define would have caught half of
#451 and nothing else.

What it does not cover, and CI still does: a module missing from
`docs/source/index.rst`, and a cross-reference from an unchanged docstring into
a changed one. The gate is the changed files, per issue #451's scope.

Usage::

    uv run python infra/gate_docstring_markup.py --base origin/main
"""

from __future__ import annotations

import argparse
import io
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
#: The Sphinx source the `docs` CI job builds; its `conf.py` is what this uses.
SOURCE = REPO_ROOT / "docs" / "source"
#: The package root, put first on `sys.path` so autodoc imports this checkout
#: rather than whatever a shared environment resolves the package to.
PACKAGE_PATH = REPO_ROOT / "python"
_AUTOMODULE = re.compile(r"^\.\. automodule:: (\S+)", re.MULTILINE)


def changed_modules(base: str) -> list[str]:
    """The modules whose docstrings this branch could have broken.

    The changed Python files under ``python/``, and -- when ``docs/source/``
    itself changed -- every module the index names, since the module list and
    the configuration are then what moved.

    Parameters
    ----------
    base : str
        The branch the pull request targets.

    Returns
    -------
    list[str]
        Dotted module names, sorted and deduplicated.
    """
    changed = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    if any(name.startswith("docs/source/") for name in changed):
        return indexed_modules()
    modules: set[str] = set()
    for name in changed:
        path = Path(name)
        if path.suffix != ".py" or not path.is_relative_to("python"):
            continue
        if not (REPO_ROOT / path).is_file():  # deleted by the branch
            continue
        dotted = ".".join(path.relative_to("python").with_suffix("").parts)
        modules.add(dotted.removesuffix(".__init__"))
    return sorted(modules)


def indexed_modules() -> list[str]:
    """Every module ``docs/source/index.rst`` autodocuments.

    Returns
    -------
    list[str]
    """
    return _AUTOMODULE.findall((SOURCE / "index.rst").read_text())


def warnings_for(modules: list[str]) -> str:
    """Build the repository's Sphinx configuration over ``modules`` alone.

    A `dummy` builder: the documents are read, parsed and transformed, which is
    where a markup error is reported, and nothing is written.

    Parameters
    ----------
    modules : list[str]
        Dotted module names to autodocument.

    Returns
    -------
    str
        Everything Sphinx wrote to its warning stream; empty on a clean build.
    """
    if not modules:
        return ""
    from sphinx.application import Sphinx
    from sphinx.util.console import nocolor
    from sphinx.util.docutils import docutils_namespace

    # The stream is read back and printed by a caller that is not a terminal;
    # Sphinx colours it regardless of where it is going.
    nocolor()
    if str(PACKAGE_PATH) not in sys.path:
        sys.path.insert(0, str(PACKAGE_PATH))
    stream = io.StringIO()
    with tempfile.TemporaryDirectory() as name:
        tmp = Path(name)
        shutil.copy(SOURCE / "conf.py", tmp / "conf.py")
        body = "".join(
            f".. automodule:: {module}\n   :members:\n\n" for module in modules
        )
        (tmp / "index.rst").write_text(f"changed modules\n===============\n\n{body}")
        with docutils_namespace():
            app = Sphinx(
                srcdir=str(tmp),
                confdir=str(tmp),
                outdir=str(tmp / "_build"),
                doctreedir=str(tmp / "_doctrees"),
                buildername="dummy",
                status=None,
                warning=stream,
                freshenv=True,
            )
            app.build()
    # The temporary path is an implementation detail; a reader needs the
    # docstring's own location, which Sphinx already names after it.
    return stream.getvalue().replace(f"{name}/", "")


def main(argv: list[str] | None = None) -> int:
    """Report the changed docstrings the documentation build rejects.

    Returns
    -------
    int
        ``0`` when every changed module's docstrings parse without a warning.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="origin/main")
    args = parser.parse_args(argv)
    modules = changed_modules(args.base)
    reported = warnings_for(modules)
    if not reported.strip():
        return 0
    print(f"  {len(modules)} changed modules; sphinx reported:", file=sys.stderr)
    for line in reported.strip().splitlines():
        print(f"  {line}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
