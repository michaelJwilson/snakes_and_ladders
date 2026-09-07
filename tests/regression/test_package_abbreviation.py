"""``import snakes_and_ladders as sal`` is the stated form, and it resolves (issue #301).

Two properties, both structural. The top-level import brings in no
subpackage -- and therefore no ``torch`` -- until one is touched, which is
what keeps ``import snakes_and_ladders`` as cheap as it was; and every
subpackage's modules resolve by attribute after that one import, which is
what makes ``sal.likelihood.pruning`` a name and not a typo. The prose rule
is checked by the script that applies it: no Markdown, LaTeX, notebook cell
or docstring literal still spells the long form, and every Sphinx role does.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "infra" / "abbreviate_package.py"


@pytest.mark.structural
def test_the_top_level_import_loads_no_subpackage() -> None:
    # In a fresh interpreter, so nothing this process has already imported
    # can make the check pass by accident.
    probe = (
        "import sys, snakes_and_ladders as sal; "
        "print(sorted(m for m in sys.modules if m.startswith('snakes_and_ladders.')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    loaded = eval(result.stdout)
    assert loaded == [
        "snakes_and_ladders._lazy",
        "snakes_and_ladders.oxi_snakes_and_ladders",
    ]


@pytest.mark.structural
@pytest.mark.parametrize(
    "dotted",
    [
        "sim.graph",
        "likelihood.pruning",
        "opt.fit",
        "search.infer",
        "learn.environment",
        "qa.manifest",
        "emissions",
    ],
)
def test_every_module_resolves_by_attribute_after_one_import(dotted: str) -> None:
    sal = importlib.import_module("snakes_and_ladders")

    reached = sal
    for part in dotted.split("."):
        reached = getattr(reached, part)

    assert reached is importlib.import_module(f"snakes_and_ladders.{dotted}")


@pytest.mark.edge_case
def test_an_unknown_attribute_is_refused_by_name() -> None:
    sal = importlib.import_module("snakes_and_ladders")

    with pytest.raises(
        AttributeError, match="'snakes_and_ladders' has no attribute 'nonesuch'"
    ):
        getattr(sal, "nonesuch")  # noqa: B009 - the dynamic path is the one under test
    with pytest.raises(
        AttributeError, match="'snakes_and_ladders.sim' has no attribute"
    ):
        getattr(sal.sim, "nonesuch")  # noqa: B009


@pytest.mark.structural
def test_no_prose_still_spells_the_long_form_and_the_roles_keep_it() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    # The roles must keep the full name, or Sphinx cannot resolve them.
    roles = subprocess.run(
        [
            "git",
            "grep",
            "-c",
            r":mod:`snakes_and_ladders\.\|automodule:: snakes_and_ladders",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert roles.stdout.strip(), "no Sphinx role names the package in full"
