"""Every issue template is one an author can file from, and one the label
taxonomy admits (issue #381).

Tickets of one shape had been restating the same fields — the heads of a
mis-merged pull request, the profile behind an optimization, the survey rows
behind a seam — so each shape has a template that asks for them. The guard
reads the directory rather than a list, so a template added without these
properties fails here and not at filing time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = REPO_ROOT / ".github" / "ISSUE_TEMPLATE"
LABELS = REPO_ROOT / ".github" / "labels.yml"

#: The shapes `DEV.md` names, each its own form.
EXPECTED = {
    "task.yml",
    "release.yml",
    "documents.yml",
    "mis_merge.yml",
    "optimization.yml",
    "seam.yml",
    "coverage.yml",
    "markers.yml",
}


def _templates() -> list[str]:
    return sorted(p.name for p in TEMPLATES.glob("*.yml") if p.name != "config.yml")


def _load(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load((TEMPLATES / name).read_text())
    return loaded


@pytest.mark.infra
def test_the_shapes_dev_md_names_each_have_a_template() -> None:
    assert set(_templates()) == EXPECTED


@pytest.mark.infra
@pytest.mark.parametrize("name", _templates())
def test_a_template_names_itself_and_asks_for_something(name: str) -> None:
    form = _load(name)
    assert form["name"], f"{name} has no name"
    assert form["description"], f"{name} has no description"
    fields = [b for b in form["body"] if b["type"] != "markdown"]
    assert fields, f"{name} asks for nothing"
    required = [b for b in fields if b.get("validations", {}).get("required")]
    assert required, f"{name} can be filed empty: no field is required"
    ids = [b["id"] for b in fields]
    assert len(ids) == len(set(ids)), f"{name} repeats a field id: {ids}"


@pytest.mark.infra
@pytest.mark.parametrize("name", _templates())
def test_every_label_a_template_applies_is_one_the_taxonomy_defines(name: str) -> None:
    defined = {entry["name"] for entry in yaml.safe_load(LABELS.read_text())}
    applied = set(_load(name).get("labels", []))
    assert applied <= defined, (
        f"{name} applies {applied - defined}, absent from labels.yml"
    )


@pytest.mark.infra
def test_blank_issues_stay_disabled() -> None:
    config = yaml.safe_load((TEMPLATES / "config.yml").read_text())
    assert config["blank_issues_enabled"] is False
