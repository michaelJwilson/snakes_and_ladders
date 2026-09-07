"""The experiment ledger holds to its template, and the index to the files (issue #314).

Structural guards: every file under ``docs/experiments/`` parses, carries
every field and section the template names with the vocabularies it fixes,
and names a ticket for every action; the generated index matches what the
generator would write; and the guard itself catches a file that drifted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "infra"))

import experiments  # noqa: E402

VALID = (
    REPO_ROOT / "docs" / "experiments" / "001-potts-cluster-autocorrelation.md"
).read_text()


@pytest.mark.structural
def test_every_experiment_file_is_valid() -> None:
    found = experiments.experiments()
    assert found, "the ledger has no experiments"
    for experiment in found:
        assert experiments.problems(experiment) == [], experiment.path.name


@pytest.mark.structural
def test_the_index_is_what_the_generator_writes() -> None:
    index = (experiments.EXPERIMENTS_DIR / experiments.INDEX).read_text()
    assert index == experiments.render_index(experiments.experiments())


@pytest.mark.structural
def test_the_template_names_every_section_and_field() -> None:
    template = experiments.load(experiments.EXPERIMENTS_DIR / experiments.TEMPLATE)
    assert set(experiments.SECTIONS) <= set(template.sections)
    assert set(experiments.REQUIRED_FIELDS) <= set(template.fields)


@pytest.mark.edge_case
def test_the_guard_catches_a_file_that_drifted(tmp_path: Path) -> None:
    good = tmp_path / "001-potts-cluster-autocorrelation.md"
    good.write_text(VALID)
    assert experiments.problems(experiments.load(good)) == []
    drifted = tmp_path / "002-drifted.md"
    text = VALID.replace("status: confirmed", "status: done").replace(
        "## What is not claimed", "## Caveats"
    )
    text = text.replace("commit: ", "commit: x")
    drifted.write_text(text)
    found = experiments.problems(experiments.load(drifted))
    assert any("status" in problem for problem in found)
    assert any("What is not claimed" in problem for problem in found)
    assert any("commit" in problem for problem in found)
    assert any("does not match" in problem or "id" in problem for problem in found)
    no_front_matter = tmp_path / "003-bare.md"
    no_front_matter.write_text("# Bare\n\n## Setup\n\nnothing\n")
    with pytest.raises(ValueError, match="front matter"):
        experiments.load(no_front_matter)
    assert experiments.main(["--check", "--directory", str(tmp_path)]) == 1
