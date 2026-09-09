"""The experiment ledger holds to its template, and the index to the files (issue #314).

Structural guards: every file under ``docs/experiments/`` parses, carries
every field and section the template names with the vocabularies it fixes,
names a ticket for every action, and keeps its body inside the cap of ten
content lines (issue #458); the generated index matches what the generator
would write; and the guard itself catches a file that drifted or ran over.
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
#: 001's front matter, reused by the cases below: the cap is a claim about
#: the body, so the record above it is held fixed rather than invented.
FRONT_MATTER = VALID[: VALID.index("\n---\n") + len("\n---\n")]
#: A body of exactly the cap: a question, an eight-line table and a finding.
#: Fourteen non-blank lines, four of them a title and its section headings,
#: which the cap charges for none of; ten of them content.
AT_THE_CAP = """
# A title, which is not one of the ten

## Question

One line.

## Numbers

| metric | a | b |
| --- | --- | --- |
| the instance | 1 | 2 |
| a second | 3 | 4 |
| a third | 5 | 6 |
| a fourth | 7 | 8 |
| a fifth | 9 | 10 |
| a sixth | 11 | 12 |

## Finding

One line, and its action, #458.
"""


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
    assert template.body_lines <= experiments.BODY_LINE_CAP


@pytest.mark.edge_case
def test_a_body_over_ten_lines_fails_and_no_heading_is_one_of_them(
    tmp_path: Path,
) -> None:
    at_the_cap = tmp_path / "001-at-the-cap.md"
    at_the_cap.write_text(FRONT_MATTER + AT_THE_CAP)
    experiment = experiments.load(at_the_cap)
    assert experiment.body_lines == experiments.BODY_LINE_CAP == 10
    non_blank = [line for line in AT_THE_CAP.splitlines() if line.strip()]
    headings = [line for line in non_blank if line.startswith("#")]
    assert (len(non_blank), len(headings)) == (14, 4), "a title and three headings"
    assert experiments.problems(experiment) == []
    over = tmp_path / "001-over-the-cap.md"
    over.write_text(
        FRONT_MATTER
        + AT_THE_CAP.replace(
            "| a sixth | 11 | 12 |", "| a sixth | 11 | 12 |\n| a seventh | 13 | 14 |"
        )
    )
    found = experiments.problems(experiments.load(over))
    assert found == ["body is 11 content lines, over the cap of 10 (issue #458)"]
    assert experiments.main(["--check", "--directory", str(tmp_path)]) == 1


@pytest.mark.edge_case
def test_the_guard_catches_a_file_that_drifted(tmp_path: Path) -> None:
    good = tmp_path / "001-potts-cluster-autocorrelation.md"
    good.write_text(VALID)
    assert experiments.problems(experiments.load(good)) == []
    drifted = tmp_path / "002-drifted.md"
    text = VALID.replace("status: confirmed", "status: done").replace(
        "## Numbers", "## Results"
    )
    text = text.replace("commit: ", "commit: x")
    drifted.write_text(text)
    found = experiments.problems(experiments.load(drifted))
    assert any("status" in problem for problem in found)
    assert any("Numbers" in problem for problem in found)
    assert any("commit" in problem for problem in found)
    assert any("does not match" in problem or "id" in problem for problem in found)
    no_front_matter = tmp_path / "003-bare.md"
    no_front_matter.write_text("# Bare\n\n## Numbers\n\nnothing\n")
    with pytest.raises(ValueError, match="front matter"):
        experiments.load(no_front_matter)
    assert experiments.main(["--check", "--directory", str(tmp_path)]) == 1
