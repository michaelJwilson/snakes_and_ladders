"""The experiment ledger: validate every experiment file, and generate the index (issue #314).

An experiment under ``docs/experiments/`` is a Markdown file with YAML front
matter and fixed sections, written from ``TEMPLATE.md``. This module is the
one reading of that format: :func:`load` parses a file, :func:`problems`
lists what it gets wrong, and :func:`render_index` writes the table
``README.md`` shows -- problem, size, methods, status, the best-known result
-- so the index is generated and never hand-edited. Infrastructure, not
science: it knows the field names and nothing about what a Potts lattice is.

Imported the way ``select_tests.py`` is, by inserting ``infra/`` onto
``sys.path``; run as ``python infra/experiments.py`` to rewrite the index and
``python infra/experiments.py --check`` to fail if a file is invalid or the
committed index is stale.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "docs" / "experiments"
FIXTURES_DIR = REPO_ROOT / "tests" / "regression" / "fixtures"
TEMPLATE = "TEMPLATE.md"
INDEX = "README.md"

REQUIRED_FIELDS = (
    "id",
    "date",
    "commit",
    "branch",
    "pr",
    "tickets",
    "problem",
    "fixture",
    "size",
    "methods",
    "budget",
    "seeds",
    "hardware",
    "status",
)
PROBLEMS = ("potts-lattice", "hmm-path", "coupled", "tree", "mixture")
SIZES = ("ci", "stress", "release")
STATUSES = ("open", "confirmed", "retracted", "superseded")
SECTIONS = (
    "Feature under test",
    "Setup",
    "Results",
    "Figures",
    "Finding",
    "Conclusion and actions",
    "What is not claimed",
)
#: What a ``fixture`` field must name: a fixture file, or the problem
#: directory holding one where the experiment swept sizes around it rather
#: than running the declared instance (issue #382). Prose may follow; the
#: reference is what is checked.
_FIXTURE_REFERENCE = re.compile(
    r"tests/regression/fixtures/[a-z0-9_]+(?:/[a-z]+\.yaml)?"
)
_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
_TICKET = re.compile(r"#\d+")
_COMMIT = re.compile(r"\A[0-9a-f]{40}\Z")


@dataclass(frozen=True)
class Experiment:
    """One experiment file, parsed."""

    path: Path
    fields: dict[str, object]
    title: str
    sections: dict[str, str]


def load(path: Path) -> Experiment:
    """Parse ``path``: front matter, the ``#`` title, and the ``##`` sections."""
    text = path.read_text()
    match = _FRONT_MATTER.match(text)
    if match is None:
        msg = f"{path.name}: no YAML front matter between --- lines at the top"
        raise ValueError(msg)
    fields = yaml.safe_load(match.group(1)) or {}
    if not isinstance(fields, dict):
        msg = f"{path.name}: front matter is not a mapping"
        raise ValueError(msg)
    body = text[match.end() :]
    title_match = re.search(r"^# (.+)$", body, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else ""
    sections: dict[str, str] = {}
    for heading, content in re.findall(
        r"^## (.+?)\n(.*?)(?=^## |\Z)", body, re.MULTILINE | re.DOTALL
    ):
        sections[heading.strip()] = content.strip()
    return Experiment(path, {str(k): v for k, v in fields.items()}, title, sections)


def problems(experiment: Experiment) -> list[str]:
    """Everything the file gets wrong, as one line each; empty when it is valid."""
    found: list[str] = []
    fields = experiment.fields
    for name in REQUIRED_FIELDS:
        if name not in fields or fields[name] in (None, "", []):
            found.append(f"front matter lacks {name!r}")
    commit = str(fields.get("commit", ""))
    if not _COMMIT.match(commit):
        found.append(f"commit {commit!r} is not a 40-character lowercase hex SHA")
    if fields.get("problem") not in PROBLEMS:
        found.append(f"problem {fields.get('problem')!r} is not one of {PROBLEMS}")
    found += _fixture_problems(fields.get("fixture", ""))
    if fields.get("size") not in SIZES:
        found.append(f"size {fields.get('size')!r} is not one of {SIZES}")
    if fields.get("status") not in STATUSES:
        found.append(f"status {fields.get('status')!r} is not one of {STATUSES}")
    for name in ("tickets", "methods", "seeds"):
        if not isinstance(fields.get(name), list) or not fields.get(name):
            found.append(f"{name} must be a non-empty list")
    if not isinstance(fields.get("budget"), int) or isinstance(
        fields.get("budget"), bool
    ):
        found.append("budget must be an integer count of evaluations")
    if not experiment.title:
        found.append("no '# ' title")
    for section in SECTIONS:
        if section not in experiment.sections:
            found.append(f"section {section!r} missing")
        elif not experiment.sections[section]:
            found.append(f"section {section!r} is empty")
    actions = experiment.sections.get("Conclusion and actions", "")
    if actions and actions.strip().lower() != "none" and not _TICKET.search(actions):
        found.append(
            "every action in 'Conclusion and actions' names a ticket, or the section says none"
        )
    stem = experiment.path.stem
    if not re.match(r"\A\d{3}-[a-z0-9-]+\Z", stem):
        found.append(f"file name {experiment.path.name!r} is not NNN-slug.md")
    elif str(fields.get("id", "")).zfill(3) != stem[:3]:
        found.append(
            f"id {fields.get('id')!r} does not match the file name's {stem[:3]}"
        )
    return found


def _fixture_problems(field: object) -> list[str]:
    """What the ``fixture`` field gets wrong, as one line each.

    An experiment is a measurement on an instance, and an instance that is
    only described in prose is one nothing else can be run on. The field
    therefore names a registry fixture --- a file, or the problem directory
    when the experiment swept sizes around the declared instance.
    """
    references = _FIXTURE_REFERENCE.findall(str(field))
    if not references:
        return [
            "fixture must name a fixture under tests/regression/fixtures/, "
            "a file or the problem directory (issue #382)"
        ]
    return [
        f"fixture names {reference!r}, which does not exist"
        for reference in references
        if not (REPO_ROOT / reference).exists()
    ]


def experiments(directory: Path = EXPERIMENTS_DIR) -> list[Experiment]:
    """Every experiment file in ``directory``, the template and the index excluded, by name."""
    return [
        load(path)
        for path in sorted(directory.glob("*.md"))
        if path.name not in (TEMPLATE, INDEX)
    ]


def _joined(values: object) -> str:
    """A list field as ``a, b, c``."""
    return (
        ", ".join(str(v) for v in values) if isinstance(values, list) else str(values)
    )


def render_index(found: list[Experiment]) -> str:
    """The ``README.md`` body: what the ledger is for, and one row per experiment."""
    rows = [
        "| id | problem | size | methods | budget | status | finding |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for experiment in found:
        fields = experiment.fields
        finding = (
            experiment.sections.get("Finding", "").split("\n\n")[0].replace("\n", " ")
        )
        if len(finding) > 160:
            finding = finding[:157].rstrip() + "..."
        rows.append(
            f"| [{fields['id']}]({experiment.path.name}) | {fields['problem']} | {fields['size']} | "
            f"{_joined(fields['methods'])} | {fields['budget']} | {fields['status']} | "
            f"{finding} |"
        )
    header = (
        "# docs/experiments/\n\n"
        "One file per experiment, written from [`TEMPLATE.md`](TEMPLATE.md): the commit,\n"
        "the feature under test, the fixture and its size tier, the methods compared at\n"
        "one budget over shared seeds, the results, the finding, and the actions it\n"
        "filed (issue #314). `STATUS.md` cites an experiment rather than restating its\n"
        "table, and `tests/regression/test_experiments.py` holds every file to the\n"
        "template and this index to the files.\n\n"
        "**This index is generated.** Rewrite it with `python infra/experiments.py`;\n"
        "`--check` fails when a file is invalid or the index is stale.\n\n"
    )
    return header + "\n".join(rows) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Rewrite the index, or check the files and the index."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="validate and compare, write nothing"
    )
    parser.add_argument("--directory", type=Path, default=EXPERIMENTS_DIR)
    arguments = parser.parse_args(argv)
    failed = False
    found: list[Experiment] = []
    for path in sorted(arguments.directory.glob("*.md")):
        if path.name in (TEMPLATE, INDEX):
            continue
        try:
            found.append(load(path))
        except ValueError as error:
            failed = True
            print(error, file=sys.stderr)
    for experiment in found:
        for problem in problems(experiment):
            failed = True
            print(f"{experiment.path.name}: {problem}", file=sys.stderr)
    index = render_index(found)
    index_path = arguments.directory / INDEX
    if arguments.check:
        if not index_path.is_file() or index_path.read_text() != index:
            failed = True
            print(f"{INDEX} is stale; run python infra/experiments.py", file=sys.stderr)
        return 1 if failed else 0
    index_path.write_text(index)
    print(f"wrote {index_path} ({len(found)} experiments)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
