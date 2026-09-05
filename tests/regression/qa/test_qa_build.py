"""What the citation-driven figure selection must guarantee.

The build regenerates only the figures the documents under ``docs/tex/`` cite,
and the release gate regenerates the rest (issue #154). That trade is only
sound if three things hold, and each is asserted here: a document can never cite a
figure the build skips, no committed figure falls outside the release gate's
reach, and a rotted figure is still caught -- by the per-PR path when the
document cites it, and by the release path when it does not.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from snakes_and_ladders.qa import build, manifest
from snakes_and_ladders.qa.build import UncitedFigureError, compare, selected
from snakes_and_ladders.qa.manifest import FIGURES, cited_stems

DOCUMENTS = build.DEFAULT_DOCUMENTS
COMMITTED_FIGURES = build.DEFAULT_OUTPUT_DIR

# The cheapest entry in the manifest, so the tests that actually render
# something cost a second rather than two minutes.
CHEAP_STEM = "sim_problem_sizes"


def _document_arguments() -> list[str]:
    """``--document`` flags naming every document the repository builds."""
    return [
        argument for document in DOCUMENTS for argument in ("--document", str(document))
    ]


def test_every_figure_the_documents_cite_has_a_manifest_entry() -> None:
    # The failure this prevents: a figure added to a document that no build
    # regenerates, left to drift from the code that produced it while the
    # staleness check passes because nothing rebuilt it.
    assert manifest.unknown_stems(cited_stems(*DOCUMENTS)) == set()


def test_every_committed_figure_has_a_manifest_entry() -> None:
    # The release gate renders the manifest, so a committed figure absent from
    # it would be checked by nothing at all -- neither per PR nor at release.
    committed = {
        path.stem.removesuffix("_caption") for path in COMMITTED_FIGURES.iterdir()
    }
    known = {spec.stem for spec in FIGURES}
    assert committed - known == set()


def test_the_document_selects_fewer_figures_than_the_release_gate() -> None:
    # The whole point of the change. If these were equal the per-PR build
    # would be doing the release gate's work, which is what it cost before.
    cited = selected(DOCUMENTS, every=False)
    every = selected(DOCUMENTS, every=True)

    assert set(cited) < set(every)
    assert len(every) == len(FIGURES)


def test_a_cited_figure_is_selected_whichever_way_it_is_included(
    tmp_path: Path,
) -> None:
    # `\includegraphics` for a plot, `\input` for a typeset table, and
    # `\qacaptionread` for a caption are all references, and the selection
    # keys on the path rather than the command so a fourth mechanism still
    # counts.
    document = tmp_path / "main.tex"
    document.write_text(
        r"\includegraphics{figures/sim_tree}"
        "\n"
        r"\input{figures/sim_problem_sizes}"
        "\n"
        r"\qacaptionread{figures/opt_coverage_caption.txt}{\x}"
        "\n"
    )

    assert {spec.stem for spec in selected([document], every=False)} == {
        "sim_tree",
        "sim_problem_sizes",
        "opt_coverage",
    }


def test_a_document_citing_an_unknown_figure_is_refused(tmp_path: Path) -> None:
    # Refused rather than skipped: skipping is exactly the silent failure the
    # selection would otherwise introduce.
    document = tmp_path / "main.tex"
    document.write_text(r"\includegraphics{figures/no_such_figure}")

    with pytest.raises(UncitedFigureError, match="no_such_figure"):
        selected([document], every=False)


def test_a_perturbed_figure_is_reported_as_stale(tmp_path: Path) -> None:
    # The check that has to keep working for the release gate to substitute
    # for the per-PR one: a committed figure whose bytes no longer match a
    # rebuild is named, not passed over.
    rebuilt = tmp_path / "rebuilt"
    committed = tmp_path / "committed"
    rebuilt.mkdir()
    committed.mkdir()
    (rebuilt / "figure.pdf").write_bytes(b"rendered")
    (committed / "figure.pdf").write_bytes(b"rotted")

    assert compare(rebuilt, committed) == ["figure.pdf"]


def test_a_figure_missing_from_the_committed_set_is_reported_as_stale(
    tmp_path: Path,
) -> None:
    rebuilt = tmp_path / "rebuilt"
    committed = tmp_path / "committed"
    rebuilt.mkdir()
    committed.mkdir()
    (rebuilt / "figure.pdf").write_bytes(b"rendered")

    assert compare(rebuilt, committed) == ["figure.pdf"]


def test_matching_figures_are_reported_as_clean(tmp_path: Path) -> None:
    rebuilt = tmp_path / "rebuilt"
    committed = tmp_path / "committed"
    rebuilt.mkdir()
    committed.mkdir()
    (rebuilt / "figure.pdf").write_bytes(b"same")
    (committed / "figure.pdf").write_bytes(b"same")

    assert compare(rebuilt, committed) == []


def test_check_catches_an_uncited_figure_that_has_rotted(tmp_path: Path) -> None:
    # Both directions of the trade, on a real rendering. `sim_problem_sizes`
    # is committed and *not* cited by the document, so it is exactly the case
    # the release gate exists to cover: `--check` without `--all` passes over
    # it, and `--check --all` catches it.
    output_dir = tmp_path / "figures"
    output_dir.mkdir()
    for path in COMMITTED_FIGURES.iterdir():
        shutil.copy(path, output_dir / path.name)
    (output_dir / f"{CHEAP_STEM}.tex").write_text("rotted")

    cited_only = build.main(
        [
            *_document_arguments(),
            "--output-dir",
            str(output_dir),
            "--check",
        ]
    )
    release_gate = build.main(
        [
            *_document_arguments(),
            "--output-dir",
            str(output_dir),
            "--check",
            "--only",
            CHEAP_STEM,
        ]
    )

    assert cited_only == 0, "the per-PR check does not cover an uncited figure"
    assert release_gate == 1, "the release gate must catch it"


def test_a_figure_only_the_textbook_cites_is_still_selected(
    tmp_path: Path,
) -> None:
    # The seam the split turns on. The selection is the *union* of what the
    # documents cite, so a figure the paper does not mention is regenerated
    # per pull request because the textbook does. Deriving it from one
    # document would stop regenerating the other's figures and fail nothing --
    # issue #154's defect in mirror image (issue #249).
    paper = tmp_path / "paper.tex"
    paper.write_text(r"\includegraphics{figures/sim_example}")
    textbook = tmp_path / "textbook.tex"
    textbook.write_text(r"\includegraphics{figures/sim_tree}")

    together = {spec.stem for spec in selected([paper, textbook], every=False)}

    assert together == {"sim_example", "sim_tree"}


def test_leaving_a_document_out_selects_the_wrong_set(tmp_path: Path) -> None:
    # The paired half: the guard above is only worth having if the mistake it
    # forbids is one that changes the answer. It is -- the textbook's figure
    # disappears from the selection, silently, and every other check still
    # passes.
    paper = tmp_path / "paper.tex"
    paper.write_text(r"\includegraphics{figures/sim_example}")
    textbook = tmp_path / "textbook.tex"
    textbook.write_text(r"\includegraphics{figures/sim_tree}")

    partial = {spec.stem for spec in selected([paper], every=False)}

    assert "sim_tree" not in partial
    assert partial < {spec.stem for spec in selected([paper, textbook], every=False)}


def test_a_selection_over_no_document_is_refused() -> None:
    # An empty union cites nothing and would render nothing, while passing
    # every check that asks whether the cited figures are fresh.
    with pytest.raises(ValueError, match="at least one document"):
        cited_stems()


def test_the_documents_the_build_defaults_to_all_exist() -> None:
    # `DEFAULT_DOCUMENTS` is what the build script and the release gate agree
    # on. A path renamed on one side only would raise far from its cause.
    assert [document.name for document in DOCUMENTS] == ["paper.tex", "textbook.tex"]
    assert all(document.is_file() for document in DOCUMENTS)


def test_the_textbook_names_no_code() -> None:
    # The separation the split is for (issue #249): the textbook states
    # problem formulations, algorithms and the properties that referee them,
    # and none of that depends on how any of it is implemented. A module path,
    # a filename or a function call in it is application documentation wearing
    # a textbook's clothes.
    textbook = next(
        document for document in DOCUMENTS if document.name == "textbook.tex"
    )
    text = textbook.read_text()

    offenders = [
        needle
        for needle in ("snakes_and_ladders.", ".py", "\\texttt{")
        if needle in text
    ]

    assert offenders == []
