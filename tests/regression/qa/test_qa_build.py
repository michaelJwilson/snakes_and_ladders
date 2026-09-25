"""What the citation-driven figure selection must guarantee.

The build regenerates the figures ``docs/tex/`` cites and the release gate the
rest (issue #154): no cited figure is skipped, no committed figure escapes the
release gate, and a rotted figure is caught on one path or the other. Since
#492 the correspondence is a bijection; ``infra/check_citations.py`` and
``test_document_build_outputs.py`` hold its two directions (#982). The
uncited case is built in ``tmp_path``, testing the mechanism.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from sal.qa import build, manifest
from sal.qa.build import UncitedFigureError, compare, selected
from sal.qa.manifest import FIGURES, cited_stems

from tests._scale import stress_only

DOCUMENTS = build.DEFAULT_DOCUMENTS
COMMITTED_FIGURES = build.DEFAULT_OUTPUT_DIR

# The cheapest entry in the manifest, so the tests that actually render
# something cost a second rather than two minutes. It is cited since #492, so
# a test needing an *uncited* figure writes its own document rather than
# naming this one.
CHEAP_STEM = "sim_problem_sizes"


def _document_arguments() -> list[str]:
    """``--document`` flags naming every document the repository builds."""
    return [
        argument for document in DOCUMENTS for argument in ("--document", str(document))
    ]


@pytest.mark.critical
@pytest.mark.infra
def test_every_figure_the_documents_cite_has_a_manifest_entry() -> None:
    # The failure this prevents: a figure added to a document that no build
    # regenerates, left to drift from the code that produced it while the
    # staleness check passes because nothing rebuilt it.
    assert manifest.unknown_stems(cited_stems(*DOCUMENTS)) == set()


@pytest.mark.infra
def test_the_release_gate_selects_the_whole_manifest() -> None:
    # `--all` ignores the citations entirely, which is what makes the per-PR
    # selection a cost decision rather than a substitute for the gate. It held
    # when two figures were uncited and it holds now that none is.
    assert selected(DOCUMENTS, every=True) == FIGURES


@pytest.mark.infra
def test_a_document_citing_less_selects_less(tmp_path: Path) -> None:
    # What the strict subset above used to assert against the repository:
    # the selection tracks the citations rather than returning the manifest.
    # Constructed here, because every figure is cited on `main` since #492 and
    # a test that read the repository would now be asserting the empty case.
    document = tmp_path / "main.tex"
    document.write_text(r"\includegraphics{figures/sim_tree}")

    assert set(selected([document], every=False)) < set(selected(DOCUMENTS, every=True))


@pytest.mark.infra
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


@pytest.mark.smoke
def test_a_document_citing_an_unknown_figure_is_refused(tmp_path: Path) -> None:
    # Refused rather than skipped: skipping is exactly the silent failure the
    # selection would otherwise introduce.
    document = tmp_path / "main.tex"
    document.write_text(r"\includegraphics{figures/no_such_figure}")

    with pytest.raises(UncitedFigureError, match="no_such_figure"):
        selected([document], every=False)


@pytest.mark.infra
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


@pytest.mark.smoke
def test_a_figure_missing_from_the_committed_set_is_reported_as_stale(
    tmp_path: Path,
) -> None:
    rebuilt = tmp_path / "rebuilt"
    committed = tmp_path / "committed"
    rebuilt.mkdir()
    committed.mkdir()
    (rebuilt / "figure.pdf").write_bytes(b"rendered")

    assert compare(rebuilt, committed) == ["figure.pdf"]


@pytest.mark.infra
def test_matching_figures_are_reported_as_clean(tmp_path: Path) -> None:
    rebuilt = tmp_path / "rebuilt"
    committed = tmp_path / "committed"
    rebuilt.mkdir()
    committed.mkdir()
    (rebuilt / "figure.pdf").write_bytes(b"same")
    (committed / "figure.pdf").write_bytes(b"same")

    assert compare(rebuilt, committed) == []


@pytest.mark.infra
@stress_only(
    "renders every figure in the manifest, which is the release "
    "gate's job; the cited-figure paths are checked at CI tier above"
)
def test_check_catches_an_uncited_figure_that_has_rotted(tmp_path: Path) -> None:
    # `--check` alone passes a figure the document does not cite, `--check
    # --all` catches it. Built here: every committed figure is cited (#492),
    # and a copied stamp would let the cache skip it.
    output_dir = tmp_path / "figures"
    output_dir.mkdir()
    for path in COMMITTED_FIGURES.iterdir():
        shutil.copy(path, output_dir / path.name)
    (output_dir / f"{CHEAP_STEM}.tex").write_text("rotted")
    (output_dir / f"{CHEAP_STEM}.inputs").write_text("stale-on-purpose\n")

    document = tmp_path / "cites-something-else.tex"
    document.write_text(r"\includegraphics{figures/sim_tree}")

    cited_only = build.main(
        [
            "--document",
            str(document),
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


@pytest.mark.infra
def test_a_figure_only_the_textbook_cites_is_still_selected(
    tmp_path: Path,
) -> None:
    # The selection is the union over documents; one document alone would
    # drop the other's figures silently (#154 mirrored, #249).
    paper = tmp_path / "paper.tex"
    paper.write_text(r"\includegraphics{figures/sim_example}")
    textbook = tmp_path / "textbook.tex"
    textbook.write_text(r"\includegraphics{figures/sim_tree}")

    together = {spec.stem for spec in selected([paper, textbook], every=False)}

    assert together == {"sim_example", "sim_tree"}


@pytest.mark.infra
def test_leaving_a_document_out_selects_the_wrong_set(tmp_path: Path) -> None:
    # The paired half: the mistake the guard forbids changes the answer -- the
    # textbook's figure disappears from the selection silently, and every other
    # check still passes.
    paper = tmp_path / "paper.tex"
    paper.write_text(r"\includegraphics{figures/sim_example}")
    textbook = tmp_path / "textbook.tex"
    textbook.write_text(r"\includegraphics{figures/sim_tree}")

    partial = {spec.stem for spec in selected([paper], every=False)}

    assert "sim_tree" not in partial
    assert partial < {spec.stem for spec in selected([paper, textbook], every=False)}


@pytest.mark.smoke
def test_a_selection_over_no_document_is_refused() -> None:
    # An empty union cites nothing and would render nothing, while passing
    # every check that asks whether the cited figures are fresh.
    with pytest.raises(ValueError, match="at least one document"):
        cited_stems()


@pytest.mark.infra
def test_the_documents_the_build_defaults_to_all_exist() -> None:
    # `DEFAULT_DOCUMENTS` is what the build script and the release gate agree
    # on. A path renamed on one side only would raise far from its cause.
    assert [document.name for document in DOCUMENTS] == ["paper.tex", "textbook.tex"]
    assert all(document.is_file() for document in DOCUMENTS)
