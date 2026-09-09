"""What the citation-driven figure selection must guarantee.

The build regenerates only the figures the documents under ``docs/tex/`` cite,
and the release gate regenerates the rest (issue #154). That trade is only
sound if three things hold, and each is asserted here: a document can never cite a
figure the build skips, no committed figure falls outside the release gate's
reach, and a rotted figure is still caught -- by the per-PR path when the
document cites it, and by the release path when it does not.

Since issue #492 the correspondence is a bijection and is asserted as one.
``infra/check_citations.py`` covers cited-but-missing; the other direction,
a figure the manifest renders and no document cites, is
``test_every_manifest_figure_is_cited_by_a_document`` below. The two orphans
that direction was written for, ``sim_problem_sizes`` and
``topology_accuracy``, were rendered on every release and read by nobody. The
selection tests that used to rely on their existence now construct the
uncited case in ``tmp_path`` instead, so they check the mechanism rather than
a state of the repository that is now forbidden.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from snakes_and_ladders.qa import build, manifest
from snakes_and_ladders.qa.build import UncitedFigureError, compare, selected
from snakes_and_ladders.qa.manifest import FIGURES, cited_stems

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
@pytest.mark.structural
def test_every_figure_the_documents_cite_has_a_manifest_entry() -> None:
    # The failure this prevents: a figure added to a document that no build
    # regenerates, left to drift from the code that produced it while the
    # staleness check passes because nothing rebuilt it.
    assert manifest.unknown_stems(cited_stems(*DOCUMENTS)) == set()


@pytest.mark.critical
@pytest.mark.structural
def test_every_committed_figure_has_a_manifest_entry() -> None:
    # The release gate renders the manifest, so a committed figure absent from
    # it would be checked by nothing at all -- neither per PR nor at release.
    committed = {
        path.stem.removesuffix("_caption") for path in COMMITTED_FIGURES.iterdir()
    }
    known = {spec.stem for spec in FIGURES}
    assert committed - known == set()


@pytest.mark.critical
@pytest.mark.structural
def test_every_manifest_figure_is_cited_by_a_document() -> None:
    # Issue #492's invariant, and the half `infra/check_citations.py` does not
    # cover: that script fails a citation with no figure, and this fails a
    # figure with no citation. An uncited figure is rendered by the release
    # gate and read by nobody, so nothing ever decides whether it is right --
    # which is how `sim_problem_sizes` and `topology_accuracy` sat stale on
    # `main` for eight releases. A figure whose document has no room for it is
    # deleted with its renderer, not left rendering.
    uncited = {spec.stem for spec in FIGURES} - cited_stems(*DOCUMENTS)

    assert uncited == set(), (
        f"rendered and cited by nothing: {sorted(uncited)}; cite each from the "
        "document that asked for it, or remove it with its renderer (DEV.md, "
        "'A figure exists because a document asked for it')"
    )


@pytest.mark.structural
def test_the_release_gate_selects_the_whole_manifest() -> None:
    # `--all` ignores the citations entirely, which is what makes the per-PR
    # selection a cost decision rather than a substitute for the gate. It held
    # when two figures were uncited and it holds now that none is.
    assert selected(DOCUMENTS, every=True) == FIGURES


@pytest.mark.structural
def test_a_document_citing_less_selects_less(tmp_path: Path) -> None:
    # What the strict subset above used to assert against the repository:
    # the selection tracks the citations rather than returning the manifest.
    # Constructed here, because every figure is cited on `main` since #492 and
    # a test that read the repository would now be asserting the empty case.
    document = tmp_path / "main.tex"
    document.write_text(r"\includegraphics{figures/sim_tree}")

    assert set(selected([document], every=False)) < set(selected(DOCUMENTS, every=True))


@pytest.mark.structural
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


@pytest.mark.edge_case
def test_a_document_citing_an_unknown_figure_is_refused(tmp_path: Path) -> None:
    # Refused rather than skipped: skipping is exactly the silent failure the
    # selection would otherwise introduce.
    document = tmp_path / "main.tex"
    document.write_text(r"\includegraphics{figures/no_such_figure}")

    with pytest.raises(UncitedFigureError, match="no_such_figure"):
        selected([document], every=False)


@pytest.mark.structural
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


@pytest.mark.edge_case
def test_a_figure_missing_from_the_committed_set_is_reported_as_stale(
    tmp_path: Path,
) -> None:
    rebuilt = tmp_path / "rebuilt"
    committed = tmp_path / "committed"
    rebuilt.mkdir()
    committed.mkdir()
    (rebuilt / "figure.pdf").write_bytes(b"rendered")

    assert compare(rebuilt, committed) == ["figure.pdf"]


@pytest.mark.structural
def test_matching_figures_are_reported_as_clean(tmp_path: Path) -> None:
    rebuilt = tmp_path / "rebuilt"
    committed = tmp_path / "committed"
    rebuilt.mkdir()
    committed.mkdir()
    (rebuilt / "figure.pdf").write_bytes(b"same")
    (committed / "figure.pdf").write_bytes(b"same")

    assert compare(rebuilt, committed) == []


@pytest.mark.structural
@stress_only(
    "renders every figure in the manifest, which is the release "
    "gate's job; the cited-figure paths are checked at CI tier above"
)
def test_check_catches_an_uncited_figure_that_has_rotted(tmp_path: Path) -> None:
    # Both directions of the trade, on a real rendering: `--check` without
    # `--all` passes over a figure the given document does not cite, and
    # `--check --all` catches it.
    #
    # The document is written here rather than taken from `docs/tex/`. Every
    # committed figure is cited since #492, so the repository no longer
    # supplies this case -- and a version of this test that kept reading the
    # real documents would still have passed, for the unrelated reason that
    # the rotted figure's *stamp* was copied intact and the staleness cache
    # skipped it. That would be the release gate's guarantee asserted by the
    # cache's behaviour, which is the substitution this module exists to deny.
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


@pytest.mark.structural
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


@pytest.mark.structural
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


@pytest.mark.edge_case
def test_a_selection_over_no_document_is_refused() -> None:
    # An empty union cites nothing and would render nothing, while passing
    # every check that asks whether the cited figures are fresh.
    with pytest.raises(ValueError, match="at least one document"):
        cited_stems()


@pytest.mark.structural
def test_the_documents_the_build_defaults_to_all_exist() -> None:
    # `DEFAULT_DOCUMENTS` is what the build script and the release gate agree
    # on. A path renamed on one side only would raise far from its cause.
    assert [document.name for document in DOCUMENTS] == ["paper.tex", "textbook.tex"]
    assert all(document.is_file() for document in DOCUMENTS)


@pytest.mark.structural
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
