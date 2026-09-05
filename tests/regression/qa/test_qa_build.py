"""What the citation-driven figure selection must guarantee.

The build regenerates only the figures ``docs/tex/main.tex`` cites, and the
release gate regenerates the rest (issue #154). That trade is only sound if
three things hold, and each is asserted here: the document can never cite a
figure the build skips, no committed figure falls outside the release gate's
reach, and a rotted figure is still caught -- by the per-PR path when the
document cites it, and by the release path when it does not.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from phylo.qa import build, manifest
from phylo.qa.build import UncitedFigureError, compare, selected
from phylo.qa.manifest import FIGURES, cited_stems

from tests._scale import stress_only

MAIN_TEX = build.DEFAULT_MAIN_TEX
COMMITTED_FIGURES = build.DEFAULT_OUTPUT_DIR

# The cheapest entry in the manifest, so the tests that actually render
# something cost a second rather than two minutes.
CHEAP_STEM = "sim_problem_sizes"


def test_every_figure_the_document_cites_has_a_manifest_entry() -> None:
    # The failure this prevents: a figure added to the document that no build
    # regenerates, left to drift from the code that produced it while the
    # staleness check passes because nothing rebuilt it.
    assert manifest.unknown_stems(cited_stems(MAIN_TEX)) == set()


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
    cited = selected(MAIN_TEX, every=False)
    every = selected(MAIN_TEX, every=True)

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

    assert {spec.stem for spec in selected(document, every=False)} == {
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
        selected(document, every=False)


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


@stress_only(
    "renders every figure in the manifest, which is the release "
    "gate's job; the cited-figure paths are checked at CI tier above"
)
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
        ["--main-tex", str(MAIN_TEX), "--output-dir", str(output_dir), "--check"]
    )
    release_gate = build.main(
        [
            "--main-tex",
            str(MAIN_TEX),
            "--output-dir",
            str(output_dir),
            "--check",
            "--only",
            CHEAP_STEM,
        ]
    )

    assert cited_only == 0, "the per-PR check does not cover an uncited figure"
    assert release_gate == 1, "the release gate must catch it"
