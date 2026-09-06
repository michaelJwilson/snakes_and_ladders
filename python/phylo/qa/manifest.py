"""Which QA figures exist, and what renders each one.

This is the single statement of that. It used to be the sequence of thirteen
invocations in ``infra/build_technical_doc.sh``, which nothing connected to
the document: when the document stopped citing eleven of the figures, the
build kept regenerating all thirteen and no check noticed.

Two consumers read this. Per pull request, ``phylo.qa.build`` regenerates only
what ``docs/tex/main.tex`` cites, so the cost tracks the document rather than
drifting from it. At release, ``infra/release.sh`` regenerates every entry, so
a figure the document has stopped citing still cannot rot unnoticed -- the
check moves rather than disappearing.

The fixtures are named here rather than in the build script because which
alignment a figure was rendered from is what its caption reports, and that is
the application's knowledge, not the build's (``qa/CLAUDE.md``).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

FIXTURES = "tests/regression/fixtures"


@dataclass(frozen=True)
class FigureSpec:
    """One QA output, and the command that renders it.

    Parameters
    ----------
    stem : str
        Output basename, without extension. ``docs/tex/main.tex`` refers to
        the figure by this name, and the script writes ``<stem>.pdf`` (or
        ``.tex``) and ``<stem>_caption.txt``.
    module : str
        Module run with ``python -m``. Each figure renders in its own process,
        as it did when a shell script invoked them, so no figure inherits
        matplotlib state from the one before it.
    arguments : tuple[str, ...]
        Everything but ``--output-dir``, which the runner supplies. Paths are
        relative to the repository root.
    """

    stem: str
    module: str
    arguments: tuple[str, ...]

    def command(self, output_dir: Path) -> list[str]:
        """Build the argument vector that renders this figure.

        Returns
        -------
        list[str]
            A ``python -m`` command, less the interpreter.
        """
        return [
            "-m",
            self.module,
            *self.arguments,
            "--output-dir",
            str(output_dir),
        ]


FIGURES: tuple[FigureSpec, ...] = (
    FigureSpec(
        "sim_tree",
        "phylo.qa.sim_tree",
        ("--params", f"{FIXTURES}/simulation_params_8taxa.yaml"),
    ),
    FigureSpec(
        "sim_example",
        "phylo.qa.sim_example",
        ("--params", f"{FIXTURES}/simulation_params.yaml"),
    ),
    FigureSpec(
        "jc_transition",
        "phylo.qa.jc_transition",
        ("--params", f"{FIXTURES}/simulation_params.yaml"),
    ),
    # Brute-force marginalization costs k**m for m internal nodes, so this
    # runs on the 4-taxon fixture and nowhere larger.
    FigureSpec(
        "backend_agreement",
        "phylo.qa.backend_agreement",
        ("--params", f"{FIXTURES}/simulation_params.yaml"),
    ),
    FigureSpec(
        "likelihood_footprint",
        "phylo.qa.likelihood_footprint",
        (),
    ),
    FigureSpec(
        "sim_problem_sizes",
        "phylo.qa.sim_problem_sizes",
        (
            "--params",
            f"{FIXTURES}/simulation_params.yaml",
            "--params",
            f"{FIXTURES}/simulation_params_small_sites.yaml",
            "--params",
            f"{FIXTURES}/simulation_params_8taxa.yaml",
        ),
    ),
    # The optimization figures refit both reference instances many times over.
    FigureSpec(
        "opt_recovery",
        "phylo.qa.opt_recovery",
        (
            "--potts-params",
            f"{FIXTURES}/potts_params.yaml",
            "--hmm-params",
            f"{FIXTURES}/hmm_params.yaml",
        ),
    ),
    FigureSpec(
        "opt_coverage",
        "phylo.qa.opt_coverage",
        (
            "--potts-params",
            f"{FIXTURES}/potts_params.yaml",
            "--hmm-params",
            f"{FIXTURES}/hmm_params.yaml",
        ),
    ),
    FigureSpec(
        "opt_branch_recovery",
        "phylo.qa.opt_branch_recovery",
        (
            "--unrooted-params",
            f"{FIXTURES}/simulation_params_small_sites.yaml",
            "--rooted-params",
            f"{FIXTURES}/simulation_params_8taxa.yaml",
        ),
    ),
    FigureSpec(
        "opt_model_recovery",
        "phylo.qa.opt_model_recovery",
        ("--params", f"{FIXTURES}/simulation_params_8taxa.yaml"),
    ),
    # The search figures each sweep all 105 unrooted topologies on the
    # 6-taxon fixture.
    FigureSpec(
        "search_trajectory",
        "phylo.qa.search_trajectory",
        ("--params", f"{FIXTURES}/simulation_params_6taxa.yaml"),
    ),
    FigureSpec(
        "search_topologies",
        "phylo.qa.search_topologies",
        ("--params", f"{FIXTURES}/simulation_params_6taxa.yaml"),
    ),
    FigureSpec(
        "rl_reward_surface",
        "phylo.qa.rl_reward_surface",
        ("--params", f"{FIXTURES}/simulation_params_6taxa.yaml"),
    ),
    FigureSpec(
        "topology_accuracy",
        "phylo.qa.topology_accuracy",
        ("--params", f"{FIXTURES}/simulation_params_6taxa.yaml"),
    ),
)

# Any reference to `figures/<stem>` in the document, whichever way it is
# pulled in: `\includegraphics` for a plot, `\input` for a typeset table, and
# `\qacaptionread` for the caption beside either. Matching the path rather
# than the command means a figure included by some future fourth mechanism
# still counts as cited, which is the safe direction to be wrong in -- the
# failure it prevents is a cited figure going unregenerated.
_FIGURE_REFERENCE = re.compile(r"figures/([A-Za-z0-9_]+)")

_CAPTION_SUFFIX = "_caption"


def cited_stems(main_tex: Path) -> set[str]:
    """Find the figure stems ``main_tex`` refers to.

    A caption reference (``figures/<stem>_caption.txt``) counts as a
    reference to ``<stem>``: the caption is an output of the same script, and
    a document quoting a caption needs the figure regenerated with it.

    Parameters
    ----------
    main_tex : Path
        The LaTeX source to scan.

    Returns
    -------
    set[str]
        Every stem referred to, whether or not this manifest knows it.
    """
    text = main_tex.read_text()
    stems = set()
    for match in _FIGURE_REFERENCE.finditer(text):
        stem = match.group(1)
        if stem.endswith(_CAPTION_SUFFIX):
            stem = stem[: -len(_CAPTION_SUFFIX)]
        stems.add(stem)
    return stems


def select(stems: Iterable[str]) -> tuple[FigureSpec, ...]:
    """Pick the manifest entries rendering ``stems``, in manifest order.

    Order is the manifest's rather than the caller's so a build runs the
    cheap figures first and the same way every time.

    Returns
    -------
    tuple[FigureSpec, ...]
        The matching specs.
    """
    wanted = set(stems)
    return tuple(spec for spec in FIGURES if spec.stem in wanted)


def unknown_stems(stems: Iterable[str]) -> set[str]:
    """Find which of ``stems`` this manifest cannot render.

    Returns
    -------
    set[str]
        Stems with no manifest entry.
    """
    known = {spec.stem for spec in FIGURES}
    return {stem for stem in stems if stem not in known}
