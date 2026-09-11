"""Which QA figures exist, and what renders each one.

The single statement of that. It used to be the thirteen invocations in
``infra/build_documents.sh``, which nothing connected to the document: when the
document stopped citing eleven of the figures, the build kept regenerating all
thirteen and no check noticed.

Two consumers read this. Per pull request, ``snakes_and_ladders.qa.build``
regenerates only what the documents under ``docs/tex/`` cite, so the cost
tracks them. At release, ``infra/release.sh`` regenerates every entry, so a
figure the document has stopped citing cannot rot unnoticed.

Since issue #492 the two sets coincide: every entry here is cited by one of the
documents, and a guard fails one that is not
(``tests/regression/qa/test_qa_build.py``). The selection is kept as the
mechanism rather than the current count --- a document that drops a citation
narrows it again the same day. Adding an entry means citing it; ``DEV.md``
states the path from the request to the citation.

The fixtures are named here rather than in the build script because which
alignment a figure was rendered from is what its caption reports, which is the
application's knowledge and not the build's (``qa/CLAUDE.md``).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

FIXTURES = "tests/regression/fixtures"

#: The most a figure the documents cite may take to render, in seconds on the
#: reference host (4 cores). A documents build is the sum of its cited figures
#: --- every one since issue #490 removed the stamps that skipped some --- and
#: one figure over this cap is a third of the 300 s budget `DEV.md` gives the
#: whole validation (issue #372). "Cited by nothing" was the other way out
#: until issue #492 closed it, so a figure over the cap is waived below with
#: the ticket that will bring it under, or it is cut.
CITED_RENDER_CAP = 30.0

#: Cited figures over the cap, each with the ticket that owns cutting it.
#: A waiver is a debt with a name, not an exemption. Issue #498 tried to pay
#: the first two by rendering at five taxa and could pay neither:
#: `rl_tree_policy` renders from the 7-taxon fixture issue #177 chose, so the
#: taxon count is not its term to cut, and `search_trajectory` loses its
#: trajectory panel at five taxa (the comment on its entry below).
#: `topology_accuracy` is the third since #492 cited it: its cost is
#: attributed and unreduced --- no topology sweep at all, but 91.5% L-BFGS
#: branch-length fitting over 48 inferences (#506) --- and #509 owns cutting
#: the term the profile names.
CAP_WAIVERS: dict[str, str] = {
    "rl_tree_policy": "#372",
    "search_trajectory": "#372",
    "topology_accuracy": "#509",
}


@dataclass(frozen=True)
class FigureSpec:
    """One QA output, and the command that renders it.

    Parameters
    ----------
    stem : str
        Output basename, without extension. A document under ``docs/tex/``
        refers to the figure by this name, and the script writes ``<stem>.pdf``
        (or ``.tex``) and ``<stem>_caption.txt``.
    module : str
        Module run with ``python -m``. Each figure renders in its own process,
        so none inherits matplotlib state from the one before it.
    arguments : tuple[str, ...]
        Everything but ``--output-dir``, which the runner supplies. Paths are
        relative to the repository root.
    seconds : float
        Wall clock of one render on the reference host, measured alone with
        ``infra/measure_build.sh`` and stated here so the cap on a cited
        figure is checked against a number rather than an impression.
    """

    stem: str
    module: str
    arguments: tuple[str, ...]
    seconds: float

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
        "snakes_and_ladders.qa.sim_tree",
        ("--params", f"{FIXTURES}/tree_jc/release.yaml"),
        seconds=2.7,
    ),
    FigureSpec(
        "sim_example",
        "snakes_and_ladders.qa.sim_example",
        ("--params", f"{FIXTURES}/tree_jc/stress.yaml"),
        seconds=2.7,
    ),
    FigureSpec(
        "jc_transition",
        "snakes_and_ladders.qa.jc_transition",
        ("--params", f"{FIXTURES}/tree_jc/stress.yaml"),
        seconds=2.7,
    ),
    # Brute-force marginalization costs k**m for m internal nodes, so this
    # runs on the 4-taxon fixture and nowhere larger.
    FigureSpec(
        "backend_agreement",
        "snakes_and_ladders.qa.backend_agreement",
        ("--params", f"{FIXTURES}/tree_jc/stress.yaml"),
        seconds=3.1,
    ),
    FigureSpec(
        "likelihood_footprint",
        "snakes_and_ladders.qa.likelihood_footprint",
        ("--params", f"{FIXTURES}/tree_jc/ci.yaml"),
        seconds=2.6,
    ),
    FigureSpec(
        "sim_problem_sizes",
        "snakes_and_ladders.qa.sim_problem_sizes",
        (
            "--params",
            f"{FIXTURES}/tree_jc/stress.yaml",
            "--params",
            f"{FIXTURES}/tree_jc/ci.yaml",
            "--params",
            f"{FIXTURES}/tree_jc/release.yaml",
        ),
        seconds=2.5,
    ),
    # The optimization figures refit both reference instances many times over.
    FigureSpec(
        "opt_recovery",
        "snakes_and_ladders.qa.opt_recovery",
        (
            "--potts-params",
            f"{FIXTURES}/potts_chain/ci.yaml",
            "--hmm-params",
            f"{FIXTURES}/hmm/ci.yaml",
        ),
        seconds=5.0,
    ),
    FigureSpec(
        "opt_coverage",
        "snakes_and_ladders.qa.opt_coverage",
        (
            "--potts-params",
            f"{FIXTURES}/potts_chain/ci.yaml",
            "--hmm-params",
            f"{FIXTURES}/hmm/ci.yaml",
        ),
        seconds=27.3,
    ),
    FigureSpec(
        "opt_branch_recovery",
        "snakes_and_ladders.qa.opt_branch_recovery",
        (
            "--unrooted-params",
            f"{FIXTURES}/tree_jc/ci.yaml",
            "--rooted-params",
            f"{FIXTURES}/tree_jc/release.yaml",
        ),
        seconds=5.0,
    ),
    FigureSpec(
        "opt_model_recovery",
        "snakes_and_ladders.qa.opt_model_recovery",
        ("--params", f"{FIXTURES}/tree_jc/release.yaml"),
        seconds=7.5,
    ),
    # Issue #498 asked whether these three could sweep the 5-taxon fixture's
    # 15 unrooted topologies instead of the 6-taxon fixture's 105. One can.
    #
    # `search_trajectory` stays at 6 taxa: its panel (a) is the climb, and at 5
    # taxa there is none -- hill climbing starts at or beside the optimum, both
    # move sets terminate on the first evaluation, and the panel is a single
    # marker at zero fits. The sweep is 61% of its 41.7 s, so the saving was
    # real and the figure was not.
    FigureSpec(
        "search_trajectory",
        "snakes_and_ladders.qa.search_trajectory",
        ("--params", f"{FIXTURES}/tree_search/stress.yaml"),
        seconds=39.6,
    ),
    # `search_topologies` moves: it draws two trees and never the sweep, which
    # only picks the runner-up, so exhaustiveness is all it needs from the
    # enumeration and 15 topologies are exhaustive. Every claim its caption
    # makes survives -- the search recovers the generating topology, the
    # runner-up differs by one split, and that split is supportable only by
    # collapsing an internal edge to 0.000.
    FigureSpec(
        "search_topologies",
        "snakes_and_ladders.qa.search_topologies",
        ("--params", f"{FIXTURES}/tree_search/ci.yaml"),
        seconds=7.7,
    ),
    # `rl_reward_surface` stays at 6 taxa: its panel (a) *is* a distribution
    # over the topology space, and 15 points state it less precisely than 105.
    # `tree_search/ci.yaml` reserves the 6-taxon instance for it, the
    # resolution being what the release gate pays for.
    FigureSpec(
        "rl_reward_surface",
        "snakes_and_ladders.qa.rl_reward_surface",
        ("--params", f"{FIXTURES}/tree_search/stress.yaml"),
        seconds=28.4,
    ),
    # Trains eight policies, so it is the most expensive entry here; the
    # budget it trains at is chosen in the module for that reason.
    FigureSpec(
        "rl_tree_policy",
        "snakes_and_ladders.qa.rl_tree_policy",
        ("--params", f"{FIXTURES}/tree_search/release.yaml"),
        seconds=101.4,
    ),
    # Enumerates no topology at all: its cost is 48 NNI hill-climbing
    # inferences over six site counts, 91% of it L-BFGS branch-length fitting
    # (issue #498). The fixture supplies the generating tree, so the taxon
    # count is not the term to cut here. Cited by `paper.tex` since #492 and
    # over the cap, so it is waived above under #509.
    FigureSpec(
        "topology_accuracy",
        "snakes_and_ladders.qa.topology_accuracy",
        ("--params", f"{FIXTURES}/tree_search/stress.yaml"),
        seconds=124.0,
    ),
    # The textbook's problem-statement figures (issue #358). The parsimony
    # figure scores every topology of the 8-taxon fixture at a reduced site
    # count; the other three name the fixture their instance is declared in,
    # as every entry here does since issue #382.
    FigureSpec(
        "tropical_relaxation",
        "snakes_and_ladders.qa.tropical_relaxation",
        ("--params", f"{FIXTURES}/tree_search/ci.yaml"),
        seconds=5.4,
    ),
    FigureSpec(
        "parsimony_zones",
        "snakes_and_ladders.qa.parsimony_zones",
        ("--params", f"{FIXTURES}/tree_jc/release.yaml"),
        seconds=4.6,
    ),
    FigureSpec(
        "frustrated_lattices",
        "snakes_and_ladders.qa.frustrated_lattices",
        ("--params", f"{FIXTURES}/frustrated_lattice/ci.yaml"),
        seconds=3.9,
    ),
    FigureSpec(
        "mixture_seeding",
        "snakes_and_ladders.qa.mixture_seeding",
        ("--params", f"{FIXTURES}/mixture/ci.yaml"),
        seconds=27.4,
    ),
    FigureSpec(
        "optimizer_landscapes",
        "snakes_and_ladders.qa.optimizer_landscapes",
        ("--params", f"{FIXTURES}/test_functions/ci.yaml"),
        seconds=6.6,
    ),
    # The two rendered instances of issue #394, drawn beside the textbook's
    # hand-drawn sketches of the same two problems. Both are a drawing of one
    # fixture rather than a study over many, so both are cheap.
    FigureSpec(
        "tanner_graph",
        "snakes_and_ladders.qa.tanner_graph",
        ("--params", f"{FIXTURES}/ldpc/ci.yaml"),
        seconds=2.7,
    ),
    FigureSpec(
        "turbo_waterfall",
        "snakes_and_ladders.qa.turbo_waterfall",
        ("--params", f"{FIXTURES}/turbo/stress.yaml"),
        seconds=16.0,
    ),
    FigureSpec(
        "coupled_labelling",
        "snakes_and_ladders.qa.coupled_labelling",
        ("--params", f"{FIXTURES}/spatio_sequential/stress.yaml"),
        seconds=3.0,
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


def cited_stems(*sources: Path) -> set[str]:
    """Find the figure stems the given LaTeX sources refer to, as one set.

    A caption reference (``figures/<stem>_caption.txt``) counts as a
    reference to ``<stem>``: the caption is an output of the same script, and
    a document quoting a caption needs the figure regenerated with it.

    **Several sources rather than one, and the union rather than each.** The
    repository builds a paper and a textbook (issue #249). A per-pull-request
    selection derived from one of them stops regenerating every figure the
    other cites, and nothing notices --- issue #154's defect in mirror image,
    where the build regenerated figures the document had stopped citing.
    Taking the union makes leaving a document out a selection that is *wrong*
    rather than one that is quietly smaller.

    Parameters
    ----------
    *sources : Path
        The LaTeX sources to scan. At least one.

    Returns
    -------
    set[str]
        Every stem referred to by any of them, whether or not this manifest
        knows it.

    Raises
    ------
    ValueError
        If no source is given. An empty union selects nothing, which would
        pass every check while regenerating no figure at all.
    """
    if not sources:
        msg = "cited_stems needs at least one document; an empty set cites nothing"
        raise ValueError(msg)
    stems = set()
    for source in sources:
        for match in _FIGURE_REFERENCE.finditer(source.read_text()):
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
