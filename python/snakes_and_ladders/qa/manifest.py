"""Which QA figures exist, and what renders each one.

This is the single statement of that. It used to be the sequence of thirteen
invocations in ``infra/build_documents.sh``, which nothing connected to
the document: when the document stopped citing eleven of the figures, the
build kept regenerating all thirteen and no check noticed.

Two consumers read this. Per pull request, ``snakes_and_ladders.qa.build`` regenerates only
what the documents under ``docs/tex/`` cite, so the cost tracks them rather than
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

from snakes_and_ladders.inputs import Reach, digest, library_versions, reachable

FIXTURES = "tests/regression/fixtures"

#: The most a figure the documents cite may take to render, in seconds on the
#: reference host (4 cores). A per-pull-request build is the sum of its stale
#: cited figures, and one figure over this cap is a third of the 300 s budget
#: `DEV.md` gives the whole validation (issue #372). A figure that cannot fit
#: is cited by nothing and rendered at the release gate, as `topology_accuracy`
#: is; one cited and over the cap is refused by a guard unless it is waived
#: here with the ticket that will bring it under.
CITED_RENDER_CAP = 30.0

#: Cited figures over the cap, each with the ticket that owns cutting it.
#: A waiver is a debt with a name, not an exemption.
CAP_WAIVERS: dict[str, str] = {"rl_tree_policy": "#372", "search_trajectory": "#372"}


@dataclass(frozen=True)
class FigureSpec:
    """One QA output, and the command that renders it.

    Parameters
    ----------
    stem : str
        Output basename, without extension. A document under ``docs/tex/``
        refers to
        the figure by this name, and the script writes ``<stem>.pdf`` (or
        ``.tex``) and ``<stem>_caption.txt``.
    module : str
        Module run with ``python -m``. Each figure renders in its own process,
        as it did when a shell script invoked them, so no figure inherits
        matplotlib state from the one before it.
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

    def reach(self, root: Path) -> Reach:
        """The definitions this renderer executes, fingerprinted.

        Parameters
        ----------
        root : Path
            The repository root.

        Returns
        -------
        Reach
            See :func:`snakes_and_ladders.inputs.reachable`.
        """
        return reachable([self.module], root)

    def data(self, root: Path, reach: Reach) -> list[Path]:
        """The inputs hashed as bytes rather than as code.

        Parameters
        ----------
        root : Path
            The repository root.
        reach : Reach
            This renderer's reachable code, which says whether the compiled
            extension is among what it runs.

        Returns
        -------
        list[Path]
            Every argument naming an existing file, then the Rust sources and
            ``Cargo.lock`` when a reachable definition names the extension.
        """
        files = [root / argument for argument in self.arguments]
        found = [path for path in files if path.is_file()]
        if reach.extension:
            found += sorted((root / "src").rglob("*.rs"))
            found.append(root / "Cargo.lock")
        return found

    def inputs(self, root: Path) -> list[Path]:
        """Every file whose change can change what this figure renders.

        The sources are hashed by :meth:`reach` as code rather than as bytes,
        so this is what they cover, for reporting and for tests; the digest
        below is what the stamp records.

        Parameters
        ----------
        root : Path
            The repository root.

        Returns
        -------
        list[Path]
            The package sources the renderer executes, then the fixtures its
            arguments name and the Rust sources when it reaches the extension.
        """
        reach = self.reach(root)
        return [*reach.modules, *self.data(root, reach)]

    def input_digest(self, root: Path) -> str:
        """Hash of the inputs, the spec itself and the drawing libraries.

        The package sources enter as :attr:`Reach.fingerprint` --- the ASTs of
        the definitions the renderer reaches, docstrings and comments removed
        --- and not as their bytes, so a reworded docstring or a constant the
        renderer never reads is not a new figure (issues #372, #394). The
        modules the walk had to hash whole enter by name, so a module that
        becomes opaque to it restamps rather than quietly widening.

        Returns
        -------
        str
            A SHA-256 hex digest; equal for two trees that render the same
            figure, different when any input differs.
        """
        reach = self.reach(root)
        return digest(
            self.data(root, reach),
            root,
            reach.fingerprint,
            *reach.fallbacks,
            self.stem,
            self.module,
            *self.arguments,
            *library_versions(),
        )

    def stamp(self, output_dir: Path) -> Path:
        """The file recording the digest the committed figure was rendered from.

        Returns
        -------
        Path
            ``<output_dir>/<stem>.inputs``.
        """
        return output_dir / f"{self.stem}.inputs"

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
    # The search figures each sweep all 105 unrooted topologies on the
    # 6-taxon fixture.
    FigureSpec(
        "search_trajectory",
        "snakes_and_ladders.qa.search_trajectory",
        ("--params", f"{FIXTURES}/tree_search/stress.yaml"),
        seconds=39.6,
    ),
    FigureSpec(
        "search_topologies",
        "snakes_and_ladders.qa.search_topologies",
        ("--params", f"{FIXTURES}/tree_search/stress.yaml"),
        seconds=29.6,
    ),
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
