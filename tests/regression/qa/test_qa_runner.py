"""The shared QA entry point's own contract, and every script's `main` through it.

The figure and table scripts exercise :mod:`sal.qa.runner`
end-to-end, so what is pinned here is what routing them through one function
is supposed to guarantee and what no individual figure's test would catch.
Each script's own `main` is one row of `SCRIPTS` (issue #982 moved thirteen
per-module copies of that test here): it writes its output and a caption
equal to the one it returns, reporting what it was handed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure
from sal.fixtures import load_params
from sal.qa import (
    backend_agreement,
    opt_branch_recovery,
    opt_coverage,
    opt_model_recovery,
    opt_recovery,
    parsimony_zones,
    rl_reward_surface,
    search_topologies,
    search_trajectory,
    sim_example,
    sim_problem_sizes,
    sim_tree,
    topology_accuracy,
)
from sal.qa.figure import QAFigure, QATable, latex_integer
from sal.qa.runner import (
    Option,
    ParamsArgument,
    figure_main,
    table_main,
)
from sal.sim.params import SimulationParams

from tests._fixtures import FIXTURES_DIR

#: The fixtures the rows read, by the path each script's test module named.
TREE_JC_STRESS = FIXTURES_DIR / "tree_jc/stress.yaml"
TREE_JC_RELEASE = FIXTURES_DIR / "tree_jc/release.yaml"
TREE_SEARCH_CI = FIXTURES_DIR / "tree_search/ci.yaml"
POTTS_CHAIN = FIXTURES_DIR / "potts_chain/ci.yaml"
HMM = FIXTURES_DIR / "hmm/ci.yaml"
SIZES_FIXTURES = [
    FIXTURES_DIR / name
    for name in ("tree_jc/stress.yaml", "tree_jc/ci.yaml", "tree_jc/release.yaml")
]

# A 5-taxon tree (15 topologies against 105) at a third of the sites: cheap
# renders. Nothing is asserted against its truth; it is read through the
# registry at a problem's tier (issue #863).
_SMALL_PARAMS = """
model: jukes-cantor
oracle: enumeration
seed: 20260906
n_sites: 400
tolerance: 0.01
n_states: 4
pi: [0.25, 0.25, 0.25, 0.25]
tau:
  name: root
  children:
    - name: A
      branch_length: 0.12
    - name: B
      branch_length: 0.28
    - name: ancestor_CDE
      branch_length: 0.06
      children:
        - name: C
          branch_length: 0.21
        - name: ancestor_DE
          branch_length: 0.09
          children:
            - name: D
              branch_length: 0.07
            - name: E
              branch_length: 0.33
"""


def _small_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "tree_search" / "ci.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_SMALL_PARAMS)
    return path


def _simulation(path: Path) -> SimulationParams:
    return load_params(path, SimulationParams)


@dataclass(frozen=True)
class Script:
    """One QA script's `main`, the arguments it is run with, and its caption."""

    main: Callable[[list[str] | None], QAFigure | QATable]
    argv: Callable[[Path], list[str]]
    #: What the caption must contain.
    needles: Callable[[], tuple[str, ...]] = tuple
    #: The whole caption, where the script's `build_caption` states it.
    caption: Callable[[], str] | None = None
    #: ``(module, attribute, value)``: a sweep cut to per-PR size.
    patches: tuple[tuple[ModuleType, str, object], ...] = ()


def _params(path: Path) -> Callable[[Path], list[str]]:
    return lambda out: ["--params", str(path), "--output-dir", str(out)]


def _potts_and_hmm(out: Path) -> list[str]:
    return [
        *("--potts-params", str(POTTS_CHAIN), "--hmm-params", str(HMM)),
        *("--output-dir", str(out)),
    ]


def _sizes(out: Path) -> list[str]:
    argv = [argument for path in SIZES_FIXTURES for argument in ("--params", str(path))]
    return [*argv, "--output-dir", str(out)]


SCRIPTS: dict[str, Script] = {
    "backend_agreement": Script(backend_agreement.main, _params(TREE_JC_STRESS)),
    "opt_branch_recovery": Script(
        opt_branch_recovery.main,
        lambda out: [
            *("--unrooted-params", str(FIXTURES_DIR / "tree_jc/ci.yaml")),
            *("--rooted-params", str(TREE_JC_RELEASE), "--output-dir", str(out)),
        ],
    ),
    # The real sweep refits both models dozens of times and belongs to the
    # document build; the sizes are patched down so the wiring still runs.
    "opt_coverage": Script(
        opt_coverage.main,
        _potts_and_hmm,
        patches=(
            (opt_coverage, "POTTS_SIZES", ((50, 2), (100, 2))),
            (opt_coverage, "HMM_SIZES", ((600, 1), (900, 1))),
        ),
    ),
    "opt_model_recovery": Script(opt_model_recovery.main, _params(TREE_JC_RELEASE)),
    "opt_recovery": Script(opt_recovery.main, _potts_and_hmm),
    "parsimony_zones": Script(
        parsimony_zones.main,
        _params(TREE_JC_RELEASE),
        lambda: ("10\\_395 unrooted topologies", "seed 20260904"),
    ),
    # At 5 taxa, so both surfaces, the sweep, the caption and the render run
    # per PR rather than only behind the release gate.
    "rl_reward_surface": Script(
        rl_reward_surface.main, _params(TREE_SEARCH_CI), lambda: ("correlation",)
    ),
    "search_trajectory": Script(
        search_trajectory.main,
        lambda out: _params(_small_fixture(out))(out),
        lambda: ("All 15 unrooted topologies",),
    ),
    "search_topologies": Script(
        search_topologies.main,
        lambda out: _params(_small_fixture(out))(out),
        lambda: ("log units",),
    ),
    "sim_example": Script(
        sim_example.main,
        lambda out: [*_params(TREE_JC_STRESS)(out), "--n-sites-shown", "10"],
        lambda: (
            str(_simulation(TREE_JC_STRESS).seed),
            latex_integer(_simulation(TREE_JC_STRESS).n_sites),
            "4-taxon",
            "10",
        ),
        lambda: sim_example.build_caption(
            _simulation(TREE_JC_STRESS), n_sites_shown=10
        ),
    ),
    "sim_problem_sizes": Script(
        sim_problem_sizes.main,
        _sizes,
        lambda: (str(len(SIZES_FIXTURES)),),
        lambda: sim_problem_sizes.build_caption(
            ["tree_jc/stress.yaml", "tree_jc/ci.yaml", "tree_jc/release.yaml"]
        ),
    ),
    "sim_tree": Script(
        sim_tree.main,
        _params(TREE_JC_RELEASE),
        lambda: (str(_simulation(TREE_JC_RELEASE).seed), "8 taxa", "Jukes-Cantor"),
        lambda: sim_tree.build_caption(_simulation(TREE_JC_RELEASE)),
    ),
    # At 5 taxa and a two-by-two sweep (issue #154): at its committed size
    # this ran 48 searches, 27.7 s. The caption reports the sweep that ran,
    # not the module's defaults, which is what makes the cut safe to assert.
    "topology_accuracy": Script(
        topology_accuracy.main,
        _params(TREE_SEARCH_CI),
        lambda: (
            "Robinson-Foulds",
            str(topology_accuracy.REQUIREMENT),
            "2",
            str(min(topology_accuracy.SITE_COUNTS)),
        ),
        patches=(
            (
                topology_accuracy,
                "SITE_COUNTS",
                (min(topology_accuracy.SITE_COUNTS), 250),
            ),
            (topology_accuracy, "REPLICATES", 2),
        ),
    ),
}


def _one_line(path: Path) -> str:
    """Load a file as its stripped contents."""
    return path.read_text().strip()


def _write_params(tmp_path: Path, name: str, text: str) -> Path:
    """Write a stand-in parameters file and return its path."""
    path = tmp_path / name
    path.write_text(text)
    return path


@pytest.mark.smoke
def test_a_figure_is_closed_even_when_writing_it_fails(tmp_path: Path) -> None:
    # The reason the close lives in a `finally` in the runner rather than in
    # each builder. A caption carrying an unescaped LaTeX special is refused
    # by `check_latex_safe` inside the write, and a figure leaked on that path
    # accumulates across a build until matplotlib warns and memory grows.
    params_path = _write_params(tmp_path, "params.txt", "ignored")
    leaked: list[Figure] = []

    def build(_: str) -> tuple[Figure, str]:
        fig = plt.figure()
        leaked.append(fig)
        # `_` is exactly what check_latex_safe rejects.
        return fig, "a caption with an unescaped under_score"

    with pytest.raises(ValueError, match="unescaped LaTeX special"):
        figure_main(
            stem="unwritable",
            description=None,
            params=[ParamsArgument("params", _one_line)],
            build=build,
            argv=["--params", str(params_path), "--output-dir", str(tmp_path)],
        )

    assert len(leaked) == 1
    assert not plt.fignum_exists(leaked[0].number)


@pytest.mark.smoke
def test_repeated_parameters_reach_the_builder_in_the_order_given(
    tmp_path: Path,
) -> None:
    # The problem-sizes table is one row per fixture, and the row order is the
    # order the flags were given, so the runner must not reorder or dedupe.
    first = _write_params(tmp_path, "first.txt", "alpha")
    second = _write_params(tmp_path, "second.txt", "beta")
    third = _write_params(tmp_path, "third.txt", "alpha")

    def build(loaded: list[str]) -> tuple[str, str]:
        return " ".join(loaded), "caption"

    written = table_main(
        stem="ordered",
        description=None,
        params=[ParamsArgument("params", _one_line, repeated=True)],
        build=build,
        argv=[
            "--params",
            str(second),
            "--params",
            str(first),
            "--params",
            str(third),
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert written.table_path.read_text().strip() == "beta alpha alpha"


@pytest.mark.smoke
def test_parameters_reach_the_builder_in_declaration_order_not_argv_order(
    tmp_path: Path,
) -> None:
    # A builder takes its parameters positionally, so the declaration order is
    # its signature. If the runner passed them in the order the flags happened
    # to appear, two files through the same loader -- the rooted and unrooted
    # fixtures of `opt_branch_recovery` -- would silently swap.
    left = _write_params(tmp_path, "left.txt", "left-value")
    right = _write_params(tmp_path, "right.txt", "right-value")

    def build(first: str, second: str) -> tuple[str, str]:
        return f"{first}|{second}", "caption"

    written = table_main(
        stem="declaration_order",
        description=None,
        params=[
            ParamsArgument("first-params", _one_line),
            ParamsArgument("second-params", _one_line),
        ],
        build=build,
        argv=[
            "--second-params",
            str(right),
            "--first-params",
            str(left),
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert written.table_path.read_text().strip() == "left-value|right-value"


@pytest.mark.smoke
def test_an_absent_option_reaches_the_builder_as_its_default(
    tmp_path: Path,
) -> None:
    # `sim_example` keeps a `--n-sites-shown` default of 10, and the build
    # script never passes the flag, so the default is the value every
    # committed rendering of that figure was produced with.
    params_path = _write_params(tmp_path, "params.txt", "ignored")

    def build(_: str, n_sites_shown: int = -1) -> tuple[str, str]:
        return str(n_sites_shown), "caption"

    written = table_main(
        stem="defaulted",
        description=None,
        params=[ParamsArgument("params", _one_line)],
        build=build,
        options=[Option("n-sites-shown", int, 10)],
        argv=["--params", str(params_path), "--output-dir", str(tmp_path)],
    )

    assert written.table_path.read_text().strip() == "10"


@pytest.mark.smoke
def test_an_option_given_on_the_command_line_overrides_its_default(
    tmp_path: Path,
) -> None:
    params_path = _write_params(tmp_path, "params.txt", "ignored")

    def build(_: str, n_sites_shown: int = -1) -> tuple[str, str]:
        return str(n_sites_shown), "caption"

    written = table_main(
        stem="overridden",
        description=None,
        params=[ParamsArgument("params", _one_line)],
        build=build,
        options=[Option("n-sites-shown", int, 10)],
        argv=[
            "--params",
            str(params_path),
            "--output-dir",
            str(tmp_path),
            "--n-sites-shown",
            "3",
        ],
    )

    assert written.table_path.read_text().strip() == "3"


@pytest.mark.smoke
def test_a_missing_parameters_file_argument_is_refused(tmp_path: Path) -> None:
    # Every parameters file is required: a figure rendered from a default
    # fixture nobody named would carry a caption claiming provenance it was
    # not given.
    def build(_: str) -> tuple[str, str]:
        return "body", "caption"

    with pytest.raises(SystemExit):
        table_main(
            stem="incomplete",
            description=None,
            params=[ParamsArgument("params", _one_line)],
            build=build,
            argv=["--output-dir", str(tmp_path)],
        )


@pytest.mark.smoke
@pytest.mark.parametrize("name", sorted(SCRIPTS))
def test_main_writes_its_output_and_caption(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = SCRIPTS[name]
    needles, caption = script.needles(), script.caption
    for module, attribute, value in script.patches:
        monkeypatch.setattr(module, attribute, value)

    written = script.main(script.argv(tmp_path))

    output = (
        written.figure_path if isinstance(written, QAFigure) else written.table_path
    )
    assert output.is_file()
    assert output.stat().st_size > 0
    assert written.caption_path.read_text() == written.caption
    if caption is not None:
        assert written.caption == caption()
    for needle in needles:
        assert needle in written.caption, needle


@pytest.mark.smoke
@pytest.mark.parametrize("runner", ["figure_main", "table_main"])
def test_main_reads_sys_argv_when_no_argv_is_given(
    runner: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # One script per runner: `sim_tree` through `figure_main` and
    # `sim_problem_sizes` through `table_main`, each run with no argv.
    stem, output = {
        "figure_main": ("sim_tree", "sim_tree.pdf"),
        "table_main": ("sim_problem_sizes", "sim_problem_sizes.tex"),
    }[runner]
    script = SCRIPTS[stem]
    monkeypatch.setattr("sys.argv", [stem, *script.argv(tmp_path)])

    # `None` is each `main`'s default: argv is read from `sys.argv`.
    script.main(None)

    output_path = tmp_path / output
    caption_path = tmp_path / f"{stem}_caption.txt"
    assert output_path.is_file()
    assert script.caption is not None
    assert caption_path.read_text() == script.caption()
    # The runner reports what it wrote through the run logger (issue #311),
    # which writes to stderr; nothing goes to stdout.
    captured = capsys.readouterr()
    written = captured.out + captured.err
    assert str(output_path) in written
    assert str(caption_path) in written
