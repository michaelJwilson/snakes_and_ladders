"""Write the textbook's applicability tables from ``PROBLEMS.md`` (issue #358).

``PROBLEMS.md`` names, per problem, the code that simulates, evaluates, fits,
searches and learns on it. The textbook may name no code (``docs/CLAUDE.md``),
so the three tables it typesets -- algorithms against problems, oracles
against problems, and problems against the four method families -- carry the
algorithm or oracle each symbol *is*, and this module is the one place that
naming lives. A symbol the catalogue gains that
this module cannot name fails the generation rather than vanishing from the
tables, and the guard in ``tests/regression/docs/test_problems_tables.py``
holds the committed file to a regeneration, as ``CHECKS.md`` and the QA
figures are held.

What referees each method is read from the suite, never typed: the tests
``CHECKS.md`` lists (kind ``oracle`` or ``simulated_truth``) that import a
catalogue symbol are the tests that pin it, and their ``stress`` and
``release`` markers are the size tier they run at. A method pinned by the
simulated truth alone, or by neither kind, is marked, because there an oracle
is wanted and absent.

Infrastructure, not science: it reads a Markdown table, a dictionary and the
test tree, and knows nothing about what a Potts lattice is. Run::

    python infra/problems_tables.py --write   # regenerate
    python infra/problems_tables.py --check   # exit 1 if stale
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections.abc import Iterable
from pathlib import Path

import checks_ledger
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOGUE = REPO_ROOT / "PROBLEMS.md"
GENERATED = REPO_ROOT / "docs" / "tex" / "generated" / "problems_tables.tex"
#: The one hand-written input: a sentence per pairing on when the method wins.
METHOD_NOTES = REPO_ROOT / "docs" / "tex" / "generated" / "method_notes.yaml"
EXPERIMENTS = REPO_ROOT / "docs" / "experiments"

#: The prefix every code symbol in the catalogue carries. A backticked cell
#: without it is a path (a fixture, a notebook) and names no algorithm.
PACKAGE = "snakes_and_ladders."

ALGORITHM_COLUMNS = (
    "simulation",
    "exact evaluation",
    "message passing",
    "gradient fit",
    "expectation--maximization",
    "initializer",
    "hill climbing",
    "sampling and tempering",
    "descent and expansion",
    "exact ground state",
    "relaxation",
    "policy learning",
)

#: The four families the textbook's algorithm appendix groups by, in its
#: order. A problem is paired with a family when the catalogue names a symbol
#: of that family for it.
METHOD_FAMILIES = ("initializers", "samplers", "optimizers", "surrogates")

#: Which family each algorithm column belongs to. A column naming what is
#: *evaluated* rather than a method that moves --- simulation and exact
#: evaluation --- belongs to none and does not reach the third table. Message
#: passing does: on a code it is the decoder, which is where the appendix
#: groups it.
FAMILY_BY_COLUMN: dict[str, str] = {
    "initializer": "initializers",
    "sampling and tempering": "samplers",
    "gradient fit": "optimizers",
    "expectation--maximization": "optimizers",
    "hill climbing": "optimizers",
    "descent and expansion": "optimizers",
    "exact ground state": "optimizers",
    "message passing": "optimizers",
    "policy learning": "optimizers",
    "relaxation": "surrogates",
}

#: The symbols whose family is not the one their algorithm column implies. A
#: fitted surrogate is a surrogate wherever it is learned.
FAMILY_BY_SYMBOL: dict[str, str] = {"learn.surrogate.fit_surrogate": "surrogates"}

ORACLE_COLUMNS = (
    "enumeration or brute force",
    "closed form or published value",
    "independent algorithm",
    "planted truth",
)

#: The size tiers ``DEV.md`` defines, by the scheduling marker that selects
#: each; a test with neither marker runs at the CI tier.
TIERS = ("ci", "stress", "release")

#: How a cell is marked: pinned by an oracle; by the simulated truth only,
#: an oracle wanted; by neither significant kind, an oracle wanted.
ORACLE_MARK = r"$\checkmark$"
TRUTH_MARK = r"$\checkmark^{\dagger}$"
UNPINNED_MARK = r"$\circ$"

#: Every code symbol the catalogue may name, and the column it fills. A
#: symbol may fill one algorithm column, one oracle column, or both (an
#: independent algorithm is an oracle for the fit it referees). The mapping is
#: by the symbol's name under the package, so a module moving keeps its row.
ALGORITHMS: dict[str, str] = {
    "sim.simulate.simulate_alignment": "simulation",
    "sim.jc.jc_transition_probabilities": "simulation",
    "sim.gtr.gtr_rate_matrix": "simulation",
    "opt.potts.simulate_chains": "simulation",
    "sim.potts.simulate_potts": "simulation",
    "sim.graph.lattice_graph": "simulation",
    "sim.graph.erdos_renyi_graph": "simulation",
    "sim.canonical.frustrated_triangular_lattice": "simulation",
    "sim.canonical.planted_spin_glass": "simulation",
    "sim.hmm.simulate_sequences": "simulation",
    "emissions.CategoricalEmission": "simulation",
    "emissions.GaussianEmission": "simulation",
    "emissions.PoissonEmission": "simulation",
    "emissions.BinomialEmission": "simulation",
    "emissions.BetaBinomialEmission": "simulation",
    "emissions.NegativeBinomialEmission": "simulation",
    "sim.spatio_sequential.simulate_spatio_sequential": "simulation",
    "sim.spatio_sequential.canonical_spatio_sequential": "simulation",
    "sim.mixture.simulate_mixture": "simulation",
    "sim.ldpc.gallager_code": "simulation",
    "sim.ldpc.BinarySymmetricChannel": "simulation",
    "sim.ldpc.BinaryErasureChannel": "simulation",
    "sim.ldpc.BinaryInputGaussianChannel": "simulation",
    "sim.ldpc.encode": "simulation",
    "sim.ldpc.all_zero_transmission": "simulation",
    "likelihood.ldpc.decode": "message passing",
    "sim.factor_graph.from_parity_check": "message passing",
    "likelihood.pruning.log_likelihood": "exact evaluation",
    "likelihood.pruning_torch.log_likelihood": "exact evaluation",
    "likelihood.pruning_rust.log_likelihood": "exact evaluation",
    "likelihood.parsimony.fitch_score": "exact evaluation",
    "likelihood.parsimony.sankoff_score": "exact evaluation",
    "opt.potts.log_partition": "exact evaluation",
    "opt.hmm.forward_log_likelihood": "exact evaluation",
    "likelihood.forward_backward.forward_backward": "exact evaluation",
    "likelihood.message_passing.max_product": "exact evaluation",
    "likelihood.spatio_sequential.class_posteriors": "exact evaluation",
    "likelihood.spatio_sequential.log_evidence_by_forward": "exact evaluation",
    "opt.mixture.mixture_log_likelihood": "exact evaluation",
    "opt.testfunctions.Rosenbrock": "exact evaluation",
    "opt.testfunctions.Rastrigin": "exact evaluation",
    "opt.testfunctions.Himmelblau": "exact evaluation",
    "likelihood.belief_propagation.belief_propagation": "message passing",
    "likelihood.message_passing.sum_product": "message passing",
    "sim.factor_graph.FactorGraph": "message passing",
    "sim.spatio_sequential.coupled_factor_graph": "message passing",
    "likelihood.objective.BranchLengthObjective": "gradient fit",
    "likelihood.objective.SubstitutionModelObjective": "gradient fit",
    "opt.potts.PottsObjective": "gradient fit",
    "opt.potts.PottsLatticeObjective": "gradient fit",
    "opt.hmm.HmmObjective": "gradient fit",
    "opt.mixture.GaussianMixtureObjective": "gradient fit",
    "opt.fit.fit": "gradient fit",
    "opt.hmm.baum_welch_family": "expectation--maximization",
    "opt.mixture.expectation_maximization": "expectation--maximization",
    "search.spatio_sequential.fit_spatio_sequential": "expectation--maximization",
    "opt.mixture.KMeansPlusPlus": "initializer",
    "search.spatio_sequential.graph_burn_in": "initializer",
    "opt.initialize.RandomRestart": "initializer",
    # #364's tree starts: a distance or a spectral estimate, then a joining.
    "search.initialize.FromDistances": "initializer",
    "search.initialize.FromHadamard": "initializer",
    "search.neighbor_joining.neighbor_joining": "initializer",
    "likelihood.distance.distance_matrix": "initializer",
    "likelihood.distance.log_det_distance": "initializer",
    "likelihood.hadamard.hadamard_conjugation": "initializer",
    "search.infer.infer": "hill climbing",
    "search.infer.Model": "hill climbing",
    "search.infer.parsimony_search": "hill climbing",
    "search.topology.nni_neighbours": "hill climbing",
    "search.topology.spr_neighbours": "hill climbing",
    "search.gibbs.anneal_topology": "sampling and tempering",
    "search.potts_mcmc.sample_potts": "sampling and tempering",
    "search.potts_mcmc.anneal_potts": "sampling and tempering",
    "search.potts_mcmc.parallel_tempering": "sampling and tempering",
    "search.gibbs.sample_factor_graph": "sampling and tempering",
    "search.gibbs.chain_block_sweep": "sampling and tempering",
    "opt.hmc.anneal": "sampling and tempering",
    "search.alpha_expansion.alpha_expansion": "descent and expansion",
    "search.alpha_expansion.iterated_conditional_modes": "descent and expansion",
    "search.maxflow.ising_ground_state": "exact ground state",
    "search.max_cut.goemans_williamson": "relaxation",
    "learn.relaxed.RelaxedPotts": "relaxation",
    "learn.relaxed.RelaxedHmmPath": "relaxation",
    "search.rl.TopologyEnvironment": "policy learning",
    "learn.potts.PottsLandscape": "policy learning",
    "learn.hmm.StatePathLandscape": "policy learning",
    "learn.surrogate.fit_surrogate": "policy learning",
}

ORACLES: dict[str, str] = {
    "likelihood.brute_force.brute_force_log_likelihood": "enumeration or brute force",
    "likelihood.parsimony.brute_force_parsimony_score": "enumeration or brute force",
    "search.topology.enumerate_topologies": "enumeration or brute force",
    "likelihood.potts.enumerate_potts": "enumeration or brute force",
    "learn.potts.enumerate_configurations": "enumeration or brute force",
    "likelihood.hmm_paths.enumerate_hidden_paths": "enumeration or brute force",
    "learn.hmm.enumerate_paths": "enumeration or brute force",
    "likelihood.spatio_sequential.enumerate_spatio_sequential": (
        "enumeration or brute force"
    ),
    "sim.canonical.minimum_frustrated_edges": "closed form or published value",
    "likelihood.ldpc.erasure_threshold": "closed form or published value",
    "likelihood.ldpc.exact_decoding": "enumeration or brute force",
    "opt.testfunctions.HIMMELBLAU_MINIMA": "closed form or published value",
    "likelihood.potts.strip_log_partition": "independent algorithm",
    "opt.hmm.baum_welch_family": "independent algorithm",
    "opt.mixture.expectation_maximization": "independent algorithm",
    "opt.mixture.optimal_clustering_cost": "independent algorithm",
    "sim.canonical.PlantedSpinGlass": "planted truth",
}

_SYMBOL = re.compile(r"`([^`]+)`")


class UnnamedSymbolError(ValueError):
    """A catalogue symbol this module cannot name as an algorithm or an oracle."""


def rows(catalogue: Path = CATALOGUE) -> list[tuple[str, list[str]]]:
    """``(problem, symbols)`` per table row of the catalogue, in order.

    Only backticked names under the package count; a path names a fixture or
    a notebook and fills no column.
    """
    found: list[tuple[str, list[str]]] = []
    for line in catalogue.read_text().splitlines():
        if not line.startswith("| ") or line.startswith("| Problem") or "---" in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        problem, rest = cells[0], " ".join(cells[1:])
        symbols = [
            name[len(PACKAGE) :]
            for name in _SYMBOL.findall(rest)
            if name.startswith(PACKAGE)
        ]
        found.append((problem, symbols))
    return found


def unnamed(symbols: list[str]) -> list[str]:
    """The symbols neither table names, sorted."""
    return sorted({s for s in symbols if s not in ALGORITHMS and s not in ORACLES})


# --- what the suite says referees each symbol ---------------------------------


def _bindings(tree: ast.Module) -> tuple[dict[str, str], dict[str, str]]:
    """Names a test file binds to package symbols, and to package modules.

    ``from snakes_and_ladders.a.b import c as d`` binds ``d`` to the symbol
    ``a.b.c``; ``from snakes_and_ladders.a import b`` and
    ``import snakes_and_ladders.a.b`` bind a name to the module ``a.b``, whose
    attributes are then symbols.
    """
    symbols: dict[str, str] = {}
    modules: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            if not node.module.startswith(PACKAGE.rstrip(".")):
                continue
            base = node.module[len(PACKAGE) :]
            for alias in node.names:
                bound = alias.asname or alias.name
                target = f"{base}.{alias.name}" if base else alias.name
                if base and base.count(".") == 0 and "." not in alias.name:
                    # `from snakes_and_ladders.likelihood import pruning`: a
                    # module of the package, or a symbol of a top-level module.
                    modules[bound] = target
                symbols[bound] = target
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(PACKAGE):
                    modules[alias.asname or alias.name] = alias.name[len(PACKAGE) :]
    return symbols, modules


def _referenced(
    function: ast.FunctionDef, symbols: dict[str, str], modules: dict[str, str]
) -> tuple[set[str], set[str]]:
    """The package symbols one function's body names, and the local names it calls.

    The second set is the module-level helpers a test reaches a symbol
    through, so a fixture built in a helper still counts for the test.
    """
    found: set[str] = set()
    called: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Name):
            if node.id in symbols:
                found.add(symbols[node.id])
            else:
                called.add(node.id)
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in modules
        ):
            found.add(f"{modules[node.value.id]}.{node.attr}")
    return found, called


def _references_by_function(tree: ast.Module) -> dict[str, set[str]]:
    """``function name -> symbols`` for every module-level function, helpers followed.

    A test that names a helper inherits the helper's symbols, transitively,
    so the reading does not depend on whether a fixture is built inline.
    """
    symbols, modules = _bindings(tree)
    direct: dict[str, set[str]] = {}
    calls: dict[str, set[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            direct[node.name], calls[node.name] = _referenced(node, symbols, modules)
    closed = {name: set(found) for name, found in direct.items()}
    changed = True
    while changed:
        changed = False
        for name, callees in calls.items():
            for callee in callees & closed.keys():
                if not closed[callee] <= closed[name]:
                    closed[name] |= closed[callee]
                    changed = True
    return closed


def _tier(markers: set[str]) -> str:
    if "release" in markers:
        return "release"
    if "stress" in markers:
        return "stress"
    return "ci"


def referees(
    tests: Path = checks_ledger.TESTS,
) -> dict[str, set[tuple[str, str]]]:
    """``symbol -> {(kind, tier)}`` over the significant tests that name it.

    The significant tests are exactly the rows of ``CHECKS.md``; each names a
    symbol when its body uses a name the file bound to it, so the reading is
    the file's own imports and not a guess from a spelling.
    """
    significant = {
        (file, name): kind.split(", ") for file, name, kind, _ in checks_ledger.rows()
    }
    found: dict[str, set[tuple[str, str]]] = {}
    for path in sorted(tests.rglob("test_*.py")):
        tree = ast.parse(path.read_text())
        references = _references_by_function(tree)
        relative = str(path.relative_to(REPO_ROOT))
        for node in tree.body:
            if not (
                isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
            ):
                continue
            kinds = significant.get((relative, node.name))
            if kinds is None:
                continue
            tier = _tier(checks_ledger._markers(node))
            for symbol in references[node.name]:
                for kind in kinds:
                    found.setdefault(symbol, set()).add((kind, tier))
    return found


def _mark(pins: Iterable[tuple[str, str]]) -> str:
    kinds = {kind for kind, _ in pins}
    if "oracle" in kinds:
        return ORACLE_MARK
    if "simulated_truth" in kinds:
        return TRUTH_MARK
    return UNPINNED_MARK


def _referee_at(pins: Iterable[tuple[str, str]], tier: str) -> str:
    kinds = {kind for kind, at in pins if at == tier}
    if "oracle" in kinds:
        return "oracle"
    if "simulated_truth" in kinds:
        return r"truth$^{\dagger}$"
    return "--"


# --- rendering -----------------------------------------------------------------


def _tex_text(text: str) -> str:
    """Escape a problem name for LaTeX text mode."""
    return (
        text.replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("\u2013", "--")
        .replace("\u2014", "---")
    )


def _table(
    columns: tuple[str, ...],
    body: list[tuple[str, list[str]]],
    label: str,
    caption: str,
) -> list[str]:
    """One ``table`` environment: problems down, ``columns`` across, rotated."""
    header = " & ".join(
        rf"\rotatebox{{90}}{{\footnotesize {column}}}" for column in columns
    )
    # Scaled to the text width: twelve rotated columns beside a full problem
    # name overflow an A4 column at the body size.
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{l" + "c" * len(columns) + "}",
        r"    \toprule",
        f"    & {header} \\\\",
        r"    \midrule",
    ]
    for problem, cells in body:
        lines.append(f"    {_tex_text(problem)} & {' & '.join(cells)} \\\\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}}",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"\end{table}",
    ]
    return lines


class MissingNoteError(ValueError):
    """A problem-and-family pairing the notes file does not carry."""


def family(symbol: str) -> str | None:
    """The method family ``symbol`` belongs to, or ``None`` if it is not a method."""
    if symbol in FAMILY_BY_SYMBOL:
        return FAMILY_BY_SYMBOL[symbol]
    return FAMILY_BY_COLUMN.get(ALGORITHMS.get(symbol, ""))


def notes(path: Path = METHOD_NOTES) -> dict[str, dict[str, str]]:
    """``problem -> family -> note``, from the committed YAML."""
    loaded = yaml.safe_load(path.read_text())
    return {} if loaded is None else loaded


_EXPERIMENT = re.compile(r"experiment (\d+)")


def cited_experiments(note: str) -> list[str]:
    """The experiment ids one note cites, zero-padded as the filenames are."""
    return [f"{int(number):03d}" for number in _EXPERIMENT.findall(note)]


def missing_experiments(
    note_map: dict[str, dict[str, str]] | None = None,
    experiments: Path = EXPERIMENTS,
) -> list[str]:
    """Every experiment a note cites that has no file, sorted.

    A note is the only hand-written cell of the tables, so the one thing it
    can get wrong on its own is the citation; a pull request that renumbers
    or retracts an experiment must not leave a table pointing at it.
    """
    present = {path.name[:3] for path in experiments.glob("[0-9][0-9][0-9]-*.md")}
    wanted = {
        identifier
        for families in (notes() if note_map is None else note_map).values()
        for note in families.values()
        for identifier in cited_experiments(note)
    }
    return sorted(wanted - present)


def method_cells(
    catalogue: Path = CATALOGUE, note_map: dict[str, dict[str, str]] | None = None
) -> list[tuple[str, str, str, str, str]]:
    """``(problem, family, tier, referee, note)`` for every problem and family.

    Every pairing appears: one the suite names with no test of either
    significant kind is ``untested``, and one the catalogue has no symbol for
    is ``--``. Omitting either would make the table read as though the
    question had not been asked.

    Raises
    ------
    MissingNoteError
        If a pairing the catalogue carries has no note.
    """
    known = notes() if note_map is None else note_map
    pins = referees()
    found: list[tuple[str, str, str, str, str]] = []
    for problem, symbols in rows(catalogue):
        for name in METHOD_FAMILIES:
            here = [symbol for symbol in symbols if family(symbol) == name]
            note = known.get(problem, {}).get(name, "")
            if here and not note:
                msg = (
                    f"{problem!r} has {name} in PROBLEMS.md and no note in "
                    f"{METHOD_NOTES.name}; add one saying when the family wins here"
                )
                raise MissingNoteError(msg)
            pairs = [pin for symbol in here for pin in pins.get(symbol, ())]
            kinds = {kind for kind, _ in pairs}
            tiers = [tier for tier in TIERS if any(at == tier for _, at in pairs)]
            if "oracle" in kinds:
                referee = "oracle"
            elif "simulated_truth" in kinds:
                referee = r"truth$^{\dagger}$"
            elif here:
                referee = "untested"
            else:
                referee = "--"
            found.append(
                (problem, name, tiers[0] if tiers else "--", referee, note.strip())
            )
    return found


def _method_table(cells: list[tuple[str, str, str, str, str]]) -> list[str]:
    """The third table, one part per method family: problems down, what pins each.

    Four parts rather than one grid, for the reason the appendix groups the
    algorithms the same way: a reader compares problems within a family, and
    a single grid of eleven problems by four sentences does not fit a page.
    """
    lines: list[str] = []
    for name in METHOD_FAMILIES:
        here = [cell for cell in cells if cell[1] == name]
        lines += [
            r"\begin{table}[htbp]",
            r"  \centering",
            r"  \footnotesize",
            r"  \begin{tabular}{@{}p{0.21\textwidth}llp{0.44\textwidth}@{}}",
            r"    \toprule",
            r"    Problem & Tier & Referee & When it wins, and when it does not \\",
            r"    \midrule",
        ]
        for problem, _, tier, referee, note in here:
            lines.append(
                f"    {_tex_text(problem)} & {tier} & {referee} & "
                f"{_tex_text(note) if note else '--'} \\\\"
            )
        lines += [
            r"    \bottomrule",
            r"  \end{tabular}",
            f"  \\caption{{The {name} each problem carries: the size tier at "
            "which the pairing is validated, the kind of referee that "
            "validates it, and what the pairing is worth. Tier and referee are "
            "read from the suite, as in the two tables above --- "
            "\\emph{oracle} a test pinned to an independent exact answer, "
            "truth$^{\\dagger}$ a test pinned to the simulated truth only, "
            "\\emph{untested} a family the problem carries that no test of "
            "either kind names, and -- a family it carries no method of. The "
            "last column is the one hand-written part of these tables, and "
            "names the experiment that measured the comparison where one "
            f"did.}}",
            f"  \\label{{tab:methods-{name}}}",
            r"\end{table}",
            "",
        ]
    return lines[:-1]


def truth_only(catalogue: Path = CATALOGUE) -> list[tuple[str, str]]:
    """``(problem, algorithm column)`` cells pinned by the simulated truth or by neither kind.

    The list ``STATUS.md`` reports and the follow-up tickets name an oracle
    for.
    """
    pins = referees()
    found: list[tuple[str, str]] = []
    for problem, symbols in rows(catalogue):
        for column in ALGORITHM_COLUMNS:
            here = [s for s in symbols if ALGORITHMS.get(s) == column]
            if not here:
                continue
            mark = _mark(pin for s in here for pin in pins.get(s, ()))
            if mark != ORACLE_MARK:
                found.append((problem, column))
    return found


def render(catalogue: Path = CATALOGUE) -> str:
    """The generated LaTeX, from the catalogue and the suite.

    Raises
    ------
    UnnamedSymbolError
        If the catalogue names a symbol neither dictionary can place. The
        table cannot be silently narrower than the catalogue.
    """
    missing: list[str] = []
    for _, symbols in rows(catalogue):
        missing += unnamed(symbols)
    if missing:
        msg = (
            f"PROBLEMS.md names {sorted(set(missing))}, which infra/problems_tables.py "
            "cannot place in either table; add each to ALGORITHMS or ORACLES"
        )
        raise UnnamedSymbolError(msg)

    dangling = missing_experiments()
    if dangling:
        msg = (
            f"{METHOD_NOTES.name} cites experiments with no file under "
            f"docs/experiments/: {dangling}"
        )
        raise UnnamedSymbolError(msg)

    pins = referees()
    algorithms: list[tuple[str, list[str]]] = []
    oracles: list[tuple[str, list[str]]] = []
    for problem, symbols in rows(catalogue):
        cells = []
        for column in ALGORITHM_COLUMNS:
            here = [s for s in symbols if ALGORITHMS.get(s) == column]
            cells.append(
                _mark(pin for s in here for pin in pins.get(s, ())) if here else "--"
            )
        algorithms.append((problem, cells))
        kinds = {ORACLES[s] for s in symbols if s in ORACLES}
        every_pin = [pin for s in symbols for pin in pins.get(s, ())]
        oracles.append(
            (
                problem,
                [ORACLE_MARK if column in kinds else "--" for column in ORACLE_COLUMNS]
                + [_referee_at(every_pin, tier) for tier in TIERS],
            )
        )

    lines = [
        "% Generated from PROBLEMS.md and the regression suite by the",
        "% problems-tables generator under infra/. Do not edit; regenerate",
        "% with its --write flag.",
        *_table(
            ALGORITHM_COLUMNS,
            algorithms,
            "tab:algorithms-problems",
            "The algorithms that apply to each problem, read from the problem "
            "catalogue, and what referees each: $\\checkmark$ a test pinned to "
            "an oracle, $\\checkmark^{\\dagger}$ a test pinned to the simulated "
            "truth only, $\\circ$ neither kind of test, and -- not applicable "
            "or not built. A dagger or a circle is a cell where an oracle is "
            "wanted and absent.",
        ),
        "",
        *_table(
            ORACLE_COLUMNS + tuple(f"referee, {tier} tier" for tier in TIERS),
            oracles,
            "tab:oracles-problems",
            "The oracle each problem's evaluators and searches are pinned to, "
            "and, per size tier, what referees the tests that pin them: an "
            "oracle, the simulated truth only (dagger: an oracle is wanted), or "
            "no test of either kind at that tier (--). A planted truth is an "
            "upper bound on the optimum rather than the optimum.",
        ),
        "",
        *_method_table(method_cells(catalogue)),
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Regenerate or check the tables. Returns 1 from ``--check`` when stale."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="regenerate the tables")
    mode.add_argument("--check", action="store_true", help="exit 1 if stale")
    arguments = parser.parse_args(argv)
    text = render()
    if arguments.write:
        GENERATED.parent.mkdir(parents=True, exist_ok=True)
        GENERATED.write_text(text)
        print(f"wrote {GENERATED}")
        return 0
    if not GENERATED.is_file() or GENERATED.read_text() != text:
        print(
            f"{GENERATED} is stale; regenerate with: python infra/problems_tables.py --write",
            file=sys.stderr,
        )
        return 1
    print(f"{GENERATED} is current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
