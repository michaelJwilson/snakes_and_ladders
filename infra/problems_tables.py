"""Write the textbook's applicability tables from ``PROBLEMS.md`` (issue #358).

``PROBLEMS.md`` names, per problem, the code that simulates, evaluates, fits,
searches and learns on it. The textbook may name no code (``docs/CLAUDE.md``),
so the three tables it typesets -- algorithms against problems, oracles
against problems, and problems against the four method families -- carry the
algorithm or oracle each symbol *is*, and this module is the one place that
naming lives. A symbol the catalogue gains that
this module cannot name fails the generation rather than vanishing from the
tables, and the guard in ``tests/regression/docs/test_problems_tables.py``
holds what it writes to the properties the textbook needs. The file itself is
not committed (issue #425): ``infra/ledgers.sh`` writes it in the document
build, at the release gate and in CI.

What referees each method is read from the suite, never typed: the tests
``CHECKS.md`` lists (kind ``oracle`` or ``simulated_truth``) that import a
catalogue symbol are the tests that pin it, and their ``stress`` and
``release`` markers are the size tier they run at. A method pinned by the
simulated truth alone, or by neither kind, is marked, because there an oracle
is wanted and absent.

Infrastructure, not science: it reads a Markdown table, a dictionary and the
test tree, and knows nothing about what a Potts lattice is. Run::

    python infra/problems_tables.py --write   # regenerate
    python infra/problems_tables.py --check   # exit 1 if the tree's copy is stale
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections.abc import Iterable
from functools import cache
from pathlib import Path

import checks_ledger
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOGUE = REPO_ROOT / "PROBLEMS.md"
GENERATED = REPO_ROOT / "docs" / "tex" / "generated" / "problems_tables.tex"
#: The one hand-written input: a sentence per pairing on when the method wins.
METHOD_NOTES = REPO_ROOT / "docs" / "tex" / "method_notes.yaml"
#: The block of that file holding the reason each untested pairing is
#: untested, keyed ``"<fixture> / <family>"``.
UNTESTED_KEY = "untested pairings"
EXPERIMENTS = REPO_ROOT / "docs" / "experiments"
#: The fixture registry's files, read as data: which problems exist, and at
#: which tiers. Read from the tree rather than imported, because nothing here
#: may hold an application reference (``infra/CLAUDE.md``).
FIXTURES = REPO_ROOT / "tests" / "regression" / "fixtures"

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

#: How a fixture is named in a test: through one of the registry's entry
#: points that take a problem and a tier, as a problem parameterized over
#: every tier it declares, or as the file's path. All three are read, because
#: a pairing must not read as untested for the way its test spells the fixture
#: (issue #382).
_REGISTRY_CALLS = ("fixture", "path_of", "baseline_path", "baseline")
_FIXTURE_CALL = re.compile(
    r"(?<!at_)(?:"
    + "|".join(_REGISTRY_CALLS)
    + r')\(\s*"([a-z_0-9]+)"\s*,\s*"([a-z]+)"'
)
#: A ``StrEnum`` member *is* its string, so ``Scale.CI`` names the CI tier as
#: plainly as ``"ci"`` does; 19 tests in the suite spell it that way. Resolved
#: only when the member's lowercased name is a tier ``DEV.md`` declares, so an
#: unrelated enum is left alone rather than guessed at.
_SCALE_MEMBER = re.compile(r"\bScale\.([A-Z_]+)\b")
_AT_FIXTURE_CALL = re.compile(r'at_fixture\(\s*"[a-z_0-9]+"\s*,\s*"([a-z_0-9]+)"')
#: A path is preceded by a quote or a separator, never by a backslash: the
#: escaped ``tree\_jc/release.yaml`` a LaTeX assertion carries would
#: otherwise read as a problem named ``_jc``.
_FIXTURE_PATH = re.compile(r"(?<![a-z_0-9\\])([a-z_0-9]+)/([a-z]+)\.yaml")

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
    "sim.graph.triangular_lattice_graph": "simulation",
    "sim.potts.spatio_only_field": "simulation",
    "sim.canonical.frustrated_triangular_lattice": "simulation",
    "sim.canonical.planted_spin_glass": "simulation",
    "sim.hmm.simulate_sequences": "simulation",
    "emissions.CategoricalEmission": "simulation",
    "emissions.GaussianEmission": "simulation",
    "emissions.PoissonEmission": "simulation",
    "emissions.BinomialEmission": "simulation",
    "emissions.BetaBinomialEmission": "simulation",
    "emissions.NegativeBinomialEmission": "simulation",
    "emissions.CountPairEmission": "simulation",
    "sim.spatio_sequential.simulate_spatio_sequential": "simulation",
    "sim.count_pairs.simulate_count_pairs": "simulation",
    "sim.count_pairs_rust.simulate_count_pairs": "simulation",
    "sim.count_pairs.IndependentCountPair": "simulation",
    "sim.count_pairs.coarsen": "simulation",
    "sim.spatio_sequential.canonical_spatio_sequential": "simulation",
    "sim.mixture.simulate_mixture": "simulation",
    "sim.emission_mixture.simulate_emission_mixture": "simulation",
    "sim.ldpc.gallager_code": "simulation",
    "sim.ldpc.bicycle_code": "simulation",
    "sim.css.CssCode": "simulation",
    "sim.css.sample_x_error": "simulation",
    "sim.ldpc.BinarySymmetricChannel": "simulation",
    "sim.ldpc.BinaryErasureChannel": "simulation",
    "sim.ldpc.BinaryInputGaussianChannel": "simulation",
    "sim.ldpc.encode": "simulation",
    "sim.ldpc.all_zero_transmission": "simulation",
    "sim.convolutional.recursive_systematic_trellis": "simulation",
    "sim.convolutional.turbo_code": "simulation",
    "sim.convolutional.turbo_encode": "simulation",
    "sim.convolutional.parity_check": "simulation",
    "sim.ldpc.ParityCheck": "simulation",
    "likelihood.ldpc.decode": "message passing",
    "likelihood.css.decode_syndrome": "message passing",
    "sim.factor_graph.from_parity_check": "message passing",
    "likelihood.turbo.decode_turbo": "message passing",
    "sim.factor_graph.from_trellis": "message passing",
    "likelihood.convolutional.bcjr": "exact evaluation",
    "likelihood.convolutional.viterbi": "exact evaluation",
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
    "likelihood.spatio_sequential_rust.class_posteriors": "exact evaluation",
    "likelihood.spatio_sequential_rust.external_field": "exact evaluation",
    "likelihood.spatio_sequential.log_evidence_by_forward": "exact evaluation",
    "opt.mixture.mixture_log_likelihood": "exact evaluation",
    "opt.mixture.responsibilities": "exact evaluation",
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
    "opt.emission_mixture.expectation_maximization": "expectation--maximization",
    "search.spatio_sequential.fit_spatio_sequential": "expectation--maximization",
    "opt.mixture.KMeansPlusPlus": "initializer",
    "opt.emission_mixture.plus_plus_start": "initializer",
    "opt.emission_mixture.uniform_start": "initializer",
    "opt.mixture.emission_mixture_plus_plus": "initializer",
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
    "sandbox.tropical.optimize": "relaxation",
    "sandbox.tropical.quartet_table": "relaxation",
    "search.rl.TopologyEnvironment": "policy learning",
    "learn.potts.PottsEnvironment": "policy learning",
    "learn.hmm.HmmEnvironment": "policy learning",
    "learn.surrogate.fit_surrogate": "policy learning",
}

ORACLES: dict[str, str] = {
    "likelihood.brute_force.brute_force_log_likelihood": "enumeration or brute force",
    "likelihood.parsimony.brute_force_parsimony_score": "enumeration or brute force",
    "search.topology.enumerate_topologies": "enumeration or brute force",
    "likelihood.potts.enumerate_potts": "enumeration or brute force",
    "learn.potts.enumerate_configurations": "enumeration or brute force",
    "likelihood.hmm_paths.enumerate_hidden_paths": "enumeration or brute force",
    "likelihood.mixture_assignments.enumerate_mixture_assignments": (
        "enumeration or brute force"
    ),
    "learn.hmm.enumerate_paths": "enumeration or brute force",
    "likelihood.spatio_sequential.enumerate_spatio_sequential": (
        "enumeration or brute force"
    ),
    "sim.canonical.minimum_frustrated_edges": "closed form or published value",
    "likelihood.ldpc.erasure_threshold": "closed form or published value",
    "likelihood.ldpc.exact_decoding": "enumeration or brute force",
    "likelihood.css.error_cosets": "enumeration or brute force",
    "likelihood.convolutional.exact_bitwise_posterior": "enumeration or brute force",
    "likelihood.turbo.exact_turbo_posterior": "enumeration or brute force",
    "likelihood.turbo.uncoded_bit_error_rate": "closed form or published value",
    "opt.testfunctions.HIMMELBLAU_MINIMA": "closed form or published value",
    "likelihood.forward_backward.forward_backward": "independent algorithm",
    "likelihood.potts.strip_log_partition": "independent algorithm",
    "opt.hmm.baum_welch_family": "independent algorithm",
    "opt.mixture.expectation_maximization": "independent algorithm",
    "opt.mixture.optimal_clustering_cost": "independent algorithm",
    "opt.emission_mixture.enumerated_posterior": "enumeration or brute force",
    "sim.canonical.PlantedSpinGlass": "planted truth",
}

_SYMBOL = re.compile(r"`([^`]+)`")
#: The two cells the tables read. A key is a bare fixture directory; a
#: defining name carries a dot and no package prefix, since the catalogue
#: names code under the package and says so once rather than per cell.
_KEY_CELL = re.compile(r"`([a-z_0-9]+)`")
_DEFINES_CELL = re.compile(r"`([a-z_]+\.[A-Za-z0-9_.]+)`")
#: A fixture named by a catalogue row: the problem, then the tier.
_CATALOGUE_FIXTURE = re.compile(
    r"tests/regression/fixtures/([a-z_0-9]+)/([a-z]+)\.yaml"
)


class UnnamedSymbolError(ValueError):
    """A catalogue symbol this module cannot name as an algorithm or an oracle."""


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


def _constants(source: str) -> dict[str, str]:
    """The module-level names a file binds to a bare string, ``name -> value``.

    A file that writes ``PROBLEM = "planted_glass"`` and then
    ``fixture(PROBLEM, TIER)`` is naming its fixture as plainly as one that
    inlines the literal, and reading only the literal form attributes the file
    to no problem at all --- the defect issue #640 is about, one level down.
    Tuple assignments are read because ``PROBLEM, TIER = "x", "ci"`` is the
    common spelling; an attribute value such as ``Scale.CI`` binds nothing and
    is left to the call's own regex, which then does not match and costs only
    a tier.

    Returns
    -------
    dict[str, str]
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    found: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        targets, values = node.targets[0], node.value
        pairs: list[tuple[ast.expr, ast.expr]] = []
        if isinstance(targets, ast.Tuple) and isinstance(values, ast.Tuple):
            pairs = list(zip(targets.elts, values.elts, strict=False))
        else:
            pairs = [(targets, values)]
        for target, value in pairs:
            if (
                isinstance(target, ast.Name)
                and isinstance(value, ast.Constant)
                and isinstance(value.value, str)
            ):
                found[target.id] = value.value
    return found


def _with_constants(source: str) -> str:
    """``source`` with every module-level string constant spelled out.

    Rewriting the text rather than walking the calls keeps one reading of a
    fixture call --- the regexes below --- instead of two that can disagree.

    Returns
    -------
    str
    """
    resolved = _SCALE_MEMBER.sub(
        lambda match: f'"{match.group(1).lower()}"'
        if match.group(1).lower() in TIERS
        else match.group(0),
        source,
    )
    bound = _constants(resolved)
    if not bound:
        return resolved
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, bound)) + r")\b")
    return pattern.sub(lambda match: f'"{bound[match.group(1)]}"', resolved)


def fixtures_named(source: str) -> set[tuple[str, str]]:
    """The ``(problem, tier)`` fixtures a test file names, however it names them.

    Read per file rather than per function: a module that loads its instance
    once at the top and uses it in every test is naming it for all of them,
    and attributing it to the one line would say the tests below run on
    nothing.

    Returns
    -------
    set[tuple[str, str]]
        A tier of ``""`` means the file named the problem without a tier ---
        ``at_fixture`` parameterizes over every tier the problem declares.
    """
    resolved = _with_constants(source)
    found = set(_FIXTURE_CALL.findall(resolved))
    found |= {(problem, "") for problem in _AT_FIXTURE_CALL.findall(resolved)}
    return found | set(_FIXTURE_PATH.findall(resolved))


def fixture_tiers(problem: str, directory: Path = FIXTURES) -> list[str]:
    """The tiers one fixture problem declares, smallest first.

    Returns
    -------
    list[str]
    """
    present = {path.stem for path in (directory / problem).glob("*.yaml")}
    return [tier for tier in TIERS if tier in present]


def tier_of(markers: set[str], named: set[tuple[str, str]]) -> str:
    """The size tier a test validates a pairing at.

    The scheduling marker decides where it carries one, because that is what
    the selection obeys. Where it does not, the fixtures the test names do:
    an unmarked test runs per pull request, and the smallest instance it
    names is the size its claim is made at.

    Returns
    -------
    str
        One of :data:`TIERS`.
    """
    if markers & {"release", "stress"}:
        return _tier(markers)
    tiers = {
        tier or found
        for problem, tier in named
        for found in fixture_tiers(problem) or [""]
    }
    return next((tier for tier in TIERS if tier in tiers), "ci")


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
        source = path.read_text()
        tree = ast.parse(source)
        named = fixtures_named(source)
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
            tier = tier_of(checks_ledger._markers(node), named)
            for symbol in references[node.name]:
                for kind in kinds:
                    found.setdefault(symbol, set()).add((kind, tier))
    return found


def fixtures_by_symbol(
    tests: Path = checks_ledger.TESTS,
) -> dict[str, set[str]]:
    """``symbol -> {fixture problem}`` over the significant tests that name both.

    What makes "every method is applied to every supported problem" a
    question the tables can answer rather than one a reviewer has to: a
    method is applied to a problem when a test of either significant kind
    names a symbol of the method and a fixture of the problem.

    Returns
    -------
    dict[str, set[str]]
    """
    significant = {(file, name) for file, name, _, _ in checks_ledger.rows()}
    found: dict[str, set[str]] = {}
    for path in sorted(tests.rglob("test_*.py")):
        source = path.read_text()
        problems = {problem for problem, _ in fixtures_named(source)}
        if not problems:
            continue
        tree = ast.parse(source)
        references = _references_by_function(tree)
        relative = str(path.relative_to(REPO_ROOT))
        for node in tree.body:
            if not (
                isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
            ):
                continue
            if (relative, node.name) not in significant:
                continue
            for symbol in references[node.name]:
                found.setdefault(symbol, set()).update(problems)
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
    if loaded is None:
        return {}
    return {key: value for key, value in loaded.items() if key != UNTESTED_KEY}


def untested_notes(path: Path = METHOD_NOTES) -> dict[str, str]:
    """``"<fixture> / <family>" -> reason``, from the committed YAML.

    Why a method the catalogue pairs with a problem is not run on that
    problem's fixture. A pairing may be untested for a good reason --- the
    method has no meaning there, the instance is past the oracle --- and the
    reason is stated rather than left to be rediscovered.

    Returns
    -------
    dict[str, str]
    """
    loaded = yaml.safe_load(path.read_text())
    return {} if loaded is None else dict(loaded.get(UNTESTED_KEY, {}))


def catalogue_rows(
    catalogue: Path = CATALOGUE,
) -> list[tuple[str, list[str], list[str]]]:
    """``(problem, fixture keys, defining code)`` per row of the catalogue.

    The two hand-written columns the tables need. **Key** is the fixture
    directory and the marker, which are one name; a row with two keys is one
    problem declared at two instances. **Defines** is the code that is this
    problem and no other, which the tables do not place in a column --- a
    simulator is not a method --- and which is read here so one reader parses
    the table.

    Returns
    -------
    list[tuple[str, list[str], list[str]]]
    """
    found: list[tuple[str, list[str], list[str]]] = []
    for line in catalogue.read_text().splitlines():
        if not line.startswith("| ") or line.startswith("| Problem") or "---" in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        found.append(
            (
                cells[0],
                sorted(set(_KEY_CELL.findall(cells[1]))),
                sorted(set(_DEFINES_CELL.findall(cells[3]))),
            )
        )
    return found


@cache
def applied(catalogue: Path = CATALOGUE) -> dict[str, frozenset[str]]:
    """``problem -> the methods its own tests reach``, read from the suite.

    The catalogue no longer inventories a problem's methods (issue #640), so
    the tables derive them: a method is applied to a problem when a test of
    either significant kind names a symbol of the method and a fixture of one
    of the problem's keys. The reading is the suite's, which is what the
    tables were always meant to report --- the inventory only ever claimed it.

    Returns
    -------
    dict[str, frozenset[str]]
        Keyed by row title, holding the symbols either table can place.
    """
    by_symbol = fixtures_by_symbol()
    reached: dict[str, set[str]] = {}
    for symbol, problems in by_symbol.items():
        if symbol not in ALGORITHMS and symbol not in ORACLES:
            continue
        for problem in problems:
            reached.setdefault(problem, set()).add(symbol)
    return {
        title: frozenset().union(*(reached.get(key, set()) for key in keys))
        if keys
        else frozenset()
        for title, keys, _ in catalogue_rows(catalogue)
    }


def rows(catalogue: Path = CATALOGUE) -> list[tuple[str, list[str]]]:
    """``(problem, symbols)`` per table row, the symbols derived from the suite.

    Returns
    -------
    list[tuple[str, list[str]]]
    """
    reached = applied(catalogue)
    return [
        (title, sorted(reached.get(title, frozenset())))
        for title, _, _ in catalogue_rows(catalogue)
    ]


def untested_pairs(
    catalogue: Path = CATALOGUE, note_map: dict[str, dict[str, str]] | None = None
) -> list[tuple[str, str]]:
    """``(fixture key, method family)`` pairings claimed but not made.

    "Every compatible method is applied to every supported problem" is a
    claim, and the two halves of it now come from two places, which is what
    stops the check being circular. **The claim** is
    ``docs/tex/method_notes.yaml``: a note for a problem and a family says the
    family applies there. **The coverage** is the suite: a test of either
    significant kind naming a symbol of the family and a fixture of the
    problem's key. A pairing the first asserts and the second does not is
    listed, with the reason it is untested, and loses its entry only by
    gaining a test (issue #382).

    Returns
    -------
    list[tuple[str, str]]
        Sorted, without repeats.
    """
    known = notes() if note_map is None else note_map
    by_symbol = fixtures_by_symbol()
    of_family = {
        name: {symbol for symbol in ALGORITHMS if family(symbol) == name}
        for name in METHOD_FAMILIES
    }
    found = set()
    for title, keys, _ in catalogue_rows(catalogue):
        for name in known.get(title, {}):
            for key in keys:
                if not any(
                    key in by_symbol.get(symbol, ()) for symbol in of_family[name]
                ):
                    found.add((key, name))
    return sorted(found)


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

    Every pairing appears: one the notes claim that no test of either
    significant kind makes is ``untested``, and one nothing claims and nothing
    tests is ``--``. Omitting either would make the table read as though the
    question had not been asked.

    Raises
    ------
    MissingNoteError
        If the suite applies a family to a problem the notes do not claim.
        The table cannot go quiet about a method that is running.
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
                    f"{problem!r} runs {name} in the suite and has no note in "
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
            elif note:
                # Claimed by the notes and reached by no significant test.
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
            # \scriptsize, not \footnotesize: the optimizers part carries a
            # sentence for every problem and overflows a page at the larger
            # size, and LaTeX reports that as a warning rather than an error.
            r"  \scriptsize",
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
    unclaimed = sorted(
        f"{problem} / {name}"
        for problem, name, _, referee, note in method_cells(catalogue)
        if referee not in ("--", "untested") and not note
    )
    if unclaimed:
        msg = (
            f"the suite applies {unclaimed} and {METHOD_NOTES.name} claims neither; "
            "the table cannot be narrower than what is running"
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
            "truth only, $\\circ$ neither kind of test, and -- the problem "
            "carries no method of that family. The tables report what the "
            "tree holds, so -- reads as \\emph{not built} and never as "
            "\\emph{not applicable}; Section~\\ref{sec:applicability} says "
            "which pairings are the second. A dagger or a circle is a cell "
            "where an oracle is wanted and absent.",
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
    for problem, name in untested_pairs():
        print(f"untested: {problem} / {name}", file=sys.stderr)
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
