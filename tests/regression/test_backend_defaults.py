"""A public ``backend=`` default is ``RUST`` wherever a Rust route exists (issue #1283).

Read from the source tree, as `test_api_vocabulary.py` reads it: a default is
what a caller gets without asking, so it is the route the package is judged
on. The owner's rule is that a default flips to ``RUST`` where the Rust route
is measured at 2x or more against the current default at stress size, min of
3. Every default that is not ``RUST`` is listed below with its reason, so a new
one fails here until it states one.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[2] / "python" / "sal"

#: Each public ``backend``-like default that is not ``RUST``, keyed
#: ``<module path>::<function>(<parameter>)``, and why.
_NO_RUST = "no Rust route"
_GIBBS = "no Rust route; NUMBA is the compiled route"
_RETIRED_HMM = "no Rust route on the retired objective (#1189); deprecated"
_GRADIENT = "differentiates; JAX and TORCH are the routes it admits"
_COUNTS_ONLY = (
    "the Rust route takes the two-channel count emission alone and raises "
    "on the categorical model the default serves"
)
EXCEPTIONS = {
    "likelihood/objective.py::BranchLengthObjective.__init__(backend)": _GRADIENT,
    "likelihood/objective.py::SubstitutionModelObjective.__init__(backend)": _GRADIENT,
    "likelihood/spatio_sequential/__init__.py::class_posteriors(backend)": _COUNTS_ONLY,
    "likelihood/spatio_sequential/__init__.py::external_field(backend)": _COUNTS_ONLY,
    "likelihood/spatio_sequential/__init__.py::labelled_log_likelihood(backend)": (
        _COUNTS_ONLY
    ),
    "opt/hmm/objectives.py::HmmObjective.__init__(backend)": _RETIRED_HMM,
    "opt/hmm/objectives.py::GaussianHmmObjective.__init__(backend)": _RETIRED_HMM,
    "opt/hmm/objectives.py::PoissonHmmObjective.__init__(backend)": _RETIRED_HMM,
    "opt/hmm/objectives.py::BinomialHmmObjective.__init__(backend)": _RETIRED_HMM,
    "opt/hmm/objectives.py::BetaBinomialHmmObjective.__init__(backend)": _RETIRED_HMM,
    "opt/hmm/objectives.py::NegativeBinomialHmmObjective.__init__(backend)": (
        _RETIRED_HMM
    ),
    "sample/gibbs/__init__.py::Indexed.log_density(backend)": _GIBBS,
    "sample/gibbs/__init__.py::gibbs_sweep(backend)": _GIBBS,
    "sample/gibbs/__init__.py::sample_factor_graph(backend)": _GIBBS,
    "sample/gibbs/__init__.py::anneal_factor_graph(backend)": _GIBBS,
    "sample/gibbs/__init__.py::heat_bath(backend)": _GIBBS,
    "sample/slice.py::slice_sample(backend)": _NO_RUST,
    "search/icm/__init__.py::colouring(backend)": _GIBBS,
    "search/icm/__init__.py::iterated_conditional_modes(backend)": _GIBBS,
    "search/icm/__init__.py::merge_small_labels(backend)": _GIBBS,
    "search/trws/__init__.py::trws(backend)": _GIBBS,
    "search/spatio_sequential.py::fit_spatio_sequential(backend)": _COUNTS_ONLY,
    "sim/count_pairs/__init__.py::simulate_count_pairs(backend)": (
        "<2x at stress: 1.05x (30.7 s against 29.1 s, "
        "spatio_sequential_counts/stress.yaml)"
    ),
}

#: The defaults #1283 flipped, and the ratio each was flipped on: stress
#: size, min of 3, the oracle's time over the Rust route's.
FLIPPED = {
    "search/maxflow/__init__.py::max_flow(backend)": "131x, open 512^2",
    "search/maxflow/__init__.py::ising_ground_state(backend)": "315x, open 512^2",
    "sample/potts_mcmc/sweeps.py::swendsen_wang_sweep(backend)": "47.2x, 64^2 q=3",
    "sample/potts_mcmc/chains.py::sample_potts(cluster_backend)": "35.6x, 64^2 q=3",
    "sample/potts_mcmc/chains.py::anneal_potts(cluster_backend)": "49.1x, 64^2 q=3",
    "sample/potts_mcmc/chains.py::parallel_tempering(cluster_backend)": "34.2x",
    "sample/potts_mcmc/chains.py::sample_potts_pair(cluster_backend)": "19.8x",
    "sample/potts_mcmc/chains.py::sweep_for(cluster_backend)": "38.6x, 64^2 q=3",
    "sample/annealed.py::annealed_importance_sampling(cluster_backend)": "38.9x",
    "sample/annealed.py::population_annealing(cluster_backend)": "33.4x",
    "sample/annealed.py::simulated_tempering(cluster_backend)": "47.8x",
    "sample/tempered.py::tempered_potts_pair(cluster_backend)": "23.0x",
    "sample/potts_keyed.py::SwendsenWangMove.__init__(backend)": "45.9x, 64^2",
    "sample/potts_keyed.py::cluster_moves(backend)": "13.5x, 64^2",
    "likelihood/pruning/__init__.py::log_likelihood(backend)": "15.5x, tree_jc/stress",
    "learn/ranking.py::ground_state_target(backend)": "32.8x, spatio_only/stress",
}

#: The dunder methods a caller writes the parameters of.
_PUBLIC_DUNDERS = ("__init__", "__call__")


def _backend_defaults(source: str) -> dict[str, str]:
    """Every public ``backend`` or ``*_backend`` default naming a ``Backend`` member, by key."""
    found: dict[str, str] = {}

    def visit(node: ast.AST, owner: str | None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                if not child.name.startswith("_"):
                    visit(child, child.name)
            elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                if child.name.startswith("_") and child.name not in _PUBLIC_DUNDERS:
                    continue
                arguments = child.args
                positional = arguments.posonlyargs + arguments.args
                pairs = list(
                    zip(
                        positional[len(positional) - len(arguments.defaults) :],
                        arguments.defaults,
                        strict=True,
                    )
                ) + [
                    (argument, default)
                    for argument, default in zip(
                        arguments.kwonlyargs, arguments.kw_defaults, strict=True
                    )
                    if default is not None
                ]
                name = f"{owner}.{child.name}" if owner else child.name
                for argument, default in pairs:
                    if not (
                        argument.arg == "backend" or argument.arg.endswith("_backend")
                    ):
                        continue
                    if (
                        isinstance(default, ast.Attribute)
                        and isinstance(default.value, ast.Name)
                        and default.value.id == "Backend"
                    ):
                        found[f"{name}({argument.arg})"] = default.attr

    visit(ast.parse(source), None)
    return found


def _package_defaults() -> dict[str, str]:
    """:func:`_backend_defaults` over the package, keyed by module path."""
    return {
        f"{path.relative_to(PACKAGE).as_posix()}::{key}": member
        for path in sorted(PACKAGE.rglob("*.py"))
        for key, member in _backend_defaults(path.read_text()).items()
    }


@pytest.mark.critical
@pytest.mark.infra
def test_every_public_backend_default_is_rust_or_states_why_not() -> None:
    defaults = _package_defaults()
    off = {key for key, member in defaults.items() if member != "RUST"}

    unlisted = sorted(off - EXCEPTIONS.keys())
    stale = sorted(EXCEPTIONS.keys() - off)
    assert unlisted == [], f"non-RUST defaults with no stated reason: {unlisted}"
    assert stale == [], f"listed exceptions that are RUST or gone: {stale}"
    assert all(reason for reason in EXCEPTIONS.values())


@pytest.mark.critical
@pytest.mark.infra
def test_the_defaults_issue_1283_flipped_stay_rust() -> None:
    defaults = _package_defaults()

    assert {key: defaults.get(key) for key in FLIPPED} == dict.fromkeys(FLIPPED, "RUST")


@pytest.mark.infra
def test_the_guard_reads_each_form_of_a_backend_default() -> None:
    source = """
def a(x, backend: Backend = Backend.PYTHON): ...
def b(*, cluster_backend: Backend = Backend.RUST): ...
def _c(backend: Backend = Backend.PYTHON): ...
def d(backend: Backend): ...
class E:
    def __init__(self, backend: Backend = Backend.NUMBA): ...
    def _f(self, backend: Backend = Backend.PYTHON): ...
class _G:
    def h(self, backend: Backend = Backend.PYTHON): ...
"""

    # Private functions, private methods and private classes are skipped; a
    # parameter with no default is not a default.
    assert _backend_defaults(source) == {
        "a(backend)": "PYTHON",
        "b(cluster_backend)": "RUST",
        "E.__init__(backend)": "NUMBA",
    }
