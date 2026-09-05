"""Regression tests for the optimization interface and its constraint maps.

Two things are asserted here that no other module can assert: that the
constraint maps are bijections onto their feasible sets (so a fitted
parameter has an identifiable value), and that ``phylo.opt`` contains no
application knowledge -- the structural claim issue #63 exists to make.
"""

from __future__ import annotations

import ast
from pathlib import Path

import phylo.opt
import pytest
import torch
from numpy.testing import assert_allclose
from phylo.opt.constrain import free_from_log_simplex, log_simplex
from phylo.opt.hmm import HmmObjective
from phylo.opt.objective import Objective
from phylo.opt.potts import PottsObjective

# The whole point of the abstraction: the optimizer may not know what it is
# optimizing. Stated as module prefixes rather than names so a new
# application module is covered the day it is added.
FORBIDDEN_PREFIXES = ("phylo.sim", "phylo.likelihood", "phylo.search")


def _imported_modules(source: Path) -> set[str]:
    tree = ast.parse(source.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    return imported


@pytest.mark.critical
@pytest.mark.structural
def test_opt_imports_nothing_from_the_application_modules() -> None:
    # `opt/CLAUDE.md` says nothing phylogenetic belongs here. That rule is
    # only worth stating if something checks it: a single `from phylo.sim
    # import ...` added in a hurry is invisible to ruff and mypy, and turns
    # the abstraction back into a phylogenetics-specific optimizer.
    package = Path(phylo.opt.__file__).parent
    offenders: dict[str, set[str]] = {}
    for source in sorted(package.glob("*.py")):
        bad = {
            name
            for name in _imported_modules(source)
            if name.startswith(FORBIDDEN_PREFIXES)
        }
        if bad:
            offenders[source.name] = bad
    assert offenders == {}


@pytest.mark.critical
@pytest.mark.structural
def test_the_check_would_catch_an_application_import(tmp_path: Path) -> None:
    # Guards the guard: a structural test that cannot fail is worse than no
    # test, because it reads as evidence.
    source = tmp_path / "leaky.py"
    source.write_text("from phylo.sim.tree import Node\nimport phylo.likelihood\n")
    imported = _imported_modules(source)
    assert {n for n in imported if n.startswith(FORBIDDEN_PREFIXES)} == {
        "phylo.sim.tree",
        "phylo.likelihood",
    }


@pytest.mark.mathematical
@pytest.mark.parametrize("n", [2, 3, 5])
def test_log_simplex_yields_a_normalized_distribution(n: int) -> None:
    free = torch.linspace(-1.5, 2.0, n - 1, dtype=torch.float64)
    log_probs = log_simplex(free)
    assert log_probs.shape == (n,)
    assert_allclose(float(torch.exp(log_probs).sum()), 1.0, rtol=1e-15)


@pytest.mark.mathematical
def test_log_simplex_normalizes_each_row_of_a_batch() -> None:
    # Transition and emission matrices are mapped a whole matrix at a time,
    # so the batched path is the one that is actually used.
    free = torch.tensor([[0.5, -1.0], [2.0, 0.25], [-0.75, 0.0]], dtype=torch.float64)
    log_probs = log_simplex(free)
    assert log_probs.shape == (3, 3)
    assert_allclose(
        torch.exp(log_probs).sum(dim=1).numpy(), [1.0, 1.0, 1.0], rtol=1e-15
    )


@pytest.mark.mathematical
def test_free_from_log_simplex_inverts_log_simplex() -> None:
    free = torch.tensor([0.3, -1.2, 0.75], dtype=torch.float64)
    assert_allclose(
        free_from_log_simplex(log_simplex(free)).numpy(), free.numpy(), rtol=1e-14
    )


@pytest.mark.mathematical
def test_the_pinned_gauge_leaves_no_flat_direction() -> None:
    # A plain softmax over n logits is invariant to adding a constant to all
    # of them, which makes the observed information singular and an interval
    # undefined. Pinning the first logit removes exactly that direction: the
    # same shift now changes the answer.
    free = torch.tensor([0.4, -0.9], dtype=torch.float64)
    shifted = free + 1.0
    assert not torch.allclose(log_simplex(free), log_simplex(shifted))

    logits = torch.cat([torch.zeros(1, dtype=torch.float64), free])
    assert torch.allclose(
        torch.log_softmax(logits, dim=0), torch.log_softmax(logits + 1.0, dim=0)
    )


@pytest.mark.structural
def test_both_reference_instances_satisfy_the_protocol() -> None:
    chains = torch.zeros((2, 3), dtype=torch.long).numpy()
    assert isinstance(PottsObjective(chains, n_states=2), Objective)
    assert isinstance(HmmObjective(chains, n_states=2, n_symbols=2), Objective)
