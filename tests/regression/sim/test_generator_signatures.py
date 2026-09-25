"""Randomness enters through a generator, and this is what says so.

Issue #240 (`sim/CLAUDE.md`): seeding inside a call makes an ensemble one
draw. Per function: two draws from one generator differ (fails under a seed
signature) and two generators seeded alike agree. The guard: #230 counted 12
seed signatures against 10 at filing, 16 against 16 at implementation. #337
converted the `torch` stream (`sample.hmc.sample`, `sample.hmc.anneal`,
`search.max_cut.goemans_williamson`); no exemption remains.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import torch
from sal.sample.hmc import anneal, sample
from sal.sample.potts_mcmc import PottsMove, sample_potts
from sal.sample.schedule import ConstantTempSchedule
from sal.search.icm import iterated_conditional_modes
from sal.search.max_cut import goemans_williamson
from sal.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from sal.sim.potts import simulate_potts
from sal.sim.simulate import simulate_alignment
from sal.sim.tree import Node

from tests._paths import REPO_ROOT
from tests._posteriors import GAUSSIAN

PACKAGE = REPO_ROOT / "python" / "sal"


def _tree() -> Node:
    """A four-taxon tree, the smallest an alignment is drawn on."""
    leaves = [Node(name=f"t{i}", branch_length=0.1) for i in range(4)]
    inner = Node(name="i", branch_length=0.1, children=(leaves[2], leaves[3]))
    return Node(name="root", branch_length=None, children=(leaves[0], leaves[1], inner))


def _graph() -> PottsGraph:
    return lattice_graph((3, 3), BoundaryCondition.OPEN, 0.4)


#: A per-node field of the same order as the coupling, so the two disagree and
#: the descent has somewhere to land other than one uniform answer.
ICM_FIELD = np.random.default_rng(1).normal(size=(9, 3))


def _alignment_draw(rng: np.random.Generator) -> tuple[int, ...]:
    dataset = simulate_alignment(_tree(), 4, np.full(4, 0.25), rng, 30)
    return tuple(int(v) for v in dataset.alignment["t0"])


def _potts_draw(rng: np.random.Generator) -> tuple[int, ...]:
    dataset = simulate_potts(_graph(), np.zeros(2), rng, 3, burn_in=5)
    return tuple(int(v) for v in dataset.configurations.reshape(-1))


def _mcmc_draw(rng: np.random.Generator) -> tuple[int, ...]:
    chain = sample_potts(_graph(), np.zeros(2), PottsMove.SINGLE_SITE, rng, 3)
    return tuple(int(v) for v in chain.states.reshape(-1))


def _icm_draw(rng: np.random.Generator) -> tuple[int, ...]:
    """One descent from a random start, on a surface where the start matters.

    A random field against a comparable coupling: 6 draws reach 3 optima.
    """
    labelling, *_ = iterated_conditional_modes(_graph(), ICM_FIELD, 3, rng)
    return tuple(int(v) for v in labelling)


#: A 2-D target with a closed form, so an HMC draw is cheap and its stream is
#: the only thing that varies between chains.


def _hmc_draw(generator: torch.Generator) -> tuple[float, ...]:
    chain = sample(GAUSSIAN, generator, 4, step_size=0.2, n_steps=5)
    return tuple(float(v) for v in chain.draws.reshape(-1))


def _anneal_draw(generator: torch.Generator) -> tuple[float, ...]:
    annealed = anneal(
        GAUSSIAN, ConstantTempSchedule(1.0, 4), generator, step_size=0.2, n_steps=5
    )
    return tuple(float(v) for v in annealed.final)


def _max_cut_draw(generator: torch.Generator) -> tuple[float, ...]:
    """The relaxation an under-solved ascent reaches from a random start.

    Rounded cuts all agree on small graphs; 20 ascent steps remember the start.
    """
    result = goemans_williamson(_graph(), generator, iterations=20, roundings=4)
    return (result.relaxation, result.value)


NUMPY_DRAWS = {
    "simulate_alignment": _alignment_draw,
    "simulate_potts": _potts_draw,
    "sample_potts": _mcmc_draw,
    "iterated_conditional_modes": _icm_draw,
}

TORCH_DRAWS = {
    "sample": _hmc_draw,
    "anneal": _anneal_draw,
    "goemans_williamson": _max_cut_draw,
}


def _draws_from_one_stream(
    name: str, seed: int, count: int
) -> list[tuple[int, ...] | tuple[float, ...]]:
    """``count`` draws of ``name`` from one `numpy` or `torch` stream at ``seed``."""
    if name in NUMPY_DRAWS:
        rng = np.random.default_rng(seed)
        return [NUMPY_DRAWS[name](rng) for _ in range(count)]
    generator = torch.Generator().manual_seed(seed)
    return [TORCH_DRAWS[name](generator) for _ in range(count)]


@pytest.mark.end2end
@pytest.mark.parametrize("name", sorted({**NUMPY_DRAWS, **TORCH_DRAWS}))
def test_two_draws_from_one_generator_differ(name: str) -> None:
    """The property the rule exists for, per converted function.

    Fails under a seed signature: eight draws would be one.
    """
    drawn = set(_draws_from_one_stream(name, 20260905, 6))

    assert len(drawn) > 1, f"{name} returns the same draw from one generator"


@pytest.mark.smoke
@pytest.mark.parametrize("name", sorted({**NUMPY_DRAWS, **TORCH_DRAWS}))
def test_generators_seeded_alike_agree(name: str) -> None:
    """Reproducibility survives the conversion, or the test above passes on noise."""
    assert _draws_from_one_stream(name, 7, 1) == _draws_from_one_stream(name, 7, 1)


def _seed_parameters(path: Path) -> list[str]:
    """Public function parameters ``seed: int`` in ``path``; fields are not read."""
    tree = ast.parse(path.read_text())
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
            continue
        for argument in [*node.args.args, *node.args.kwonlyargs]:
            annotation = ast.unparse(argument.annotation) if argument.annotation else ""
            if argument.arg == "seed" and annotation == "int":
                found.append(f"{path.name}::{node.name}")
    return found


@pytest.mark.critical
@pytest.mark.smoke
def test_no_public_signature_takes_a_seed() -> None:
    """The rule, enforced where it can be, over both streams.

    No exemption since #337; a fixture's declared `seed` field is not a parameter.
    """
    offenders = [
        entry
        for path in sorted(PACKAGE.rglob("*.py"))
        for entry in _seed_parameters(path)
    ]
    assert not offenders, (
        f"{len(offenders)} public signature(s) still take a seed where "
        f"`sim/CLAUDE.md` says a generator: {offenders}"
    )


@pytest.mark.smoke
def test_the_guard_fails_on_a_signature_that_takes_a_seed(tmp_path: Path) -> None:
    """The guard rejects what it exists to reject, and spares what it should.

    A dataclass field `seed` is spared: the fixture seed is a non-goal.
    """
    offending = tmp_path / "offending.py"
    offending.write_text("def draw(n: int, seed: int) -> int:\n    return n\n")
    converted = tmp_path / "converted.py"
    converted.write_text(
        "import numpy as np\n\n\n"
        "def draw(n: int, rng: np.random.Generator) -> int:\n    return n\n"
    )
    declared = tmp_path / "declared.py"
    declared.write_text(
        "from dataclasses import dataclass\n\n\n"
        "@dataclass\nclass Params:\n    seed: int\n"
    )

    assert _seed_parameters(offending) == ["offending.py::draw"]
    assert _seed_parameters(converted) == []
    assert _seed_parameters(declared) == [], (
        "a declared fixture seed is not a signature"
    )
