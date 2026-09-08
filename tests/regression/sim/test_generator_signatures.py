"""Randomness enters through a generator, and this is what says so.

Issue #240. `sim/CLAUDE.md` states the rule -- a generator, never a seed --
because seeding inside a call makes every draw of an ensemble identical, which
looks like a passing test over many draws and is one draw. That mistake has
been made in this repository, which is why `erdos_renyi_graph` was given the
generator signature deliberately and
`test_independent_draws_come_from_one_generator` pins it.

Five public functions kept the seed shape. These tests extend that pairing to
each of them: two draws from one generator differ, and two generators seeded
alike agree. The first half is the one that matters -- under the old signature
it fails, because every call rebuilt the same stream -- so it is a real
discriminator rather than a restatement of what the code does.

The guard below is the other half. A rule nothing checks is a rule the next
module quietly breaks: at filing, #230 counted 12 seed-taking signatures
against 10 generator-taking ones, and by the time this ticket was implemented
it was 16 against 16. The seed side had grown while the ticket waited.

Issue #337 converted the three the first pass deferred, the `torch` stream --
`opt.hmc.sample`, `opt.hmc.anneal` and `search.max_cut.goemans_williamson` --
so the pairing runs over both streams and the guard has no exemption left.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import torch
from snakes_and_ladders.opt.hmc import anneal, sample
from snakes_and_ladders.opt.schedule import Constant
from snakes_and_ladders.search.alpha_expansion import iterated_conditional_modes
from snakes_and_ladders.search.max_cut import goemans_williamson
from snakes_and_ladders.search.potts_mcmc import PottsMove, sample_potts
from snakes_and_ladders.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from snakes_and_ladders.sim.potts import simulate_potts
from snakes_and_ladders.sim.simulate import simulate_alignment
from snakes_and_ladders.sim.tree import Node

from tests._objective_checks import AnalyticGaussian

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE = REPO_ROOT / "python" / "snakes_and_ladders"

#: The fixture parameter objects a yaml declares. Their `seed` field is how a
#: run is declared reproducible and stays: only the boundary at which it
#: becomes a generator moved, which is the caller.
DECLARED_PARAMS = frozenset(
    {"SimulationParams", "PottsLatticeParams", "HmmParams", "PottsParams"}
)


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

    Iterated conditional modes is an optimizer, not a sampler, so it only
    reports the generator it was given where the surface has more than one
    local optimum to fall into. A *uniform* field has one -- every site takes
    the same label and every start converges there -- and so does a field
    strong enough to decide each site alone. A random field against a coupling
    of comparable size is the case with content: the two terms disagree, the
    surface is rugged, and 6 draws reach 3 distinct optima.
    """
    labelling, _ = iterated_conditional_modes(_graph(), ICM_FIELD, 3, rng)
    return tuple(int(v) for v in labelling)


#: A 2-D target with a closed form, so an HMC draw is cheap and its stream is
#: the only thing that varies between chains.
GAUSSIAN = AnalyticGaussian([1.0, -2.0], [[2.0, 0.6], [0.6, 0.5]])


def _hmc_draw(generator: torch.Generator) -> tuple[float, ...]:
    chain = sample(GAUSSIAN, generator, 4, step_size=0.2, n_steps=5)
    return tuple(float(v) for v in chain.theta.reshape(-1))


def _anneal_draw(generator: torch.Generator) -> tuple[float, ...]:
    annealed = anneal(GAUSSIAN, Constant(1.0, 4), generator, step_size=0.2, n_steps=5)
    return tuple(float(v) for v in annealed.final)


def _max_cut_draw(generator: torch.Generator) -> tuple[float, ...]:
    """The relaxation an under-solved ascent reaches from a random start.

    The rounded cut is the wrong witness: on a small graph every hyperplane
    finds the optimum, so two draws agree on the cut whatever stream they
    came from. The relaxation after 20 ascent steps still remembers its
    starting vectors, which is where the generator shows.
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
    """``count`` draws of ``name`` from one stream seeded with ``seed``.

    The stream is a `numpy` or a `torch` generator by the function's
    convention; what the tests state does not depend on which.
    """
    if name in NUMPY_DRAWS:
        rng = np.random.default_rng(seed)
        return [NUMPY_DRAWS[name](rng) for _ in range(count)]
    generator = torch.Generator().manual_seed(seed)
    return [TORCH_DRAWS[name](generator) for _ in range(count)]


@pytest.mark.simulated_truth
@pytest.mark.parametrize("name", sorted({**NUMPY_DRAWS, **TORCH_DRAWS}))
def test_two_draws_from_one_generator_differ(name: str) -> None:
    """The property the rule exists for, per converted function.

    This is the half that fails under the old signature: a function seeding
    itself returns the same draw every call, so an ensemble of eight is one
    draw reported eight times.
    """
    drawn = set(_draws_from_one_stream(name, 20260905, 6))

    assert len(drawn) > 1, f"{name} returns the same draw from one generator"


@pytest.mark.structural
@pytest.mark.parametrize("name", sorted({**NUMPY_DRAWS, **TORCH_DRAWS}))
def test_generators_seeded_alike_agree(name: str) -> None:
    """Reproducibility survives the conversion.

    A declared seed still determines the run; what moved is where it becomes a
    generator. Without this the test above would pass for a function that had
    simply become non-deterministic.
    """
    assert _draws_from_one_stream(name, 7, 1) == _draws_from_one_stream(name, 7, 1)


def _seed_parameters(path: Path) -> list[str]:
    """Public function parameters named ``seed`` and typed ``int`` in ``path``.

    Dataclass fields are excluded by construction -- only ``FunctionDef``
    arguments are read -- so a fixture's declared seed does not register.
    """
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
@pytest.mark.structural
def test_no_public_signature_takes_a_seed() -> None:
    """The rule, enforced where it can be, over both streams.

    No function is exempt: the `torch` stream #254 deferred -- `opt.hmc.sample`,
    `opt.hmc.anneal` and `search.max_cut.goemans_williamson` -- takes a
    `torch.Generator` since #337. The one thing that keeps its `seed` is a
    declared fixture parameter's *field*, which is how a run is declared
    reproducible and which `_seed_parameters` does not read; only the
    boundary at which it becomes a generator moved.
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


@pytest.mark.structural
def test_the_guard_fails_on_a_signature_that_takes_a_seed(tmp_path: Path) -> None:
    """The guard rejects what it exists to reject, and spares what it should.

    A guard that only passes on the converted tree says nothing about the next
    module -- and one that could not tell a function's parameter from a
    dataclass's field would force the fixture seed out too, which the ticket
    names as a non-goal.
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
