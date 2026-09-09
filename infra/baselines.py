"""Compute the baseline numbers beside each fixture, or check the committed ones.

A fixture's baseline is what a *reference* algorithm achieves on the instance
--- the enumerated maximum over its topologies, the rate at which random-restart
hill climbing reaches that maximum, the rate an untrained policy reaches it,
the exact target every learned surrogate is fitted against. Each is a
deterministic function of the fixture, the code and a seed, and each is
expensive: enumerating 945 topologies is 1.4 s and 200 uniform rollouts are
8.3 s, paid before a test measures anything about the thing under test. That
is most of why three claims left the per-pull-request tier (issue #401).

So the numbers are computed once and committed, and a test reads them:
``<problem>/<tier>.baseline.json`` beside the fixture, written here and read by
``snakes_and_ladders.sim.fixtures.baseline``, which refuses a record whose
digest is not the current tree's.

**Where the recomputation went.** Per pull request nothing here runs; the
release gate runs ``infra/baselines.py`` with no flag, which recomputes every
record and fails on any drift. That is the trade
``snakes_and_ladders.qa.build`` states for figures --- the check moves to the
release gate rather than disappearing --- and it is made for the same reason:
a cached number no test ever recomputes is a claim with no referee, and one
recomputed on every pull request costs the seconds the tier move removed. The
digest closes the gap between the two: a change to the fixture, to the
computing modules, or to a recorded budget makes every reader raise on the
pull request that made it, without any of them recomputing.

The record is a second file rather than a block in the fixture yaml. The
fixture declares an instance and is written by hand; the record states a
measurement and is written by this script, so a regeneration never rewrites a
declaration.

Run::

    uv run python infra/baselines.py --write   # compute and commit
    uv run python infra/baselines.py           # recompute and compare
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from snakes_and_ladders.fixtures import Scale
from snakes_and_ladders.inputs import library_versions, module_closure
from snakes_and_ladders.learn.exact import exact_expected_return
from snakes_and_ladders.learn.policy import LinearPolicy
from snakes_and_ladders.learn.potts import (
    PottsLandscape,
    enumerate_configurations,
    optimum,
)
from snakes_and_ladders.learn.rollout import greedy_rollout, rollout
from snakes_and_ladders.likelihood.potts import log_weights
from snakes_and_ladders.log import get_logger, phase
from snakes_and_ladders.search.alpha_expansion import iterated_conditional_modes
from snakes_and_ladders.search.infer import MoveSet
from snakes_and_ladders.search.rl import FeatureSet, RewardModel, TopologyEnvironment
from snakes_and_ladders.search.surrogate import maximized_target
from snakes_and_ladders.search.topology import Topology, enumerate_topologies
from snakes_and_ladders.sim.fixtures import (
    BASELINE_LIBRARIES,
    Baseline,
    Fixture,
    Measurement,
    StaleBaselineError,
    baseline_path,
    fixture,
    path_of,
    read_baseline,
    write_baseline,
)
from snakes_and_ladders.sim.graph import PottsGraph
from snakes_and_ladders.sim.params import SimulationParams
from snakes_and_ladders.sim.simulate import simulate_alignment
from snakes_and_ladders.sim.tree import edges

REPO_ROOT = Path(__file__).resolve().parents[1]

# The tree measurements' budget, which is `test_search_hard_fixture.py`'s and
# issue #178's: 50 starting topologies drawn from `seed + 1000`, 30 decisions
# per episode. A rate at another budget is a different number, so the numbers
# and the names travel together into every record below.
STARTS = 50
HORIZON = 30
START_SEED_OFFSET = 1000
#: Rollouts per start for a stochastic policy, averaged rather than best-of,
#: so the rate is budget-matched against greedy's one deterministic run.
ROLLOUTS_PER_START = 4
#: The generator an untrained policy is rolled out under.
UNTRAINED_SEED = 99

# The surrogate training set: every topology of each of the first three
# alignments, fitted at 200 sites. Three rather than issue #308's six,
# because the per-pull-request sibling splits one alignment to each of train,
# validation and test and a fourth would only make the fit slower.
SURROGATE_ALIGNMENTS = 3
SURROGATE_SITES = 200
SURROGATE_SEED_OFFSET = 1000

# The enumerable reinforcement-learning environment: the declared chain at the
# length exhaustive enumeration reaches. The fixture's own `chain_length` is
# the estimation study's (12 sites, 531,441 configurations); a policy's exact
# expected return is a sum over every configuration, so the environment the
# learners are refereed on is the same model at length 4, which is 81.
RL_CHAIN_LENGTH = 4
RL_RETURN_HORIZON = 3
RL_ROLLOUT_HORIZON = 6

# The spin-glass measurement's budget (issue #406): 50 seeded restarts of
# single-site descent per seed, over 16 seeds, which is the interval §2.4
# asks for. The seeds are consecutive from the fixture's own.
GLASS_RESTART_SEEDS = 16
#: Zero field: the instance's difficulty is in the couplings.
GLASS_FIELD = np.zeros(2)
#: Configurations hashed per pass of the ground-state enumeration, so a
#: 2**18-state instance is one contiguous array at a time rather than 18
#: columns of a million rows.
GLASS_CHUNK = 1 << 16


def _tree_environment(
    params: SimulationParams, features: FeatureSet, moves: MoveSet = MoveSet.NNI
) -> tuple[TopologyEnvironment, list[str]]:
    """The reward surface an agent sees on a tree fixture, and its taxa."""
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        rng=np.random.default_rng(params.seed),
        n_sites=params.n_sites,
    )
    alignment = dict(dataset.alignment)
    built = TopologyEnvironment(
        alignment,
        params.k,
        np.asarray(params.pi),
        branch_length=float(
            np.mean([child.branch_length for _, child in edges(params.tau)])
        ),
        reward=RewardModel.KNOWN,
        moves=moves,
        features=features,
    )
    return built, sorted(alignment)


def _starts(built: TopologyEnvironment, seed: int) -> list[Topology]:
    """`STARTS` starting topologies, drawn as every tree measurement draws them."""
    generator = np.random.default_rng(seed + START_SEED_OFFSET)
    return [built.reset(generator) for _ in range(STARTS)]


def _rate(
    built: TopologyEnvironment, endpoints: Sequence[Topology], best: float
) -> float:
    """The fraction of ``endpoints`` scoring the enumerated maximum."""
    return float(
        np.mean([abs(built.score(state) - best) < 1e-9 for state in endpoints])
    )


def tree_policy_baseline(loaded: Fixture) -> dict[str, Measurement]:
    """The enumerated maximum and the two reference rates on a tree fixture.

    The three numbers a policy is judged against: what the best topology
    scores, how often hill climbing finds it, and how often a policy that has
    learned nothing finds it. The last is the control without which "the
    policy ties greedy" is consistent with the instance being trivial.
    """
    params = loaded.params
    built, taxa = _tree_environment(params, FeatureSet.FULL)
    budget = {
        "starts": STARTS,
        "horizon": HORIZON,
        "start_seed_offset": START_SEED_OFFSET,
    }

    best = max(built.score(topology) for topology in enumerate_topologies(taxa))
    starts = _starts(built, params.seed)
    greedy = _rate(
        built,
        [greedy_rollout(built, start, HORIZON).states[-1] for start in starts],
        best,
    )
    generator = np.random.default_rng(UNTRAINED_SEED)
    untrained_policy = LinearPolicy(built.n_features())
    untrained = _rate(
        built,
        [
            rollout(built, untrained_policy, generator, HORIZON, start=start).states[-1]
            for start in starts
            for _ in range(ROLLOUTS_PER_START)
        ],
        best,
    )
    return {
        "enumerated_maximum": Measurement(
            algorithm=(
                f"exhaustive enumeration of the "
                f"{len(list(enumerate_topologies(taxa)))} unrooted topologies, "
                f"scored at RewardModel.KNOWN"
            ),
            value=best,
            seed=None,
            budget={"taxa": len(taxa), "reward": str(RewardModel.KNOWN)},
        ),
        "greedy_rate": Measurement(
            algorithm="NNI hill climbing from each seeded start, to a local optimum",
            value=greedy,
            seed=int(params.seed),
            budget=budget,
        ),
        "untrained_rate": Measurement(
            algorithm=(
                "a LinearPolicy at its initial weights, uniform over the same "
                "moves, averaged over rollouts per start"
            ),
            value=untrained,
            seed=UNTRAINED_SEED,
            budget={**budget, "rollouts_per_start": ROLLOUTS_PER_START},
        ),
    }


def tree_surrogate_baseline(loaded: Fixture) -> dict[str, Measurement]:
    """The maximized log-likelihood of every topology of every training alignment.

    The exact targets a learned surrogate is fitted against and scored on. One
    fit per topology per alignment, which is what makes the training set
    expensive rather than the surrogate.
    """
    params = loaded.params
    target = maximized_target(params.k)
    targets = []
    for index in range(SURROGATE_ALIGNMENTS):
        alignment = dict(
            simulate_alignment(
                params.tau,
                params.k,
                params.pi,
                np.random.default_rng(SURROGATE_SEED_OFFSET + index),
                SURROGATE_SITES,
            ).alignment
        )
        targets += [
            target(topology, alignment)
            for topology in enumerate_topologies(sorted(alignment))
        ]
    return {
        "maximized_log_likelihood": Measurement(
            algorithm=(
                "one maximum-likelihood fit per topology per alignment, "
                "topologies in enumeration order and alignments in seed order"
            ),
            value=tuple(targets),
            seed=SURROGATE_SEED_OFFSET,
            budget={
                "alignments": SURROGATE_ALIGNMENTS,
                "n_sites": SURROGATE_SITES,
                "alignment_seed_offset": SURROGATE_SEED_OFFSET,
            },
        )
    }


def potts_landscape(loaded: Fixture) -> PottsLandscape:
    """The declared Potts chain as a single-flip search, at an enumerable length."""
    params = loaded.params
    return PottsLandscape(
        coupling=params.coupling, field=params.field, chain_length=RL_CHAIN_LENGTH
    )


def potts_landscape_baseline(loaded: Fixture) -> dict[str, Measurement]:
    """What an untrained policy achieves on the enumerable Potts chain.

    The optimum is a sum over every configuration and so is the expected
    return of a policy, which is why both are computed here rather than twice
    per training test.
    """
    landscape = potts_landscape(loaded)
    states = list(enumerate_configurations(landscape.n_states, landscape.chain_length))
    untrained = LinearPolicy(2)
    generator = np.random.default_rng(1)
    reached = float(
        np.mean(
            [
                abs(
                    landscape.energy(
                        rollout(
                            landscape, untrained, generator, RL_ROLLOUT_HORIZON, start=s
                        ).states[-1]
                    )
                    - optimum(landscape)[1]
                )
                < 1e-9
                for s in states
            ]
        )
    )
    expected = float(
        np.mean(
            [
                float(
                    exact_expected_return(
                        landscape, untrained, s, RL_RETURN_HORIZON
                    ).detach()
                )
                for s in states
            ]
        )
    )
    chain = {"chain_length": RL_CHAIN_LENGTH, "states": len(states)}
    return {
        "enumerated_optimum": Measurement(
            algorithm="exhaustive enumeration of every configuration of the chain",
            value=float(optimum(landscape)[1]),
            seed=None,
            budget=chain,
        ),
        "untrained_expected_return": Measurement(
            algorithm=(
                "the exact expected return of a LinearPolicy at its initial "
                "weights, averaged over every configuration as a start"
            ),
            value=expected,
            seed=None,
            budget={**chain, "horizon": RL_RETURN_HORIZON},
        ),
        "untrained_reached": Measurement(
            algorithm=(
                "the fraction of configurations from which that policy's "
                "rollout ends at the enumerated optimum"
            ),
            value=reached,
            seed=1,
            budget={**chain, "horizon": RL_ROLLOUT_HORIZON},
        ),
    }


def glass_ground_energy(graph: PottsGraph) -> float:
    """The minimum energy over every configuration, by exhaustive enumeration.

    The oracle the whole fixture rests on: a descent that falls short has
    demonstrably fallen short, rather than possibly having found the answer.
    Walked in chunks so the configuration table is bounded whatever the size.

    Returns
    -------
    float
    """
    nodes = graph.n_nodes
    columns = np.arange(nodes)
    best = math.inf
    for start in range(0, 1 << nodes, GLASS_CHUNK):
        index = np.arange(start, min(start + GLASS_CHUNK, 1 << nodes), dtype=np.int64)
        configurations = ((index[:, None] >> columns[None, :]) & 1).astype(np.int64)
        best = min(
            best, float((-log_weights(graph, GLASS_FIELD, configurations)).min())
        )
    return best


def planted_glass_baseline(loaded: Fixture) -> dict[str, Measurement]:
    """The ground state of the declared glass, and how often descent reaches it.

    The instance is hard *and* refereed, which is the conjunction issue #406
    needs: the energy comes from enumeration and the rate from the baseline
    the instance is meant to defeat.
    """
    params = loaded.params
    (frustration,) = params.glass_frustrations
    glass = params.glass(frustration, np.random.default_rng(params.seed))
    best = glass_ground_energy(glass.graph)
    budget = {
        "restarts": params.glass_restarts,
        "seeds": GLASS_RESTART_SEEDS,
        "nodes": params.glass_nodes,
        "frustration": frustration,
    }
    rates, restarts = [], []
    for offset in range(GLASS_RESTART_SEEDS):
        generator = np.random.default_rng(params.seed + offset)
        found = [
            iterated_conditional_modes(glass.graph, GLASS_FIELD, 2, generator)[1]
            for _ in range(params.glass_restarts)
        ]
        hits = [abs(energy - best) < 1e-9 for energy in found]
        rates.append(float(np.mean(hits)))
        restarts.append(float(any(hits)))
    return {
        "enumerated_ground_energy": Measurement(
            algorithm=(
                f"exhaustive enumeration of all 2**{params.glass_nodes} "
                f"configurations at zero field"
            ),
            value=best,
            seed=None,
            budget={"nodes": params.glass_nodes, "states": 2},
        ),
        "planted_energy": Measurement(
            algorithm=(
                "the energy of the state the couplings were built around, an "
                "upper bound on the ground state and not the ground state here"
            ),
            value=float(glass.planted_energy),
            seed=int(params.seed),
            budget={"nodes": params.glass_nodes, "frustration": frustration},
        ),
        "descent_rate": Measurement(
            algorithm=(
                "single-site descent from each seeded random start, to a "
                "local minimum, averaged over restarts and over seeds"
            ),
            value=float(np.mean(rates)),
            seed=int(params.seed),
            budget=budget,
        ),
        "descent_rate_half_width": Measurement(
            algorithm="half the 95% normal interval of the per-seed rates",
            value=float(1.96 * np.std(rates, ddof=1) / math.sqrt(GLASS_RESTART_SEEDS)),
            seed=int(params.seed),
            budget=budget,
        ),
        "restart_success": Measurement(
            algorithm=(
                "the fraction of seeds where at least one of the restarts "
                "reached the ground state --- random-restart descent, which "
                "is the baseline a policy has actually to beat"
            ),
            value=float(np.mean(restarts)),
            seed=int(params.seed),
            budget=budget,
        ),
    }


@dataclass(frozen=True)
class BaselineSpec:
    """One record: which instance, which modules computed it, and how.

    Parameters
    ----------
    problem, tier : str, Scale
        The fixture the record sits beside.
    modules : tuple[str, ...]
        The dotted names whose import closure the digest covers. The
        *computing* modules, not this script and not the registry that reads
        the record: a change to the search being measured must invalidate
        the number, and a change to the reader must not.
    compute : Callable[[Fixture], dict[str, Measurement]]
        The measurement, given the loaded instance.
    """

    problem: str
    tier: Scale
    modules: tuple[str, ...]
    compute: Callable[[Fixture], dict[str, Measurement]]


#: What the tree policy measurements reach: the environment, the rollouts and
#: the enumeration that referees them, and the simulator that makes the data.
_TREE_POLICY_MODULES = (
    "snakes_and_ladders.learn.policy",
    "snakes_and_ladders.learn.rollout",
    "snakes_and_ladders.search.rl",
    "snakes_and_ladders.search.topology",
    "snakes_and_ladders.sim.simulate",
)

SPECS: tuple[BaselineSpec, ...] = (
    BaselineSpec(
        problem="tree_search",
        tier=Scale.STRESS,
        modules=_TREE_POLICY_MODULES,
        compute=tree_policy_baseline,
    ),
    BaselineSpec(
        problem="tree_search",
        tier=Scale.RELEASE,
        modules=_TREE_POLICY_MODULES,
        compute=tree_policy_baseline,
    ),
    BaselineSpec(
        problem="tree_search",
        tier=Scale.CI,
        modules=(
            "snakes_and_ladders.search.surrogate",
            "snakes_and_ladders.sim.simulate",
        ),
        compute=tree_surrogate_baseline,
    ),
    BaselineSpec(
        problem="planted_glass",
        tier=Scale.CI,
        modules=(
            "snakes_and_ladders.likelihood.potts",
            "snakes_and_ladders.search.alpha_expansion",
            "snakes_and_ladders.sim.canonical",
        ),
        compute=planted_glass_baseline,
    ),
    BaselineSpec(
        problem="potts_chain",
        tier=Scale.CI,
        modules=(
            "snakes_and_ladders.learn.exact",
            "snakes_and_ladders.learn.policy",
            "snakes_and_ladders.learn.potts",
            "snakes_and_ladders.learn.rollout",
        ),
        compute=potts_landscape_baseline,
    ),
)


def compute(spec: BaselineSpec, root: Path = REPO_ROOT) -> Baseline:
    """Run one spec's reference algorithms and build the record they make.

    Returns
    -------
    Baseline
        Not yet written; the caller decides between committing it and
        comparing it with what is committed.
    """
    directory = root / "tests" / "regression" / "fixtures"
    measurements = spec.compute(fixture(spec.problem, spec.tier, directory))
    return Baseline(
        problem=spec.problem,
        tier=spec.tier,
        path=baseline_path(spec.problem, spec.tier, directory),
        modules=spec.modules,
        libraries=tuple(library_versions(BASELINE_LIBRARIES)),
        measurements=measurements,
    )


def differences(computed: Baseline, committed: Baseline) -> list[str]:
    """Every way the two records disagree, in words.

    Parameters
    ----------
    computed : Baseline
        Freshly measured.
    committed : Baseline
        Read from the tree.

    Returns
    -------
    list[str]
        One line per disagreement, empty when they match. A budget, a seed or
        an algorithm is reported like a value: the record is then describing a
        measurement other than the one it holds, which is the way a
        hand-edited record goes wrong.
    """
    found = []
    for name in sorted(set(computed.measurements) | set(committed.measurements)):
        fresh = computed.measurements.get(name)
        stored = committed.measurements.get(name)
        if fresh is None or stored is None:
            found.append(
                f"{name}: {'only computed' if stored is None else 'only committed'}"
            )
            continue
        if fresh.value != stored.value:
            found.append(
                f"{name}: computed {fresh.value!r}, committed {stored.value!r}"
            )
        if (fresh.algorithm, fresh.seed, dict(fresh.budget)) != (
            stored.algorithm,
            stored.seed,
            dict(stored.budget),
        ):
            found.append(
                f"{name}: computed under seed {fresh.seed!r} and budget "
                f"{dict(fresh.budget)!r}, committed under seed {stored.seed!r} "
                f"and budget {dict(stored.budget)!r}"
            )
    if computed.libraries != committed.libraries:
        found.append(
            f"libraries: computed under {list(computed.libraries)}, "
            f"committed under {list(committed.libraries)}"
        )
    return found


def selected(only: Sequence[str]) -> tuple[BaselineSpec, ...]:
    """The specs to run, all of them or the named ``problem/tier`` pairs.

    Returns
    -------
    tuple[BaselineSpec, ...]

    Raises
    ------
    ValueError
        If a name matches no spec. Skipping it silently would report a pass
        over nothing.
    """
    if not only:
        return SPECS
    known = {f"{spec.problem}/{spec.tier}": spec for spec in SPECS}
    missing = sorted(set(only) - known.keys())
    if missing:
        msg = f"no baseline spec for {missing}; known: {sorted(known)}"
        raise ValueError(msg)
    return tuple(known[name] for name in only)


#: Changes outside a record's import closure that can still move its numbers:
#: this script holds the reference algorithms, and the environment decides the
#: arithmetic. Everything else that matters is in the closure, which is exact.
GLOBAL_TRIGGERS: tuple[str, ...] = (
    "infra/baselines.py",
    "pyproject.toml",
    "uv.lock",
    "Cargo.lock",
    "Cargo.toml",
    "rust-toolchain.toml",
    "src/",
)


def reached_by(
    changed: Sequence[str], root: Path = REPO_ROOT
) -> tuple[BaselineSpec, ...]:
    """The specs whose numbers ``changed`` could have moved (issue #460).

    A record's numbers are a function of its fixture, the transitive import
    closure of its computing modules, this script and the environment. A
    change outside all four provably cannot move them, so recomputing then
    would measure nothing --- which is the whole reason the numbers are
    committed. This is the selection the committed digest used to stand in
    for: the digest re-keyed on the same changes, but said only that the tree
    had moved, where recomputing says whether the number did.

    Parameters
    ----------
    changed : Sequence[str]
        Paths relative to the repository root, as ``git diff --name-only``
        gives them.
    root : Path
        The repository root the closures are taken relative to.

    Returns
    -------
    tuple[BaselineSpec, ...]
        In :data:`SPECS` order, so a run is reproducible.
    """
    paths = {path.strip() for path in changed if path.strip()}
    if any(path.startswith(trigger) for path in paths for trigger in GLOBAL_TRIGGERS):
        return SPECS
    directory = root / "tests" / "regression" / "fixtures"
    found = []
    for spec in SPECS:
        closure = {
            str(path.relative_to(root)) for path in module_closure(spec.modules, root)
        }
        closure.add(str(path_of(spec.problem, spec.tier, directory).relative_to(root)))
        closure.add(
            str(baseline_path(spec.problem, spec.tier, directory).relative_to(root))
        )
        if paths & closure:
            found.append(spec)
    return tuple(found)


def check(specs: Sequence[BaselineSpec]) -> None:
    """Recompute ``specs`` and raise if a record no longer holds.

    Raises
    ------
    StaleBaselineError
        If a recomputed value, budget, seed or algorithm disagrees with what
        is committed, or a record is missing or unreadable. This is the
        referee the removed digest stood in for, and it is stronger: the
        digest said the tree had moved, this says the number has.
    """
    log = get_logger(__name__, start_time=time.time())
    stale: list[str] = []
    for spec in specs:
        name = f"{spec.problem}/{spec.tier}"
        with phase(f"measure {name}"):
            computed = compute(spec)
        try:
            committed = read_baseline(computed.path)
        except (FileNotFoundError, ValueError) as error:
            stale.append(f"{name}: {error}")
            continue
        stale += [f"{name}: {line}" for line in differences(computed, committed)]
    if stale:
        msg = "; ".join(stale) + (
            "; recompute with: uv run python infra/baselines.py --write"
        )
        raise StaleBaselineError(msg)
    log.info("%d baseline records match a recomputation", len(specs))


def main(argv: list[str] | None = None) -> int:
    """Write or check the baseline records.

    Returns
    -------
    int
        ``0`` on success; ``1`` if a check found drift or a missing record.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--write",
        action="store_true",
        help="compute and commit the records, instead of comparing with them",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="PROBLEM/TIER",
        help="act on this record alone; repeatable",
    )
    parser.add_argument(
        "--changed",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "act on the records these changed paths could have moved; "
            "repeatable, and reads one path per line from stdin when given "
            "as '-'"
        ),
    )
    arguments = parser.parse_args(argv)
    log = get_logger(__name__, start_time=time.time())

    if arguments.changed and arguments.only:
        log.error("--changed and --only name the records two different ways")
        return 2
    if arguments.changed:
        changed = [
            line
            for item in arguments.changed
            for line in (sys.stdin.read().splitlines() if item == "-" else [item])
        ]
        specs = reached_by(changed)
        if not specs:
            log.info(
                "no baseline record reaches any of the %d changed paths", len(changed)
            )
            return 0
        log.info(
            "%d of %d records reached: %s",
            len(specs),
            len(SPECS),
            ", ".join(f"{spec.problem}/{spec.tier}" for spec in specs),
        )
    else:
        specs = selected(arguments.only)

    if arguments.write:
        for spec in specs:
            with phase(f"measure {spec.problem}/{spec.tier}"):
                computed = compute(spec)
            write_baseline(computed)
            log.info("wrote %s", computed.path)
        return 0
    try:
        check(specs)
    except StaleBaselineError as error:
        log.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
