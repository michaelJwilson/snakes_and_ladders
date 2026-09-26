"""The arms a surrogate is read against, and the paired 2-SE verdicts (issue #1067).

Every arm runs on the same held-out field, and its **gap** is its energy
minus the TRW-S lower bound on that field, so a gap is never negative and a
difference of gaps is a difference of energies. Arms:

- (a) the surrogate's argmax labelling;
- (b) (a) polished by ICM to convergence;
- (c) ICM to convergence from the field argmax;
- (d) alpha-expansion from its default start, the field argmax;
- (e) Swendsen--Wang under a budget of 1000 sweeps' site visits;
- (f) the TRW-S bound, the reference every gap is read against;
- (g) alpha-expansion warm-started at (a);
- (h) alpha-expansion started explicitly at the field argmax, which is (d)'s
  start, so (h) equals (d) bitwise and prices the start argument alone.

A surrogate arm's time includes its forward pass. A verdict is on the paired
difference over fields: *lower* or *higher* where the mean is beyond two
standard errors, *no difference* otherwise.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from sal.backend import Backend
from sal.cost import Cost
from sal.opt.budget import Budget
from sal.sandbox.surrogate.potts.fields import SeedRange
from sal.sandbox.surrogate.potts.model import Grid, Surrogate, hierarchy, predict
from sal.search.alpha_expansion import alpha_expansion
from sal.search.ground_state import ground_state
from sal.search.icm import iterated_conditional_modes
from sal.search.trws import trws
from sal.sim.graph import PottsGraph
from sal.sim.potts import energy

#: Every arm, in the order they run.
ARMS = ("a", "b", "c", "d", "e", "f", "g", "h")

#: The bound arm every gap is read against.
BOUND = "f"

#: The comparisons the trial reported: the surrogate against ICM, expansion
#: and Swendsen--Wang, and the warm start against the cold one.
PAIRS = (("b", "c"), ("b", "d"), ("b", "e"), ("g", "d"), ("g", "h"))

#: Swendsen--Wang's budget, in sweeps of ``n_nodes + 2 n_edges`` site visits.
SW_SWEEPS = 1000

#: ICM's sweep cap, standing in for convergence.
ICM_SWEEPS = 100_000

#: A field's lattice and field, from a side and a seed.
FieldOf = Callable[[int, int], tuple[PottsGraph, np.ndarray]]


@dataclass(frozen=True)
class ArmResult:
    """One arm on one field.

    Parameters
    ----------
    value : float
        The labelling's energy; for the bound arm, the bound.
    seconds : float
        Wall clock, the forward pass included where the arm uses one.
    cycles : int
        Alpha-expansion's sweeps over the labels; zero for any other arm.
    """

    value: float
    seconds: float
    cycles: int = 0


@dataclass(frozen=True)
class Verdict:
    """A paired difference ``x - y`` over fields.

    Parameters
    ----------
    mean, se : float
        The mean difference and its standard error.
    outcome : str
        ``"lower"``, ``"higher"`` or ``"no difference"``, at two standard errors.
    wins : int
        Fields where ``x`` is strictly below ``y``.
    """

    mean: float
    se: float
    outcome: str
    wins: int


def paired(x: np.ndarray, y: np.ndarray) -> Verdict:
    """The verdict on ``x - y``, one entry per field.

    Returns
    -------
    Verdict
    """
    difference = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    mean = float(difference.mean())
    se = float(difference.std(ddof=1) / np.sqrt(difference.size))
    if mean < -2 * se:
        outcome = "lower"
    elif mean > 2 * se:
        outcome = "higher"
    else:
        outcome = "no difference"
    return Verdict(mean, se, outcome, int((difference < 0).sum()))


def _icm(graph: PottsGraph, field: np.ndarray, start: np.ndarray) -> float:
    """ICM's converged energy from ``start``, in index order."""
    return iterated_conditional_modes(
        graph,
        field,
        n_states=field.shape[1],
        rng=np.random.default_rng(0),
        start=np.asarray(start, dtype=np.int64),
        max_sweeps=ICM_SWEEPS,
    ).energy


def _timed[T](call: Callable[[], T]) -> tuple[T, float]:
    """``call()`` and its wall clock."""
    start = time.perf_counter()
    result = call()
    return result, time.perf_counter() - start


def evaluate_field(
    model: Surrogate,
    grids: list[Grid],
    graph: PottsGraph,
    field: np.ndarray,
    seed: int,
    arms: Sequence[str] = ARMS,
) -> dict[str, ArmResult]:
    """Run ``arms`` on one field; ``seed`` drives Swendsen--Wang.

    Returns
    -------
    dict[str, ArmResult]
    """
    n_states = field.shape[1]
    argmax = field.argmax(1).astype(np.int64)
    out: dict[str, ArmResult] = {}
    labelling, forward = _timed(lambda: predict(model, grids, field))
    if "a" in arms:
        out["a"] = ArmResult(energy(graph, field, labelling), forward)
    if "b" in arms:
        value, seconds = _timed(lambda: _icm(graph, field, labelling))
        out["b"] = ArmResult(value, forward + seconds)
    if "c" in arms:
        value, seconds = _timed(lambda: _icm(graph, field, argmax))
        out["c"] = ArmResult(value, seconds)
    expansions = {"d": (None, 0.0), "g": (labelling, forward), "h": (argmax, 0.0)}
    for arm, (start, lead) in expansions.items():
        if arm in arms:
            start_time = time.perf_counter()
            result = alpha_expansion(
                graph,
                field,
                n_states=n_states,
                start=None if start is None else start.copy(),
                backend=Backend.RUST,
            )
            seconds = time.perf_counter() - start_time
            out[arm] = ArmResult(result.energy, lead + seconds, result.cycles)
    if "e" in arms:
        budget = Budget(
            Cost.SITE_VISITS, SW_SWEEPS * (graph.n_nodes + 2 * len(graph.edges))
        )
        run, seconds = _timed(
            lambda: ground_state(
                graph, field, "swendsen-wang", budget, np.random.default_rng(seed)
            )
        )
        out["e"] = ArmResult(run.energy, seconds)
    if BOUND in arms:
        bound, seconds = _timed(lambda: trws(graph, field).bound)
        out[BOUND] = ArmResult(bound, seconds)
    return {arm: out[arm] for arm in ARMS if arm in out}


def evaluate(
    model: Surrogate,
    side: int,
    seeds: SeedRange,
    n_fields: int,
    field_of: FieldOf,
    arms: Sequence[str] = ARMS,
) -> list[dict[str, ArmResult]]:
    """Every arm on draws ``1 .. n_fields`` of ``seeds`` at ``side``; draw 0 warms the compiled kernels and is dropped.

    Returns
    -------
    list[dict[str, ArmResult]]
        One record per field.
    """
    graph, _ = field_of(side, seeds.seed(side, 0))
    grids = hierarchy(graph)
    model.eval()
    records = []
    for index in range(n_fields + 1):
        seed = seeds.seed(side, index)
        graph, field = field_of(side, seed)
        record = evaluate_field(model, grids, graph, field, seed, arms)
        if index > 0:
            records.append(record)
    return records


def gaps(records: Sequence[dict[str, ArmResult]]) -> dict[str, np.ndarray]:
    """Each arm's energy minus the TRW-S bound, per field.

    Returns
    -------
    dict[str, np.ndarray]
        Every arm but the bound, shape ``(n_fields,)``.
    """
    bound = np.array([record[BOUND].value for record in records])
    return {
        arm: np.array([record[arm].value for record in records]) - bound
        for arm in records[0]
        if arm != BOUND
    }


def verdicts(
    records: Sequence[dict[str, ArmResult]],
) -> dict[tuple[str, str], tuple[Verdict, Verdict]]:
    """The gap and the time verdict for each of :data:`PAIRS` both arms of which ran.

    Returns
    -------
    dict[tuple[str, str], tuple[Verdict, Verdict]]
    """
    gap = gaps(records)
    seconds = {
        arm: np.array([record[arm].seconds for record in records]) for arm in records[0]
    }
    return {
        (x, y): (paired(gap[x], gap[y]), paired(seconds[x], seconds[y]))
        for x, y in PAIRS
        if x in gap and y in gap
    }


def load_surrogate(path: str, coupling: float) -> Surrogate:
    """A :class:`~sal.sandbox.surrogate.potts.model.Surrogate` from a state dict :func:`~sal.sandbox.surrogate.potts.train.curriculum` saved.

    Returns
    -------
    Surrogate
    """
    model = Surrogate(coupling)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model
