"""One budgeted comparison, so every equal-budget claim is the same comparison.

Four pull requests measured "does the method beat restarts at the same cost"
by hand, each with its own loop and its own derivation of the restart count
(issue #281). This module is the one loop. A :class:`Budget` names its unit
and its size; a method takes an instance, the budget and a generator and
reports what it spent in that unit; :func:`compare` runs every method on
every instance over the seeds and reports hits against a known optimum or
against the best value any method found; :func:`mcnemar` is the paired test
on two methods' per-instance hits.

Two refusals carry the discipline. A method that spends more than its budget
is refused rather than rounded, since "wins by running longer" is the error
this exists to make impossible. And a budget is in one unit: a sweep and a
likelihood evaluation are not exchangeable, so a comparison across units is
two comparisons.

Model-agnostic, per ``opt/CLAUDE.md``: nothing here knows what an instance is.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

from snakes_and_ladders.parallel import Backend, map_tasks

InstanceT = TypeVar("InstanceT")

# How the (method, instance, seed) cells of a comparison run beside each
# other: processes, because a method is a fit or a sweep loop in Python that
# holds the GIL. The intra-op thread count is left at the process default in
# workers and serial alike, so the two runs reduce in the same order on one
# machine. Each cell already seeds its own generator, so the cells are
# independent by construction. No pool reached 2x at 4 workers at the
# mid-size tier; STATUS.md carries the measurement (issue #344).
_COMPARE_BACKEND: Backend = "processes"
_COMPARE_INTRA_OP_THREADS: int | None = None


class OverspendError(ValueError):
    """A method reported spending more than its budget."""


@dataclass(frozen=True)
class Budget:
    """A cost ceiling in one declared unit.

    Parameters
    ----------
    unit : str
        What is counted: ``"sweeps"``, ``"evaluations"``, ``"fits"``. Named
        so a comparison states what it holds equal.
    size : int
        How many of them, at least one.
    """

    unit: str
    size: int

    def __post_init__(self) -> None:
        if not self.unit:
            msg = "a budget names its unit"
            raise ValueError(msg)
        if self.size < 1:
            msg = f"a budget is at least one {self.unit}, got {self.size}"
            raise ValueError(msg)


@dataclass(frozen=True)
class Outcome:
    """What one run of a method returns.

    Parameters
    ----------
    value : float
        The objective reached; lower is better, so an energy or a negative
        log-likelihood.
    spent : int
        What the run cost, in the budget's unit, as the method counted it.
    """

    value: float
    spent: int


Method = Callable[[InstanceT, Budget, np.random.Generator], Outcome]


def restarts(single: Method[InstanceT], cost: int) -> Method[InstanceT]:
    """The random-restart baseline at the same budget.

    Runs ``single`` with a budget of ``cost`` as many times as the budget
    allows, ``size // cost``, each from the generator's next state, and keeps
    the best. The count is derived from the declared cost rather than
    chosen.

    Raises
    ------
    ValueError
        If ``cost`` is not positive, or at call time if one run would already
        exceed the budget.
    """
    if cost < 1:
        msg = f"a restart costs at least one unit, got {cost}"
        raise ValueError(msg)
    return Restarts(single, cost)


@dataclass(frozen=True)
class Restarts(Generic[InstanceT]):
    """What :func:`restarts` returns: a method, and picklable where ``single`` is.

    A closure would do the same arithmetic but cannot cross a process
    boundary, and :func:`compare` runs its cells on a process pool.
    """

    single: Method[InstanceT]
    cost: int

    def __call__(
        self, instance: InstanceT, budget: Budget, rng: np.random.Generator
    ) -> Outcome:
        if self.cost > budget.size:
            msg = (
                f"one restart costs {self.cost} {budget.unit}, above the budget "
                f"of {budget.size}"
            )
            raise ValueError(msg)
        best = np.inf
        spent = 0
        for _ in range(budget.size // self.cost):
            outcome = self.single(instance, Budget(budget.unit, self.cost), rng)
            best = min(best, outcome.value)
            spent += outcome.spent
        return Outcome(best, spent)


@dataclass(frozen=True)
class Comparison:
    """Every method on every instance, at one budget.

    Parameters
    ----------
    budget : Budget
        What was held equal.
    methods : tuple[str, ...]
        In the order of the rows below.
    best : np.ndarray
        Best value per method and instance over the seeds, shape
        ``(n_methods, n_instances)``.
    spent : np.ndarray
        Largest spend per method and instance over the seeds, same shape; at
        most ``budget.size``, or the comparison would have been refused.
    reference : np.ndarray
        Per instance, the value a hit is scored against: the known optimum
        where one was given, else the best any method found.
    """

    budget: Budget
    methods: tuple[str, ...]
    best: np.ndarray
    spent: np.ndarray
    reference: np.ndarray

    def reached(
        self, tolerance: float = 1e-9, *, relative: bool = False
    ) -> dict[str, np.ndarray]:
        """Per method, which instances it reached the reference on, shape ``(n_instances,)``.

        ``relative`` scales the tolerance by ``|reference|`` per instance ---
        the form a summed log-likelihood needs, since an absolute bound fixed
        at one data size does not transfer to another (``DEV.md``, issue
        #111) --- and leaves it absolute for an energy scored against zero.
        """
        margin = tolerance * np.abs(self.reference) if relative else tolerance
        return {
            name: self.best[row] <= self.reference + margin
            for row, name in enumerate(self.methods)
        }

    def hits(
        self, tolerance: float = 1e-9, *, relative: bool = False
    ) -> dict[str, int]:
        """Instances on which each method reached the reference, within ``tolerance``."""
        return {
            name: int(hit.sum())
            for name, hit in self.reached(tolerance, relative=relative).items()
        }

    def paired_p(
        self,
        first: str,
        second: str,
        tolerance: float = 1e-9,
        *,
        relative: bool = False,
    ) -> float:
        """:func:`mcnemar` on ``first``'s and ``second``'s hits, instance by instance."""
        reached = self.reached(tolerance, relative=relative)
        return mcnemar(reached[first], reached[second])

    def mean_gap(self) -> dict[str, float]:
        """Mean of ``best - reference`` per method, zero when every instance is a hit."""
        return {
            name: float(np.mean(self.best[row] - self.reference))
            for row, name in enumerate(self.methods)
        }

    def table(self) -> str:
        """The markdown row set ``STATUS.md`` cites: hits, mean gap and spend per method."""
        n_instances = self.reference.shape[0]
        hits = self.hits()
        gaps = self.mean_gap()
        lines = [
            f"| method | hits of {n_instances} | mean gap | {self.budget.unit} spent |",
            "| --- | --- | --- | --- |",
        ]
        for row, name in enumerate(self.methods):
            lines.append(
                f"| {name} | {hits[name]}/{n_instances} | {gaps[name]:.3g} | "
                f"{int(self.spent[row].max())} of {self.budget.size} |"
            )
        return "\n".join(lines)


def _run_cell(task: tuple[Method[InstanceT], InstanceT, Budget, int, int]) -> Outcome:
    """One (method, instance, seed) cell, importable so a process pool can run it."""
    method, instance, budget, seed, index = task
    return method(instance, budget, np.random.default_rng([seed, index]))


def compare(
    methods: Mapping[str, Method[InstanceT]],
    instances: Sequence[InstanceT],
    budget: Budget,
    seeds: Sequence[int],
    *,
    workers: int,
    known: Sequence[float] | None = None,
) -> Comparison:
    """Run every method on every instance over the seeds, at one budget.

    Parameters
    ----------
    methods : Mapping[str, Method]
        Name to method. Each is called once per (instance, seed) with the
        generator ``np.random.default_rng([seed, index])``, so a method sees
        the same stream whichever others run beside it. Under ``workers > 1``
        a method must be picklable: a module-level function, or
        :func:`restarts` of one, not a closure.
    instances : Sequence
        The instance set. What an instance is, the methods know.
    budget : Budget
        Held equal across methods.
    seeds : Sequence[int]
        At least one. The best over seeds is what each method is scored on.
    workers : int
        Cells run at once, through :func:`snakes_and_ladders.parallel.map_tasks` on
        a process pool; ``1`` is the serial loop. Explicit rather than
        defaulted, so a comparison does not change with the machine; bitwise
        equal at every count because each cell seeds itself. Measured under
        2x at 4 workers at the mid-size tier (``STATUS.md`` §0), so callers
        pass ``1`` until a measurement on their hardware says otherwise.
    known : Sequence[float] | None
        A known optimum per instance, where one exists (a planted energy, an
        enumerated minimum). ``None`` scores against the best any method found.

    Raises
    ------
    OverspendError
        If any run reports spending more than ``budget.size``.
    ValueError
        If there are no methods, no instances or no seeds, or ``known`` does
        not carry one value per instance.
    """
    if not methods:
        msg = "compare needs at least one method"
        raise ValueError(msg)
    if not instances:
        msg = "compare needs at least one instance"
        raise ValueError(msg)
    if not seeds:
        msg = "compare needs at least one seed"
        raise ValueError(msg)
    if known is not None and len(known) != len(instances):
        msg = f"known carries {len(known)} values for {len(instances)} instances"
        raise ValueError(msg)
    names = tuple(methods)
    best = np.full((len(names), len(instances)), np.inf)
    spent = np.zeros((len(names), len(instances)), dtype=np.int64)
    cells = [
        (row, index, seed)
        for row in range(len(names))
        for index in range(len(instances))
        for seed in seeds
    ]
    outcomes = map_tasks(
        _run_cell,
        [
            (methods[names[row]], instances[index], budget, seed, index)
            for row, index, seed in cells
        ],
        workers=workers,
        backend=_COMPARE_BACKEND,
        intra_op_threads=_COMPARE_INTRA_OP_THREADS,
    )
    for (row, index, _seed), outcome in zip(cells, outcomes, strict=True):
        if outcome.spent > budget.size:
            msg = (
                f"{names[row]} spent {outcome.spent} {budget.unit} on instance "
                f"{index}, above the budget of {budget.size}"
            )
            raise OverspendError(msg)
        best[row, index] = min(best[row, index], outcome.value)
        spent[row, index] = max(spent[row, index], outcome.spent)
    reference = (
        np.asarray(known, dtype=float) if known is not None else best.min(axis=0)
    )
    return Comparison(budget, names, best, spent, reference)


def mcnemar(first: np.ndarray, second: np.ndarray) -> float:
    """Exact two-sided McNemar p-value for two methods' hits on the same instances.

    The paired test ``ROADMAP.md`` §2.4 requires before a method displaces a
    baseline. Only the discordant instances carry evidence, and under the null
    they split evenly, so the p-value is the two-sided binomial tail on the
    smaller count (McNemar, 1947, exact rather than the chi-square
    approximation, which is what 40 starts support). No discordant instance
    means no evidence either way, and the p-value is 1.

    Parameters
    ----------
    first, second : np.ndarray
        Boolean hit per instance, one entry per instance, the same length.

    Returns
    -------
    float
        The p-value, in ``(0, 1]``.

    Raises
    ------
    ValueError
        If the two do not cover the same instances.
    """
    hits_first = np.asarray(first, dtype=bool).reshape(-1)
    hits_second = np.asarray(second, dtype=bool).reshape(-1)
    if hits_first.shape != hits_second.shape:
        msg = (
            f"{hits_first.shape[0]} and {hits_second.shape[0]} instances are not paired"
        )
        raise ValueError(msg)
    only_first = int((hits_first & ~hits_second).sum())
    only_second = int((~hits_first & hits_second).sum())
    discordant = only_first + only_second
    if discordant == 0:
        return 1.0
    smaller = min(only_first, only_second)
    tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / 2.0**discordant
    return min(1.0, 2.0 * tail)
