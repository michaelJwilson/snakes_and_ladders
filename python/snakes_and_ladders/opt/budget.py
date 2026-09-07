"""One budgeted comparison, so every equal-budget claim is the same comparison.

Four pull requests measured "does the method beat restarts at the same cost"
by hand, each with its own loop, its own accounting and its own derivation
of the restart count (issue #281). This module is the one loop. A
:class:`Budget` names its unit and its size; a method is a callable that
takes an instance, the budget and a generator and reports what it spent in
that unit; :func:`compare` runs every method on every instance over the
seeds and reports hits against the best value any method found, or against
a known optimum where the instance carries one.

Two refusals carry the discipline. A method that spends more than its budget
is refused, not rounded, because "wins by running longer" is the error the
utility exists to make impossible. And a budget is in one unit: a sweep and a
likelihood evaluation are not exchangeable, and a harness that converted
between them would be asserting a cost model it has not measured. Two methods
compared under one budget report in its unit, and a comparison across units
is two comparisons.

Model-agnostic, per ``opt/CLAUDE.md``: nothing here knows what an instance is.
The methods that do live beside the models they compare, in the tests.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

InstanceT = TypeVar("InstanceT")


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
    the best. The count is derived from the declared cost rather than chosen,
    which is what every hand-rolled study did with a number it then had to
    justify.

    Raises
    ------
    ValueError
        If ``cost`` is not positive, or at call time if one run would already
        exceed the budget.
    """
    if cost < 1:
        msg = f"a restart costs at least one unit, got {cost}"
        raise ValueError(msg)

    def run(instance: InstanceT, budget: Budget, rng: np.random.Generator) -> Outcome:
        if cost > budget.size:
            msg = f"one restart costs {cost} {budget.unit}, above the budget of {budget.size}"
            raise ValueError(msg)
        best = np.inf
        spent = 0
        for _ in range(budget.size // cost):
            outcome = single(instance, Budget(budget.unit, cost), rng)
            best = min(best, outcome.value)
            spent += outcome.spent
        return Outcome(best, spent)

    return run


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
        The largest spend per method and instance over the seeds, same shape;
        every entry is at most ``budget.size`` or the comparison would have
        been refused.
    reference : np.ndarray
        Per instance, the value a hit is scored against: the known optimum
        where one was given, else the best any method found.
    """

    budget: Budget
    methods: tuple[str, ...]
    best: np.ndarray
    spent: np.ndarray
    reference: np.ndarray

    def hits(self, tolerance: float = 1e-9) -> dict[str, int]:
        """Instances on which each method reached the reference, within ``tolerance``."""
        return {
            name: int((self.best[row] <= self.reference + tolerance).sum())
            for row, name in enumerate(self.methods)
        }

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


def compare(
    methods: Mapping[str, Method[InstanceT]],
    instances: Sequence[InstanceT],
    budget: Budget,
    seeds: Sequence[int],
    *,
    known: Sequence[float] | None = None,
) -> Comparison:
    """Run every method on every instance over the seeds, at one budget.

    Parameters
    ----------
    methods : Mapping[str, Method]
        Name to method. Each is called once per (instance, seed) with the
        generator ``np.random.default_rng([seed, index])``, so a method sees
        the same stream whichever other methods run beside it.
    instances : Sequence
        The instance set. What an instance is, the methods know.
    budget : Budget
        Held equal across methods.
    seeds : Sequence[int]
        At least one. The best over seeds is what each method is scored on.
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
    for row, name in enumerate(names):
        for index, instance in enumerate(instances):
            for seed in seeds:
                outcome = methods[name](
                    instance, budget, np.random.default_rng([seed, index])
                )
                if outcome.spent > budget.size:
                    msg = (
                        f"{name} spent {outcome.spent} {budget.unit} on instance "
                        f"{index}, above the budget of {budget.size}"
                    )
                    raise OverspendError(msg)
                best[row, index] = min(best[row, index], outcome.value)
                spent[row, index] = max(spent[row, index], outcome.spent)
    reference = (
        np.asarray(known, dtype=float) if known is not None else best.min(axis=0)
    )
    return Comparison(budget, names, best, spent, reference)
