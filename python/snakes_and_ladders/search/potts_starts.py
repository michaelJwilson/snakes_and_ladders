"""The ground-state solvers as starts of the `opt.starts` seam, polished by ICM (issue #906).

:class:`~snakes_and_ladders.opt.starts.StartsBenchmark` runs a start, scores
what it offers, and hands it to a polisher, over an
:class:`~snakes_and_ladders.opt.objective.Objective` and a ``theta``. A Potts
labelling is a vector of integers with an energy, and nothing the seam does
with ``theta`` --- score it, detach it, count its bytes, hand it on --- needs
more than a tensor. So the lattice enters the seam through an adapter, not a
second type parameter on it: :class:`PottsObjective` is the energy as an
``Objective`` whose ``theta`` is the labelling, :class:`SolverStart` is one
entry of :data:`~snakes_and_ladders.search.ground_state.METHODS` as an
``Initializer``, and :func:`polish_by_icm` is iterated conditional modes as a
``Polisher``. Generalizing the four protocols over a state type would change
every signature the other two consumers read, for one consumer this module
serves in full.

**The polish is descent to a local minimum.** Iterated conditional modes
from the solver's labelling, one index-order sweep a unit, until a sweep
changes nothing: the counterpart of EM's polish, so what separates two
solvers is where they leave the labelling.

**The objective is the energy, which the seam minimizes.** Differentiability,
which the protocol states and a labelling has none of, is not read by the
seam.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import torch

from snakes_and_ladders.cost import Cost
from snakes_and_ladders.opt import starts as opt_starts
from snakes_and_ladders.opt.budget import Budget
from snakes_and_ladders.opt.objective import Objective
from snakes_and_ladders.opt.starts import refuse_start
from snakes_and_ladders.opt.termination import Termination
from snakes_and_ladders.sample.potts_mcmc import PottsMove
from snakes_and_ladders.sample.schedule import ScheduleParams
from snakes_and_ladders.search.ground_state import (
    METHODS,
    MethodRun,
    Rung,
    run_annealed,
    rung_field,
    warm_anneal,
)
from snakes_and_ladders.search.icm import iterated_conditional_modes
from snakes_and_ladders.search.maxflow import ising_ground_state
from snakes_and_ladders.sim.graph import lattice_graph
from snakes_and_ladders.sim.potts import (
    PottsLatticeParams,
    SpatioOnlyParams,
    energy,
    spatio_only_field,
)
from snakes_and_ladders.track import current


def describe(method: str) -> str:
    """The first paragraph of a solver's docstring, on one line.

    The three annealed entries are one function under three move sets
    (`search.ground_state`), so a ``functools.partial`` is described by the
    function it wraps and the move it fixes.

    Returns
    -------
    str
    """
    target = METHODS[method]
    suffix = ""
    if isinstance(target, functools.partial):
        suffix = f" Move set: {target.keywords['move']}."
        target = target.func
    document = inspect.getdoc(target) or ""
    return " ".join(document.split("\n\n")[0].split()) + suffix


def rung_of(params: PottsLatticeParams, name: str) -> Rung:
    """The :class:`~snakes_and_ladders.search.ground_state.Rung` a size-tilted lattice fixture declares.

    Raises
    ------
    ValueError
        If the fixture's field is a table rather than a size tilt, since the
        structural referee reads the ladder and the sizes.
    """
    if params.alpha is None or params.sizes is None:
        msg = f"{name}: the rung's structure is read off a size-tilted field"
        raise ValueError(msg)
    return Rung(
        name=name,
        graph=lattice_graph(params.shape, params.boundary, params.coupling),
        field=params.field,
        alpha=params.alpha,
        sizes=params.sizes,
        n_states=params.n_states,
        optimum=None,
    )


def spatio_rung(params: SpatioOnlyParams, name: str) -> Rung:
    """The :class:`~snakes_and_ladders.search.ground_state.Rung` a ``spatio_only`` fixture declares, at its own class count (issue #927).

    The field and the ladder are
    :func:`~snakes_and_ladders.search.ground_state.rung_field`'s at the
    fixture's ``n_classes``, so the rung is the declared instance and not a
    re-derivation of it; no exact optimum is known above two states.

    Returns
    -------
    Rung
    """
    field, alpha = rung_field(params, params.n_classes)
    return Rung(
        name=name,
        graph=params.graph,
        field=field,
        alpha=alpha,
        sizes=params.sizes,
        n_states=params.n_classes,
        optimum=None,
    )


def binary_sibling(rung: Rung, name: str) -> Rung:
    """The same lattice and sizes at two states, the ladder's ends, with its exact optimum by graph cut.

    At two states the energy is an Ising model in a field, and the minimum
    cut is its exact ground state (:func:`~snakes_and_ladders.search.maxflow.ising_ground_state`),
    so this rung carries an exact gap where the three-state one carries a
    bracket.

    Returns
    -------
    Rung
    """
    alpha = np.array([rung.alpha[0], rung.alpha[-1]])
    field = spatio_only_field(alpha, rung.sizes)
    _, optimum = ising_ground_state(rung.graph, field)
    return Rung(
        name=name,
        graph=rung.graph,
        field=field,
        alpha=alpha,
        sizes=rung.sizes,
        n_states=2,
        optimum=float(optimum),
    )


@dataclass(frozen=True)
class PottsObjective:
    """The energy of a rung as an :class:`~snakes_and_ladders.opt.objective.Objective` over labellings.

    ``theta`` is the labelling, one ``int64`` per site; the value is
    :func:`~snakes_and_ladders.sim.potts.energy` there, lower being better.

    Parameters
    ----------
    rung : Rung
        The lattice, its field and what referees it.
    """

    rung: Rung

    def initial(self) -> torch.Tensor:
        """Every site at state 0."""
        return torch.zeros(self.rung.n_nodes, dtype=torch.int64)

    def constrain(self, theta: torch.Tensor) -> Mapping[str, torch.Tensor]:
        """``{"labelling": theta}``: a labelling needs no constraint."""
        return {"labelling": theta}

    def theta_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """The labelling under ``"labelling"``."""
        return named["labelling"]

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        """The energy of the labelling ``theta``, a float64 scalar."""
        labelling = np.asarray(theta.numpy(), dtype=np.int64)
        return torch.tensor(
            energy(self.rung.graph, self.rung.field, labelling), dtype=torch.float64
        )


@dataclass(frozen=True)
class SolverStart:
    """One ground-state solver as a start: the labelling it returns at its budget.

    Built per cell from the cell's generator, as
    ``functools.partial(SolverStart, name, budget)`` called with it, so each
    ``(instance, seed)`` draws its own stream (`opt.starts`).

    Parameters
    ----------
    method : str
        A key of :data:`~snakes_and_ladders.search.ground_state.METHODS`.
    budget : Budget
        The solver's own, in :attr:`~snakes_and_ladders.cost.Cost.SITE_VISITS`.
    rng : np.random.Generator
        The cell's generator.
    """

    method: str
    budget: Budget
    rng: np.random.Generator

    def starts(self, objective: Objective) -> list[torch.Tensor]:
        """The solver's labelling on the objective's rung, one start.

        Records the site visits it spent and its own energy into the
        enclosing ``track`` run.

        Returns
        -------
        list[torch.Tensor]
        """
        run = METHODS[self.method](_rung(objective, self.method), self.budget, self.rng)
        current().record(0, solver_energy=run.energy, site_visits=float(run.spent))
        return [torch.as_tensor(np.asarray(run.labelling, dtype=np.int64))]


@dataclass(frozen=True)
class ScheduleStart:
    """One annealed run on a schedule of the caller's, cold from a uniform draw or warm from ICM, as a start (issue #1038).

    :class:`SolverStart` runs a :data:`~snakes_and_ladders.search.ground_state.METHODS`
    entry, whose schedule is fixed; this runs
    :func:`~snakes_and_ladders.search.ground_state.run_annealed` or, with
    ``warm``, :func:`~snakes_and_ladders.search.ground_state.warm_anneal`, on
    ``schedule``. Built per cell as ``functools.partial(ScheduleStart, move,
    budget, schedule, steps, warm)`` called with the generator.

    Parameters
    ----------
    move : PottsMove
        The move set.
    budget : Budget
        In :attr:`~snakes_and_ladders.cost.Cost.SITE_VISITS`.
    schedule : ScheduleParams
        The temperatures, built at the step count.
    steps : int | None
        A step count fixed beforehand, or ``None`` for the budget's rule.
    warm : bool
        Whether ICM's labelling is the chain's start.
    rng : np.random.Generator
        The cell's generator.
    """

    move: PottsMove
    budget: Budget
    schedule: ScheduleParams
    steps: int | None
    warm: bool
    rng: np.random.Generator

    def starts(self, objective: Objective) -> list[torch.Tensor]:
        """The run's labelling on the objective's rung, one start.

        Records the site visits it spent and its own energy into the
        enclosing ``track`` run, as :meth:`SolverStart.starts` does.

        Returns
        -------
        list[torch.Tensor]
        """
        rung = _rung(objective, f"{self.move} on a schedule")
        run = (
            warm_anneal(
                rung, self.budget, self.rng, self.move, self.schedule, steps=self.steps
            )
            if self.warm
            else run_annealed(
                rung,
                self.budget,
                self.rng,
                self.move,
                schedule=self.schedule,
                steps=self.steps,
            )
        )
        current().record(0, solver_energy=run.energy, site_visits=float(run.spent))
        return [torch.as_tensor(np.asarray(run.labelling, dtype=np.int64))]


@dataclass(frozen=True)
class RunStart:
    """Any run shaped as a :data:`~snakes_and_ladders.search.ground_state.METHODS` entry, as a start (issue #1041).

    :class:`SolverStart` looks its run up by name in ``METHODS``; this takes
    the run itself, so an arm with bound keywords --- a
    ``functools.partial`` of a module-level function, which crosses a process
    boundary --- runs through the same seam without a table entry. Built per
    cell as ``functools.partial(RunStart, run, budget)`` called with the
    generator.

    Parameters
    ----------
    run : Callable[[Rung, Budget, np.random.Generator], MethodRun]
        The arm.
    budget : Budget
        In :attr:`~snakes_and_ladders.cost.Cost.SITE_VISITS`.
    rng : np.random.Generator
        The cell's generator.
    """

    run: Callable[[Rung, Budget, np.random.Generator], MethodRun]
    budget: Budget
    rng: np.random.Generator

    def starts(self, objective: Objective) -> list[torch.Tensor]:
        """The run's labelling on the objective's rung, one start.

        Records the site visits it spent and its own energy into the
        enclosing ``track`` run, as :meth:`SolverStart.starts` does.

        Returns
        -------
        list[torch.Tensor]
        """
        run = self.run(_rung(objective, "a run"), self.budget, self.rng)
        current().record(0, solver_energy=run.energy, site_visits=float(run.spent))
        if run.trace:
            # A run's clusters, pooled over its trace: the mean size and the
            # accept rate a cluster move reports beside its energy.
            sizes = [size for counter in run.trace for size in counter.sizes]
            proposals = sum(counter.proposals for counter in run.trace)
            current().record(
                0,
                cluster_mean=float(np.mean(sizes)) if sizes else float("nan"),
                cluster_accept=sum(counter.accepts for counter in run.trace) / proposals
                if proposals
                else float("nan"),
            )
        return [torch.as_tensor(np.asarray(run.labelling, dtype=np.int64))]


def _rung(objective: Objective, who: str) -> Rung:
    """The rung a :class:`PottsObjective` carries, or a refusal naming ``who``."""
    if not isinstance(objective, PottsObjective):
        refuse_start(who, objective, "it is not a Potts energy over labellings")
    return objective.rung


def polish_by_icm(
    objective: Objective,
    theta: torch.Tensor,
    budget: Budget,
    *,
    min_sites: int = 0,
) -> opt_starts.PolishedPoint:
    """Iterated conditional modes from ``theta`` until a sweep changes nothing, one sweep a unit.

    One index-order sweep at a time, recorded into the enclosing ``track``
    run as ``objective``, so the cell's curve carries the descent. The
    descent is converged at the first sweep that leaves the labelling as it
    found it, and that sweep is charged; ``budget.size`` sweeps is the cap.

    ``min_sites`` is the descent's floor
    (:func:`~snakes_and_ladders.search.icm.iterated_conditional_modes`,
    issue #1055), its uniforms drawn from one generator seeded ``0`` across
    the sweeps. ``0``, the default, draws nothing and is the polish before
    the floor existed, bitwise.

    Returns
    -------
    snakes_and_ladders.opt.starts.PolishedPoint

    Raises
    ------
    ValueError
        If ``budget`` is not in :attr:`~snakes_and_ladders.cost.Cost.SWEEPS`,
        or ``min_sites`` is negative or exceeds the sites.
    """
    if budget.unit is not Cost.SWEEPS:
        msg = f"an ICM polish is counted in sweeps, not {budget.unit}"
        raise ValueError(msg)
    rung = _rung(objective, "icm")
    tracked = current()
    labelling = np.asarray(theta.numpy(), dtype=np.int64)
    value = energy(rung.graph, rung.field, labelling)
    # Read by the floor alone: a start is given, and the index order draws no
    # permutation.
    floor_draws = np.random.default_rng(0)
    converged = False
    sweeps = 0
    while sweeps < budget.size:
        settled = iterated_conditional_modes(
            rung.graph,
            rung.field,
            rung.n_states,
            floor_draws,
            start=labelling,
            max_sweeps=1,
            min_sites=min_sites,
        )
        sweeps += 1
        tracked.record(sweeps, objective=settled.energy)
        if np.array_equal(settled.labelling, labelling):
            converged = True
            break
        labelling, value = settled.labelling, settled.energy
    return opt_starts.PolishedPoint(
        value=float(value),
        termination=Termination.after(sweeps, converged=converged),
        theta=torch.as_tensor(labelling),
    )
