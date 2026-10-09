"""One Potts problem held across solves: the graph, the field and the buffers in Rust, one crossing per solve (issue #1413).

A caller that solves one graph many times --- a labelling step refreshing its
field per outer iteration, or several solvers from one start --- paid a
:class:`~sal.sim.graph.PottsGraph` and its compressed adjacency per call, and
the floor's ``max_iterations * n_nodes`` uniforms up front.
:class:`PottsProblem` holds the adjacency and the field in
``sal.oxisal.PottsProblem`` and each method crosses once, returning the
:class:`~sal.search.alpha_expansion.Labelling` its sibling returns.

**Where the shape departs from the HMM classes.** Given graph and field a
Potts solve is a pure function: there are no parameters to fit, so the object
holds the *problem* and its buffers, never a result, and :meth:`set_field`
replaces the one input a labelling loop changes. Composed solvers (an anneal
and its polish, a descent and its floor) run inside one method.

**Oracles, and where the stream differs.** :meth:`PottsProblem.icm` in
:attr:`~sal.search.icm.SweepOrder.INDEX` order without a floor, or with the
draw-free :attr:`~sal.search.icm.FloorPolicy.SMALLEST_FIRST_BEST_FIELD`
floor, is :func:`~sal.search.icm.iterated_conditional_modes` and
:func:`~sal.search.icm.merge_small_labels`' rule bitwise; the uniform floor
and :attr:`~sal.search.icm.SweepOrder.RANDOM` draw from a ChaCha8 stream
seeded by one draw of ``rng``, a chain of the same law on another stream.
:meth:`PottsProblem.merge` under :attr:`MergeGain.FULL` and
:attr:`MergePairs.ALL` is :func:`sal.sample.potts_mcmc.chains.merge_labels`
bitwise; :meth:`PottsProblem.anneal` is
:func:`~sal.sample.potts_mcmc.anneal_potts`' Rust loop on the held graph,
bitwise. Every energy is the Rust loop's (each edge read from both ends and
halved), within ``1e-9`` relative of :func:`~sal.sim.potts.energy`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import numpy as np

from sal.opt.budget import Budget
from sal.opt.termination import Stop, Termination
from sal.sample.potts_mcmc.chains import loop_codes, site_visits
from sal.sample.potts_mcmc.moves import (
    PottsMove,
    PottsMoves,
    Recolour,
    move_set,
    refuse_negative_coupling,
)
from sal.sample.schedule import Polish, TempSchedule
from sal.search.alpha_expansion import Labelling
from sal.search.icm import FloorPolicy, SweepOrder, check_min_sites
from sal.sim.graph import PottsGraph
from sal.sim.potts import SiteField, check_labelling, log_weight_of, site_field


class FloorAt(StrEnum):
    """When the floor runs inside :meth:`PottsProblem.icm` (issue #1413)."""

    SWEEP = "sweep"
    """After every sweep: :func:`~sal.search.icm.iterated_conditional_modes`' placement."""
    EPOCH_GUARDED = "epoch-guarded"
    """Once a sweep is clean; the descent resumes where the floor moved a site, until it moves none or ``max_iterations`` sweeps are spent (the guard)."""


class MergeGain(StrEnum):
    """The boundary gain a whole-label merge credits an edge between the two labels (issue #1413)."""

    FULL = "full"
    """Its coupling once: the energy change exactly (:func:`sal.sample.potts_mcmc.chains.merge_labels`)."""
    HALVED = "halved"
    """Half its coupling: the downstream labelling step's ``merge_assignment`` rule, which misses a merge the energy takes (experiment 040)."""


class MergePairs(StrEnum):
    """The label pairs a whole-label merge considers (issue #1413)."""

    ALL = "all"
    """Every pair of labels in use."""
    ADJACENT = "adjacent"
    """Only pairs sharing at least one edge, as ``merge_assignment`` reads them."""


#: The Rust codes of each enum member.
_ORDER = {
    SweepOrder.INDEX: 0,
    SweepOrder.RANDOM: 1,
    SweepOrder.CHECKERBOARD: 2,
    SweepOrder.WORKLIST: 3,
}
_POLICY = {FloorPolicy.UNIFORM: 0, FloorPolicy.SMALLEST_FIRST_BEST_FIELD: 1}
_AT = {FloorAt.SWEEP: 0, FloorAt.EPOCH_GUARDED: 1}
_STOP = (Stop.CONVERGED, Stop.BUDGET, Stop.INFEASIBLE)


def _held() -> Any:
    """``sal.oxisal.PottsProblem``, imported at the call as the other Rust routes import it."""
    from sal import oxisal

    return oxisal.PottsProblem


class PottsProblem:
    """A Potts graph and field held in Rust across solves (issue #1413).

    Parameters
    ----------
    graph : PottsGraph
        The instance; its compressed adjacency is copied once.
    field : SiteField | np.ndarray
        ``h`` as a log-weight, ``(n_states,)`` or ``(n_nodes, n_states)``.
    n_states : int | None
        Checked against the field's state axis where given.
    """

    def __init__(
        self,
        graph: PottsGraph,
        field: SiteField | np.ndarray,
        *,
        n_states: int | None = None,
    ) -> None:
        self.graph = graph
        rows = self._rows(field, n_states)
        self.n_states = int(rows.shape[1])
        offsets, neighbours, couplings = graph.compressed_adjacency()
        self._problem = _held()(offsets, neighbours, couplings, rows)

    def _rows(self, field: SiteField | np.ndarray, n_states: int | None) -> np.ndarray:
        rows = site_field(
            np.asarray(log_weight_of(field), dtype=float),
            self.graph.n_nodes,
            n_states=n_states,
        )
        return np.ascontiguousarray(rows, dtype=np.float64)

    def set_field(self, field: SiteField | np.ndarray) -> None:
        """Replace the field, same state count; the graph is kept."""
        self._problem.set_field(self._rows(field, self.n_states))

    def energy(self, labelling: np.ndarray) -> float:
        """``E(s)`` of ``labelling``, checked by :func:`~sal.sim.potts.check_labelling`."""
        return float(self._problem.energy(self._labels(labelling)))

    def footprint(self) -> dict[str, int]:
        """Bytes held per structure: adjacency, couplings, field, labels, buffers."""
        names = ("adjacency", "couplings", "field", "labels", "buffers")
        return dict(zip(names, self._problem.footprint(), strict=True))

    def _labels(self, labelling: np.ndarray) -> np.ndarray:
        checked = check_labelling(labelling, self.graph.n_nodes, self.n_states)
        return np.ascontiguousarray(checked, dtype=np.int64)

    def argmax(self) -> Labelling:
        """Each site's first label of greatest field; zero sweeps, converged."""
        labels, value = self._problem.argmax()
        return Labelling(
            labels, float(value), 0, termination=Termination.after(0, converged=True)
        )

    def icm(
        self,
        rng: np.random.Generator,
        *,
        start: np.ndarray | None = None,
        order: SweepOrder = SweepOrder.INDEX,
        min_sites: int = 0,
        policy: FloorPolicy = FloorPolicy.UNIFORM,
        floor_at: FloorAt = FloorAt.SWEEP,
        max_iterations: int = 200,
    ) -> Labelling:
        """Single-site descent to a labelling a clean sweep leaves, with a floor of ``min_sites``.

        A site whose field row and neighbours have not changed since it was
        last visited is skipped: it would recompute the label it holds, so
        the result is the full sweep's, bitwise.

        Parameters
        ----------
        rng : np.random.Generator
            Draws the start where ``start`` is ``None``, then one seed of the
            Rust stream where ``order`` is ``RANDOM`` or a ``UNIFORM`` floor
            runs; nothing otherwise.
        start : np.ndarray | None
            The labelling to descend from, or ``None`` to draw one uniformly.
        order : SweepOrder
            ``INDEX``, ``RANDOM``, ``CHECKERBOARD`` or ``WORKLIST``;
            ``RESIDUAL`` is :func:`~sal.search.icm.iterated_conditional_modes`'
            Python route alone.
        min_sites : int
            The floor; ``0`` dissolves nothing.
        policy : FloorPolicy
            How the floor moves a state's sites.
        floor_at : FloorAt
            When it runs.
        max_iterations : int
            Sweeps allowed; a worklist's visits are ``max_iterations *
            n_nodes``, its sweeps those visits over ``n_nodes``, rounded up.

        Returns
        -------
        Labelling
            Converged where no sweep would change it and no state is below
            the floor, infeasible where a site below it allows no state at
            it, the budget otherwise (:func:`~sal.search.icm.descended`'s
            reading).

        Raises
        ------
        ValueError
            If ``order`` is ``RESIDUAL``, if ``min_sites`` is negative or
            exceeds the sites, if a floor leaves no state to dissolve into, or
            as :func:`~sal.sim.potts.check_labelling` raises.
        """
        check_min_sites(min_sites, self.graph.n_nodes)
        if order not in _ORDER:
            msg = f"{order} order is iterated_conditional_modes' Python route"
            raise ValueError(msg)
        begin = (
            rng.integers(0, self.n_states, size=self.graph.n_nodes)
            if start is None
            else start
        )
        drawn = order is SweepOrder.RANDOM or (
            min_sites > 0 and policy is FloorPolicy.UNIFORM
        )
        seed = int(rng.integers(0, 2**62)) if drawn else 0
        labels, value, sweeps, stop = self._problem.icm(
            self._labels(begin),
            _ORDER[order],
            min_sites,
            _POLICY[policy],
            _AT[floor_at],
            max_iterations,
            seed,
        )
        reason = _STOP[stop]
        return Labelling(
            labels,
            float(value),
            sweeps,
            termination=Termination(reason is Stop.CONVERGED, sweeps, reason),
        )

    def merge(
        self,
        labelling: np.ndarray,
        *,
        gain: MergeGain = MergeGain.FULL,
        pairs: MergePairs = MergePairs.ALL,
    ) -> Labelling:
        """``labelling`` with whole labels merged greedily while a merge's change, under ``gain``, is negative.

        Each round merges the admissible pair ``(u, v)`` of most negative
        ``-(U[u, v] - U[u, u]) - g (B[u, v] + B[v, u]) / 2``, ``g`` one under
        ``FULL`` and a half under ``HALVED``, ties to the lowest ``u`` then
        ``v``; ``v`` must be allowed at every site of ``u``, and under
        ``ADJACENT`` the two must share an edge.

        Returns
        -------
        Labelling
            ``sweeps`` the rounds, the last finding no pair; converged.
        """
        labels, value, rounds = self._problem.merge(
            self._labels(labelling),
            gain is MergeGain.HALVED,
            pairs is MergePairs.ADJACENT,
        )
        return Labelling(
            labels,
            float(value),
            rounds,
            termination=Termination.after(rounds, converged=True),
        )

    def anneal(
        self,
        schedule: TempSchedule,
        rng: np.random.Generator,
        *,
        start: np.ndarray | None = None,
        move: PottsMoves = PottsMove.SINGLE_SITE,
        recolour: Recolour = Recolour.PER_MOVE,
        budget: Budget | None = None,
        polish: Polish | None = None,
        min_sites: int = 0,
    ) -> Labelling:
        """:func:`~sal.sample.potts_mcmc.anneal_potts`' Rust loop on the held graph: the same draws, the same labelling.

        Parameters are ``anneal_potts``' for the moves the Rust loop runs;
        ``schedule`` is a :class:`~sal.sample.schedule.TempSchedule` (no
        ``"auto"``).

        Returns
        -------
        Labelling
            The last stage's labelling (the best, polished, merged), its
            energy, the main steps run, and ``anneal_potts``' termination.

        Raises
        ------
        ValueError
            If a move has no Rust kernel, if ``min_sites`` is given without
            ``polish``, or as ``anneal_potts`` raises.
        """
        site_visits(budget)
        if min_sites and polish is None:
            msg = f"min_sites={min_sites} is the polish's floor; it needs a polish"
            raise ValueError(msg)
        moves = move_set(move, recolour)
        refuse_negative_coupling(moves, self.graph)
        codes = loop_codes(moves, None)
        if codes is None:
            msg = f"{move} has a move the Rust loop does not run"
            raise ValueError(msg)
        state = np.ascontiguousarray(
            rng.integers(0, self.n_states, size=self.graph.n_nodes)
            if start is None
            else self._labels(start),
            dtype=np.int64,
        )
        temperatures = np.array(
            [schedule(index) for index in range(schedule.n_steps)], dtype=np.float64
        )
        ran = self._problem.anneal(
            state,
            codes,
            temperatures,
            0 if budget is None else budget.size,
            schedule.n_steps,
            int(rng.integers(0, 2**62)),
            polish is not None,
            min_sites,
            polish is Polish.ICM_MERGE,
        )
        n_main = int(ran["n_main"])
        if polish is None:
            best = np.asarray(ran["best"])
            termination = Termination.after(n_main, converged=False)
        else:
            reason = _STOP[int(ran["polish_stop"])]
            sweeps = int(ran["polish_sweeps"])
            termination = Termination(reason is Stop.CONVERGED, sweeps, reason)
            best = np.asarray(ran["best_polished"])
        if polish is Polish.ICM_MERGE:
            best = np.asarray(ran["merged"])
            rounds = int(ran["merge_rounds"])
            termination = (
                Termination.after(rounds, converged=True)
                if termination.converged
                else Termination(False, rounds, termination.reason)
            )
        return Labelling(
            best,
            float(ran["stage_energies"][-1]),
            n_main,
            termination=termination,
        )
