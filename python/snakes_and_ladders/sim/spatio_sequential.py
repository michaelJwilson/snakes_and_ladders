"""The coupled spatio-sequential model of issue #290: parameters and simulator.

A Potts prior over class labels on a spatial graph, one hidden Markov chain
per class, and a *gated* emission: the observation at node ``n`` and position
``s`` is emitted by the chain of the class ``n`` belongs to. The joint is
``eq:joint`` in the textbook, and every piece here is a
composition of parts the repository already holds to their distributions --
the single-site heat bath for the labels, a categorical draw per transition
for the chains, and an :class:`~snakes_and_ladders.emissions.EmissionFamily`
per class for the observations -- under one generator, per ``sim/CLAUDE.md``.

What draws data lives here; what evaluates it, the enumeration oracle, lives
in :mod:`snakes_and_ladders.likelihood.spatio_sequential`, because ``sim`` may
not import ``likelihood``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from snakes_and_ladders.emissions import CategoricalEmission, EmissionFamily
from snakes_and_ladders.fixtures import load_declared
from snakes_and_ladders.sim.factor_graph import FactorGraph, from_coupled
from snakes_and_ladders.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from snakes_and_ladders.sim.potts import simulate_potts


def circulant_transition(n_states: int, self_transition: float) -> np.ndarray:
    """The circulant transition: ``t`` on the diagonal, the rest spread evenly.

    Raises
    ------
    ValueError
        If ``n_states < 2`` or ``self_transition`` is not strictly inside
        ``(0, 1)``: at either end a row of the matrix is a point mass and the
        rate is not a parameter the data could move.
    """
    if n_states < 2:
        msg = f"a chain needs at least two states, got {n_states}"
        raise ValueError(msg)
    if not 0.0 < self_transition < 1.0:
        msg = f"self_transition must lie strictly in (0, 1), got {self_transition}"
        raise ValueError(msg)
    off = (1.0 - self_transition) / (n_states - 1)
    return np.full((n_states, n_states), off) + (self_transition - off) * np.eye(
        n_states
    )


@dataclass(frozen=True)
class SpatioSequentialParams:
    """Fully-specified truth for the coupled model.

    Parameters
    ----------
    graph : PottsGraph
        The spatial graph; its couplings are ``J >= 0``.
    n_classes : int
        ``M``, the number of class labels.
    n_states : int
        ``K``, hidden states per chain.
    n_positions : int
        ``S``, the length of every chain.
    beta : float
        Inverse temperature on the spatial prior of ``eq:joint``.
    self_transition : float
        The circulant self-transition rate ``t``, shared by every class.
    initial : np.ndarray
        ``Pi``, shape ``(M, K)``, one initial distribution per class.
    emissions : tuple[EmissionFamily, ...]
        One family per class, each over ``K`` states.

    Raises
    ------
    ValueError
        If a coupling is negative (the prior would be antiferromagnetic and
        the cluster moves refuse it), ``initial`` is not ``(M, K)`` row
        stochastic, there is not one family per class, or a family does not
        carry ``K`` states.
    """

    graph: PottsGraph
    n_classes: int
    n_states: int
    n_positions: int
    beta: float
    self_transition: float
    initial: np.ndarray
    emissions: tuple[EmissionFamily, ...]

    def __post_init__(self) -> None:
        if any(coupling < 0.0 for coupling in self.graph.coupling):
            msg = "the spatial prior is ferromagnetic: every coupling must be >= 0"
            raise ValueError(msg)
        if self.beta < 0.0:
            msg = f"beta must be >= 0, got {self.beta}"
            raise ValueError(msg)
        if self.n_positions < 1:
            msg = f"a chain needs at least one position, got {self.n_positions}"
            raise ValueError(msg)
        circulant_transition(self.n_states, self.self_transition)  # validates both
        initial = np.asarray(self.initial, dtype=float)
        if initial.shape != (self.n_classes, self.n_states):
            msg = (
                f"initial has shape {initial.shape}, expected "
                f"({self.n_classes}, {self.n_states})"
            )
            raise ValueError(msg)
        if (initial < 0).any() or not np.allclose(initial.sum(axis=1), 1.0):
            msg = "every row of initial must be a distribution"
            raise ValueError(msg)
        object.__setattr__(self, "initial", initial)
        if len(self.emissions) != self.n_classes:
            msg = (
                f"{len(self.emissions)} emission families for {self.n_classes} classes"
            )
            raise ValueError(msg)
        for m, family in enumerate(self.emissions):
            if family.n_states != self.n_states:
                msg = (
                    f"class {m}'s emission family has {family.n_states} states, "
                    f"expected {self.n_states}"
                )
                raise ValueError(msg)

    @property
    def transition(self) -> np.ndarray:
        """The circulant transition ``A_m`` of ``eq:joint``, the same for every class."""
        return circulant_transition(self.n_states, self.self_transition)

    def scaled_graph(self) -> PottsGraph:
        """The graph with ``beta * J`` as couplings: the prior at temperature one."""
        return PottsGraph(
            n_nodes=self.graph.n_nodes,
            edges=self.graph.edges,
            coupling=tuple(self.beta * coupling for coupling in self.graph.coupling),
        )


@dataclass(frozen=True)
class SimulatedSpatioSequential:
    """One draw from the model.

    Parameters
    ----------
    labels : np.ndarray
        ``l_n``, shape ``(n_nodes,)``, entries in ``[0, M)``.
    states : np.ndarray
        ``k_{s,m}``, shape ``(M, S)``, entries in ``[0, K)``; every class's
        chain is drawn, whether or not a node belongs to it.
    observations : np.ndarray
        ``x_{sn}``, shape ``(S, n_nodes)``, in the emission families' dtype.
    params : SpatioSequentialParams
        The generating truth.
    """

    labels: np.ndarray
    states: np.ndarray
    observations: np.ndarray
    params: SpatioSequentialParams


def simulate_spatio_sequential(
    params: SpatioSequentialParams,
    rng: np.random.Generator,
    *,
    labels: np.ndarray | None = None,
    burn_in: int = 500,
) -> SimulatedSpatioSequential:
    """Draw labels, chains and observations under one generator.

    Parameters
    ----------
    params : SpatioSequentialParams
        The truth.
    rng : np.random.Generator
        Passed in rather than seeded here (``sim/CLAUDE.md``, issue #240).
    labels : np.ndarray | None
        Planted labels, shape ``(n_nodes,)``. ``None`` draws them from the
        Potts prior at ``beta`` with no field, by the single-site heat bath
        :func:`snakes_and_ladders.sim.potts.simulate_potts` runs. Planting is
        for recovery studies, where the truth must be known and the prior's
        own draw may not separate the classes.
    burn_in : int
        Sweeps the label chain runs before its state is taken.

    Returns
    -------
    SimulatedSpatioSequential
    """
    n_nodes = params.graph.n_nodes
    if labels is None:
        drawn = simulate_potts(
            params.scaled_graph(),
            np.zeros(params.n_classes),
            rng,
            n_samples=1,
            burn_in=burn_in,
        )
        labels = np.asarray(drawn.configurations[0], dtype=np.int64)
    else:
        labels = np.asarray(labels, dtype=np.int64)
        if (
            labels.shape != (n_nodes,)
            or (labels < 0).any()
            or (labels >= params.n_classes).any()
        ):
            msg = f"planted labels must be {n_nodes} entries in [0, {params.n_classes})"
            raise ValueError(msg)

    transition = params.transition
    states = np.empty((params.n_classes, params.n_positions), dtype=np.int64)
    for m in range(params.n_classes):
        states[m, 0] = rng.choice(params.n_states, p=params.initial[m])
        for s in range(1, params.n_positions):
            states[m, s] = rng.choice(
                params.n_states, p=transition[int(states[m, s - 1])]
            )

    columns: dict[int, np.ndarray] = {}
    for m, family in enumerate(params.emissions):
        nodes = np.flatnonzero(labels == m)
        if nodes.size == 0:
            continue
        emitting = np.repeat(states[m], nodes.size)  # (S * n_m,), position-major
        drawn_obs = family.sample(emitting, rng).reshape(params.n_positions, nodes.size)
        for column, node in enumerate(nodes):
            columns[int(node)] = drawn_obs[:, column]
    first = next(iter(columns.values()))
    observations = np.empty((params.n_positions, n_nodes), dtype=first.dtype)
    for index, column_values in columns.items():
        observations[:, index] = column_values
    return SimulatedSpatioSequential(
        labels=labels,
        states=states,
        observations=observations,
        params=params,
    )


def gated_log_density(
    params: SpatioSequentialParams, observations: np.ndarray
) -> np.ndarray:
    """``log P(x_{sn} | k, theta_m)`` for every node, position, class and state.

    Returns
    -------
    np.ndarray
        Shape ``(n_nodes, S, M, K)``, the layout :func:`from_coupled` takes.
    """
    n_positions, n_nodes = observations.shape
    table = np.empty((n_nodes, n_positions, params.n_classes, params.n_states))
    for m, family in enumerate(params.emissions):
        scores = family.log_density(
            torch.as_tensor(observations, dtype=family.observation_dtype)
        )  # (S, n_nodes, K)
        table[:, :, m, :] = scores.detach().numpy().transpose(1, 0, 2)
    return table


def coupled_factor_graph(
    params: SpatioSequentialParams, observations: np.ndarray
) -> FactorGraph:
    """The model on ``observations`` as the factor graph of ``eq:joint``."""
    transition = params.transition
    return from_coupled(
        params.graph,
        params.beta,
        [np.log(params.initial[m]) for m in range(params.n_classes)],
        [np.log(transition) for _ in range(params.n_classes)],
        gated_log_density(params, observations),
    )


def canonical_spatio_sequential() -> SpatioSequentialParams:
    """The instance small enough to enumerate: a 2x2 open lattice, ``M = K = 2``, ``S = 6``.

    ``2**4`` labellings times ``2**12`` joint chain paths is 65,536 states,
    inside the enumeration limit. Two categorical families over three symbols,
    separated so that a draw at ``S = 6`` identifies the classes.
    """
    return SpatioSequentialParams(
        graph=lattice_graph((2, 2), BoundaryCondition.OPEN, 1.0),
        n_classes=2,
        n_states=2,
        n_positions=6,
        beta=0.6,
        self_transition=0.7,
        initial=np.array([[0.6, 0.4], [0.3, 0.7]]),
        emissions=(
            CategoricalEmission(np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])),
            CategoricalEmission(np.array([[0.1, 0.1, 0.8], [0.45, 0.45, 0.1]])),
        ),
    )


_REQUIRED_FIELDS = frozenset(
    {
        "seed",
        "shape",
        "boundary",
        "coupling",
        "n_classes",
        "n_states",
        "n_positions",
        "beta",
        "self_transition",
        "initial",
        "emissions",
    }
)


def load_spatio_sequential_params(path: Path) -> SpatioSequentialParams:
    """Load and validate a coupled spatio-sequential fixture yaml.

    The spatial graph is declared as a lattice --- extent, boundary and a
    uniform coupling --- rather than as an edge list, because that is the
    family the model's enumerable instances come from and an edge list would
    make the file unreadable at the sizes the oracle reaches.

    Parameters
    ----------
    path : Path
        Path to the yaml file.

    Returns
    -------
    SpatioSequentialParams
        The parsed, validated truth. Every further check --- non-negative
        couplings, a row-stochastic ``initial``, one emission family per class
        --- is :class:`SpatioSequentialParams`'s own.

    Raises
    ------
    ValueError
        If a required field is missing, or the file declares a number of
        emission families other than ``n_classes``.
    """
    raw = load_declared(path, _REQUIRED_FIELDS)

    n_classes = int(raw["n_classes"])
    emissions = tuple(
        CategoricalEmission(np.asarray(matrix, dtype=np.float64))
        for matrix in raw["emissions"]
    )
    if len(emissions) != n_classes:
        msg = f"{path}: {len(emissions)} emission families for {n_classes} classes"
        raise ValueError(msg)

    return SpatioSequentialParams(
        graph=lattice_graph(
            tuple(int(extent) for extent in raw["shape"]),
            BoundaryCondition(str(raw["boundary"])),
            float(raw["coupling"]),
        ),
        n_classes=n_classes,
        n_states=int(raw["n_states"]),
        n_positions=int(raw["n_positions"]),
        beta=float(raw["beta"]),
        self_transition=float(raw["self_transition"]),
        initial=np.asarray(raw["initial"], dtype=np.float64),
        emissions=emissions,
    )
