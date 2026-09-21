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

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np
import torch

from snakes_and_ladders.emissions import CategoricalEmission, EmissionFamily
from snakes_and_ladders.ragged import MINIMUM_LENGTH
from snakes_and_ladders.sim.factor_graph import FactorGraph, from_coupled
from snakes_and_ladders.sim.graph import (
    BoundaryCondition,
    PottsGraph,
    boundary_from_declared,
    lattice_graph,
)
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


@dataclass(frozen=True)
class SpatioSequentialParams:
    """Fully-specified truth for the coupled model.

    Under the seam rule: issue #399's fixtures and emission add its consumers.

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
    self_transition : float | np.ndarray
        The per-class kernel, in one of three forms (issue #658):

        ``float``
            The circulant self-transition rate ``t``, one kernel for the whole
            chain, built by :func:`circulant_transition`. Every construction
            before #658 passed this, and it stays the default.
        ``(K, K)``
            One matrix for the whole chain, row stochastic and **not**
            required to be circulant. A kernel assembled from parts --- a base
            over one latent and a kernel over another, combined into the
            product space --- is not a circulant at any rate, so the rate above
            cannot express one even when it does not vary.
        ``(S - 1, K, K)``
            One matrix per transition, the same matrices the chain recursions
            take since #656.

        The name is the rate's, and the two matrix forms outgrow it. They are
        read as the kernel itself, and how a caller assembled one --- from two
        kernels over a product space, from a distance between sites --- is the
        caller's; this carries the result and does not reconstruct it.
    initial : np.ndarray
        ``Pi``, shape ``(M, K)``, one initial distribution per class.
    emissions : tuple[EmissionFamily, ...]
        One family per class, each over ``K`` states.
    covariate : np.ndarray | None
        What each observation is scored *against* --- an exposure for a rate
        family, a trial count for a bounded one (issue #652). Its leading two
        axes are ``(S, n_nodes)`` as the observations' are, followed by
        whatever the family takes: none for a scalar observation, a channel
        axis for the two-channel count emission, which takes one covariate per
        channel (issue #658). It belongs to the observation and not to the class, so
        every class's family reads the same array. ``None``, the default, is
        the model every construction before this field described: the
        families condition on nothing.

        It lives on the params rather than on the family for the two reasons
        #631 refused it there: ``log_density`` is documented for any leading
        shape, so a family cannot know how to broadcast a stored covariate;
        and ``reestimate`` returns a new family, so a family would carry the
        dataset through every M step. Neither applies here --- these params
        *are* the instance, and the spatial M step rebuilds them with
        ``replace(params, emissions=...)``, which preserves every other field
        by construction.

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
    self_transition: float | np.ndarray
    initial: np.ndarray
    emissions: tuple[EmissionFamily, ...]
    covariate: np.ndarray | None = None
    segments: tuple[int, ...] | None = None

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
        if self.segments is not None:
            # `n_positions` stays the total, so the kernel stack and the
            # observation shape are what they always were; what the segments
            # add is where the chain restarts (issue #666).
            short = [
                (index, length)
                for index, length in enumerate(self.segments)
                if length < MINIMUM_LENGTH
            ]
            if short:
                index, length = short[0]
                msg = (
                    f"segment {index} has length {length}; a segment carries at "
                    f"least {MINIMUM_LENGTH} positions, since one position is an "
                    "initial distribution and no transition (issue #666)"
                )
                raise ValueError(msg)
            if sum(self.segments) != self.n_positions:
                msg = (
                    f"segments sum to {sum(self.segments)} and the chain has "
                    f"{self.n_positions} positions; they must tile it exactly"
                )
                raise ValueError(msg)
        given = np.asarray(self.self_transition, dtype=float)
        steps = max(self.n_positions - 1, 0)
        square = (self.n_states, self.n_states)
        if given.ndim == 0:
            circulant_transition(self.n_states, float(given))  # validates both
        elif given.shape in (square, (steps, *square)):
            if (given < 0).any() or not np.allclose(given.sum(axis=-1), 1.0):
                msg = "every row of every transition must be a distribution"
                raise ValueError(msg)
        else:
            msg = (
                f"self_transition has shape {given.shape}, expected a scalar "
                f"rate, {square} one matrix for the chain, or {(steps, *square)} "
                "one matrix per transition"
            )
            raise ValueError(msg)
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
        if self.covariate is not None:
            covariate = np.asarray(self.covariate, dtype=float)
            expected = (self.n_positions, self.graph.n_nodes)
            # The leading two axes are position and node, as the observations'
            # are. What follows them is the family's own: a scalar-observation
            # family carries none, and the two-channel count emission carries
            # a channel axis, one covariate per channel (issue #658). Checking
            # only the leading two is the same latitude `gated_log_density`
            # gives the observations, and for the same reason -- what the
            # trailing axes mean is the family's to say, not this class's.
            if covariate.shape[:2] != expected:
                msg = (
                    f"covariate has shape {covariate.shape}, expected "
                    f"{expected} in its leading two axes -- one value per "
                    "position and node, as the observations are, with whatever "
                    "trailing axes the emission family takes"
                )
                raise ValueError(msg)
            object.__setattr__(self, "covariate", covariate)

    @property
    def segment_lengths(self) -> tuple[int, ...]:
        """The chain's segments, which is one whole chain where none is declared.

        Undeclared, the model is what it has always been: a single chain of
        `n_positions`. Declared, the chain restarts at each boundary, and the
        number of boundaries is part of the problem rather than of its size.
        """
        return self.segments if self.segments is not None else (self.n_positions,)

    @property
    def transition(self) -> np.ndarray:
        """The circulant transition ``A_m`` of ``eq:joint``, the same for every class.

        Shape ``(K, K)`` where ``self_transition`` is a scalar, and
        ``(S - 1, K, K)`` where it is one rate per step (issue #658) --- the
        two shapes ``forward_backward`` takes, so a chain whose kernel is a
        function of position can be written down here too. A rate per step
        stays a *circulant*, which is what makes it one number rather than
        ``K * (K - 1)``: the model that varies is how sticky the chain is at
        each position, not which states it prefers to move between.
        """
        given = np.asarray(self.self_transition, dtype=float)
        if given.ndim == 0:
            return circulant_transition(self.n_states, float(given))
        return given

    def scaled_graph(self) -> PottsGraph:
        """The graph with ``beta * J`` as couplings: the prior at temperature one."""
        return PottsGraph(
            n_nodes=self.graph.n_nodes,
            edges=self.graph.edges,
            coupling=tuple(self.beta * coupling for coupling in self.graph.coupling),
        )

    #: The fields :func:`snakes_and_ladders.fixtures.load_params` checks are present before
    #: calling :meth:`from_declared`.
    required_fields: ClassVar[frozenset[str]] = _REQUIRED_FIELDS

    @classmethod
    def from_declared(cls, declared: Mapping[str, Any], path: Path, /) -> Self:
        """Load and validate a coupled spatio-sequential fixture yaml.  The
        spatial graph is declared as a lattice --- extent, boundary and a
        uniform coupling --- rather than as an edge list, because that is
        the family the model's enumerable instances come from and an edge
        list would make the file unreadable at the sizes the oracle reaches.

        ``declared`` is the mapping
        :func:`snakes_and_ladders.fixtures.load_params` read from ``path``
        with :attr:`required_fields` present; ``path`` names the file in
        every error.

        Every further check --- non-negative couplings, a row-stochastic
        ``initial``, one emission family per class --- is
        :class:`SpatioSequentialParams`'s own.

        Raises
        ------
        ValueError
            If a required field is missing, or the file declares a number of
            emission families other than ``n_classes``.
        """
        n_classes = int(declared["n_classes"])
        emissions = tuple(
            CategoricalEmission(np.asarray(matrix, dtype=np.float64))
            for matrix in declared["emissions"]
        )
        if len(emissions) != n_classes:
            msg = f"{path}: {len(emissions)} emission families for {n_classes} classes"
            raise ValueError(msg)

        return cls(
            graph=lattice_graph(
                tuple(int(extent) for extent in declared["shape"]),
                boundary_from_declared(path, declared["boundary"]),
                float(declared["coupling"]),
            ),
            n_classes=n_classes,
            n_states=int(declared["n_states"]),
            n_positions=int(declared["n_positions"]),
            beta=float(declared["beta"]),
            self_transition=float(declared["self_transition"]),
            initial=np.asarray(declared["initial"], dtype=np.float64),
            emissions=emissions,
            # Optional: a fixture that declares no segmentation is one chain, which
            # is every instance that existed before #666.
            segments=(
                tuple(int(one) for one in declared["segments"])
                if "segments" in declared
                else None
            ),
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
        ``x_{sn}``, shape ``(S, n_nodes)`` for a scalar-observation family and
        ``(S, n_nodes, ...)`` for one carrying axes of its own --- the
        two-channel count pair of :mod:`snakes_and_ladders.sim.count_pairs` is
        ``(S, n_nodes, 2)`` (issue #672). In the emission families' dtype.
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
        at = 0
        # The chain restarts at each segment: the first position of every one
        # is drawn from the initial distribution, not from the transition out
        # of the position before it, which belongs to another chain entirely
        # (issue #666). With no segments declared this is one pass and the
        # draws are the ones this simulator has always made.
        for length in params.segment_lengths:
            states[m, at] = rng.choice(params.n_states, p=params.initial[m])
            for s in range(at + 1, at + length):
                states[m, s] = rng.choice(
                    params.n_states, p=transition[int(states[m, s - 1])]
                )
            at += length

    columns: dict[int, np.ndarray] = {}
    for m, family in enumerate(params.emissions):
        nodes = np.flatnonzero(labels == m)
        if nodes.size == 0:
            continue
        emitting = np.repeat(states[m], nodes.size)  # (S * n_m,), position-major
        # The covariate is laid out the way the states are --- this class's
        # columns, position-major --- so each draw is made under the value that
        # belongs to it (issue #658). Without this no planted instance exists
        # under a varying covariate, and a fit conditioned on one has nothing
        # to recover.
        flat = family.sample(emitting, rng, covariate=_drawing_covariate(params, nodes))
        # Only the leading axis is the flattened draw; what follows belongs to
        # the family and is carried through rather than assumed absent, which
        # is what lets a pair family be drawn here at all (issue #672).
        drawn_obs = flat.reshape(params.n_positions, nodes.size, *flat.shape[1:])
        for column, node in enumerate(nodes):
            columns[int(node)] = drawn_obs[:, column]
    first = next(iter(columns.values()))
    channels = tuple(first.shape[1:])
    for vertex, column_values in columns.items():
        if tuple(column_values.shape[1:]) != channels:
            msg = (
                f"vertex {vertex} draws an observation of shape "
                f"{column_values.shape[1:]} where another draws {channels}; one "
                f"instance holds one observation shape across its classes"
            )
            raise ValueError(msg)
    observations = np.empty((params.n_positions, n_nodes, *channels), dtype=first.dtype)
    for index, column_values in columns.items():
        observations[:, index] = column_values
    return SimulatedSpatioSequential(
        labels=labels,
        states=states,
        observations=observations,
        params=params,
    )


def _scoring_covariate(params: SpatioSequentialParams) -> torch.Tensor | None:
    """``params.covariate`` ready to score every vertex against.

    The singleton is added only where the covariate has no axes of its own,
    the contract :mod:`snakes_and_ladders.emissions` states and enforces
    (issues #658, #677). Both this and
    :func:`snakes_and_ladders.likelihood.spatio_sequential.covariate_block`
    obey it, and neither restates it: ``sim`` cannot import ``likelihood``, so
    the rule lives where the families that consume the covariate are.
    """
    if params.covariate is None:
        return None
    covariate = params.covariate
    return torch.as_tensor(covariate[..., None] if covariate.ndim == 2 else covariate)


def _drawing_covariate(
    params: SpatioSequentialParams, nodes: np.ndarray
) -> torch.Tensor | None:
    """``params.covariate`` for one class's nodes, flattened as the states are.

    The draw runs over ``(S * n_m,)`` position-major entries, so the covariate
    is selected by the same nodes and flattened the same way. A family whose
    observation carries its own trailing axes keeps them after the flatten,
    which is why the reshape names only the leading axis.

    The singleton is added **only** where the covariate has no axes of its own
    (:mod:`snakes_and_ladders.emissions`). Appending it to a per-channel
    covariate makes ``(S * n_m, 2)`` into ``(S * n_m, 2, 1)``, whose last axis
    names no channel, and
    :func:`snakes_and_ladders.sim.count_pairs.split_covariate` refuses it ---
    which is how a pair family could not be drawn here at all (issue #672).

    Returns
    -------
    torch.Tensor | None
        ``(S * n_m, 1)`` for a scalar-observation family, ``(S * n_m, ...)``
        for one with its own axes, or ``None`` where the params carry none.
    """
    if params.covariate is None:
        return None
    block = params.covariate[:, nodes]
    flat = block.reshape(-1, *block.shape[2:])
    return torch.as_tensor(flat[..., None] if block.ndim == 2 else flat)


def gated_log_density(
    params: SpatioSequentialParams, observations: np.ndarray
) -> np.ndarray:
    """``log P(x_{sn} | k, theta_m)`` for every node, position, class and state.

    The leading two axes are position and node; a family whose observation is
    not a scalar carries its own trailing axes after them --- the two-channel
    count emission of :mod:`snakes_and_ladders.sim.count_pairs` is
    ``(S, n_nodes, 2)`` --- so only the two this function indexes are
    unpacked here.

    ``params.covariate`` reaches every family here, which is the only route a
    caller has to condition a score on an exposure or a trial count
    (issue #652). The family broadcasts a covariate along the states, so it wants a trailing singleton axis; the covariate is stored with the observations' own axes and the singleton is added here, where the observation layout is known. A caller should not have to carry a shape that exists for the family's broadcast.

    Returns
    -------
    np.ndarray
        Shape ``(n_nodes, S, M, K)``, the layout :func:`from_coupled` takes.
    """
    n_positions, n_nodes = observations.shape[:2]
    table = np.empty((n_nodes, n_positions, params.n_classes, params.n_states))
    covariate = _scoring_covariate(params)
    for m, family in enumerate(params.emissions):
        scores = family.log_density(
            torch.as_tensor(observations, dtype=family.observation_dtype),
            covariate=covariate,
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
