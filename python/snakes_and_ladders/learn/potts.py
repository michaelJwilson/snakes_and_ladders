"""A Potts landscape searched by single-site flips: reference environment.

Not phylogenetics, and that is its job -- the same job ``snakes_and_ladders.opt.potts``
does for :class:`~snakes_and_ladders.opt.objective.Objective`. An interface justified by
one application is shaped by it, so the environment the RL machinery is
developed against is deliberately not a tree.

Reusing *this* model rather than inventing a gridworld makes the parallel
structural instead of rhetorical: the same Potts chain is ``snakes_and_ladders.opt``'s
reference **objective**, fitted continuously, and ``snakes_and_ladders.learn``'s
reference **environment**, searched discretely. One model, both halves of
the problem this repository is about. Local search over an Ising or Potts
landscape is classical (Newman & Barkema, ch. 3).

The model is the one ``snakes_and_ladders.opt.potts`` documents, read as an energy to be
maximized over configurations at *known* parameters::

    E(s) = J * sum_i delta(s_i, s_{i+1}) + sum_i h[s_i]

**Known parameters, no inner solve.** ``E`` is evaluated at the fixture's
true ``J`` and ``h``. Nothing here calls ``snakes_and_ladders.opt.fit``. That is issue
#131's simplification stated in the reference instance, and the reason an
episode costs microseconds rather than seconds.

**Two gauges, both benign here, both worth naming.**

* Adding a constant ``c`` to every entry of ``h`` shifts ``E`` by ``L * c``
  for every configuration alike, so it leaves every *reward* unchanged --
  a reward is a difference. The landscape is therefore insensitive to the
  gauge ``snakes_and_ladders.opt.potts`` has to fix, and a test pins that.
* ``delta_energy = J * agreement_delta + field_delta`` exactly, so the two
  features below span the reward. A greedy searcher is the weight vector
  proportional to ``(J, 1)``, which puts the classical baseline *inside*
  the policy class rather than beside it -- and makes "did the agent learn
  the physics" a recovery test against ``J``, not a claim about a loss curve.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence

import numpy as np
import torch

from snakes_and_ladders.opt.potts import PottsParams

# (site, new_state): flip one site to a state it is not already in.
Flip = tuple[int, int]
Configuration = tuple[int, ...]


class PottsLandscape:
    """Single-flip search over a 1-D Potts chain at known parameters.

    Parameters
    ----------
    coupling : float
        The true ``J``.
    field : np.ndarray
        The true ``h``, shape ``(n_states,)``.
    chain_length : int
        Sites in the chain, >= 2. A length-1 chain has no coupling term, so
        the coupling feature would be identically zero and its weight
        unidentifiable.

    Raises
    ------
    ValueError
        If ``chain_length < 2`` or ``field`` is not 1-D with at least two
        entries.
    """

    def __init__(self, coupling: float, field: np.ndarray, chain_length: int) -> None:
        field = np.asarray(field, dtype=np.float64)
        if field.ndim != 1 or field.shape[0] < 2:
            msg = f"field must be 1-D with >= 2 entries, got shape {field.shape}"
            raise ValueError(msg)
        if chain_length < 2:
            msg = f"chain_length must be >= 2, got {chain_length}"
            raise ValueError(msg)
        self._coupling = float(coupling)
        self._field = field
        self._chain_length = chain_length
        self._n_states = int(field.shape[0])
        self._neighbours = _chain_neighbours(chain_length)

    @classmethod
    def on_graph(
        cls,
        coupling: float,
        field: np.ndarray,
        edges: Sequence[tuple[int, int]],
        n_nodes: int,
    ) -> PottsLandscape:
        """The same landscape over an arbitrary graph rather than a chain.

        A chain is the case ``edges == [(0, 1), (1, 2), ...]``, so this is a
        second constructor and not a second class: the energy, the move set,
        the features and the reward are shared, and only the adjacency
        differs. `snakes_and_ladders.sim.graph.lattice_graph` builds the N-D lattices this
        is for, and an adapter in ``snakes_and_ladders.search`` supplies its edges ---
        ``snakes_and_ladders.learn`` may not import ``snakes_and_ladders.sim`` (``learn/CLAUDE.md``),
        so the graph arrives as plain indices.

        Parameters
        ----------
        coupling : float
            ``J``, uniform across edges. `lattice_graph` takes a scalar for
            the same reason: a per-edge coupling would leave the agreement
            feature unable to span the reward, and the greedy searcher would
            fall outside the policy class (``learn/CLAUDE.md``).
        field : np.ndarray
            ``h``, shape ``(n_states,)``.
        edges : Sequence[tuple[int, int]]
            Undirected edges as ``(i, j)``. A repeated pair counts twice, as
            a doubled bond, matching `lattice_graph`'s edge count for a
            periodic dimension of extent 2.
        n_nodes : int
            Number of sites, >= 2.

        Returns
        -------
        PottsLandscape
            The landscape over that graph.

        Raises
        ------
        ValueError
            If ``n_nodes < 2``, or an edge names a node outside
            ``[0, n_nodes)``.
        """
        landscape = cls(coupling, field, n_nodes)
        for first, second in edges:
            if not (0 <= first < n_nodes and 0 <= second < n_nodes):
                msg = f"edge {(first, second)} names a node outside [0, {n_nodes})"
                raise ValueError(msg)
        neighbours: list[list[int]] = [[] for _ in range(n_nodes)]
        for first, second in edges:
            neighbours[first].append(second)
            neighbours[second].append(first)
        landscape._neighbours = tuple(tuple(row) for row in neighbours)
        return landscape

    @classmethod
    def from_params(cls, params: PottsParams) -> PottsLandscape:
        """Build the landscape a Potts fixture describes.

        The same yaml that supplies ``snakes_and_ladders.opt``'s reference objective, read
        as a search problem instead of a fitting problem.
        """
        return cls(params.coupling, params.field, params.chain_length)

    @property
    def n_states(self) -> int:
        """States available at each site."""
        return self._n_states

    @property
    def chain_length(self) -> int:
        """Sites in the chain."""
        return self._chain_length

    def energy(self, state: Configuration) -> float:
        """``E(s)``: the unnormalized log-density of one configuration.

        Absolute energies are reported for tests and for the exhaustive
        oracle; an agent only ever sees differences of them.
        """
        # Each edge once: the adjacency lists it from both ends.
        agreement = 0.5 * sum(
            float(state[node] == state[neighbour])
            for node, row in enumerate(self._neighbours)
            for neighbour in row
        )
        return self._coupling * agreement + float(self._field[list(state)].sum())

    def reset(self, rng: np.random.Generator) -> Configuration:
        """Draw a uniform random configuration."""
        return tuple(
            int(value)
            for value in rng.integers(self._n_states, size=self._chain_length)
        )

    def actions(self, state: Configuration) -> Sequence[Flip]:
        """Every single-site flip to a different state.

        ``chain_length * (n_states - 1)`` of them, so the neighbourhood grows
        with the problem -- which is the property that forces a policy to
        score actions rather than index them.
        """
        return [
            (site, value)
            for site in range(self._chain_length)
            for value in range(self._n_states)
            if value != state[site]
        ]

    def step(self, state: Configuration, action: Flip) -> tuple[Configuration, float]:
        """Apply a flip and return the new configuration and its reward.

        The reward is ``E(s') - E(s)``, computed locally in ``O(1)`` from the
        two affected bonds and the one affected site rather than by
        re-evaluating ``E``. A test pins the two against each other.
        """
        site, value = action
        successor = list(state)
        successor[site] = value
        agreement_delta, field_delta = self._deltas(state, action)
        reward = self._coupling * agreement_delta + field_delta
        return tuple(successor), reward

    def features(self, state: Configuration, actions: Sequence[Flip]) -> torch.Tensor:
        """``(len(actions), 2)``: the change in agreement, and in field.

        These two span the reward exactly, so the policy class contains the
        greedy searcher. Neither is constant across a state's actions in
        general, so neither is the unidentifiable direction the softmax would
        swallow; there is deliberately no third, constant feature.
        """
        rows = [self._deltas(state, action) for action in actions]
        return torch.tensor(rows, dtype=torch.float64).reshape(len(actions), 2)

    def n_features(self) -> int:
        """Two: the agreement change and the field change."""
        return 2

    def is_terminal(self, state: Configuration) -> bool:
        """Whether ``state`` is a local maximum under single flips.

        ``docs/tex`` defines the episode as ending "on a step budget or when
        no move improves the score", which is a property of the state rather
        than of the policy. So an agent can route *around* a barrier by
        accepting a negative reward and climbing elsewhere, but cannot step
        off a local maximum -- the same stopping rule the greedy baseline
        obeys, which is what makes the two comparable at all.
        """
        return not any(
            self._coupling * agreement + field > 0.0
            for agreement, field in (
                self._deltas(state, action) for action in self.actions(state)
            )
        )

    def greedy_weights(self) -> torch.Tensor:
        """The weight vector whose policy is greedy, up to temperature.

        ``delta_energy = J * agreement_delta + field_delta``, so scoring with
        ``(J, 1)`` scores exactly by reward and the argmax is the greedy
        move. Exposed because it is the truth a recovery test compares a
        learned policy against.
        """
        return torch.tensor([self._coupling, 1.0], dtype=torch.float64)

    def _deltas(self, state: Configuration, action: Flip) -> tuple[float, float]:
        """Change in agreeing-neighbour count and in field, for one flip."""
        site, value = action
        agreement = sum(
            float(value == state[neighbour]) - float(state[site] == state[neighbour])
            for neighbour in self._neighbours[site]
        )
        return agreement, float(self._field[value] - self._field[state[site]])


def _chain_neighbours(length: int) -> tuple[tuple[int, ...], ...]:
    """Adjacency of a 1-D open chain: each site next to the ones beside it."""
    return tuple(
        tuple(
            neighbour for neighbour in (site - 1, site + 1) if 0 <= neighbour < length
        )
        for site in range(length)
    )


def enumerate_configurations(
    n_states: int, chain_length: int
) -> Iterator[Configuration]:
    """Every configuration of the chain, in lexicographic order.

    ``n_states ** chain_length`` of them, so this is an oracle for small
    chains and nothing else -- the same role exhaustive topology enumeration
    plays for tree search in ``snakes_and_ladders.search.topology``.
    """
    return itertools.product(range(n_states), repeat=chain_length)


def optimum(landscape: PottsLandscape) -> tuple[Configuration, float]:
    """The global maximum of the landscape, by exhaustive enumeration.

    The independent oracle for "did the search find the best configuration".
    Affordable only because the reference instance is deliberately small.

    Returns
    -------
    tuple[Configuration, float]
        The maximizing configuration and its energy. Ties resolve to the
        lexicographically first, so the answer is deterministic.
    """
    best_state, best_energy = None, -np.inf
    for candidate in enumerate_configurations(
        landscape.n_states, landscape.chain_length
    ):
        energy = landscape.energy(candidate)
        if energy > best_energy:
            best_state, best_energy = candidate, energy
    assert best_state is not None
    return best_state, float(best_energy)
