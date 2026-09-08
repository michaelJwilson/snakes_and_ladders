"""Single-position search over a hidden Markov state path at known parameters.

The third instance of :class:`~snakes_and_ladders.learn.environment.Environment`, beside
the Potts landscape and (in ``snakes_and_ladders.search``) the tree. It exists for the same
reason ``snakes_and_ladders.opt`` keeps a Potts chain and an HMM beside branch lengths: an
interface justified by one model is shaped by that model, and the only way to
show that :class:`Environment` is not shaped by lattices is to put something
that is not a lattice behind it.

**The parameters arrive as plain arrays.** ``learn/CLAUDE.md`` forbids
importing ``snakes_and_ladders.sim``, so this module never sees an ``HmmParams``; a caller
in ``snakes_and_ladders.search`` --- which may import both halves --- unpacks one. That is
the same reason :meth:`PottsLandscape.on_graph` takes an edge list rather
than a ``PottsGraph``.

**The objective is the joint log-probability of a path and the observations
it explains**, not the marginal likelihood. Searching over paths at fixed
parameters is the decoding problem, so the state is a path, the reward is what
changing one position buys, and the maximum is the one Viterbi computes
(issue #175) and brute-force enumeration confirms at small lengths.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence

import numpy as np
import torch

# (position, new_state): change one position to a state it is not already in.
Revision = tuple[int, int]
Path = tuple[int, ...]


class StatePathLandscape:
    """Search over hidden state paths, one position at a time.

    Parameters
    ----------
    log_initial : np.ndarray
        ``log pi``, shape ``(n_states,)``.
    log_transition : np.ndarray
        ``log A``, shape ``(n_states, n_states)``; entry ``(i, j)`` is the
        log-probability of moving from ``i`` to ``j``.
    log_emission : np.ndarray
        ``log B``, shape ``(n_states, n_symbols)``.
    observations : np.ndarray
        One observed sequence, shape ``(length,)``, entries in
        ``[0, n_symbols)``.

    Raises
    ------
    ValueError
        If the shapes disagree, if the sequence is shorter than two positions
        --- a length-1 path has no transition term, so the transition feature
        would be identically zero and its weight unidentifiable, exactly as a
        length-1 Potts chain has no coupling --- or if an observation lies
        outside the emission alphabet.
    """

    def __init__(
        self,
        log_initial: np.ndarray,
        log_transition: np.ndarray,
        log_emission: np.ndarray,
        observations: np.ndarray,
    ) -> None:
        log_initial = np.asarray(log_initial, dtype=np.float64)
        log_transition = np.asarray(log_transition, dtype=np.float64)
        log_emission = np.asarray(log_emission, dtype=np.float64)
        observations = np.asarray(observations, dtype=np.int64)

        n_states = int(log_initial.shape[0])
        if log_initial.ndim != 1 or n_states < 2:
            msg = f"log_initial must be 1-D with >= 2 entries, got {log_initial.shape}"
            raise ValueError(msg)
        if log_transition.shape != (n_states, n_states):
            msg = (
                f"log_transition must have shape {(n_states, n_states)}, got "
                f"{log_transition.shape}"
            )
            raise ValueError(msg)
        if log_emission.ndim != 2 or log_emission.shape[0] != n_states:
            msg = (
                f"log_emission must have shape (n_states, n_symbols) with "
                f"n_states {n_states}, got {log_emission.shape}"
            )
            raise ValueError(msg)
        if observations.ndim != 1 or observations.shape[0] < 2:
            msg = (
                f"observations must be 1-D with >= 2 entries, got {observations.shape}"
            )
            raise ValueError(msg)
        n_symbols = int(log_emission.shape[1])
        if int(observations.min()) < 0 or int(observations.max()) >= n_symbols:
            msg = (
                f"observations must lie in [0, {n_symbols}), got "
                f"[{int(observations.min())}, {int(observations.max())}]"
            )
            raise ValueError(msg)

        self._log_initial = log_initial
        self._log_transition = log_transition
        self._log_emission = log_emission
        self._observations = observations
        self._n_states = n_states
        self._length = int(observations.shape[0])

    @property
    def n_states(self) -> int:
        """Hidden states available at each position."""
        return self._n_states

    @property
    def length(self) -> int:
        """Positions in the path."""
        return self._length

    def energy(self, state: Path) -> float:
        """Joint log-probability of the path and the observed sequence.

        ``log pi[s_0] + sum_t log A[s_{t-1}, s_t] + sum_t log B[s_t, o_t]``.
        Reported in full for the exhaustive oracle; an agent sees only
        differences of it.
        """
        total = float(self._log_initial[state[0]])
        for previous, current in itertools.pairwise(state):
            total += float(self._log_transition[previous, current])
        for position, hidden in enumerate(state):
            total += float(self._log_emission[hidden, self._observations[position]])
        return total

    def reset(self, rng: np.random.Generator) -> Path:
        """Draw a uniform random path."""
        return tuple(
            int(value) for value in rng.integers(self._n_states, size=self._length)
        )

    def actions(self, state: Path) -> Sequence[Revision]:
        """Every single-position change to a different hidden state.

        ``length * (n_states - 1)`` of them, so the neighbourhood grows with
        the sequence rather than being a fixed action set --- the property
        that forces a policy to score actions rather than index them.
        """
        return [
            (position, value)
            for position in range(self._length)
            for value in range(self._n_states)
            if value != state[position]
        ]

    def step(self, state: Path, action: Revision) -> tuple[Path, float]:
        """Apply a revision and return the new path and its reward.

        The reward is the change in joint log-probability, computed from the
        at most two transitions and the one emission the change touches
        rather than by re-evaluating :meth:`energy`. A test pins the two
        against each other, because an ``O(1)`` update that disagrees with a
        full evaluation is the failure this class is most exposed to.
        """
        position, value = action
        successor = list(state)
        successor[position] = value
        transition_delta, emission_delta = self._deltas(state, action)
        return tuple(successor), transition_delta + emission_delta

    def features(self, state: Path, actions: Sequence[Revision]) -> torch.Tensor:
        """``(len(actions), 2)``: the change in transition and in emission terms.

        The two span the reward exactly --- their sum *is* it --- so the
        greedy searcher sits inside the policy class at weights ``(1, 1)``,
        which ``learn/CLAUDE.md`` requires. Neither is constant across a
        state's actions in general, so neither is the direction the softmax
        would swallow, and there is deliberately no third constant feature.
        """
        rows = [self._deltas(state, action) for action in actions]
        return torch.tensor(rows, dtype=torch.float64).reshape(len(actions), 2)

    def n_features(self) -> int:
        """Two: the transition change and the emission change."""
        return 2

    def is_terminal(self, state: Path) -> bool:
        """Whether no single-position change raises the joint log-probability."""
        return not any(
            transition + emission > 0.0
            for transition, emission in (
                self._deltas(state, action) for action in self.actions(state)
            )
        )

    def greedy_weights(self) -> torch.Tensor:
        """The weight vector whose policy is greedy, up to temperature.

        The reward is the plain sum of the two features, so ``(1, 1)`` scores
        exactly by reward. Unlike the Potts landscape, whose weights are
        ``(J, 1)``, this carries no parameter --- which makes it the cleaner
        of the two recovery targets.
        """
        return torch.ones(2, dtype=torch.float64)

    def _deltas(self, state: Path, action: Revision) -> tuple[float, float]:
        """Change in the transition terms and in the emission term."""
        position, value = action
        transition = 0.0
        if position > 0:
            previous = state[position - 1]
            transition += float(
                self._log_transition[previous, value]
                - self._log_transition[previous, state[position]]
            )
        else:
            transition += float(
                self._log_initial[value] - self._log_initial[state[position]]
            )
        if position + 1 < self._length:
            following = state[position + 1]
            transition += float(
                self._log_transition[value, following]
                - self._log_transition[state[position], following]
            )
        symbol = int(self._observations[position])
        emission = float(
            self._log_emission[value, symbol]
            - self._log_emission[state[position], symbol]
        )
        return transition, emission


def enumerate_paths(n_states: int, length: int) -> Iterator[Path]:
    """Every hidden path, in lexicographic order.

    ``n_states ** length`` of them, so this is an oracle for short sequences
    and nothing else --- the role exhaustive topology enumeration plays for
    tree search, and the reference issue #175's Viterbi decoder is pinned
    against.
    """
    return itertools.product(range(n_states), repeat=length)


def optimum(landscape: StatePathLandscape) -> tuple[Path, float]:
    """The maximum-a-posteriori path, by exhaustive enumeration.

    Returns
    -------
    tuple[Path, float]
        The maximizing path and its joint log-probability. Ties resolve to
        the lexicographically first, so the answer is deterministic.
    """
    best_path: Path | None = None
    best_energy = -float("inf")
    for path in enumerate_paths(landscape.n_states, landscape.length):
        energy = landscape.energy(path)
        if energy > best_energy:
            best_path, best_energy = path, energy
    assert best_path is not None
    return best_path, best_energy
