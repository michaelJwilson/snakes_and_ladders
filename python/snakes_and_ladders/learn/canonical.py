"""Canonical control problems, with an exact optimum written a second way.

Issue #597, the control arm of #596. Every environment the learners here are
measured on is a research problem: a Potts landscape, a hidden path, a
topology search. When a learner ties with greedy on one of those, two readings
are open --- the problem is hard, or the learner is broken --- and nothing in
the tree separates them. These fixtures do: each has an optimum known from
outside, so a learner that fails here is wrong, and a learner that passes here
and ties there has measured the problem.

Four problems, each exposing something the others do not:

* **Chain** --- the greedy optimum is the near reward and the true optimum is
  at the far end, so it separates a method that explores from one that does
  not;
* **Gridworld** --- credit assignment over a long horizon, the plain case the
  rest is read against;
* **Cliff walk** --- the optimal path runs along a row of terminal penalties,
  which is where an on-policy and an off-policy learner part (#597's second
  pull request, where SARSA and Q-learning are compared here);
* **Towers of Hanoi** --- a combinatorial state space whose optimal solution
  length is ``2^d - 1`` exactly, the shape closest to the research problems
  and the only fixture whose oracle is an integer.

**The oracle is a second computation, not a second opinion.**
:func:`value_iteration` sweeps the Bellman optimality operator over an
explicit state set to a residual; `learn.exact.exact_optimal_value` recurses
over trajectories to a horizon. They share no line of code, and the suite
holds them to each other to 1e-12 --- which is what root `CLAUDE.md` means by
an oracle, rather than running one implementation twice.

**What does not fit, and is reported rather than forced.**
:class:`~snakes_and_ladders.learn.environment.Environment` declares
``step(state, action) -> (state, reward)``, which is deterministic by
contract: `learn.exact`'s enumeration depends on it, and every environment in
the tree satisfies it. The two fixtures #597's plan lists that need
randomness *inside* a transition --- the ``k``-armed bandit, whose whole
tension is a stochastic reward, and the slippery Frozen Lake --- therefore
cannot be written against this protocol without widening it for every
implementer and invalidating the enumeration oracle. They are left out and
carried as an issue rather than approximated with a deterministic
stand-in, which would be a fixture that looks like a bandit and tests nothing
about exploration.

Constructed in code and not on disk: these carry no simulated data and no
seed-dependent truth --- a chain is its transition table --- so a yaml file
would make a machine-written record a merge participant for nothing
(`sim/CLAUDE.md`'s two clauses, read for a problem with one instance per
size).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from snakes_and_ladders.learn.environment import Environment

#: How small a sweep's largest value change has to be for
#: :func:`value_iteration` to stop. 1e-13 rather than 1e-12: the suite holds
#: this against `learn.exact`'s recursion at 1e-12, so the oracle's own
#: convergence has to be tighter than the agreement asserted of it.
DEFAULT_RESIDUAL = 1e-13
DEFAULT_SWEEPS = 100_000


class SweepError(RuntimeError):
    """Raised when value iteration did not reach its residual.

    A value function read off unfinished sweeps is not an optimum, and a
    caller cannot tell it from one that is --- `likelihood/CLAUDE.md`'s
    refusal, applied to the oracle rather than to an evaluator.
    """

    def __init__(self, sweeps: int, residual: float, tolerance: float) -> None:
        super().__init__(
            f"value iteration did not settle in {sweeps} sweeps: largest change "
            f"{residual:.3e} against a tolerance of {tolerance:.3e}"
        )
        self.sweeps = sweeps
        self.residual = residual
        self.tolerance = tolerance


@dataclass(frozen=True)
class ValueFunction[S]:
    """``V*`` over an enumerated state set, and what it cost to reach.

    Parameters
    ----------
    values : dict[S, float]
        The optimal value of every state, terminal states at zero.
    sweeps : int
        Sweeps taken.
    residual : float
        Largest change on the final sweep.
    """

    values: dict[S, float]
    sweeps: int
    residual: float

    def optimal_actions[A](self, environment: Environment[S, A], state: S) -> list[A]:
        """Every action attaining the maximum at ``state``.

        A list rather than one action: a tie is the interesting case --- the
        cliff walk has several optimal first moves --- and picking one would
        hide it.
        """
        if environment.is_terminal(state):
            return []
        scored = [
            (
                reward + self.values[successor],
                action,
            )
            for action, (successor, reward) in (
                (action, environment.step(state, action))
                for action in environment.actions(state)
            )
        ]
        best = max(value for value, _ in scored)
        return [action for value, action in scored if value >= best - 1e-12]


def reachable_states[S, A](environment: Environment[S, A], start: S) -> list[S]:
    """Every state reachable from ``start``, breadth first and in that order.

    The state set :func:`value_iteration` sweeps. Derived rather than declared
    so a fixture cannot disagree with its own transition table.
    """
    seen = {start}
    order = [start]
    queue = deque([start])
    while queue:
        state = queue.popleft()
        if environment.is_terminal(state):
            continue
        for action in environment.actions(state):
            successor, _ = environment.step(state, action)
            if successor not in seen:
                seen.add(successor)
                order.append(successor)
                queue.append(successor)
    return order


def value_iteration[S, A](
    environment: Environment[S, A],
    start: S,
    *,
    residual: float = DEFAULT_RESIDUAL,
    max_sweeps: int = DEFAULT_SWEEPS,
) -> ValueFunction[S]:
    """``V*`` by sweeping the Bellman optimality operator, undiscounted.

    ``V(s) <- max_a [ r(s, a) + V(s') ]`` over every non-terminal reachable
    state until no value moves by more than ``residual``. Undiscounted because
    the rewards here are improvements and the episodes terminate, which is the
    convention
    :class:`~snakes_and_ladders.learn.environment.Episode` states; a discount
    would change the optimum rather than only the arithmetic.

    Parameters
    ----------
    environment : Environment
        Deterministic, as the protocol declares.
    start : S
        The state the reachable set is taken from.
    residual : float
        Stopping tolerance on the largest single-sweep change.
    max_sweeps : int
        Sweeps before refusing.

    Returns
    -------
    ValueFunction

    Raises
    ------
    SweepError
        If the residual is still above ``residual`` at ``max_sweeps``. A
        cycle of positive reward makes the undiscounted optimum unbounded and
        this is where that shows, loudly.
    """
    states = reachable_states(environment, start)
    values: dict[S, float] = dict.fromkeys(states, 0.0)
    last = 0.0
    for sweep in range(1, max_sweeps + 1):
        largest = 0.0
        for state in states:
            if environment.is_terminal(state):
                continue
            best = -np.inf
            for action in environment.actions(state):
                successor, reward = environment.step(state, action)
                best = max(best, reward + values[successor])
            largest = max(largest, abs(best - values[state]))
            values[state] = float(best)
        last = largest
        if largest <= residual:
            return ValueFunction(values=values, sweeps=sweep, residual=largest)
    raise SweepError(max_sweeps, last, residual)


def shortest_path_length[S, A](
    environment: Environment[S, A], start: S, *, goal: S | None = None
) -> int:
    """Decisions on the shortest route from ``start`` to a terminal state.

    Breadth first, so it counts moves and ignores reward entirely --- which is
    what makes it an independent check on Hanoi, whose optimum is a *length*
    and not a value.

    Parameters
    ----------
    environment : Environment
    start : S
    goal : S | None
        A specific state to reach. ``None`` accepts any terminal state.

    Returns
    -------
    int
        The number of moves. Zero when ``start`` already qualifies.

    Raises
    ------
    ValueError
        If no route exists, which for these fixtures means the transition
        table is wrong rather than the question unanswerable.
    """
    if goal is None and environment.is_terminal(start):
        return 0
    if goal is not None and start == goal:
        return 0
    seen = {start}
    queue = deque([(start, 0)])
    while queue:
        state, depth = queue.popleft()
        for action in environment.actions(state):
            successor, _ = environment.step(state, action)
            if successor in seen:
                continue
            if successor == goal or (
                goal is None and environment.is_terminal(successor)
            ):
                return depth + 1
            seen.add(successor)
            queue.append((successor, depth + 1))
    msg = "no route from the start to a terminal state"
    raise ValueError(msg)


@dataclass(frozen=True)
class ChainMdp(Environment[int, int]):
    """``n`` states in a line: a small reward at the near end, the prize at the far.

    Action ``+1`` walks away from the start and ``-1`` back. Stepping off the
    far end ends the episode with ``prize``; stepping off the near end ends it
    with ``consolation``. Every other step pays nothing, so a method that
    takes the first reward it can see collects ``consolation`` and a method
    that explores collects ``prize``.

    The classic NChain has a slip probability, which this protocol's
    deterministic ``step`` cannot express (module docstring); what is kept is
    the near-versus-far tension, which is the distinction the fixture is here
    for.

    Parameters
    ----------
    n_states : int
        At least two.
    prize, consolation : float
        The far and near payoffs. The default prize is worth the walk.
    """

    n_states: int = 8
    prize: float = 10.0
    consolation: float = 1.0

    def __post_init__(self) -> None:
        if self.n_states < 2:
            msg = f"a chain has at least two states, got {self.n_states}"
            raise ValueError(msg)

    def reset(self, rng: np.random.Generator) -> int:
        """The near end, always: the fixture's point is the walk, not the start."""
        del rng
        return 0

    def actions(self, state: int) -> Sequence[int]:
        return () if self.is_terminal(state) else (-1, 1)

    def step(self, state: int, action: int) -> tuple[int, float]:
        position = state + action
        if position >= self.n_states:
            return self.n_states, self.prize
        if position < 0:
            return -1, self.consolation
        return position, 0.0

    def features(self, state: int, actions: Sequence[int]) -> torch.Tensor:
        """``(len(actions), 2)``: the step's reward, and how far it goes out.

        The second is what a policy needs to prefer the far end before it has
        ever seen the prize, and it is not constant across a state's actions,
        so neither feature sits in the direction the softmax cancels.
        """
        if not actions:
            return torch.empty((0, 2), dtype=torch.float64)
        rows = []
        for action in actions:
            successor, reward = self.step(state, action)
            rows.append([reward, successor / self.n_states])
        return torch.tensor(rows, dtype=torch.float64)

    def n_features(self) -> int:
        return 2

    def is_terminal(self, state: int) -> bool:
        return state < 0 or state >= self.n_states


@dataclass(frozen=True)
class GridWorld(Environment[tuple[int, int], tuple[int, int]]):
    """A rectangle walked one cell at a time, with one goal and a step cost.

    Every move costs ``step_cost`` and reaching the goal pays ``goal_reward``,
    so the optimal return is the goal reward less the Manhattan distance ---
    a closed form the suite checks value iteration against, which is the
    cheapest possible independent oracle and the reason this is the plain
    case.

    Parameters
    ----------
    shape : tuple[int, int]
        Rows and columns, both at least one.
    goal : tuple[int, int]
        Inside the rectangle.
    step_cost : float
        Charged per move, as a negative reward.
    goal_reward : float
        Paid on entering the goal.
    """

    shape: tuple[int, int] = (4, 4)
    goal: tuple[int, int] = (3, 3)
    step_cost: float = -1.0
    goal_reward: float = 10.0

    def __post_init__(self) -> None:
        rows, columns = self.shape
        if rows < 1 or columns < 1:
            msg = f"a grid has at least one cell, got {self.shape}"
            raise ValueError(msg)
        if not (0 <= self.goal[0] < rows and 0 <= self.goal[1] < columns):
            msg = f"the goal {self.goal} lies outside {self.shape}"
            raise ValueError(msg)

    def reset(self, rng: np.random.Generator) -> tuple[int, int]:
        """A uniform non-goal cell, so credit assignment is measured from anywhere."""
        rows, columns = self.shape
        while True:
            cell = (int(rng.integers(rows)), int(rng.integers(columns)))
            if cell != self.goal:
                return cell

    def actions(self, state: tuple[int, int]) -> Sequence[tuple[int, int]]:
        if self.is_terminal(state):
            return ()
        rows, columns = self.shape
        moves = []
        for step in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            row, column = state[0] + step[0], state[1] + step[1]
            if 0 <= row < rows and 0 <= column < columns:
                moves.append(step)
        return tuple(moves)

    def step(
        self, state: tuple[int, int], action: tuple[int, int]
    ) -> tuple[tuple[int, int], float]:
        cell = (state[0] + action[0], state[1] + action[1])
        reward = self.step_cost + (self.goal_reward if cell == self.goal else 0.0)
        return cell, reward

    def features(
        self, state: tuple[int, int], actions: Sequence[tuple[int, int]]
    ) -> torch.Tensor:
        """``(len(actions), 2)``: the step's reward, and the distance it leaves.

        The distance is the instrumentation's, not the learner's problem: what
        is measured here is that a learner optimizes the features it is given,
        and a fixture whose features cannot express the optimum measures the
        features instead.
        """
        if not actions:
            return torch.empty((0, 2), dtype=torch.float64)
        rows = []
        for action in actions:
            cell, reward = self.step(state, action)
            distance = abs(cell[0] - self.goal[0]) + abs(cell[1] - self.goal[1])
            rows.append([reward, -float(distance)])
        return torch.tensor(rows, dtype=torch.float64)

    def n_features(self) -> int:
        return 2

    def is_terminal(self, state: tuple[int, int]) -> bool:
        return state == self.goal


@dataclass(frozen=True)
class CliffWalk(Environment[tuple[int, int], tuple[int, int]]):
    """A grid whose bottom row, between the start and the goal, ends the episode.

    The shortest route runs along the edge of that row; a route one row up is
    two steps longer and cannot fall in. Both are optimal under different
    policies, which is the distinction this fixture exists for and which #597's
    second pull request measures with SARSA against Q-learning --- the
    difference there comes from the *policy's* exploration, not from the
    transitions, so a deterministic cliff is the right fixture and not a
    compromise.

    Parameters
    ----------
    shape : tuple[int, int]
        Rows and columns; at least two rows and three columns, or there is no
        cliff to walk beside.
    step_cost, cliff_cost, goal_reward : float
        Per move, on falling, and on arriving.
    """

    shape: tuple[int, int] = (4, 6)
    step_cost: float = -1.0
    cliff_cost: float = -100.0
    goal_reward: float = 0.0

    def __post_init__(self) -> None:
        rows, columns = self.shape
        if rows < 2 or columns < 3:
            msg = f"a cliff walk needs at least (2, 3), got {self.shape}"
            raise ValueError(msg)

    @property
    def start(self) -> tuple[int, int]:
        """The bottom-left cell."""
        return (self.shape[0] - 1, 0)

    @property
    def goal(self) -> tuple[int, int]:
        """The bottom-right cell."""
        return (self.shape[0] - 1, self.shape[1] - 1)

    def is_cliff(self, state: tuple[int, int]) -> bool:
        """Whether ``state`` is one of the bottom row's interior cells."""
        row, column = state
        return row == self.shape[0] - 1 and 0 < column < self.shape[1] - 1

    def reset(self, rng: np.random.Generator) -> tuple[int, int]:
        """The bottom-left cell, always: the cliff is defined relative to it."""
        del rng
        return self.start

    def actions(self, state: tuple[int, int]) -> Sequence[tuple[int, int]]:
        if self.is_terminal(state):
            return ()
        rows, columns = self.shape
        moves = []
        for step in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            row, column = state[0] + step[0], state[1] + step[1]
            if 0 <= row < rows and 0 <= column < columns:
                moves.append(step)
        return tuple(moves)

    def step(
        self, state: tuple[int, int], action: tuple[int, int]
    ) -> tuple[tuple[int, int], float]:
        cell = (state[0] + action[0], state[1] + action[1])
        if self.is_cliff(cell):
            return cell, self.step_cost + self.cliff_cost
        reward = self.step_cost + (self.goal_reward if cell == self.goal else 0.0)
        return cell, reward

    def features(
        self, state: tuple[int, int], actions: Sequence[tuple[int, int]]
    ) -> torch.Tensor:
        """``(len(actions), 2)``: the step's reward, and the distance it leaves."""
        if not actions:
            return torch.empty((0, 2), dtype=torch.float64)
        rows = []
        for action in actions:
            cell, reward = self.step(state, action)
            distance = abs(cell[0] - self.goal[0]) + abs(cell[1] - self.goal[1])
            rows.append([reward, -float(distance)])
        return torch.tensor(rows, dtype=torch.float64)

    def n_features(self) -> int:
        return 2

    def is_terminal(self, state: tuple[int, int]) -> bool:
        return state == self.goal or self.is_cliff(state)


@dataclass(frozen=True)
class TowersOfHanoi(Environment[tuple[int, ...], tuple[int, int]]):
    """``d`` disks over three pegs; the state is which peg each disk sits on.

    The only fixture here whose oracle is an **integer**: the shortest
    solution is ``2^d - 1`` moves, so the assertion carries no tolerance. The
    state space is ``3^d``, which is the combinatorial shape the research
    problems have and the gridworld does not.

    A move lifts the smallest disk on one peg onto another whose smallest disk
    is larger, which is the rule; the state is indexed smallest disk first, so
    ``state[0]`` is the peg the smallest disk is on.

    Parameters
    ----------
    n_disks : int
        At least one.
    step_cost : float
        Charged per move, so the optimal return is ``-(2^d - 1)`` plus the
        goal reward and value iteration and breadth-first search must agree.
    goal_reward : float
        Paid on stacking every disk on the last peg.
    """

    n_disks: int = 4
    step_cost: float = -1.0
    goal_reward: float = 0.0

    def __post_init__(self) -> None:
        if self.n_disks < 1:
            msg = f"there is at least one disk, got {self.n_disks}"
            raise ValueError(msg)

    @property
    def start(self) -> tuple[int, ...]:
        """Every disk on peg zero."""
        return (0,) * self.n_disks

    @property
    def goal(self) -> tuple[int, ...]:
        """Every disk on peg two."""
        return (2,) * self.n_disks

    @property
    def optimal_moves(self) -> int:
        """``2^d - 1``: the known shortest solution length."""
        return int(2**self.n_disks) - 1

    def reset(self, rng: np.random.Generator) -> tuple[int, ...]:
        """The classic start, every disk on one peg."""
        del rng
        return self.start

    def _top(self, state: tuple[int, ...], peg: int) -> int | None:
        """The smallest disk on ``peg``, or ``None`` when it is empty."""
        for disk, where in enumerate(state):
            if where == peg:
                return disk
        return None

    def actions(self, state: tuple[int, ...]) -> Sequence[tuple[int, int]]:
        if self.is_terminal(state):
            return ()
        moves = []
        for source in range(3):
            lifted = self._top(state, source)
            if lifted is None:
                continue
            for target in range(3):
                if target == source:
                    continue
                resting = self._top(state, target)
                if resting is None or resting > lifted:
                    moves.append((source, target))
        return tuple(moves)

    def step(
        self, state: tuple[int, ...], action: tuple[int, int]
    ) -> tuple[tuple[int, ...], float]:
        source, target = action
        lifted = self._top(state, source)
        if lifted is None:
            msg = f"peg {source} is empty in {state}"
            raise ValueError(msg)
        moved = list(state)
        moved[lifted] = target
        successor = tuple(moved)
        reward = self.step_cost + (self.goal_reward if successor == self.goal else 0.0)
        return successor, reward

    def features(
        self, state: tuple[int, ...], actions: Sequence[tuple[int, int]]
    ) -> torch.Tensor:
        """``(len(actions), 2)``: the step's reward, and disks left off the goal peg."""
        if not actions:
            return torch.empty((0, 2), dtype=torch.float64)
        rows = []
        for action in actions:
            successor, reward = self.step(state, action)
            away = sum(1 for peg in successor if peg != 2)
            rows.append([reward, -float(away)])
        return torch.tensor(rows, dtype=torch.float64)

    def n_features(self) -> int:
        return 2

    def is_terminal(self, state: tuple[int, ...]) -> bool:
        return state == self.goal


__all__ = [
    "DEFAULT_RESIDUAL",
    "DEFAULT_SWEEPS",
    "ChainMdp",
    "CliffWalk",
    "GridWorld",
    "SweepError",
    "TowersOfHanoi",
    "ValueFunction",
    "reachable_states",
    "shortest_path_length",
    "value_iteration",
]
