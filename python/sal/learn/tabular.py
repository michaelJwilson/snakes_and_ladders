"""Tabular Q-learning and SARSA, which differ in one expression.

Issue #597. These two are here for one reason: the cliff walk's claim ---
*an on-policy learner and an off-policy learner disagree, predictably* --- is
untestable with only one of them, and it is the sharpest checkable claim in
elementary reinforcement learning. Both take the same experience and differ in
the continuation term of their update, one line marked in :func:`_control`:

* **Q-learning** bootstraps from ``max_a' Q(s', a')``, the value of the
  *greedy* successor action, whatever the behaviour policy then does. Its fixed
  point is ``q*``, so its greedy policy is optimal.
* **SARSA** bootstraps from ``Q(s', a')`` for the action the behaviour policy
  actually took next. Its fixed point is ``q_pi`` for that policy, ``pi``
  being epsilon-greedy, so it values the exploration it will actually do.

That difference is the whole content of the module, and it is why
``max |V - V*|`` is asserted small for one of them and asserted *non-zero* for
the other: SARSA's values are not ``V*`` and a test claiming they were would
claim something false --- `likelihood/CLAUDE.md`'s rule that an approximate
evaluator states which regime carries its correctness, applied to a learner.

Tabular by construction: the table is keyed on the state, so the method holds
only where the reachable set is enumerable. That is the boundary #596 draws
between these and the function approximators --- a research problem's state
space is not enumerable, which is exactly why the learners measured there are
`LinearPolicy` and `MLPPolicy` and why these two stay on the canonical
fixtures.

Exploration is epsilon-greedy over the table, plus the initial value. Both
matter and the chain shows why: with ``initial_value = 0`` the near reward is
a trap that neither learner escapes, since reaching the far prize needs a
run of exploratory steps whose probability is ``(epsilon / 2)**(n - 1)``; with
an optimistic initial value the same code walks out and finds the prize. The
fixture exists for that distinction.

See ``eq:q-learning`` and ``eq:sarsa`` of ``docs/tex/textbook.tex``, whose
``app:rl:td`` states both updates and why the difference is not cosmetic for a
search that must take a worsening move (Sutton & Barto ch. 6).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from sal.learn.canonical import ValueFunction
from sal.learn.environment import Environment, Episode

#: Constant step size. Constant rather than decayed: a schedule is a second
#: thing to tune and the claims here do not need one -- at this rate
#: Q-learning's greedy policy is optimal from every start of the 4x4 grid and
#: its values sit 2.14e-05 from ``V*``, which a decayed step size would
#: tighten and no claim here rests on
#: (`tests/regression/learn/test_tabular.py`).
DEFAULT_LEARNING_RATE = 0.1
#: Exploration rate of the behaviour policy. Held fixed rather than annealed to
#: zero: annealing it away makes SARSA converge to ``q*`` and erases the
#: distinction both learners are here to expose.
DEFAULT_EPSILON = 0.1
DEFAULT_EPISODES = 2_000
#: Steps before an episode is truncated. A learner that has not yet found a
#: terminal state wanders, and an unbounded episode would never end.
DEFAULT_MAX_STEPS = 200
#: Pessimistic by default, so a caller who wants optimism asks for it and the
#: chain's trap is the default rather than a surprise.
DEFAULT_INITIAL_VALUE = 0.0
#: Two action values within this are a tie. The fixtures' rewards are integers,
#: so a tie here is a real tie and not a rounding artefact.
TIE = 1e-12


@dataclass(frozen=True)
class ActionValues[S, A]:
    """A learned ``Q`` table, with what was spent reaching it.

    Parameters
    ----------
    values : dict[S, dict[A, float]]
        ``Q(s, a)``, one inner mapping per state the run touched. A terminal
        state maps to an empty mapping: it has no actions, and giving it a
        zero-valued one would invent an action the environment does not offer.
    visits : dict[S, int]
        How many updates each state received. Reported beside every deviation,
        because a value read off a state visited twice is not an estimate of
        anything and a caller cannot tell it from one visited ten thousand
        times.
    episodes : int
        Episodes run.
    updates : int
        Update applications, which is the work done and is not
        ``episodes`` times anything: episodes end at different lengths.
    """

    values: dict[S, dict[A, float]]
    visits: dict[S, int]
    episodes: int
    updates: int

    def value(self, state: S) -> float:
        """``max_a Q(state, a)``, and ``0.0`` at a terminal or unseen state.

        Zero rather than ``-inf`` at a terminal state: that is the value the
        Bellman recursion assigns it, and it is what
        :func:`~sal.learn.canonical.value_iteration` holds
        there too, so the two are comparable without a special case at the
        call site.
        """
        row = self.values.get(state)
        return max(row.values()) if row else 0.0

    def greedy_actions(self, state: S) -> list[A]:
        """Every action attaining the maximum at ``state``.

        A list, for the same reason
        :meth:`~sal.learn.canonical.ValueFunction.optimal_actions`
        returns one: a tie is the interesting case. On the chain the exact
        optimum ties at every interior state, so a learner that ties there has
        agreed with its oracle rather than failed to decide.
        """
        row = self.values.get(state)
        if not row:
            return []
        best = max(row.values())
        return [action for action, value in row.items() if value >= best - TIE]

    def greedy_episode(
        self, environment: Environment[S, A], start: S, *, max_steps: int
    ) -> Episode[S, A]:
        """Run the greedy policy from ``start``, breaking ties by insertion order.

        Deterministic, and takes no generator: a seeded tie-break would make
        the learned policy's route depend on a second seed nobody declared.
        Insertion order is the order ``environment.actions`` first listed them.

        The returned episode carries ``terminated``, and a caller reading
        ``total_reward`` has to look at it: an undiscounted return from an
        episode that ran out of budget is a truncation, not the policy's value.
        A greedy policy can cycle forever without being wrong --- see
        :meth:`greedy_actions` on the chain --- so this reports rather than
        refuses.
        """
        state = start
        states: list[S] = [state]
        actions: list[A] = []
        rewards: list[float] = []
        while not environment.is_terminal(state) and len(actions) < max_steps:
            greedy = self.greedy_actions(state)
            if not greedy:
                break
            action = greedy[0]
            state, reward = environment.step(state, action)
            actions.append(action)
            rewards.append(reward)
            states.append(state)
        return Episode(
            states=tuple(states),
            actions=tuple(actions),
            rewards=tuple(rewards),
            terminated=environment.is_terminal(state),
        )

    def behaviour_episode(
        self,
        environment: Environment[S, A],
        start: S,
        rng: np.random.Generator,
        *,
        epsilon: float = DEFAULT_EPSILON,
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> Episode[S, A]:
        """Run the epsilon-greedy policy the table was learned under.

        The greedy episode is what a learner *proposes*; this is what it would
        actually collect while still exploring, and the two are different
        numbers. On the cliff walk they are the difference the fixture exists
        to show: the optimal route runs beside the cliff, so an exploratory
        step off it is expensive, and the route SARSA prefers gives up two
        steps to avoid that.

        Parameters
        ----------
        environment : Environment[S, A]
        start : S
        rng : np.random.Generator
            The only source of randomness.
        epsilon : float
            Exploration rate. Pass the rate the table was learned at, or the
            comparison is against a policy that never ran.
        max_steps : int
            Truncation budget, as for :meth:`greedy_episode`.
        """
        state = start
        states: list[S] = [state]
        actions: list[A] = []
        rewards: list[float] = []
        while not environment.is_terminal(state) and len(actions) < max_steps:
            available = environment.actions(state)
            if not available:
                break
            action = _choose(self.values.get(state), available, rng, epsilon)
            state, reward = environment.step(state, action)
            actions.append(action)
            rewards.append(reward)
            states.append(state)
        return Episode(
            states=tuple(states),
            actions=tuple(actions),
            rewards=tuple(rewards),
            terminated=environment.is_terminal(state),
        )

    def largest_deviation(
        self, optimal: ValueFunction[S], *, at_least: int = 1
    ) -> float:
        """``max_s |max_a Q(s, a) - V*(s)|`` over states visited enough times.

        Parameters
        ----------
        optimal : ValueFunction[S]
            From :func:`~sal.learn.canonical.value_iteration`.
        at_least : int
            Skip states updated fewer times than this. The default counts
            every state the run touched; raising it separates "this learner's
            fixed point is not ``V*``" from "this state was seen twice".

        Returns
        -------
        float
            ``0.0`` when no state qualifies, which a caller distinguishes by
            the visit counts it passed.
        """
        deviations = [
            abs(self.value(state) - optimal.values[state])
            for state, count in self.visits.items()
            if count >= at_least
            and state in optimal.values
            and self.values[state]  # a terminal state has no actions to read
        ]
        return max(deviations, default=0.0)


def _choose[A](
    row: dict[A, float] | None,
    available: Sequence[A],
    rng: np.random.Generator,
    epsilon: float,
) -> A:
    """An epsilon-greedy action over ``row``, falling back to uniform.

    ``row`` is ``None`` for a state this run has not reached, where every
    action is equally unknown and uniform is the only honest choice. Ties
    among the greedy actions are broken uniformly, so a table that has learned
    nothing does not silently prefer whichever action the environment lists
    first.
    """
    if row is None or not row or rng.random() < epsilon:
        return available[int(rng.integers(len(available)))]
    best = max(row.values())
    tied = [action for action, value in row.items() if value >= best - TIE]
    return tied[int(rng.integers(len(tied)))]


def _control[S, A](
    environment: Environment[S, A],
    *,
    on_policy: bool,
    start: S | None,
    episodes: int,
    learning_rate: float,
    epsilon: float,
    max_steps: int,
    initial_value: float,
    rng: np.random.Generator,
) -> ActionValues[S, A]:
    """The shared loop. ``on_policy`` picks the continuation term, and nothing else."""
    if episodes < 1:
        msg = f"episodes must be >= 1, got {episodes}"
        raise ValueError(msg)
    if not 0.0 < learning_rate <= 1.0:
        msg = (
            f"learning_rate must be in (0, 1], got {learning_rate}: at 0 no "
            "update moves and the table stays at its initial value"
        )
        raise ValueError(msg)
    if not 0.0 <= epsilon <= 1.0:
        msg = f"epsilon must be in [0, 1], got {epsilon}"
        raise ValueError(msg)
    if max_steps < 1:
        msg = f"max_steps must be >= 1, got {max_steps}"
        raise ValueError(msg)

    values: dict[S, dict[A, float]] = {}
    visits: dict[S, int] = {}

    def row(state: S) -> dict[A, float]:
        if state not in values:
            available = tuple(environment.actions(state))
            values[state] = dict.fromkeys(available, initial_value)
            visits[state] = 0
        return values[state]

    updates = 0
    for _ in range(episodes):
        state = environment.reset(rng) if start is None else start
        if environment.is_terminal(state):
            continue
        action = _choose(row(state), environment.actions(state), rng, epsilon)
        for _ in range(max_steps):
            successor, reward = environment.step(state, action)
            successor_row = row(successor)
            terminal = environment.is_terminal(successor)
            next_action = (
                None
                if terminal or not successor_row
                else _choose(
                    successor_row,
                    environment.actions(successor),
                    rng,
                    epsilon,
                )
            )
            # The one line the two methods disagree on. SARSA bootstraps from
            # the action its behaviour policy will take; Q-learning from the
            # best available whether or not it will be taken. Everything else
            # in this function is shared, which is the point.
            if terminal or next_action is None:
                continuation = 0.0
            elif on_policy:
                continuation = successor_row[next_action]
            else:
                continuation = max(successor_row.values())

            current = row(state)
            current[action] += learning_rate * (reward + continuation - current[action])
            visits[state] += 1
            updates += 1
            if terminal or next_action is None:
                break
            state, action = successor, next_action

    return ActionValues(
        values=values, visits=visits, episodes=episodes, updates=updates
    )


def q_learning[S, A](
    environment: Environment[S, A],
    rng: np.random.Generator,
    *,
    start: S | None = None,
    episodes: int = DEFAULT_EPISODES,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    epsilon: float = DEFAULT_EPSILON,
    max_steps: int = DEFAULT_MAX_STEPS,
    initial_value: float = DEFAULT_INITIAL_VALUE,
) -> ActionValues[S, A]:
    """Off-policy control: bootstrap from the greedy successor action.

    ``Q(s, a) <- Q(s, a) + alpha [ r + max_a' Q(s', a') - Q(s, a) ]``, with
    ``a'`` the best action at ``s'`` whatever the behaviour policy does there.
    The fixed point is ``q*``, so the greedy policy this leaves is optimal and
    its values are comparable to
    :func:`~sal.learn.canonical.value_iteration`'s directly.

    Undiscounted, matching the rest of :mod:`sal.learn`: the
    rewards are improvements and the episodes terminate, so ``gamma = 1``
    needs no separate justification
    (:class:`~sal.learn.environment.Episode`).

    Parameters
    ----------
    environment : Environment[S, A]
        Deterministic, as the protocol declares, and with an enumerable
        reachable set: the table is keyed on the state.
    rng : np.random.Generator
        The only source of randomness -- the starts, the exploration, and
        every tie-break.
    start : S | None
        Fixed starting state, or ``None`` to draw one per episode from
        ``environment.reset``. ``None`` is what visits the whole state space
        on a fixture whose ``reset`` is uniform, and it is what makes a
        deviation from ``V*`` a statement about the method rather than about
        which states were reached.
    episodes : int
        Episodes to run.
    learning_rate : float
        Constant step size ``alpha``, in ``(0, 1]``.
    epsilon : float
        Exploration rate of the behaviour policy, in ``[0, 1]``.
    max_steps : int
        Steps before an episode is truncated.
    initial_value : float
        What an unseen action is worth before it is tried. Above the true
        optimum this is optimistic initialisation, which explores
        systematically rather than by luck; at zero a positive near reward is
        absorbing on any fixture that has one.

    Returns
    -------
    ActionValues[S, A]

    Raises
    ------
    ValueError
        If ``episodes``, ``learning_rate``, ``epsilon`` or ``max_steps`` is
        outside its stated range.
    """
    return _control(
        environment,
        on_policy=False,
        start=start,
        episodes=episodes,
        learning_rate=learning_rate,
        epsilon=epsilon,
        max_steps=max_steps,
        initial_value=initial_value,
        rng=rng,
    )


def sarsa[S, A](
    environment: Environment[S, A],
    rng: np.random.Generator,
    *,
    start: S | None = None,
    episodes: int = DEFAULT_EPISODES,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    epsilon: float = DEFAULT_EPSILON,
    max_steps: int = DEFAULT_MAX_STEPS,
    initial_value: float = DEFAULT_INITIAL_VALUE,
) -> ActionValues[S, A]:
    """On-policy control: bootstrap from the action the behaviour policy takes.

    ``Q(s, a) <- Q(s, a) + alpha [ r + Q(s', a') - Q(s, a) ]``, with ``a'``
    the action epsilon-greedy selection actually drew at ``s'``. The name is
    the quintuple it uses.

    The fixed point is ``q_pi`` for that epsilon-greedy policy and **not**
    ``q*``, which is not a defect but the method: it values the exploratory
    steps it will really take. Where those steps are expensive its greedy
    policy differs from Q-learning's, and the cliff walk is the fixture that
    makes the difference a theorem rather than an observation.

    Parameters are :func:`q_learning`'s, and both raise on the same ranges.

    Returns
    -------
    ActionValues[S, A]
    """
    return _control(
        environment,
        on_policy=True,
        start=start,
        episodes=episodes,
        learning_rate=learning_rate,
        epsilon=epsilon,
        max_steps=max_steps,
        initial_value=initial_value,
        rng=rng,
    )
