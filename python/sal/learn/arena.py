"""One table per problem: five learners on one axis, each stating what it spent.

Issue #705. The five rows exist --- greedy, REINFORCE, actor-critic, PPO and
MLP+PPO --- but each was written for its own experiment, so the Potts chain has
all five, the tree has three and the HMM path none, and a new problem costs a
new harness. This module is the harness: a :class:`Learner` per row, a
:func:`table` that runs them all on one environment, and one metric read the
same way for each.

**A seam, not a second implementation.** Every entry in :data:`LEARNERS` builds
the policy a direct caller would build and calls the same function with the same
arguments in the same order, so a row is bitwise what the published number was
measured on. `tests/regression/learn/test_arena.py` asserts exactly that, and it
is the whole claim the table rests on: a harness that re-derived anything would
make its rows incomparable with the numbers already stated against these
functions.

**The metric reports a cost beside a fraction, because a match is a result.**
`ROADMAP.md`'s Milestone 8 asks a policy to beat hill climbing, and #596 is
right that nothing beats an exact baseline --- Viterbi, a graph cut at two
labels, enumeration under twelve bits. But a learner that *matches* one has
shown the policy class contains the classical method, and a learner that
matches it more cheaply has shown something stronger: the planner reaches
92.6% of the chain's starts at 8.3 action evaluations an episode against
greedy's 80.2% at 48 (#135). Neither statement is available from a fraction
alone, so :class:`Row` carries both and :class:`Outcome` names which of the
three it is.

**Greedy is a row but not a policy.** `greedy_rollout` stops at a local
maximum; a policy rolled out walks past one, taking the least-bad move, which
:class:`~sal.learn.policy.EpsilonGreedyPolicy`'s docstring
states and issue #194 measured. So the greedy row runs `greedy_rollout` rather
than a "greedy policy", because the second is a different baseline from the one
80.2% was measured on.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import torch

from sal.learn.actor_critic import actor_critic
from sal.learn.critic import Critic, n_state_features
from sal.learn.environment import Environment, Episode
from sal.learn.policy import LinearPolicy, MLPPolicy, Policy
from sal.learn.ppo import ppo
from sal.learn.reinforce import reinforce
from sal.learn.rollout import greedy_restarts, greedy_rollout, rollout

#: Hidden width of the MLP row's scorer, as issue #313 measured it. A declared
#: constant rather than a parameter of :func:`table`: the row is comparable
#: across problems only if the scorer is the same one.
MLP_HIDDEN = 8

#: Adam step size the MLP row trains at. Smaller than
#: :func:`~sal.learn.ppo.ppo`'s default, and declared rather
#: than defaulted because it is what 97.5% was measured at: the deeper scorer
#: diverges from the linear row's step, and at ``0.05`` the row reads 78 of 81
#: rather than 79 --- one start, which is the whole margin between the MLP and
#: linear rows, so the hyper-parameter is part of the published number.
MLP_LEARNING_RATE = 0.01

#: Hidden width of every row's critic. ``None`` is linear in the state
#: features, which is what the published actor-critic and PPO rows used.
CRITIC_HIDDEN: int | None = None

#: How close a value must sit to the declared optimum to count as having
#: reached it. Tight because the optimum is an *enumerated* value at these
#: sizes, not an estimate: the two are the same float or the episode did not
#: get there.
DEFAULT_TOLERANCE = 1e-9


class Scoring(StrEnum):
    """Which state in an episode decides whether it reached the optimum.

    ``VISITED`` is `learn/CLAUDE.md`'s rule --- an episode that may leave a
    local optimum is scored on its best state --- expressed without a
    direction, since an optimum may be a minimum energy or a maximum
    likelihood and "reached it" is the same claim either way.

    ``FINAL`` reads the last state alone, which is what the published chain
    numbers were measured on. The two differ exactly where a policy walks out
    of the optimum before its budget runs out, so a row states which it used
    and the reproduction of #313 uses ``FINAL``.
    """

    VISITED = "visited"
    FINAL = "final"


class Outcome(StrEnum):
    """A row against its problem's baseline: beat, match or lose.

    ``MATCH`` is a first-class outcome and not a near-miss. Against an exact
    baseline it is the only one available, and it says the learner's policy
    class contains the classical method --- shown analytically on the chain,
    where hill climbing is the weight vector proportional to ``(J, 1)``, and
    never by measurement anywhere else.
    """

    BEAT = "beat"
    MATCH = "match"
    LOSE = "lose"


@dataclass(frozen=True)
class TrainingBudget:
    """What a learned row is given, in the three numbers its trainer takes.

    Not one number: ``learn/CLAUDE.md`` counts a budget in decisions, and
    ``iterations x batch x max_steps`` is that count --- but the three factors
    are not interchangeable at a fixed product, so collapsing them would make
    two rows with the same budget incomparable. :attr:`decisions` is the
    number a row reports; the three are what it was given.

    Parameters
    ----------
    iterations : int
        Gradient updates.
    batch : int
        Episodes per update.
    max_steps : int
        Decision budget per episode, which is also the horizon a row is
        evaluated at.
    stop_at_local_optimum : bool
        Whether an episode ends where nothing improves, in training and in
        evaluation alike. ``True`` is what every published row was measured
        under. ``False`` lets a learner walk out of a local optimum, and
        changes the baseline with it: the greedy row is then hill climbing
        restarted until the same budget is spent, since one greedy run stops
        after a few decisions and a learner read against it overstates itself
        (`learn/CLAUDE.md`, issue #820).

    Raises
    ------
    ValueError
        If any of the three is under one.
    """

    iterations: int
    batch: int
    max_steps: int
    stop_at_local_optimum: bool = True

    def __post_init__(self) -> None:
        if min(self.iterations, self.batch, self.max_steps) < 1:
            msg = (
                "iterations, batch and max_steps must be >= 1, got "
                f"{self.iterations}, {self.batch}, {self.max_steps}"
            )
            raise ValueError(msg)

    @property
    def decisions(self) -> int:
        """Decisions the training run costs, at most."""
        return self.iterations * self.batch * self.max_steps


@dataclass(frozen=True)
class Streams:
    """The four generators one row consumes, each the caller's own.

    Four and not one, because the published rows were measured on four
    independent streams and a row is only bitwise what was published if it
    reads the same ones. A seed would be shorter and `sim/CLAUDE.md` refuses
    it: seeding inside a call makes an ensemble of runs one run.

    :func:`table` takes a *factory* of these rather than an instance, since a
    generator is consumed --- five rows sharing one would leave the fifth
    reading a stream the first four advanced, and each published row started
    from a freshly seeded one.

    Parameters
    ----------
    training : np.random.Generator
        Starting states and every sampled action of the training run.
    evaluation : np.random.Generator
        The evaluation rollouts alone, so a row's fraction is read on the same
        stream whatever its training consumed --- which is what makes two
        rows' fractions a comparison rather than two samples.
    policy : torch.Generator
        Initial weights of the scorer. Unread by a row whose scorer starts at
        zero, which :class:`~sal.learn.policy.LinearPolicy`
        does.
    critic : torch.Generator
        Initial weights of the critic.
    """

    training: np.random.Generator
    evaluation: np.random.Generator
    policy: torch.Generator
    critic: torch.Generator


@dataclass(frozen=True)
class Row:
    """One learner's result on one problem, with what it cost.

    Parameters
    ----------
    name : str
        The learner's, as :data:`LEARNERS` keys it.
    reached : int
        Starts from which the episode reached the declared optimum.
    starts : int
        Starts offered. Every row is evaluated from the same set, since a
        fraction over different starts is not a comparison.
    evaluations : float
        Mean scored actions per start --- over every run the start bought,
        which is one episode for every row but restarted greedy --- the
        quantity #135's planner result is stated in, and the one that
        separates "the same answer" from "the same answer more cheaply".
    training_decisions : int
        Decisions the training run spent, zero for a row that does not train.
    scoring : Scoring
        Which state decided each episode.
    """

    name: str
    reached: int
    starts: int
    evaluations: float
    training_decisions: int
    scoring: Scoring

    @property
    def fraction(self) -> float:
        """Reached over offered."""
        return self.reached / self.starts

    def against(self, baseline: Row) -> Outcome:
        """Beat, match or lose against ``baseline``.

        A strictly larger fraction beats; a strictly smaller one loses. An
        equal fraction is a match, and it is still a match when it cost fewer
        evaluations --- the cost is reported beside the outcome rather than
        folded into it, because "the same answer at an eighth of the price" is
        a sentence a reader should see rather than a rank a function invented.
        """
        if self.reached * baseline.starts > baseline.reached * self.starts:
            return Outcome.BEAT
        if self.reached * baseline.starts < baseline.reached * self.starts:
            return Outcome.LOSE
        return Outcome.MATCH


def reached_optimum[S, A](
    episode: Episode[S, A],
    value: Callable[[S], float],
    optimum: float,
    *,
    scoring: Scoring = Scoring.VISITED,
    tolerance: float = DEFAULT_TOLERANCE,
) -> bool:
    """Whether ``episode`` reached ``optimum``, under ``scoring``.

    Parameters
    ----------
    episode : Episode[S, A]
        The trajectory.
    value : Callable[[S], float]
        The objective, as the problem states it --- an energy, a
        log-likelihood. Passed rather than read off the environment because
        :class:`~sal.learn.environment.Environment` declares
        rewards and not levels, and ``Episode.total_reward`` is a difference.
    optimum : float
        The declared optimum, from enumeration or a closed form at these
        sizes.
    scoring : Scoring
        Which state decides.
    tolerance : float
        How close counts.
    """
    if scoring is Scoring.FINAL:
        return abs(value(episode.states[-1]) - optimum) < tolerance
    return any(abs(value(state) - optimum) < tolerance for state in episode.states)


def evaluations[S, A](environment: Environment[S, A], episode: Episode[S, A]) -> int:
    """Actions ``episode`` scored: the sum of the neighbourhood at each decision.

    The unit #135 reports the planner in and #313 reports greedy in --- 48 an
    episode on the 4-site chain, eight actions at each of six steps. Counted
    from the trajectory rather than instrumented into the learners, so every
    row is counted the same way whatever it does between decisions.
    """
    return sum(
        len(environment.actions(state))
        for state in episode.states[: len(episode.actions)]
    )


class Learner:
    """One row of the table: how to train, and what to roll out afterwards.

    Parameters
    ----------
    name : str
        The row's label.
    train : Callable | None
        Given ``(environment, budget, rng, generator)``, returns the policy to
        evaluate. ``None`` for a row that does not train, whose episodes come
        from :func:`~sal.learn.rollout.greedy_rollout` instead.

    Under the seam rule: two consuming modules rather than three --- this one
    and its test. It is kept because the alternative is the state #705 found:
    five callers of five signatures, none of which can be listed, so no table
    can be built from them without a sixth. The registry is the list.
    """

    def __init__(
        self,
        name: str,
        train: Callable[[Environment[object, object], TrainingBudget, Streams], Policy]
        | None,
    ) -> None:
        self.name = name
        self._train = train

    @property
    def trains(self) -> bool:
        """Whether this row has a training run to spend a budget on."""
        return self._train is not None

    def episodes[S, A](
        self,
        environment: Environment[S, A],
        starts: Sequence[S],
        budget: TrainingBudget,
        streams: Streams,
    ) -> list[tuple[Episode[S, A], ...]]:
        """Train if this row trains, then the runs each of ``starts`` buys.

        One run per start for every row but one: under
        ``budget.stop_at_local_optimum = False`` the greedy row is
        :func:`~sal.learn.rollout.greedy_restarts`, whose
        restarts are the runs, so a start's result is the best state over
        them and its cost their sum (issue #820). Under the default the
        greedy row reads no stream at all: `greedy_rollout` takes no
        generator, since given a start it is deterministic and ties break
        towards the first action the environment lists.
        """
        if self._train is None:
            if budget.stop_at_local_optimum:
                return [
                    (greedy_rollout(environment, start, budget.max_steps),)
                    for start in starts
                ]
            return [
                greedy_restarts(
                    environment, start, budget.max_steps, streams.evaluation
                )
                for start in starts
            ]
        policy = self._train(
            environment,  # type: ignore[arg-type]
            budget,
            streams,
        )
        return [
            (
                rollout(
                    environment,
                    policy,
                    streams.evaluation,
                    budget.max_steps,
                    start=start,
                    stop_at_local_optimum=budget.stop_at_local_optimum,
                ),
            )
            for start in starts
        ]


def _critic[S, A](environment: Environment[S, A], streams: Streams) -> Critic:
    """The critic every value-using row builds, at the published width."""
    return Critic(
        n_state_features(environment), hidden=CRITIC_HIDDEN, generator=streams.critic
    )


def _reinforce[S, A](
    environment: Environment[S, A],
    budget: TrainingBudget,
    streams: Streams,
) -> Policy:
    policy = LinearPolicy(environment.n_features())
    reinforce(
        environment,
        policy,
        streams.training,
        budget.iterations,
        budget.batch,
        budget.max_steps,
        stop_at_local_optimum=budget.stop_at_local_optimum,
    )
    return policy


def _actor_critic[S, A](
    environment: Environment[S, A],
    budget: TrainingBudget,
    streams: Streams,
) -> Policy:
    policy = LinearPolicy(environment.n_features())
    actor_critic(
        environment,
        policy,
        _critic(environment, streams),
        streams.training,
        iterations=budget.iterations,
        batch=budget.batch,
        max_steps=budget.max_steps,
        stop_at_local_optimum=budget.stop_at_local_optimum,
    )
    return policy


def _ppo[S, A](
    environment: Environment[S, A],
    budget: TrainingBudget,
    streams: Streams,
) -> Policy:
    policy = LinearPolicy(environment.n_features())
    ppo(
        environment,
        policy,
        _critic(environment, streams),
        streams.training,
        iterations=budget.iterations,
        batch=budget.batch,
        max_steps=budget.max_steps,
        stop_at_local_optimum=budget.stop_at_local_optimum,
    )
    return policy


def _mlp_ppo[S, A](
    environment: Environment[S, A],
    budget: TrainingBudget,
    streams: Streams,
) -> Policy:
    policy = MLPPolicy(
        environment.n_features(), hidden=MLP_HIDDEN, generator=streams.policy
    )
    ppo(
        environment,
        policy,
        _critic(environment, streams),
        streams.training,
        iterations=budget.iterations,
        batch=budget.batch,
        max_steps=budget.max_steps,
        learning_rate=MLP_LEARNING_RATE,
        stop_at_local_optimum=budget.stop_at_local_optimum,
    )
    return policy


#: The five rows, in the order the table reports them: the baseline first, then
#: the learners in the order they were added to the tree. The order is declared
#: rather than incidental, because a table read across problems is read down its
#: rows.
LEARNERS: dict[str, Learner] = {
    "greedy": Learner("greedy", None),
    "reinforce": Learner("reinforce", _reinforce),
    "actor-critic": Learner("actor-critic", _actor_critic),
    "ppo": Learner("ppo", _ppo),
    "mlp-ppo": Learner("mlp-ppo", _mlp_ppo),
}


def row[S, A](
    learner: Learner,
    environment: Environment[S, A],
    starts: Sequence[S],
    value: Callable[[S], float],
    optimum: float,
    budget: TrainingBudget,
    streams: Streams,
    *,
    scoring: Scoring = Scoring.VISITED,
    tolerance: float = DEFAULT_TOLERANCE,
) -> Row:
    """One learner's row: its success fraction and what it cost."""
    runs = learner.episodes(environment, starts, budget, streams)
    return Row(
        name=learner.name,
        reached=sum(
            any(
                reached_optimum(
                    episode, value, optimum, scoring=scoring, tolerance=tolerance
                )
                for episode in group
            )
            for group in runs
        ),
        starts=len(starts),
        evaluations=float(
            np.mean(
                [
                    sum(evaluations(environment, episode) for episode in group)
                    for group in runs
                ]
            )
        ),
        training_decisions=budget.decisions if learner.trains else 0,
        scoring=scoring,
    )


def table[S, A](
    environment: Environment[S, A],
    starts: Sequence[S],
    value: Callable[[S], float],
    optimum: float,
    budget: TrainingBudget,
    streams: Callable[[], Streams],
    *,
    scoring: Scoring = Scoring.VISITED,
    names: Sequence[str] | None = None,
) -> tuple[Row, ...]:
    """Every registered row on one problem, in :data:`LEARNERS` order.

    Parameters
    ----------
    environment, starts, value, optimum, budget, scoring
        As :func:`row`.
    streams : Callable[[], Streams]
        Called once per row, so each starts from a freshly seeded set --- which
        is what the published rows did and what makes them comparable.
    names : Sequence[str] | None
        Rows to run, defaulting to all of them. Named rather than sliced, so a
        partial table says which rows it is.

    Raises
    ------
    KeyError
        If a name is not registered.
    """
    chosen = list(LEARNERS) if names is None else list(names)
    unknown = [name for name in chosen if name not in LEARNERS]
    if unknown:
        msg = f"unregistered learners {unknown}; the table carries {list(LEARNERS)}"
        raise KeyError(msg)
    return tuple(
        row(
            LEARNERS[name],
            environment,
            starts,
            value,
            optimum,
            budget,
            streams(),
            scoring=scoring,
        )
        for name in chosen
    )


__all__ = [
    "CRITIC_HIDDEN",
    "DEFAULT_TOLERANCE",
    "LEARNERS",
    "MLP_HIDDEN",
    "MLP_LEARNING_RATE",
    "Learner",
    "Outcome",
    "Row",
    "Scoring",
    "Streams",
    "TrainingBudget",
    "evaluations",
    "reached_optimum",
    "row",
    "table",
]
