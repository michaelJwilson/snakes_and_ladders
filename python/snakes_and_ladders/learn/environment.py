"""The reinforcement-learning interface, and nothing that knows what it searches.

This module is to ``snakes_and_ladders.learn`` what ``objective.py`` is to ``snakes_and_ladders.opt``,
and deliberately so. ``opt/CLAUDE.md`` records why the optimizer may not know
what it optimizes; the same argument applies here with one more step. An
interface justified by a single application is shaped by that application, so
this one is written against no application at all: ``snakes_and_ladders.learn`` imports
nothing from ``snakes_and_ladders.sim``, ``snakes_and_ladders.likelihood`` or ``snakes_and_ladders.search``, asserted
by a test rather than left to review.

Four pieces are enough, and they are the ones ``docs/tex`` already names:

* a **state**, whose type the environment owns;
* an **action set that varies with the state**, because a move neighbourhood's
  size varies with the problem -- so the policy *scores* candidate actions
  rather than indexing a fixed action space;
* a **step** returning the next state and a scalar reward;
* **features** of each available action, which is what a policy consumes.

The reward is a difference of an objective, so the undiscounted return
telescopes to the total improvement an episode achieved and ``gamma = 1``
needs no separate justification.

**No inner optimization.** A reward here is a closed-form function of the
state under *known* parameters, never a quantity solved for by ``snakes_and_ladders.opt``.
That is issue #131's simplification, and it is what makes an RL loop
affordable at all: a fitted reward costs one L-BFGS solve per action, and a
single episode evaluates the whole neighbourhood at every step.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class Environment[S, A](Protocol):
    """A discrete search problem an agent proposes moves in.

    Implementations are free to be stateful in the sense of holding data and
    fixed structure; they must not be stateful in the episode, since
    :func:`rollout` and the exact enumeration in :mod:`snakes_and_ladders.learn.exact`
    both revisit states and would otherwise disagree.

    Every member is a method rather than a property: a ``runtime_checkable``
    protocol can only ``isinstance``-check methods, and the check is used.
    """

    def reset(self, rng: np.random.Generator) -> S:
        """Draw a starting state.

        Parameters
        ----------
        rng : np.random.Generator
            The only source of randomness, so an episode is reproducible
            from its seed.

        Returns
        -------
        S
            A starting state.
        """
        ...  # pragma: no cover

    def actions(self, state: S) -> Sequence[A]:
        """The moves available in ``state``.

        Its length may vary with the state, which is why the policy scores
        actions instead of indexing them.
        """
        ...  # pragma: no cover

    def step(self, state: S, action: A) -> tuple[S, float]:
        """Apply ``action`` and return the next state and its reward.

        The reward is an *improvement* -- a difference of the objective
        being searched -- so that the undiscounted return telescopes.
        """
        ...  # pragma: no cover

    def features(self, state: S, actions: Sequence[A]) -> torch.Tensor:
        """Features of each available action, shape ``(len(actions), n)``.

        Batched over the neighbourhood because a policy scores the whole
        neighbourhood at once; ``n`` is :meth:`n_features`.

        A feature that takes the same value for every action in a state is
        **unidentifiable**: the policy is a softmax over these scores, and a
        constant shared by every action cancels. That is the same gauge
        ``snakes_and_ladders.opt.constrain.log_simplex`` fixes, and the reason no
        implementation here supplies a bias term.
        """
        ...  # pragma: no cover

    def n_features(self) -> int:
        """Width of the feature vector :meth:`features` returns."""
        ...  # pragma: no cover

    def is_terminal(self, state: S) -> bool:
        """Whether the episode ends on reaching ``state``.

        A property of the state, not of the action taken to reach it.
        """
        ...  # pragma: no cover


@dataclass(frozen=True)
class Episode[S, A]:
    """One trajectory, recorded in full.

    Parameters
    ----------
    states : tuple[S, ...]
        Visited states, starting with the initial one. Length is one more
        than ``actions``.
    actions : tuple[A, ...]
        Actions taken, in order.
    rewards : tuple[float, ...]
        Reward of each action, aligned with ``actions``.
    terminated : bool
        Whether the episode ended because the environment said the state was
        terminal, rather than because the step budget ran out. A truncated
        episode has not finished, and ``docs/tex`` notes why that matters:
        a truncation landing between a sacrifice and its payoff teaches the
        opposite of the truth.
    """

    states: tuple[S, ...]
    actions: tuple[A, ...]
    rewards: tuple[float, ...]
    terminated: bool

    @property
    def total_reward(self) -> float:
        """The undiscounted return.

        By the telescoping in ``docs/tex`` this is exactly the improvement in
        the underlying objective between the first and last state, whatever
        path was taken between them.
        """
        return float(sum(self.rewards))

    def returns_to_go(self) -> tuple[float, ...]:
        """``G_t = sum_{u >= t} R_u`` for each step, undiscounted.

        Undiscounted because ``gamma < 1`` breaks the telescoping and would
        prefer improvement found early over the same improvement found late.
        Episodes are capped by a step budget, so the sum is finite without
        discounting.
        """
        suffix, running = [], 0.0
        for reward in reversed(self.rewards):
            running += reward
            suffix.append(running)
        return tuple(reversed(suffix))
