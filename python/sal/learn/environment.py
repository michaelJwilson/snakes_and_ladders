"""The reinforcement-learning interface, and nothing that knows what it searches.

This module is to ``sal.learn`` what ``objective.py`` is to ``sal.opt``,
and deliberately so. ``opt/CLAUDE.md`` records why the optimizer may not know
what it optimizes; the same argument applies here with one more step. An
interface justified by a single application is shaped by that application, so
this one is written against no application at all: ``sal.learn`` imports
nothing from ``sal.sim``, ``sal.likelihood`` or ``sal.search``, asserted
by a test rather than left to review.

Four pieces are enough, and they are the ones ``sec:policy-gradient`` of ``docs/tex/textbook.tex`` names:

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
state under *known* parameters, never a quantity solved for by ``sal.opt``.
That is issue #131's simplification, and it is what makes an RL loop
affordable at all: a fitted reward costs one L-BFGS solve per action, and a
single episode evaluates the whole neighbourhood at every step.
"""

from __future__ import annotations

import warnings
from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import torch


@runtime_checkable
class Environment[S, A](Protocol):
    """A discrete search problem an agent proposes moves in.

    Implementations are free to be stateful in the sense of holding data and
    fixed structure; they must not be stateful in the episode, since
    :func:`rollout` and the exact enumeration in :mod:`sal.learn.exact`
    both revisit states and would otherwise disagree.

    Every member is a method rather than a property: a ``runtime_checkable``
    protocol can only ``isinstance``-check methods, and the check is used.
    """

    @abstractmethod
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

    @abstractmethod
    def actions(self, state: S) -> Sequence[A]:
        """The moves available in ``state``.

        Its length may vary with the state, which is why the policy scores
        actions instead of indexing them.
        """
        ...  # pragma: no cover

    @abstractmethod
    def step(self, state: S, action: A) -> tuple[S, float]:
        """Apply ``action`` and return the next state and its reward.

        The reward is an *improvement* -- a difference of the objective
        being searched -- so that the undiscounted return telescopes.
        """
        ...  # pragma: no cover

    @abstractmethod
    def features(self, state: S, actions: Sequence[A]) -> NDArray[np.float64]:
        """Features of each available action, shape ``(len(actions), n)``.

        Batched over the neighbourhood because a policy scores the whole
        neighbourhood at once; ``n`` is :meth:`n_features`.

        An array, not a tensor: no gradient flows into a feature, which is a
        constant to every loss the learners differentiate, so the policy
        converts it once where it scores (issue #1011). A tensor an
        implementation still returns is accepted there too.

        A feature that takes the same value for every action in a state is
        **unidentifiable**: the policy is a softmax over these scores, and a
        constant shared by every action cancels. That is the same gauge
        ``sal.opt.constrain.log_simplex`` fixes, and the reason no
        implementation here supplies a bias term.
        """
        ...  # pragma: no cover

    @abstractmethod
    def n_features(self) -> int:
        """Width of the feature vector :meth:`features` returns."""
        ...  # pragma: no cover

    @abstractmethod
    def is_terminal(self, state: S) -> bool:
        """Whether the episode ends on reaching ``state``.

        A property of the state, not of the action taken to reach it.
        """
        ...  # pragma: no cover


def features_tensor[S, A](
    environment: Environment[S, A], state: S, actions: Sequence[A], /
) -> torch.Tensor:
    """Deprecated: :meth:`Environment.features` as the tensor release 0.3.0 returned.

    Kept for one release after 0.3.0 and removed in the one after that
    (issue #1011). ``torch.as_tensor`` shares the array's memory, so the values
    are the array's bit for bit.

    Parameters
    ----------
    environment : Environment[S, A]
        The environment whose features to read.
    state : S
        The state the actions are available in.
    actions : Sequence[A]
        The actions to describe.

    Returns
    -------
    torch.Tensor
        Shape ``(len(actions), environment.n_features())``, ``float64``.
    """
    warnings.warn(
        "features_tensor is deprecated: Environment.features returns a NumPy "
        "array; call torch.as_tensor on it where a tensor is needed",
        DeprecationWarning,
        stacklevel=2,
    )
    import torch

    return torch.as_tensor(environment.features(state, actions))


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
        episode has not finished, and ``sec:policy-gradient`` of ``docs/tex/textbook.tex``
        notes why that matters:
        a truncation landing between a sacrifice and its payoff teaches the
        opposite of the truth.
    """

    states: tuple[S, ...]
    actions: tuple[A, ...]
    rewards: tuple[float, ...]
    terminated: bool

    @classmethod
    def from_rollout(
        cls,
        states: Sequence[S],
        actions: Sequence[A],
        rewards: Sequence[float],
        environment: Environment[S, A],
        /,
    ) -> Episode[S, A]:
        """The episode a loop just walked, ``terminated`` read from the environment.

        Four loops built this record field by field and each decided
        ``terminated`` for itself; it is a property of the last state, not of
        the loop, so it is read here from the state the walk ended in.

        Parameters
        ----------
        states : Sequence[S]
            Visited states, starting with the initial one and ending with the
            state the walk stopped in.
        actions : Sequence[A]
            Actions taken, in order; one fewer than ``states``.
        rewards : Sequence[float]
            Reward of each action, aligned with ``actions``.
        environment : Environment[S, A]
            The environment walked, asked whether the final state is terminal.

        Returns
        -------
        Episode[S, A]
        """
        return cls(
            states=tuple(states),
            actions=tuple(actions),
            rewards=tuple(rewards),
            terminated=environment.is_terminal(states[-1]),
        )

    @property
    def total_reward(self) -> float:
        """The undiscounted return.

        By ``eq:return`` of ``docs/tex/textbook.tex`` this is exactly the improvement in
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
