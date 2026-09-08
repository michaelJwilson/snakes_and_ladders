"""The policy: a softmax over scored actions, and the gauge that entails.

``sec:policy-gradient`` of ``docs/tex/textbook.tex`` states the form directly --- ``pi(a | s)`` proportional to
``exp f(s, a)`` over the *available* actions, because a move neighbourhood's
size varies with the problem and a fixed action index would not survive it.

The scorer here is linear in the action features. That is not a placeholder
for something deeper: a linear scorer keeps the learned parameters
interpretable, so recovery can be stated against a known truth the way
``opt/CLAUDE.md`` requires of a fit, rather than against "the loss went
down". A deeper scorer is a drop-in replacement for :class:`LinearPolicy`
and changes nothing else.

**The gauge.** A softmax over scores is invariant to adding a constant to
every score in a state, exactly as a softmax over logits is invariant to
adding a constant to all of them. So a feature that is constant across the
available actions has no effect and no identifiable weight -- there is no
bias term here, and :class:`Environment` implementations are told not to
supply one. This is the same failure ``snakes_and_ladders.opt.constrain.log_simplex``
exists to prevent, one module over.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch


class Policy(Protocol):
    """What :func:`snakes_and_ladders.learn.rollout.rollout` needs of a policy.

    Narrower than :class:`LinearPolicy` on purpose. Rolling out an episode
    requires only the ability to choose among scored actions;
    :func:`snakes_and_ladders.learn.reinforce.reinforce` needs the weights and the
    autograd graph besides, and keeps the concrete type. Separating the two
    is what lets a non-differentiable wrapper such as
    :class:`EpsilonGreedyPolicy` be rolled out without being trainable, which
    is exactly what it is.
    """

    def sample(self, features: torch.Tensor, rng: np.random.Generator) -> int:
        """Choose an action index from the scored actions."""
        ...


class LinearPolicy:
    """Scores each action linearly in its features, then softmaxes.

    Parameters
    ----------
    n_features : int
        Width of the feature vector the environment emits.
    dtype : torch.dtype
        Working precision. ``float64`` by default: the exact-enumeration
        oracle in :mod:`snakes_and_ladders.learn.exact` compares against finite
        differences, which ``float32`` cannot support.

    Raises
    ------
    ValueError
        If ``n_features`` is not positive.
    """

    def __init__(self, n_features: int, dtype: torch.dtype = torch.float64) -> None:
        if n_features < 1:
            msg = f"n_features must be >= 1, got {n_features}"
            raise ValueError(msg)
        self._weights = torch.zeros(n_features, dtype=dtype, requires_grad=True)
        self._dtype = dtype

    @property
    def weights(self) -> torch.Tensor:
        """The learned parameter vector, with ``requires_grad`` set."""
        return self._weights

    @property
    def n_features(self) -> int:
        """Width of the feature vector this policy consumes."""
        return int(self._weights.shape[0])

    @property
    def dtype(self) -> torch.dtype:
        """Working precision."""
        return self._dtype

    def parameters(self) -> list[torch.Tensor]:
        """The tensors an optimizer updates: the one weight vector."""
        return [self._weights]

    def set_weights(self, values: torch.Tensor) -> None:
        """Replace the weights in place, keeping the same leaf tensor.

        Used by the optimizer and by tests that place a known truth in the
        parameters; assigning a new tensor instead would detach the graph
        every caller already holds.
        """
        if values.shape != self._weights.shape:
            msg = f"expected weights of shape {tuple(self._weights.shape)}, got {tuple(values.shape)}"
            raise ValueError(msg)
        with torch.no_grad():
            self._weights.copy_(values.to(self._dtype))

    def log_probabilities(self, features: torch.Tensor) -> torch.Tensor:
        """Log ``pi(a | s)`` over the actions ``features`` describes.

        Parameters
        ----------
        features : torch.Tensor
            Shape ``(n_actions, n_features)``, as
            :meth:`snakes_and_ladders.learn.environment.Environment.features` returns.

        Returns
        -------
        torch.Tensor
            Shape ``(n_actions,)``, summing to 1 after exponentiation and
            differentiable with respect to :attr:`weights`.

        Raises
        ------
        ValueError
            If ``features`` is not 2-D with ``n_features`` columns.
        """
        if features.ndim != 2 or features.shape[1] != self.n_features:
            msg = (
                f"expected features of shape (n_actions, {self.n_features}), "
                f"got {tuple(features.shape)}"
            )
            raise ValueError(msg)
        scores = features.to(self._dtype) @ self._weights
        return torch.log_softmax(scores, dim=0)

    def sample(self, features: torch.Tensor, rng: np.random.Generator) -> int:
        """Draw an action index from ``pi(. | s)``.

        Sampled through ``rng`` rather than through torch's global generator
        so an episode is reproducible from the one seed its caller declared,
        with no second source of randomness nobody named.
        """
        with torch.no_grad():
            probabilities = torch.exp(self.log_probabilities(features))
        return int(rng.choice(probabilities.shape[0], p=probabilities.numpy()))

    def greedy(self, features: torch.Tensor) -> int:
        """The most probable action, which is the zero-temperature limit.

        Scaling the weights by ``beta`` and letting ``beta`` grow drives the
        softmax to this action, so a greedy searcher is a limit of this
        policy class rather than a different kind of thing.
        """
        with torch.no_grad():
            return int(torch.argmax(self.log_probabilities(features)))


class TrainablePolicy(Protocol):
    """What the actor-critic and PPO trainers need: differentiable log-probabilities and the tensors behind them."""

    @property
    def dtype(self) -> torch.dtype: ...

    def parameters(self) -> list[torch.Tensor]: ...

    def log_probabilities(self, features: torch.Tensor) -> torch.Tensor: ...

    def sample(self, features: torch.Tensor, rng: np.random.Generator) -> int: ...

    def greedy(self, features: torch.Tensor) -> int: ...


class MLPPolicy:
    """Scores each action by a small multilayer perceptron on its features, then softmaxes (issue #313).

    A drop-in for :class:`LinearPolicy` where the reward is not linear in
    the features. The linear scorer stays the reference every claim is
    stated against first, because its weights are interpretable; this one
    is measured against it, not assumed better.

    Parameters
    ----------
    n_features : int
        Width of the feature vector the environment emits.
    hidden : int
        Width of the two hidden layers.
    generator : torch.Generator
        Source of the initial weights, so a run is reproducible from the
        seeds its caller declared.
    dtype : torch.dtype
        Working precision, ``float64`` by default for the same reason as
        :class:`LinearPolicy`.
    """

    def __init__(
        self,
        n_features: int,
        *,
        hidden: int,
        generator: torch.Generator,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if n_features < 1 or hidden < 1:
            msg = f"n_features and hidden must be >= 1, got {n_features}, {hidden}"
            raise ValueError(msg)
        self._dtype = dtype
        self._n_features = n_features
        self._net = torch.nn.Sequential(
            torch.nn.Linear(n_features, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, 1, bias=False),
        ).to(dtype)
        for parameter in self._net.parameters():
            if parameter.dim() > 1:
                torch.nn.init.xavier_uniform_(parameter, generator=generator)
            else:
                torch.nn.init.zeros_(parameter)

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def n_features(self) -> int:
        return self._n_features

    def parameters(self) -> list[torch.Tensor]:
        return list(self._net.parameters())

    def log_probabilities(self, features: torch.Tensor) -> torch.Tensor:
        """Log ``pi(a | s)`` over the actions ``features`` describes, as :meth:`LinearPolicy.log_probabilities`."""
        if features.ndim != 2 or features.shape[1] != self._n_features:
            msg = (
                f"expected features of shape (n_actions, {self._n_features}), "
                f"got {tuple(features.shape)}"
            )
            raise ValueError(msg)
        scores: torch.Tensor = self._net(features.to(self._dtype))[:, 0]
        return torch.log_softmax(scores, dim=0)

    def sample(self, features: torch.Tensor, rng: np.random.Generator) -> int:
        with torch.no_grad():
            probabilities = torch.exp(self.log_probabilities(features))
        return int(rng.choice(probabilities.shape[0], p=probabilities.numpy()))

    def greedy(self, features: torch.Tensor) -> int:
        with torch.no_grad():
            return int(torch.argmax(self.log_probabilities(features)))


class EpsilonGreedyPolicy:
    """Takes the wrapped policy's best action, except a fraction of the time.

    With probability ``epsilon`` it draws uniformly from the available
    actions, and otherwise takes the wrapped policy's greedy action. The
    uniform draw is what makes it able to accept a *worsening* move, which is
    the property issue #194 exists for: on a rugged landscape every episode
    otherwise ends at the first state no action improves, so an agent chooses
    which local optimum to enter and can never leave one.

    It is not trainable and deliberately exposes no weights. Exploration here
    is a fixed, declared property of the search rather than something learned,
    so that a measured escape rate is attributable to ``epsilon`` and not to a
    training run that happened alongside it. :class:`LinearPolicy` remains the
    thing REINFORCE fits.

    Parameters
    ----------
    policy : LinearPolicy
        Supplies the greedy action.

        **An untrained policy does not give hill climbing.** Weights start at
        zero, so every action scores alike and :meth:`LinearPolicy.greedy`
        returns the first one; wrapping it explores around "always take
        action 0", which resembles a searcher without being one. It produces
        plausible numbers rather than an error, which is what makes it worth
        stating here --- the measurement in issue #194 was first run this way
        and reported an escape rate of 0.000 where the baseline it had to
        reproduce was 0.480. Wrap a trained policy, or set the weights to a
        vector whose argmax is the action you mean.
        ``tests/regression/search/test_search_escape.py`` pins both halves.
    epsilon : float
        Probability of the uniform draw, in ``[0, 1]``. At ``0`` this is the
        wrapped policy's greedy action every step, which is *not* the same as
        hill climbing once the episode is allowed to continue past a local
        optimum: there the greedy action is the least-bad move, so a
        zero-epsilon agent still walks, deterministically.

    Raises
    ------
    ValueError
        If ``epsilon`` is outside ``[0, 1]``.
    """

    def __init__(self, policy: TrainablePolicy, epsilon: float) -> None:
        if not 0.0 <= epsilon <= 1.0:
            msg = f"epsilon must lie in [0, 1], got {epsilon}"
            raise ValueError(msg)
        self._policy = policy
        self._epsilon = epsilon

    @property
    def epsilon(self) -> float:
        """The exploration probability this policy was built with."""
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            msg = f"epsilon must lie in [0, 1], got {value}"
            raise ValueError(msg)
        self._epsilon = value

    def log_probabilities(self, features: torch.Tensor) -> torch.Tensor:
        """Log of the mixture this policy samples from: ``(1 - eps)`` on the greedy action plus ``eps / n`` everywhere.

        What an off-policy learner divides by when the episodes were
        collected under this policy (issue #313): the ratio
        ``pi(a | s) / beta(a | s)`` needs ``beta`` as a distribution, not
        only as a sampler. Detached, since nothing here is trained.
        """
        n_actions = int(features.shape[0])
        with torch.no_grad():
            probabilities = torch.full(
                (n_actions,), self._epsilon / n_actions, dtype=torch.float64
            )
            probabilities[self._policy.greedy(features)] += 1.0 - self._epsilon
        return torch.log(probabilities)

    def sample(self, features: torch.Tensor, rng: np.random.Generator) -> int:
        """Choose an action index: uniform with probability ``epsilon``.

        Parameters
        ----------
        features : torch.Tensor
            Action features, shape ``(n_actions, n_features)``.
        rng : np.random.Generator
            The only source of randomness, as for :meth:`LinearPolicy.sample`.

        Returns
        -------
        int
            Index into the available actions.
        """
        n_actions = int(features.shape[0])
        if self._epsilon > 0.0 and rng.random() < self._epsilon:
            return int(rng.integers(n_actions))
        return self._policy.greedy(features)
