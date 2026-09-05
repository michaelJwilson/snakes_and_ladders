"""The policy: a softmax over scored actions, and the gauge that entails.

``docs/tex`` states the form directly --- ``pi(a | s)`` proportional to
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

import numpy as np
import torch


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
