"""What a hidden state emits, separated from how the model is fitted.

An HMM's emission is the only part of the model that knows what an
observation *is*. The rest --- the forward recursion, the Baum-Welch E step,
path enumeration --- needs three things from it and nothing else: draw an
observation given a state, score an observation given a state, and
re-estimate itself from posterior state weights. This module is that
interface and its implementations; ``snakes_and_ladders.opt.hmm`` holds the
recursions that consume it.

**Why it is here rather than in a module that owns a model.** ``opt/CLAUDE.md``
forbids ``snakes_and_ladders.opt`` from importing ``snakes_and_ladders.sim``, and
``tests/regression/opt/test_opt_objective.py`` asserts it, so a family defined
in ``sim`` would be unreachable from the objective that has to score with it.
A family names no tree, no alignment and no lattice, so it sits beside
:mod:`snakes_and_ladders.numerics` and :mod:`snakes_and_ladders.enumeration` on
the same terms: importable from anywhere, inverting no layering.

**A density is not a probability, and the difference is load-bearing.**
:meth:`EmissionFamily.log_density` returns a log-probability for a family over
a countable alphabet and a log *density* for one over the reals. Only the
first is bounded above by zero, so the evidence ``log P(observations)`` of a
continuous-emission HMM may be positive and an assertion that it is not fails
on correct code. :attr:`EmissionFamily.is_discrete` states which case a family
is, so a test can assert the bound exactly where it holds.

**An unbounded likelihood is a property of the model, not a bug in the fit.**
A Gaussian emission's likelihood has no maximum: put one state's mean on a
single observation and let its variance go to zero and the likelihood
diverges (Bishop, *Pattern Recognition and Machine Learning*, section 9.2.1).
:class:`GaussianEmission` therefore carries an explicit variance floor and
**refuses** rather than clamps when a re-estimate reaches it, because a
clamped fit returns normally and its intervals mean nothing --- issue #122's
case, arrived at from a second direction.

Parameterization for an unconstrained optimizer is deliberately *not* here.
That is what ``snakes_and_ladders.opt.constrain`` is for, and putting it here
would make :mod:`snakes_and_ladders.sim` import ``snakes_and_ladders.opt``
transitively to draw a sequence.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import numpy as np
import torch

from snakes_and_ladders.numerics_rust import sample_rows

#: Nearest-neighbour spacing, in units of the pooled standard deviation, that
#: :func:`pooled_variance_floor` treats as the smallest scale a state can
#: genuinely occupy. ``n`` observations spread over a pooled standard
#: deviation ``s`` sit at a typical spacing ``s / n``, so a state explaining a
#: *neighbourhood* of the data has variance of at least that order while a
#: state collapsed onto a *single* observation has variance heading to zero.
#: The floor is the boundary between the two and is derived from the data
#: rather than fixed, so it transfers across fixture sizes.
COLLAPSE_EXPONENT = 2


@runtime_checkable
class EmissionFamily(Protocol):
    """The emission distribution of every hidden state, as one object.

    An implementation is immutable: :meth:`reestimate` returns a new family
    rather than updating in place, so an EM iterate cannot be aliased by the
    iterate before it.
    """

    @property
    def n_states(self) -> int:
        """Hidden states this family emits from."""
        ...  # pragma: no cover

    @property
    def is_discrete(self) -> bool:
        """Whether :meth:`log_density` returns a probability rather than a density.

        ``True`` for a family over a countable alphabet, where the evidence of
        a sequence is a probability and ``log P <= 0``. ``False`` for a family
        over the reals, where it is a density and that bound does not hold.
        """
        ...  # pragma: no cover

    @property
    def observation_dtype(self) -> torch.dtype:
        """Type an observation is carried in: integral or floating point."""
        ...  # pragma: no cover

    def sample(self, states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draw one observation per entry of ``states``.

        Parameters
        ----------
        states : np.ndarray
            Emitting state per draw, shape ``(n_draws,)``.
        rng : np.random.Generator
            Generator, passed in rather than seeded here (``sim/CLAUDE.md``).

        Returns
        -------
        np.ndarray
            One observation per entry of ``states``.
        """
        ...  # pragma: no cover

    def log_density(self, observations: torch.Tensor) -> torch.Tensor:
        """Score every observation under every state.

        Parameters
        ----------
        observations : torch.Tensor
            Observations of any leading shape ``(...)``.

        Returns
        -------
        torch.Tensor
            Shape ``(..., n_states)``, differentiable with respect to this
            family's parameters. A log-probability where :attr:`is_discrete`,
            a log-density otherwise.
        """
        ...  # pragma: no cover

    def validate(self, observations: np.ndarray) -> None:
        """Raise if ``observations`` cannot have come from this family.

        Raises
        ------
        ValueError
            If an observation lies outside the family's support.
        """
        ...  # pragma: no cover

    def reestimate(
        self, observations: torch.Tensor, posterior: torch.Tensor
    ) -> EmissionFamily:
        """The Baum-Welch M step for this family alone.

        Parameters
        ----------
        observations : torch.Tensor
            Observations, shape ``(n_sequences, length)``.
        posterior : torch.Tensor
            State posteriors ``P(state_t | observations)`` as probabilities,
            shape ``(n_sequences, length, n_states)``.

        Returns
        -------
        EmissionFamily
            The re-estimated family, of the same type and shape.
        """
        ...  # pragma: no cover

    def alignment_key(self) -> torch.Tensor:
        """Per-state signature the hidden-state permutation is matched on.

        Shape ``(n_states, d)``. Label switching is unidentifiable in any HMM,
        so a recovery test aligns before comparing; the emission is the
        discriminating signal, since two states emitting identically are the
        same state. What "identically" means is the family's to say --- a row
        of symbol probabilities for a categorical emission, a mean for a
        Gaussian one --- which is why this is on the family and not in the
        aligner.
        """
        ...  # pragma: no cover

    def named_parameters(self) -> Mapping[str, torch.Tensor]:
        """This family's parameters under the names the model states them in."""
        ...  # pragma: no cover


class CategoricalEmission:
    """A symbol drawn from a per-state distribution over a fixed alphabet.

    The family every HMM claim in this repository rested on before there was
    an interface to state it against.

    Parameters
    ----------
    matrix : np.ndarray | torch.Tensor
        Row-stochastic emission matrix, shape ``(n_states, n_symbols)``.
    """

    def __init__(self, matrix: np.ndarray | torch.Tensor) -> None:
        values = torch.as_tensor(matrix, dtype=torch.float64)
        if values.ndim != 2 or values.shape[1] < 1:
            msg = f"emission matrix must be 2-D, got shape {tuple(values.shape)}"
            raise ValueError(msg)
        # Both forms are stored rather than one derived on demand. A sampler
        # reads the probabilities and a recursion reads their logarithm, and
        # a round trip through ``exp(log(p))`` moves the last bits of a
        # probability -- enough to move an inverse-CDF draw at a cell
        # boundary, which would change a pinned simulated sequence for no
        # reason a reader could find.
        self._matrix = values
        self._log_matrix = torch.log(values)

    @classmethod
    def from_log(cls, log_matrix: torch.Tensor) -> CategoricalEmission:
        """Build from log-probabilities, the form the recursions carry.

        Parameters
        ----------
        log_matrix : torch.Tensor
            Log emission matrix, shape ``(n_states, n_symbols)``.

        Returns
        -------
        CategoricalEmission
            A family holding exactly these log-probabilities, without a
            round trip through ``exp`` and ``log``.
        """
        family = cls.__new__(cls)
        family._log_matrix = log_matrix
        family._matrix = torch.exp(log_matrix)
        return family

    @property
    def n_states(self) -> int:
        """Hidden states this family emits from."""
        return int(self._log_matrix.shape[0])

    @property
    def n_symbols(self) -> int:
        """Size of the emission alphabet."""
        return int(self._log_matrix.shape[1])

    @property
    def is_discrete(self) -> bool:
        """True: the alphabet is finite, so the evidence is a probability."""
        return True

    @property
    def observation_dtype(self) -> torch.dtype:
        """Symbols are indices into the alphabet."""
        return torch.long

    @property
    def log_matrix(self) -> torch.Tensor:
        """Log emission matrix, shape ``(n_states, n_symbols)``."""
        return self._log_matrix

    @property
    def matrix(self) -> torch.Tensor:
        """Emission matrix as probabilities, shape ``(n_states, n_symbols)``."""
        return self._matrix

    def sample(self, states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draw one symbol per entry of ``states`` by inverse-CDF sampling."""
        return sample_rows(rng, self._matrix.numpy(), states)

    def log_density(self, observations: torch.Tensor) -> torch.Tensor:
        """Gather ``log B[:, symbol]`` for every observation."""
        return self._log_matrix.t()[observations]

    def validate(self, observations: np.ndarray) -> None:
        """Raise if a symbol lies outside the alphabet."""
        low, high = int(observations.min()), int(observations.max())
        if low < 0 or high >= self.n_symbols:
            msg = f"observations must lie in [0, {self.n_symbols}), got [{low}, {high}]"
            raise ValueError(msg)

    def reestimate(
        self, observations: torch.Tensor, posterior: torch.Tensor
    ) -> CategoricalEmission:
        """Normalized expected symbol counts, in log space."""
        mask = torch.nn.functional.one_hot(
            observations.reshape(-1).to(torch.long), self.n_symbols
        ).to(posterior.dtype)
        weights = posterior.reshape(-1, self.n_states)
        counts = torch.log(weights.t() @ mask + torch.finfo(posterior.dtype).tiny)
        return CategoricalEmission.from_log(
            counts - torch.logsumexp(counts, dim=1, keepdim=True)
        )

    def alignment_key(self) -> torch.Tensor:
        """The emission rows themselves, as probabilities."""
        return self.matrix

    def named_parameters(self) -> Mapping[str, torch.Tensor]:
        """``log_emission``, the form the forward recursion consumes."""
        return {"log_emission": self._log_matrix}


class GaussianEmission:
    """A real observation drawn from ``Normal(mean[state], scale[state])``.

    The family whose likelihood has **no maximum**. With ``mean[s]`` on a
    single observation and ``scale[s] -> 0`` the density at that point
    diverges, so a Gaussian-emission fit that converges was stopped by its
    initialization or by a floor, and which one has to be knowable. This class
    makes it knowable: the floor is explicit, derived from the data, and
    reaching it is a refusal rather than a clamp.

    Parameters
    ----------
    mean : np.ndarray | torch.Tensor
        Per-state mean, shape ``(n_states,)``.
    scale : np.ndarray | torch.Tensor
        Per-state standard deviation, shape ``(n_states,)``, strictly
        positive.
    variance_floor : float
        Variance at or below which :meth:`reestimate` refuses. Derive it from
        the data with :func:`pooled_variance_floor` rather than choosing a
        constant, so it scales with the fixture.

    Raises
    ------
    ValueError
        If the shapes disagree, a scale is not positive, or the floor is not
        positive.
    """

    def __init__(
        self,
        mean: np.ndarray | torch.Tensor,
        scale: np.ndarray | torch.Tensor,
        variance_floor: float,
    ) -> None:
        self._mean = torch.as_tensor(mean, dtype=torch.float64).reshape(-1)
        self._scale = torch.as_tensor(scale, dtype=torch.float64).reshape(-1)
        if self._mean.shape != self._scale.shape:
            msg = (
                f"mean and scale must have the same shape, got "
                f"{tuple(self._mean.shape)} and {tuple(self._scale.shape)}"
            )
            raise ValueError(msg)
        if bool((self._scale <= 0.0).any()):
            msg = f"every scale must be positive, got {self._scale.tolist()}"
            raise ValueError(msg)
        if variance_floor <= 0.0:
            msg = f"variance_floor must be positive, got {variance_floor}"
            raise ValueError(msg)
        self._variance_floor = variance_floor

    @property
    def n_states(self) -> int:
        """Hidden states this family emits from."""
        return int(self._mean.shape[0])

    @property
    def is_discrete(self) -> bool:
        """False: the support is the real line, so the evidence is a density."""
        return False

    @property
    def observation_dtype(self) -> torch.dtype:
        """Observations are real."""
        return torch.float64

    @property
    def mean(self) -> torch.Tensor:
        """Per-state mean, shape ``(n_states,)``."""
        return self._mean

    @property
    def scale(self) -> torch.Tensor:
        """Per-state standard deviation, shape ``(n_states,)``."""
        return self._scale

    @property
    def variance_floor(self) -> float:
        """Variance at or below which a re-estimate is refused."""
        return self._variance_floor

    def sample(self, states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draw one real observation per entry of ``states``."""
        return np.asarray(
            rng.normal(
                loc=self._mean.numpy()[states], scale=self._scale.numpy()[states]
            )
        )

    def log_density(self, observations: torch.Tensor) -> torch.Tensor:
        """The Normal log-density of every observation under every state.

        Unbounded above: as a scale shrinks with its mean on an observation,
        the value at that observation grows without bound. That is the model,
        not a defect, and a test exhibits it.
        """
        centred = observations.unsqueeze(-1).to(self._mean.dtype) - self._mean
        return (
            -0.5 * torch.log(torch.tensor(2.0 * torch.pi, dtype=self._mean.dtype))
            - torch.log(self._scale)
            - 0.5 * (centred / self._scale) ** 2
        )

    def validate(self, observations: np.ndarray) -> None:
        """Raise if an observation is not finite; the support is all of R."""
        if not bool(np.isfinite(observations).all()):
            msg = "observations must be finite"
            raise ValueError(msg)

    def reestimate(
        self, observations: torch.Tensor, posterior: torch.Tensor
    ) -> GaussianEmission:
        """Posterior-weighted mean and variance, in closed form.

        Raises
        ------
        ValueError
            If a state's re-estimated variance reaches
            :attr:`variance_floor`. Refusal rather than clamping: the
            likelihood is unbounded in that direction, so a fit that clamped
            and returned normally would report a point estimate at a
            degenerate optimum and a Wald interval around it that summarizes
            nothing --- issue #122's case.
        """
        values = observations.reshape(-1).to(posterior.dtype)
        weights = posterior.reshape(-1, self.n_states)
        mass = weights.sum(dim=0)
        mean = (weights * values.unsqueeze(-1)).sum(dim=0) / mass
        variance = (weights * (values.unsqueeze(-1) - mean) ** 2).sum(dim=0) / mass
        collapsed = variance <= self._variance_floor
        if bool(collapsed.any()):
            states = torch.nonzero(collapsed).reshape(-1).tolist()
            msg = (
                f"state(s) {states} re-estimated to variance "
                f"{variance[collapsed].tolist()}, at or below the floor "
                f"{self._variance_floor:.6g}: the Gaussian likelihood is "
                f"unbounded as a variance goes to zero, so this fit is "
                f"heading to a degenerate optimum rather than converging"
            )
            raise ValueError(msg)
        return GaussianEmission(mean, torch.sqrt(variance), self._variance_floor)

    def alignment_key(self) -> torch.Tensor:
        """The per-state means, as a column."""
        return self._mean.reshape(-1, 1)

    def named_parameters(self) -> Mapping[str, torch.Tensor]:
        """``mean`` and ``scale``, the parameters the model is stated in."""
        return {"mean": self._mean, "scale": self._scale}


def pooled_variance_floor(observations: np.ndarray) -> float:
    """The variance below which a Gaussian state has collapsed onto a point.

    ``n`` observations spread over a pooled standard deviation ``s`` sit at a
    typical nearest-neighbour spacing ``s / n``, so a state explaining a
    *neighbourhood* of the data carries variance of at least that order, while
    a state collapsed onto a *single* observation carries variance heading to
    zero. The floor is ``s**2 / n**2``: the boundary between the two, derived
    from the data rather than chosen, so it transfers across fixture sizes
    instead of being retuned per fixture.

    Parameters
    ----------
    observations : np.ndarray
        Every observation the fit will see, of any shape.

    Returns
    -------
    float
        The floor, strictly positive.

    Raises
    ------
    ValueError
        If fewer than two observations are supplied, or they are all equal ---
        in both cases there is no scale to derive a floor from.
    """
    values = np.asarray(observations, dtype=np.float64).reshape(-1)
    if values.size < 2:
        msg = f"need at least 2 observations to derive a floor, got {values.size}"
        raise ValueError(msg)
    pooled = float(values.var(ddof=1))
    if pooled <= 0.0:
        msg = "observations have zero spread, so no variance floor follows"
        raise ValueError(msg)
    return pooled / float(values.size) ** COLLAPSE_EXPONENT
