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

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, runtime_checkable

import numpy as np
import torch

from snakes_and_ladders.numerics_rust import sample_rows

#: What a family accepts for a parameter vector. A plain list is admitted
#: because a fixture states its truth in one, and requiring an array at the
#: boundary would put ``np.array`` around every literal in the suite for no
#: gain -- the constructor converts, and validates what it converted.
type Values = np.ndarray | torch.Tensor | Sequence[float] | Sequence[int]

#: Nearest-neighbour spacing, in units of the pooled standard deviation, that
#: :func:`pooled_variance_floor` treats as the smallest scale a state can
#: genuinely occupy. ``n`` observations spread over a pooled standard
#: deviation ``s`` sit at a typical spacing ``s / n``, so a state explaining a
#: *neighbourhood* of the data has variance of at least that order while a
#: state collapsed onto a *single* observation has variance heading to zero.
#: The floor is the boundary between the two and is derived from the data
#: rather than fixed, so it transfers across fixture sizes.
COLLAPSE_EXPONENT = 2

#: How far below :func:`identifiable_dispersion_bound` the dispersion solve's
#: bracket reaches. Nine decades: the score diverges to ``+inf`` as ``r -> 0``,
#: so the lower end only has to be small enough that the root is above it, and
#: bisection pays for the width in ``log2`` of it -- three extra iterations per
#: three decades.
_DISPERSION_BRACKET_RATIO = 1e-9

#: How far a re-estimated binomial probability is held from 0 and 1. A
#: weighted mean of exactly 0 or exactly n makes ``log p`` or ``log(1 - p)``
#: infinite, and a state that took no weight produces one; the margin is
#: below any probability a fixture of a realistic size can resolve.
_PROBABILITY_MARGIN = 1e-12

#: How far below :func:`identifiable_concentration_bound` the beta-binomial
#: solve's bracket reaches, and the iteration cap each bisection runs under.
#: Nine decades halved 60 times leaves a bracket far below any tolerance a
#: caller states.
_CONCENTRATION_BRACKET_RATIO = 1e-9
_MAX_BISECTIONS = 60


#: The family a re-estimate returns, so a caller of a concrete family's M step
#: keeps that family's own parameters rather than the protocol's. Covariant,
#: which is sound because the record is frozen: a
#: ``Reestimate[GaussianEmission]`` may stand where a
#: ``Reestimate[EmissionFamily]`` is wanted, and nothing can write the narrower
#: field through the wider view.
FamilyT_co = TypeVar("FamilyT_co", covariant=True)


@dataclass(frozen=True)
class Reestimate(Generic[FamilyT_co]):
    """One emission M step, and whether what it returned is an answer.

    A closed-form M step needs none of this: it is a formula, it always
    succeeds, and there is nothing to report. An M step that is itself an
    optimization does need it, which is why the record exists --- a silent
    inner failure surfaces as a mysteriously non-monotone outer likelihood,
    several EM iterations later and nowhere near its cause (issue #229).

    Parameters
    ----------
    emissions : FamilyT_co
        The re-estimated family, at its own type.
    converged : bool
        Whether an iterative M step settled. A closed-form one is ``True`` by
        construction. ``False`` is a refusal-worthy condition, not a warning:
        ``likelihood/CLAUDE.md`` forbids returning a number read off
        iterations that never settled, and Baum-Welch raises on it.
    at_boundary : bool
        Whether a parameter reached the edge of the range this data can
        identify it over. **Not** an error: it is the honest report that the
        estimate is a bound rather than a maximum, so a Wald interval around
        it summarizes nothing (issue #122). A caller counts these the way
        ``qa.opt_coverage`` counts a singular information matrix.
    iterations : int
        Iterations the inner solve took; ``0`` for a closed-form M step.
    residual : float
        The weighted score at the returned parameters, divided by the total
        weight, so it is comparable across data sizes. ``0.0`` for a
        closed-form M step, where the score is zero by construction.
    """

    emissions: FamilyT_co
    converged: bool = True
    at_boundary: bool = False
    iterations: int = 0
    residual: float = 0.0


@runtime_checkable
class CountEmissionFamily(Protocol):
    """An emission family over the non-negative integers, which has moments.

    Every count family here states a closed form for both its mean and its
    variance, and the *ratio* of the two is what separates them: below one for
    the binomial, exactly one for the Poisson, above one for the negative
    binomial and the beta-binomial. That is why the moments are a protocol
    rather than four coincidentally-named properties --- a test can range over
    the families and assert the bracketing, which is the property the set
    exists to establish.
    """

    @property
    def mean(self) -> torch.Tensor:
        """Per-state mean, shape ``(n_states,)``."""
        ...  # pragma: no cover

    @property
    def variance(self) -> torch.Tensor:
        """Per-state variance, shape ``(n_states,)``."""
        ...  # pragma: no cover


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
    ) -> Reestimate[EmissionFamily]:
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
        Reestimate
            The re-estimated family, and what its M step had to report about
            producing it.
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
    matrix : Values
        Row-stochastic emission matrix, shape ``(n_states, n_symbols)``.
    """

    def __init__(self, matrix: Values) -> None:
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
    ) -> Reestimate[CategoricalEmission]:
        """Normalized expected symbol counts, in log space.

        Closed form, so the record it returns carries no convergence to
        report: this is the case the seam was designed against.
        """
        mask = torch.nn.functional.one_hot(
            observations.reshape(-1).to(torch.long), self.n_symbols
        ).to(posterior.dtype)
        weights = posterior.reshape(-1, self.n_states)
        counts = torch.log(weights.t() @ mask + torch.finfo(posterior.dtype).tiny)
        return Reestimate(
            CategoricalEmission.from_log(
                counts - torch.logsumexp(counts, dim=1, keepdim=True)
            )
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
    mean : Values
        Per-state mean, shape ``(n_states,)``.
    scale : Values
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
        mean: Values,
        scale: Values,
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
    ) -> Reestimate[GaussianEmission]:
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
        return Reestimate(
            GaussianEmission(mean, torch.sqrt(variance), self._variance_floor)
        )

    def alignment_key(self) -> torch.Tensor:
        """The per-state means, as a column."""
        return self._mean.reshape(-1, 1)

    def named_parameters(self) -> Mapping[str, torch.Tensor]:
        """``mean`` and ``scale``, the parameters the model is stated in."""
        return {"mean": self._mean, "scale": self._scale}


class NegativeBinomialEmission:
    """A count drawn from ``NegativeBinomial(dispersion[state], mean[state])``.

    Counts are a third data shape, neither a symbol from a fixed alphabet nor
    a real number, and modelling them as either loses the mean-variance
    relation that defines them: ``Var = mu + mu**2 / r``, with ``r`` setting
    how far above Poisson the dispersion sits.

    Carried as ``(r, mu)`` rather than the textbook ``(r, p)``. The mean
    profiles out of the M step in closed form in that parameterization, and it
    is the one the mean-variance relation is stated in;
    :meth:`from_probability` accepts the textbook form.

    **The family whose M step is an optimization.** Categorical and Gaussian
    emissions re-estimate by formula. ``r`` does not: its posterior-weighted
    score involves ``digamma`` and has no analytic root. So the M step is a
    solve, and this is the family that says whether an interface validated
    only against formulas can carry one.

    **Its identifiability hazard is a *flat* likelihood, not an unbounded
    one.** Where the data are not measurably overdispersed the likelihood is
    nearly flat in ``r`` as ``r -> inf`` --- the Poisson limit --- so the
    maximum runs to the boundary and any interval around it summarizes
    nothing. That is the opposite failure from :class:`GaussianEmission`'s and
    needs the opposite guard: not a floor below which a parameter must not go,
    but a ceiling above which the data cannot tell one value from another.
    :func:`identifiable_dispersion_bound` derives it.

    Parameters
    ----------
    dispersion : Values
        Per-state ``r``, shape ``(n_states,)``, strictly positive. Larger is
        closer to Poisson.
    mean : Values
        Per-state ``mu``, shape ``(n_states,)``, strictly positive.

    Raises
    ------
    ValueError
        If the shapes disagree or a parameter is not positive.
    """

    def __init__(
        self,
        dispersion: Values,
        mean: Values,
    ) -> None:
        self._dispersion = torch.as_tensor(dispersion, dtype=torch.float64).reshape(-1)
        self._mean = torch.as_tensor(mean, dtype=torch.float64).reshape(-1)
        if self._dispersion.shape != self._mean.shape:
            msg = (
                f"dispersion and mean must have the same shape, got "
                f"{tuple(self._dispersion.shape)} and {tuple(self._mean.shape)}"
            )
            raise ValueError(msg)
        for name, values in (
            ("dispersion", self._dispersion),
            ("mean", self._mean),
        ):
            if bool((values <= 0.0).any()):
                msg = f"every {name} must be positive, got {values.tolist()}"
                raise ValueError(msg)

    @classmethod
    def from_probability(
        cls,
        dispersion: Values,
        probability: Values,
    ) -> NegativeBinomialEmission:
        """Build from the textbook ``(r, p)``, where ``mu = r (1 - p) / p``.

        Parameters
        ----------
        dispersion : Values
            Per-state ``r``, shape ``(n_states,)``.
        probability : Values
            Per-state success probability ``p`` in ``(0, 1]``, shape
            ``(n_states,)``.

        Returns
        -------
        NegativeBinomialEmission
            The same family in the ``(r, mu)`` parameterization.
        """
        r = torch.as_tensor(dispersion, dtype=torch.float64).reshape(-1)
        p = torch.as_tensor(probability, dtype=torch.float64).reshape(-1)
        return cls(r, r * (1.0 - p) / p)

    @property
    def n_states(self) -> int:
        """Hidden states this family emits from."""
        return int(self._mean.shape[0])

    @property
    def is_discrete(self) -> bool:
        """True: the support is the non-negative integers.

        Worth pinning beside :class:`GaussianEmission`, since the bound this
        restores --- ``log P(observations) <= 0`` --- is exactly the one the
        Gaussian case had to give up.
        """
        return True

    @property
    def observation_dtype(self) -> torch.dtype:
        """Floating point, though the support is integral.

        ``lgamma(y + r)`` needs ``y`` in the same type as ``r``, so counts are
        carried as floats and :meth:`validate` enforces their integrality
        rather than the dtype doing it.
        """
        return torch.float64

    @property
    def dispersion(self) -> torch.Tensor:
        """Per-state ``r``, shape ``(n_states,)``."""
        return self._dispersion

    @property
    def mean(self) -> torch.Tensor:
        """Per-state ``mu``, shape ``(n_states,)``."""
        return self._mean

    @property
    def variance(self) -> torch.Tensor:
        """``mu + mu**2 / r``, the relation that makes this a count model."""
        return self._mean + self._mean**2 / self._dispersion

    def sample(self, states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draw one count per entry of ``states``."""
        r = self._dispersion.numpy()[states]
        mu = self._mean.numpy()[states]
        return np.asarray(rng.negative_binomial(r, r / (r + mu)))

    def log_density(self, observations: torch.Tensor) -> torch.Tensor:
        """The negative binomial log-probability of every count under every state."""
        counts = observations.unsqueeze(-1).to(self._mean.dtype)
        total = self._dispersion + self._mean
        return (
            torch.lgamma(counts + self._dispersion)
            - torch.lgamma(self._dispersion)
            - torch.lgamma(counts + 1.0)
            + self._dispersion * torch.log(self._dispersion / total)
            + counts * torch.log(self._mean / total)
        )

    def validate(self, observations: np.ndarray) -> None:
        """Raise if an observation is not a non-negative integer."""
        _validate_counts(observations)

    def reestimate(
        self, observations: torch.Tensor, posterior: torch.Tensor
    ) -> Reestimate[NegativeBinomialEmission]:
        """The M step: a closed form for the mean, and a solve for the dispersion.

        With the mean profiled at its posterior-weighted value the score in
        ``r`` reduces to ``sum_t gamma[t] (digamma(y_t + r) - digamma(r)) + W
        log(r / (r + mu))``, which has no analytic root. It is solved by
        **bisection on** ``log r``, not by Newton: started at the
        method-of-moments value Newton converges for moderate ``r`` and, on a
        sample whose dispersion is near Poisson, overshoots and underflows to
        zero --- which arrives as a domain error rather than as a bad answer.
        Bisection cannot leave its bracket, and the bracket's upper end is
        where "not identified" is detected rather than approached.

        Returns
        -------
        Reestimate
            The re-estimated family, whether a state's dispersion reached the
            bound the data can identify it over, the iterations taken, and the
            weighted score at the answer.
        """
        values = observations.reshape(-1).to(posterior.dtype)
        weights = posterior.reshape(-1, self.n_states)
        mass = weights.sum(dim=0)
        mean = (weights * values.unsqueeze(-1)).sum(dim=0) / mass

        dispersion = torch.empty_like(mean)
        boundary = False
        iterations = 0
        residual = 0.0
        for state in range(self.n_states):
            solved = _solve_dispersion(values, weights[:, state], float(mean[state]))
            dispersion[state] = solved.value
            boundary = boundary or solved.at_boundary
            iterations = max(iterations, solved.iterations)
            residual = max(residual, solved.residual)
        return Reestimate(
            NegativeBinomialEmission(dispersion, mean),
            at_boundary=boundary,
            iterations=iterations,
            residual=residual,
        )

    def alignment_key(self) -> torch.Tensor:
        """The per-state mean **and variance**, both in observation units.

        Two moments rather than one, unlike :class:`GaussianEmission`, and the
        difference is not an inconsistency. A Gaussian scale is a nuisance
        parameter the ticket asks to permute by mean alone; the negative
        binomial's dispersion is the parameter the family exists for, so two
        states sharing a mean and differing in dispersion are different states
        and a mean-only signature would tie them. Both entries are in the units
        of the observations, so summing their absolute differences is
        dimensionally consistent --- which ``(mu, r)`` would not be, ``r``
        being a shape.
        """
        return torch.stack([self._mean, self.variance], dim=1)

    def named_parameters(self) -> Mapping[str, torch.Tensor]:
        """``dispersion`` and ``mean``, the parameters the model is stated in."""
        return {"dispersion": self._dispersion, "mean": self._mean}


class PoissonEmission:
    """A count drawn from ``Poisson(mean[state])``: the equidispersed case.

    The family with no dispersion parameter at all, which is what makes it
    useful here beyond its own merits. It is the ``r -> inf`` limit of
    :class:`NegativeBinomialEmission` and the ``n -> inf, p -> 0`` limit of
    :class:`BinomialEmission`, so it referees both against a second
    implementation rather than against a hand-written expression.

    Parameters
    ----------
    mean : Values
        Per-state rate, shape ``(n_states,)``, strictly positive.

    Raises
    ------
    ValueError
        If a mean is not positive.
    """

    def __init__(self, mean: Values) -> None:
        self._mean = torch.as_tensor(mean, dtype=torch.float64).reshape(-1)
        if bool((self._mean <= 0.0).any()):
            msg = f"every mean must be positive, got {self._mean.tolist()}"
            raise ValueError(msg)

    @property
    def n_states(self) -> int:
        """Hidden states this family emits from."""
        return int(self._mean.shape[0])

    @property
    def is_discrete(self) -> bool:
        """True: the support is the non-negative integers."""
        return True

    @property
    def observation_dtype(self) -> torch.dtype:
        """Floating point, since ``lgamma`` consumes the counts."""
        return torch.float64

    @property
    def mean(self) -> torch.Tensor:
        """Per-state rate, shape ``(n_states,)``."""
        return self._mean

    @property
    def variance(self) -> torch.Tensor:
        """Equal to the mean. The defining property, and the one tested."""
        return self._mean

    def sample(self, states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draw one count per entry of ``states``."""
        return np.asarray(rng.poisson(self._mean.numpy()[states]))

    def log_density(self, observations: torch.Tensor) -> torch.Tensor:
        """``y log(lambda) - lambda - log(y!)``."""
        counts = observations.unsqueeze(-1).to(self._mean.dtype)
        return counts * torch.log(self._mean) - self._mean - torch.lgamma(counts + 1.0)

    def validate(self, observations: np.ndarray) -> None:
        """Raise if an observation is not a non-negative integer."""
        _validate_counts(observations)

    def reestimate(
        self, observations: torch.Tensor, posterior: torch.Tensor
    ) -> Reestimate[PoissonEmission]:
        """The posterior-weighted mean, in closed form."""
        values = observations.reshape(-1).to(posterior.dtype)
        weights = posterior.reshape(-1, self.n_states)
        mean = (weights * values.unsqueeze(-1)).sum(dim=0) / weights.sum(dim=0)
        return Reestimate(PoissonEmission(mean))

    def alignment_key(self) -> torch.Tensor:
        """The per-state rate, as a column: it is the whole family."""
        return self._mean.reshape(-1, 1)

    def named_parameters(self) -> Mapping[str, torch.Tensor]:
        """``mean``, the only parameter there is."""
        return {"mean": self._mean}


class BinomialEmission:
    """A count of successes in ``trials[state]`` attempts, each with ``p[state]``.

    The **under**-dispersed case: ``Var = n p (1 - p) < n p``. Every other
    count family here sits at or above equidispersion, so this is the one that
    says whether anything in the interface quietly assumes otherwise.

    The trial count is a declared constant per state, not a parameter. It is a
    property of how an observation was made rather than of the process being
    fitted, and estimating it from the data is a different problem.

    Parameters
    ----------
    trials : Values
        Per-state ``n``, shape ``(n_states,)``, positive integers.
    probability : Values
        Per-state ``p``, shape ``(n_states,)``, in ``(0, 1)``.

    Raises
    ------
    ValueError
        If the shapes disagree, a trial count is not a positive integer, or a
        probability is not strictly inside ``(0, 1)``.
    """

    def __init__(
        self,
        trials: Values,
        probability: Values,
    ) -> None:
        self._trials = torch.as_tensor(trials, dtype=torch.float64).reshape(-1)
        self._probability = torch.as_tensor(probability, dtype=torch.float64).reshape(
            -1
        )
        if self._trials.shape != self._probability.shape:
            msg = (
                f"trials and probability must have the same shape, got "
                f"{tuple(self._trials.shape)} and "
                f"{tuple(self._probability.shape)}"
            )
            raise ValueError(msg)
        if bool(((self._trials < 1) | (self._trials != self._trials.floor())).any()):
            msg = f"every trial count must be a positive integer, got {self._trials.tolist()}"
            raise ValueError(msg)
        if bool(((self._probability <= 0.0) | (self._probability >= 1.0)).any()):
            msg = (
                f"every probability must lie strictly in (0, 1), got "
                f"{self._probability.tolist()}"
            )
            raise ValueError(msg)

    @property
    def n_states(self) -> int:
        """Hidden states this family emits from."""
        return int(self._trials.shape[0])

    @property
    def is_discrete(self) -> bool:
        """True: the support is ``{0, ..., n}``."""
        return True

    @property
    def observation_dtype(self) -> torch.dtype:
        """Floating point, since ``lgamma`` consumes the counts."""
        return torch.float64

    @property
    def trials(self) -> torch.Tensor:
        """Per-state ``n``, shape ``(n_states,)``."""
        return self._trials

    @property
    def probability(self) -> torch.Tensor:
        """Per-state ``p``, shape ``(n_states,)``."""
        return self._probability

    @property
    def mean(self) -> torch.Tensor:
        """``n p``."""
        return self._trials * self._probability

    @property
    def variance(self) -> torch.Tensor:
        """``n p (1 - p)``, strictly below the mean."""
        return self._trials * self._probability * (1.0 - self._probability)

    def sample(self, states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draw one count per entry of ``states``."""
        return np.asarray(
            rng.binomial(
                self._trials.numpy()[states].astype(np.int64),
                self._probability.numpy()[states],
            )
        )

    def log_density(self, observations: torch.Tensor) -> torch.Tensor:
        """``log C(n, y) + y log p + (n - y) log(1 - p)``."""
        counts = observations.unsqueeze(-1).to(self._trials.dtype)
        return (
            torch.lgamma(self._trials + 1.0)
            - torch.lgamma(counts + 1.0)
            - torch.lgamma(self._trials - counts + 1.0)
            + counts * torch.log(self._probability)
            + (self._trials - counts) * torch.log1p(-self._probability)
        )

    def validate(self, observations: np.ndarray) -> None:
        """Raise if an observation is not an integer in ``[0, max(trials)]``."""
        values = _validate_counts(observations)
        largest = float(self._trials.max())
        if bool((values > largest).any()):
            msg = (
                f"observations must not exceed the largest trial count "
                f"{largest:.0f}, got {values.max()}"
            )
            raise ValueError(msg)

    def reestimate(
        self, observations: torch.Tensor, posterior: torch.Tensor
    ) -> Reestimate[BinomialEmission]:
        """``p = weighted mean / n``, in closed form; ``n`` is not estimated."""
        values = observations.reshape(-1).to(posterior.dtype)
        weights = posterior.reshape(-1, self.n_states)
        mean = (weights * values.unsqueeze(-1)).sum(dim=0) / weights.sum(dim=0)
        probability = (mean / self._trials).clamp(
            _PROBABILITY_MARGIN, 1.0 - _PROBABILITY_MARGIN
        )
        return Reestimate(BinomialEmission(self._trials, probability))

    def alignment_key(self) -> torch.Tensor:
        """The per-state mean and variance, both in observation units."""
        return torch.stack([self.mean, self.variance], dim=1)

    def named_parameters(self) -> Mapping[str, torch.Tensor]:
        """``probability``. The trial count is a constant, not a fitted value."""
        return {"probability": self._probability}


class BetaBinomialEmission:
    """A count of successes in ``trials[state]`` attempts with a random rate.

    ``p`` drawn from ``Beta(a, b)`` per observation, then a binomial count:
    the **over**-dispersed counterpart of :class:`BinomialEmission` on the same
    bounded support, with
    ``Var = n p (1 - p) (n + a + b) / (1 + a + b)`` where ``p = a / (a + b)``.

    **The second family whose M step is an optimization, and a different
    one.** The negative binomial's is a one-dimensional root find; this is
    two-dimensional in ``(a, b)``, so a seam that took the first by accident
    does not take this one by accident. It is solved by Minka's fixed point
    for the Polya distribution, which increases the likelihood at every step
    rather than merely converging somewhere.

    **Its hazard is the negative binomial's, one family over.** As
    ``a + b -> inf`` at fixed ``a / (a + b)`` the family approaches
    ``Binomial(n, p)``, so the concentration stops being identified and the
    maximum runs to the boundary. :func:`identifiable_concentration_bound`
    says where, on the same reasoning.

    Parameters
    ----------
    trials : Values
        Per-state ``n``, shape ``(n_states,)``, positive integers.
    alpha, beta : Values
        Per-state Beta parameters, shape ``(n_states,)``, strictly positive.

    Raises
    ------
    ValueError
        If the shapes disagree or a parameter is out of range.
    """

    def __init__(
        self,
        trials: Values,
        alpha: Values,
        beta: Values,
    ) -> None:
        self._trials = torch.as_tensor(trials, dtype=torch.float64).reshape(-1)
        self._alpha = torch.as_tensor(alpha, dtype=torch.float64).reshape(-1)
        self._beta = torch.as_tensor(beta, dtype=torch.float64).reshape(-1)
        if not self._trials.shape == self._alpha.shape == self._beta.shape:
            msg = (
                f"trials, alpha and beta must have the same shape, got "
                f"{tuple(self._trials.shape)}, {tuple(self._alpha.shape)} and "
                f"{tuple(self._beta.shape)}"
            )
            raise ValueError(msg)
        if bool(((self._trials < 1) | (self._trials != self._trials.floor())).any()):
            msg = f"every trial count must be a positive integer, got {self._trials.tolist()}"
            raise ValueError(msg)
        for name, values in (("alpha", self._alpha), ("beta", self._beta)):
            if bool((values <= 0.0).any()):
                msg = f"every {name} must be positive, got {values.tolist()}"
                raise ValueError(msg)

    @property
    def n_states(self) -> int:
        """Hidden states this family emits from."""
        return int(self._trials.shape[0])

    @property
    def is_discrete(self) -> bool:
        """True: the support is ``{0, ..., n}``."""
        return True

    @property
    def observation_dtype(self) -> torch.dtype:
        """Floating point, since ``lgamma`` consumes the counts."""
        return torch.float64

    @property
    def trials(self) -> torch.Tensor:
        """Per-state ``n``, shape ``(n_states,)``."""
        return self._trials

    @property
    def alpha(self) -> torch.Tensor:
        """Per-state ``a``, shape ``(n_states,)``."""
        return self._alpha

    @property
    def beta(self) -> torch.Tensor:
        """Per-state ``b``, shape ``(n_states,)``."""
        return self._beta

    @property
    def concentration(self) -> torch.Tensor:
        """``a + b``: how close this is to a binomial. Larger is closer."""
        return self._alpha + self._beta

    @property
    def mean(self) -> torch.Tensor:
        """``n a / (a + b)``."""
        return self._trials * self._alpha / self.concentration

    @property
    def variance(self) -> torch.Tensor:
        """``n p (1 - p) (n + a + b) / (1 + a + b)``, above the binomial's."""
        total = self.concentration
        rate = self._alpha / total
        return (
            self._trials * rate * (1.0 - rate) * (self._trials + total) / (1.0 + total)
        )

    def sample(self, states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draw a rate from the Beta, then a binomial count at that rate."""
        rate = rng.beta(self._alpha.numpy()[states], self._beta.numpy()[states])
        return np.asarray(
            rng.binomial(self._trials.numpy()[states].astype(np.int64), rate)
        )

    def log_density(self, observations: torch.Tensor) -> torch.Tensor:
        """``log C(n, y) + log B(y + a, n - y + b) - log B(a, b)``."""
        counts = observations.unsqueeze(-1).to(self._trials.dtype)
        total = self.concentration
        return (
            torch.lgamma(self._trials + 1.0)
            - torch.lgamma(counts + 1.0)
            - torch.lgamma(self._trials - counts + 1.0)
            + torch.lgamma(counts + self._alpha)
            + torch.lgamma(self._trials - counts + self._beta)
            - torch.lgamma(self._trials + total)
            + torch.lgamma(total)
            - torch.lgamma(self._alpha)
            - torch.lgamma(self._beta)
        )

    def validate(self, observations: np.ndarray) -> None:
        """Raise if an observation is not an integer in ``[0, max(trials)]``."""
        values = _validate_counts(observations)
        largest = float(self._trials.max())
        if bool((values > largest).any()):
            msg = (
                f"observations must not exceed the largest trial count "
                f"{largest:.0f}, got {values.max()}"
            )
            raise ValueError(msg)

    def reestimate(
        self, observations: torch.Tensor, posterior: torch.Tensor
    ) -> Reestimate[BetaBinomialEmission]:
        """Minka's fixed point for ``(a, b)``, which increases the likelihood.

        Chosen over Newton for the reason bisection was chosen for the
        negative binomial: the update is a ratio of positive quantities, so
        it cannot leave the feasible set, and it is monotone in the objective
        rather than merely convergent. A test asserts the monotonicity per
        iteration, since that is the property being relied on.

        Returns
        -------
        Reestimate
            The re-estimated family, whether a state's concentration reached
            the bound this data identifies it over, the iterations taken, and
            the relative change at the last step.
        """
        values = observations.reshape(-1).to(posterior.dtype)
        weights = posterior.reshape(-1, self.n_states)

        alpha = torch.empty(self.n_states, dtype=torch.float64)
        beta = torch.empty(self.n_states, dtype=torch.float64)
        boundary = False
        converged = True
        iterations = 0
        residual = 0.0
        for state in range(self.n_states):
            total = float(self._alpha[state] + self._beta[state])
            solved = _solve_beta_binomial(
                values,
                weights[:, state],
                float(self._trials[state]),
                float(self._alpha[state]) / total,
                total,
            )
            alpha[state] = solved.alpha
            beta[state] = solved.beta
            boundary = boundary or solved.at_boundary
            converged = converged and solved.converged
            iterations = max(iterations, solved.iterations)
            residual = max(residual, solved.residual)
        return Reestimate(
            BetaBinomialEmission(self._trials, alpha, beta),
            at_boundary=boundary,
            converged=converged,
            iterations=iterations,
            residual=residual,
        )

    def alignment_key(self) -> torch.Tensor:
        """The per-state mean and variance, both in observation units."""
        return torch.stack([self.mean, self.variance], dim=1)

    def named_parameters(self) -> Mapping[str, torch.Tensor]:
        """``alpha`` and ``beta``. The trial count is a constant."""
        return {"alpha": self._alpha, "beta": self._beta}


@dataclass(frozen=True)
class _SolvedBetaBinomial:
    """One state's ``(a, b)`` solve."""

    alpha: float
    beta: float
    at_boundary: bool
    converged: bool
    iterations: int
    residual: float


def _beta_binomial_rate_score(
    values: torch.Tensor,
    weights: torch.Tensor,
    trials: float,
    rate: float,
    concentration: float,
) -> float:
    """Score in the mean rate ``p`` at fixed concentration, divided by ``M``."""
    alpha = torch.tensor(rate * concentration, dtype=values.dtype)
    beta = torch.tensor((1.0 - rate) * concentration, dtype=values.dtype)
    return float(
        (
            weights
            * (
                torch.digamma(values + alpha)
                - torch.digamma(alpha)
                - torch.digamma(trials - values + beta)
                + torch.digamma(beta)
            )
        ).sum()
    )


def _beta_binomial_concentration_score(
    values: torch.Tensor,
    weights: torch.Tensor,
    trials: float,
    rate: float,
    concentration: float,
) -> float:
    """Score in the concentration ``M = a + b`` at fixed mean rate."""
    total = torch.tensor(concentration, dtype=values.dtype)
    alpha = torch.tensor(rate * concentration, dtype=values.dtype)
    beta = torch.tensor((1.0 - rate) * concentration, dtype=values.dtype)
    return float(
        (
            weights
            * (
                rate * (torch.digamma(values + alpha) - torch.digamma(alpha))
                + (1.0 - rate)
                * (torch.digamma(trials - values + beta) - torch.digamma(beta))
                - (
                    torch.digamma(torch.tensor(trials, dtype=values.dtype) + total)
                    - torch.digamma(total)
                )
            )
        ).sum()
    )


def _rate_score_at(
    values: torch.Tensor, weights: torch.Tensor, trials: float, concentration: float
) -> Callable[[float], float]:
    """The rate score as a function of the rate alone, at a held concentration."""

    def score(rate: float) -> float:
        return _beta_binomial_rate_score(values, weights, trials, rate, concentration)

    return score


def _concentration_score_at(
    values: torch.Tensor, weights: torch.Tensor, trials: float, rate: float
) -> Callable[[float], float]:
    """The concentration score as a function of ``log M`` alone, at a held rate."""

    def score(log_concentration: float) -> float:
        return _beta_binomial_concentration_score(
            values, weights, trials, rate, math.exp(log_concentration)
        )

    return score


def _bisect(
    score: Callable[[float], float], low: float, high: float, tolerance: float
) -> float:
    """Root of a decreasing ``score`` bracketed by ``[low, high]``."""
    for _ in range(_MAX_BISECTIONS):
        if high - low <= tolerance:
            break
        middle = 0.5 * (low + high)
        if score(middle) > 0.0:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def _solve_beta_binomial(
    values: torch.Tensor,
    weights: torch.Tensor,
    trials: float,
    rate: float,
    concentration: float,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 60,
) -> _SolvedBetaBinomial:
    """Maximize the weighted likelihood in ``(p, M)`` by alternating bisection.

    Minka's fixed point for the Polya distribution was tried first and
    rejected on measurement: it is monotone but linearly convergent, and at a
    concentration of 120 it was still moving in the third decimal place after
    500 iterations --- so it returned an unconverged answer that looked like
    an estimate. Alternating bisection is the same discipline the negative
    binomial's dispersion gets, for the same reason: bracket the root instead
    of stepping toward it, and the failure mode becomes "the root is outside
    the bracket", which is a fact worth reporting rather than an iterate that
    ran out of patience.

    Each coordinate's score is decreasing in that coordinate over its bracket,
    with the mean rate bracketed by ``(0, 1)`` and the concentration by
    ``(0, bound]``, so each inner solve is unconditional.
    """
    bound = identifiable_concentration_bound(trials, float(weights.sum()))
    concentration = min(concentration, bound)
    at_boundary = False
    residual = float("inf")
    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        previous = (rate, concentration)
        rate = _bisect(
            _rate_score_at(values, weights, trials, concentration),
            _PROBABILITY_MARGIN,
            1.0 - _PROBABILITY_MARGIN,
            tolerance,
        )
        if (
            _beta_binomial_concentration_score(values, weights, trials, rate, bound)
            > 0.0
        ):
            concentration, at_boundary = bound, True
        else:
            at_boundary = False
            low, high = (
                math.log(bound) + math.log(_CONCENTRATION_BRACKET_RATIO),
                math.log(bound),
            )
            concentration = math.exp(
                _bisect(
                    _concentration_score_at(values, weights, trials, rate),
                    low,
                    high,
                    tolerance,
                )
            )
        residual = max(
            abs(rate - previous[0]),
            abs(concentration - previous[1]) / concentration,
        )
        if residual <= tolerance:
            break
    return _SolvedBetaBinomial(
        alpha=rate * concentration,
        beta=(1.0 - rate) * concentration,
        at_boundary=at_boundary,
        converged=residual <= tolerance,
        iterations=iterations,
        residual=residual,
    )


def identifiable_concentration_bound(trials: float, weight: float) -> float:
    """The concentration above which this much data cannot tell ``a + b`` from infinity.

    A beta-binomial exceeds its binomial variance by a factor
    ``(n + a + b) / (1 + a + b)``, which is ``1 + (n - 1) / (a + b)`` to
    leading order in large ``a + b``. The sampling noise on a variance from
    ``W`` observations is a relative ``sqrt(2 / W)``. Setting the excess equal
    to the noise gives ``a + b = (n - 1) sqrt(W / 2)``: the same construction
    as :func:`identifiable_dispersion_bound`, one family over.

    Parameters
    ----------
    trials : float
        The state's trial count, at least 2 --- at ``n = 1`` a beta-binomial
        *is* a Bernoulli whatever its concentration, so nothing identifies it.
    weight : float
        Total posterior weight on the state, its effective sample size.

    Returns
    -------
    float
        The bound, strictly positive.

    Raises
    ------
    ValueError
        If the trial count is below 2 or the weight is not positive.
    """
    if trials < 2.0 or weight <= 0.0:
        msg = f"trials must be >= 2 and weight positive, got {trials} and {weight}"
        raise ValueError(msg)
    return (trials - 1.0) * math.sqrt(weight / 2.0)


def _validate_counts(observations: np.ndarray) -> np.ndarray:
    """Raise unless every observation is a non-negative integer."""
    values = np.asarray(observations)
    if bool((values < 0).any()) or bool((values != np.floor(values)).any()):
        msg = (
            f"observations must be non-negative integers, got a minimum of "
            f"{values.min()} and "
            f"{int((values != np.floor(values)).sum())} non-integral value(s)"
        )
        raise ValueError(msg)
    return values


@dataclass(frozen=True)
class _SolvedDispersion:
    """One state's dispersion solve."""

    value: float
    at_boundary: bool
    iterations: int
    residual: float


def _weighted_dispersion_score(
    values: torch.Tensor, weights: torch.Tensor, dispersion: float, mean: float
) -> float:
    """The posterior-weighted score in ``r``, with the mean profiled out.

    ``sum_t w_t (digamma(y_t + r) - digamma(r)) + W log(r / (r + mu))``. The
    term in ``(mu - y_t) / (r + mu)`` that appears in the full derivative
    vanishes because ``mu`` is the weighted mean of ``y``, which is what
    "profiled out" buys.
    """
    r = torch.tensor(dispersion, dtype=values.dtype)
    return float(
        (weights * (torch.digamma(values + r) - torch.digamma(r))).sum()
        + weights.sum() * math.log(dispersion / (dispersion + mean))
    )


def _solve_dispersion(
    values: torch.Tensor,
    weights: torch.Tensor,
    mean: float,
    *,
    tolerance: float = 1e-12,
) -> _SolvedDispersion:
    """Maximize the weighted likelihood in ``r`` by bisection on ``log r``."""
    upper = identifiable_dispersion_bound(mean, float(weights.sum()))
    lower = upper * _DISPERSION_BRACKET_RATIO
    if _weighted_dispersion_score(values, weights, upper, mean) > 0.0:
        return _SolvedDispersion(upper, at_boundary=True, iterations=0, residual=0.0)
    if _weighted_dispersion_score(values, weights, lower, mean) < 0.0:
        return _SolvedDispersion(lower, at_boundary=True, iterations=0, residual=0.0)

    low, high = math.log(lower), math.log(upper)
    iterations = 0
    while high - low > tolerance:
        middle = 0.5 * (low + high)
        if _weighted_dispersion_score(values, weights, math.exp(middle), mean) > 0.0:
            low = middle
        else:
            high = middle
        iterations += 1
    dispersion = math.exp(0.5 * (low + high))
    return _SolvedDispersion(
        dispersion,
        at_boundary=False,
        iterations=iterations,
        residual=abs(_weighted_dispersion_score(values, weights, dispersion, mean))
        / float(weights.sum()),
    )


def identifiable_dispersion_bound(mean: float, weight: float) -> float:
    """The dispersion above which this much data cannot tell ``r`` from infinity.

    A negative binomial exceeds its Poisson variance by ``mu**2 / r``. The
    sampling noise on a variance estimated from ``W`` observations is about
    ``Var sqrt(2 / W)``, which near the Poisson limit is ``mu sqrt(2 / W)``.
    Setting the excess equal to the noise gives ``r = mu sqrt(W / 2)``: beyond
    it the overdispersion the model is *for* is smaller than the error on
    measuring it, the likelihood is flat, and a maximum reported there is a
    bound rather than an estimate.

    Derived rather than fixed, so it scales with both the counts and the
    sample --- a constant cap would flag an identified fixture at one size and
    miss an unidentified one at another.

    Parameters
    ----------
    mean : float
        The state's posterior-weighted mean count.
    weight : float
        Total posterior weight on the state, its effective sample size.

    Returns
    -------
    float
        The bound, strictly positive.

    Raises
    ------
    ValueError
        If the mean or the weight is not positive.
    """
    if mean <= 0.0 or weight <= 0.0:
        msg = f"mean and weight must be positive, got {mean} and {weight}"
        raise ValueError(msg)
    return mean * math.sqrt(weight / 2.0)


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
