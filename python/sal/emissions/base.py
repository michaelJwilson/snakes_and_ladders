"""The emission interface: the protocols a family implements, the record a re-estimate returns, the refusals, and the covariate rule.

Every family in :mod:`sal.emissions` and every implementer
outside it --- :class:`~sal.sim.count_pairs.IndependentCountPair`
among them --- is written against what this module declares. It imports no
other submodule of the package, so each of them can import it.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, runtime_checkable

import numpy as np
import torch
from numpy.typing import ArrayLike

#: What a family accepts for a parameter vector. A plain list is admitted
#: because a fixture states its truth in one; the constructor converts, and
#: validates what it converted.
type Values = np.ndarray | torch.Tensor | Sequence[float] | Sequence[int]


#: The family a re-estimate returns, so a caller of a concrete family's M step
#: keeps that family's own parameters rather than the protocol's. Covariance is
#: sound because the record is frozen: nothing can write the narrower field
#: through the wider view.
FamilyT_co = TypeVar("FamilyT_co", covariant=True)


@dataclass(frozen=True)
class Reestimate(Generic[FamilyT_co]):
    """One emission M step, and whether what it returned is an answer.

    A closed-form M step needs none of this. An M step that is itself an
    optimization does: a silent inner failure surfaces as a non-monotone outer
    likelihood several EM iterations later, nowhere near its cause (issue
    #229).

    Parameters
    ----------
    emissions : FamilyT_co
        The re-estimated family, at its own type.
    converged : bool
        Whether an iterative M step settled; ``True`` by construction for a
        closed-form one. ``False`` is refusal-worthy, not a warning:
        ``likelihood/CLAUDE.md`` forbids returning a number read off
        iterations that never settled, and Baum-Welch raises on it.
    at_boundary : bool
        Whether a parameter reached the edge of the range this data can
        identify it over. **Not** an error: the estimate is a bound rather
        than a maximum, so a Wald interval around it summarizes nothing (issue
        #122). A caller counts these the way ``qa.opt_coverage`` counts a
        singular information matrix.
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


class ParameterDomainError(ValueError):
    """A family was given a parameter outside its domain: a scale, a rate or a shape not positive.

    A ``ValueError``, so every caller that refused one before still does. It
    has its own type because one caller reads it as information rather than
    a mistake: a Hamiltonian trajectory that drives a parameter out of its
    domain has diverged, and :mod:`sal.sample.hmc` rejects that
    proposal rather than stopping the chain (#912).
    """


class CovariateNotSupportedError(TypeError):
    """A family was given a per-observation covariate it cannot condition on.

    Refused rather than ignored. A covariate silently dropped is a model
    fitted to a different likelihood than the caller asked for, and the
    failure would show up as a recovery that is merely worse rather than as an
    error (root ``CLAUDE.md``: no silent behaviour changes).
    """


def trial_count(covariate: torch.Tensor | None, declared: torch.Tensor) -> torch.Tensor:
    """The trial count to score at: the covariate's, or the declared one.

    The covariate **overrides** ``declared`` rather than replacing it. A family
    keeps a per-state trial count so :attr:`BetaBinomialEmission.mean` and
    :attr:`~BetaBinomialEmission.variance` keep a value; a covariate says what
    the count actually was for each observation (issue #631).

    **The shape is the whole cost.** ``_beta_binomial_log_density`` is nine
    ``lgamma`` calls, and PyTorch evaluates each on its own operand's shape and
    broadcasts at the ``+``, not before. A covariate shaped ``(..., 1)``
    broadcasts along the state axis and leaves three terms at ``n_obs * K``,
    which is what the per-state count costs today. Materialised ``(..., K)`` by
    the caller, all nine become ``n_obs * K`` and the term count triples.

    Parameters
    ----------
    covariate : torch.Tensor | None
        One trial count per observation, trailing axis of length one.
    declared : torch.Tensor
        The family's per-state count, shape ``(n_states,)``.

    Returns
    -------
    torch.Tensor

    Raises
    ------
    ValueError
        If the covariate does not end in a singleton axis, or is not a
        positive integer count.
    """
    if covariate is None:
        return declared
    if covariate.ndim == 0 or covariate.shape[-1] != 1:
        msg = (
            f"a trial count per observation must end in a singleton axis so it "
            f"broadcasts along the states, got {tuple(covariate.shape)}; a "
            f"count materialised per state triples the lgamma term count"
        )
        raise ValueError(msg)
    return validated_trials(covariate, declared)


def exposure(covariate: torch.Tensor, declared: torch.Tensor) -> torch.Tensor:
    """The exposure to scale the rate by, shaped to broadcast along the states.

    The offset half of :func:`trial_count`, and the same layout rule for the
    same reason: shaped ``(..., 1)`` it broadcasts along the state axis, and
    materialised ``(..., n_states)`` by the caller it makes every term of the
    density ``n_obs * n_states`` (issue #631).

    Returns
    -------
    torch.Tensor

    Raises
    ------
    ValueError
        If the covariate does not end in a singleton axis, or is not positive.
    """
    if covariate.ndim == 0 or covariate.shape[-1] != 1:
        msg = (
            f"an exposure per observation must end in a singleton axis so it "
            f"broadcasts along the states, got {tuple(covariate.shape)}"
        )
        raise ValueError(msg)
    return validated_exposure(covariate, declared)


def validated_exposure(covariate: torch.Tensor, declared: torch.Tensor) -> torch.Tensor:
    """``covariate`` as exposures, in ``declared``'s dtype.

    The support check alone, without :func:`exposure`'s trailing-axis rule,
    for the M step --- which flattens over sequences and positions and never
    broadcasts along the states. A zero exposure marks the channel
    **unobserved** at that observation (issue #933): it scores log 1 and the
    M step drops it, whatever count it carries.

    Returns
    -------
    torch.Tensor

    Raises
    ------
    ValueError
        If an exposure is negative or not a number.
    """
    offsets = covariate.to(declared.dtype)
    if bool(((offsets < 0.0) | offsets.isnan()).any()):
        msg = "every exposure must be non-negative; zero marks the channel unobserved"
        raise ValueError(msg)
    return offsets


def validated_trials(covariate: torch.Tensor, declared: torch.Tensor) -> torch.Tensor:
    """``covariate`` as trial counts, in ``declared``'s dtype.

    The support check alone, without :func:`trial_count`'s trailing-axis rule.
    An M step flattens its covariate over sequences and positions and never
    broadcasts it along the states, so that rule would refuse the shape the
    protocol documents for :meth:`EmissionFamily.reestimate`.

    A trial count of zero marks the channel **unobserved** at that
    observation (issue #933), as a zero exposure does the total's.

    Returns
    -------
    torch.Tensor

    Raises
    ------
    ValueError
        If a count is not a non-negative integer.
    """
    counts = covariate.to(declared.dtype)
    if bool(((counts < 0) | (counts != counts.floor())).any()):
        msg = (
            "every trial count must be a non-negative integer; zero marks the "
            "channel unobserved"
        )
        raise ValueError(msg)
    return counts


def as_tensor(values: ArrayLike, dtype: torch.dtype | None = None) -> torch.Tensor:
    """``values`` as the tensor an M step or a draw computes on; a tensor is returned as given.

    The array half of :meth:`EmissionFamily.reestimate` and
    :meth:`EmissionFamily.sample` (issue #1011): neither takes a derivative,
    so a caller holding arrays hands them over as they are, and the family
    converts once at entry and computes as it did on a tensor. A tensor is
    **not** cast, so a caller that passed one before gets the same arithmetic.
    An array is read in ``dtype`` --- a family passes its
    :attr:`~EmissionFamily.observation_dtype` for the observations --- which
    is the tensor a caller built with that dtype before. A read-only array is
    copied, since a tensor over it would be writable.

    Returns
    -------
    torch.Tensor
    """
    if isinstance(values, torch.Tensor):
        return values
    array = np.asarray(values)
    if not array.flags.writeable:
        array = array.copy()
    return torch.as_tensor(array, dtype=dtype)


def as_array(values: ArrayLike, dtype: torch.dtype | None = None) -> np.ndarray:
    """``values`` as a NumPy array, in ``dtype`` where one is given.

    The inverse seam to :func:`as_tensor`, for an implementer whose own M step
    is array arithmetic before it calls a family's
    (:class:`~sal.sim.count_pairs.ReflectedEmission`). A tensor
    is detached, since nothing on this path is differentiated.

    Returns
    -------
    np.ndarray
    """
    if isinstance(values, torch.Tensor):
        tensor = values.detach() if dtype is None else values.detach().to(dtype)
        return tensor.numpy()
    if dtype is None:
        return np.asarray(values)
    return np.asarray(values, dtype=torch.empty(0, dtype=dtype).numpy().dtype)


def refuse_covariate(family: object, covariate: ArrayLike | None) -> None:
    """Raise unless ``covariate`` is ``None``.

    Public because an implementer outside this module calls it ---
    :class:`~sal.sim.count_pairs.IndependentCountPair` is the
    eighth :class:`EmissionFamily` and does not live here.

    Raises
    ------
    CovariateNotSupportedError
        If a covariate is supplied.
    """
    if covariate is None:
        return
    name = type(family).__name__
    msg = (
        f"{name} conditions on no per-observation covariate; pass None. An "
        "exposure or a trial count per observation is issue #631, and lands on "
        "NegativeBinomialEmission and BetaBinomialEmission."
    )
    raise CovariateNotSupportedError(msg)


@runtime_checkable
class CountEmissionFamily(Protocol):
    """An emission family over the non-negative integers, which has moments.

    Every count family here states a closed form for both moments, and their
    *ratio* separates them: below one for the binomial, exactly one for the
    Poisson, above one for the negative binomial and the beta-binomial. The
    moments are a protocol so a test can range over the families and assert
    that bracketing.

    Under the seam rule: the count families' shared moments, which bracket
    equidispersion in five families and carry a per-channel pair (issue #399).
    """

    @property
    def mean(self) -> torch.Tensor:
        """Per-state mean, shape ``(n_states,)``.

        A family whose observation is a tuple states one moment per channel,
        shape ``(n_states, n_channels)``: :class:`CountPairEmission` emits a
        depth and an allele count, and a single mean over the two would be a
        number in no unit.
        """
        ...  # pragma: no cover

    @property
    def variance(self) -> torch.Tensor:
        """Per-state variance, shape ``(n_states,)``, or per channel."""
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

    @abstractmethod
    def sample(
        self,
        states: np.ndarray,
        rng: np.random.Generator,
        covariate: ArrayLike | None = None,
    ) -> np.ndarray:
        """Draw one observation per entry of ``states``.

        Parameters
        ----------
        states : np.ndarray
            Emitting state per draw, shape ``(n_draws,)``.
        rng : np.random.Generator
            Generator, passed in rather than seeded here (``sim/CLAUDE.md``).
        covariate : ArrayLike | None
            One conditioning value per draw, shape ``(n_draws,)``: an array or
            a tensor, converted by :func:`as_tensor`. Conditioned on and never
            fitted. A family that cannot use one raises
            :class:`CovariateNotSupportedError` rather than dropping it
            (issue #631).

        Returns
        -------
        np.ndarray
            One observation per entry of ``states``.
        """
        ...  # pragma: no cover

    @abstractmethod
    def log_density(
        self, observations: torch.Tensor, covariate: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Score every observation under every state.

        Parameters
        ----------
        observations : torch.Tensor
            Observations of any leading shape ``(...)``.
        covariate : torch.Tensor | None
            One conditioning value per observation, shape ``(..., 1)`` so it
            broadcasts along the state axis rather than being materialised per
            state (issue #631). A family that cannot use one raises
            :class:`CovariateNotSupportedError`.

        Returns
        -------
        torch.Tensor
            Shape ``(..., n_states)``, differentiable with respect to this
            family's parameters. A log-probability where :attr:`is_discrete`,
            a log-density otherwise.
        """
        ...  # pragma: no cover

    @abstractmethod
    def bregman_divergence(self, observations: torch.Tensor) -> torch.Tensor:
        """``D_phi(T(y), mu_state)``: the divergence of this family's log-partition.

        The quantity ``Emission_Mixture++`` scores a candidate by
        (``eq:kmeanspp``; Banerjee et al., 2005). An exponential family writes
        ``p(y | theta) = exp(-D_phi(T(y), mu)) b_phi(T(y))``, so the divergence
        is the negative log density **less** ``-log b_phi(T(y))``, the log
        density's largest value over the family's own mean parameter, which
        depends on the observation alone. D-squared sampling normalizes its
        scores rather than shifting them, so that term is not a constant a
        seeding absorbs (issue #560).

        Parameters
        ----------
        observations : torch.Tensor
            Observations, of the shape :meth:`log_density` takes.

        Returns
        -------
        torch.Tensor
            Shape ``(..., n_states)``, non-negative, and zero at the state
            whose mean parameter is the observation.
        """
        ...  # pragma: no cover

    @abstractmethod
    def validate(self, observations: np.ndarray) -> None:
        """Raise if ``observations`` cannot have come from this family.

        Raises
        ------
        ValueError
            If an observation lies outside the family's support.
        """
        ...  # pragma: no cover

    @abstractmethod
    def reestimate(
        self,
        observations: ArrayLike,
        posterior: ArrayLike,
        covariate: ArrayLike | None = None,
    ) -> Reestimate[EmissionFamily]:
        """The Baum-Welch M step for this family alone.

        No derivative is taken through it, so every argument is an array or a
        tensor (issue #1011): an array is converted by :func:`as_tensor`, the
        observations in :attr:`observation_dtype`, and a tensor is used as
        given. The family it returns holds tensors, as every family does.

        Parameters
        ----------
        observations : ArrayLike
            Observations, shape ``(n_sequences, length)``.
        posterior : ArrayLike
            State posteriors ``P(state_t | observations)`` as probabilities,
            shape ``(n_sequences, length, n_states)``.
        covariate : ArrayLike | None
            One conditioning value per observation, shape
            ``(n_sequences, length)``. A family that cannot use one raises
            :class:`CovariateNotSupportedError`.

        Returns
        -------
        Reestimate
            The re-estimated family, and what its M step had to report about
            producing it.
        """
        ...  # pragma: no cover

    @abstractmethod
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

    @abstractmethod
    def named_parameters(self) -> Mapping[str, torch.Tensor]:
        """This family's parameters under the names the model states them in."""
        ...  # pragma: no cover
