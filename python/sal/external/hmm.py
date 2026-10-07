"""The HMM's external solvers, in ``opt.hmm``'s terms (issue #1282, step 5).

``sal.external`` is namespaced by problem family: this module holds the HMM
calls, each through hmmlearn (:attr:`~sal.external.solvers.Solver.HMMLEARN`)
in a subprocess, and each taking its sibling's arguments in its order with a
:class:`~sal.external.solvers.Solver` after them.

- :func:`fit` is :func:`sal.opt.hmm.baum_welch_family`'s Baum-Welch, and
  returns its :class:`~sal.opt.hmm.EmFit` as :class:`ExternalFit`.
- :func:`viterbi` is :func:`sal.likelihood.hmm.viterbi`'s most probable path;
  the sibling's ``(states, log_probability)`` pair is :class:`ExternalPaths`,
  which unpacks to it.
- :func:`forward_log_likelihood` is hmmlearn's ``score``, the forward
  algorithm's log p(X), under :func:`sal.opt.hmm.forward_log_likelihood`'s
  name; :class:`ExternalLogLikelihood` reads as its ``float``.

Each result carries what it spent, a :class:`~sal.opt.termination.Termination`
and the answer's :class:`~sal.external.solvers.Provenance`.

**Families.** hmmlearn's ``CategoricalHMM``, one-channel ``GaussianHMM`` and
``PoissonHMM`` are :class:`~sal.emissions.CategoricalEmission`,
:class:`~sal.emissions.GaussianEmission` and
:class:`~sal.emissions.PoissonEmission`, by exact type. Every other family
needs a :class:`~sal.external.solvers.Capability` hmmlearn does not declare
(:func:`family_capability`), and is refused before any subprocess starts.

**Ragged segments.** As the siblings take them: :func:`fit` and
:func:`viterbi` a :class:`~sal.ragged.Ragged` batch, as
:func:`~sal.opt.hmm.baum_welch_family` and
:func:`sal.likelihood.ragged.viterbi` do; :func:`forward_log_likelihood`
``lengths``, as :func:`~sal.opt.hmm.forward_log_likelihood_ragged` does.
hmmlearn takes a segment of one position in each call. A fit whose every
segment has one position observes no transition, and hmmlearn returns a
transition matrix of zero rows for it; that fit is refused.

**The bytes.** The parameters are sent as probabilities,
:func:`probabilities` of the log tensors, posed with
:func:`~sal.external.hmm_inputs.hmm_inputs`, which the adapter
:mod:`sal.validation.hmmlearn` sends too, so each answer is the adapter's
bitwise (``tests/validation/test_hmmlearn.py``). A fitted probability is read
back through ``np.log``, a Gaussian variance through ``np.sqrt``.

**The fit's loop.** ``config`` is :class:`~sal.opt.em.EmConfig`: at most
``max_iterations`` iterations, stopped where the log-likelihood changes by at
most ``tolerance`` relative to its magnitude, :func:`sal.opt.em.em_loop`'s
test, run inside hmmlearn's own loop. ``log_likelihood`` is the last E
step's, at the parameters that iteration was handed, as ``em_loop`` reports
it. hmmlearn applies no variance floor and no prior: a Gaussian state whose
variance it drives to zero is a :class:`~sal.emissions.ParameterDomainError`
on the returned family.

**Cost.** :func:`fit` spends its EM iterations, in
:attr:`~sal.cost.Cost.ITERATIONS`, as its sibling does; :func:`viterbi` and
:func:`forward_log_likelihood` one pass each, :data:`PASS_UNIT`.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

import numpy as np
import torch

from sal.cost import Cost
from sal.emissions import (
    BetaBinomialEmission,
    BinomialEmission,
    CategoricalEmission,
    CountPairEmission,
    EmissionFamily,
    GaussianEmission,
    NegativeBinomialEmission,
    PoissonEmission,
)
from sal.external.hmm_inputs import hmm_inputs
from sal.external.runner import Run
from sal.external.sessions import Session, served_by
from sal.external.solvers import (
    Capability,
    ExternalUnavailable,
    Provenance,
    Solver,
    available,
    invoke,
    provenance,
    require,
)
from sal.opt.em import EM, EmConfig
from sal.opt.hmm import EmFit
from sal.opt.termination import Termination
from sal.ragged import Ragged

#: The unit :func:`viterbi` and :func:`forward_log_likelihood` are charged in:
#: one pass of hmmlearn's recursion over every segment.
PASS_UNIT = Cost.PASS

#: The families hmmlearn's three models are, by exact type.
_NAMED: Mapping[type, Capability] = {
    CategoricalEmission: Capability.CATEGORICAL_EMISSIONS,
    GaussianEmission: Capability.GAUSSIAN_EMISSIONS,
    PoissonEmission: Capability.POISSON_EMISSIONS,
}

#: Every other family the package defines, and its capability; a subclass
#: is its parent's family.
_OTHERS: tuple[tuple[type, Capability], ...] = (
    (BinomialEmission, Capability.BINOMIAL_EMISSIONS),
    (NegativeBinomialEmission, Capability.NEGATIVE_BINOMIAL_EMISSIONS),
    (BetaBinomialEmission, Capability.BETA_BINOMIAL_EMISSIONS),
    (CountPairEmission, Capability.COUNT_PAIR_EMISSIONS),
)


def family_capability(components: EmissionFamily) -> Capability:
    """The :class:`~sal.external.solvers.Capability` a solver needs to take ``components``.

    A Gaussian over more than one channel, or with a flat one, needs
    :attr:`~sal.external.solvers.Capability.MULTI_CHANNEL_EMISSIONS`; a
    family the package does not define,
    :attr:`~sal.external.solvers.Capability.UNNAMED_EMISSIONS`.
    """
    if isinstance(components, GaussianEmission) and (
        components.n_channels > 1 or components.flat
    ):
        return Capability.MULTI_CHANNEL_EMISSIONS
    named = _NAMED.get(type(components))
    if named is not None:
        return named
    for family, capability in _OTHERS:
        if isinstance(components, family):
            return capability
    return Capability.UNNAMED_EMISSIONS


def probabilities(log: torch.Tensor) -> np.ndarray:
    """``log``'s probabilities, as the script receives them: ``np.exp`` in ``float64``."""
    return np.exp(np.asarray(log.detach().numpy(), dtype=np.float64))


def emission_parameters(components: EmissionFamily) -> dict[str, np.ndarray]:
    """``components``'s parameters, named as :func:`~sal.external.hmm_inputs.family_of` reads them.

    A categorical's :func:`probabilities`, a Gaussian's ``mean`` and its
    ``scale`` squared, a Poisson's ``rate``.
    """
    if isinstance(components, CategoricalEmission):
        return {"emission": probabilities(components.log_matrix)}
    if isinstance(components, GaussianEmission):
        scale = components.scale.detach().numpy().reshape(-1)
        return {
            "mean": components.mean.detach().numpy().reshape(-1),
            "variance": scale * scale,
        }
    if isinstance(components, PoissonEmission):
        return {"rate": components.mean.detach().numpy()}
    msg = f"hmmlearn has no model for {type(components).__name__}"
    raise TypeError(msg)


def _fitted(start: EmissionFamily, outputs: Mapping[str, np.ndarray]) -> EmissionFamily:
    """The family hmmlearn fitted, in ``start``'s type and settings."""
    if isinstance(start, CategoricalEmission):
        return CategoricalEmission.from_log(
            torch.from_numpy(np.log(outputs["emission"]))
        )
    if isinstance(start, GaussianEmission):
        return GaussianEmission(
            outputs["mean"],
            np.sqrt(outputs["variance"]),
            start.variance_floor,
            on_collapse=start.on_collapse,
        )
    return PoissonEmission(outputs["rate"])


def _kernel(log_initial: torch.Tensor, log_transition: torch.Tensor) -> int:
    """The number of states, once ``log_transition`` is one ``(m, m)`` kernel.

    hmmlearn holds one kernel for every step; a per-step or per-sequence
    kernel, which :func:`~sal.opt.hmm.baum_welch_family` conditions on, is
    refused with :class:`ValueError`.
    """
    m = int(log_initial.shape[0])
    if log_initial.ndim != 1 or tuple(log_transition.shape) != (m, m):
        msg = (
            f"hmmlearn takes one ({m}, {m}) kernel after a ({m},) initial "
            f"distribution; got {tuple(log_initial.shape)} and "
            f"{tuple(log_transition.shape)}"
        )
        raise ValueError(msg)
    return m


def _segments(
    observations: np.ndarray | torch.Tensor | Ragged,
    lengths: Sequence[int] | None = None,
) -> tuple[np.ndarray, tuple[int, ...] | None]:
    """The observations as the script reads them, and their lengths where ragged.

    A rectangular batch is ``(n_sequences, length)``; a :class:`~sal.ragged.Ragged`
    batch, or ``(total,)`` with ``lengths``, is the segments end to end, each
    at least one position long (:class:`~sal.ragged.Ragged` checks it).
    """
    if isinstance(observations, Ragged):
        batch = observations
    elif lengths is not None:
        batch = Ragged(np.asarray(observations), tuple(int(n) for n in lengths))
    else:
        values = np.asarray(observations)
        if values.ndim != 2:
            msg = f"observations are (n_sequences, length); got shape {values.shape}"
            raise ValueError(msg)
        return values, None
    if batch.values.ndim != 1:
        msg = (
            f"ragged observations are one scalar per position, (total,); got "
            f"shape {batch.values.shape}"
        )
        raise ValueError(msg)
    return batch.values, batch.lengths


def _call(
    solver: Solver,
    needs: set[Capability],
    session: Session | None,
    inputs: Mapping[str, np.ndarray],
    timeout: float,
) -> Run:
    """``solver``'s script on ``inputs``, through ``session`` where one is open."""
    if session is None:
        return invoke(solver, needs, inputs, timeout=timeout)
    return session.invoke(needs, inputs, timeout=timeout)


def _checked(solver: Solver, session: Session | None) -> None:
    """The refusals every call makes after its capabilities, before it sends anything."""
    served_by(session, solver)
    if session is None and not available(solver):
        raise ExternalUnavailable(solver)


@dataclass(frozen=True)
class ExternalFit(EmFit):
    """A :class:`~sal.opt.hmm.EmFit` from an external solver, with its seconds and provenance."""

    #: Wall seconds of the framework's ``fit`` alone, as the script measured them.
    seconds: float = dataclass_field(kw_only=True)
    #: The framework, installed version and licence the fit came from.
    provenance: Provenance = dataclass_field(kw_only=True)


def fit(
    observations: np.ndarray | Ragged,
    log_initial: torch.Tensor,
    log_transition: torch.Tensor,
    components: EmissionFamily,
    solver: Solver,
    config: EmConfig = EM,
    *,
    timeout: float = 600.0,
    session: Session | None = None,
) -> ExternalFit:
    """``solver``'s Baum-Welch fit, as :func:`sal.opt.hmm.baum_welch_family` returns one.

    The sibling's first five arguments in its order, with the solver after
    the family; its keywords --- a covariate, a held transition, a backend,
    an E or M step --- set its own route, and hmmlearn has none of them.

    Parameters
    ----------
    observations : np.ndarray | Ragged
        ``(n_sequences, length)``, or a :class:`~sal.ragged.Ragged` batch of
        scalars: symbols, real values or counts, as ``components`` scores them.
    log_initial, log_transition : torch.Tensor
        The start, as log-probabilities: ``(m,)`` and one ``(m, m)`` kernel.
    components : EmissionFamily
        The starting family: categorical, one-channel Gaussian or Poisson.
    solver : Solver
        One that declares :attr:`~sal.external.solvers.Capability.HMM_FIT`
        and the family's capability.
    config : EmConfig
        At most ``max_iterations`` iterations, at least one, and the relative
        tolerance; :data:`~sal.opt.em.EM` by default, as the sibling's.
    timeout : float
        Seconds the subprocess may take.
    session : Session | None
        A worker :func:`sal.external.session` opened on ``solver``.

    Returns
    -------
    ExternalFit
        The fitted parameters, the last E step's log-likelihood, the
        :class:`~sal.opt.termination.Termination` after hmmlearn's count of
        iterations, ``spent`` that count, hmmlearn's ``fit`` seconds and the
        :class:`~sal.external.solvers.Provenance`.

    Raises
    ------
    CapabilityRefused
        If ``components`` is a family ``solver`` does not declare, or
        ``solver`` does not fit.
    ExternalUnavailable
        If the framework is not installed.
    ValueError
        If the kernel is not one ``(m, m)`` matrix, ``config`` allows no
        iteration, every segment has one position, the observations are
        neither shape, or ``session`` serves another solver.
    ScriptError
        If hmmlearn fails.
    """
    needs = {Capability.HMM_FIT, family_capability(components)}
    require(solver, needs)
    _kernel(log_initial, log_transition)
    if config.max_iterations < 1:
        msg = "hmmlearn reports a log-likelihood only after an iteration; got none"
        raise ValueError(msg)
    values, lengths = _segments(observations)
    if (lengths is not None and max(lengths) == 1) or (
        lengths is None and values.shape[1] == 1
    ):
        msg = (
            "every segment has one position, so no transition is observed and "
            "hmmlearn's re-estimate of the kernel is a matrix of zero rows"
        )
        raise ValueError(msg)
    _checked(solver, session)
    inputs = hmm_inputs(
        values,
        probabilities(log_initial),
        probabilities(log_transition),
        emission_parameters(components),
        call="fit",
        n_iter=config.max_iterations,
        tolerance=config.tolerance,
        lengths=lengths,
    )
    result = _call(solver, needs, session, inputs, timeout)
    outputs = result.outputs
    iterations = int(outputs["iterations"])
    return ExternalFit(
        torch.from_numpy(np.log(outputs["initial"])),
        torch.from_numpy(np.log(outputs["transition"])),
        _fitted(components, outputs),
        float(outputs["log_likelihood"]),
        termination=Termination.after(iterations, converged=bool(outputs["settled"])),
        spent=iterations,
        seconds=result.seconds,
        provenance=provenance(solver),
    )


@dataclass(frozen=True, kw_only=True)
class ExternalPaths:
    """:func:`sal.likelihood.hmm.viterbi`'s ``(states, log_probability)``, with what it spent and its provenance.

    It unpacks to the sibling's pair, ``states, log_probability = ...``, so
    a caller of either reads both the same way; the rest are read by name.
    """

    #: The most probable path: ``(n_sequences, length)`` ``int64``, or
    #: ``(total,)`` for a ragged batch, the segments end to end.
    states: np.ndarray
    #: The sum over segments of each path's joint log-probability.
    log_probability: float
    #: Passes spent, in :attr:`unit`: one.
    spent: int
    #: :data:`PASS_UNIT`.
    unit: Cost = PASS_UNIT
    #: One pass, run to its end.
    termination: Termination
    #: Wall seconds of the framework's ``decode`` alone.
    seconds: float
    #: The framework, installed version and licence the path came from.
    provenance: Provenance

    def __iter__(self) -> Iterator[Any]:
        """``(states, log_probability)``: the sibling's pair, in its order."""
        yield from (self.states, self.log_probability)


def viterbi(
    observations: np.ndarray | Ragged,
    log_initial: torch.Tensor,
    log_transition: torch.Tensor,
    components: EmissionFamily,
    solver: Solver,
    *,
    timeout: float = 600.0,
    session: Session | None = None,
) -> ExternalPaths:
    """``solver``'s most probable path of every segment, as :func:`sal.likelihood.hmm.viterbi` returns it.

    The sibling's first four arguments in its order, with the solver after
    them in place of its ``backend``. A tie is broken as hmmlearn breaks it.

    Parameters
    ----------
    observations : np.ndarray | Ragged
        ``(n_sequences, length)``, or a :class:`~sal.ragged.Ragged` batch of
        scalars, as ``components`` scores them.
    log_initial, log_transition : torch.Tensor
        Log-probabilities, ``(m,)`` and ``(m, m)``.
    components : EmissionFamily
        Categorical, one-channel Gaussian or Poisson.
    solver : Solver
        One that declares :attr:`~sal.external.solvers.Capability.VITERBI`
        and the family's capability.
    timeout : float
        Seconds the subprocess may take.
    session : Session | None
        A worker :func:`sal.external.session` opened on ``solver``.

    Returns
    -------
    ExternalPaths
        The paths and their total joint log-probability, one pass spent,
        and the :class:`~sal.external.solvers.Provenance`.

    Raises
    ------
    CapabilityRefused
        If ``components`` is a family ``solver`` does not declare, or
        ``solver`` does not decode.
    ExternalUnavailable
        If the framework is not installed.
    ValueError
        If the kernel is not one ``(m, m)`` matrix, the observations are
        neither shape, or ``session`` serves another solver.
    """
    needs = {Capability.VITERBI, family_capability(components)}
    require(solver, needs)
    _kernel(log_initial, log_transition)
    values, lengths = _segments(observations)
    _checked(solver, session)
    inputs = hmm_inputs(
        values,
        probabilities(log_initial),
        probabilities(log_transition),
        emission_parameters(components),
        call="decode",
        lengths=lengths,
    )
    result = _call(solver, needs, session, inputs, timeout)
    return ExternalPaths(
        states=result.outputs["states"],
        log_probability=float(result.outputs["log_probability"]),
        spent=1,
        termination=Termination.after(1, converged=True),
        seconds=result.seconds,
        provenance=provenance(solver),
    )


@dataclass(frozen=True, kw_only=True)
class ExternalLogLikelihood:
    """:func:`sal.opt.hmm.forward_log_likelihood`'s scalar, with what it spent and its provenance.

    No gradient is taken through a subprocess, so the value is a ``float``,
    and ``float(result)`` reads it as ``float`` reads the sibling's tensor.
    """

    #: The summed log-likelihood over segments; a density where the family is continuous.
    log_likelihood: float
    #: Passes spent, in :attr:`unit`: one.
    spent: int
    #: :data:`PASS_UNIT`.
    unit: Cost = PASS_UNIT
    #: One pass, run to its end.
    termination: Termination
    #: Wall seconds of the framework's ``score`` alone.
    seconds: float
    #: The framework, installed version and licence the value came from.
    provenance: Provenance

    def __float__(self) -> float:
        """:attr:`log_likelihood`."""
        return self.log_likelihood


def forward_log_likelihood(
    observations: np.ndarray | torch.Tensor,
    log_initial: torch.Tensor,
    log_transition: torch.Tensor,
    log_emission: torch.Tensor | EmissionFamily,
    solver: Solver,
    *,
    lengths: Sequence[int] | None = None,
    timeout: float = 600.0,
    session: Session | None = None,
) -> ExternalLogLikelihood:
    """``solver``'s forward log-likelihood, hmmlearn's ``score``, as :func:`sal.opt.hmm.forward_log_likelihood` returns it.

    The sibling's four arguments in its order, then the solver; ``lengths``
    reads the observations as segments end to end, as
    :func:`~sal.opt.hmm.forward_log_likelihood_ragged` takes them.

    Parameters
    ----------
    observations : np.ndarray | torch.Tensor
        ``(n_sequences, length)``, or ``(total,)`` with ``lengths``.
    log_initial, log_transition : torch.Tensor
        Log-probabilities, ``(m,)`` and ``(m, m)``.
    log_emission : torch.Tensor | EmissionFamily
        The sibling's log emission matrix, ``(m, n_symbols)``, for a
        categorical HMM; or a one-channel Gaussian or Poisson family.
    solver : Solver
        One that declares
        :attr:`~sal.external.solvers.Capability.LOG_LIKELIHOOD` and the
        family's capability.
    lengths : Sequence[int] | None
        One length per segment, each at least one, summing to ``total``.
    timeout : float
        Seconds the subprocess may take.
    session : Session | None
        A worker :func:`sal.external.session` opened on ``solver``.

    Returns
    -------
    ExternalLogLikelihood
        The summed log-likelihood, one pass spent, and the
        :class:`~sal.external.solvers.Provenance`.

    Raises
    ------
    CapabilityRefused
        If the family is one ``solver`` does not declare, or ``solver`` does
        not score.
    ExternalUnavailable
        If the framework is not installed.
    ValueError
        If the kernel is not one ``(m, m)`` matrix, the observations are
        neither shape, ``lengths`` do not tile them, or ``session`` serves
        another solver.
    """
    components = (
        CategoricalEmission.from_log(log_emission)
        if isinstance(log_emission, torch.Tensor)
        else log_emission
    )
    needs = {Capability.LOG_LIKELIHOOD, family_capability(components)}
    require(solver, needs)
    _kernel(log_initial, log_transition)
    values, segments = _segments(observations, lengths)
    _checked(solver, session)
    inputs = hmm_inputs(
        values,
        probabilities(log_initial),
        probabilities(log_transition),
        emission_parameters(components),
        call="score",
        lengths=segments,
    )
    result = _call(solver, needs, session, inputs, timeout)
    return ExternalLogLikelihood(
        log_likelihood=float(result.outputs["log_likelihood"]),
        spent=1,
        termination=Termination.after(1, converged=True),
        seconds=result.seconds,
        provenance=provenance(solver),
    )
