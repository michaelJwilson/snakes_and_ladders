"""The EM drivers: Baum-Welch, categorical and per family, and the route each takes between the torch recursion and the compiled ``oxisal`` steps.

A driver here owns the iteration and the record a fit returns; the recursion
it runs is :mod:`~sal.opt.hmm.forward`'s, or a compiled kernel
under :data:`~sal.backend.Backend.RUST`. Imports
:mod:`~sal.opt.hmm.forward` alone.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch

from sal import oxisal
from sal.backend import Backend, refuse_backend
from sal.emissions import (
    BetaBinomialEmission,
    BinomialEmission,
    CategoricalEmission,
    EmissionFamily,
    GaussianEmission,
    NegativeBinomialEmission,
    PoissonEmission,
    refuse_collapsed,
)
from sal.opt.em import EM, EmConfig, em_loop
from sal.opt.termination import Termination
from sal.ragged import Ragged


@dataclass(frozen=True)
class EmFit:
    """What one Baum-Welch run produced, and what it had to report.

    A tuple sufficed while every M step was a formula, and stopped when one
    became a solve (issue #229): a dispersion at the edge of what the data
    identifies is not an error, but a caller that cannot see it will build a
    Wald interval around a bound and call it an estimate.

    Parameters
    ----------
    log_initial, log_transition : torch.Tensor
        Fitted transition parameters, as log-probabilities.
    emissions : EmissionFamily
        The fitted emission family.
    log_likelihood : float
        The final log-likelihood. A density, and so possibly positive, where
        the family is continuous.
    emission_at_boundary : bool
        Whether any M step returned a parameter at the edge of the range this
        data identifies it over.
    termination : Termination | None
        Whether the outer loop met its relative tolerance and after how many
        EM iterations (issue #860). An unconverged emission M step is refused
        rather than reported, and still is: this answers for the outer loop.
    """

    log_initial: torch.Tensor
    log_transition: torch.Tensor
    emissions: EmissionFamily
    log_likelihood: float
    emission_at_boundary: bool = False
    termination: Termination | None = None


@dataclass(frozen=True)
class CategoricalFit:
    """What :func:`baum_welch` fitted, as log-probabilities.

    :class:`EmFit` is the general form, carrying a family rather than a
    matrix and a field a categorical M step cannot fill: no categorical
    re-estimate sits at a boundary. The outer loop's termination is carried
    as every EM fit carries it (issue #1059); the four-tuple an unpacking
    reads is what it was (issue #865).

    Parameters
    ----------
    log_initial : torch.Tensor
        Shape ``(m,)``, the fitted initial distribution.
    log_transition : torch.Tensor
        Shape ``(m, m)``, the fitted transition kernel.
    log_emission : torch.Tensor
        Shape ``(m, n_symbols)``, the fitted emission matrix.
    log_likelihood : float
        The final log-likelihood.
    termination : Termination | None
        Why the EM loop stopped and after how many iterations, as
        :class:`EmFit` reports it.
    """

    log_initial: torch.Tensor
    log_transition: torch.Tensor
    log_emission: torch.Tensor
    log_likelihood: float
    termination: Termination | None = None

    def __iter__(self) -> Iterator[Any]:
        """The declared order (#865): the three parameters, the value, the termination.

        ``Any`` and not a union: an unpacking gives every name the element
        type, so a union would mistype each of them.
        """
        yield from (
            self.log_initial,
            self.log_transition,
            self.log_emission,
            self.log_likelihood,
            self.termination,
        )


def baum_welch(
    observations: np.ndarray,
    log_initial: torch.Tensor,
    log_transition: torch.Tensor,
    log_emission: torch.Tensor,
    config: EmConfig = EM,
    backend: Backend = Backend.RUST,
) -> CategoricalFit:
    """Fit an HMM by expectation-maximization, with no autodiff involved.

    Baum-Welch is an independent fitting algorithm for the same model, so a
    gradient fit checked against it is checked against something other than
    itself. It shares no code with ``fit`` -- not the optimizer, not the
    parameterization (EM works directly in probabilities) -- only the model.

    Parameters
    ----------
    observations : np.ndarray
        Integer symbols, shape ``(n_sequences, sequence_length)``.
    log_initial, log_transition, log_emission : torch.Tensor
        Starting parameters, as log-probabilities.
    config : EmConfig
        The EM budget and its relative tolerance; :data:`~sal.opt.em.EM`,
        500 iterations at 1e-12, by default. Relative, since an absolute
        tolerance does not transfer across data sizes (``DEV.md``, issue #111).
    backend : Backend
        :data:`~sal.backend.Backend.RUST`, the default since
        issue #986, runs each E and M step in
        ``oxisal.categorical_em_step``: scaled messages
        streamed one sequence at a time into expected counts, so no array
        over every position is held. At 10^5 positions of three states one
        fit peaked at 62.6 MB on the batched route.
        :data:`~sal.backend.Backend.PYTHON` is that route,
        :func:`baum_welch_family` in log space, and the oracle that pins the
        compiled one within 1e-10.

    Returns
    -------
    CategoricalFit
        Fitted log initial, log transition and log emission, and the final
        log-likelihood.
    """
    refuse_backend("baum_welch", backend, (Backend.PYTHON, Backend.RUST))
    if backend is Backend.RUST:
        return _streamed_baum_welch(
            observations,
            log_initial,
            log_transition,
            log_emission,
            config=config,
        )
    result = baum_welch_family(
        observations,
        log_initial,
        log_transition,
        CategoricalEmission.from_log(log_emission),
        config=config,
    )
    family = result.emissions
    if not isinstance(family, CategoricalEmission):  # pragma: no cover
        msg = f"expected a categorical M step, got {type(family).__name__}"
        raise TypeError(msg)
    return CategoricalFit(
        result.log_initial,
        result.log_transition,
        family.log_matrix,
        result.log_likelihood,
        termination=result.termination,
    )


def _ragged_e_step(
    emit: torch.Tensor,
    mask: torch.Tensor,
    lengths: np.ndarray,
    log_initial: torch.Tensor,
    log_transition: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """The E step in the compiled ragged kernel, returned in the padded layout (issue #933).

    ``emit`` is the masked ``(n, length, m)`` block; its live rows, in
    sequence order, are what the kernel walks. The log marginals are
    scattered back to the block with ``-inf`` at padding, as the torch
    recursion leaves them, so the M step reads one layout either way.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, float]
        Log gamma ``(n, length, m)``, log transition counts ``(m, m)``
        summed over sequences, and the log-likelihood.
    """
    values = np.ascontiguousarray(emit[mask].numpy(), dtype=np.float64)
    m = values.shape[1]
    gamma = np.empty_like(values)
    counts = np.empty((m, m), dtype=np.float64)
    evidence = np.empty(lengths.shape[0], dtype=np.float64)
    oxisal.ragged_posteriors(
        values,
        lengths,
        np.ascontiguousarray(log_initial.numpy(), dtype=np.float64),
        np.ascontiguousarray(log_transition.numpy(), dtype=np.float64),
        gamma,
        counts,
        evidence,
    )
    padded = torch.full(emit.shape, -float("inf"), dtype=emit.dtype)
    padded[mask] = torch.as_tensor(gamma)
    return padded, torch.as_tensor(counts), float(evidence.sum())


class CovariateUpdate(Protocol):
    """A covariate that depends on the parameters, recomputed before every E step (issue #933).

    Called with the current emission family, the posterior of the previous E
    step --- shape ``(n_sequences, length, m)``, zero at padded positions,
    and uniform over the states before the first --- and the covariate as
    :func:`baum_welch_family` holds it. It returns the covariate the next E
    step scores against and the M step after it conditions on, in the same
    layout. A normalizer that is a function of the parameters, as
    :class:`ExpectedRateNormalizer` is, enters the fit this way.
    """

    def __call__(
        self,
        emissions: EmissionFamily,
        posterior: torch.Tensor,
        covariate: torch.Tensor,
    ) -> torch.Tensor:
        """The covariate to score the next E step against."""
        ...


@dataclass(frozen=True)
class ExpectedRateNormalizer:
    """Divide each sequence's exposure by ``Z_c = sum_g lambda_g E[mu_{s_c(g)}]`` (issue #933).

    A consumer whose rate at position ``g`` of sequence ``c`` is ``lambda_g
    mu_s / Z_c`` normalizes each sequence to one: ``Z_c`` sums the unnormalized
    rates over the sequence's positions, and it depends on the states the
    sequence takes, so no fixed covariate can hold it. This is its plug-in
    form: the state at each position is averaged over the previous E step's
    posterior, so ``Z_c`` moves with the parameters and the posterior and the
    fit is expectation--conditional-maximization around it, the emission M
    step conditioning on ``Z`` held at its current value.

    ``mu_k`` is the family's first alignment coordinate: the mean, in
    observation units, of every count family and of a count pair's total
    channel. The rates are unchanged by scaling every ``mu_k`` together,
    since ``Z_c`` scales with them, so only their ratios are identified.

    Parameters
    ----------
    weights : torch.Tensor
        ``lambda``, shape ``(length,)`` for one per position or
        ``(n_sequences, length)`` for one per position of each sequence.
    channel : int | None
        The covariate channel holding the exposure, for a family whose
        covariate carries one per channel (a count pair's is channel 0);
        ``None`` for a single-channel family.
    """

    weights: torch.Tensor
    channel: int | None = None

    def normalizer(
        self, emissions: EmissionFamily, posterior: torch.Tensor
    ) -> torch.Tensor:
        """``Z_c`` per sequence, shape ``(n_sequences,)``."""
        means = emissions.alignment_key()[:, 0].to(posterior.dtype)
        expected = posterior @ means  # (n_sequences, length)
        return (self.weights.to(posterior.dtype) * expected).sum(dim=-1)

    def __call__(
        self,
        emissions: EmissionFamily,
        posterior: torch.Tensor,
        covariate: torch.Tensor,
    ) -> torch.Tensor:
        """``covariate`` with its exposure divided by each sequence's ``Z_c``."""
        divisor = self.normalizer(emissions, posterior)[:, None]
        updated = covariate.clone()
        if self.channel is None:
            updated[..., 0] = covariate[..., 0] / divisor
        else:
            updated[..., self.channel] = covariate[..., self.channel] / divisor
        return updated


def _streamed_baum_welch(
    observations: np.ndarray,
    log_initial: torch.Tensor,
    log_transition: torch.Tensor,
    log_emission: torch.Tensor,
    *,
    config: EmConfig,
) -> CategoricalFit:
    """:func:`baum_welch` on the compiled step, driven by the same :func:`em_loop`."""
    # Borrowed where NumPy already holds int64 rows; a copy only otherwise.
    symbols = np.ascontiguousarray(observations, dtype=np.int64)
    m, n_symbols = log_emission.shape

    def step(
        state: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], float]:
        initial, transition, emission, log_likelihood = oxisal.categorical_em_step(
            symbols, *state
        )
        return (initial, transition, emission), log_likelihood

    def flat(tensor: torch.Tensor) -> np.ndarray:
        return np.ascontiguousarray(tensor.detach().numpy(), dtype=np.float64).reshape(
            -1
        )

    (initial, transition, emission), log_likelihood, termination = em_loop(
        step,
        (flat(log_initial), flat(log_transition), flat(log_emission)),
        config=config,
    )
    # Shaped in NumPy and wrapped without a copy: a first torch `reshape` in a
    # process costs 2.1 MB of resident memory, 60 times the fit's own.
    return CategoricalFit(
        torch.from_numpy(initial),
        torch.from_numpy(transition.reshape(m, m)),
        torch.from_numpy(emission.reshape(m, n_symbols)),
        log_likelihood,
        termination=termination,
    )


#: The count families :func:`baum_welch_family` streams through a table: one
#: channel, no covariate, and an M step that reads only weighted counts.
_TABLED = (
    PoissonEmission,
    BinomialEmission,
    NegativeBinomialEmission,
    BetaBinomialEmission,
)


#: The most cells a count table may span, ``(max count + 1) * (max covariate
#: + 1)``: its row lookup is four bytes a cell, so 4 MB at most.
_MOST_CELLS = 1 << 20


def _streams(
    observations: np.ndarray | Ragged,
    log_transition: torch.Tensor,
    emissions: EmissionFamily,
    covariate: np.ndarray | Ragged | None,
) -> bool:
    """Whether :func:`baum_welch_family` has a streamed step for this case (issue #997)."""
    m = emissions.n_states
    if (
        not isinstance(observations, np.ndarray)
        or observations.ndim != 2
        or observations.size == 0
        or log_transition.shape != (m, m)
    ):
        return False
    if type(emissions) is GaussianEmission:
        return covariate is None and emissions.n_channels == 1
    # Integer counts and an integer covariate: the table is indexed by them.
    if type(emissions) not in _TABLED or not _whole(observations):
        return False
    stride = 1
    if covariate is not None:
        if not (
            isinstance(covariate, np.ndarray)
            and covariate.shape == observations.shape
            and _whole(covariate)
        ):
            return False
        stride = int(covariate.max()) + 1
    return (int(observations.max()) + 1) * stride <= _MOST_CELLS


def _whole(values: np.ndarray) -> bool:
    """Whether ``values`` is an integer array with nothing below zero."""
    return np.issubdtype(values.dtype, np.integer) and int(values.min()) >= 0


def _direct_scoring(family: EmissionFamily) -> dict[str, Any]:
    """The compiled step's name and stacked parameters for ``family`` (issue #997)."""
    rows: list[torch.Tensor]
    if isinstance(family, PoissonEmission):
        name, rows = "poisson", [family.mean]
    elif isinstance(family, BinomialEmission):
        name, rows = "binomial", [family.trials, family.probability]
    elif isinstance(family, NegativeBinomialEmission):
        name, rows = "negative_binomial", [family.dispersion, family.mean]
    else:
        assert isinstance(family, BetaBinomialEmission)
        name, rows = "beta_binomial", [family.trials, family.alpha, family.beta]
    parameters = np.concatenate([row.detach().numpy().reshape(-1) for row in rows])
    return {"family": name, "parameters": parameters.astype(np.float64)}


def _streamed_family(
    observations: np.ndarray,
    log_initial: torch.Tensor,
    log_transition: torch.Tensor,
    emissions: EmissionFamily,
    *,
    config: EmConfig,
    covariate: np.ndarray | None = None,
    with_table: bool = True,
    table_size: int | None = None,
    approx: bool = False,
    stirling_from: float = 10.0,
) -> EmFit:
    """:func:`baum_welch_family` on a compiled step that streams (issue #997).

    A one-channel Gaussian is scored and re-estimated in
    ``oxisal.gaussian_em_step``. A count family is scored in
    ``count_em_step``, term for term its ``log_density`` with a compiled
    ``lgamma``: once per cell for the ``table_size`` cells the data repeats
    most --- a cell is a distinct count, or a distinct pair of a count and its
    covariate --- and at every observation for the rest. The step returns each state's posterior
    weight on each cell, and the family's own ``reestimate`` runs on those,
    which is its M step exactly, the order of summation aside. No gradient is
    taken here, so torch's ``log_density`` is not needed on this route; it
    stays the batched oracle and the autodiff objectives' density.
    """
    m = emissions.n_states
    at_boundary = False

    def flat(tensor: torch.Tensor) -> np.ndarray:
        return np.ascontiguousarray(tensor.detach().numpy(), dtype=np.float64).reshape(
            -1
        )

    if isinstance(emissions, GaussianEmission):
        values = np.ascontiguousarray(observations, dtype=np.float64)
        floor = emissions.variance_floor

        def gaussian(
            state: tuple[np.ndarray, np.ndarray, EmissionFamily],
        ) -> tuple[tuple[np.ndarray, np.ndarray, EmissionFamily], float]:
            initial, transition, family = state
            assert isinstance(family, GaussianEmission)
            initial, transition, mean, variance, log_likelihood = (
                oxisal.gaussian_em_step(
                    values, initial, transition, flat(family.mean), flat(family.scale)
                )
            )
            refuse_collapsed(torch.from_numpy(variance), floor)
            fitted = GaussianEmission(mean, np.sqrt(variance), floor)
            return (initial, transition, fitted), log_likelihood

        step = gaussian
    else:
        # Borrowed where NumPy already holds int64 rows; a copy only otherwise.
        counts = np.ascontiguousarray(observations, dtype=np.int64)
        given = (
            None
            if covariate is None
            else np.ascontiguousarray(covariate, dtype=np.int64)
        )
        stride = 1 if given is None else int(given.max()) + 1
        cells, multiplicity = oxisal.count_cells(counts, given, stride)
        rows = np.zeros((int(counts.max()) + 1) * stride, dtype=np.uint32)
        rows[cells] = np.arange(cells.size, dtype=np.uint32)
        # The table holds the `table_size` cells that repeat most; ties go to
        # the lower cell, so the choice depends on the data alone.
        size = 0 if not with_table else cells.size if table_size is None else table_size
        in_table = np.zeros(cells.size, dtype=bool)
        in_table[np.argsort(-multiplicity, kind="stable")[: max(size, 0)]] = True
        support = torch.from_numpy((cells // stride).astype(np.float64))
        # The family broadcasts a covariate along its states, as
        # `baum_welch_family` hands it one.
        exposure = (
            None
            if given is None
            else torch.from_numpy((cells % stride).astype(np.float64)).unsqueeze(1)
        )

        def tabled(
            state: tuple[np.ndarray, np.ndarray, EmissionFamily],
        ) -> tuple[tuple[np.ndarray, np.ndarray, EmissionFamily], float]:
            nonlocal at_boundary
            initial, transition, family = state
            initial, transition, histogram, log_likelihood = oxisal.count_em_step(
                counts,
                given,
                stride,
                cells,
                rows,
                initial,
                transition,
                **_direct_scoring(family),
                tabled=in_table,
                approx=approx,
                stirling_from=stirling_from,
            )
            reestimate = family.reestimate(
                support, torch.from_numpy(histogram.reshape(-1, m)), covariate=exposure
            )
            if not reestimate.converged:
                msg = (
                    f"the emission M step did not settle after "
                    f"{reestimate.iterations} iterations, at a relative change "
                    f"of {reestimate.residual:.3e}"
                )
                raise ValueError(msg)
            at_boundary = at_boundary or reestimate.at_boundary
            return (initial, transition, reestimate.emissions), log_likelihood

        step = tabled

    (initial, transition, fitted), log_likelihood, termination = em_loop(
        step,
        (flat(log_initial), flat(log_transition), emissions),
        config=config,
    )
    return EmFit(
        log_initial=torch.from_numpy(initial),
        log_transition=torch.from_numpy(transition.reshape(m, m)),
        emissions=fitted,
        log_likelihood=log_likelihood,
        emission_at_boundary=at_boundary,
        termination=termination,
    )


def compiled_family(emissions: EmissionFamily) -> tuple[str, np.ndarray] | None:
    """The compiled HMM kernels' family name and parameters for ``emissions``, or ``None``.

    Exactly a categorical, a one-channel Gaussian or a count family is
    compiled; ``None`` for any other, whose caller takes its oracle.
    :mod:`sal.likelihood.hmm`'s Viterbi and evidence read it (issue #1059).
    """
    if type(emissions) is CategoricalEmission:
        return "categorical", np.ascontiguousarray(
            emissions.log_matrix.detach().numpy(), dtype=np.float64
        ).reshape(-1)
    if type(emissions) is GaussianEmission and emissions.n_channels == 1:
        stacked = torch.cat([emissions.mean, emissions.scale]).detach().numpy()
        return "gaussian", np.ascontiguousarray(stacked, dtype=np.float64)
    if type(emissions) in _TABLED:
        scoring = _direct_scoring(emissions)
        return str(scoring["family"]), scoring["parameters"]
    return None


def baum_welch_family(
    observations: np.ndarray | Ragged,
    log_initial: torch.Tensor,
    log_transition: torch.Tensor,
    emissions: EmissionFamily,
    config: EmConfig = EM,
    covariate: np.ndarray | Ragged | None = None,
    *,
    update: CovariateUpdate | None = None,
    fit_transition: bool = True,
    backend: Backend = Backend.RUST,
    with_table: bool = True,
    table_size: int | None = None,
    approx: bool = False,
    stirling_from: float = 10.0,
) -> EmFit:
    """Baum-Welch over any emission family, with no autodiff involved.

    Takes a rectangular ``(n_sequences, length, ...)`` array as it always has,
    or a `Ragged` batch whose segments differ in length. There is one recursion
    underneath either: the rectangular form converts, and the route it replaced
    is conserved in `sandbox.rectangular_hmm` as the referee of that case
    (issue #666).

    The E step is the model: forward and backward messages in log space,
    identical whatever a state emits, and so is the M step for the initial
    distribution and the transitions, both simplex-valued for every family.
    Only the emission M step differs, and it is delegated to the family. The
    alternation around them is :func:`sal.opt.em.em_loop`'s
    (issue #859); what this function computes in one iteration is below.

    Parameters
    ----------
    observations : np.ndarray
        Observations, shape ``(n_sequences, length)`` followed by whatever
        trailing axes the family's observation carries --- none for a scalar
        observation, a channel axis for a family over a pair of counts.
        Symbol indices or real values, as the family says.
    emissions : EmissionFamily
        Starting emission family.
    config : EmConfig
        The EM budget and its relative tolerance; :data:`~sal.opt.em.EM`,
        500 iterations at 1e-12, by default. Relative, since an absolute
        tolerance does not transfer across data sizes (``DEV.md``, issue #111).
    log_initial, log_transition : torch.Tensor
        Starting parameters, as log-probabilities. ``log_transition`` is
        ``(m, m)``, one kernel for the whole chain; ``(length - 1, m, m)``, one
        per step (issue #658); or ``(n_sequences, length - 1, m, m)``, one per
        step of each sequence (issue #933), where a sequence's steps past its
        own end are never read. ``length`` is the longest sequence's.

        **A per-step kernel is conditioned on, not fitted.** It carries
        ``(length - 1) * m * (m - 1)`` free values against ``length - 1``
        transitions per sequence, so at the sequence counts this repository
        runs it is not identifiable and an M step that re-estimated it would
        return the posterior it was handed. Given one, this function holds it
        fixed and fits the initial distribution and the emissions --- the same
        standing a covariate has, and for the same reason. A kernel per
        sequence is held fixed on the same terms. Given a single matrix it
        fits that matrix, exactly as before.
    covariate : np.ndarray | None
        What each observation is scored against, carrying the observations'
        leading ``(n_sequences, length)`` -- an exposure for a rate family, a trial count for a
        bounded one (issue #652). It reaches both seams of the loop, the E
        step's scoring and the emission M step, because a fit that scores
        against an exposure and re-estimates without it is fitting two
        different models. ``None`` is the model this function had before.
        The family broadcasts a covariate along the states, so it wants a trailing singleton axis; the covariate is stored with the observations' own axes and the singleton is added here, where the observation layout is known. A caller should not have to carry a shape that exists for the family's broadcast.
    update : CovariateUpdate | None
        A covariate that depends on the parameters (issue #933): before every
        E step it maps the family, the previous posterior and ``covariate`` to
        the covariate that E step and the next M step use. It needs a
        ``covariate`` to update. The reported log-likelihood is each E step's,
        at the covariate it was scored against, so it is the objective of the
        plug-in fit and need not rise monotonically.
    fit_transition : bool
        Whether the M step re-estimates a single ``(m, m)`` transition. With
        ``False`` it is held, as a per-step or per-sequence kernel always is
        (issue #933).
    backend : Backend
        The E step's recursion. The default,
        :data:`~sal.backend.Backend.RUST`, walks each sequence
        in the compiled ragged kernel (issue #933): 19.4x the torch recursion
        on 200 chains of 100-3,000 positions. It takes one kernel for the
        whole chain, so a per-step or per-sequence kernel keeps the torch
        recursion under either. :data:`~sal.backend.Backend.PYTHON`
        is the padded torch recursion and the oracle, agreeing within 2e-12.

    backend : Backend
        :data:`~sal.backend.Backend.RUST`, the default since
        issue #997, streams the E step one sequence at a time into
        sufficient statistics, where the family is exactly a one-channel
        :class:`GaussianEmission`, or a :class:`PoissonEmission`,
        :class:`BinomialEmission`, :class:`NegativeBinomialEmission` or
        :class:`BetaBinomialEmission` over integer counts with no covariate or
        an integer one, the observations are a rectangular array, and the
        kernel is one ``(m, m)`` matrix.
    with_table : bool
        How the streamed count step scores an observation, read only there.
        ``True``, the default, scores each occupied (count, covariate) cell
        once and gathers, which pays where counts repeat. ``False`` scores
        every observation, which pays where they do not. Both hand the same
        weighted cells to the family's M step.
    table_size : int | None
        How many cells the table holds under ``with_table``: the ones the
        data repeats most, which for counts are mostly the low ones; every
        other observation is scored where it stands. ``None``, the default,
        tables every occupied cell; ``0`` is ``with_table=False``.
    approx : bool
        Whether the streamed count step takes ``lgamma`` from the Stirling
        series at ``stirling_from`` and above --- one logarithm and one
        division, where the Lanczos sum below takes fourteen divisions --- and
        is read only there. ``False`` by default: the Lanczos sum throughout,
        within 1e-13 of the exact log-factorials (issue #999).
    stirling_from : float
        Where the Stirling series takes over under ``approx``. At the default
        10 it is within 4e-15 relative of the Lanczos sum; its first omitted
        term is ``3617 / (122400 x^13)``, 2.4e-11 absolute at 5.

    Returns
    -------
    EmFit
        The fitted parameters, the final log-likelihood, and whether any M
        step reported a parameter at the edge of what the data identifies.

    Raises
    ------
    ValueError
        If the family refuses its own re-estimate. A Gaussian family does so
        when a state's variance reaches its floor, which is an approach to a
        degenerate optimum rather than a convergence, and is reported as such
        rather than clamped away.
    """
    refuse_backend("the Baum-Welch E step", backend, (Backend.PYTHON, Backend.RUST))
    # The streamed step re-estimates every block and updates no covariate;
    # a fit that holds the transition or updates the exposure takes the
    # general route.
    if (
        backend is Backend.RUST
        and update is None
        and fit_transition
        and _streams(observations, log_transition, emissions, covariate)
    ):
        assert isinstance(observations, np.ndarray)
        assert covariate is None or isinstance(covariate, np.ndarray)
        return _streamed_family(
            observations,
            log_initial,
            log_transition,
            emissions,
            config=config,
            covariate=covariate,
            with_table=with_table,
            table_size=table_size,
            approx=approx,
            stirling_from=stirling_from,
        )
    # One batch form, and a rectangular argument converts to it (issue #666).
    # The segments are padded to the longest and masked; the mask is not a
    # convenience but the whole of the claim that padding never reaches a
    # likelihood, so it is applied in the E step *and*, through `gamma`, in the
    # emission M step. The conserved rectangular route in `sandbox` referees
    # the equal-length case bit for bit.
    batch = (
        observations
        if isinstance(observations, Ragged)
        else Ragged.from_rectangular(np.asarray(observations))
    )
    # Zero is a value every family admits and every padded position is masked
    # out of the arithmetic below, so the fill is arbitrary and never scored.
    block, present = batch.padded(fill=0)
    # The leading two axes are the sequence and the position. What follows them
    # is the family's own: none where an observation is a scalar, and one or
    # more where it is not --- a family over a pair of counts carries a channel
    # axis. Unpacking the whole shape refused every such family outright, so a
    # family could be made to condition on a covariate and still not be
    # fittable here (issue #658).
    data = torch.as_tensor(block, dtype=emissions.observation_dtype)
    mask = torch.as_tensor(present)
    n_sequences, length = batch.n_segments, batch.longest
    #: The last live position of each segment, which is where its chain ends.
    final = torch.as_tensor(batch.lengths, dtype=torch.long) - 1
    rows = torch.arange(n_sequences)
    steps = torch.arange(length)
    m = emissions.n_states
    varying = log_transition.shape != (m, m)
    per_sequence = log_transition.shape == (n_sequences, max(length - 1, 0), m, m)
    if (
        varying
        and not per_sequence
        and log_transition.shape
        != (
            max(length - 1, 0),
            m,
            m,
        )
    ):
        msg = (
            f"log_transition {tuple(log_transition.shape)} is neither ({m}, {m}), "
            f"({max(length - 1, 0)}, {m}, {m}) nor ({n_sequences}, "
            f"{max(length - 1, 0)}, {m}, {m}) for {n_sequences} chains of up to "
            f"{length} positions over {m} states"
        )
        raise ValueError(msg)
    kernels = (
        log_transition if varying else log_transition.expand(max(length - 1, 0), m, m)
    )
    # The trailing singleton the single-channel families broadcast over their
    # states with is added only where the covariate has no axes of its own; a
    # covariate carrying the family's axes is passed through, because the
    # singleton then belongs inside each of them and the family is what puts it
    # there (issue #658).
    exposure: torch.Tensor | None = None
    if covariate is not None:
        # The covariate is segmented exactly as the observations are, and is
        # padded with one rather than zero: it multiplies a rate or replaces a
        # trial count, and zero would be a division where the mask has not yet
        # removed the row (issue #658's neutral value, issue #666's padding).
        carried = (
            covariate
            if isinstance(covariate, Ragged)
            else Ragged.from_rectangular(np.asarray(covariate))
        )
        given, _ = carried.padded(fill=1)
        exposure = torch.as_tensor(given, dtype=torch.float64)
        if exposure.ndim == data.ndim == 2:
            exposure = exposure[..., None]
    if update is not None and exposure is None:
        msg = "a covariate update needs a covariate to update"
        raise ValueError(msg)
    compiled = backend is Backend.RUST and not varying
    lengths = np.asarray(batch.lengths, dtype=np.int64)
    # The posterior an update reads before the first E step: uniform over the
    # states, and zero at padded positions as every later one is.
    previous = mask.unsqueeze(2).expand(-1, -1, m).to(torch.float64) / m

    at_boundary = False

    def iterate(
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, EmissionFamily],
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, EmissionFamily], float]:
        """One E step, one M step, and the log-likelihood at the state given.

        The state is the initial distribution, the transition matrix, the
        per-step kernels it is expanded to and the emission family --- every
        parameter the recursion below reads and the M step rewrites.
        """
        nonlocal at_boundary, previous
        log_initial, log_transition, kernels, emissions = state
        scored = (
            exposure
            if update is None or exposure is None
            else update(emissions, previous, exposure)
        )
        # --- E step: forward and backward messages in log space ----------
        emit = emissions.log_density(data, covariate=scored)
        # A padded position scores log 1, so it adds nothing wherever it is
        # reached. Its `alpha` beyond the segment's end is still nonsense, which
        # is why the evidence is gathered at each segment's own last position
        # rather than read off the block's last column.
        emit = torch.where(mask.unsqueeze(2), emit, torch.zeros_like(emit))
        transition_counts: torch.Tensor | None
        if compiled:
            # The ragged kernel walks each sequence in place and returns the
            # log marginals, the log transition counts summed over sequences,
            # and each sequence's evidence (issue #933). It takes one kernel.
            gamma, transition_counts, log_likelihood = _ragged_e_step(
                emit, mask, lengths, log_initial, log_transition
            )
        else:
            alpha = torch.empty((n_sequences, length, m), dtype=log_initial.dtype)
            alpha[:, 0] = log_initial.unsqueeze(0) + emit[:, 0]
            for t in range(1, length):
                # One kernel for every sequence, or each sequence's own (#933).
                kernel = (
                    kernels[:, t - 1]
                    if per_sequence
                    else (kernels[t - 1] if varying else log_transition).unsqueeze(0)
                )
                alpha[:, t] = (
                    torch.logsumexp(alpha[:, t - 1].unsqueeze(2) + kernel, dim=1)
                    + emit[:, t]
                )
            beta = torch.zeros((n_sequences, length, m), dtype=log_initial.dtype)
            for t in range(length - 2, -1, -1):
                kernel = (
                    kernels[:, t]
                    if per_sequence
                    else (kernels[t] if varying else log_transition).unsqueeze(0)
                )
                onward = torch.logsumexp(
                    kernel + (emit[:, t + 1] + beta[:, t + 1]).unsqueeze(1),
                    dim=2,
                )
                # At or past a segment's last position the chain has ended: beta is
                # one, not whatever the next column carries. This is the backward
                # half of "the recursions restart at each boundary".
                ended = (t >= final).unsqueeze(1)
                beta[:, t] = torch.where(ended, torch.zeros_like(onward), onward)

            evidence = torch.logsumexp(alpha[rows, final], dim=1)
            log_likelihood = float(evidence.sum())

            gamma = torch.where(
                mask.unsqueeze(2),
                alpha + beta - evidence[:, None, None],
                torch.full_like(alpha, -float("inf")),
            )
            # A pair spans positions t and t+1, so it exists only where t is before
            # the segment's last position. The pair that would straddle a boundary
            # is not a transition the model took and is not counted as one.
            pairs = (steps[:-1] < final.unsqueeze(1))[:, :, None, None]
            xi = torch.where(
                pairs,
                alpha[:, :-1].unsqueeze(3)
                + (kernels if per_sequence else kernels.unsqueeze(0))
                + (emit[:, 1:] + beta[:, 1:]).unsqueeze(2)
                - evidence[:, None, None, None],
                torch.full(
                    (n_sequences, length - 1, m, m), -float("inf"), dtype=alpha.dtype
                ),
            )

            transition_counts = (
                None if varying else torch.logsumexp(xi.reshape(-1, m, m), dim=0)
            )

        # --- M step: normalized expected counts, then the family's own ---
        log_initial = torch.logsumexp(gamma[:, 0], dim=0) - torch.log(
            torch.tensor(float(n_sequences), dtype=gamma.dtype)
        )
        if fit_transition and transition_counts is not None:
            log_transition = transition_counts - torch.logsumexp(
                transition_counts, dim=1, keepdim=True
            )
            kernels = log_transition.expand(max(length - 1, 0), m, m)
        previous = torch.exp(gamma)
        step = emissions.reestimate(data, previous, covariate=scored)
        if not step.converged:
            msg = (
                f"the emission M step did not settle after {step.iterations} "
                f"iterations, at a relative change of {step.residual:.3e}: a "
                f"parameter read off iterations that never converged is not an "
                f"estimate, and a monotone outer likelihood would not have "
                f"shown it"
            )
            raise ValueError(msg)
        at_boundary = at_boundary or step.at_boundary
        return (log_initial, log_transition, kernels, step.emissions), log_likelihood

    (log_initial, log_transition, _, emissions), log_likelihood, termination = em_loop(
        iterate,
        (log_initial, log_transition, kernels, emissions),
        config=config,
    )
    return EmFit(
        log_initial=log_initial,
        log_transition=log_transition,
        emissions=emissions,
        log_likelihood=log_likelihood,
        emission_at_boundary=at_boundary,
        termination=termination,
    )
