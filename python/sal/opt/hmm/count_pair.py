"""The unphased count-pair HMM as one compiled object: a Baum-Welch fit in one FFI crossing (issue #1412).

:class:`CountPairHmm` holds the data, the emission, the chain and the
buffers in ``sal.oxisal.CountPairHmm`` (``src/count_pair_hmm.rs``), so
:meth:`~CountPairHmm.fit` runs every E step, M step and stopping test in
Rust and returns once. The model is
:class:`~sal.emissions.CountPairEmission`'s independent form under a
per-position exposure and trial count, ``K <= 255`` states.

**What it adds to** :func:`~sal.opt.hmm.estimation.baum_welch_family` **on
this model:** the initial distribution held (``fit_initial=False``); one
dispersion and one concentration shared by every state (``tied=True``),
solved in Rust where ``sal.qa.hmm_fit_semantics.tied_m_step`` solves it in
Python; a parameter-change stopping rule beside the relative log-likelihood
one; and a joint L-BFGS emission M step (:attr:`MStepSolver.LBFGS`) beside
the per-block one. ``baum_welch_family`` and ``tied_m_step`` stay as its
oracles: ``tests/regression/opt/test_count_pair_hmm.py`` pins each path.

**The two solvers' optima differ under a varying exposure.** The per-block
step takes each mean in closed form, ``sum w y / sum w e``, which is the
mean's maximum only where the exposure is constant; L-BFGS maximizes the
expected log-likelihood jointly, so its M step is never below the per-block
one's.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np
import torch

from sal import oxisal
from sal.emissions import CountPairEmission
from sal.opt.em import EM, EmConfig, Unsettled
from sal.opt.hmm.estimation import EmFit
from sal.opt.hmm.forward import Posteriors
from sal.opt.termination import Stop, Termination
from sal.ragged import Ragged

__all__ = ["CountPairHmm", "MStepSolver"]


class MStepSolver(StrEnum):
    """How :class:`CountPairHmm` re-estimates its emission."""

    NEWTON = "newton"
    """Each mean in closed form, each dispersion and ``(a, b)`` by a bracketed solve: the family's own ``reestimate``."""

    LBFGS = "lbfgs"
    """One L-BFGS over ``(log mu, logit p, log r, log tau)`` on the closed-form gradient: the joint maximum."""


class CountPairHmm:
    """The unphased count-pair HMM, its fit, posteriors, M step, likelihood and decode.

    Parameters
    ----------
    observations : Ragged
        ``(total, 2)`` non-negative integers: the count, then the successes.
    log_initial : np.ndarray
        ``(K,)`` starting log initial distribution.
    log_transition : np.ndarray
        ``(K, K)`` starting log transition.
    components : CountPairEmission
        The starting emission, independent form (``joint=False``).
    covariate : Ragged
        ``(total, 2)``: the exposure, then the trial count, segmented as
        ``observations``. Required: the trial count is per position here, and
        ``components.trials`` is not read. A zero exposure marks the count
        unobserved, a zero trial count the successes.
    tied : bool
        One dispersion and one concentration across states. The start must
        hold one of each.
    fit_initial, fit_transition : bool
        Whether the M step re-estimates the initial distribution and the
        transition; ``baum_welch_family`` always fits the first.

    Raises
    ------
    ValueError
        A joint-form family, shapes that disagree, more than 255 states, or a
        tied start whose dispersions or concentrations differ.
    """

    def __init__(
        self,
        observations: Ragged,
        log_initial: np.ndarray,
        log_transition: np.ndarray,
        components: CountPairEmission,
        *,
        covariate: Ragged,
        tied: bool = False,
        fit_initial: bool = True,
        fit_transition: bool = True,
    ) -> None:
        if components.joint:
            msg = "CountPairHmm fits the independent form; got joint=True"
            raise ValueError(msg)
        values = np.asarray(observations.values)
        given = np.asarray(covariate.values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 2 or given.shape != values.shape:
            msg = f"observations and covariate are (total, 2), got {values.shape} and {given.shape}"
            raise ValueError(msg)
        if tuple(observations.lengths) != tuple(covariate.lengths):
            msg = "the covariate is segmented as the observations are"
            raise ValueError(msg)
        trials = given[:, 1]
        if bool((trials != np.round(trials)).any()) or bool((trials < 0).any()):
            msg = "trial counts are non-negative integers"
            raise ValueError(msg)
        self._lengths = tuple(int(n) for n in observations.lengths)
        self._model = oxisal.CountPairHmm(
            np.ascontiguousarray(values[:, 0], dtype=np.uint32),
            np.ascontiguousarray(given[:, 0]),
            np.ascontiguousarray(values[:, 1], dtype=np.uint32),
            np.ascontiguousarray(trials, dtype=np.uint32),
            np.asarray(self._lengths, dtype=np.int64),
            np.ascontiguousarray(components.total.dispersion.numpy()),
            np.ascontiguousarray(components.total.mean.numpy()),
            np.ascontiguousarray(components.alpha.numpy()),
            np.ascontiguousarray(components.beta.numpy()),
            np.ascontiguousarray(log_initial, dtype=np.float64),
            np.ascontiguousarray(log_transition, dtype=np.float64),
            tied=tied,
            fit_initial=fit_initial,
            fit_transition=fit_transition,
        )
        self._trials = components.trials

    @property
    def n_states(self) -> int:
        """Hidden states."""
        return int(self._model.n_states)

    @property
    def components(self) -> CountPairEmission:
        """The held emission, as the family."""
        dispersion, mean, alpha, beta, _, _ = self._model.parameters()
        assert self._trials is not None
        return CountPairEmission(
            dispersion, mean, alpha, beta, self._trials, joint=False
        )

    @property
    def log_initial(self) -> np.ndarray:
        """``(K,)`` held log initial distribution."""
        return np.asarray(self._model.parameters()[4])

    @property
    def log_transition(self) -> np.ndarray:
        """``(K, K)`` held log transition."""
        return np.asarray(self._model.parameters()[5])

    def fit(
        self,
        config: EmConfig = EM,
        *,
        solver: MStepSolver = MStepSolver.NEWTON,
        parameter_tolerance: float = 0.0,
    ) -> EmFit:
        """Baum-Welch from the held parameters, in one crossing; the model keeps the result.

        ``em_loop``'s alternation: each iteration's log-likelihood is at the
        parameters it was handed, and it stops when that moves by at most
        ``config.tolerance`` relative to its magnitude, when a positive
        ``parameter_tolerance`` bounds every emission parameter's relative
        change, or after ``config.max_iterations``. An emission M step that
        does not settle ends it with :attr:`~sal.opt.termination.Stop.DEGENERATE`
        on the parameters before it.

        Returns
        -------
        EmFit
            ``spent`` in EM iterations and its ``Termination``, as
            ``baum_welch_family`` returns.
        """
        log_likelihood, iterations, stop, at_boundary, frozen, unsettled = (
            self._model.fit(
                config.max_iterations,
                config.tolerance,
                parameter_tolerance,
                str(solver),
            )
        )
        reason = Stop(stop)
        termination = Termination(
            converged=reason is Stop.CONVERGED, iterations=iterations, reason=reason
        )
        return EmFit(
            log_initial=torch.as_tensor(self.log_initial),
            log_transition=torch.as_tensor(self.log_transition),
            components=self.components,
            log_likelihood=float(log_likelihood),
            at_boundary=bool(at_boundary),
            termination=termination,
            spent=iterations,
            frozen=tuple(frozen),
            unsettled=None
            if unsettled is None
            else Unsettled(unsettled[0], unsettled[1], tuple(unsettled[2])),
        )

    def posteriors(self) -> Posteriors:
        """The E step at the held parameters, in the log domain."""
        log_posterior, log_counts, log_evidence = self._model.posteriors()
        return Posteriors(
            np.asarray(log_posterior), np.asarray(log_counts), np.asarray(log_evidence)
        )

    def m_step(
        self, posterior: np.ndarray, *, solver: MStepSolver = MStepSolver.NEWTON
    ) -> CountPairEmission:
        """Re-estimate the held emission on ``(total, K)`` posterior probabilities; the result.

        The initial distribution and transition are not touched: they read the
        transition counts, which :meth:`posteriors` returns.

        Raises
        ------
        ValueError
            If ``posterior`` is not ``(total, K)``, or the solve did not settle.
        """
        converged, _, iterations, residual, _, degenerate = self._model.m_step(
            np.ascontiguousarray(posterior, dtype=np.float64), str(solver)
        )
        if not converged:
            msg = (
                f"the emission M step did not settle after {iterations} iterations, "
                f"residual {residual:.3e}, degenerate states {degenerate}"
            )
            raise ValueError(msg)
        return self.components

    def log_likelihood(self) -> float:
        """The log-likelihood at the held parameters."""
        return float(self._model.log_likelihood())

    def viterbi(self) -> tuple[np.ndarray, np.ndarray]:
        """The most probable path, ``(total,)``, and each segment's joint log-probability."""
        path, log_joint = self._model.viterbi()
        return np.asarray(path), np.asarray(log_joint)
