"""hmmlearn's Baum--Welch, Viterbi and forward log-likelihood on an HMM the adapter wrote (issues #975, #997, #1282).

Inputs, as :func:`sal.external.hmm_inputs.hmm_inputs` writes them:
``observations``, ``(n_sequences, length)``, or the segments end to end with
``lengths``; ``initial`` and ``transition``, the start as probabilities;
``family``, a code into :data:`~sal.external.hmm_inputs.FAMILIES`
(categorical when absent), with its start: ``emission`` as probabilities,
``mean`` and ``variance`` per state, or ``rate`` per state. ``call`` is a code
into :data:`~sal.external.hmm_inputs.CALLS`: ``fit`` (the default),
``decode`` or ``score``, the summed log-likelihood at the given parameters.

``fit`` runs ``n_iter`` iterations in log space from that start
(``init_params=""``), with every prior and floor hmmlearn applies set to
zero, so the update is the maximum-likelihood one the package runs. Without
``tolerance`` it runs all ``n_iter`` (``tol=-inf``); with it, it stops where
the log-likelihood changes by at most ``tolerance`` relative to its
magnitude, :func:`sal.opt.em.em_loop`'s test. Outputs: the fitted
``initial``, ``transition``, the family's parameters, ``iterations``, the
last E step's ``log_likelihood`` and whether the relative test stopped it,
``settled``. ``decode`` runs Viterbi at the given parameters; outputs
``states`` and ``log_probability``. The measured seconds and peak resident
memory are the one call alone.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sal.external.hmm_inputs import CALLS, FAMILIES
from sal.external.protocol import dump, measured, received


def _settled(history: Any, tolerance: float) -> bool:
    """Whether the last two log-likelihoods differ by at most ``tolerance`` relative to the last."""
    return len(history) >= 2 and abs(history[-1] - history[-2]) <= tolerance * abs(
        history[-1]
    )


def _relative(tolerance: float, n_iter: int) -> Any:
    """hmmlearn's monitor, stopping on :func:`sal.opt.em.em_loop`'s relative test."""
    from hmmlearn.base import ConvergenceMonitor  # only here

    class Relative(ConvergenceMonitor):  # type: ignore[misc]
        """Converged at the budget, or where the relative change is within ``tol``."""

        @property
        def converged(self) -> bool:
            return bool(self.iter == self.n_iter or _settled(self.history, self.tol))

    return Relative(tolerance, n_iter, verbose=False)


def _model(inputs: dict[str, np.ndarray], family: str) -> Any:
    """The hmmlearn model for ``family``, at the given start, fitting every parameter."""
    from hmmlearn.hmm import CategoricalHMM, GaussianHMM, PoissonHMM  # only here

    shared = {
        "n_components": int(inputs["initial"].shape[0]),
        "implementation": "log",
        "init_params": "",
        "n_iter": int(inputs.get("n_iter", np.asarray(1))),
        "tol": -np.inf,
    }
    model: Any
    if family == "categorical":
        emission = inputs["emission"]
        model = CategoricalHMM(n_features=emission.shape[1], params="ste", **shared)
        model.emissionprob_ = emission
    elif family == "gaussian":
        # No prior on the means or the variances and no variance floor: the
        # maximum-likelihood update `GaussianEmission.reestimate` makes.
        model = GaussianHMM(
            covariance_type="diag",
            min_covar=0.0,
            covars_prior=0.0,
            covars_weight=1.0,
            means_prior=0.0,
            means_weight=0.0,
            params="stmc",
            **shared,
        )
        model.means_ = inputs["mean"].reshape(-1, 1)
        model.covars_ = inputs["variance"].reshape(-1, 1)
    elif family == "poisson":
        model = PoissonHMM(
            lambdas_prior=0.0, lambdas_weight=0.0, params="stl", **shared
        )
        model.lambdas_ = inputs["rate"].reshape(-1, 1)
    else:
        message = f"family {family!r} is none of categorical, gaussian, poisson"
        raise ValueError(message)
    model.startprob_ = inputs["initial"]
    model.transmat_ = inputs["transition"]
    if "tolerance" in inputs:
        model.monitor_ = _relative(float(inputs["tolerance"]), int(model.n_iter))
    return model


def _parameters(model: Any, family: str) -> dict[str, np.ndarray]:
    """The family's fitted parameters, as the inputs name them."""
    if family == "categorical":
        return {"emission": np.asarray(model.emissionprob_, dtype=np.float64)}
    if family == "gaussian":
        return {
            "mean": np.asarray(model.means_, dtype=np.float64).reshape(-1),
            "variance": np.asarray(model.covars_, dtype=np.float64).reshape(-1),
        }
    return {"rate": np.asarray(model.lambdas_, dtype=np.float64).reshape(-1)}


def main() -> None:
    """Fit or decode under the timer, and write the result back."""
    inputs, returned = received()
    family = FAMILIES[int(inputs.get("family", np.asarray(0)))]
    call = CALLS[int(inputs.get("call", np.asarray(0)))]
    observations = inputs["observations"]
    if "lengths" in inputs:
        lengths = [int(length) for length in inputs["lengths"]]
    else:
        n_sequences, length = observations.shape
        lengths = [length] * n_sequences
    model = _model(inputs, family)
    column = observations.reshape(-1, 1)
    if call == "score":
        log_likelihood, seconds, peak_bytes = measured(
            lambda: model.score(column, lengths)
        )
        outputs = {"log_likelihood": np.asarray(log_likelihood, dtype=np.float64)}
    elif call == "decode":
        (log_probability, states), seconds, peak_bytes = measured(
            lambda: model.decode(column, lengths, algorithm="viterbi")
        )
        outputs = {
            "states": np.asarray(states, dtype=np.int64).reshape(observations.shape),
            "log_probability": np.asarray(log_probability, dtype=np.float64),
        }
    else:
        _, seconds, peak_bytes = measured(lambda: model.fit(column, lengths=lengths))
        outputs = {
            "initial": np.asarray(model.startprob_, dtype=np.float64),
            "transition": np.asarray(model.transmat_, dtype=np.float64),
            **_parameters(model, family),
            "iterations": np.asarray(model.monitor_.iter, dtype=np.int64),
            "log_likelihood": np.asarray(
                model.monitor_.history[-1] if model.monitor_.history else -np.inf,
                dtype=np.float64,
            ),
            "settled": np.asarray(
                "tolerance" in inputs
                and _settled(model.monitor_.history, float(inputs["tolerance"]))
            ),
        }
    dump(returned, outputs, seconds, peak_bytes)


if __name__ == "__main__":
    main()
