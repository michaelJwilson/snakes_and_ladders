"""scikit-learn's EM on a one-dimensional Gaussian mixture the adapter wrote (issue #975).

Inputs: ``observations``, ``(n_samples,)``; ``weights``, ``mean``, ``scale``,
the start; ``n_iter``. The fit runs exactly ``n_iter`` iterations from that
start with no covariance regularization (``reg_covar=0``, ``tol=0``).
Outputs: the fitted ``weights``, ``mean``, ``scale`` and ``iterations``. The
measured seconds are ``fit`` alone.

With ``call`` set to ``score`` (issue #997) the start is the fitted model
and nothing is fitted: the output is ``log_likelihood``, the summed
``score_samples``, and the measured seconds and peak are that call alone.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from sal.validation.protocol import dump, measured, received


def main() -> None:
    """Fit from the given start under the timer, and write the parameters back."""
    from sklearn.exceptions import ConvergenceWarning  # the framework, only here
    from sklearn.mixture import GaussianMixture

    inputs, returned = received()
    scale = inputs["scale"]
    if str(inputs.get("call", np.asarray("fit"))) == "score":
        _score(inputs, returned)
        return
    model = GaussianMixture(
        n_components=scale.size,
        covariance_type="diag",
        reg_covar=0.0,
        tol=0.0,
        max_iter=int(inputs["n_iter"]),
        weights_init=inputs["weights"],
        means_init=inputs["mean"][:, None],
        precisions_init=1.0 / scale[:, None] ** 2,
    )
    column = inputs["observations"][:, None]
    with warnings.catch_warnings():
        # A fixed iteration count is asked for; "not converged" is expected.
        warnings.simplefilter("ignore", ConvergenceWarning)
        _, seconds, peak_bytes = measured(lambda: model.fit(column))
    dump(
        returned,
        {
            "weights": np.asarray(model.weights_, dtype=np.float64),
            "mean": np.asarray(model.means_, dtype=np.float64).ravel(),
            "scale": np.sqrt(np.asarray(model.covariances_, dtype=np.float64)).ravel(),
            "iterations": np.asarray(model.n_iter_, dtype=np.int64),
        },
        seconds,
        peak_bytes,
    )


def _score(inputs: dict[str, np.ndarray], returned: Path) -> None:
    """The summed log-likelihood at the given parameters, ``score_samples`` timed."""
    from sklearn.mixture import GaussianMixture  # the framework, only here

    scale = inputs["scale"]
    model = GaussianMixture(n_components=scale.size, covariance_type="diag")
    model.weights_ = inputs["weights"]
    model.means_ = inputs["mean"][:, None]
    model.covariances_ = scale[:, None] ** 2
    model.precisions_cholesky_ = 1.0 / scale[:, None]
    column = inputs["observations"][:, None]
    scores, seconds, peak_bytes = measured(lambda: model.score_samples(column))
    dump(
        returned,
        {"log_likelihood": np.asarray(float(np.sum(scores)))},
        seconds,
        peak_bytes,
    )


if __name__ == "__main__":
    main()
