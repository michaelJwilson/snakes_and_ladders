"""scikit-learn's EM on a one-dimensional Gaussian mixture the adapter wrote (issue #975).

Inputs: ``observations``, ``(n_samples,)``; ``weights``, ``mean``, ``scale``,
the start; ``n_iter``. The fit runs exactly ``n_iter`` iterations from that
start with no covariance regularization (``reg_covar=0``, ``tol=0``).
Outputs: the fitted ``weights``, ``mean``, ``scale`` and ``iterations``. The
measured seconds are ``fit`` alone.
"""

from __future__ import annotations

import warnings

import numpy as np

from snakes_and_ladders.validation.protocol import dump, load, paths, peaked, timed


def main() -> None:
    """Fit from the given start under the timer, and write the parameters back."""
    from sklearn.exceptions import ConvergenceWarning  # the framework, only here
    from sklearn.mixture import GaussianMixture

    given, returned = paths()
    inputs = load(given)
    scale = inputs["scale"]
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
        (_, seconds), peak_bytes = peaked(lambda: timed(lambda: model.fit(column)))
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


if __name__ == "__main__":
    main()
