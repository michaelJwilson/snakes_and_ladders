"""hmmlearn's Baum--Welch on a categorical HMM the adapter wrote (issue #975).

Inputs: ``observations``, ``(n_sequences, length)`` symbols; ``initial``,
``transition``, ``emission``, the start as probabilities; ``n_iter``. The
fit runs exactly ``n_iter`` iterations in log space from that start
(``init_params=""``, ``tol=-inf``). Outputs: the fitted ``initial``,
``transition``, ``emission`` and ``iterations``. The measured seconds are
``fit`` alone.
"""

from __future__ import annotations

import numpy as np

from snakes_and_ladders.validation.protocol import dump, load, paths, timed


def main() -> None:
    """Fit from the given start under the timer, and write the parameters back."""
    from hmmlearn.hmm import CategoricalHMM  # the framework, only here

    given, returned = paths()
    inputs = load(given)
    observations = inputs["observations"]
    n_sequences, length = observations.shape
    emission = inputs["emission"]
    model = CategoricalHMM(
        n_components=emission.shape[0],
        n_features=emission.shape[1],
        implementation="log",
        init_params="",
        params="ste",
        n_iter=int(inputs["n_iter"]),
        tol=-np.inf,
    )
    model.startprob_ = inputs["initial"]
    model.transmat_ = inputs["transition"]
    model.emissionprob_ = emission
    column = observations.reshape(-1, 1)
    _, seconds = timed(lambda: model.fit(column, lengths=[length] * n_sequences))
    dump(
        returned,
        {
            "initial": np.asarray(model.startprob_, dtype=np.float64),
            "transition": np.asarray(model.transmat_, dtype=np.float64),
            "emission": np.asarray(model.emissionprob_, dtype=np.float64),
            "iterations": np.asarray(model.monitor_.iter, dtype=np.int64),
        },
        seconds,
    )


if __name__ == "__main__":
    main()
