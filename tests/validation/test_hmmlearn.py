"""`opt.hmm.baum_welch` against hmmlearn's, run in a subprocess (issue #975).

hmmlearn's `CategoricalHMM` is an independent implementation of the same
recursion. Referees:

- ten iterations from one start on `hmm/ci.yaml` (3 states, 4 symbols, 600
  sequences of 15): every probability within 1e-11, and one iteration alone
  within the same, so the agreement is per step and not a shared fixed point.

The runtime goal hmmlearn sets is in `test_goals.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from snakes_and_ladders.fixtures import load_params
from snakes_and_ladders.opt.hmm import baum_welch
from snakes_and_ladders.sim.hmm import HmmParams, simulate_sequences
from snakes_and_ladders.validation import hmmlearn
from snakes_and_ladders.validation.runner import available

from tests._fixtures import FIXTURES_DIR

pytestmark = [
    pytest.mark.validation,
    pytest.mark.skipif(
        not available("hmmlearn"), reason="hmmlearn is the validation-hmmlearn extra"
    ),
]

FIXTURE: Path = FIXTURES_DIR / "hmm" / "ci.yaml"

#: The float64 tolerance two implementations of one recursion are held to.
ATOL = 1e-11


def _start(params: HmmParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A start away from the truth: each row a perturbed uniform, seeded."""
    rng = np.random.default_rng(975)

    def simplex(*shape: int) -> np.ndarray:
        draw = rng.random(shape) + 0.5
        return np.asarray(draw / draw.sum(axis=-1, keepdims=True))

    return (
        simplex(params.n_states),
        simplex(params.n_states, params.n_states),
        simplex(params.n_states, params.n_symbols),
    )


def _ours(
    observations: np.ndarray, start: tuple[np.ndarray, ...], n_iter: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    initial, transition, emission = (torch.log(torch.as_tensor(p)) for p in start)
    fit = baum_welch(
        observations,
        initial,
        transition,
        emission,
        max_iterations=n_iter,
        tolerance=-np.inf,
    )
    return (
        fit.log_initial.exp().numpy(),
        fit.log_transition.exp().numpy(),
        fit.log_emission.exp().numpy(),
    )


@pytest.mark.oracle
@pytest.mark.parametrize("n_iter", [1, 10])
def test_baum_welch_is_hmmlearns_iteration_for_iteration(n_iter: int) -> None:
    params = load_params(FIXTURE, HmmParams)
    observations = simulate_sequences(params).observations
    start = _start(params)
    theirs = hmmlearn.baum_welch(observations, *start, n_iter)
    ours = _ours(observations, start, n_iter)
    assert theirs.iterations == n_iter
    for mine, other in zip(
        ours, (theirs.initial, theirs.transition, theirs.emission), strict=False
    ):
        np.testing.assert_allclose(mine, other, rtol=0.0, atol=ATOL)
    # The start is not a fixed point: ten iterations moved the emissions.
    assert np.abs(theirs.emission - start[2]).max() > 1e-3
