"""`oxisal.ragged_posteriors` over segments in parallel, the same at every thread count (issue #1191).

The kernel cuts the segments into blocks by their lengths alone, runs the
blocks over `rayon`, and sums the blocks' transition counts in block order.
Referees: the three outputs are equal, bitwise, on pools of 1, 2, 4 and 8
threads and on the global pool, under every `SwitchKind`, on a layout of
many blocks (`smoke`, an invariant the kernel chose); they stay within
`CROSS_DEVICE_RTOL_FLOAT64` of the NumPy oracle, which sums every pair into
one accumulator (`oracle`); and a Baum--Welch fit in a process whose global
pool is one thread is the fit on eight, bitwise (`smoke`).
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest
from sal import oxisal
from sal.likelihood.device import CROSS_DEVICE_RTOL_FLOAT64
from sal.likelihood.ragged import Posteriors, SwitchKind, posteriors_oracle
from sal.ragged import Ragged

#: Thread counts the issue names.
THREADS = (1, 2, 4, 8)

#: 300 segments of 2 to 91 positions: about 14,000 positions, cut into 60 blocks.
LENGTHS = tuple(2 + (one * 53) % 90 for one in range(300))

#: Slow states; the stay-or-move kind runs at twice this, as the Kronecker kinds do.
SLOW = 3


def _instance(
    kind: SwitchKind, lengths: tuple[int, ...] = LENGTHS
) -> tuple[Ragged, np.ndarray, np.ndarray, np.ndarray]:
    """Scores over ``2 K`` states, a prior, a transition of the kind's width, a switch."""
    rng = np.random.default_rng(1191)
    total, n = sum(lengths), 2 * SLOW
    width = n if kind is SwitchKind.STAY_OR_MOVE else SLOW
    return (
        Ragged(np.log(rng.random((total, n))), lengths),
        np.log(rng.dirichlet(np.ones(n))),
        np.log(rng.dirichlet(np.ones(width), size=width)),
        rng.uniform(size=total),
    )


def _run(
    kind: SwitchKind, threads: int | None, lengths: tuple[int, ...] = LENGTHS
) -> Posteriors:
    """The kernel's three outputs on a pool of ``threads``, or on the global pool."""
    density, initial, transition, switch = _instance(kind, lengths)
    values = np.ascontiguousarray(density.values)
    n = values.shape[1]
    gamma, counts = np.empty_like(values), np.empty((n, n))
    evidence = np.empty(density.n_segments)
    oxisal.ragged_posteriors(
        values,
        np.asarray(lengths, dtype=np.int64),
        initial,
        transition,
        gamma,
        counts,
        evidence,
        switch,
        kind.value,
        threads=threads,
    )
    return Posteriors(gamma, counts, evidence)


@pytest.mark.critical
@pytest.mark.smoke
@pytest.mark.backend
@pytest.mark.parametrize("kind", list(SwitchKind), ids=str)
def test_every_thread_count_returns_the_same_bits(kind: SwitchKind) -> None:
    """Posteriors, counts and evidence at 1, 2, 4, 8 threads and the global pool."""
    serial = _run(kind, 1)
    for threads in (*THREADS[1:], None):
        for got, want in zip(_run(kind, threads), serial, strict=True):
            np.testing.assert_array_equal(got, want, err_msg=f"threads={threads}")


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.parametrize("kind", list(SwitchKind), ids=str)
def test_the_blocked_counts_stay_within_the_declared_tolerance(
    kind: SwitchKind,
) -> None:
    """Against the oracle's one accumulator: 60 segments, so many blocks.

    Measured: counts within 3e-14 relative of the kernel before #1191 on 700
    segments; gamma and evidence equal bitwise (the PR records both).
    """
    lengths = LENGTHS[:60]
    density, initial, transition, switch = _instance(kind, lengths)
    want = posteriors_oracle(density, initial, transition, switch, kind)
    for got, expected in zip(_run(kind, 4, lengths), want, strict=True):
        np.testing.assert_allclose(got, expected, rtol=CROSS_DEVICE_RTOL_FLOAT64)


_FIT = """
from dataclasses import replace

import numpy as np
import torch
from sal.emissions import GaussianEmission
from sal.opt.em import EM
from sal.opt.hmm import baum_welch_family
from sal.ragged import Ragged

rng = np.random.default_rng(1191)
lengths = tuple(2 + (one * 53) % 90 for one in range(300))
observations = Ragged(rng.normal(size=sum(lengths)), lengths)
log_initial = torch.log(torch.full((3,), 1.0 / 3.0, dtype=torch.float64))
log_transition = torch.log(torch.as_tensor(rng.dirichlet(np.full(3, 5.0), size=3)))
family = GaussianEmission(np.array([-1.0, 0.0, 1.0]), np.ones(3), variance_floor=1e-6)
fit = baum_welch_family(
    observations, log_initial, log_transition, family, replace(EM, max_iterations=20)
)
values = [fit.log_likelihood, *fit.log_initial.numpy().ravel(),
          *fit.log_transition.numpy().ravel(), *np.asarray(fit.components.mean),
          *np.asarray(fit.components.scale)]
print(" ".join(float(one).hex() for one in values))
"""


def _fit_on(threads: int) -> str:
    """The fit's fields, in hex, from a process whose global pool has ``threads``."""
    environment = {**os.environ, "RAYON_NUM_THREADS": str(threads)}
    return subprocess.run(
        [sys.executable, "-c", _FIT],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@pytest.mark.smoke
@pytest.mark.backend
def test_a_baum_welch_fit_is_the_same_on_one_thread_and_on_eight() -> None:
    """300 sequences, 20 iterations: every field of the fit, bitwise."""
    assert _fit_on(1) == _fit_on(8)
