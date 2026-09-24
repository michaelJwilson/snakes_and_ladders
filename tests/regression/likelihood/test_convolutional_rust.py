"""The Rust BCJR pass against the NumPy oracle and against enumeration (issue #754).

`likelihood/CLAUDE.md`: the reference implementation is the oracle and it
stays, and a backend is accepted or rejected against a stated bound rather
than adjusted until it matches. Two claims are separated here rather than
merged into the looser one.

* **Bitwise** on every register this repository declares. `src/bcjr.rs`
  takes NumPy's `npy_logaddexp` branch for branch and
  `numerics.logsumexp`'s shift by the row maximum, and at `memory` 1 and 2
  --- four states, the `(7, 5)` encoder of every fixture and both
  benchmark lengths --- the two agree to the last bit.
* **Inside `CROSS_DEVICE_RTOL_FLOAT64`** at `memory` 3 and 4, where NumPy's
  pairwise sum switches to eight accumulators and this kernel stays left to
  right. That is a reassociation of a floating sum: realized 2.3e-13
  absolute and 2.2e-12 relative over `K` up to 1,024, with the log evidence
  still bitwise.

Both backends are then pinned to `exact_bitwise_posterior`, which shares no
recursion with either, so the pair is not established by agreeing with each
other alone: `test_convolutional.py::test_bcjr_posteriors_are_the_exact_bitwise_map`
runs that enumeration on both backends. End to end, `bcjr` and
`decode_turbo` default to this kernel, so
`test_convolutional.py`'s planted-message test and `test_turbo.py`'s
recorded-rate test judge it against the planted truth (issue #982 dropped the
copies of all three here).
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from snakes_and_ladders.backend import Backend
from snakes_and_ladders.likelihood import convolutional_rust
from snakes_and_ladders.likelihood.convolutional import (
    bcjr,
)
from snakes_and_ladders.likelihood.device import CROSS_DEVICE_RTOL_FLOAT64
from snakes_and_ladders.likelihood.turbo import (
    split_streams,
)
from snakes_and_ladders.sim.convolutional import (
    Trellis,
    recursive_systematic_trellis,
    turbo_encode,
)
from snakes_and_ladders.sim.fixtures import fixture
from snakes_and_ladders.sim.ldpc import BinaryInputGaussianChannel

from tests._rows import every_row

#: The declared register and the three above it, as octal `(feedback,
#: feedforward)` pairs. `memory` 2 is the `(7, 5)` encoder every fixture
#: uses; 1, 3 and 4 are here because a backend measured at one state count
#: is measured at one state count.
REGISTERS = {1: (0o3, 0o2), 2: (0o7, 0o5), 3: (0o13, 0o15), 4: (0o23, 0o35)}

#: Where the two backends agree to the last bit: four states or fewer, which
#: is where NumPy's pairwise sum over the states is a left-to-right loop.
BITWISE_MEMORY = (1, 2)


def _ratios(length: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three log-likelihood ratio streams of one length, under one seed."""
    rng = np.random.default_rng(seed)
    return (
        1.7 * rng.standard_normal(length),
        1.7 * rng.standard_normal(length),
        0.5 * rng.standard_normal(length),
    )


# --- against the NumPy oracle ----------------------------------------------------


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.backend
def test_the_rust_pass_is_the_numpy_oracle_bitwise_on_the_declared_registers() -> None:
    """Every output, to the last bit, at the state count the fixtures use.

    The rung `infra/ladder.py` records under `BCJR`: the kernel is not
    accepted against a tolerance where it does not need one. Equality is the
    strongest thing a referee can assert (root `CLAUDE.md`), and at four
    states the two implementations reassociate nothing relative to each
    other, so it is what is asserted.
    """

    def check(memory: int, message_length: int, terminated: bool) -> None:
        feedback, feedforward = REGISTERS[memory]
        trellis = recursive_systematic_trellis(feedback, feedforward, memory)
        systematic, parity, apriori = _ratios(message_length + memory, memory + 17)

        oracle = bcjr(
            trellis,
            systematic,
            parity,
            apriori,
            terminated=terminated,
            backend=Backend.PYTHON,
        )
        ported = convolutional_rust.bcjr(
            trellis, systematic, parity, apriori, terminated=terminated
        )

        np.testing.assert_array_equal(ported.posterior_llr, oracle.posterior_llr)
        np.testing.assert_array_equal(ported.extrinsic_llr, oracle.extrinsic_llr)
        assert ported.log_evidence == oracle.log_evidence

    every_row(product(BITWISE_MEMORY, [8, 256, 1024], [True, False]), check)


@pytest.mark.oracle
@pytest.mark.backend
def test_past_four_states_the_two_part_only_by_numpys_pairwise_sum() -> None:
    """Inside the float64 cross-implementation bound, and the evidence is exact.

    NumPy's pairwise reduction switches to eight accumulators at eight terms
    and the kernel's state loop stays left to right, so the two reassociate
    the sum over states differently. Realized over these six cells:
    **2.3e-13** absolute and **2.2e-12** relative in the ratios, against a
    bound of 1e-11 relative. The log evidence reduces once per call and
    stays bitwise, which is what says the departure is the reduction and not
    the recursions feeding it.
    """

    def check(memory: int, message_length: int) -> None:
        feedback, feedforward = REGISTERS[memory]
        trellis = recursive_systematic_trellis(feedback, feedforward, memory)
        systematic, parity, apriori = _ratios(message_length + memory, memory + 17)

        oracle = bcjr(trellis, systematic, parity, apriori, backend=Backend.PYTHON)
        ported = bcjr(trellis, systematic, parity, apriori, backend=Backend.RUST)

        np.testing.assert_allclose(
            ported.posterior_llr, oracle.posterior_llr, rtol=CROSS_DEVICE_RTOL_FLOAT64
        )
        np.testing.assert_allclose(
            ported.extrinsic_llr, oracle.extrinsic_llr, rtol=CROSS_DEVICE_RTOL_FLOAT64
        )
        assert ported.log_evidence == oracle.log_evidence

    every_row(product([3, 4], [64, 256, 1024]), check)


@pytest.mark.oracle
@pytest.mark.backend
def test_the_rust_pass_is_the_oracle_on_the_ci_turbo_fixture() -> None:
    """The declared instance, both constituent streams, bitwise.

    The fixture rather than a constructed trellis: a backend measured only
    on inputs its author chose is measured on its author.
    """
    code = fixture("turbo", "ci").params.code()
    rng = np.random.default_rng(233)
    message = rng.integers(0, 2, code.message_length).astype(np.uint8)
    word = turbo_encode(code, message)
    streams = split_streams(
        code, BinaryInputGaussianChannel(0.9).log_likelihood_ratios(word, rng)
    )

    for parity in (streams.parity_first, streams.parity_second):
        oracle = bcjr(code.trellis, streams.systematic, parity, backend=Backend.PYTHON)
        ported = bcjr(code.trellis, streams.systematic, parity, backend=Backend.RUST)

        np.testing.assert_array_equal(ported.posterior_llr, oracle.posterior_llr)
        np.testing.assert_array_equal(ported.extrinsic_llr, oracle.extrinsic_llr)
        assert ported.log_evidence == oracle.log_evidence


# --- refusals --------------------------------------------------------------------


@pytest.mark.smoke
def test_a_backend_the_pass_does_not_have_is_refused() -> None:
    """`Backend.NUMBA` names no BCJR implementation, so it raises rather than runs."""
    trellis = recursive_systematic_trellis(*REGISTERS[2], 2)
    systematic, parity, apriori = _ratios(10, 3)

    with pytest.raises(ValueError, match="bcjr runs on"):
        bcjr(trellis, systematic, parity, apriori, backend=Backend.NUMBA)


@pytest.mark.smoke
@pytest.mark.parametrize("backend", [Backend.PYTHON, Backend.RUST])
def test_both_backends_refuse_streams_of_disagreeing_length(backend: Backend) -> None:
    """The shapes are validated before the dispatch, so the refusal is the same one."""
    trellis = recursive_systematic_trellis(*REGISTERS[2], 2)

    with pytest.raises(ValueError, match="must be the same one-dimensional shape"):
        bcjr(trellis, np.zeros(6), np.zeros(5), backend=backend)


@pytest.mark.smoke
def test_a_next_state_table_that_is_not_a_permutation_is_refused() -> None:
    """The kernel derives the gather's inverse and checks it rather than trusting it.

    `Trellis.source` is built by a scatter into an uninitialized array, so a
    table that is not a permutation leaves a state reading whatever was in
    the slot. The kernel derives the inverse itself and refuses instead.
    """
    broken = Trellis(
        memory=1,
        next_state=np.array([[0, 1], [0, 1]], dtype=np.int64),
        parity=np.array([[0, 1], [1, 0]], dtype=np.uint8),
        tail_input=np.array([0, 0], dtype=np.uint8),
        source=np.array([[0, 0], [0, 0]], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="not a permutation"):
        bcjr(broken, np.zeros(4), np.zeros(4), backend=Backend.RUST)
