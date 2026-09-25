"""The Rust BCJR pass against the NumPy oracle and against enumeration (issue #754).

Bitwise at `memory` 1 and 2 (four states, every declared register):
`src/bcjr.rs` follows `npy_logaddexp` and `numerics.logsumexp`'s shift. Inside
`CROSS_DEVICE_RTOL_FLOAT64` at `memory` 3 and 4, where NumPy's pairwise sum
switches to eight accumulators: 2.3e-13 absolute, 2.2e-12 relative over `K`
up to 1,024, the log evidence still bitwise. Both backends are pinned to
enumeration in `test_convolutional.py`; end to end, `bcjr` and `decode_turbo`
default to this kernel, judged there and in `test_turbo.py` (#982).
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from sal.backend import Backend
from sal.likelihood.convolutional import (
    bcjr,
)
from sal.likelihood.device import CROSS_DEVICE_RTOL_FLOAT64
from sal.likelihood.rust import convolutional as convolutional_rust
from sal.likelihood.turbo import (
    split_streams,
)
from sal.sim.convolutional import (
    Trellis,
    recursive_systematic_trellis,
    turbo_encode,
)
from sal.sim.fixtures import fixture
from sal.sim.ldpc import BinaryInputGaussianChannel

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

    The `BCJR` rung of `infra/ladder.py`; at four states nothing reassociates.
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

    Pairwise against left-to-right: 2.3e-13 absolute, 2.2e-12 relative, bound 1e-11.
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
    """The declared instance, both constituent streams, bitwise."""
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

    A non-permutation would leave `Trellis.source` reading uninitialized slots.
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
