"""Regression tests for ``sal.likelihood.pruning_rust`` (the Rust CPU backend).

What is the compiled route's alone. The checks every route shares are one
body each, parametrised over ``ROUTES`` (issue #982): agreement with the NumPy
oracle in ``test_pruning_common.py``, with brute force at ``n <= 6`` taxa and
rescaled against unrescaled in ``test_likelihood_pruning.py``, and the
refusals in ``test_likelihood_validation.py``.

Requires the compiled extension (``maturin develop`` / ``pip install .``),
like ``tests/test_oxisal_bindings.py``.
"""

from __future__ import annotations

import pytest
from numpy.testing import assert_allclose
from sal.backend import Backend
from sal.likelihood import pruning, pruning_rust
from sal.likelihood.device import CROSS_DEVICE_RTOL_FLOAT64
from sal.likelihood.patterns import compress

from tests._fixtures import FOUR_TAXA, SMALL_SITES, load_fixture, simulated_alignment

# Relative (#111): ~8e-13 relative at every size is 7.4e-07 absolute at
# 200,000 sites. The bound is `sec:tolerance`'s, never relaxed.
_RTOL_ORACLE = CROSS_DEVICE_RTOL_FLOAT64


@pytest.mark.smoke
@pytest.mark.release
def test_relative_tolerance_transfers_to_fixture_scale() -> None:
    """The tolerance holds at 200,000 sites, where an absolute one would not.

    Issue #111 in executable form; release-gated, a full-fixture score (#109).
    """
    params, alignment = simulated_alignment(FOUR_TAXA)

    numpy_ll = pruning.log_likelihood(params.tau, params.k, params.pi, alignment)
    rust_ll = pruning_rust.log_likelihood(params.tau, params.k, params.pi, alignment)

    # The relative bound holds, unchanged from `test_pruning_common.py`'s
    # 20,000-site check.
    assert_allclose(rust_ll, numpy_ll, rtol=_RTOL_ORACLE)

    # And the absolute bound this file used to carry would have failed here,
    # on backends that are behaving correctly.
    absolute = abs(rust_ll - numpy_ll)
    assert absolute > 1e-9, (
        "expected the absolute deviation to exceed the old 1e-9 bound at "
        f"fixture scale, got {absolute:.3e}; if this no longer holds the "
        "motivating example in issue #111 needs revisiting"
    )


@pytest.mark.backend
@pytest.mark.oracle
def test_the_enum_reaches_the_rust_kernel_bitwise() -> None:
    # #860: `Backend.RUST` at the oracle's entry point is this module's call,
    # bit for bit, weighted route included; any other member is refused by name.
    for name in (SMALL_SITES, "tree_search/ci.yaml"):
        params, alignment = simulated_alignment(name)
        patterns = compress(alignment)

        assert pruning.log_likelihood(
            params.tau, params.k, params.pi, alignment, backend=Backend.RUST
        ) == pruning_rust.log_likelihood(params.tau, params.k, params.pi, alignment)
        assert pruning.log_likelihood(
            params.tau,
            params.k,
            params.pi,
            patterns.alignment,
            weights=patterns.weights,
            rescale=False,
            backend=Backend.RUST,
        ) == pruning_rust.log_likelihood(
            params.tau,
            params.k,
            params.pi,
            patterns.alignment,
            weights=patterns.weights,
            rescale=False,
        )

    params = load_fixture(SMALL_SITES)
    with pytest.raises(ValueError, match="not numba"):
        pruning.log_likelihood(
            params.tau, params.k, params.pi, {}, backend=Backend.NUMBA
        )
