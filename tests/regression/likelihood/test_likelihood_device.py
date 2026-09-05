"""Regression tests for device dispatch and the cross-device tolerance.

CI runners have neither CUDA nor Metal, so the parts that can be checked here
are the selection *policy* -- pure, with availability injected -- and the
float32/float64 agreement that sets the tolerance, which is the same
arithmetic Metal will do and runs fine on CPU. The device-specific numerical
checks are marked and skip when the hardware is absent.
"""

from __future__ import annotations

import pytest
import torch
from numpy.testing import assert_allclose
from phylo.likelihood import pruning, pruning_torch
from phylo.likelihood.device import (
    CROSS_DEVICE_RTOL_FLOAT32,
    CROSS_DEVICE_RTOL_FLOAT64,
    available_device,
    cross_device_rtol,
    default_dtype,
    select_device,
)
from phylo.sim.simulate import simulate_alignment

from tests._fixtures import FOUR_TAXA, SMALL_SITES, load_fixture

# --- selection policy: pure, so it is testable without the hardware ------


@pytest.mark.structural
@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [
        (True, True, "cuda"),
        (True, False, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_selection_prefers_cuda_then_mps_then_cpu(
    cuda: bool, mps: bool, expected: str
) -> None:
    assert select_device(cuda_available=cuda, mps_available=mps) == expected


@pytest.mark.structural
def test_selection_always_yields_a_device() -> None:
    # CPU is the last preference and always present, so selection cannot fail.
    assert available_device() in {"cuda", "mps", "cpu"}


# --- dtype policy: the constraint that shapes the whole ticket -----------


@pytest.mark.structural
def test_metal_gets_float32_because_it_cannot_do_float64() -> None:
    # PyTorch's Metal backend rejects float64 outright. This is not a
    # preference; it is why the float32 tolerance has to exist.
    assert default_dtype("mps") == torch.float32


@pytest.mark.structural
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_float64_is_kept_wherever_it_is_supported(device: str) -> None:
    assert default_dtype(device) == torch.float64


@pytest.mark.edge_case
def test_an_unknown_device_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown device"):
        default_dtype("tpu")


# --- tolerance policy ----------------------------------------------------


@pytest.mark.structural
def test_tolerance_is_keyed_on_the_lowest_precision_involved() -> None:
    # A comparison is only as accurate as its least accurate side. Holding a
    # float32 comparison to the float64 bound would fail correct code; the
    # reverse would let a broken float64 backend pass.
    assert cross_device_rtol(torch.float64, torch.float64) == CROSS_DEVICE_RTOL_FLOAT64
    assert cross_device_rtol(torch.float32, torch.float64) == CROSS_DEVICE_RTOL_FLOAT32
    assert cross_device_rtol(torch.float32) == CROSS_DEVICE_RTOL_FLOAT32


@pytest.mark.structural
def test_the_float32_tolerance_is_the_looser_one() -> None:
    assert CROSS_DEVICE_RTOL_FLOAT32 > CROSS_DEVICE_RTOL_FLOAT64


@pytest.mark.edge_case
def test_tolerance_needs_a_dtype() -> None:
    with pytest.raises(ValueError, match="at least one dtype"):
        cross_device_rtol()


@pytest.mark.edge_case
def test_an_unsupported_dtype_is_refused() -> None:
    with pytest.raises(ValueError, match="unsupported dtype"):
        cross_device_rtol(torch.float16)


# --- the tolerance against real arithmetic, on CPU -----------------------


@pytest.mark.mathematical
@pytest.mark.parametrize("fixture", [SMALL_SITES, FOUR_TAXA])
def test_float32_agrees_with_float64_inside_the_stated_tolerance(
    fixture: str,
) -> None:
    # The check that makes the float32 tolerance evidence rather than a
    # guess: float32 on CPU is the same arithmetic Metal will do, so this
    # runs on a GPU-less runner and still exercises the number.
    params = load_fixture(fixture)
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        seed=params.seed,
        n_sites=params.n_sites,
    )
    alignment = dict(dataset.alignment)

    wide = pruning_torch.log_likelihood(
        params.tau,
        params.k,
        params.pi,
        alignment,
        pruning_torch.branch_lengths_from_tree(params.tau),
    )
    narrow = pruning_torch.log_likelihood(
        params.tau,
        params.k,
        params.pi,
        alignment,
        pruning_torch.branch_lengths_from_tree(params.tau, dtype=torch.float32),
    )

    assert narrow.dtype == torch.float32
    assert_allclose(
        float(narrow),
        float(wide),
        rtol=cross_device_rtol(torch.float32, torch.float64),
    )


@pytest.mark.structural
def test_float32_would_fail_an_absolute_bound_that_float64_passes() -> None:
    # Why the tolerance is relative (issue #111). At fixture scale the
    # float32 discrepancy is ~1e-2 absolute, so any absolute bound tight
    # enough to be meaningful for float64 rejects correct float32 code.
    params = load_fixture(FOUR_TAXA)
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        seed=params.seed,
        n_sites=params.n_sites,
    )
    alignment = dict(dataset.alignment)

    wide = float(
        pruning_torch.log_likelihood(
            params.tau,
            params.k,
            params.pi,
            alignment,
            pruning_torch.branch_lengths_from_tree(params.tau),
        )
    )
    narrow = float(
        pruning_torch.log_likelihood(
            params.tau,
            params.k,
            params.pi,
            alignment,
            pruning_torch.branch_lengths_from_tree(params.tau, dtype=torch.float32),
        )
    )

    absolute = abs(wide - narrow)
    relative = absolute / abs(wide)

    assert absolute > 1e-3
    assert relative < CROSS_DEVICE_RTOL_FLOAT32


@pytest.mark.structural
def test_float64_default_is_unchanged_by_the_dtype_parameter() -> None:
    # No silent behaviour change: a caller who passes nothing still gets
    # float64, and still matches the NumPy oracle.
    params = load_fixture(SMALL_SITES)
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        seed=params.seed,
        n_sites=200,
    )
    alignment = dict(dataset.alignment)

    lengths = pruning_torch.branch_lengths_from_tree(params.tau)
    assert lengths.dtype == torch.float64

    torch_value = float(
        pruning_torch.log_likelihood(
            params.tau, params.k, params.pi, alignment, lengths
        )
    )
    numpy_value = pruning.log_likelihood(params.tau, params.k, params.pi, alignment)
    assert_allclose(torch_value, numpy_value, rtol=CROSS_DEVICE_RTOL_FLOAT64)


# --- device-specific, skipped where the hardware is absent ---------------


@pytest.mark.mathematical
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA device on this machine"
)
def test_cuda_agrees_with_cpu() -> None:  # pragma: no cover
    params = load_fixture(SMALL_SITES)
    dataset = simulate_alignment(
        tau=params.tau, k=params.k, pi=params.pi, seed=params.seed, n_sites=500
    )
    alignment = dict(dataset.alignment)

    dtype = default_dtype("cuda")
    on_cpu = pruning_torch.log_likelihood(
        params.tau,
        params.k,
        params.pi,
        alignment,
        pruning_torch.branch_lengths_from_tree(params.tau, dtype=dtype),
    )
    on_cuda = pruning_torch.log_likelihood(
        params.tau,
        params.k,
        params.pi,
        alignment,
        pruning_torch.branch_lengths_from_tree(params.tau, dtype=dtype, device="cuda"),
    )
    assert_allclose(float(on_cuda), float(on_cpu), rtol=cross_device_rtol(dtype, dtype))


@pytest.mark.mathematical
@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="no Metal device on this machine"
)
def test_mps_agrees_with_cpu() -> None:  # pragma: no cover
    params = load_fixture(SMALL_SITES)
    dataset = simulate_alignment(
        tau=params.tau, k=params.k, pi=params.pi, seed=params.seed, n_sites=500
    )
    alignment = dict(dataset.alignment)

    # float32 on both sides: Metal cannot do float64, so the CPU side is
    # narrowed to match rather than the comparison being cross-precision.
    dtype = default_dtype("mps")
    on_cpu = pruning_torch.log_likelihood(
        params.tau,
        params.k,
        params.pi,
        alignment,
        pruning_torch.branch_lengths_from_tree(params.tau, dtype=dtype),
    )
    on_mps = pruning_torch.log_likelihood(
        params.tau,
        params.k,
        params.pi,
        alignment,
        pruning_torch.branch_lengths_from_tree(params.tau, dtype=dtype, device="mps"),
    )
    assert_allclose(float(on_mps), float(on_cpu), rtol=cross_device_rtol(dtype, dtype))
