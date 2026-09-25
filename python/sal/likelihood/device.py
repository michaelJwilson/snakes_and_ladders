"""Device selection and the cross-device agreement tolerance.

`ROADMAP.md` requires native dispatch across CUDA, Metal/MPS and CPU, and
root `CLAUDE.md` requires agreement between them to be checked against a
stated tolerance rather than bitwise. This module holds both halves: which
device to run on, and how close two devices must agree to count as agreeing.

Precision is the constraint that shapes everything here. **PyTorch's Metal
backend does not support float64**, so an Apple Silicon path necessarily runs
the pruning recursion in float32, and the tolerance has to admit that. CUDA
supports float64 and should therefore be held to a much tighter bound; one
tolerance covering both would let a genuinely broken float64 backend pass.

The tolerances are *relative*. The log-likelihood is a sum over sites, so its
magnitude -- and any absolute discrepancy in it -- grows with the site count,
and an absolute bound fixed at one problem size does not transfer to another
(issue #111). Measured float32-against-float64 agreement on CPU, which is the
same arithmetic Metal will do:

===================================  =========  ==========  ==========
fixture                              ``|lnL|``  absolute    relative
===================================  =========  ==========  ==========
4 taxa, 20,000 sites                 8.1e+04    4.38e-03    5.38e-08
4 taxa, 200,000 sites                8.2e+05    3.61e-02    4.41e-08
8 taxa, 200,000 sites                1.5e+06    4.99e-02    3.36e-08
===================================  =========  ==========  ==========

The absolute column spans an order of magnitude; the relative column is flat
at roughly 0.4 times float32 epsilon, which is the floor. That is why the
tolerance is relative, and why it is set where it is.

An absolute bound is not merely inconvenient at these magnitudes, it is
unreachable: near ``|lnL| = 2.4e5`` adjacent float32 values are ``0.0156``
apart, so no absolute bound tighter than that is achievable in float32 at
all, however good the kernel.
"""

from __future__ import annotations

import torch

# Relative tolerance for a comparison where both sides are float64. The
# backends agree to 7.8e-13 relative at fixture scale, so this leaves better
# than an order of magnitude of headroom for a device reordering operations.
CROSS_DEVICE_RTOL_FLOAT64 = 1e-11

# Relative tolerance for a comparison where either side is float32 -- every
# Metal comparison, since MPS cannot do float64. Measured agreement is
# ~4.4e-08 (above), so this leaves well over an order of magnitude for a
# device that reorders reductions differently from CPU.
CROSS_DEVICE_RTOL_FLOAT32 = 1e-6

# Devices in preference order. CPU is last and always available, so selection
# never fails.
_PREFERENCE = ("cuda", "mps", "cpu")


def select_device(*, cuda_available: bool, mps_available: bool) -> str:
    """Choose a device from what is available, preferring the fastest.

    Pure, so CI can test the policy on a runner with neither accelerator by
    passing the availability flags directly.

    Parameters
    ----------
    cuda_available : bool
        Whether a CUDA device is present.
    mps_available : bool
        Whether a Metal/MPS device is present.

    Returns
    -------
    str
        ``"cuda"``, ``"mps"`` or ``"cpu"``.
    """
    if cuda_available:
        return "cuda"
    if mps_available:
        return "mps"
    return "cpu"


def available_device() -> str:
    """The device this machine should run on.

    Returns
    -------
    str
        The result of :func:`select_device` for the current machine.
    """
    return select_device(
        cuda_available=torch.cuda.is_available(),
        mps_available=torch.backends.mps.is_available(),
    )


def default_dtype(device: str) -> torch.dtype:
    """The widest dtype ``device`` supports.

    Parameters
    ----------
    device : str
        Device name, as returned by :func:`select_device`.

    Returns
    -------
    torch.dtype
        ``torch.float32`` on Metal, which rejects float64 outright;
        ``torch.float64`` everywhere else.

    Raises
    ------
    ValueError
        If ``device`` is not one this module dispatches to -- an unknown
        device would otherwise silently take the float64 branch and fail
        deep inside the recursion.
    """
    if device not in _PREFERENCE:
        msg = f"unknown device {device!r}, expected one of {_PREFERENCE}"
        raise ValueError(msg)
    return torch.float32 if device == "mps" else torch.float64


def cross_device_rtol(*dtypes: torch.dtype) -> float:
    """The relative tolerance a comparison across ``dtypes`` is held to.

    Keyed on the lowest precision involved: a comparison is only as accurate
    as its least accurate side, and holding a float32 comparison to the
    float64 bound would fail on correct code.

    Parameters
    ----------
    *dtypes : torch.dtype
        The dtypes taking part in the comparison.

    Returns
    -------
    float
        :data:`CROSS_DEVICE_RTOL_FLOAT32` if any side is float32, otherwise
        :data:`CROSS_DEVICE_RTOL_FLOAT64`.

    Raises
    ------
    ValueError
        If no dtypes are given, or one is neither float32 nor float64.
    """
    if not dtypes:
        msg = "cross_device_rtol needs at least one dtype"
        raise ValueError(msg)
    for dtype in dtypes:
        if dtype not in (torch.float32, torch.float64):
            msg = f"unsupported dtype {dtype}, expected float32 or float64"
            raise ValueError(msg)
    return (
        CROSS_DEVICE_RTOL_FLOAT32
        if torch.float32 in dtypes
        else CROSS_DEVICE_RTOL_FLOAT64
    )
