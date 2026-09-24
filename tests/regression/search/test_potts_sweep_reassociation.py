"""Why the tempered sweep scales the accumulated field, not its parts (#651).

``beta h + sum(beta J)`` would drop two temporaries per sweep against
``(h + sum J) beta``; #571 refused it as not bitwise, and #651 re-measured
under the backed-off rule. Refused on mechanism: the Rust sweep's ``GUARD``
absorbs last-place differences, but ``h + sum J`` cancels, so the error is set
by the parts' magnitudes while the result is near zero. It concentrates on
sites whose conditional is nearest uniform, the ones most likely to flip.
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.sample.potts_mcmc.sweeps import GUARD

SEED = 20260916
#: Sites drawn per trial, over neighbourhoods and alphabets the lattice fixtures
#: span. Large enough that the tail below is measured rather than sampled.
N_TRIALS = 50_000


def _reassociation_ulps() -> tuple[np.ndarray, np.ndarray]:
    """``(error in units of the last place, |accumulated field|)`` per state."""
    rng = np.random.default_rng(SEED)
    errors: list[float] = []
    fields: list[float] = []
    for _ in range(N_TRIALS):
        degree = int(rng.integers(3, 9))
        states = int(rng.integers(2, 6))
        field = rng.normal(0.0, 1.0, states)
        couplings = rng.normal(0.0, 1.0, (degree, states))
        beta = float(rng.uniform(0.05, 5.0))

        accumulated = (field + couplings.sum(axis=0)) * beta
        scaled_parts = beta * field + (beta * couplings).sum(axis=0)
        for reference, alternative in zip(accumulated, scaled_parts, strict=True):
            if reference == alternative:
                continue
            errors.append(abs(reference - alternative) / np.spacing(abs(reference)))
            fields.append(abs(reference))
    return np.asarray(errors), np.asarray(fields)


@pytest.mark.critical
@pytest.mark.analytic
def test_scaling_the_parts_leaves_the_guard_behind() -> None:
    # The refusal, as a number. The guard absorbs 16 units of the last place;
    # the reassociation reaches five orders past that, so the cheaper form
    # cannot be handed to the kernel and left to the hand-back path.
    errors, _ = _reassociation_ulps()

    assert np.median(errors) <= 2.0, "the typical site is a last-place difference"
    assert errors.max() > 1_000.0 * GUARD, (
        f"worst reassociation {errors.max():.0f} ulps against a guard of "
        f"{GUARD}; if this ever falls inside the guard the cheaper form is "
        "available and #651's first candidate should be re-read"
    )


@pytest.mark.critical
@pytest.mark.analytic
def test_the_error_concentrates_where_the_conditional_is_finely_balanced() -> None:
    # Why the tail is not a tail to be tolerated. `h + sum J` cancels, so the
    # scaled-parts form's absolute error is set by the parts while the result
    # is near zero -- and a near-zero accumulated field is a near-uniform
    # conditional, the site whose decision a perturbation is likeliest to flip.
    errors, fields = _reassociation_ulps()
    past_guard = errors > GUARD

    assert 0.01 < past_guard.mean() < 0.10, (
        f"{past_guard.mean():.2%} of sites past the guard; the measurement this "
        "refusal rests on has moved"
    )
    assert np.median(fields[past_guard]) < 0.1 * np.median(fields[~past_guard]), (
        "the sites past the guard are the cancelling ones, which is the whole "
        "argument; if they are not, the mechanism has changed"
    )
