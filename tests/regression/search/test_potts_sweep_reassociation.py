"""Why the tempered sweep scales the accumulated field, not its parts (#651).

Dropping two whole-array temporaries per sweep is available for free in
`search/potts_mcmc.py`: pass ``beta`` into the parts and compute
``beta h + sum(beta J)`` instead of ``(h + sum J) beta``. The two are equal in
real arithmetic, and #571 refused the cheaper one for not being bitwise.

`CLAUDE.md` now permits backing off bitwise to the declared tolerance, so #651
re-opened the question. **The measurement refuses it, and on mechanism rather
than on caution.**

The Rust sweep already tolerates a last-place disagreement: it decides a site
only where the draw clears every cumulative boundary by ``_GUARD`` units of the
last place, and hands any site it declines back to the oracle's own update. So
a reassociation inside the guard would cost extra hand-backs and nothing else.
This one is not inside the guard, and it is not a rounding difference at all:
``h + sum J`` cancels, so the scaled-parts form carries an absolute error set
by the magnitudes of the *parts* while the result is near zero.

The error therefore concentrates exactly where it does most harm --- a site
whose accumulated field is near zero is a site whose conditional is closest to
uniform, which is the site most likely to flip. That is why this is refused
under a rule written to admit exactly this kind of trade.
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.search.potts_mcmc import _GUARD

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
@pytest.mark.mathematical
def test_scaling_the_parts_leaves_the_guard_behind() -> None:
    # The refusal, as a number. The guard absorbs 16 units of the last place;
    # the reassociation reaches five orders past that, so the cheaper form
    # cannot be handed to the kernel and left to the hand-back path.
    errors, _ = _reassociation_ulps()

    assert np.median(errors) <= 2.0, "the typical site is a last-place difference"
    assert errors.max() > 1_000.0 * _GUARD, (
        f"worst reassociation {errors.max():.0f} ulps against a guard of "
        f"{_GUARD}; if this ever falls inside the guard the cheaper form is "
        "available and #651's first candidate should be re-read"
    )


@pytest.mark.critical
@pytest.mark.mathematical
def test_the_error_concentrates_where_the_conditional_is_finely_balanced() -> None:
    # Why the tail is not a tail to be tolerated. `h + sum J` cancels, so the
    # scaled-parts form's absolute error is set by the parts while the result
    # is near zero -- and a near-zero accumulated field is a near-uniform
    # conditional, the site whose decision a perturbation is likeliest to flip.
    errors, fields = _reassociation_ulps()
    past_guard = errors > _GUARD

    assert 0.01 < past_guard.mean() < 0.10, (
        f"{past_guard.mean():.2%} of sites past the guard; the measurement this "
        "refusal rests on has moved"
    )
    assert np.median(fields[past_guard]) < 0.1 * np.median(fields[~past_guard]), (
        "the sites past the guard are the cancelling ones, which is the whole "
        "argument; if they are not, the mechanism has changed"
    )
