"""The Rust count-pair simulator against the NumPy one (issue #399).

PCG64 and ChaCha8 keyed by ``[seed, vertex]``: two samples of one model, so
the comparison is distributional. Per ``(class, state)`` and channel: the mean
and the index of dispersion, which separates these families from a Poisson.
Both are also held to the closed-form moments (`sim/CLAUDE.md`), the NumPy
one in ``test_count_pairs.py``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from sal.backend import Backend
from sal.emissions import PoissonEmission
from sal.sim.count_pairs import (
    SUCCESSES,
    TOTAL,
    CountPairInstance,
    IndependentCountPair,
    counts_digest,
)
from sal.sim.count_pairs import (
    simulate_count_pairs as simulate_numpy,
)
from sal.sim.fixtures import fixture
from sal.sim.rust.count_pairs import fine_instance
from sal.sim.rust.count_pairs import (
    simulate_count_pairs as simulate_rust,
)

PROBLEM = "spatio_sequential_counts"

#: Relative agreement required of a sample mean, at the roughly 16,000 draws
#: per (class, state) the ci instance carries in each channel. Two independent
#: samples, so the bound is about four standard errors of their difference.
MEAN_TOLERANCE = 0.04

#: Relative agreement required of an index of dispersion, whose standard error
#: is several times a mean's at the same sample size.
DISPERSION_TOLERANCE = 0.15


def _moments(instance: CountPairInstance, channel: int) -> np.ndarray:
    """Per ``(class, state)``, one channel's mean and dispersion index, ``(M, K, 2)``."""
    model = instance.params
    moments = np.empty((model.n_classes, model.n_states, 2))
    for m in range(model.n_classes):
        members = np.flatnonzero(instance.labels == m)
        for k in range(model.n_states):
            positions = np.flatnonzero(instance.states[m] == k)
            values = instance.observations[np.ix_(positions, members)][
                ..., channel
            ].astype(np.float64)
            moments[m, k] = (values.mean(), values.var() / values.mean())
    return moments


@pytest.mark.oracle
def test_the_rust_simulator_draws_what_the_numpy_one_draws() -> None:
    # Both channels, per (class, state), at the ci instance. The two draws
    # share a seed and a keying and share no arithmetic after that, so what
    # can agree is the distribution and this is the reading of it.
    declared = fixture(PROBLEM, "ci").params
    rust = simulate_rust(declared)
    numpy = simulate_numpy(declared)

    np.testing.assert_array_equal(rust.labels, numpy.labels)
    np.testing.assert_array_equal(rust.states, numpy.states)
    for channel in (TOTAL, SUCCESSES):
        expected, actual = _moments(numpy, channel), _moments(rust, channel)
        np.testing.assert_allclose(
            actual[..., 0], expected[..., 0], rtol=MEAN_TOLERANCE
        )
        np.testing.assert_allclose(
            actual[..., 1], expected[..., 1], rtol=DISPERSION_TOLERANCE
        )


@pytest.mark.oracle
@pytest.mark.backend
def test_the_backend_seam_reaches_the_rust_draw_bitwise() -> None:
    # `simulate_count_pairs(..., backend=Backend.RUST)` is the one way a caller
    # names the kernel (#813); it must hand back the twin's own draw and not a
    # third one, so the comparison is equality and not a moment.
    declared = fixture(PROBLEM, "ci").params
    through_seam = simulate_numpy(declared, backend=Backend.RUST)
    twin = simulate_rust(declared)

    np.testing.assert_array_equal(through_seam.observations, twin.observations)
    np.testing.assert_array_equal(through_seam.labels, twin.labels)
    np.testing.assert_array_equal(through_seam.states, twin.states)
    with pytest.raises(ValueError, match="runs on"):
        simulate_numpy(declared, backend=Backend.NUMBA)


@pytest.mark.end2end
def test_the_rust_draw_has_the_families_own_moments() -> None:
    # The comparison above is against a second sample; this one is against the
    # closed forms the families state, which is what `sim/CLAUDE.md` requires
    # of a simulator and what stops both simulators being wrong together.
    declared = fixture(PROBLEM, "ci").params
    instance = simulate_rust(declared)

    for m, family in enumerate(declared.model.emissions):
        assert isinstance(family, IndependentCountPair)
        for channel, side in ((TOTAL, family.total), (SUCCESSES, family.successes)):
            moments = _moments(instance, channel)[m]
            np.testing.assert_allclose(
                moments[:, 0], side.mean.numpy(), rtol=MEAN_TOLERANCE
            )
            np.testing.assert_allclose(
                moments[:, 1],
                side.variance.numpy() / side.mean.numpy(),
                rtol=DISPERSION_TOLERANCE,
            )


@pytest.mark.smoke
def test_the_draw_is_a_function_of_the_seed_and_the_vertex_alone() -> None:
    # The contract that lets a fixture be a file rather than a committed
    # array: the same declaration draws the same counts, in this process and
    # in the next one. The Rust unit tests carry the other half --- that a
    # vertex's counts do not depend on how many vertices are drawn beside it.
    declared = fixture(PROBLEM, "ci").params

    first = simulate_rust(declared)
    second = simulate_rust(declared)

    np.testing.assert_array_equal(first.observations, second.observations)
    assert counts_digest(first) == counts_digest(second)


@pytest.mark.smoke
@pytest.mark.parametrize("tier", ["ci", "release"])
def test_the_declared_digest_is_the_draw_the_file_names(tier: str) -> None:
    # The fixture records what its counts hash to, so a changed simulator is
    # a red test at the fixture rather than a drift in whatever was measured
    # on it three tests later. The release file draws in 0.06 s, so both
    # tiers are checked per pull request (issue #891).
    entry = fixture(PROBLEM, tier)

    assert entry.params.counts_digest is not None
    assert counts_digest(fine_instance(entry.path)) == entry.params.counts_digest


@pytest.mark.smoke
def test_a_draw_that_disagrees_with_the_recorded_digest_is_refused(
    tmp_path: Path,
) -> None:
    # The guard's own trigger: a file whose digest names a draw this simulator
    # does not produce is refused where it is loaded, not where it is used.
    raw = yaml.safe_load(fixture(PROBLEM, "ci").path.read_text())
    raw["counts_digest"] = "0" * 16
    path = tmp_path / "ci.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="the simulator or the parameters moved"):
        fine_instance(path)


@pytest.mark.smoke
def test_a_family_that_is_not_a_count_pair_is_refused() -> None:
    # The kernel draws two channels from five parameter tables. A declaration
    # it cannot fill is refused by name rather than by a shape error inside.
    declared = fixture(PROBLEM, "ci").params
    broken = replace(
        declared,
        model=replace(
            declared.model,
            emissions=tuple(
                PoissonEmission(np.full(declared.model.n_states, 3.0))
                for _ in range(declared.model.n_classes)
            ),
        ),
    )
    with pytest.raises(TypeError, match="two-channel count emission"):
        simulate_rust(broken)
