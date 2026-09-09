"""The Potts field, one row per site: the capability, and the three instances (issue #413).

Two claims, refereed differently at each size.

**That a per-site field is scored correctly.** The widened shape reaches four
routines --- the exact open-chain sampler, the Gibbs sweep, enumeration and
the strip transfer matrix --- and each is checked against one of the others.
Enumeration shares no recursion with any of them, so it referees the two
samplers; the strip shares no configuration walk with enumeration, so the two
referee each other where both reach; and a field whose rows are all equal
must reproduce the shared-field result exactly, which is the reduction that
catches a broadcast applied to the wrong axis.

**That the covariate is identified.** ``h[n, m] = alpha[m] * log(size[n] /
size_bar)`` is only a model if the sizes vary: with one size every row of the
field is equal and ``alpha`` is a constant in disguise. At ``ci`` the exact
marginals under the declared field are compared against the exact marginals
under its site-average, and ``alpha`` is recovered against enumeration; at
``stress`` the exact normalizer past enumeration referees the sampler; at
``release`` the tilt survives 5,041 vertices, where a field indexed by the
wrong site would leave none.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import minimize
from snakes_and_ladders.likelihood.potts import enumerate_potts, strip_log_partition
from snakes_and_ladders.sim.fixtures import fixture
from snakes_and_ladders.sim.graph import BoundaryCondition, lattice_graph
from snakes_and_ladders.sim.potts import (
    PottsSpotsParams,
    simulate_potts,
    site_field,
    spots_field,
)

#: Deviation two exact routes may differ by: both sum the same weights in a
#: different order, so the gap is the last bits of a float64 reduction.
_EXACT = 1e-12

#: The instances, loaded once each --- a stress draw is 15 s and a release
#: draw 14 s, and the tests below would otherwise repeat them.
CI: PottsSpotsParams = fixture("potts_spots", "ci").params
STRESS: PottsSpotsParams = fixture("potts_spots", "stress").params
RELEASE: PottsSpotsParams = fixture("potts_spots", "release").params


def _strip_shape(params: PottsSpotsParams) -> tuple[int, int]:
    """The declared lattice's ``(columns, width)``, which the strip transfers over."""
    shape = params.graph.shape
    if shape is None or len(shape) != 2:
        msg = f"the strip oracle needs a 2-D lattice, got shape {shape}"
        raise AssertionError(msg)
    return shape[0], shape[1]


def _marginals(configurations: np.ndarray, n_states: int) -> np.ndarray:
    """Empirical single-site frequencies, shape ``(n_nodes, n_states)``."""
    return np.stack(
        [
            np.bincount(column, minlength=n_states) / configurations.shape[0]
            for column in configurations.T
        ]
    )


# --- the widened shape ------------------------------------------------------


@pytest.mark.mathematical
def test_a_field_repeated_at_every_site_is_the_shared_field() -> None:
    # The reduction the widening has to satisfy: a per-site field whose rows
    # are equal is the same model as the shared field it was built from. A
    # broadcast applied along the state axis rather than the site axis passes
    # every shape check and fails this.
    graph = lattice_graph((3, 3), BoundaryCondition.OPEN, 0.6)
    shared = np.array([0.30, -0.10, -0.20])

    one = enumerate_potts(graph, shared)
    many = enumerate_potts(graph, np.tile(shared, (graph.n_nodes, 1)))

    assert many.log_partition == pytest.approx(one.log_partition, abs=_EXACT)
    assert_allclose(many.single_site, one.single_site, atol=_EXACT)


@pytest.mark.oracle
def test_the_strip_transfer_matrix_matches_enumeration_on_a_per_site_field() -> None:
    # The two exact routes, on a lattice small enough for both. The strip
    # walks columns and enumeration walks configurations, so they share only
    # the model convention -- which is what makes the strip usable as the
    # oracle at the stress instance, where enumeration cannot go.
    shape = (3, 3)
    graph = lattice_graph(shape, BoundaryCondition.OPEN, 0.6)
    field = np.random.default_rng(413).normal(size=(graph.n_nodes, 3))

    exact = enumerate_potts(graph, field)
    strip = strip_log_partition(shape, BoundaryCondition.OPEN, 0.6, field)

    assert strip == pytest.approx(exact.log_partition, abs=_EXACT)


@pytest.mark.oracle
def test_the_exact_open_chain_sampler_carries_the_field_of_each_site() -> None:
    # Where the shape change is most likely to be silently wrong: site 0's
    # field enters at the first draw and every other site's inside a transfer
    # matrix, so an off-by-one in that indexing shifts the whole chain's
    # field by one site. Enumeration is the referee, at a chain both reach.
    length, n_states = 6, 3
    chain = lattice_graph((length,), BoundaryCondition.OPEN, 0.7)
    field = np.random.default_rng(4131).normal(size=(length, n_states))
    n_samples = 200_000

    drawn = simulate_potts(chain, field, np.random.default_rng(1), n_samples)

    exact = enumerate_potts(chain, field)
    # 3 sqrt(p (1 - p) / n) at p = 0.5 and n = 200,000 is 0.0034.
    assert_allclose(
        _marginals(drawn.configurations, n_states), exact.single_site, atol=0.0034
    )


@pytest.mark.edge_case
def test_a_field_of_neither_shape_is_refused() -> None:
    # A field with one row per *state* on a graph whose node count differs is
    # the mistake the two accepted shapes make possible; it is refused rather
    # than broadcast into a different model.
    with pytest.raises(ValueError, match="expected"):
        site_field(np.zeros((4, 3)), 9)


@pytest.mark.edge_case
def test_a_size_that_is_not_positive_is_refused() -> None:
    # `log(size)` is the whole construction, so a size of zero is refused
    # where it is declared rather than becoming a field of -inf.
    with pytest.raises(ValueError, match="every size must be positive"):
        spots_field(np.array([1.0, 0.0]), np.array([1.0, 0.0]))


# --- the ci instance: enumeration referees everything -----------------------


@pytest.mark.oracle
def test_gibbs_matches_enumeration_at_the_declared_spots_instance() -> None:
    # The sampler under the declared per-site field against the exact
    # marginals of the same model, at the size all 19,683 configurations
    # enumerate. Measured deviation 0.0153 against the declared 0.04.
    drawn = simulate_potts(
        CI.graph,
        CI.field,
        np.random.default_rng(CI.seed),
        CI.n_samples,
        CI.burn_in,
    )

    exact = enumerate_potts(CI.graph, CI.field)

    assert_allclose(
        _marginals(drawn.configurations, CI.n_classes),
        exact.single_site,
        atol=CI.tolerance,
    )


@pytest.mark.mathematical
def test_the_declared_sizes_move_the_marginals() -> None:
    # The instance the ticket asks for rather than one where the field is a
    # constant in disguise: replacing the per-site field by its site average
    # has to change the exact marginals by more than the tolerance the
    # sampler is checked within, or the fixture is a uniform-field instance
    # under another name. The average is exactly zero here, because the
    # covariate is centred, so the control is the uniform distribution.
    exact = enumerate_potts(CI.graph, CI.field)
    averaged = enumerate_potts(
        CI.graph, np.tile(CI.field.mean(axis=0), (CI.graph.n_nodes, 1))
    )

    deviation = np.abs(exact.single_site - averaged.single_site).max()

    assert deviation == pytest.approx(0.2555, rel=0.01)
    assert deviation > CI.tolerance


def _fit_alpha(params: PottsSpotsParams, configurations: np.ndarray) -> np.ndarray:
    """Maximum likelihood ``alpha``, normalized exactly by enumeration.

    The model is an exponential family in ``alpha`` with sufficient statistic
    ``T_m = sum_n log(size_n / size_bar) [s_n = m]``, so the fit is a convex
    problem whose gradient is the difference between the model's ``T`` and
    the sample's. Both come from :func:`enumerate_potts`, which shares no
    code with the sampler that drew ``configurations``.

    ``alpha`` and ``alpha + c`` give the same model --- a constant shifts
    every class at a site by the same amount, which cancels --- so the
    estimate is returned in the sum-zero gauge the fixture declares its own
    ``alpha`` in.
    """
    covariate = np.log(params.sizes) - float(np.log(params.sizes).mean())
    indicator = configurations[:, :, np.newaxis] == np.arange(params.n_classes)
    observed = np.einsum("n,cnm->m", covariate, indicator) / configurations.shape[0]

    def objective(free: np.ndarray) -> tuple[float, np.ndarray]:
        alpha = np.concatenate([free, [0.0]])
        alpha = alpha - alpha.mean()
        exact = enumerate_potts(params.graph, spots_field(alpha, params.sizes))
        gradient = covariate @ exact.single_site - observed
        return exact.log_partition - float(alpha @ observed), gradient[:-1] - float(
            gradient.mean()
        )

    fitted = minimize(
        objective, np.zeros(params.n_classes - 1), jac=True, method="L-BFGS-B"
    )
    estimate = np.concatenate([fitted.x, [0.0]])
    return np.asarray(estimate - estimate.mean())


@pytest.mark.simulated_truth
def test_the_covariate_coefficients_are_recovered_at_the_ci_instance() -> None:
    # What makes the sizes part of the model rather than decoration: alpha is
    # identified from a draw, and the flat class's interval covers zero. The
    # interval is the maximum-likelihood one, 1.96 standard errors from the
    # observed information of the sufficient statistic in the sum-zero gauge.
    drawn = simulate_potts(
        CI.graph, CI.field, np.random.default_rng(CI.seed), CI.n_samples, CI.burn_in
    )
    fitted = _fit_alpha(CI, drawn.configurations)

    covariate = np.log(CI.sizes) - float(np.log(CI.sizes).mean())
    statistic = np.einsum(
        "n,cnm->cm",
        covariate,
        drawn.configurations[:, :, np.newaxis] == np.arange(CI.n_classes),
    )
    gauge = np.eye(CI.n_classes) - np.ones((CI.n_classes, CI.n_classes)) / CI.n_classes
    information = CI.n_samples * gauge @ np.cov(statistic.T) @ gauge
    half_width = 1.96 * np.sqrt(np.diag(np.linalg.pinv(information)))

    inside = np.abs(fitted - CI.alpha) < half_width
    assert inside.all(), f"{fitted} against {CI.alpha}, half-widths {half_width}"
    flat = int(np.flatnonzero(CI.alpha == 0.0)[0])
    assert abs(fitted[flat]) < half_width[flat]


# --- the stress instance: the exact normalizer past enumeration -------------


@pytest.mark.oracle
@pytest.mark.stress
def test_the_sampler_matches_the_exact_normalizer_past_enumeration() -> None:
    # 72 sites is 3**72 configurations and no enumeration, but a strip six
    # sites wide still transfers exactly. The quantity both routes produce is
    # the mean field energy: it is `d log Z(t h) / dt` at `t = 1`, which two
    # more transfer-matrix evaluations give, and it is a sample mean under
    # the sampler. A per-site field carried wrongly by either route moves one
    # of the two. Measured deviation 0.005 against the declared 0.25.
    step = 1e-3
    columns, width = _strip_shape(STRESS)
    scaled = [
        strip_log_partition(
            (columns, width),
            BoundaryCondition.OPEN,
            STRESS.graph.coupling[0],
            factor * STRESS.field,
        )
        for factor in (1.0 + step, 1.0 - step)
    ]
    exact = (scaled[0] - scaled[1]) / (2.0 * step)

    drawn = simulate_potts(
        STRESS.graph,
        STRESS.field,
        np.random.default_rng(STRESS.seed),
        STRESS.n_samples,
        STRESS.burn_in,
    )
    sites = np.arange(STRESS.graph.n_nodes)[np.newaxis, :]
    sampled = STRESS.field[sites, drawn.configurations].sum(axis=1)

    assert sampled.mean() == pytest.approx(exact, abs=STRESS.tolerance)


# --- the release instance: the tilt at 5,041 vertices -----------------------


@pytest.mark.simulated_truth
@pytest.mark.release
def test_the_covariate_still_tilts_the_labels_at_five_thousand_vertices() -> None:
    # That the field survives the size. Nothing is exact at 5,041 vertices,
    # so what is asserted is the property the covariate is in the model for:
    # the mean alpha of a site's label rises with the site's size, monotonely
    # across size quartiles. A field indexed by the wrong site leaves a tilt
    # of zero, which is 49 standard errors from the measured 0.4935.
    drawn = simulate_potts(
        RELEASE.graph,
        RELEASE.field,
        np.random.default_rng(RELEASE.seed),
        RELEASE.n_samples,
        RELEASE.burn_in,
    )

    quartile = np.digitize(RELEASE.sizes, np.quantile(RELEASE.sizes, [0.25, 0.5, 0.75]))
    by_quartile = [
        float(RELEASE.alpha[drawn.configurations[:, quartile == which]].mean())
        for which in range(4)
    ]
    tilt = np.array(
        [
            RELEASE.alpha[row[quartile == 3]].mean()
            - RELEASE.alpha[row[quartile == 0]].mean()
            for row in drawn.configurations
        ]
    )

    assert by_quartile == sorted(by_quartile)
    assert tilt.mean() == pytest.approx(0.4935, abs=RELEASE.tolerance)
    assert tilt.mean() > RELEASE.tolerance
