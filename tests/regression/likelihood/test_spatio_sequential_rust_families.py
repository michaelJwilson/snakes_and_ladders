"""The Rust route on Gaussian, negative-binomial and beta-binomial classes (issue #1308).

The kernel reads one table per call and adds nothing to it: a row is a count,
a ``(count, trial count)`` pair, or a site, and each table is the family's own
``log_density`` at those rows. The referee is the NumPy route, at #1307's
tolerances: ``rtol`` 1e-12 on the evidence, the field and the labelled joint,
``atol`` 1e-8 on the posterior and the pairwise. Measured over every case,
tier and layout here: 1.9e-14 on the posterior and the pairwise, 1.0e-14
relative on the field, 3.6e-16 on the evidence, 2.4e-16 on the labelled joint.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
from sal import oxisal
from sal.backend import Backend
from sal.emissions import (
    BetaBinomialEmission,
    CategoricalEmission,
    EmissionFamily,
    GaussianEmission,
    NegativeBinomialEmission,
)
from sal.likelihood.spatio_sequential import (
    CovariateRows,
    class_posteriors,
    external_field,
    labelled_log_likelihood,
    log_prior,
    observation_rows,
    rust,
)
from sal.search.spatio_sequential import fit_spatio_sequential
from sal.sim.fixtures import fixture
from sal.sim.spatio_sequential import (
    SpatioSequentialParams,
    simulate_spatio_sequential,
)

#: Relative, on sums over sites (`test_spatio_sequential_rust.py`).
LOG_TOLERANCE = 1e-12

#: Absolute, on probabilities.
POSTERIOR_TOLERANCE = 1e-8

#: Each family under test, its covariate, and the row its table is keyed on
#: at the stress instance. Trial counts in ``[10, 40)`` make a pair table of
#: 1,230 rows against 400 sites, so it is scored by site; in ``[28, 31)`` the
#: pair table is 93 rows and is read.
CASES = {
    "gaussian": ("site", None),
    "nb": ("count", None),
    "nb-exposure": ("site", "exposure"),
    "bb": ("count", None),
    "bb-trials-wide": ("site", (10, 40)),
    "bb-trials-narrow": ("pair", (28, 31)),
}


def _families(name: str) -> tuple[EmissionFamily, EmissionFamily]:
    """Two classes of the family ``name`` names, their states separated."""
    if name == "gaussian":
        return (
            GaussianEmission([-1.0, 1.0], [1.0, 0.7], 1e-6),
            GaussianEmission([0.0, 2.5], [0.8, 1.2], 1e-6),
        )
    if name.startswith("nb"):
        return (
            NegativeBinomialEmission([4.0, 6.0], [3.0, 12.0]),
            NegativeBinomialEmission([5.0, 3.0], [7.0, 20.0]),
        )
    return (
        BetaBinomialEmission([30.0, 30.0], [2.0, 6.0], [6.0, 2.0]),
        BetaBinomialEmission([30.0, 30.0], [4.0, 1.0], [4.0, 3.0]),
    )


def _instance(
    name: str, tier: str = "stress"
) -> tuple[SpatioSequentialParams, np.ndarray, np.ndarray]:
    """The ``spatio_sequential`` lattice at ``tier`` with ``name``'s classes, one draw."""
    base = fixture("spatio_sequential", tier).params
    shape = (base.n_positions, base.graph.n_nodes)
    rng = np.random.default_rng(7)
    covariate = CASES[name][1]
    drawn = None
    if covariate == "exposure":
        drawn = rng.uniform(0.5, 2.0, size=shape)
        # A zero exposure marks a count unobserved (#933), and is scored too.
        drawn[0, 0] = 0.0
    elif isinstance(covariate, tuple):
        drawn = rng.integers(*covariate, size=shape).astype(float)
    params = replace(base, emissions=_families(name), covariate=drawn)
    data = simulate_spatio_sequential(params, np.random.default_rng(0))
    return params, data.observations, data.labels


def _assert_routes_agree(
    params: SpatioSequentialParams,
    observations: np.ndarray,
    labels: np.ndarray,
    layout: CovariateRows = "range",
) -> None:
    """The four quantities, the Rust route against the NumPy oracle."""
    expected = class_posteriors(params, observations, labels)
    actual = class_posteriors(
        params, observations, labels, backend=Backend.RUST, covariate_rows=layout
    )
    np.testing.assert_allclose(
        actual.log_evidence, expected.log_evidence, rtol=LOG_TOLERANCE
    )
    np.testing.assert_allclose(
        actual.posterior, expected.posterior, atol=POSTERIOR_TOLERANCE
    )
    np.testing.assert_allclose(
        actual.pairwise, expected.pairwise, atol=POSTERIOR_TOLERANCE
    )
    np.testing.assert_allclose(
        external_field(
            params, observations, labels, backend=Backend.RUST, covariate_rows=layout
        ),
        external_field(params, observations, labels),
        rtol=LOG_TOLERANCE,
    )
    np.testing.assert_allclose(
        labelled_log_likelihood(
            params, observations, labels, backend=Backend.RUST, covariate_rows=layout
        ),
        labelled_log_likelihood(params, observations, labels),
        rtol=LOG_TOLERANCE,
    )


@pytest.mark.oracle
@pytest.mark.parametrize("tier", ["ci", "stress"])
@pytest.mark.parametrize("name", list(CASES))
def test_the_rust_route_matches_the_oracle_on_every_family(
    name: str, tier: str
) -> None:
    # Issue #1308: the Rust route refused these classes. Every quantity the
    # fit reads now agrees with the NumPy oracle, covariate rows included.
    params, observations, labels = _instance(name, tier)

    _assert_routes_agree(params, observations, labels)


@pytest.mark.oracle
@pytest.mark.parametrize("layout", ["range", "distinct", "factored"])
@pytest.mark.parametrize("name", ["nb-exposure", "bb-trials-narrow"])
def test_every_covariate_layout_matches_the_oracle(
    name: str, layout: CovariateRows
) -> None:
    # `range` and `distinct` key a beta-binomial's table on the pair;
    # `factored` and a continuous exposure are scored by site. Each agrees.
    params, observations, labels = _instance(name)

    _assert_routes_agree(params, observations, labels, layout)


@pytest.mark.oracle
@pytest.mark.backend
@pytest.mark.parametrize("name", list(CASES))
def test_each_table_is_keyed_as_stated_and_is_the_family_s_density(name: str) -> None:
    # The row each case is read by, and every row's entry is the family's own
    # `log_density` at that observation, bit for bit: the table is the
    # oracle's arithmetic, not a second implementation of it.
    import torch

    params, observations, labels = _instance(name)
    table = rust.one_channel_table(params, observations)

    assert table.key == CASES[name][0]
    scores = table.table[table.rows]  # (S, V, M, K)
    for m, family in enumerate(params.emissions):
        covariate = (
            None
            if params.covariate is None
            else torch.as_tensor(params.covariate)[..., None]
        )
        np.testing.assert_array_equal(
            scores[:, :, m, :],
            family.log_density(
                torch.as_tensor(observations, dtype=family.observation_dtype),
                covariate=covariate,
            ).numpy(),
        )


@pytest.mark.oracle
def test_classes_of_different_families_are_scored_by_site() -> None:
    # Mixed families align by site: a Gaussian class beside a negative
    # binomial, and a categorical beside a Gaussian, each agree with the
    # oracle. Observations are counts, which all three score.
    params, observations, labels = _instance("nb", "ci")
    gaussian = _families("gaussian")[0]
    categorical = CategoricalEmission(np.full((params.n_states, 128), 1.0 / 128.0))

    for emissions in [(params.emissions[0], gaussian), (categorical, gaussian)]:
        mixed = replace(params, emissions=emissions)
        assert rust.one_channel_table(mixed, observations).key == "site"
        _assert_routes_agree(mixed, observations, labels)


@pytest.mark.oracle
@pytest.mark.backend
@pytest.mark.parametrize("name", ["gaussian", "categorical"])
def test_rows_built_once_hold_the_table_for_the_same_parameters(name: str) -> None:
    # A fit hands every call the rows it built once; the table is kept with
    # the parameters it was built under, returned while they are the same
    # object, and rebuilt for others. Held or rebuilt, the E step is the same.
    if name == "categorical":
        params = fixture("spatio_sequential", "ci").params
        data = simulate_spatio_sequential(params, np.random.default_rng(0))
        observations, labels = data.observations, data.labels
        build: Callable[..., object] = rust.symbol_table
    else:
        params, observations, labels = _instance(name, "ci")
        build = rust.one_channel_table
    rows = observation_rows(observations, params.covariate)

    first = build(params, observations, covariate_rows=rows)
    assert build(params, observations, covariate_rows=rows) is first
    assert build(replace(params), observations, covariate_rows=rows) is not first
    held = rust.class_posteriors(params, observations, labels, covariate_rows=rows)
    fresh = rust.class_posteriors(params, observations, labels)
    np.testing.assert_array_equal(held.log_evidence, fresh.log_evidence)
    np.testing.assert_array_equal(held.posterior, fresh.posterior)


@pytest.mark.oracle
@pytest.mark.parametrize("name", ["gaussian", "nb-exposure", "bb-trials-narrow"])
def test_the_rust_fit_is_the_python_fit_on_every_family(name: str) -> None:
    # The fit, from one generator, reaches the labels the NumPy fit reaches,
    # and the same joint at every block.
    params, observations, _ = _instance(name)

    fits = [
        fit_spatio_sequential(
            params, observations, np.random.default_rng(1), backend=backend, n_blocks=4
        )
        for backend in (Backend.PYTHON, Backend.RUST)
    ]

    np.testing.assert_array_equal(fits[1].labels, fits[0].labels)
    np.testing.assert_allclose(
        fits[1].log_likelihoods, fits[0].log_likelihoods, rtol=LOG_TOLERANCE
    )


def _log_prior_by_edge(
    params: SpatioSequentialParams, labellings: np.ndarray
) -> np.ndarray:
    """The loop over edges `log_prior` was before issue #1308, written out."""
    total = np.zeros(labellings.shape[0])
    for (first, second), coupling in params.graph.weighted_edges():
        total += (
            params.beta * coupling * (labellings[:, first] == labellings[:, second])
        )
    return total


@pytest.mark.oracle
@pytest.mark.parametrize("tier", ["ci", "stress"])
def test_the_label_prior_is_the_loop_over_edges_bit_for_bit(tier: str) -> None:
    # Vectorised in NumPy and ported to Rust, each term `(beta * J_e) [l_i ==
    # l_j]` summed in edge order: both are the loop's sum, bitwise, under
    # couplings that are not integers.
    params = fixture("spatio_sequential", tier).params
    rng = np.random.default_rng(3)
    graph = replace(
        params.graph,
        coupling=tuple(rng.uniform(0.1, 2.0, size=len(params.graph.edges))),
    )
    params = replace(params, graph=graph, beta=0.37)
    labellings = rng.integers(0, params.n_classes, size=(256, graph.n_nodes))
    expected = _log_prior_by_edge(params, labellings)

    np.testing.assert_array_equal(log_prior(params, labellings), expected)
    ported = [
        oxisal.coupled_log_prior(
            np.ascontiguousarray(labelling),
            graph.edge_index.reshape(-1),
            graph.edge_coupling,
            params.beta,
        )
        for labelling in labellings
    ]
    np.testing.assert_array_equal(ported, expected)
