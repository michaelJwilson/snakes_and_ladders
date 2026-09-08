"""The two-channel count emission and the binned coupled instances (issue #399).

What is checked here, at the ci fixture's size: that binning is the sum it
claims to be, that the pair's density is a distribution, that the simulator
draws from the families the file declares, and that aggregation is exact for
the negative-binomial channel and is not for the beta-binomial one --- which
is the reason a coarse instance asserts a parameter of the first and only a
label of the second.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml
from snakes_and_ladders.emissions import BetaBinomialEmission, NegativeBinomialEmission
from snakes_and_ladders.sim.count_pairs import (
    SUCCESSES,
    TOTAL,
    CountPairInstance,
    IndependentCountPair,
    SpatioSequentialCountsParams,
    aggregate,
    binned_instance,
    coarsen,
    fine_instance,
    load_spatio_sequential_counts_params,
    planted_labels,
)
from snakes_and_ladders.sim.fixtures import fixture
from snakes_and_ladders.sim.spatio_sequential import circulant_transition

CI = "spatio_sequential_counts"


def _fine() -> tuple[SpatioSequentialCountsParams, CountPairInstance]:
    """The ci fixture's declaration and its fine draw."""
    entry = fixture(CI, "ci")
    declared: SpatioSequentialCountsParams = entry.params
    return declared, fine_instance(entry.path)


def _families(declared: SpatioSequentialCountsParams) -> list[IndependentCountPair]:
    """The declared families at their own type, one per class."""
    families = declared.model.emissions
    assert all(isinstance(family, IndependentCountPair) for family in families)
    return [family for family in families if isinstance(family, IndependentCountPair)]


@pytest.mark.oracle
@pytest.mark.parametrize("factor", [5, 10])
def test_binning_is_the_sum_over_each_block_of_positions(factor: int) -> None:
    # Pinned against the sum written out position by position, in both
    # channels. The reshape-and-sum is the whole of the coarsening, and it is
    # what makes the three declared instances one draw rather than three.
    _, fine = _fine()
    binned = coarsen(fine, factor)

    n_bins = fine.observations.shape[0] // factor
    expected = np.zeros_like(binned.observations)
    for index in range(n_bins):
        for offset in range(factor):
            expected[index] += fine.observations[index * factor + offset]

    np.testing.assert_array_equal(binned.observations, expected)
    assert binned.params.n_positions == n_bins
    assert binned.observations.shape == (n_bins, fine.observations.shape[1], 2)


@pytest.mark.mathematical
def test_the_pair_density_is_a_distribution_over_its_support() -> None:
    # The two channels multiply, so the pair's mass is the product of two
    # masses and must sum to one over the joint support. The negative
    # binomial's support is unbounded, so the sum is truncated far enough out
    # that the remaining tail is below the tolerance asserted.
    family = IndependentCountPair(
        NegativeBinomialEmission(dispersion=[3.0, 6.0], mean=[4.0, 9.0]),
        BetaBinomialEmission(trials=[5.0, 5.0], alpha=[2.0, 4.0], beta=[3.0, 1.0]),
    )
    totals = np.arange(400)
    successes = np.arange(6)
    grid = np.stack(
        np.meshgrid(totals, successes, indexing="ij"), axis=-1
    )  # (n_totals, n_successes, 2)

    mass = torch.exp(family.log_density(torch.as_tensor(grid, dtype=torch.float64)))

    np.testing.assert_allclose(mass.sum(dim=(0, 1)).numpy(), [1.0, 1.0], atol=1e-9)


@pytest.mark.simulated_truth
def test_the_simulator_draws_from_the_declared_families() -> None:
    # Per (class, state), both channels' sample mean and variance against the
    # closed forms the families state. The tolerances are Monte Carlo bounds
    # at about 16,000 draws per group: 2% on a mean, 8% on a variance, which
    # is roughly four standard errors of each.
    declared, fine = _fine()
    model = declared.model
    for m, family in enumerate(_families(declared)):
        members = np.flatnonzero(fine.labels == m)
        for k in range(model.n_states):
            positions = np.flatnonzero(fine.states[m] == k)
            block = fine.observations[np.ix_(positions, members)].astype(np.float64)
            for channel, channel_family in (
                (TOTAL, family.total),
                (SUCCESSES, family.successes),
            ):
                values = block[..., channel]
                assert values.size > 8_000
                np.testing.assert_allclose(
                    values.mean(), float(channel_family.mean[k]), rtol=0.02
                )
                np.testing.assert_allclose(
                    values.var(), float(channel_family.variance[k]), rtol=0.08
                )


def _pmf(
    family: NegativeBinomialEmission | BetaBinomialEmission, support: np.ndarray
) -> np.ndarray:
    """One state's probability mass over ``support``, per state, shape ``(n, n_states)``."""
    values = torch.as_tensor(support, dtype=torch.float64)
    return np.asarray(torch.exp(family.log_density(values)).numpy())


def _convolved(pmf: np.ndarray, factor: int) -> np.ndarray:
    """The mass of a sum of ``factor`` independent draws, by direct convolution.

    The oracle for :func:`aggregate`: it shares no code with the closed forms
    the families state, so a family that is closed under addition can be
    checked against the addition itself.
    """
    total = pmf
    for _ in range(factor - 1):
        total = np.convolve(total, pmf)
    return total


@pytest.mark.oracle
@pytest.mark.parametrize("factor", [5, 10])
def test_the_negative_binomial_channel_aggregates_exactly(factor: int) -> None:
    # A sum of f independent NB(r, p) counts is NB(f r, p): both the mean and
    # the dispersion scale by f. Checked against the f-fold convolution of the
    # base mass, which is what "a sum of f draws" means and shares no code
    # with the family's closed form. Exact to the truncation of the support.
    declared, _ = _fine()
    family = _families(declared)[0]
    aggregated = aggregate(family, factor).total
    support = np.arange(600)

    base = _pmf(family.total, support)
    for k in range(declared.model.n_states):
        convolved = _convolved(base[:, k], factor)
        coarse = _pmf(aggregated, np.arange(convolved.size))
        np.testing.assert_allclose(convolved, coarse[:, k], atol=1e-12)


@pytest.mark.simulated_truth
def test_the_binned_counts_follow_the_aggregated_negative_binomial() -> None:
    # The claim above, on the data rather than on the family: over the bins
    # whose positions share a hidden state --- two states are two values of p,
    # and only within one state are the summands identically distributed ---
    # the binned totals' mean and variance are the aggregated family's. The
    # tolerances are Monte Carlo bounds at the few thousand such draws the ci
    # instance carries at factor 5.
    declared, fine = _fine()
    model = declared.model
    factor = 5
    binned = coarsen(fine, factor)
    blocks = fine.states.reshape(model.n_classes, -1, factor)
    for m, family in enumerate(_families(declared)):
        aggregated = aggregate(family, factor).total
        members = np.flatnonzero(fine.labels == m)
        constant = blocks[m].min(axis=1) == blocks[m].max(axis=1)
        for k in range(model.n_states):
            bins = np.flatnonzero(constant & (blocks[m, :, 0] == k))
            values = binned.observations[np.ix_(bins, members)][..., TOTAL].astype(
                np.float64
            )
            assert values.size > 500
            np.testing.assert_allclose(
                values.mean(), float(aggregated.mean[k]), rtol=0.05
            )
            np.testing.assert_allclose(
                values.var(), float(aggregated.variance[k]), rtol=0.20
            )


@pytest.mark.oracle
def test_the_beta_binomial_channel_is_misspecified_under_aggregation() -> None:
    # A sum of f beta-binomials is not beta-binomial. Against the same
    # convolution oracle: the mean of BetaBinomial(f n, a, b) --- the family
    # `aggregate` returns, and the closest member of the family to the truth
    # --- is right, its variance is several times too large, and the two
    # distributions are far apart in total variation. That is why a coarse
    # instance's second channel is checked for label recovery and never for a
    # parameter.
    declared, _ = _fine()
    model = declared.model
    factor = 10
    family = _families(declared)[0]
    aggregated = aggregate(family, factor).successes
    trials = int(family.successes.trials[0])
    base = _pmf(family.successes, np.arange(trials + 1))
    coarse = _pmf(aggregated, np.arange(trials * factor + 1))

    for k in range(model.n_states):
        convolved = _convolved(base[:, k], factor)
        support = np.arange(convolved.size)
        truth_mean = float(support @ convolved)
        truth_variance = float(support**2 @ convolved) - truth_mean**2
        distance = 0.5 * float(np.abs(convolved - coarse[:, k]).sum())

        np.testing.assert_allclose(truth_mean, float(aggregated.mean[k]), rtol=1e-9)
        assert float(aggregated.variance[k]) > 4.0 * truth_variance
        assert distance > 0.3


@pytest.mark.mathematical
def test_the_chain_is_the_circulant_walk_the_transition_states() -> None:
    # The path is drawn as a walk on Z_K rather than by a categorical draw per
    # position, which is exact for a circulant transition and is what makes a
    # 20,000-position fixture load. Checked against the transition matrix
    # itself, at 999 steps per class.
    declared, fine = _fine()
    model = declared.model
    transition = circulant_transition(model.n_states, model.self_transition)
    counts = np.zeros((model.n_states, model.n_states))
    for m in range(model.n_classes):
        np.add.at(counts, (fine.states[m, :-1], fine.states[m, 1:]), 1.0)

    empirical = counts / counts.sum(axis=1, keepdims=True)

    np.testing.assert_allclose(empirical, transition, atol=0.03)


@pytest.mark.structural
def test_the_planted_labels_are_contiguous_bands_of_rows() -> None:
    # The prior is ferromagnetic, so the planted labelling must be smooth or a
    # recovery test measures the prior fighting the truth. Bands are that, and
    # each class gets the same number of rows to within one.
    declared, _ = _fine()
    model = declared.model
    assert model.graph.shape is not None
    rows, columns = model.graph.shape

    labels = planted_labels(model, model.n_classes).reshape(rows, columns)

    assert (labels == labels[:, :1]).all(), "a band spans a whole row"
    band = labels[:, 0]
    assert (np.diff(band) >= 0).all(), "bands are contiguous and ordered"
    sizes = np.bincount(band, minlength=model.n_classes)
    assert sizes.max() - sizes.min() <= 1


@pytest.mark.structural
def test_the_fine_draw_is_simulated_once_and_binned_from() -> None:
    # "Simulated once at load and held in memory" is the fixture's claim, and
    # the cache is what makes it true: a second reader gets the same array,
    # not a second 400 MB draw, and every coarse instance is a bin of it.
    entry = fixture(CI, "ci")

    assert fine_instance(entry.path) is fine_instance(entry.path)
    assert binned_instance(entry.path, 5) is binned_instance(entry.path, 5)
    np.testing.assert_array_equal(
        binned_instance(entry.path, 5).observations,
        coarsen(fine_instance(entry.path), 5).observations,
    )


@pytest.mark.structural
def test_the_key_instance_is_the_one_the_5k_file_marks() -> None:
    # `fixture(problem, "key")` is not a fourth size: it resolves to whichever
    # file marks one of its own instances the key one, and the tier it reports
    # is that file's own.
    entry = fixture(CI, "key")

    assert entry.path.stem == "stress"
    assert str(entry.tier) == "stress"
    assert entry.params.key_factor == 5
    assert entry.params.marker(5) == "key"
    assert entry.params.marker(1) == "release"
    assert entry.params.marker(10) == "stress"
    assert fixture(CI, "ci").params.factors == (1, 5, 10)


@pytest.mark.edge_case
@pytest.mark.parametrize(
    ("edit", "message"),
    [
        ({"bin": [{"factor": 3, "marker": "key"}]}, "does not divide"),
        ({"bin": [{"factor": 1, "marker": "hourly"}]}, "is not one of"),
        (
            {
                "bin": [
                    {"factor": 1, "marker": "key"},
                    {"factor": 5, "marker": "key"},
                ]
            },
            "at most one",
        ),
        ({"emissions": {"form": "joint"}}, "not implemented here"),
    ],
)
def test_a_fixture_that_cannot_mean_what_it_says_is_refused(
    tmp_path: Path, edit: dict[str, Any], message: str
) -> None:
    # Each of these parses as yaml and is not an instance: a bin that does not
    # divide the positions has a short last bin drawn from another
    # distribution, an unknown marker deselects silently, two key instances is
    # two defaults, and the joint form is not written yet.
    raw = yaml.safe_load(fixture(CI, "ci").path.read_text())
    for key, value in edit.items():
        raw[key] = {**raw[key], **value} if key == "emissions" else value
    path = tmp_path / "ci.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match=message):
        load_spatio_sequential_counts_params(path)


@pytest.mark.edge_case
def test_a_shifted_beta_binomial_rate_outside_the_unit_interval_is_refused(
    tmp_path: Path,
) -> None:
    # The per-class shift is what separates the classes in the second channel;
    # a shift that takes a rate out of (0, 1) is a file that does not describe
    # a beta-binomial, and clipping it would be the silent behaviour change
    # `CLAUDE.md` forbids.
    raw = yaml.safe_load(fixture(CI, "ci").path.read_text())
    raw["emissions"]["class_rate_shift"] = [0.9, 0.1]
    path = tmp_path / "ci.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="leaves"):
        load_spatio_sequential_counts_params(path)
