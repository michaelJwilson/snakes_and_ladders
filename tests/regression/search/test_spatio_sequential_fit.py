"""Block ascent on the coupled model: the label step against enumeration, monotonicity, and the studies.

With the parameters at the truth the label block is pinned against the
enumerated MAP labelling on the canonical instance; with the parameters
fitted the labelled joint is asserted non-decreasing across every block for
every solver; past enumeration, on a planted 10x10 lattice, the three label
solvers and the annealed start are measured against the planted labels at
equal blocks. The measurement is the finding: the annealed start is what
helps, and no cluster move beats single-site descent from a cold start
(issue #306).
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.emissions import CategoricalEmission
from snakes_and_ladders.likelihood.spatio_sequential import (
    enumerate_spatio_sequential,
    labelled_log_likelihood,
    map_labelling,
)
from snakes_and_ladders.opt.mixture import emission_mixture_plus_plus, kmeans_plus_plus
from snakes_and_ladders.sample.schedule import ExponentialTempSchedule
from snakes_and_ladders.search.spatio_sequential import (
    LabelSolver,
    fit_spatio_sequential,
    graph_burn_in,
    label_accuracy,
    redraw_small_labels,
    seed_emissions,
)
from snakes_and_ladders.sim.fixtures import fixture
from snakes_and_ladders.sim.spatio_sequential import (
    SpatioSequentialParams,
    simulate_spatio_sequential,
)
from snakes_and_ladders.track import MemoryRun, track

WOLFF = ExponentialTempSchedule(2.0, 0.2, 12)


def _planted_lattice() -> tuple[SpatioSequentialParams, np.ndarray]:
    """The stress fixture (past enumeration, weak emissions), the notebook's, and a planting."""
    params = fixture("spatio_sequential", "stress").params
    side = params.graph.shape[0]
    planted = (np.arange(side * side) % side < side // 2).astype(np.int64)
    return params, planted


@pytest.mark.oracle
@pytest.mark.parametrize("solver", list(LabelSolver))
def test_the_label_step_reaches_the_enumerated_map_from_the_planted_labels(
    solver: LabelSolver,
) -> None:
    # Labels at true parameters: the enumerated MAP on 5 of 6 draws from the
    # planting, 3 of 6 from uniform; asserted at four of six.
    params = fixture("spatio_sequential", "ci").params
    hits = 0
    for seed in range(6):
        data = simulate_spatio_sequential(params, np.random.default_rng(10 + seed))
        target = map_labelling(params, data.observations)

        fit = fit_spatio_sequential(
            params,
            data.observations,
            np.random.default_rng(seed),
            solver=solver,
            n_blocks=6,
            labels=data.labels,
            fit_parameters=False,
            wolff_schedule=WOLFF,
        )

        hits += int(np.array_equal(fit.labels, target))
        assert bool((np.diff(fit.log_likelihoods) >= -1e-9).all())
    assert hits >= 4, hits


BURN_IN = ExponentialTempSchedule(4.0, 1.0, 30)


def _enumerated_log_posterior(
    params: SpatioSequentialParams, observations: np.ndarray, labels: np.ndarray
) -> float:
    """``log p(l | x)`` exactly: the labelled joint, less both normalizers."""
    exact = enumerate_spatio_sequential(params, observations)
    return (
        labelled_log_likelihood(params, observations, labels)
        - exact.log_prior_normalizer
        - exact.log_evidence
    )


@pytest.mark.oracle
def test_the_burn_in_start_reaches_the_enumerated_map_of_the_model_it_reached() -> None:
    # The 2x2 mode, over twelve draws: Graph_BurnIn++ 9 (others 0.66, 0.84,
    # 1.05 nats below); uniform 2, mean 1,020.8 below. Asserted 8 of 12, 1.5 nats.
    params = fixture("spatio_sequential", "ci").params
    hits = 0
    gaps = []
    cold_gaps = []
    for seed in range(12):
        data = simulate_spatio_sequential(params, np.random.default_rng(10 + seed))

        warm = graph_burn_in(
            params, data.observations, np.random.default_rng(seed), BURN_IN
        )

        target = map_labelling(warm.params, data.observations)
        best = _enumerated_log_posterior(warm.params, data.observations, target)
        hits += int(np.array_equal(warm.labels, target))
        gaps.append(
            best
            - _enumerated_log_posterior(warm.params, data.observations, warm.labels)
        )
        cold = np.random.default_rng(seed).integers(
            0, params.n_classes, size=params.graph.n_nodes
        )
        cold_gaps.append(
            best - _enumerated_log_posterior(warm.params, data.observations, cold)
        )
    assert hits >= 8, hits
    assert max(gaps) < 1.5, gaps
    assert float(np.mean(cold_gaps)) > 100.0 * float(np.mean(gaps)), (gaps, cold_gaps)


@pytest.mark.analytic
@pytest.mark.parametrize("solver", list(LabelSolver))
def test_the_labelled_joint_never_decreases_across_blocks(solver: LabelSolver) -> None:
    params = fixture("spatio_sequential", "ci").params
    data = simulate_spatio_sequential(params, np.random.default_rng(3))

    fit = fit_spatio_sequential(
        params,
        data.observations,
        np.random.default_rng(0),
        solver=solver,
        n_blocks=8,
        wolff_schedule=WOLFF,
    )

    assert fit.log_likelihoods.shape == (17,)
    assert bool((np.diff(fit.log_likelihoods) >= -1e-9).all()), fit.log_likelihoods
    assert fit.log_likelihoods[-1] > fit.log_likelihoods[0]
    assert (
        abs(
            labelled_log_likelihood(fit.params, data.observations, fit.labels)
            - fit.log_likelihoods[-1]
        )
        < 1e-9
    )
    assert 0.0 < fit.params.self_transition < 1.0
    np.testing.assert_allclose(fit.params.initial.sum(axis=1), 1.0)


@pytest.mark.end2end
def test_the_label_step_recovers_planted_labels_when_the_parameters_are_known() -> None:
    # The label problem alone is easy: with theta at the truth, the field
    # separates the classes on 98 to 99 percent of nodes over six draws.
    params, planted = _planted_lattice()
    accuracies = []
    for seed in range(6):
        data = simulate_spatio_sequential(
            params, np.random.default_rng(50 + seed), labels=planted
        )
        fit = fit_spatio_sequential(
            params,
            data.observations,
            np.random.default_rng(seed),
            n_blocks=10,
            labels=planted,
            fit_parameters=False,
        )
        accuracies.append(label_accuracy(fit.labels, planted, 2))

    assert float(np.mean(accuracies)) >= 0.95, accuracies


@pytest.mark.end2end
def test_the_annealed_start_beats_every_cold_solver_at_equal_blocks() -> None:
    # From uniform, EM at the truth freezes for every solver: accuracy 0.66
    # (expansion), 0.78 (descent), 0.68 (annealed Wolff), six draws; the trap
    # is the parameters. Graph_BurnIn++: 0.97. Asserted: it above all, all < 0.9.
    params, planted = _planted_lattice()
    accuracy: dict[str, list[float]] = {solver.value: [] for solver in LabelSolver}
    accuracy["burn_in"] = []
    for seed in range(6):
        data = simulate_spatio_sequential(
            params, np.random.default_rng(50 + seed), labels=planted
        )
        for solver in LabelSolver:
            fit = fit_spatio_sequential(
                params,
                data.observations,
                np.random.default_rng(seed),
                solver=solver,
                n_blocks=10,
                wolff_schedule=ExponentialTempSchedule(3.0, 0.3, 30),
            )
            accuracy[solver.value].append(label_accuracy(fit.labels, planted, 2))
        warm = graph_burn_in(
            params,
            data.observations,
            np.random.default_rng(seed),
            ExponentialTempSchedule(4.0, 1.0, 30),
        )
        polished = fit_spatio_sequential(
            warm.params,
            data.observations,
            np.random.default_rng(seed),
            n_blocks=10,
            labels=warm.labels,
        )
        accuracy["burn_in"].append(label_accuracy(polished.labels, planted, 2))

    means = {name: float(np.mean(values)) for name, values in accuracy.items()}
    for solver in LabelSolver:
        assert means["burn_in"] > means[solver.value], means
        assert means[solver.value] < 0.9, means
    assert means["burn_in"] >= 0.9, means


@pytest.mark.analytic
def test_emission_mixture_plus_plus_is_kmeans_plus_plus_under_a_squared_distance() -> (
    None
):
    values = np.random.default_rng(0).normal(size=200)

    seeded = kmeans_plus_plus(values, 4, np.random.default_rng(3))
    general = emission_mixture_plus_plus(
        values,
        4,
        lambda centre, points: (points - centre) ** 2,
        np.random.default_rng(3),
    )

    np.testing.assert_array_equal(seeded, general)


@pytest.mark.smoke
def test_seeding_replaces_every_categorical_row_with_a_seeded_one() -> None:
    params = fixture("spatio_sequential", "ci").params
    data = simulate_spatio_sequential(params, np.random.default_rng(1))

    seeded = seed_emissions(params, data.observations, np.random.default_rng(2))

    for family in seeded.emissions:
        assert isinstance(family, CategoricalEmission)
        matrix = family.matrix.numpy()
        np.testing.assert_allclose(matrix.sum(axis=1), 1.0)
        assert (matrix.max(axis=1) == 0.9).all()


@pytest.mark.analytic
def test_label_accuracy_is_taken_over_permutations() -> None:
    planted = np.array([0, 0, 1, 1, 2])
    assert label_accuracy(np.array([2, 2, 0, 0, 1]), planted, 3) == 1.0
    assert label_accuracy(np.array([0, 0, 1, 1, 1]), planted, 3) == 0.8


@pytest.mark.smoke
def test_the_wolff_solver_needs_a_schedule_and_a_block_count_is_positive() -> None:
    params = fixture("spatio_sequential", "ci").params
    data = simulate_spatio_sequential(params, np.random.default_rng(1))
    with pytest.raises(ValueError, match="needs a schedule"):
        fit_spatio_sequential(
            params,
            data.observations,
            np.random.default_rng(0),
            solver=LabelSolver.WOLFF,
        )
    with pytest.raises(ValueError, match="at least one block"):
        fit_spatio_sequential(
            params, data.observations, np.random.default_rng(0), n_blocks=0
        )
    with pytest.raises(ValueError, match="n_centres"):
        emission_mixture_plus_plus(
            np.zeros(3), 4, lambda _c, v: v * 0, np.random.default_rng(0)
        )


@pytest.mark.smoke
def test_the_tracked_curve_is_the_block_s_own_joint() -> None:
    # One entry per block, each the value that block ended on: the series is
    # `log_likelihoods` read at the blocks, so the notebook comparing starts
    # reads what the fit already reports (issue #887).
    params = fixture("spatio_sequential", "ci").params
    data = simulate_spatio_sequential(params, np.random.default_rng(1))
    blocks = 4

    with track(MemoryRun()) as tracked:
        fit = fit_spatio_sequential(
            params, data.observations, np.random.default_rng(0), n_blocks=blocks
        )
    run = tracked.run
    assert isinstance(run, MemoryRun)

    recorded = run.series("log_likelihood")
    assert [step for step, _ in recorded] == list(range(blocks + 1))
    assert [value for _, value in recorded] == [
        fit.log_likelihoods[0],
        *fit.log_likelihoods[2::2].tolist(),
    ]
    assert run.last("state_bytes") == float(fit.labels.nbytes)


@pytest.mark.smoke
@pytest.mark.patch
def test_an_untracked_block_ascent_is_the_fit_before_the_hook() -> None:
    # The seam moves no number: outside a `track` block the binding is
    # `track.NULL` and `record` returns on its first line.
    params = fixture("spatio_sequential", "ci").params
    data = simulate_spatio_sequential(params, np.random.default_rng(1))

    outside = fit_spatio_sequential(
        params, data.observations, np.random.default_rng(0), n_blocks=4
    )
    with track(MemoryRun()):
        inside = fit_spatio_sequential(
            params, data.observations, np.random.default_rng(0), n_blocks=4
        )

    assert outside.log_likelihoods.tolist() == inside.log_likelihoods.tolist()


@pytest.mark.end2end
def test_the_seeded_start_recovers_planted_labels_at_the_enumerable_size() -> None:
    # #887's enumerable claim (moved by #891): from Emission_Mixture++ the
    # stripes beat 0.5 chance and the joint is not below the truth's: 0.792
    # and -7.0 nats over six draws.
    params = fixture("spatio_sequential", "ci").params
    planted = (np.arange(params.graph.n_nodes) % params.n_classes == 0).astype(np.int64)
    accuracies = []
    gaps = []
    for index in range(6):
        draw = simulate_spatio_sequential(
            params, np.random.default_rng([887, 2, index]), labels=planted
        )
        seeded = seed_emissions(params, draw.observations, np.random.default_rng(index))

        fit = fit_spatio_sequential(
            seeded, draw.observations, np.random.default_rng(index), n_blocks=6
        )

        accuracies.append(label_accuracy(fit.labels, planted, params.n_classes))
        gaps.append(
            enumerate_spatio_sequential(params, draw.observations).log_evidence
            - fit.log_likelihoods[-1]
        )
    assert float(np.mean(accuracies)) > 0.5, accuracies
    assert max(gaps) < 0.0, gaps


# --- a class too small to fit, relabelled (issue #933, R11) --------------------


@pytest.mark.analytic
def test_a_class_under_the_minimum_is_relabelled_onto_the_others() -> None:
    # Class 2 holds 3 sites and class 3 none; at a minimum of 4 exactly
    # class 2's sites move, each onto a class that is not under it (the
    # empty class 3 included), and one seed reproduces the draw.
    labels = np.array([0] * 10 + [1] * 8 + [2] * 3, dtype=np.int64)
    first = redraw_small_labels(labels, 4, 4, np.random.default_rng(933))
    again = redraw_small_labels(labels, 4, 4, np.random.default_rng(933))
    assert np.array_equal(first, again)
    moved = labels == 2
    assert np.array_equal(first[~moved], labels[~moved])
    assert set(first[moved].tolist()) <= {0, 1, 3}
    assert np.bincount(first, minlength=4)[2] == 0


@pytest.mark.analytic
def test_the_default_minimum_leaves_the_labels_and_the_generator_alone() -> None:
    labels = np.array([0, 0, 1, 2], dtype=np.int64)
    stream, reference = np.random.default_rng(1), np.random.default_rng(1)
    assert redraw_small_labels(labels, 3, 0, stream) is labels
    # Every occupied class under the minimum: nowhere to move a site to.
    assert redraw_small_labels(labels, 3, 5, stream) is labels
    assert stream.integers(1 << 62) == reference.integers(1 << 62)


@pytest.mark.smoke
def test_the_fit_redraws_small_classes_ahead_of_each_block() -> None:
    # The default is the fit as it was, bitwise; a minimum above every class
    # but one is reproducible on a seed and still ends on a labelling whose
    # joint is the recorded one.
    params = fixture("spatio_sequential", "ci").params
    data = simulate_spatio_sequential(params, np.random.default_rng(3))

    def fit(minimum: int | None) -> np.ndarray:
        rng = np.random.default_rng(0)
        if minimum is None:
            run = fit_spatio_sequential(
                params, data.observations, rng, solver=LabelSolver.ICM, n_blocks=4
            )
        else:
            run = fit_spatio_sequential(
                params,
                data.observations,
                rng,
                solver=LabelSolver.ICM,
                n_blocks=4,
                min_label_sites=minimum,
            )
        return run.log_likelihoods

    assert np.array_equal(fit(None), fit(0))
    assert np.array_equal(fit(4), fit(4))
