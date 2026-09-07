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
    labelled_log_likelihood,
    map_labelling,
)
from snakes_and_ladders.opt.mixture import emission_mixture_plus_plus, kmeans_plus_plus
from snakes_and_ladders.opt.schedule import Exponential
from snakes_and_ladders.search.spatio_sequential import (
    LabelSolver,
    fit_spatio_sequential,
    graph_burn_in,
    label_accuracy,
    seed_emissions,
)
from snakes_and_ladders.sim.graph import BoundaryCondition, lattice_graph
from snakes_and_ladders.sim.spatio_sequential import (
    SpatioSequentialParams,
    canonical_spatio_sequential,
    simulate_spatio_sequential,
)

WOLFF = Exponential(2.0, 0.2, 12)


def _planted_lattice(side: int) -> tuple[SpatioSequentialParams, np.ndarray]:
    """A lattice past enumeration with weak emissions, so the prior matters."""
    params = SpatioSequentialParams(
        graph=lattice_graph((side, side), BoundaryCondition.OPEN, 1.0),
        n_classes=2,
        n_states=2,
        n_positions=4,
        beta=1.0,
        self_transition=0.7,
        initial=np.array([[0.6, 0.4], [0.3, 0.7]]),
        emissions=(
            CategoricalEmission(np.array([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2]])),
            CategoricalEmission(np.array([[0.2, 0.2, 0.6], [0.4, 0.4, 0.2]])),
        ),
    )
    planted = (np.arange(side * side) % side < side // 2).astype(np.int64)
    return params, planted


@pytest.mark.oracle
@pytest.mark.parametrize("solver", list(LabelSolver))
def test_the_label_step_reaches_the_enumerated_map_from_the_planted_labels(
    solver: LabelSolver,
) -> None:
    # Coordinate ascent on the labels with the parameters at the truth: a
    # local optimum of the joint, which from the planted labels is the
    # enumerated MAP on 5 of 6 draws for every solver (the sixth is a draw
    # whose planted labelling sits in another basin), and from a uniform
    # start on 3 of 6. Asserted at the margin, four of six.
    params = canonical_spatio_sequential()
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


@pytest.mark.mathematical
@pytest.mark.parametrize("solver", list(LabelSolver))
def test_the_labelled_joint_never_decreases_across_blocks(solver: LabelSolver) -> None:
    params = canonical_spatio_sequential()
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


@pytest.mark.simulated_truth
def test_the_label_step_recovers_planted_labels_when_the_parameters_are_known() -> None:
    # The label problem alone is easy: with theta at the truth, the field
    # separates the classes on 98 to 99 percent of nodes over six draws.
    params, planted = _planted_lattice(10)
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


@pytest.mark.simulated_truth
def test_the_annealed_start_beats_every_cold_solver_at_equal_blocks() -> None:
    # The study the ticket asked for, and it does not say what the ticket
    # expected. From a uniform start and the true parameters as the starting
    # point of EM, block ascent freezes on this instance whichever label
    # solver runs: mean accuracy up to permutation over six planted draws of
    # 0.66 (alpha expansion), 0.78 (single-site descent), 0.68 (annealed
    # Wolff). The cluster move does not escape what descent freezes into --
    # the trap is the parameters, not the labels. Graph_BurnIn++, which
    # anneals the prior while the emissions are fitted to what the data
    # supports, reaches 0.97. Asserted at the margins: the annealed start
    # above every cold solver, and every cold solver below 0.9.
    params, planted = _planted_lattice(10)
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
                wolff_schedule=Exponential(3.0, 0.3, 30),
            )
            accuracy[solver.value].append(label_accuracy(fit.labels, planted, 2))
        warm = graph_burn_in(
            params,
            data.observations,
            np.random.default_rng(seed),
            Exponential(4.0, 1.0, 30),
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


@pytest.mark.mathematical
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


@pytest.mark.structural
def test_seeding_replaces_every_categorical_row_with_a_seeded_one() -> None:
    params = canonical_spatio_sequential()
    data = simulate_spatio_sequential(params, np.random.default_rng(1))

    seeded = seed_emissions(params, data.observations, np.random.default_rng(2))

    for family in seeded.emissions:
        assert isinstance(family, CategoricalEmission)
        matrix = family.matrix.numpy()
        np.testing.assert_allclose(matrix.sum(axis=1), 1.0)
        assert (matrix.max(axis=1) == 0.9).all()


@pytest.mark.mathematical
def test_label_accuracy_is_taken_over_permutations() -> None:
    planted = np.array([0, 0, 1, 1, 2])
    assert label_accuracy(np.array([2, 2, 0, 0, 1]), planted, 3) == 1.0
    assert label_accuracy(np.array([0, 0, 1, 1, 1]), planted, 3) == 0.8


@pytest.mark.edge_case
def test_the_wolff_solver_needs_a_schedule_and_a_block_count_is_positive() -> None:
    params = canonical_spatio_sequential()
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
