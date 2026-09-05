"""Regression tests for the phylogenetic instance of ``Objective``.

Two things are pinned here that nothing else in the suite can pin. The
optimizer is the same model-agnostic ``snakes_and_ladders.opt.fit`` the Potts chain and
the HMM use, so this file is where issue #63's claim -- that the interface
was not secretly shaped by one model -- is either confirmed or refuted.

And the branch lengths below a rooted binary tree's root are shown to be
confounded, by measurement rather than by citation. That fact decides the
parameterization: fitting them separately leaves the observed information
singular and every interval undefined.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from snakes_and_ladders.likelihood import pruning_torch
from snakes_and_ladders.likelihood.objective import (
    BranchLengthObjective,
    SubstitutionModelObjective,
)
from snakes_and_ladders.opt.fit import (
    constrained_standard_errors,
    covers,
    fit,
    observed_information,
    parameter_covariance,
)
from snakes_and_ladders.sim.gtr import gtr_rate_matrix
from snakes_and_ladders.sim.jc import jc_rate_matrix
from snakes_and_ladders.sim.simulate import simulate_alignment
from snakes_and_ladders.sim.tree import Node, preorder

from tests._fixtures import EIGHT_TAXA, FOUR_TAXA, SMALL_SITES, load_fixture
from tests._objective_checks import assert_gradient_matches_finite_differences

# Enough sites that the maximum-likelihood estimate is well determined,
# few enough that a fit is seconds rather than minutes. The regression
# fixtures carry 2e5 sites for the Monte Carlo validation tests; nothing
# here needs that.
_SITES = 5000

_RTOL_GRADIENT = 1e-6
_FINITE_DIFFERENCE_STEP = 1e-6


def _objective(fixture: str, sites: int = _SITES) -> BranchLengthObjective:
    params = load_fixture(fixture)
    dataset = simulate_alignment(
        tau=params.tau, k=params.k, pi=params.pi, seed=params.seed, n_sites=sites
    )
    return BranchLengthObjective(
        params.tau, params.k, params.pi, dict(dataset.alignment)
    )


# --- the identifiability finding, and what follows from it ---------------


def test_the_two_branches_below_a_rooted_root_are_confounded() -> None:
    # The pulley principle, measured. Under a reversible model the
    # likelihood does not depend on where the root sits along the branch it
    # subdivides, so moving mass between the two root branches at fixed sum
    # changes nothing. This is the fact the parameterization is built on.
    params = load_fixture(EIGHT_TAXA)
    dataset = simulate_alignment(
        tau=params.tau, k=params.k, pi=params.pi, seed=params.seed, n_sites=_SITES
    )
    alignment = dict(dataset.alignment)
    order = pruning_torch.branch_order(params.tau)
    lengths = pruning_torch.branch_lengths_from_tree(params.tau)

    assert len(params.tau.children) == 2, "this fixture must be a rooted binary tree"
    first = order.index(params.tau.children[0].name)
    second = order.index(params.tau.children[1].name)
    total = float(lengths[first] + lengths[second])

    def score(fraction: float) -> float:
        candidate = lengths.clone()
        candidate[first] = total * fraction
        candidate[second] = total * (1.0 - fraction)
        return float(
            pruning_torch.log_likelihood(
                params.tau, params.k, params.pi, alignment, candidate
            )
        )

    reference = score(0.5)
    for fraction in (0.1, 0.3, 0.7, 0.9):
        assert_allclose(score(fraction), reference, rtol=1e-12)


def test_two_non_root_siblings_are_not_confounded() -> None:
    # The control. Without it the test above would also pass on a likelihood
    # that ignored branch lengths entirely.
    params = load_fixture(EIGHT_TAXA)
    dataset = simulate_alignment(
        tau=params.tau, k=params.k, pi=params.pi, seed=params.seed, n_sites=_SITES
    )
    alignment = dict(dataset.alignment)
    order = pruning_torch.branch_order(params.tau)
    lengths = pruning_torch.branch_lengths_from_tree(params.tau)

    siblings = [child.name for child in params.tau.children[0].children]
    first, second = order.index(siblings[0]), order.index(siblings[1])
    total = float(lengths[first] + lengths[second])

    def score(fraction: float) -> float:
        candidate = lengths.clone()
        candidate[first] = total * fraction
        candidate[second] = total * (1.0 - fraction)
        return float(
            pruning_torch.log_likelihood(
                params.tau, params.k, params.pi, alignment, candidate
            )
        )

    assert abs(score(0.3) - score(0.5)) > 1.0


class _Unmerged:
    """The naive parameterization: one free length per branch, root included.

    Kept as a test fixture rather than shipped, to show what the merged
    parameterization avoids.
    """

    def __init__(
        self,
        tau: Node,
        k: int,
        pi: np.ndarray,
        alignment: dict[str, np.ndarray],
    ) -> None:
        self._tau, self._k, self._pi, self._alignment = tau, k, pi, alignment
        self._n = len(pruning_torch.branch_order(tau))

    def initial(self) -> torch.Tensor:
        return torch.full((self._n,), float(np.log(0.1)), dtype=torch.float64)

    def constrain(self, theta: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"branch_lengths": torch.exp(theta)}

    def theta_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.log(named["branch_lengths"])

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        return -pruning_torch.log_likelihood(
            self._tau, self._k, self._pi, self._alignment, torch.exp(theta)
        )


def test_fitting_the_root_branches_separately_has_no_intervals() -> None:
    # The consequence of the confounding: a flat direction makes the
    # observed information singular, and `snakes_and_ladders.opt.fit` refuses to invert
    # it. This is what the merged parameterization exists to avoid, and the
    # reason it is not merely a tidier way to count parameters.
    params = load_fixture(EIGHT_TAXA)
    dataset = simulate_alignment(
        tau=params.tau, k=params.k, pi=params.pi, seed=params.seed, n_sites=_SITES
    )
    alignment = dict(dataset.alignment)
    naive = _Unmerged(params.tau, params.k, params.pi, alignment)
    result = fit(naive)

    with pytest.raises(ValueError, match="not identifiable"):
        parameter_covariance(naive, result.theta)


def test_the_merged_parameterization_does_have_intervals() -> None:
    objective = _objective(EIGHT_TAXA)
    errors = constrained_standard_errors(objective, fit(objective).theta)
    assert bool((errors["branch_lengths"] > 0.0).all())
    assert bool(torch.isfinite(errors["branch_lengths"]).all())


# --- the parameterization ------------------------------------------------


def test_a_rooted_tree_loses_exactly_one_parameter() -> None:
    objective = _objective(EIGHT_TAXA)
    assert objective.n_parameters == 13
    assert len(pruning_torch.branch_order(load_fixture(EIGHT_TAXA).tau)) == 14
    assert "left+right" in objective.parameter_names
    assert "left" not in objective.parameter_names
    assert "right" not in objective.parameter_names


def test_an_unrooted_tree_keeps_every_branch() -> None:
    params = load_fixture(SMALL_SITES)
    assert len(params.tau.children) == 3, "this fixture must be trifurcating"
    objective = _objective(SMALL_SITES)
    assert objective.parameter_names == pruning_torch.branch_order(params.tau)


def test_expansion_reproduces_the_tree_s_own_branch_lengths() -> None:
    # The merge must be lossless in the direction that matters: a tree's
    # lengths, encoded and expanded, must score identically.
    for fixture in (SMALL_SITES, EIGHT_TAXA):
        params = load_fixture(fixture)
        objective = _objective(fixture)
        expanded = objective.branch_lengths(objective.theta_from_truth(params.tau))
        assert_allclose(
            expanded.detach().numpy(),
            pruning_torch.branch_lengths_from_tree(params.tau).numpy(),
            rtol=1e-12,
        )


def test_a_root_with_one_child_is_refused() -> None:
    stunted = Node(name="root", branch_length=None, children=(Node("A", 0.1),))
    with pytest.raises(ValueError, match="at least 2"):
        BranchLengthObjective(stunted, 4, np.full(4, 0.25), {"A": np.zeros(3)})


def test_the_initial_point_is_uninformative() -> None:
    objective = _objective(SMALL_SITES)
    lengths = objective.constrain(objective.initial())["branch_lengths"]
    assert_allclose(lengths.numpy(), np.full(objective.n_parameters, 0.1), rtol=1e-12)


# --- the fit -------------------------------------------------------------


@pytest.mark.parametrize("fixture", [SMALL_SITES, EIGHT_TAXA])
def test_gradient_matches_central_finite_differences(fixture: str) -> None:
    objective = _objective(fixture, sites=500)
    params = load_fixture(fixture)
    assert_gradient_matches_finite_differences(
        objective,
        objective.theta_from_truth(params.tau),
        _FINITE_DIFFERENCE_STEP,
        _RTOL_GRADIENT,
    )


@pytest.mark.parametrize("fixture", [SMALL_SITES, EIGHT_TAXA])
def test_the_fit_beats_the_generating_branch_lengths(fixture: str) -> None:
    # The optimizer never sees the truth, so an early stop fails this while
    # its own loss still went down.
    params = load_fixture(fixture)
    objective = _objective(fixture)
    result = fit(objective)
    assert result.converged
    assert result.value < float(objective(objective.theta_from_truth(params.tau)))


@pytest.mark.parametrize("fixture", [SMALL_SITES, FOUR_TAXA, EIGHT_TAXA])
def test_every_branch_length_is_recovered_to_within_four_standard_errors(
    fixture: str,
) -> None:
    # Not "every 95% interval covers": on one dataset that is a draw, and
    # with five parameters it fails about a quarter of the time on correct
    # code -- as it did while this file was being written. Four standard
    # errors is the same statement made at a tail small enough to be a
    # deterministic assertion (realized worst deviation across these three
    # fixtures: 2.45). The nominal *rate* is the release-gated test below,
    # which is where coverage belongs.
    params = load_fixture(fixture)
    objective = _objective(fixture)
    result = fit(objective)
    estimate = objective.constrain(result.theta)["branch_lengths"]
    error = constrained_standard_errors(objective, result.theta)["branch_lengths"]
    truth = torch.exp(objective.theta_from_truth(params.tau))

    deviation = (estimate - truth).abs() / error
    worst = int(deviation.argmax())
    assert float(deviation.max()) < 4.0, (
        f"{objective.parameter_names[worst]} is "
        f"{float(deviation.max()):.2f} standard errors from truth"
    )


@pytest.mark.release
def test_branch_length_intervals_cover_at_the_nominal_rate() -> None:
    # Release-gated: 40 independent alignments, each fitted and inverted.
    params = load_fixture(SMALL_SITES)
    truth = None
    covered = 0
    total = 0
    for replicate in range(40):
        dataset = simulate_alignment(
            tau=params.tau,
            k=params.k,
            pi=params.pi,
            seed=params.seed + 7919 * replicate,
            n_sites=_SITES,
        )
        objective = BranchLengthObjective(
            params.tau, params.k, params.pi, dict(dataset.alignment)
        )
        if truth is None:
            truth = torch.exp(objective.theta_from_truth(params.tau))
        result = fit(objective)
        assert result.converged
        hits = covers(
            objective.constrain(result.theta)["branch_lengths"],
            constrained_standard_errors(objective, result.theta)["branch_lengths"],
            truth,
        )
        covered += int(hits.sum())
        total += hits.numel()

    rate = covered / total
    band = 3.0 * (0.95 * 0.05 / total) ** 0.5
    assert abs(rate - 0.95) <= band, (
        f"coverage {rate:.4f}, expected 0.95 +/- {band:.4f}"
    )


# --- fitting the substitution model as well as the branches --------------

# Deliberately asymmetric, matching tests/regression/test_gtr.py: no two
# exchangeabilities equal and no two frequencies equal, so a bug that
# collapsed either would show as a failure rather than as a coincidence.
_TRUE_EXCHANGEABILITIES = np.array([1.6, 0.4, 0.9, 0.7, 2.1, 1.0])
_TRUE_PI = np.array([0.35, 0.15, 0.30, 0.20])
_GTR_SITES = 20000


def _gtr_objective(
    fixture: str = SMALL_SITES, seed_offset: int = 0, sites: int = _GTR_SITES
) -> tuple[SubstitutionModelObjective, torch.Tensor]:
    params = load_fixture(fixture)
    rate = gtr_rate_matrix(_TRUE_EXCHANGEABILITIES, _TRUE_PI)
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=_TRUE_PI,
        seed=params.seed + 7919 * seed_offset,
        n_sites=sites,
        rate_matrix=rate,
    )
    objective = SubstitutionModelObjective(
        params.tau, params.k, dict(dataset.alignment)
    )
    truth = objective.theta_from_truth(params.tau, _TRUE_EXCHANGEABILITIES, _TRUE_PI)
    return objective, truth


def test_the_torch_rate_matrix_matches_the_numpy_one() -> None:
    # Two implementations of the same construction -- the differentiable
    # torch one the optimizer uses, and the numpy one the simulator uses --
    # so agreement is evidence rather than self-consistency. A drift between
    # them would fit one model to data generated under another.
    objective, truth = _gtr_objective(sites=200)
    assert_allclose(
        objective.rate_matrix(truth).detach().numpy(),
        gtr_rate_matrix(_TRUE_EXCHANGEABILITIES, _TRUE_PI),
        atol=1e-15,
    )


def test_the_starting_point_is_exactly_jukes_cantor() -> None:
    # Not approximately: the fit begins at the model the rest of this suite
    # validates, so any departure it reaches is something the data asked for.
    params = load_fixture(SMALL_SITES)
    objective, _ = _gtr_objective(sites=200)
    assert_allclose(
        objective.rate_matrix(objective.initial()).detach().numpy(),
        jc_rate_matrix(params.k),
        atol=1e-15,
    )


def test_the_substitution_model_theta_round_trips() -> None:
    objective, truth = _gtr_objective(sites=200)
    constrained = objective.constrain(truth)
    # The truth is rescaled so the last exchangeability is 1, which is the
    # same model -- test_gtr.py pins that invariance exactly.
    assert_allclose(
        constrained["exchangeabilities"].numpy(),
        _TRUE_EXCHANGEABILITIES / _TRUE_EXCHANGEABILITIES[-1],
        rtol=1e-13,
    )
    assert_allclose(constrained["pi"].numpy(), _TRUE_PI, rtol=1e-13)
    assert float(constrained["exchangeabilities"][-1]) == pytest.approx(1.0)


def test_the_substitution_model_has_one_parameter_per_free_quantity() -> None:
    objective, _ = _gtr_objective(sites=200)
    # 5 branches + 5 free exchangeabilities (of 6) + 3 free pi entries (of 4).
    assert objective.n_parameters == 13
    assert objective.parameter_names[-4:] == ["s4", "pi1", "pi2", "pi3"]


def test_the_substitution_model_gradient_matches_finite_differences() -> None:
    objective, truth = _gtr_objective(sites=500)
    assert_gradient_matches_finite_differences(
        objective, truth, _FINITE_DIFFERENCE_STEP, _RTOL_GRADIENT
    )


def test_the_substitution_model_fit_beats_the_generating_parameters() -> None:
    objective, truth = _gtr_objective()
    result = fit(objective)
    assert result.converged
    assert result.value < float(objective(truth))


def test_the_three_gauges_leave_a_well_conditioned_problem() -> None:
    # The direct test that the rate normalization, the pinned
    # exchangeability and the simplex gauge between them remove every flat
    # direction. Drop any one and this ratio collapses toward zero and
    # `parameter_covariance` refuses the fit.
    objective, _ = _gtr_objective()
    information = observed_information(objective, fit(objective).theta)
    eigenvalues = torch.linalg.eigvalsh(information)
    assert float(eigenvalues.min() / eigenvalues.max()) > 1e-4


def test_the_substitution_model_is_recovered_to_within_four_standard_errors() -> None:
    objective, _ = _gtr_objective()
    result = fit(objective)
    estimate = objective.constrain(result.theta)
    error = constrained_standard_errors(objective, result.theta)

    reference = _TRUE_EXCHANGEABILITIES / _TRUE_EXCHANGEABILITIES[-1]
    for name, fitted, spread, expected in (
        (
            "exchangeabilities",
            estimate["exchangeabilities"][:-1],
            error["exchangeabilities"][:-1],
            torch.as_tensor(reference[:-1]),
        ),
        ("pi", estimate["pi"], error["pi"], torch.as_tensor(_TRUE_PI)),
    ):
        deviation = (fitted - expected).abs() / spread
        assert float(deviation.max()) < 4.0, (
            f"{name}: worst {float(deviation.max()):.2f} standard errors from truth"
        )


def test_fitting_jc_simulated_data_recovers_a_jc_like_model() -> None:
    # A consistency check the other direction: given data generated under
    # Jukes-Cantor, the general model must not invent structure. Stated in
    # standard errors so it transfers if the fixture size changes.
    params = load_fixture(SMALL_SITES)
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        seed=params.seed,
        n_sites=_GTR_SITES,
    )
    objective = SubstitutionModelObjective(
        params.tau, params.k, dict(dataset.alignment)
    )
    result = fit(objective)
    estimate = objective.constrain(result.theta)
    error = constrained_standard_errors(objective, result.theta)

    free = estimate["exchangeabilities"][:-1]
    deviation = (free - torch.ones_like(free)).abs() / error["exchangeabilities"][:-1]
    assert float(deviation.max()) < 4.0

    uniform = torch.full((params.k,), 1.0 / params.k, dtype=torch.float64)
    pi_deviation = (estimate["pi"] - uniform).abs() / error["pi"]
    assert float(pi_deviation.max()) < 4.0


@pytest.mark.release
def test_substitution_model_intervals_cover_at_the_nominal_rate() -> None:
    reference = torch.as_tensor(_TRUE_EXCHANGEABILITIES / _TRUE_EXCHANGEABILITIES[-1])[
        :-1
    ]
    truth_pi = torch.as_tensor(_TRUE_PI)
    covered = 0
    total = 0
    for replicate in range(30):
        objective, _ = _gtr_objective(fixture=EIGHT_TAXA, seed_offset=replicate)
        result = fit(objective)
        assert result.converged
        estimate = objective.constrain(result.theta)
        error = constrained_standard_errors(objective, result.theta)
        hits = torch.cat(
            [
                covers(
                    estimate["exchangeabilities"][:-1],
                    error["exchangeabilities"][:-1],
                    reference,
                ),
                covers(estimate["pi"], error["pi"], truth_pi),
            ]
        )
        covered += int(hits.sum())
        total += hits.numel()

    rate = covered / total
    band = 3.0 * (0.95 * 0.05 / total) ** 0.5
    assert abs(rate - 0.95) <= band, (
        f"coverage {rate:.4f}, expected 0.95 +/- {band:.4f}"
    )


def test_fitted_tree_carries_the_fitted_lengths_back_onto_the_topology() -> None:
    # The inverse of theta_from_truth, and the form anything that draws or
    # serializes a fitted tree needs: pruning_torch keeps lengths out of the
    # Node structure, which is right for differentiation and wrong for
    # display.
    params = load_fixture(SMALL_SITES)
    objective = _objective(SMALL_SITES)
    theta = objective.theta_from_truth(params.tau)

    rebuilt = objective.fitted_tree(theta)

    assert_allclose(
        pruning_torch.branch_lengths_from_tree(rebuilt).numpy(),
        pruning_torch.branch_lengths_from_tree(params.tau).numpy(),
        rtol=1e-12,
    )
    assert rebuilt.branch_length is None
    assert [node.name for node in preorder(rebuilt)] == [
        node.name for node in preorder(params.tau)
    ]


def test_fitted_tree_halves_the_merged_root_pair() -> None:
    # The estimable quantity is the pair's sum; halving it is a drawing
    # convention and the docstring says so. Pinned because a reader of the
    # picture could otherwise take either half for an estimate.
    params = load_fixture(EIGHT_TAXA)
    objective = _objective(EIGHT_TAXA)

    rebuilt = objective.fitted_tree(objective.theta_from_truth(params.tau))

    halves = [child.branch_length for child in rebuilt.children]
    assert all(half is not None for half in halves)
    first, second = (float(half) for half in halves if half is not None)
    assert first == pytest.approx(second)

    original = [child.branch_length for child in params.tau.children]
    assert all(length is not None for length in original)
    assert first + second == pytest.approx(
        sum(float(length) for length in original if length is not None)
    )
