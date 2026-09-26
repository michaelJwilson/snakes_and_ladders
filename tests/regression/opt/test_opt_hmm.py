"""Regression tests for the discrete-HMM reference instance.

The forward recursion is checked against brute-force enumeration over every
hidden path -- the independent-oracle pattern
``sal.likelihood.brute_force`` establishes for pruning: a
recursion checked only against itself is checked against nothing.
``opt/CLAUDE.md``'s finite-difference derivative check is here too.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import pairwise, permutations, product

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from sal.backend import Backend
from sal.emissions import (
    BetaBinomialEmission,
    BinomialEmission,
    CategoricalEmission,
    EmissionFamily,
    GaussianEmission,
    NegativeBinomialEmission,
    PoissonEmission,
)
from sal.fixtures import load_params
from sal.likelihood.hmm import hmm_log_likelihood, viterbi
from sal.likelihood.hmm_paths import enumerate_hidden_paths
from sal.opt.em import EM, EmConfig
from sal.opt.hmm import (
    EmFit,
    GaussianHmmObjective,
    HmmObjective,
    align_states,
    baum_welch,
    baum_welch_family,
    forward_log_likelihood,
)
from sal.opt.termination import Stop, Termination
from sal.sim.fixtures import fixture
from sal.sim.hmm import HmmParams, simulate_sequences

from tests._fixtures import FIXTURES_DIR
from tests._objective_checks import assert_gradient_matches_finite_differences
from tests._rows import every_value

FIXTURE = FIXTURES_DIR / "hmm/ci.yaml"

# Relative throughout: the log-likelihood is a sum over sequences and sites
# (`DEV.md`, issue #111).
_RTOL_ORACLE = 1e-12
_RTOL_GRADIENT = 1e-6
_FINITE_DIFFERENCE_STEP = 1e-5

# Brute force is O(n_states ** length); 3 ** 7 = 2187 paths per sequence is
# the largest that stays a fast test, and the recursion is length-agnostic,
# so a longer chain would not exercise anything new.
_BRUTE_FORCE_SEQUENCES = 3
_BRUTE_FORCE_LENGTH = 7


def _brute_force_log_likelihood(
    observations: torch.Tensor,
    log_initial: torch.Tensor,
    log_transition: torch.Tensor,
    log_emission: torch.Tensor,
) -> float:
    """Sum over every hidden path, with no message passing anywhere."""
    n_states, length = log_initial.shape[0], observations.shape[1]
    total = 0.0
    for row in observations:
        weights = [
            log_initial[path[0]]
            + sum(log_transition[path[t], path[t + 1]] for t in range(length - 1))
            + sum(log_emission[path[t], row[t]] for t in range(length))
            for path in product(range(n_states), repeat=length)
        ]
        total += float(torch.logsumexp(torch.stack(weights), dim=0))
    return total


def _log_truth(params: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(  # type: ignore[return-value]
        torch.log(torch.as_tensor(part, dtype=torch.float64))
        for part in (
            params.initial,  # type: ignore[attr-defined]
            params.transition,  # type: ignore[attr-defined]
            params.emission,  # type: ignore[attr-defined]
        )
    )


@pytest.mark.oracle
def test_forward_matches_brute_force_path_enumeration() -> None:
    params = load_params(FIXTURE, HmmParams)
    observations = torch.as_tensor(
        simulate_sequences(params).observations[
            :_BRUTE_FORCE_SEQUENCES, :_BRUTE_FORCE_LENGTH
        ]
    )
    log_initial, log_transition, log_emission = _log_truth(params)

    expected = _brute_force_log_likelihood(
        observations, log_initial, log_transition, log_emission
    )
    actual = forward_log_likelihood(
        observations, log_initial, log_transition, log_emission
    )
    assert_allclose(float(actual), expected, rtol=_RTOL_ORACLE)


@pytest.mark.analytic
def test_the_likelihood_is_invariant_to_relabelling_the_hidden_states() -> None:
    # The identifiability caveat, asserted rather than only documented: a
    # recovery test that compared parameters without aligning the
    # permutation would fail on a correct fit.
    params = load_params(FIXTURE, HmmParams)
    observations = torch.as_tensor(simulate_sequences(params).observations[:20])
    log_initial, log_transition, log_emission = _log_truth(params)
    reference = float(
        forward_log_likelihood(observations, log_initial, log_transition, log_emission)
    )

    for order in permutations(range(params.n_states)):
        index = torch.as_tensor(order)
        permuted = float(
            forward_log_likelihood(
                observations,
                log_initial[index],
                log_transition[index][:, index],
                log_emission[index],
            )
        )
        assert_allclose(permuted, reference, rtol=_RTOL_ORACLE)


@pytest.mark.analytic
def test_gradient_matches_central_finite_differences() -> None:
    def check(at_truth: bool) -> None:
        params = load_params(FIXTURE, HmmParams)
        # A short slice: the finite-difference check costs two objective
        # evaluations per parameter, and the recursion it exercises is the same
        # at any length.
        objective = HmmObjective(
            simulate_sequences(params).observations[:40],
            params.n_states,
            params.n_symbols,
        )
        theta = (
            objective.theta_from_truth(
                params.initial, params.transition, params.emission
            )
            if at_truth
            else objective.initial()
        )

        assert_gradient_matches_finite_differences(
            objective, theta, _FINITE_DIFFERENCE_STEP, _RTOL_GRADIENT
        )

    every_value([True, False], check)


@pytest.mark.oracle
def test_theta_round_trips_through_the_constraint_map() -> None:
    params = load_params(FIXTURE, HmmParams)
    objective = HmmObjective(
        simulate_sequences(params).observations, params.n_states, params.n_symbols
    )
    constrained = objective.constrain(
        objective.theta_from_truth(params.initial, params.transition, params.emission)
    )
    assert_allclose(
        torch.exp(constrained["log_initial"]).numpy(), params.initial, rtol=1e-13
    )
    assert_allclose(
        torch.exp(constrained["log_transition"]).numpy(), params.transition, rtol=1e-13
    )
    assert_allclose(
        torch.exp(constrained["log_emission"]).numpy(), params.emission, rtol=1e-13
    )


@pytest.mark.smoke
def test_theta_has_one_entry_per_free_probability() -> None:
    # 2 free initial + 3 rows x 2 free transition + 3 rows x 3 free emission.
    objective = HmmObjective(np.zeros((2, 5), dtype=np.int64), n_states=3, n_symbols=4)
    assert objective.n_parameters == 17
    assert objective.initial().shape == (17,)


@pytest.mark.smoke
def test_the_initial_point_is_uninformative_but_not_symmetric() -> None:
    # Initial and transition start uniform; the emission rows are tilted
    # apart. See the stationary-point test below for why the tilt has to be
    # there.
    objective = HmmObjective(np.zeros((2, 5), dtype=np.int64), n_states=3, n_symbols=4)
    constrained = objective.constrain(objective.initial())
    assert_allclose(
        torch.exp(constrained["log_initial"]).numpy(), np.full(3, 1 / 3), rtol=1e-14
    )
    assert_allclose(
        torch.exp(constrained["log_transition"]).numpy(),
        np.full((3, 3), 1 / 3),
        rtol=1e-14,
    )
    emission = torch.exp(constrained["log_emission"])
    assert_allclose(emission.sum(dim=1).numpy(), np.ones(3), rtol=1e-14)
    # Every state favours a different symbol, so no two rows agree.
    assert emission[0].argmax() != emission[1].argmax()
    assert emission[1].argmax() != emission[2].argmax()


@pytest.mark.analytic
def test_the_uniform_point_is_a_stationary_point_of_the_likelihood() -> None:
    # With every hidden state identical no change to the initial or transition
    # parameters moves the likelihood; a fit started there keeps one state.
    params = load_params(FIXTURE, HmmParams)
    objective = HmmObjective(
        simulate_sequences(params).observations[:40], params.n_states, params.n_symbols
    )
    uniform = torch.zeros(objective.n_parameters, dtype=torch.float64)

    point = uniform.detach().clone().requires_grad_(True)
    gradient = torch.autograd.grad(objective(point), point)[0]

    n_free_initial = params.n_states - 1
    n_free_transition = params.n_states * (params.n_states - 1)
    blocked = gradient[: n_free_initial + n_free_transition]
    assert_allclose(blocked.numpy(), np.zeros(blocked.numel()), atol=1e-12)
    # The emission block is not flat, which is why the fit appears to make
    # progress while the states stay exchangeable.
    assert float(gradient[n_free_initial + n_free_transition :].abs().max()) > 1.0


@pytest.mark.analytic
def test_baum_welch_increases_the_likelihood_monotonically() -> None:
    # An exact property of EM, not an empirical one: each iteration
    # maximizes a lower bound that is tight at the current parameters, so
    # the likelihood cannot decrease. A violation means the M step is wrong.
    params = load_params(FIXTURE, HmmParams)
    observations = simulate_sequences(params).observations[:60]
    objective = HmmObjective(observations, params.n_states, params.n_symbols)
    start = objective.constrain(objective.initial())

    previous = -float("inf")
    log_initial = start["log_initial"]
    log_transition = start["log_transition"]
    log_emission = start["log_emission"]
    for _ in range(8):
        log_initial, log_transition, log_emission, value, _ = baum_welch(
            observations,
            log_initial,
            log_transition,
            log_emission,
            config=replace(EM, max_iterations=1),
        )
        assert value >= previous - 1e-9 * abs(value)
        previous = value


@pytest.mark.oracle
def test_align_states_recovers_a_known_permutation() -> None:
    params = load_params(FIXTURE, HmmParams)
    emission = torch.as_tensor(params.emission)
    order = (2, 0, 1)
    permuted = torch.log(emission[list(order)])
    # align_states returns the order that maps the permuted matrix back.
    recovered = align_states(permuted, emission)
    assert_allclose(
        torch.exp(permuted)[list(recovered)].numpy(), emission.numpy(), atol=1e-15
    )


@pytest.mark.smoke
def test_baum_welch_stops_once_the_likelihood_stops_moving() -> None:
    # The convergence test is relative to the log-likelihood's magnitude, as
    # everywhere else. A loose tolerance must stop the iteration early, which
    # shows as a worse optimum than a tight one reaches from the same start.
    params = load_params(FIXTURE, HmmParams)
    observations = simulate_sequences(params).observations[:60]
    objective = HmmObjective(observations, params.n_states, params.n_symbols)
    start = objective.constrain(objective.initial())
    arguments = (
        observations,
        start["log_initial"],
        start["log_transition"],
        start["log_emission"],
    )

    loose = baum_welch(*arguments, config=replace(EM, tolerance=1e-1)).log_likelihood
    tight = baum_welch(*arguments, config=replace(EM, tolerance=1e-14)).log_likelihood
    assert loose < tight


#: The enumerable slice of the declared instance: 3 ** 7 = 2,187 paths per
#: sequence over 20 sequences. The instance itself is 15 positions by 600
#: sequences and is past enumeration, so the oracle is taken on a slice of it
#: rather than on a different model (issue #393).
_EM_LENGTH = 7
_EM_SEQUENCES = 20
#: Iterates whose evidence is recomputed from the enumeration. Three is enough
#: to see the increase; each one costs 20 path enumerations.
_EM_ITERATES = 3


def _enumerated_evidence(params: HmmParams, observations: np.ndarray) -> float:
    """``sum_sequences log P(sequence)``, summed over every path of each."""
    return float(
        sum(
            enumerate_hidden_paths(params, sequence).log_likelihood
            for sequence in observations
        )
    )


def _stepped(params: HmmParams, fitted: EmFit) -> HmmParams:
    """``params`` carrying what one Baum-Welch run returned."""
    return replace(
        params,
        initial=torch.exp(fitted.log_initial).numpy(),
        transition=torch.exp(fitted.log_transition).numpy(),
        emissions=fitted.components,
    )


@pytest.mark.oracle
def test_baum_welch_reaches_the_enumerated_path_evidence_and_its_fixed_point() -> None:
    # Path enumeration is the E step's sum with no recursion. Realized: equal
    # at the start (0.0); the enumerated evidence rises over three iterates,
    # -179.735, -178.981, -178.497; at convergence 8.5e-13 (one M step of lag);
    # fitted initial = enumerated first-site posterior to 2.2e-12. `lengths`
    # is the batch's only declared shape (#666).
    params = replace(
        fixture("hmm", "ci").params,
        lengths=(_EM_LENGTH,) * _EM_SEQUENCES,
    )
    observations = simulate_sequences(params).observations
    start = (
        torch.log(torch.as_tensor(params.initial)),
        torch.log(torch.as_tensor(params.transition)),
        params.emissions,
    )

    first = baum_welch_family(
        observations, *start, config=replace(EM, max_iterations=1)
    )
    assert_allclose(
        first.log_likelihood,
        _enumerated_evidence(params, observations),
        rtol=_RTOL_ORACLE,
    )

    walked = start
    evidence = []
    for _ in range(_EM_ITERATES):
        iterate = baum_welch_family(
            observations, *walked, config=replace(EM, max_iterations=1)
        )
        walked = (iterate.log_initial, iterate.log_transition, iterate.components)
        evidence.append(_enumerated_evidence(_stepped(params, iterate), observations))
    assert evidence == sorted(evidence), evidence

    fitted = baum_welch_family(observations, *start)
    settled = _stepped(params, fitted)
    assert_allclose(
        _enumerated_evidence(settled, observations),
        fitted.log_likelihood,
        rtol=1e-11,
    )
    assert_allclose(
        torch.exp(fitted.log_initial).numpy(),
        np.stack(
            [
                enumerate_hidden_paths(settled, sequence).posterior[0]
                for sequence in observations
            ]
        ).mean(axis=0),
        atol=1e-10,
    )


#: Iterations the ascent is read over. Twelve is past the point where the
#: increments have fallen by an order of magnitude, so a non-monotone step
#: would have shown.
_EM_ASCENT = 12

#: Absolute, over probabilities in [0, 1]: a fitted entry of 1e-10 makes a
#: relative bound on the re-estimation residual a statement about a number
#: the data does not identify (`DEV.md`, issue #111).
_ATOL_FIXED_POINT = 1e-9


def _re_estimated(
    observations: np.ndarray,
    log_initial: torch.Tensor,
    log_transition: torch.Tensor,
    log_emission: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Baum's re-estimation equations, written out in NumPy over the definition.

    Probability-space messages per sequence and ratios of expected counts; not `opt.hmm`.
    """
    initial = np.exp(log_initial.numpy())
    transition = np.exp(log_transition.numpy())
    emission = np.exp(log_emission.numpy())
    n_states, n_symbols = emission.shape
    length = observations.shape[1]

    start = np.zeros(n_states)
    pairs = np.zeros((n_states, n_states))
    left = np.zeros(n_states)
    emitted = np.zeros((n_states, n_symbols))
    occupancy = np.zeros(n_states)
    evidence = 0.0
    for row in observations:
        alpha = np.zeros((length, n_states))
        beta = np.zeros((length, n_states))
        alpha[0] = initial * emission[:, row[0]]
        for t in range(1, length):
            alpha[t] = (alpha[t - 1] @ transition) * emission[:, row[t]]
        beta[length - 1] = 1.0
        for t in range(length - 2, -1, -1):
            beta[t] = transition @ (emission[:, row[t + 1]] * beta[t + 1])
        total = float(alpha[length - 1].sum())
        evidence += float(np.log(total))

        gamma = alpha * beta / total
        start += gamma[0]
        for t in range(length - 1):
            pairs += (
                alpha[t][:, None]
                * transition
                * (emission[:, row[t + 1]] * beta[t + 1])[None, :]
            ) / total
        left += gamma[:-1].sum(axis=0)
        for t in range(length):
            emitted[:, row[t]] += gamma[t]
        occupancy += gamma.sum(axis=0)

    return (
        start / observations.shape[0],
        pairs / left[:, None],
        emitted / occupancy[:, None],
        evidence,
    )


@pytest.mark.oracle
@pytest.mark.critical
def test_baum_welch_ascends_and_settles_on_the_re_estimation_equations() -> None:
    # Outside the ladder (#734): EM's own properties against `_re_estimated`.
    # On the 20-sequence, length-7 slice: twelve iterations -181.992 to
    # -175.403, each rise positive (smallest 0.314 nats); evidence -173.0932201685
    # against -173.0932201686 reported, 8.5e-13 (1e-11 declared); the equations
    # return the fit to 2.2e-12, 4.1e-11, 4.3e-11 (1e-9): the stopping residue.
    params = replace(
        fixture("hmm", "ci").params,
        lengths=(_EM_LENGTH,) * _EM_SEQUENCES,
    )
    observations = simulate_sequences(params).observations
    start = (
        torch.log(torch.as_tensor(params.initial)),
        torch.log(torch.as_tensor(params.transition)),
        torch.log(torch.as_tensor(params.emission)),
    )

    walked = start
    reported = []
    for _ in range(_EM_ASCENT):
        initial_step, transition_step, emission_step, likelihood, _ = baum_welch(
            observations, *walked, config=replace(EM, max_iterations=1)
        )
        walked = (initial_step, transition_step, emission_step)
        reported.append(likelihood)
    increments = np.diff(np.array(reported))
    assert (increments > 0.0).all(), reported

    log_initial, log_transition, log_emission, log_likelihood, _ = baum_welch(
        observations, *start
    )
    initial, transition, emission, evidence = _re_estimated(
        observations, log_initial, log_transition, log_emission
    )
    assert_allclose(evidence, log_likelihood, rtol=1e-11)
    assert_allclose(torch.exp(log_initial).numpy(), initial, atol=_ATOL_FIXED_POINT)
    assert_allclose(
        torch.exp(log_transition).numpy(), transition, atol=_ATOL_FIXED_POINT
    )
    assert_allclose(torch.exp(log_emission).numpy(), emission, atol=_ATOL_FIXED_POINT)


@pytest.mark.oracle
def test_the_streamed_baum_welch_is_the_batched_one() -> None:
    # Issue #986: the compiled route streams scaled messages into expected
    # counts; the log-space batch in `baum_welch_family` is its oracle, over
    # ten iterations from a perturbed start on the fixture's model.
    def check(n_sequences: int) -> None:
        params = replace(
            load_params(FIXTURES_DIR / "hmm" / "ci.yaml", HmmParams),
            lengths=(60,) * n_sequences,
        )
        observations = simulate_sequences(params).observations
        rng = np.random.default_rng(986)
        draws = [rng.random(shape) + 0.5 for shape in ((3,), (3, 3), (3, 4))]
        initial, transition, emission = (
            torch.log(torch.as_tensor(d / d.sum(-1, keepdims=True))) for d in draws
        )
        fits = [
            baum_welch(
                observations,
                initial,
                transition,
                emission,
                backend=backend,
                config=EmConfig(max_iterations=10, tolerance=-np.inf),
            )
            for backend in (Backend.PYTHON, Backend.RUST)
        ]
        for name in ("log_initial", "log_transition", "log_emission"):
            assert_allclose(
                getattr(fits[1], name).numpy(),
                getattr(fits[0], name).numpy(),
                rtol=0.0,
                atol=1e-10,
            )
        assert_allclose(fits[1].log_likelihood, fits[0].log_likelihood, rtol=1e-12)
        # Issue #1059: both routes report the loop's termination, as every EM
        # fit does; ten steps at a tolerance nothing meets is the budget.
        for fit in fits:
            assert fit.termination == Termination(False, 10, Stop.BUDGET)

    every_value([1, 40], check)


def _count_chain(
    n_sequences: int, length: int, seed: int
) -> tuple[np.random.Generator, np.ndarray]:
    """Hidden states of a sticky three-state chain, for the streamed-family pins."""
    rng = np.random.default_rng(seed)
    cumulative = np.array(
        [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.05, 0.15, 0.8]]
    ).cumsum(axis=1)
    states = np.empty((n_sequences, length), dtype=np.int64)
    states[:, 0] = rng.choice(3, size=n_sequences)
    for t in range(1, length):
        above = rng.random(n_sequences)[:, None] > cumulative[states[:, t - 1]]
        states[:, t] = above.sum(axis=1)
    return rng, states


def _streamed_case(
    name: str, covariate: bool
) -> tuple[np.ndarray, EmissionFamily, np.ndarray | None]:
    """Observations, a start and any covariate for one family (issue #997)."""
    rng, states = _count_chain(40, 60, 997)
    depth = rng.integers(5, 40, size=states.shape) if covariate else None
    if name == "gaussian":
        values = rng.normal(np.array([-2.0, 0.0, 3.0])[states], 1.0)
        return values, GaussianEmission([-1.5, 0.5, 2.5], [1.2, 1.0, 1.4], 1e-12), None
    if name == "poisson":
        family: EmissionFamily = PoissonEmission([2.0, 4.0, 10.0])
        return rng.poisson(np.array([1.0, 5.0, 12.0])[states]), family, None
    if name == "negative_binomial":
        r, mu = np.array([2.0, 5.0, 10.0])[states], np.array([1.0, 5.0, 12.0])[states]
        mu = mu if depth is None else mu * depth / 20.0
        family = NegativeBinomialEmission([1.0, 3.0, 5.0], [2.0, 4.0, 10.0])
        return rng.negative_binomial(r, r / (r + mu)), family, depth
    if name == "binomial":
        family = BinomialEmission([30.0] * 3, [0.3, 0.5, 0.7])
        return rng.binomial(30, np.array([0.2, 0.5, 0.8])[states]), family, None
    p = rng.beta(np.array([2.0, 5.0, 8.0])[states], np.array([8.0, 5.0, 2.0])[states])
    trials = 30 if depth is None else depth
    family = BetaBinomialEmission([30.0] * 3, [1.5, 4.0, 6.0], [6.0, 4.0, 1.5])
    return rng.binomial(trials, p), family, depth


@pytest.mark.oracle
@pytest.mark.parametrize(
    ("name", "covariate"),
    [
        ("gaussian", False),
        ("poisson", False),
        ("negative_binomial", False),
        ("negative_binomial", True),
        ("binomial", False),
        ("beta_binomial", False),
        ("beta_binomial", True),
    ],
)
def test_the_streamed_family_step_is_the_batched_one(
    name: str, covariate: bool
) -> None:
    # Issue #997: a one-channel Gaussian streams moments, a count family
    # streams its posterior weight on each occupied (count, covariate) cell
    # and re-estimates on those; the batched log-space route is the oracle
    # over ten iterations from a start away from the truth.
    observations, family, depth = _streamed_case(name, covariate)
    initial = torch.log(torch.tensor([0.4, 0.3, 0.3], dtype=torch.float64))
    transition = torch.log(
        torch.tensor(
            [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]], dtype=torch.float64
        )
    )
    fits = [
        baum_welch_family(
            observations,
            initial,
            transition,
            family,
            covariate=depth,
            backend=backend,
            config=EmConfig(max_iterations=10, tolerance=-np.inf),
        )
        for backend in (Backend.PYTHON, Backend.RUST)
    ]
    if name != "gaussian":
        # Every observation scored rather than each cell, the three cells
        # that repeat most tabled and the rest scored, and each with the
        # Stirling lgamma from 10 (#997).
        fits.extend(
            baum_welch_family(
                observations,
                initial,
                transition,
                family,
                covariate=depth,
                with_table=with_table,
                table_size=size,
                approx=approx,
                config=EmConfig(max_iterations=10, tolerance=-np.inf),
            )
            for with_table, size, approx in (
                (False, None, False),
                (True, 3, False),
                (True, None, True),
                (False, None, True),
            )
        )
    assert type(fits[1].components) is type(family)
    for streamed in fits[1:]:
        for name_ in ("log_initial", "log_transition"):
            assert_allclose(
                getattr(streamed, name_).numpy(),
                getattr(fits[0], name_).numpy(),
                rtol=0.0,
                atol=1e-10,
            )
        for key, value in fits[0].components.named_parameters().items():
            assert_allclose(
                streamed.components.named_parameters()[key].numpy(),
                value.numpy(),
                rtol=1e-9,
                err_msg=key,
            )
        assert_allclose(streamed.log_likelihood, fits[0].log_likelihood, rtol=1e-12)
        assert streamed.at_boundary == fits[0].at_boundary


@pytest.mark.oracle
def test_real_valued_counts_take_the_batched_route() -> None:
    # Float counts take the batched route (#997), the compiled ragged E step
    # (#933): 2 ulps (3.0e-16) from torch, so the declared tolerance.
    observations, family, _ = _streamed_case("poisson", covariate=False)
    initial = torch.log(torch.full((3,), 1.0 / 3.0, dtype=torch.float64))
    transition = torch.log(torch.full((3, 3), 1.0 / 3.0, dtype=torch.float64))
    fits = [
        baum_welch_family(
            observations.astype(np.float64),
            initial,
            transition,
            family,
            backend=backend,
            config=EmConfig(max_iterations=3, tolerance=-np.inf),
        )
        for backend in (Backend.PYTHON, Backend.RUST)
    ]
    assert_allclose(fits[1].log_likelihood, fits[0].log_likelihood, rtol=1e-12)


@pytest.mark.oracle
@pytest.mark.parametrize(
    "name", ["gaussian", "poisson", "negative_binomial", "binomial", "beta_binomial"]
)
def test_the_compiled_viterbi_is_the_numpy_one(name: str) -> None:
    # Issue #997: the compiled route decodes sequences in parallel; the NumPy
    # recursion is its oracle, path for path and in the total log-probability.
    observations, family, _ = _streamed_case(name, covariate=False)
    initial = torch.log(torch.tensor([0.4, 0.3, 0.3], dtype=torch.float64))
    transition = torch.log(
        torch.tensor(
            [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]], dtype=torch.float64
        )
    )
    ours, theirs = (
        viterbi(observations, initial, transition, family, backend=backend)
        for backend in (Backend.RUST, Backend.PYTHON)
    )
    assert_allclose(ours[1], theirs[1], rtol=1e-12)
    # The compiled lgamma differs from torch's in the last bits, which can
    # flip a tie between two paths: a path that differs must score, under
    # the oracle's own densities, what the oracle's path scores.
    emit = family.log_density(
        torch.as_tensor(observations, dtype=family.observation_dtype)
    ).numpy()
    for row in np.flatnonzero((ours[0] != theirs[0]).any(axis=1)):
        scored = [
            float(
                initial[path[0]]
                + emit[row, np.arange(path.size), path].sum()
                + transition.numpy()[path[:-1], path[1:]].sum()
            )
            for path in (ours[0][row], theirs[0][row])
        ]
        assert_allclose(scored[0], scored[1], rtol=1e-12)


@pytest.mark.oracle
@pytest.mark.parametrize("backend", [Backend.RUST, Backend.PYTHON])
def test_viterbi_is_the_enumerated_best_path(backend: Backend) -> None:
    # Every path of four sequences of five over three states, scored by the
    # model directly: the decoded path is the argmax and its score the max.
    params = load_params(FIXTURES_DIR / "hmm" / "ci.yaml", HmmParams)
    observations = simulate_sequences(replace(params, lengths=(5,) * 4)).observations
    log_initial = torch.log(torch.as_tensor(params.initial))
    log_transition = torch.log(torch.as_tensor(params.transition))
    log_emission = torch.log(torch.as_tensor(params.emission))
    states, total = viterbi(
        observations,
        log_initial,
        log_transition,
        CategoricalEmission.from_log(log_emission),
        backend=backend,
    )
    best_total = 0.0
    for row, path in zip(observations, states, strict=True):
        scores = {
            candidate: float(
                log_initial[candidate[0]]
                + sum(log_transition[a, b] for a, b in pairwise(candidate))
                + sum(
                    log_emission[s, int(x)] for s, x in zip(candidate, row, strict=True)
                )
            )
            for candidate in product(range(3), repeat=5)
        }
        best = max(scores, key=lambda c: (scores[c], [-s for s in c]))
        assert tuple(path) == best
        best_total += scores[best]
    assert_allclose(total, best_total, rtol=1e-12)


@pytest.mark.oracle
@pytest.mark.parametrize(
    "name", ["gaussian", "poisson", "negative_binomial", "binomial", "beta_binomial"]
)
def test_the_compiled_hmm_score_is_the_forward_recursion(name: str) -> None:
    # Issue #997: the scaled forward pass over sequences in parallel against
    # the log-space recursion in torch, at the declared float64 tolerance.
    observations, family, _ = _streamed_case(name, covariate=False)
    initial = torch.log(torch.tensor([0.4, 0.3, 0.3], dtype=torch.float64))
    transition = torch.log(
        torch.tensor(
            [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]], dtype=torch.float64
        )
    )
    ours, oracle = (
        hmm_log_likelihood(observations, initial, transition, family, backend=backend)
        for backend in (Backend.RUST, Backend.PYTHON)
    )
    assert_allclose(ours, oracle, rtol=1e-12)


@pytest.mark.oracle
def test_the_gaussian_hmm_gradient_is_autograds() -> None:
    # Issue #997: `GaussianHmmObjective.gradient` by Fisher's identity from
    # one streamed pass of expected statistics; autograd through `__call__`
    # is the oracle, at points away from the start in every coordinate.
    observations, _, _ = _streamed_case("gaussian", covariate=False)
    objective = GaussianHmmObjective(observations, 3)
    rng = np.random.default_rng(997)
    for _ in range(3):
        theta = objective.initial() + 0.3 * torch.as_tensor(
            rng.normal(size=objective.n_parameters)
        )
        point = theta.clone().requires_grad_(True)
        (autograd,) = torch.autograd.grad(objective(point), point)
        assert_allclose(
            objective.gradient(theta).numpy(),
            autograd.numpy(),
            rtol=1e-10,
            atol=1e-10 * float(autograd.abs().max()),
        )
