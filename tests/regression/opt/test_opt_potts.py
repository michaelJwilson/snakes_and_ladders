"""Regression tests for the Potts-chain reference instance.

Per root ``CLAUDE.md`` ("Pin to Independent Sources"), the transfer-matrix
normalizer is checked against brute-force enumeration over every
configuration -- a computation that shares no code with the recursion under
test -- and the objective's sufficient-statistic shortcut is checked against
a naive per-chain sum. ``opt/CLAUDE.md`` makes the finite-difference
derivative check mandatory; it is here, not deferred to the optimizer.

The normalizer has two routes -- the per-site recursion and the same product
reassociated by repeated squaring (issue #754) -- and every claim above is
asserted of both, because ``log_partition`` picks between them on ``q`` and
``length`` and a fixture exercises whichever its own size selects. The two are
pinned against each other, and the squaring route against the rung below it in
``infra/ladder.py``: ``likelihood.potts.strip_log_partition`` on a strip one
site wide.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from snakes_and_ladders.likelihood.device import CROSS_DEVICE_RTOL_FLOAT64
from snakes_and_ladders.likelihood.potts import strip_log_partition
from snakes_and_ladders.opt.constrain import log_simplex
from snakes_and_ladders.opt.potts import (
    PottsObjective,
    log_partition,
    log_partition_by_recursion,
    log_partition_by_squaring,
    squaring_is_cheaper,
)
from snakes_and_ladders.sim.graph import BoundaryCondition
from snakes_and_ladders.sim.potts_chain import load_potts_params, simulate_chains

from tests._fixtures import FIXTURES_DIR
from tests._objective_checks import assert_gradient_matches_finite_differences

FIXTURE = FIXTURES_DIR / "potts_chain/ci.yaml"

#: ``(q, length)`` either side of :func:`squaring_is_cheaper`'s rule: it takes
#: the squaring route at (2, 8), (3, 2), (3, 16) and (3, 64) and declines it at
#: (3, 3) and (3, 8), so both branches of ``log_partition`` are exercised.
ROUTE_CASES = ((2, 8), (3, 2), (3, 3), (3, 8), (3, 16), (3, 64))

#: Both routes of the normalizer, named so a test says which it ran.
ROUTES = (log_partition_by_recursion, log_partition_by_squaring)

# Tolerances are relative throughout, per `DEV.md`: the log-likelihood is a
# sum over chains and sites, so an absolute bound fixed at one fixture size
# does not transfer to another (issue #111).
_RTOL_ORACLE = 1e-12
_RTOL_GRADIENT = 1e-6
_FINITE_DIFFERENCE_STEP = 1e-5


def _brute_force_log_partition(
    coupling: float, field: np.ndarray, length: int
) -> float:
    """``log Z`` by summing every one of ``q ** length`` configurations."""
    weights = [
        coupling * sum(1 for i in range(length - 1) if s[i] == s[i + 1])
        + float(field[list(s)].sum())
        for s in product(range(field.shape[0]), repeat=length)
    ]
    peak = max(weights)
    return peak + float(np.log(np.exp(np.asarray(weights) - peak).sum()))


@pytest.mark.oracle
@pytest.mark.parametrize("route", [*ROUTES, log_partition], ids=lambda f: f.__name__)
@pytest.mark.parametrize("length", [1, 2, 3, 5, 8])
def test_transfer_matrix_matches_brute_force_enumeration(
    length: int, route: object
) -> None:
    params = load_potts_params(FIXTURE)
    expected = _brute_force_log_partition(params.coupling, params.field, length)
    actual = route(  # type: ignore[operator]
        torch.tensor(params.coupling, dtype=torch.float64),
        torch.as_tensor(params.field, dtype=torch.float64),
        length,
    )
    assert_allclose(float(actual), expected, rtol=_RTOL_ORACLE)


def _gauge_fixed(values: np.ndarray) -> np.ndarray:
    """``h`` in the gauge the model is identified in, ``logsumexp(h) == 0``."""
    return values - float(np.log(np.exp(values).sum()))


@pytest.mark.oracle
@pytest.mark.patch
@pytest.mark.parametrize(("n_states", "length"), ROUTE_CASES)
def test_squaring_reproduces_the_per_site_recursion(n_states: int, length: int) -> None:
    # The reassociation is exact and changes only the order the log-space
    # sums are taken in, so the recursion is the referee and bitwise is the
    # target (root `CLAUDE.md`). It is reached wherever the two orders
    # coincide -- every chain whose exponent is 0 or 1 runs the same
    # sequence of operations on both routes -- and where they do not, the
    # guard is the declared cross-device tolerance and the realized
    # difference is asserted to sit two orders inside it rather than at it.
    rng = np.random.default_rng(11)
    field = torch.as_tensor(_gauge_fixed(rng.normal(size=n_states)))
    coupling = torch.tensor(0.6, dtype=torch.float64)

    recursion = log_partition_by_recursion(coupling, field, length)
    squaring = log_partition_by_squaring(coupling, field, length)

    if length <= 2:
        assert float(squaring) == float(recursion)
    realized = abs(float(squaring) - float(recursion)) / abs(float(recursion))
    assert realized < 1e-13
    assert_allclose(float(squaring), float(recursion), rtol=CROSS_DEVICE_RTOL_FLOAT64)


@pytest.mark.smoke
@pytest.mark.parametrize(("n_states", "length"), ROUTE_CASES)
def test_the_route_takes_no_more_logsumexp_calls_than_the_recursion(
    n_states: int, length: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # What `squaring_is_cheaper` claims, counted rather than argued: where it
    # says yes, the route it picks makes no more `logsumexp` calls and touches
    # no more elements than the recursion, so it cannot lose on either term.
    # Where it says no, the recursion is what the function always did.
    calls: dict[str, list[int]] = {}
    real = torch.logsumexp

    def _counted(values: torch.Tensor, dim: object) -> torch.Tensor:
        counted = calls.setdefault("counted", [])
        counted.append(values.numel())
        return real(values, dim)  # type: ignore[arg-type]

    monkeypatch.setattr(torch, "logsumexp", _counted)
    field = torch.zeros(n_states, dtype=torch.float64)
    coupling = torch.tensor(0.6, dtype=torch.float64)

    log_partition(coupling, field, length)
    chosen = calls.pop("counted", [])
    log_partition_by_recursion(coupling, field, length)
    recursion = calls.pop("counted", [])
    log_partition_by_squaring(coupling, field, length)
    squaring = calls.pop("counted", [])

    if squaring_is_cheaper(n_states, length):
        assert chosen == squaring
        assert len(squaring) <= len(recursion)
        assert sum(squaring) <= sum(recursion)
    else:
        assert chosen == recursion
        # Declined for a reason, and this is it: one of the two terms is
        # worse, so the rule is not conservative to the point of vacuity.
        assert len(squaring) > len(recursion) or sum(squaring) > sum(recursion)


@pytest.mark.oracle
@pytest.mark.parametrize("length", [2, 5, 16, 64])
def test_squaring_is_the_transfer_matrix_on_a_strip_one_site_wide(
    length: int,
) -> None:
    # The rung below in `infra/ladder.py`. A chain of `length` sites is an
    # open `length x 1` strip, where `strip_log_partition` transfers a whole
    # column and shares no code with either route here -- so it referees the
    # squaring at lengths brute force cannot reach.
    params = load_potts_params(FIXTURE)
    expected = strip_log_partition(
        (length, 1), BoundaryCondition.OPEN, params.coupling, params.field
    )
    actual = log_partition_by_squaring(
        torch.tensor(params.coupling, dtype=torch.float64),
        torch.as_tensor(params.field, dtype=torch.float64),
        length,
    )
    assert_allclose(float(actual), expected, rtol=CROSS_DEVICE_RTOL_FLOAT64)


@pytest.mark.analytic
@pytest.mark.parametrize("coupling", [-0.8, 0.0, 1.4])
@pytest.mark.parametrize("length", [3, 16, 64])
def test_the_squaring_gradient_matches_central_finite_differences(
    coupling: float, length: int
) -> None:
    # Autograd follows the reassociation, and what says so is the derivative
    # of the reassociated product against central differences of its own
    # value -- at lengths either side of the route rule and at couplings of
    # both signs, since the transfer matrix's off-diagonal is where the sign
    # enters.
    field = torch.as_tensor(
        _gauge_fixed(np.random.default_rng(3).normal(size=3))
    ).requires_grad_(True)
    j = torch.tensor(coupling, dtype=torch.float64, requires_grad=True)

    gradient = torch.autograd.grad(log_partition_by_squaring(j, field, length), [j])[0]

    step = 1e-6
    with torch.no_grad():
        up = float(log_partition_by_squaring(j + step, field.detach(), length))
        down = float(log_partition_by_squaring(j - step, field.detach(), length))
    assert_allclose(float(gradient), (up - down) / (2 * step), rtol=1e-6)


@pytest.mark.analytic
@pytest.mark.parametrize("shift", [-1.5, 0.75])
def test_log_partition_shifts_exactly_with_the_field_gauge(shift: float) -> None:
    # Every configuration occupies all `length` sites, so adding a constant
    # to the field multiplies every weight by exp(length * shift). This is an
    # exact analytic identity, and it is why the field has to be gauge-fixed
    # before a fitted value means anything.
    params = load_potts_params(FIXTURE)
    coupling = torch.tensor(params.coupling, dtype=torch.float64)
    field = torch.as_tensor(params.field, dtype=torch.float64)
    length = 6
    assert_allclose(
        float(log_partition(coupling, field + shift, length)),
        float(log_partition(coupling, field, length)) + length * shift,
        rtol=_RTOL_ORACLE,
    )


@pytest.mark.oracle
def test_objective_matches_a_naive_per_chain_log_likelihood() -> None:
    # The objective reduces the data to two sufficient statistics up front.
    # That shortcut is the kind of thing that is right until someone changes
    # the model, so it is pinned against the definition it claims to equal.
    params = load_potts_params(FIXTURE)
    chains = simulate_chains(params)
    objective = PottsObjective(chains, params.n_states)
    theta = objective.theta_from_truth(params.coupling, params.field)

    log_z = float(
        log_partition(
            torch.tensor(params.coupling, dtype=torch.float64),
            torch.as_tensor(params.field, dtype=torch.float64),
            params.chain_length,
        )
    )
    naive = sum(
        params.coupling * int((chain[:-1] == chain[1:]).sum())
        + float(params.field[chain].sum())
        - log_z
        for chain in chains
    )
    assert_allclose(float(objective(theta)), -naive, rtol=_RTOL_ORACLE)


@pytest.mark.analytic
@pytest.mark.parametrize("at_truth", [True, False])
def test_gradient_matches_central_finite_differences(at_truth: bool) -> None:
    params = load_potts_params(FIXTURE)
    objective = PottsObjective(simulate_chains(params), params.n_states)
    theta = (
        objective.theta_from_truth(params.coupling, params.field)
        if at_truth
        else objective.initial()
    )

    assert_gradient_matches_finite_differences(
        objective, theta, _FINITE_DIFFERENCE_STEP, _RTOL_GRADIENT
    )


@pytest.mark.oracle
def test_theta_round_trips_through_the_constraint_map() -> None:
    params = load_potts_params(FIXTURE)
    objective = PottsObjective(simulate_chains(params), params.n_states)
    constrained = objective.constrain(
        objective.theta_from_truth(params.coupling, params.field)
    )
    assert_allclose(float(constrained["coupling"]), params.coupling, rtol=1e-14)
    assert_allclose(constrained["field"].numpy(), params.field, rtol=1e-14)


@pytest.mark.smoke
def test_the_initial_point_is_a_uniform_field_and_no_coupling() -> None:
    objective = PottsObjective(np.zeros((2, 4), dtype=np.int64), n_states=3)
    constrained = objective.constrain(objective.initial())
    assert float(constrained["coupling"]) == 0.0
    assert_allclose(
        torch.exp(constrained["field"]).numpy(), np.full(3, 1 / 3), rtol=1e-14
    )


@pytest.mark.smoke
def test_the_loader_canonicalizes_the_field_gauge() -> None:
    params = load_potts_params(FIXTURE)
    assert_allclose(float(np.exp(params.field).sum()), 1.0, rtol=1e-14)


@pytest.mark.smoke
def test_simulated_chains_have_the_declared_shape_and_alphabet() -> None:
    params = load_potts_params(FIXTURE)
    chains = simulate_chains(params)
    assert chains.shape == (params.n_chains, params.chain_length)
    assert set(np.unique(chains)) <= set(range(params.n_states))


@pytest.mark.smoke
def test_simulation_is_reproducible_from_the_seed() -> None:
    params = load_potts_params(FIXTURE)
    assert np.array_equal(simulate_chains(params), simulate_chains(params))


@pytest.mark.end2end
def test_coupling_raises_the_frequency_of_adjacent_agreement() -> None:
    # A generative check with an unambiguous direction: positive coupling
    # rewards agreeing neighbours, so simulated chains must agree more often
    # than independent draws from the same field would. The independent rate
    # is computed in closed form, not simulated, so this is an analytic
    # comparison rather than two runs of the same code.
    params = load_potts_params(FIXTURE)
    chains = simulate_chains(params)
    observed = float((chains[:, :-1] == chains[:, 1:]).mean())
    independent = float((np.exp(params.field) ** 2).sum())
    assert observed > independent


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("field", "missing required field"),
        ("n_states", "missing required field"),
    ],
)
def test_a_missing_field_is_refused(field: str, message: str, tmp_path: Path) -> None:
    text = "\n".join(
        line
        for line in FIXTURE.read_text().splitlines()
        if not line.startswith(f"{field}:")
    )
    path = tmp_path / "potts.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match=message):
        load_potts_params(path)


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("replace", "with_", "message"),
    [
        ("n_states: 3", "n_states: 1", "n_states must be >= 2"),
        ("chain_length: 12", "chain_length: 1", "chain_length must be >= 2"),
        ("field: [0.40, -0.10, -0.30]", "field: [0.4, -0.1]", "field has shape"),
    ],
)
def test_an_unusable_size_is_refused(
    replace: str, with_: str, message: str, tmp_path: Path
) -> None:
    path = tmp_path / "potts.yaml"
    path.write_text(FIXTURE.read_text().replace(replace, with_))
    with pytest.raises(ValueError, match=message):
        load_potts_params(path)


@pytest.mark.smoke
def test_the_constraint_map_is_the_one_the_objective_uses() -> None:
    # Ties the instance to the shared vocabulary rather than to a private
    # copy of it: a divergence here is what would make `opt/CLAUDE.md`'s
    # constraints-by-construction rule true in one module and not another.
    objective = PottsObjective(np.zeros((2, 4), dtype=np.int64), n_states=4)
    theta = torch.tensor([0.3, 0.5, -1.0, 0.25], dtype=torch.float64)
    assert torch.equal(objective.constrain(theta)["field"], log_simplex(theta[1:]))
