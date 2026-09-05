"""Fitting a Potts model on a lattice, against an exact normalizer.

`STATUS.md`'s requirements ledger read "**Met** for the 1-D chain and the
discrete HMM; lattice outstanding". This closes it, and the thing that makes
the claim checkable is that `log Z` is *exact* here: enumerated over all
`q ** n_nodes` configurations rather than approximated, so a fitted optimum
is compared against a brute-force scan of the likelihood rather than against
the optimizer's own convergence. Those are different claims and only the
second is worth much.

Coverage is a rate, not a draw. A single dataset covering 4 of 4 intervals
says nothing about a 95% interval; what is asserted is the realized fraction
over independent replicates, and the direction it approaches nominal from --
which `opt_coverage`'s committed caption already reports for the chain and
the hidden Markov model, from opposite sides.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from phylo.opt.fit import constrained_standard_errors, covers, fit
from phylo.opt.potts import (
    PottsLatticeObjective,
    graph_statistics,
    log_partition,
    log_partition_graph,
)
from phylo.sim.graph import BoundaryCondition, lattice_graph
from phylo.sim.potts import simulate_potts

SHAPE = (3, 3)
N_STATES = 3
COUPLING = 0.6
# Gauge-fixed, as the loader would leave it: `logsumexp(h) == 0`.
FIELD = np.array([0.30, -0.10, -0.20])
FIELD = FIELD - np.log(np.exp(FIELD).sum())
SEED = 20260904

# Realized over 40 replicates at each size, 4 parameters each: 157/160 = 0.981
# at 100 samples, 153/160 = 0.956 at 400. The lattice approaches the nominal
# rate from above, as the Potts chain does and the hidden Markov model does
# not.
_COVERAGE_AT_400 = 0.956


def _graph() -> tuple[tuple[tuple[int, int], ...], int]:
    graph = lattice_graph(SHAPE, boundary=BoundaryCondition.OPEN, coupling=COUPLING)
    return graph.edges, graph.n_nodes


def _fitted(n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit one simulated dataset; return truth, estimate and standard error."""
    edges, _ = _graph()
    graph = lattice_graph(SHAPE, boundary=BoundaryCondition.OPEN, coupling=COUPLING)
    dataset = simulate_potts(graph, FIELD, seed=seed, n_samples=n_samples, burn_in=200)
    objective = PottsLatticeObjective(dataset.configurations, N_STATES, edges)
    result = fit(objective)
    assert result.converged

    estimate = objective.constrain(result.theta)
    error = constrained_standard_errors(objective, result.theta)
    truth = torch.cat(
        [torch.tensor([COUPLING], dtype=torch.float64), torch.as_tensor(FIELD)]
    )
    fitted = torch.cat([estimate["coupling"].reshape(1), estimate["field"]])
    spread = torch.cat([error["coupling"].reshape(1), error["field"]])
    return truth.numpy(), fitted.numpy(), spread.numpy()


@pytest.mark.oracle
@pytest.mark.parametrize("length", [2, 4, 6])
def test_the_graph_normalizer_reduces_to_the_transfer_matrix(length: int) -> None:
    # A chain is a 1-D lattice, and its `log Z` has a closed form by transfer
    # matrix. Two exact routes to the same number, sharing no code: one
    # enumerates `3 ** length` configurations, the other multiplies matrices.
    graph = lattice_graph((length,), boundary=BoundaryCondition.OPEN, coupling=COUPLING)
    agreements, counts = graph_statistics(N_STATES, graph.edges, graph.n_nodes)
    field = torch.as_tensor(FIELD)

    for coupling in (0.0, 0.75, -0.4):
        as_tensor = torch.tensor(coupling, dtype=torch.float64)
        enumerated = float(log_partition_graph(as_tensor, field, agreements, counts))
        transfer = float(log_partition(as_tensor, field, length))
        # Absolute, not relative: at `coupling = 0` with a gauge-fixed field
        # both are zero to rounding, and a relative deviation there divides by
        # nothing meaningful.
        assert enumerated == pytest.approx(transfer, abs=1e-12)


@pytest.mark.oracle
def test_the_enumerated_statistics_count_every_configuration() -> None:
    edges, n_nodes = _graph()
    agreements, counts = graph_statistics(N_STATES, edges, n_nodes)

    assert agreements.shape == (N_STATES**n_nodes,)
    assert counts.shape == (N_STATES**n_nodes, N_STATES)
    # Every site holds exactly one state, so the counts sum to the site count.
    assert_allclose(counts.sum(dim=1).numpy(), np.full(N_STATES**n_nodes, n_nodes))
    # A uniform configuration agrees on every edge; there are `q` of them.
    assert int((agreements == len(edges)).sum()) == N_STATES


@pytest.mark.mathematical
def test_the_gradient_matches_central_differences() -> None:
    edges, _ = _graph()
    graph = lattice_graph(SHAPE, boundary=BoundaryCondition.OPEN, coupling=COUPLING)
    dataset = simulate_potts(graph, FIELD, seed=SEED, n_samples=200, burn_in=200)
    objective = PottsLatticeObjective(dataset.configurations, N_STATES, edges)

    theta = objective.theta_from_truth(COUPLING, FIELD).requires_grad_(True)
    analytic = torch.autograd.grad(objective(theta), theta)[0]

    step = 1e-6
    numerical = torch.zeros_like(analytic)
    for index in range(theta.shape[0]):
        shift = torch.zeros_like(theta)
        shift[index] = step
        with torch.no_grad():
            numerical[index] = (objective(theta + shift) - objective(theta - shift)) / (
                2 * step
            )

    assert_allclose(analytic.detach().numpy(), numerical.numpy(), rtol=1e-6, atol=1e-6)


@pytest.mark.oracle
def test_the_fitted_optimum_beats_a_brute_force_scan() -> None:
    # Checks the optimizer against the objective rather than against itself:
    # a coarse grid over the coupling, with the field at its fitted value,
    # must not find a lower negative log-likelihood than the fit did.
    edges, _ = _graph()
    graph = lattice_graph(SHAPE, boundary=BoundaryCondition.OPEN, coupling=COUPLING)
    dataset = simulate_potts(graph, FIELD, seed=SEED, n_samples=200, burn_in=200)
    objective = PottsLatticeObjective(dataset.configurations, N_STATES, edges)
    result = fit(objective)

    with torch.no_grad():
        best = float(objective(result.theta))
        for coupling in np.linspace(-1.0, 2.0, 61):
            probe = result.theta.clone()
            probe[0] = float(coupling)
            assert float(objective(probe)) >= best - 1e-9


@pytest.mark.simulated_truth
def test_the_couplings_and_fields_are_recovered_within_their_intervals() -> None:
    # One dataset, so this is a draw and not a rate; the rate is the next test.
    truth, fitted, spread = _fitted(400, SEED)
    hits = covers(
        torch.as_tensor(fitted), torch.as_tensor(spread), torch.as_tensor(truth)
    )

    assert int(hits.sum()) == hits.numel()
    assert fitted[0] == pytest.approx(COUPLING, abs=4 * spread[0])


@pytest.mark.simulated_truth
@pytest.mark.release
def test_interval_coverage_approaches_the_nominal_rate() -> None:
    # The claim `STATUS.md`'s ledger row rests on. Release-gated because it
    # refits 40 datasets, each enumerating 19,683 configurations per gradient
    # step; the per-pull-request suite runs the single-dataset draw above.
    covered, total = 0, 0
    for replicate in range(40):
        truth, fitted, spread = _fitted(400, SEED + 7919 * replicate)
        hits = covers(
            torch.as_tensor(fitted), torch.as_tensor(spread), torch.as_tensor(truth)
        )
        covered += int(hits.sum())
        total += int(hits.numel())

    realized = covered / total
    # A binomial standard error at 160 draws is about 0.017, so this admits
    # about two of them either side rather than pinning a point estimate.
    assert abs(realized - _COVERAGE_AT_400) < 0.05, f"realized {realized}"
