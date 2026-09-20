"""The shared pruning plumbing moved no number: every route still meets the oracle (issue #858).

``snakes_and_ladders.likelihood.pruning_common`` carries the post-order, the
leaf indicator, the rescaling step and the four validations the Rust, Torch,
analytic and surrogate routes had a copy of each. The oracle,
``likelihood.pruning``, imports none of it --- so the seam is checked the way
the ladder is: each route against the oracle on ``tree_jc/ci``.

What each route is held to is what it met before the fold, and the two are not
the same claim. ``pruning_torch`` and ``pruning_analytic`` reproduce the
oracle's value *bitwise* at this fixture, so that is what is asserted; the
compiled route reassociates its sums in Rust and never did, so it is held to
:data:`~snakes_and_ladders.likelihood.device.CROSS_DEVICE_RTOL_FLOAT64`, the
float64 implementation-agreement bound of ``docs/tex/textbook.tex``
(``sec:tolerance``). Loosening either to admit a result is forbidden; both are
where the comparison lands.

``test_pruning_rust.py``, ``test_pruning_torch.py`` and
``test_pruning_analytic.py`` keep their own oracle tests on their own
fixtures. This module adds the one they cannot carry between them: the routes
through the shared plumbing, read against the oracle that does not use it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from snakes_and_ladders.likelihood import (
    pruning,
    pruning_analytic,
    pruning_common,
    pruning_rust,
    pruning_torch,
)
from snakes_and_ladders.likelihood.device import CROSS_DEVICE_RTOL_FLOAT64
from snakes_and_ladders.sim.fixtures import fixture
from snakes_and_ladders.sim.simulate import simulate_alignment
from snakes_and_ladders.sim.tree import Node


def _instance() -> tuple[Node, int, np.ndarray, dict[str, np.ndarray]]:
    """``tree_jc/ci``'s topology and an alignment simulated at its own seed."""
    params = fixture("tree_jc", "ci").params
    pi = np.asarray(params.pi, dtype=float)
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=pi,
        rng=np.random.default_rng(params.seed),
        n_sites=params.n_sites,
    )
    return params.tau, params.k, pi, dataset.alignment


@pytest.mark.oracle
def test_the_torch_route_reproduces_the_oracle_bitwise() -> None:
    tau, k, pi, alignment = _instance()
    lengths = pruning_torch.branch_lengths_from_tree(tau)

    value = pruning_torch.log_likelihood(tau, k, pi, alignment, lengths)

    assert float(value) == pruning.log_likelihood(tau, k, pi, alignment)


@pytest.mark.oracle
def test_the_analytic_route_reproduces_the_oracle_bitwise() -> None:
    tau, k, pi, alignment = _instance()
    lengths = pruning_torch.branch_lengths_from_tree(tau)

    value = pruning_analytic.log_likelihood(tau, k, pi, alignment, lengths)

    assert float(value) == pruning.log_likelihood(tau, k, pi, alignment)


@pytest.mark.oracle
def test_the_cached_route_reproduces_the_oracle_bitwise() -> None:
    tau, k, pi, alignment = _instance()
    lengths = pruning_torch.branch_lengths_from_tree(tau)

    value = pruning_torch.log_likelihood_cached(
        tau, k, pi, alignment, lengths, pruning_torch.PartialCache()
    )

    assert value == pruning.log_likelihood(tau, k, pi, alignment)


@pytest.mark.oracle
@pytest.mark.backend
def test_the_rust_route_meets_the_oracle_within_the_float64_bound() -> None:
    tau, k, pi, alignment = _instance()

    value = pruning_rust.log_likelihood(tau, k, pi, alignment)

    assert_allclose(
        value,
        pruning.log_likelihood(tau, k, pi, alignment),
        rtol=CROSS_DEVICE_RTOL_FLOAT64,
    )


@pytest.mark.analytic
def test_the_leaf_indicator_is_the_one_hot_of_the_observed_states() -> None:
    """The indicator of ``eq:pruning``: one row per site, one at what was seen."""
    states = np.array([0, 3, 1, 1], dtype=np.int64)

    table = pruning_common.leaf_indicator_array(states, 4, 4)
    tensor = pruning_common.leaf_indicator(
        states, 4, 4, torch.float64, torch.device("cpu"), index_device=None
    )

    assert table.sum() == 4.0
    assert list(np.argmax(table, axis=1)) == list(states)
    assert torch.equal(tensor, torch.as_tensor(table))


@pytest.mark.analytic
def test_rescaling_leaves_the_log_likelihood_where_it_was() -> None:
    """Dividing by the per-site maximum and logging it is a transformation, not an approximation."""
    partial = torch.tensor([[0.25, 0.5], [1e-8, 4e-8]], dtype=torch.float64)
    log_scale = torch.zeros(2, dtype=torch.float64)

    rescaled, accumulated, scale = pruning_common.rescale_partial(
        partial, log_scale, None
    )

    assert_allclose(
        (torch.log(rescaled.sum(dim=1)) + accumulated).numpy(),
        torch.log(partial.sum(dim=1)).numpy(),
        rtol=CROSS_DEVICE_RTOL_FLOAT64,
    )
    assert_allclose(scale.numpy(), partial.amax(dim=1).numpy(), rtol=0.0)


@pytest.mark.analytic
def test_a_vanished_scale_is_left_at_zero_rather_than_divided() -> None:
    """A site the model forbids keeps ``log(0) = -inf`` instead of a masked scale."""
    partial = torch.zeros((1, 3), dtype=torch.float64)
    log_scale = torch.zeros(1, dtype=torch.float64)

    rescaled, accumulated, scale = pruning_common.rescale_partial(
        partial, log_scale, None
    )

    assert float(scale) == 1.0
    assert float(accumulated) == 0.0
    assert float(torch.log(rescaled.sum(dim=1))) == float("-inf")


@pytest.mark.smoke
def test_the_shared_checks_refuse_what_each_route_refused() -> None:
    tau = Node(
        name="root",
        branch_length=None,
        children=(
            Node(name="A", branch_length=0.1),
            Node(name="B", branch_length=None),
        ),
    )
    with pytest.raises(ValueError, match="pi has shape"):
        pruning_common.check_pi_shape((3,), 4)
    with pytest.raises(ValueError, match="alignment is missing leaf"):
        pruning_common.check_alignment_covers(["A", "B"], {"A": None})
    with pytest.raises(ValueError, match="to match branch_order"):
        pruning_common.check_branch_lengths_shape((2,), 3)
    with pytest.raises(ValueError, match="has no branch_length"):
        pruning_common.require_branch_length(tau.children[1])
    assert pruning_common.require_branch_length(tau.children[0]) == 0.1
    assert [node.name for node in pruning_common.postorder(tau)] == ["A", "B", "root"]
