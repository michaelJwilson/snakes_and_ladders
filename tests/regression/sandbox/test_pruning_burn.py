"""``burn``'s taped gradient: the refusals that are this route's alone.

Route A of issue #449, declined and conserved; ``pruning_torch`` is the oracle.
The shared checks run over the routes in
``tests/regression/likelihood/test_pruning_analytic.py`` (#982); the `f64`
question is settled there and in ``src/pruning_burn.rs``.
``infra/select_tests.py`` selects this on a ``likelihood`` change (#516). Skips
without the ``sandbox`` Cargo feature; ``infra/release.sh`` compiles it.
"""

from __future__ import annotations

import pytest
import torch
from snakes_and_ladders.likelihood import pruning_torch
from snakes_and_ladders.sandbox import pruning_burn

from tests._fixtures import SMALL_SITES, simulated_alignment

pytestmark = pytest.mark.skipif(
    not pruning_burn.AVAILABLE,
    reason="extension built without the `sandbox` Cargo feature",
)


def _case(name: str, n_sites: int):  # type: ignore[no-untyped-def]
    params, alignment = simulated_alignment(name, n_sites)
    return (
        params.tau,
        params.k,
        params.pi,
        alignment,
        pruning_torch.branch_lengths_from_tree(params.tau),
    )


@pytest.mark.smoke
def test_a_general_rate_matrix_is_refused() -> None:
    """`burn` has no matrix exponential, and the route says so rather than ignoring."""
    tau, k, pi, alignment, lengths = _case(SMALL_SITES, 100)
    with pytest.raises(ValueError, match="Jukes-Cantor"):
        pruning_burn.log_likelihood(
            tau,
            k,
            pi,
            alignment,
            lengths,
            rate_matrix=torch.eye(k, dtype=torch.float64),
        )


@pytest.mark.smoke
def test_branch_lengths_of_the_wrong_length_are_refused() -> None:
    tau, k, pi, alignment, lengths = _case(SMALL_SITES, 100)
    with pytest.raises(ValueError, match="branch_order"):
        pruning_burn.log_likelihood(tau, k, pi, alignment, lengths[:-1])


@pytest.mark.smoke
def test_a_ragged_alignment_is_refused() -> None:
    tau, k, pi, alignment, lengths = _case(SMALL_SITES, 100)
    ragged = dict(alignment)
    first = next(iter(ragged))
    ragged[first] = ragged[first][:-1]
    with pytest.raises(ValueError, match="ragged"):
        pruning_burn.log_likelihood(tau, k, pi, ragged, lengths)
