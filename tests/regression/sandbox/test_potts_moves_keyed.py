"""Niedermayer's keyed move against its kernel, moved with the move to the sandbox (issues #756, #1365)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from sal.sample.potts_keyed import SwendsenWangMove, WolffMove
from sal.sandbox.potts_moves import NiedermayerMove, niedermayer_sweep

from tests._rows import every_value
from tests.regression.learn.test_cluster_arms import (
    _lattice,
)


@pytest.mark.smoke
def test_a_move_and_a_field_of_different_heights_are_refused() -> None:
    """The same check on the move's own side, where the field arrives."""
    graph, _, _ = _lattice(4, 3)
    for kind in (WolffMove, SwendsenWangMove, NiedermayerMove):
        with pytest.raises(ValueError, match="a move and its environment score"):
            kind(graph, np.zeros((9, 3)))


@pytest.mark.oracle
def test_a_niedermayer_step_is_potts_mcmcs_own_sweep_bitwise() -> None:
    """The same for issue #756's arm, and at ``T = 0`` as well.

    ``math.inf`` passes through `niedermayer_sweep`: one kernel, four temperatures.
    """

    def check(temperature: float) -> None:
        graph, field, _ = _lattice(4, 2)
        move = NiedermayerMove(graph, field)
        offsets, neighbours, couplings = graph.compressed_adjacency()
        state = np.ascontiguousarray(
            np.random.default_rng(3).integers(0, 2, size=graph.n_nodes), dtype=np.int64
        )

        keyed, charge = move.propose(
            state,
            temperature=temperature,
            site=5,
            label=1,
            rng=np.random.default_rng(99),
        )
        direct = state.copy()
        size = niedermayer_sweep(
            direct,
            field,
            offsets,
            neighbours,
            couplings,
            np.random.default_rng(99),
            beta=math.inf if temperature == 0.0 else 1.0 / temperature,
            threshold=move.threshold,
            root=5,
            partner=1,
        )

        assert np.array_equal(keyed, direct)
        assert charge == size * (1 + 2 * len(graph.edges) // graph.n_nodes)

    every_value([0.0, 0.25, 1.0, 4.0], check)
