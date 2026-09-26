"""Validation-error paths for the k-state Jukes-Cantor simulator.

Separated from tests/regression/test_jc_simulate.py, which pins scientific
correctness; these pin the guardrails around malformed inputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sal.fixtures import load_params
from sal.sim.jc import jc_rate_matrix, jc_transition_probabilities
from sal.sim.params import SimulationParams
from sal.sim.simulate import simulate_alignment
from sal.sim.tree import Node


@pytest.mark.smoke
def test_jc_transition_probabilities_rejects_k_below_two() -> None:
    with pytest.raises(ValueError, match="k must be >= 2"):
        jc_transition_probabilities(0.1, n_states=1)


@pytest.mark.smoke
def test_jc_transition_probabilities_rejects_negative_branch_length() -> None:
    with pytest.raises(ValueError, match="t must be non-negative"):
        jc_transition_probabilities(-0.1, n_states=4)


@pytest.mark.smoke
def test_jc_rate_matrix_rejects_k_below_two() -> None:
    with pytest.raises(ValueError, match="k must be >= 2"):
        jc_rate_matrix(n_states=1)


@pytest.mark.smoke
def test_simulate_alignment_rejects_mismatched_pi_shape() -> None:
    tau = Node(
        name="root", branch_length=None, children=(Node(name="A", branch_length=0.1),)
    )
    with pytest.raises(ValueError, match="pi has shape"):
        simulate_alignment(
            tau=tau,
            n_states=4,
            pi=np.full(3, 1.0 / 3),
            rng=np.random.default_rng(0),
            n_sites=10,
        )


@pytest.mark.smoke
def test_simulate_alignment_rejects_non_root_node_without_branch_length() -> None:
    tau = Node(
        name="root", branch_length=None, children=(Node(name="A", branch_length=None),)
    )
    with pytest.raises(ValueError, match="has no branch_length"):
        simulate_alignment(
            tau=tau,
            n_states=4,
            pi=np.full(4, 0.25),
            rng=np.random.default_rng(0),
            n_sites=10,
        )


@pytest.mark.smoke
def test_load_simulation_params_rejects_missing_field(tmp_path: Path) -> None:
    incomplete = tmp_path / "incomplete.yaml"
    incomplete.write_text("seed: 0\nn_sites: 10\nn_states: 4\n")
    with pytest.raises(ValueError, match="missing required field"):
        load_params(incomplete, SimulationParams)


@pytest.mark.smoke
def test_load_simulation_params_rejects_mismatched_pi_shape(tmp_path: Path) -> None:
    bad_pi = tmp_path / "bad_pi.yaml"
    bad_pi.write_text(
        "seed: 0\n"
        "n_sites: 10\n"
        "tolerance: 0.01\n"
        "n_states: 4\n"
        "pi: [0.5, 0.5]\n"
        "tau:\n"
        "  name: root\n"
        "  children:\n"
        "    - name: A\n"
        "      branch_length: 0.1\n"
    )
    with pytest.raises(ValueError, match="pi has shape"):
        load_params(bad_pi, SimulationParams)


@pytest.mark.smoke
def test_load_simulation_params_rejects_pi_not_summing_to_one(tmp_path: Path) -> None:
    bad_pi = tmp_path / "bad_pi_sum.yaml"
    bad_pi.write_text(
        "seed: 0\n"
        "n_sites: 10\n"
        "tolerance: 0.01\n"
        "n_states: 4\n"
        "pi: [0.5, 0.5, 0.5, 0.5]\n"
        "tau:\n"
        "  name: root\n"
        "  children:\n"
        "    - name: A\n"
        "      branch_length: 0.1\n"
    )
    with pytest.raises(ValueError, match="pi sums to"):
        load_params(bad_pi, SimulationParams)
