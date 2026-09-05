"""Validation-error paths for the k-state Jukes-Cantor simulator.

Separated from tests/regression/test_jc_simulate.py, which pins scientific
correctness; these pin the guardrails around malformed inputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from phylo.sim.jc import jc_rate_matrix, jc_transition_probabilities
from phylo.sim.params import load_simulation_params
from phylo.sim.simulate import simulate_alignment
from phylo.sim.tree import Node


@pytest.mark.edge_case
def test_jc_transition_probabilities_rejects_k_below_two() -> None:
    with pytest.raises(ValueError, match="k must be >= 2"):
        jc_transition_probabilities(0.1, k=1)


@pytest.mark.edge_case
def test_jc_transition_probabilities_rejects_negative_branch_length() -> None:
    with pytest.raises(ValueError, match="t must be non-negative"):
        jc_transition_probabilities(-0.1, k=4)


@pytest.mark.edge_case
def test_jc_rate_matrix_rejects_k_below_two() -> None:
    with pytest.raises(ValueError, match="k must be >= 2"):
        jc_rate_matrix(k=1)


@pytest.mark.edge_case
def test_simulate_alignment_rejects_mismatched_pi_shape() -> None:
    tau = Node(
        name="root", branch_length=None, children=(Node(name="A", branch_length=0.1),)
    )
    with pytest.raises(ValueError, match="pi has shape"):
        simulate_alignment(tau=tau, k=4, pi=np.full(3, 1.0 / 3), seed=0, n_sites=10)


@pytest.mark.edge_case
def test_simulate_alignment_rejects_non_root_node_without_branch_length() -> None:
    tau = Node(
        name="root", branch_length=None, children=(Node(name="A", branch_length=None),)
    )
    with pytest.raises(ValueError, match="has no branch_length"):
        simulate_alignment(tau=tau, k=4, pi=np.full(4, 0.25), seed=0, n_sites=10)


@pytest.mark.edge_case
def test_load_simulation_params_rejects_missing_field(tmp_path: Path) -> None:
    incomplete = tmp_path / "incomplete.yaml"
    incomplete.write_text("seed: 0\nn_sites: 10\nk: 4\n")
    with pytest.raises(ValueError, match="missing required field"):
        load_simulation_params(incomplete)


@pytest.mark.edge_case
def test_load_simulation_params_rejects_mismatched_pi_shape(tmp_path: Path) -> None:
    bad_pi = tmp_path / "bad_pi.yaml"
    bad_pi.write_text(
        "seed: 0\n"
        "n_sites: 10\n"
        "tolerance: 0.01\n"
        "k: 4\n"
        "pi: [0.5, 0.5]\n"
        "tau:\n"
        "  name: root\n"
        "  children:\n"
        "    - name: A\n"
        "      branch_length: 0.1\n"
    )
    with pytest.raises(ValueError, match="pi has shape"):
        load_simulation_params(bad_pi)


@pytest.mark.edge_case
def test_load_simulation_params_rejects_pi_not_summing_to_one(tmp_path: Path) -> None:
    bad_pi = tmp_path / "bad_pi_sum.yaml"
    bad_pi.write_text(
        "seed: 0\n"
        "n_sites: 10\n"
        "tolerance: 0.01\n"
        "k: 4\n"
        "pi: [0.5, 0.5, 0.5, 0.5]\n"
        "tau:\n"
        "  name: root\n"
        "  children:\n"
        "    - name: A\n"
        "      branch_length: 0.1\n"
    )
    with pytest.raises(ValueError, match="pi sums to"):
        load_simulation_params(bad_pi)
