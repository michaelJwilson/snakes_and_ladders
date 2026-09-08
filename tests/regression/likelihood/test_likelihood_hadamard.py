"""The Hadamard conjugation inverts the two-state model exactly, and converges from data.

Issue #364. The oracle for the exact sequence spectrum is the pruning
likelihood at ``k = 2`` evaluated on every one of the ``2^(n-1)`` site
patterns --- code that shares nothing with the transform --- so the
conjugation returning the tree's branch lengths on its splits and zero on
every other is a statement about the model, not about the transform's own
forward direction. The four-state recoding is held to the same oracle at
``k = 4``, where the returned weight must be ``2/3`` of every branch.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from numpy.testing import assert_allclose
from snakes_and_ladders.likelihood.hadamard import (
    MAX_TAXA,
    binary_recoding,
    closest_tree,
    edge_spectrum,
    expected_spectrum,
    hadamard_conjugation,
    recoding_scale,
    sequence_spectrum,
    split_weights,
    walsh_hadamard,
)
from snakes_and_ladders.likelihood.pruning import log_likelihood
from snakes_and_ladders.search.neighbor_joining import split_lengths
from snakes_and_ladders.search.topology import leaf_bipartitions
from snakes_and_ladders.sim.simulate import simulate_alignment
from snakes_and_ladders.sim.tree import Node

from tests._fixtures import EIGHT_TAXA, FOUR_TAXA, load_fixture

FIVE_TAXA = "simulation_params_5taxa.yaml"
SIX_TAXA = "simulation_params_6taxa.yaml"
HARD = "simulation_params_hard.yaml"


def _exact_spectrum(tau: Node, names: list[str], k: int) -> np.ndarray:
    """Pattern probabilities under ``tau`` at ``k`` states, by the pruning likelihood.

    The two-state spectrum indexes patterns by the taxa differing from the
    last; at ``k`` states every pattern over ``k^n`` assignments is scored
    and folded onto that index through the recoding, which is the oracle
    for the recoded spectrum too.
    """
    pi = np.full(k, 1.0 / k)
    spectrum = np.zeros(1 << (len(names) - 1))
    for states in itertools.product(range(k), repeat=len(names)):
        pattern = {
            name: np.array([state]) for name, state in zip(names, states, strict=True)
        }
        recoded = binary_recoding(pattern, k)
        index = sum(
            1 << bit
            for bit, name in enumerate(names[:-1])
            if recoded[name][0] != recoded[names[-1]][0]
        )
        spectrum[index] += np.exp(log_likelihood(tau, k, pi, pattern))
    return spectrum


@pytest.mark.mathematical
def test_the_transform_is_the_sylvester_matrix_and_its_own_inverse_up_to_size() -> None:
    """``H`` has entries ``(-1)^|A and B|`` and ``H H = N I``."""
    size = 8
    matrix = np.array(
        [[(-1) ** bin(a & b).count("1") for b in range(size)] for a in range(size)]
    )
    rng = np.random.default_rng(0)
    vector = rng.normal(size=size)

    assert_allclose(walsh_hadamard(vector), matrix @ vector, atol=1e-12)
    assert_allclose(walsh_hadamard(walsh_hadamard(vector)), size * vector, atol=1e-12)


@pytest.mark.oracle
@pytest.mark.parametrize("name", [FOUR_TAXA, FIVE_TAXA, SIX_TAXA, HARD, EIGHT_TAXA])
def test_the_conjugation_returns_the_true_split_weights_on_the_exact_spectrum(
    name: str,
) -> None:
    """On the pruning-computed two-state spectrum, ``q`` is the tree's lengths and zero elsewhere.

    Every split the tree has carries its branch length --- the root pair of
    a rooted binary fixture summed --- and every other of the
    ``2^(n-1) - n`` non-trivial splits carries zero, to ``1e-12``. Both
    directions are pinned: the inverse on the oracle spectrum, and the
    forward transform reproducing the oracle.
    """
    params = load_fixture(name)
    names, truth = edge_spectrum(params.tau)
    spectrum = _exact_spectrum(params.tau, names, 2)

    assert spectrum.sum() == pytest.approx(1.0, abs=1e-12)
    assert_allclose(hadamard_conjugation(spectrum), truth, atol=1e-12)
    assert_allclose(expected_spectrum(truth), spectrum, atol=1e-12)
    weights = split_weights(hadamard_conjugation(spectrum), names)
    lengths = split_lengths(params.tau)
    for split, weight in weights.items():
        assert weight == pytest.approx(lengths.get(split, 0.0), abs=1e-12)
    assert sum(1 for weight in weights.values() if abs(weight) > 1e-12) == len(lengths)


@pytest.mark.oracle
@pytest.mark.parametrize("name", [FOUR_TAXA, FIVE_TAXA])
def test_the_four_state_recoding_returns_two_thirds_of_every_branch(name: str) -> None:
    """The recoded four-state spectrum conjugates to ``2 t / 3`` on every split.

    Grouping the states in pairs makes a change of group a two-state
    symmetric process at rate ``2/3`` of the four-state one, so the
    conjugation of the recoded oracle spectrum is the edge spectrum scaled
    by :func:`recoding_scale`, to ``1e-12``.
    """
    params = load_fixture(name)
    names, truth = edge_spectrum(params.tau)
    spectrum = _exact_spectrum(params.tau, names, 4)

    assert recoding_scale(4) == pytest.approx(2.0 / 3.0)
    assert_allclose(
        hadamard_conjugation(spectrum), recoding_scale(4) * truth, atol=1e-12
    )


@pytest.mark.mathematical
@pytest.mark.parametrize("name", [FOUR_TAXA, FIVE_TAXA, SIX_TAXA, HARD, EIGHT_TAXA])
def test_the_closest_tree_of_the_exact_weights_is_the_tree(name: str) -> None:
    """The ``n - 3`` largest compatible weights are the tree's own splits, lengths exact."""
    params = load_fixture(name)
    names, truth = edge_spectrum(params.tau)
    weights = split_weights(truth, names)

    tree = closest_tree(weights, names, minimum_length=1e-6)

    assert leaf_bipartitions(tree) == leaf_bipartitions(params.tau)
    lengths = split_lengths(params.tau)
    for split, length in split_lengths(tree).items():
        assert length == pytest.approx(lengths[split], abs=1e-12)


@pytest.mark.simulated_truth
def test_the_split_weights_converge_with_the_site_count() -> None:
    """On simulated two-state data the weight error falls as ``1/sqrt(L)``.

    The five-taxon fixture at ``k = 2``, 1,000 to 100,000 sites, 40 seeds
    each: the root-mean-square error over every split (true weight or zero)
    and the ratio between successive sizes against ``sqrt(10)``. Realized:
    RMS 0.0265, 0.00802 and 0.00257; ratios 3.30 and 3.13.
    """
    params = load_fixture(FIVE_TAXA)
    names, truth = edge_spectrum(params.tau)
    rms: list[float] = []
    for n_sites in (1000, 10_000, 100_000):
        errors = []
        for seed in range(40):
            dataset = simulate_alignment(
                params.tau,
                2,
                np.full(2, 0.5),
                np.random.default_rng([364, seed]),
                n_sites,
            )
            _, spectrum = sequence_spectrum(dataset.alignment)
            errors.append(hadamard_conjugation(spectrum)[1:] - truth[1:])
        rms.append(float(np.sqrt(np.mean(np.square(errors)))))

    assert rms[2] < rms[1] < rms[0]
    assert 2.5 < rms[0] / rms[1] < 4.0
    assert 2.5 < rms[1] / rms[2] < 4.0


@pytest.mark.simulated_truth
@pytest.mark.parametrize("name", [FOUR_TAXA, FIVE_TAXA, SIX_TAXA, EIGHT_TAXA])
def test_the_closest_tree_of_the_fixture_alignment_is_the_generating_topology(
    name: str,
) -> None:
    """The four-state fixtures, recoded, give the generating topology at their declared sites."""
    params = load_fixture(name)
    dataset = simulate_alignment(
        params.tau,
        params.k,
        params.pi,
        np.random.default_rng(params.seed),
        params.n_sites,
    )
    names, spectrum = sequence_spectrum(binary_recoding(dataset.alignment, params.k))
    weights = split_weights(hadamard_conjugation(spectrum), names)

    tree = closest_tree(weights, names, minimum_length=1e-4)

    assert leaf_bipartitions(tree) == leaf_bipartitions(params.tau)


@pytest.mark.edge_case
def test_a_spectrum_the_logarithm_cannot_take_is_refused() -> None:
    """The hard fixture at its 2,000 sites: ``H s`` has a non-positive entry, and it is refused.

    Branches of 0.4 make the transformed entries products of small factors,
    and sampling noise pushes one below zero; a conjugation that clamped it
    would return weights for a spectrum no tree has.
    """
    params = load_fixture(HARD)
    dataset = simulate_alignment(
        params.tau,
        params.k,
        params.pi,
        np.random.default_rng(params.seed),
        params.n_sites,
    )
    _, spectrum = sequence_spectrum(binary_recoding(dataset.alignment, params.k))

    with pytest.raises(ValueError, match="non-positive"):
        hadamard_conjugation(spectrum)


@pytest.mark.edge_case
def test_unusable_inputs_are_refused() -> None:
    with pytest.raises(ValueError, match="power-of-two"):
        walsh_hadamard(np.ones(6))
    with pytest.raises(ValueError, match="even"):
        binary_recoding({"A": np.array([0, 1, 2])}, 3)
    too_many = {
        f"t{index:02d}": np.zeros(4, dtype=int) for index in range(MAX_TAXA + 1)
    }
    with pytest.raises(ValueError, match=f"{MAX_TAXA} taxa"):
        sequence_spectrum(too_many)
    with pytest.raises(ValueError, match="two-state"):
        sequence_spectrum(
            {"A": np.array([0, 2]), "B": np.array([0, 0]), "C": np.array([1, 1])}
        )
    with pytest.raises(ValueError, match="positive"):
        closest_tree({}, ["A", "B", "C"], minimum_length=0.0)
