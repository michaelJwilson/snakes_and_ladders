"""Two estimators, and the loss each one minimizes, refereed by enumeration.

Issue #696. The claim is not that one decoder is better --- it is that they
answer different questions, and that the reported metric decides which is
wanted. `label_accuracy` scores per-site agreement, so the marginal decoder is
the matching estimator, and a maximum-a-posteriori labelling reported against
a per-site score is answering something nobody asked.

Everything here is checked against exhaustive enumeration of all ``3**9``
labellings: the exact posterior, the exact marginals, and the exact maximum.
No approximation is involved, so nothing here can be an artefact of one.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from snakes_and_ladders.likelihood.potts import enumerate_potts
from snakes_and_ladders.search.decoding import expected_site_errors, marginal_decode
from snakes_and_ladders.sim.graph import (
    BoundaryCondition,
    PottsGraph,
    lattice_graph,
    triangular_lattice_graph,
)
from snakes_and_ladders.sim.potts import energy

FIELD = np.log(np.array([0.45, 0.35, 0.20]))


def _exhaustive(graph: PottsGraph) -> tuple[np.ndarray, np.ndarray]:
    """Every labelling and its energy, in one array each."""
    labellings = np.asarray(
        list(itertools.product(range(3), repeat=graph.n_nodes)), dtype=np.int64
    )
    energies = np.asarray([energy(graph, FIELD, labelling) for labelling in labellings])
    return labellings, energies


@pytest.mark.analytic
@pytest.mark.potts_lattice
def test_the_marginal_decoder_minimizes_the_expected_wrong_sites() -> None:
    # The theorem, checked rather than argued: `sum_i (1 - p_i(y_i))` is
    # minimized term by term at each site's own argmax, so no labelling beats
    # the marginal one on this loss. Checked against *every* labelling, which
    # is the only way to assert "no labelling beats it".
    graph = triangular_lattice_graph((3, 3), BoundaryCondition.OPEN, -0.9)
    exact = enumerate_potts(graph, FIELD)
    labellings, _ = _exhaustive(graph)

    decoded = marginal_decode(exact.single_site)
    mine = expected_site_errors(exact.single_site, decoded)
    best = min(
        expected_site_errors(exact.single_site, labelling) for labelling in labellings
    )

    assert mine == pytest.approx(best, abs=1e-12)


@pytest.mark.oracle
@pytest.mark.frustrated_lattice
def test_the_two_estimators_disagree_on_a_frustrated_field() -> None:
    # The case that makes the distinction operational, with both numbers. On
    # the triangular antiferromagnet the two labellings differ at six of nine
    # sites: the maximum-a-posteriori one is a proper 3-colouring, and the
    # marginal one puts every site at its own field-preferred label.
    graph = triangular_lattice_graph((3, 3), BoundaryCondition.OPEN, -0.9)
    exact = enumerate_potts(graph, FIELD)
    labellings, energies = _exhaustive(graph)

    marginal = marginal_decode(exact.single_site)
    posterior_mode = labellings[int(np.argmin(energies))]

    assert int((marginal != posterior_mode).sum()) == 6
    # Each estimator wins on its own loss and loses on the other's.
    assert expected_site_errors(exact.single_site, marginal) < expected_site_errors(
        exact.single_site, posterior_mode
    )
    assert energy(graph, FIELD, posterior_mode) < energy(graph, FIELD, marginal)


@pytest.mark.oracle
@pytest.mark.frustrated_lattice
def test_the_marginal_labelling_can_be_a_configuration_nobody_would_pick() -> None:
    # The sharpest statement of what the loss buys and what it costs. The
    # marginal labelling here ranks 19,555th of 19,683 by posterior --- inside
    # the worst one per cent --- and is still the better per-site estimator.
    # Minimizing per-site error does not require the answer to be jointly
    # plausible, and a decoder reported without its loss hides exactly this.
    graph = triangular_lattice_graph((3, 3), BoundaryCondition.OPEN, -0.9)
    exact = enumerate_potts(graph, FIELD)
    labellings, energies = _exhaustive(graph)

    marginal = marginal_decode(exact.single_site)
    rank = int((energies < energy(graph, FIELD, marginal)).sum()) + 1

    assert rank > 0.99 * len(labellings)
    assert expected_site_errors(exact.single_site, marginal) < expected_site_errors(
        exact.single_site, labellings[int(np.argmin(energies))]
    )


@pytest.mark.analytic
@pytest.mark.potts_lattice
def test_the_estimators_agree_where_the_posterior_is_unimodal() -> None:
    # The null case, stated so the difference above is not read as general: on
    # a ferromagnet at this field the posterior concentrates on one labelling
    # and both estimators return it. A distinction that appeared everywhere
    # would be a bug in one of them.
    graph = lattice_graph((3, 3), BoundaryCondition.OPEN, 0.6)
    exact = enumerate_potts(graph, FIELD)
    labellings, energies = _exhaustive(graph)

    marginal = marginal_decode(exact.single_site)

    np.testing.assert_array_equal(marginal, labellings[int(np.argmin(energies))])


@pytest.mark.smoke
@pytest.mark.potts_lattice
def test_marginals_that_are_not_a_distribution_are_refused() -> None:
    # A caller who has not normalized has not converged, and normalizing here
    # would hide an unconverged message-passing run inside a plausible
    # labelling.
    with pytest.raises(ValueError, match="not one"):
        marginal_decode(np.array([[0.5, 0.2], [0.5, 0.5]]))
    with pytest.raises(ValueError, match="negative"):
        marginal_decode(np.array([[1.5, -0.5], [0.5, 0.5]]))
    with pytest.raises(ValueError, match=r"\(n_nodes, n_states\)"):
        marginal_decode(np.array([0.5, 0.5]))


@pytest.mark.smoke
@pytest.mark.potts_lattice
def test_a_labelling_of_the_wrong_length_is_refused() -> None:
    # The loss reads one marginal per site, so a labelling of another length
    # is a caller error rather than something to broadcast.
    with pytest.raises(ValueError, match="to match the marginals"):
        expected_site_errors(np.full((4, 3), 1 / 3), np.zeros(3, dtype=np.int64))
