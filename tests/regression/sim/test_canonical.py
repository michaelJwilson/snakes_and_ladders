"""The three canonical fixtures, against the answers that come from outside.

Each of the three is admitted under both clauses of ``sim/CLAUDE.md``'s rule:
its answer is known independently of anything here, and more than one module
consumes it. The tests below are where the second clause is visible ---
``sim`` builds the instance, and ``likelihood`` and ``search`` supply the
oracle.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from phylo.likelihood.hmm_paths import enumerate_hidden_paths, path_log_probability
from phylo.search.alpha_expansion import energy, iterated_conditional_modes
from phylo.search.max_cut import enumerate_max_cut
from phylo.search.potts_mcmc import PottsMove, sample_potts
from phylo.sim.canonical import (
    AMBIGUOUS_OBSERVATIONS,
    WANNIER_RESIDUAL_ENTROPY,
    ambiguous_hmm,
    frustrated_triangular_lattice,
    minimum_frustrated_edges,
    planted_spin_glass,
)
from phylo.sim.graph import BoundaryCondition, PottsGraph, lattice_graph

ZERO_FIELD = np.zeros(2)


def _agreeing_edges(graph: PottsGraph) -> tuple[int, int]:
    """Minimum agreeing edges over all two-colourings, and how many attain it.

    Vectorized over every assignment at once, so ``2 ** 20`` stays affordable.
    """
    bits = (
        np.arange(2**graph.n_nodes, dtype=np.int64)[:, None] >> np.arange(graph.n_nodes)
    ) & 1
    agreeing = np.zeros(2**graph.n_nodes, dtype=np.int64)
    for first, second in graph.edges:
        agreeing += bits[:, first] == bits[:, second]
    lowest = int(agreeing.min())
    return lowest, int((agreeing == lowest).sum())


# --- 1. The frustrated triangular antiferromagnet ------------------------


@pytest.mark.oracle
@pytest.mark.parametrize("shape", [(3, 3), (3, 4), (4, 4)])
def test_the_periodic_ground_state_agrees_on_exactly_one_edge_in_three(
    shape: tuple[int, int],
) -> None:
    # The closed form, which is a double count rather than a search: 3N edges,
    # 2N triangles, each triangle needs an agreeing edge, each edge lies in
    # two triangles, so at least N edges agree. Enumeration shows the bound is
    # attained, at N = 9, 12 and 16.
    graph = frustrated_triangular_lattice(shape)

    lowest, _ = _agreeing_edges(graph)

    assert len(graph.edges) == 3 * graph.n_nodes
    assert lowest == minimum_frustrated_edges(graph) == graph.n_nodes
    assert lowest * 3 == len(graph.edges)


@pytest.mark.oracle
@pytest.mark.parametrize("shape", [(3, 3), (3, 4)])
def test_the_ground_state_energy_is_known_without_enumerating(
    shape: tuple[int, int],
) -> None:
    # What the closed form buys: the ground-state *energy* at any size, in the
    # convention `phylo.search.alpha_expansion.energy` uses. Every other
    # discrete claim in this repository stops where enumeration does.
    graph = frustrated_triangular_lattice(shape, coupling=-1.5)
    field = np.zeros((graph.n_nodes, 2))

    lowest, _ = _agreeing_edges(graph)
    attained = min(
        energy(graph, field, np.array(assignment, dtype=np.int64))
        for assignment in itertools.product(range(2), repeat=graph.n_nodes)
    )

    assert lowest == graph.n_nodes
    assert attained == pytest.approx(1.5 * graph.n_nodes)


@pytest.mark.mathematical
def test_a_square_lattice_is_unfrustrated_and_a_triangular_one_is_not() -> None:
    # The contrast that makes "frustration" mean something. A square lattice
    # is bipartite, so every edge can disagree and the ground-state energy is
    # exactly zero; adding the diagonal makes it non-bipartite and the ground
    # state has to pay.
    square = lattice_graph((4, 4), BoundaryCondition.PERIODIC, -1.0)
    triangular = frustrated_triangular_lattice((4, 4))

    assert _agreeing_edges(square)[0] == 0
    assert _agreeing_edges(triangular)[0] == 16


@pytest.mark.oracle
@pytest.mark.parametrize("shape", [(3, 3), (3, 4)])
def test_the_frustrated_optimum_is_the_maximum_cut(shape: tuple[int, int]) -> None:
    # The identity `max_cut.py` documents, checked on the instance it matters
    # most on: minimizing agreeing edges is maximizing the cut, so an
    # independent solver in another module must reach `2N` of the `3N` edges.
    graph = frustrated_triangular_lattice(shape)

    _, maximum = enumerate_max_cut(graph)

    assert maximum == pytest.approx(float(len(graph.edges) - graph.n_nodes))
    assert maximum == pytest.approx(2.0 * graph.n_nodes)


@pytest.mark.edge_case
@pytest.mark.parametrize("move", [PottsMove.SWENDSEN_WANG, PottsMove.WOLFF])
def test_a_cluster_move_refuses_this_instance(move: PottsMove) -> None:
    # The second job of this fixture, and the reason it is not merely a
    # physics curiosity. `1 - exp(-J)` is not a probability at `J < 0` and an
    # antiferromagnet has no like-spin clusters, so the refusal in
    # `sample_potts` is correct -- but it was reachable only by a hand-written
    # negative coupling until this fixture existed.
    graph = frustrated_triangular_lattice((3, 3))

    with pytest.raises(ValueError, match="needs every coupling >= 0"):
        sample_potts(graph, ZERO_FIELD, move=move, n_sweeps=4, burn_in=0, seed=1)


@pytest.mark.edge_case
def test_single_site_sampling_still_runs_on_the_frustrated_instance() -> None:
    # The refusal above is specific to the cluster construction, not a blanket
    # ban on a negative coupling. Without this the test above would also pass
    # if `sample_potts` rejected the graph outright.
    graph = frustrated_triangular_lattice((3, 3))

    chain = sample_potts(
        graph, ZERO_FIELD, move=PottsMove.SINGLE_SITE, n_sweeps=20, burn_in=5, seed=1
    )

    assert chain.states.shape == (20, graph.n_nodes)


@pytest.mark.structural
def test_the_residual_entropy_is_reported_and_not_asserted() -> None:
    # Wannier (1950) gives 0.3231 per site for the *infinite* lattice. The
    # finite-size values are not close to it and, measured here, are not even
    # monotone on the way: ground-state degeneracy 42 at N = 9, 68 at N = 12
    # and 42 again at N = 16, giving 0.4153, 0.3516 and 0.2336 per site. The
    # 4x4 torus is the anomaly -- neither extent is divisible by 3, so it is
    # incommensurate with the three-sublattice ground-state structure.
    #
    # So this asserts the degeneracies, which are exact, and reports the
    # entropy. Asserting convergence to Wannier's constant at these sizes
    # would be the mistake #214 already made once with a graph limit theorem.
    degeneracies = {
        shape: _agreeing_edges(frustrated_triangular_lattice(shape))[1]
        for shape in [(3, 3), (3, 4), (4, 4)]
    }

    assert degeneracies == {(3, 3): 42, (3, 4): 68, (4, 4): 42}

    entropies = [
        math.log(count) / (shape[0] * shape[1]) for shape, count in degeneracies.items()
    ]
    assert entropies == pytest.approx([0.4153, 0.3516, 0.2336], abs=1e-4)
    # Non-monotone, and every value is on the far side of the limit. Both
    # statements are the reason the constant is not asserted.
    assert entropies[1] < entropies[0]
    assert entropies[2] < entropies[1]
    assert max(entropies) > WANNIER_RESIDUAL_ENTROPY


@pytest.mark.structural
def test_an_open_boundary_is_frustrated_but_has_no_closed_form() -> None:
    # The counting argument needs every triangle complete, which an open
    # boundary breaks. The instance is still frustrated -- 4 agreeing edges of
    # 16 at 3x3 -- but the answer is enumerated rather than predicted, and
    # `minimum_frustrated_edges` refuses rather than returning `N`.
    graph = frustrated_triangular_lattice((3, 3), BoundaryCondition.OPEN)

    lowest, degeneracy = _agreeing_edges(graph)

    assert (lowest, degeneracy) == (4, 2)
    assert lowest != graph.n_nodes
    with pytest.raises(ValueError, match="only under periodic boundaries"):
        minimum_frustrated_edges(graph)


@pytest.mark.edge_case
def test_a_ferromagnetic_coupling_on_this_graph_is_refused() -> None:
    with pytest.raises(ValueError, match="coupling must be negative"):
        frustrated_triangular_lattice((3, 3), coupling=1.0)


@pytest.mark.edge_case
def test_a_periodic_extent_below_three_is_refused() -> None:
    # At extent 2 the "+1" and "-1" neighbours coincide, so the wrap doubles a
    # bond -- which `PottsGraph` permits and which silently doubles that
    # edge's weight, breaking the 3N edge count the argument rests on.
    with pytest.raises(ValueError, match="periodic extents must be >= 3"):
        frustrated_triangular_lattice((2, 4))


# --- 2. The planted Viana-Bray spin glass --------------------------------


@pytest.mark.edge_case
@pytest.mark.mathematical
def test_zero_frustration_is_a_gauge_transform_of_the_ferromagnet() -> None:
    # Stated in the docstring as the reason not to use `frustration = 0` as a
    # hard case, and checked here rather than trusted: with the gauge
    # `sigma_i = +/-1` read off the planted state, every coupling becomes
    # positive. An instance that is a relabelled ferromagnet is solved by any
    # local search.
    instance = planted_spin_glass(40, 4.0, 0.0, np.random.default_rng(1))

    sign = np.where(instance.planted == 0, 1.0, -1.0)
    gauged = [
        weight * sign[first] * sign[second]
        for (first, second), weight in zip(
            instance.graph.edges, instance.graph.coupling, strict=True
        )
    ]

    assert instance.unsatisfied == 0
    assert gauged == pytest.approx([1.0] * len(gauged))


@pytest.mark.oracle
@pytest.mark.parametrize(
    ("frustration", "expected_hits"),
    [(0.0, 60), (0.1, 41), (0.2, 23), (0.3, 10), (0.5, 0)],
)
def test_the_planted_state_stops_being_the_ground_state_as_frustration_rises(
    frustration: float, expected_hits: int
) -> None:
    # The honest limit of planting, measured against enumeration at `n = 10`.
    # The planted state is a state of *known energy* and therefore an upper
    # bound on the ground-state energy at any size. It is not the ground
    # state, and pretending otherwise would make every claim built on it
    # wrong.
    rng = np.random.default_rng(7)

    hits = 0
    for _ in range(60):
        instance = planted_spin_glass(10, 3.0, frustration, rng)
        lowest = min(
            energy(
                instance.graph,
                np.zeros((instance.graph.n_nodes, 2)),
                np.array(assignment, dtype=np.int64),
            )
            for assignment in itertools.product(range(2), repeat=instance.graph.n_nodes)
        )
        assert instance.planted_energy >= lowest - 1e-12, "planting must upper bound"
        hits += abs(instance.planted_energy - lowest) < 1e-12

    assert hits == expected_hits


@pytest.mark.oracle
def test_the_planted_energy_matches_the_energy_of_the_planted_state() -> None:
    # The construction records its own answer, so a drift between the recorded
    # energy and the model's would make every comparison against it wrong.
    # Checked against `phylo.search.alpha_expansion.energy`, which shares no
    # code with the construction.
    instance = planted_spin_glass(12, 4.0, 0.25, np.random.default_rng(3))

    assert instance.planted_energy == pytest.approx(
        energy(instance.graph, np.zeros((instance.graph.n_nodes, 2)), instance.planted)
    )


@pytest.mark.structural
def test_the_couplings_carry_both_signs_at_positive_frustration() -> None:
    # Otherwise it is not a spin glass. `+/- J` means the magnitude is fixed
    # and only the sign varies, which is also checked.
    instance = planted_spin_glass(30, 4.0, 0.3, np.random.default_rng(4))

    signs = {math.copysign(1.0, weight) for weight in instance.graph.coupling}
    assert signs == {1.0, -1.0}
    assert {abs(weight) for weight in instance.graph.coupling} == {1.0}


@pytest.mark.edge_case
@pytest.mark.parametrize(
    ("n_nodes", "mean_degree", "frustration", "magnitude", "message"),
    [
        (1, 3.0, 0.1, 1.0, "n_nodes must be at least 2"),
        (5, -1.0, 0.1, 1.0, "mean_degree must be non-negative"),
        (5, 3.0, 1.5, 1.0, r"frustration must lie in \[0, 1\]"),
        (5, 3.0, 0.1, 0.0, "magnitude must be positive"),
    ],
)
def test_an_out_of_range_parameter_is_refused(
    n_nodes: int,
    mean_degree: float,
    frustration: float,
    magnitude: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        planted_spin_glass(
            n_nodes,
            mean_degree,
            frustration,
            np.random.default_rng(0),
            magnitude=magnitude,
        )


@pytest.mark.structural
@pytest.mark.release
def test_past_enumeration_the_planted_state_is_a_reference_not_a_hard_case() -> None:
    # The claim this fixture was proposed to support, and the measurement that
    # narrows it. At `n = 100` no enumeration is possible, so the planted
    # energy is the only reference available -- that part holds.
    #
    # What does *not* hold is that it defeats a baseline. Measured against
    # 20-restart iterated conditional modes at mean degree 4:
    #
    #   frustration   ICM - planted (mean over 10 instances)
    #   0.00                  +0.30
    #   0.05                  -0.12
    #   0.10                  +0.12
    #   0.15                  -0.25
    #   0.20                  -8.20
    #   0.30                 -25.70
    #
    # Below 0.2 single-site descent lands on the planted energy, so the
    # instance is not hard. At and above it descent *beats* the planted state,
    # so the planted energy is no longer near the optimum and is a weak
    # reference. Raising connectivity does not open a window either: at mean
    # degree 12 and frustration 0.05, descent matches the planted energy
    # exactly on every instance.
    #
    # So this answers the ticket's open question in the negative: a planted
    # Viana-Bray instance does not replace #177 as the case no baseline
    # solves. It supplies a known-energy reference past enumeration, which is
    # a real and separate thing, and that is what it is used for.
    rng = np.random.default_rng(20260904)
    field = np.zeros((100, 2))

    reachable = []
    for frustration in (0.05, 0.30):
        instance = planted_spin_glass(100, 4.0, frustration, rng)
        best = min(
            iterated_conditional_modes(instance.graph, field, 2, seed=seed)[1]
            for seed in range(20)
        )
        reachable.append(best - instance.planted_energy)

    # Near the planted energy at low frustration; well past it at high.
    assert abs(reachable[0]) <= 5.0
    assert reachable[1] < -10.0


# --- 3. The ambiguous HMM ------------------------------------------------


@pytest.mark.oracle
def test_viterbi_and_posterior_decoding_disagree_on_this_fixture() -> None:
    # The whole reason the fixture exists. A decoder that computes the single
    # best path and reports it as the per-site maximum -- or the reverse --
    # passes every fixture where the two agree, which is most of them.
    params = ambiguous_hmm()

    result = enumerate_hidden_paths(params, AMBIGUOUS_OBSERVATIONS)

    assert not result.decoders_agree()
    assert list(result.viterbi) == [0, 0, 0, 0, 0]
    assert list(result.posterior_path) == [0, 1, 0, 1, 0]
    # Different at two of five sites, so this is not a single-site tie-break.
    assert int((result.viterbi != result.posterior_path).sum()) == 2


@pytest.mark.oracle
def test_the_viterbi_path_is_unique_by_a_stated_margin() -> None:
    # Without this the disagreement above could be an `argmax` tie-break, and
    # a correct decoder returning the other tied path would "fail". A
    # symmetric version of this fixture had a *three-way* tie at the maximum,
    # which is why the parameters are asymmetric.
    params = ambiguous_hmm()

    ranked = sorted(
        (
            path_log_probability(
                params, np.array(path, dtype=np.int64), AMBIGUOUS_OBSERVATIONS
            )
            for path in itertools.product(range(2), repeat=5)
        ),
        reverse=True,
    )

    assert ranked[0] == pytest.approx(-6.110066, abs=1e-6)
    assert ranked[0] - ranked[1] == pytest.approx(0.303296, abs=1e-6)


@pytest.mark.oracle
def test_the_posterior_marginals_are_decisive_at_every_site() -> None:
    # The mirror of the test above, for the other decoder. If some marginal
    # sat at 0.5 the posterior path would also be a tie-break, and the
    # disagreement would be an artifact rather than a property of the model.
    params = ambiguous_hmm()

    result = enumerate_hidden_paths(params, AMBIGUOUS_OBSERVATIONS)

    assert result.posterior.max(axis=1) == pytest.approx(
        [0.7566, 0.6256, 0.6835, 0.6256, 0.7566], abs=1e-4
    )
    assert float(result.posterior.max(axis=1).min()) > 0.6


@pytest.mark.oracle
def test_the_posterior_path_is_a_poor_path_and_that_is_the_point() -> None:
    # Posterior decoding maximizes each site's marginal, which says nothing
    # about the sequence as a whole: the path it returns here is the 5th most
    # likely of 32, 0.6066 nats below the Viterbi path. Reporting it as "the
    # most likely hidden path" is the error this fixture catches.
    params = ambiguous_hmm()
    result = enumerate_hidden_paths(params, AMBIGUOUS_OBSERVATIONS)

    posterior_log_probability = path_log_probability(
        params, result.posterior_path, AMBIGUOUS_OBSERVATIONS
    )
    ranked = sorted(
        (
            path_log_probability(
                params, np.array(path, dtype=np.int64), AMBIGUOUS_OBSERVATIONS
            )
            for path in itertools.product(range(2), repeat=5)
        ),
        reverse=True,
    )

    assert result.viterbi_log_probability - posterior_log_probability == pytest.approx(
        0.606592, abs=1e-6
    )
    rank = sum(value > posterior_log_probability + 1e-12 for value in ranked)
    assert rank == 4


@pytest.mark.structural
def test_the_fixture_declares_the_length_its_observations_have() -> None:
    # A mismatch would leave every consumer slicing or padding, and the
    # simulated dataset would not be the sequence the decodings are pinned on.
    params = ambiguous_hmm()

    assert params.sequence_length == AMBIGUOUS_OBSERVATIONS.shape[0]
    assert params.transition.sum(axis=1) == pytest.approx([1.0, 1.0])
    assert params.emission.sum(axis=1) == pytest.approx([1.0, 1.0])
