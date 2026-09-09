"""The tropical Grassmannian relaxation against enumeration and the Hadamard closed form.

Issue #408. ``ROADMAP.md`` Stage 3 records the differentiable-topology half
as blocked on an oracle. Two are used here and neither is optional.

**Enumeration**, at 5 to 8 taxa, where every unrooted topology can be scored
on the quartet surface: the relaxation at a corner is the discrete objective,
the corner is a tree metric and the softmin's weight leakage is bounded, and
gradient ascent is held to the enumerated maximum rather than to whether it
improved.

**The Hadamard conjugation**, ``eq:hadamard``, which on two-state data
returns the tree's own branch lengths on its splits and zero elsewhere. Its
metric is a point of the tropical Grassmannian computed by a route that
shares no algebra with :func:`~snakes_and_ladders.likelihood.distance.tree_distances`
--- a Walsh--Hadamard transform of pattern frequencies against a walk over
the tree --- so the four-point condition holding on it, to ``1e-12``, is a
statement about the model rather than about either implementation.

**The corner tolerance is relative and is derived.**
:func:`~snakes_and_ladders.sandbox.tropical.temperature_for` returns the
``tau`` at which the softmin leaks under ``1e-11`` of score off a corner's own
resolutions; measured, what is then left is float64 rounding of a sum over
quartets, at ``3.8e-16`` of the value across 5, 6, 7 and 8 taxa. So the
agreement is pinned *relatively*, for the reason root ``CLAUDE.md`` gives
about a log-likelihood: the absolute ``1e-11`` the Gumbel-softmax half was
held to is that half's problem size, and at 340,000 in magnitude the same
claim is 3 ulp rather than a defect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from snakes_and_ladders.likelihood.distance import distance_matrix, tree_distances
from snakes_and_ladders.likelihood.hadamard import (
    binary_recoding,
    edge_spectrum,
    expected_spectrum,
    hadamard_conjugation,
    sequence_spectrum,
    split_weights,
)
from snakes_and_ladders.sandbox.tropical import (
    MINIMUM_TEMPERATURE,
    PAIRINGS,
    QuartetTable,
    anneal,
    condensed,
    corner,
    corner_bound,
    discrete_score,
    metric_from_split_weights,
    optimize,
    pairing_positions,
    quartet_indices,
    quartet_table,
    relaxed_score,
    resolutions,
    temperature_for,
)
from snakes_and_ladders.search.infer import score_topology
from snakes_and_ladders.search.neighbor_joining import (
    four_point_violation,
    neighbor_joining,
)
from snakes_and_ladders.search.topology import (
    Topology,
    enumerate_topologies,
    leaf_bipartitions,
    normalized_robinson_foulds,
)
from snakes_and_ladders.sim.params import SimulationParams
from snakes_and_ladders.sim.simulate import simulate_alignment

from tests._fixtures import EIGHT_TAXA, load_fixture

FIVE_TAXA = "tree_search/ci.yaml"
SIX_TAXA = "tree_search/stress.yaml"
SEVEN_TAXA = "tree_search/release.yaml"

#: The corner agreement asked of :func:`temperature_for`, and the absolute
#: bound the softmin is then held to. It is the tolerance the Gumbel-softmax
#: half was held to, carried over so the two halves of the roadmap bullet
#: state the same claim.
SOFTMIN_TOLERANCE = 1e-11

#: What is left once the softmin is under :data:`SOFTMIN_TOLERANCE`: float64
#: rounding of a sum over ``C(n, 4)`` quartets. Measured at 2.9e-16, 3.0e-16,
#: 3.4e-16 and 3.8e-16 of the value at 5, 6, 7 and 8 taxa, on fixtures whose
#: scores span 25,000 to 340,000, and pinned here with a factor of 26 of
#: headroom over the worst.
CORNER_RELATIVE_TOLERANCE = 1e-14

#: Two D values inside this of each other are the same value: the quartet
#: scores are maximized log-likelihoods and agree only to the optimizer's own
#: convergence, so a topology ranking that turns on less is not a property of
#: the model (``search/CLAUDE.md``).
CONVERGENCE_TIE = 1e-6

#: Sites of the eight-taxon fixture the quartet table is fitted on. The
#: fixture declares 200,000, which is 3.91 s per four-taxon fit against 0.12 s
#: here and 210 fits either way; the shortening is
#: ``tests/regression/search/test_search_parsimony.py``'s, on the same fixture
#: and the same seed.
EIGHT_TAXON_SITES = 1000

#: Ascent budget, fixed across every size so a recovery rate is comparable
#: between sizes.
ANNEAL_FROM, ANNEAL_TO, STEPS, LEARNING_RATE = 0.5, 0.02, 300, 0.05


@dataclass(frozen=True)
class Instance:
    """One fixture, its quartet table, and the enumerated quartet surface."""

    params: SimulationParams
    alignment: dict[str, np.ndarray]
    k: int
    table: QuartetTable
    positions: np.ndarray
    scores: dict[frozenset[frozenset[str]], float]

    @property
    def names(self) -> list[str]:
        return list(self.table.names)

    @property
    def best(self) -> float:
        return max(self.scores.values())

    @property
    def best_key(self) -> frozenset[frozenset[str]]:
        return max(self.scores, key=lambda key: self.scores[key])

    def start(self) -> np.ndarray:
        """The estimated distance matrix, condensed: what a caller already has."""
        _, matrix, _ = distance_matrix(self.alignment, self.k)
        return condensed(matrix)


def _alignment(
    name: str, n_sites: int | None = None
) -> tuple[SimulationParams, dict[str, np.ndarray]]:
    params = load_fixture(name)
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        rng=np.random.default_rng(params.seed),
        n_sites=params.n_sites if n_sites is None else n_sites,
    )
    return params, dict(dataset.alignment)


def _instance(name: str, n_sites: int | None = None) -> Instance:
    """Load a fixture, fit its quartet table, and enumerate the quartet surface.

    ``n_sites`` shortens the alignment the fixture declares, as
    ``tests/regression/search/test_search_parsimony.py`` shortens the same
    eight-taxon fixture: a four-taxon fit costs 3.91 s at the declared
    200,000 sites and 0.12 s at 1,000, and 210 of them is the whole expense
    of the largest size enumeration referees.
    """
    params, alignment = _alignment(name, n_sites)
    k = params.k
    table = quartet_table(alignment, k)
    scores = {
        leaf_bipartitions(topology): discrete_score(table, topology)
        for topology in enumerate_topologies(sorted(alignment))
    }
    return Instance(
        params=params,
        alignment=alignment,
        k=k,
        table=table,
        positions=pairing_positions(table.n_taxa),
        scores=scores,
    )


@pytest.fixture(scope="module")
def five_taxon() -> Instance:
    """15 topologies and 15 four-taxon fits: 2.7 s, and every test below shares it."""
    return _instance(FIVE_TAXA)


@pytest.fixture(scope="module")
def six_taxon() -> Instance:
    """105 topologies and 45 fits: 4.1 s, behind the release gate with its users."""
    return _instance(SIX_TAXA)


@pytest.fixture(scope="module")
def seven_taxon() -> Instance:
    """945 topologies and 105 fits: 10.1 s."""
    return _instance(SEVEN_TAXA)


@pytest.fixture(scope="module")
def eight_taxon() -> Instance:
    """10,395 topologies and 210 fits at 1,000 sites: 22 s."""
    return _instance(EIGHT_TAXA, EIGHT_TAXON_SITES)


def _relative_corner_gap(instance: Instance, topology: Topology) -> float:
    """``|F_tau - D|`` at a corner, relative, with ``tau`` derived from the corner."""
    metric = corner(topology)
    temperature = temperature_for(
        instance.table, instance.positions, metric, SOFTMIN_TOLERANCE
    )
    discrete = discrete_score(instance.table, topology)
    relaxed = float(
        relaxed_score(
            instance.table,
            instance.positions,
            torch.from_numpy(metric),
            temperature,
        )
    )
    return abs(relaxed - discrete) / abs(discrete)


# --- the relaxation is an extension: every corner ------------------------


@pytest.mark.oracle
def test_the_relaxation_equals_the_discrete_score_at_every_corner(
    five_taxon: Instance,
) -> None:
    # The claim the whole method rests on, and it is checked at every one of
    # the 15 corners rather than at the generating tree: a relaxation that
    # is an extension only near the truth optimizes a different problem
    # everywhere else. `tau` is derived per corner from that corner's own
    # quartet gaps, so this is one statement and not a tuned constant.
    worst = max(
        _relative_corner_gap(five_taxon, topology)
        for topology in enumerate_topologies(five_taxon.names)
    )

    assert worst < CORNER_RELATIVE_TOLERANCE, (
        f"worst relative corner gap {worst:.3g} over 15 corners"
    )


@pytest.mark.oracle
def test_the_softmin_bound_is_what_the_derived_temperature_certifies(
    five_taxon: Instance,
) -> None:
    # `temperature_for` inverts `corner_bound`, so the bound at the returned
    # temperature must be under what was asked for. Without this the corner
    # test above could pass on rounding alone at a temperature that resolved
    # nothing.
    for topology in enumerate_topologies(five_taxon.names):
        metric = corner(topology)
        temperature = temperature_for(
            five_taxon.table, five_taxon.positions, metric, SOFTMIN_TOLERANCE
        )
        bound = corner_bound(
            five_taxon.table, five_taxon.positions, metric, temperature
        )
        assert bound <= SOFTMIN_TOLERANCE
        assert temperature > MINIMUM_TEMPERATURE


@pytest.mark.mathematical
@pytest.mark.parametrize("temperature", [0.5, 0.2, 0.1, 0.05, 0.02, 0.01])
def test_the_corner_bound_holds_at_every_temperature(
    five_taxon: Instance, temperature: float
) -> None:
    # The bound is a bound, not an estimate: it must hold where the softmin
    # is loose as well as where it is tight, and at tau = 0.5 the leakage is
    # thousands of log-likelihood units.
    for topology in enumerate_topologies(five_taxon.names):
        metric = corner(topology)
        discrete = discrete_score(five_taxon.table, topology)
        relaxed = float(
            relaxed_score(
                five_taxon.table,
                five_taxon.positions,
                torch.from_numpy(metric),
                temperature,
            )
        )
        bound = corner_bound(
            five_taxon.table, five_taxon.positions, metric, temperature
        )
        assert abs(relaxed - discrete) <= bound + abs(discrete) * 1e-14


# --- the coordinates: what makes them the tropical Grassmannian ----------


@pytest.mark.mathematical
@pytest.mark.parametrize("factor", [0.1, 3.7, 100.0])
def test_the_metric_scale_is_a_gauge(five_taxon: Instance, factor: float) -> None:
    # F_tau(c d) = F_tau(d) exactly, by the unit-mean normalization. Without
    # it ascent maximizes F by inflating the metric and never moves a
    # resolution, which is the failure the module docstring names.
    metric = five_taxon.start()
    at_one = float(
        relaxed_score(
            five_taxon.table, five_taxon.positions, torch.from_numpy(metric), 0.05
        )
    )
    at_factor = float(
        relaxed_score(
            five_taxon.table,
            five_taxon.positions,
            torch.from_numpy(factor * metric),
            0.05,
        )
    )

    assert_allclose(at_factor, at_one, rtol=1e-14)


@pytest.mark.oracle
@pytest.mark.parametrize("fixture_name", [FIVE_TAXA, SIX_TAXA, SEVEN_TAXA])
def test_the_combinatorial_resolution_is_the_tropical_plucker_argmin(
    fixture_name: str,
) -> None:
    # Two routes to a quartet's topology: the leaf bipartitions of the tree,
    # and the argmin of the three pairing sums of its metric. The second is
    # the tropical Plucker relation and is what the relaxation smooths, so
    # the relaxation is optimizing the topology only if the two agree. No
    # fits here -- this is the metric against the combinatorics -- so all
    # 945 topologies at seven taxa are affordable per pull request.
    params = load_fixture(fixture_name)
    names = sorted(_leaves(params.tau))
    quartets = quartet_indices(len(names))
    positions = pairing_positions(len(names))
    for topology in enumerate_topologies(names):
        metric = corner(topology)
        sums = metric[positions[..., 0]] + metric[positions[..., 1]]
        assert_allclose(resolutions(quartets, names, topology), sums.argmin(axis=1))


@pytest.mark.mathematical
def test_the_gradient_matches_central_differences(five_taxon: Instance) -> None:
    # There is no sampled estimator here, so there is no estimator bias to
    # measure: the relaxation is deterministic and its gradient is exact.
    # What is checked is the one thing that can still be wrong, that autodiff
    # differentiates the function actually evaluated.
    metric = five_taxon.start()
    parameters = torch.from_numpy(metric).clone().requires_grad_(True)
    value = relaxed_score(five_taxon.table, five_taxon.positions, parameters, 0.05)
    value.backward()  # type: ignore[no-untyped-call]
    assert parameters.grad is not None
    analytic = parameters.grad.numpy().copy()

    step = 1e-6
    numeric = np.empty_like(analytic)
    for index in range(metric.shape[0]):
        moved = metric.copy()
        moved[index] += step
        up = float(
            relaxed_score(
                five_taxon.table,
                five_taxon.positions,
                torch.from_numpy(moved),
                0.05,
            )
        )
        moved[index] -= 2 * step
        down = float(
            relaxed_score(
                five_taxon.table,
                five_taxon.positions,
                torch.from_numpy(moved),
                0.05,
            )
        )
        numeric[index] = (up - down) / (2 * step)

    assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-6)


# --- the quartet surface is a different surface, and must agree at the argmax


@pytest.mark.oracle
def test_the_quartet_argmax_is_the_likelihood_argmax(five_taxon: Instance) -> None:
    # `search/CLAUDE.md` licenses a cheap objective only by measuring that it
    # agrees with the expensive one at the argmax. D is a quartet
    # decomposition and not the tree's log-likelihood, so this is the
    # measurement that lets everything above be read as a statement about
    # tree search. 15 full fits, 2.1 s.
    likelihoods = {
        leaf_bipartitions(topology): score_topology(
            topology, five_taxon.alignment, five_taxon.k
        )
        for topology in enumerate_topologies(five_taxon.names)
    }
    by_likelihood = max(likelihoods, key=lambda key: likelihoods[key])
    ranked = sorted(five_taxon.scores, key=lambda key: -five_taxon.scores[key])

    assert by_likelihood == five_taxon.best_key
    assert ranked.index(by_likelihood) == 0


# --- ascent, held to the enumerated maximum ------------------------------


def _ascend(instance: Instance, metric: np.ndarray) -> float:
    return optimize(
        instance.table,
        metric,
        temperature=ANNEAL_FROM,
        final_temperature=ANNEAL_TO,
        steps=STEPS,
        learning_rate=LEARNING_RATE,
    ).score


@pytest.mark.oracle
def test_ascent_reaches_the_enumerated_maximum_from_random_metrics(
    five_taxon: Instance,
) -> None:
    # Random *metrics*, not random topologies: the relaxation's start is a
    # point of the ambient space, and starting it at the estimated distance
    # matrix would test neighbor joining's answer rather than the ascent's.
    rng = np.random.default_rng(20260909)
    start = five_taxon.start()
    reached = [
        _ascend(
            five_taxon,
            np.exp(rng.normal(np.log(start.mean()), 0.5, size=start.shape)),
        )
        for _ in range(8)
    ]

    assert all(abs(value - five_taxon.best) < CONVERGENCE_TIE for value in reached), (
        f"best {five_taxon.best:.4f}, reached {reached}"
    )


@pytest.mark.oracle
def test_ascent_leaves_the_grassmannian_and_is_projected_back(
    five_taxon: Instance,
) -> None:
    # F is optimized over the ambient space, not over the tree locus, so the
    # point ascent stops at is generally not a tree metric. The claim is not
    # that it stays on the locus -- the measured violation is 0.2 in units
    # where the metric has mean 1 -- but that the answer read off it is
    # still the enumerated maximum.
    result = optimize(
        five_taxon.table,
        five_taxon.start(),
        temperature=ANNEAL_FROM,
        final_temperature=ANNEAL_TO,
        steps=STEPS,
        learning_rate=LEARNING_RATE,
    )

    assert result.plucker_violation > 0.0
    assert abs(result.score - five_taxon.best) < CONVERGENCE_TIE
    assert normalized_robinson_foulds(result.topology, five_taxon.params.tau) == 0.0


# --- the Hadamard closed form -------------------------------------------


def _leaves(node: Topology) -> frozenset[str]:
    if node.is_leaf:
        return frozenset((node.name,))
    return frozenset().union(*(_leaves(child) for child in node.children))


@pytest.mark.oracle
@pytest.mark.parametrize("fixture_name", [FIVE_TAXA, SIX_TAXA, SEVEN_TAXA])
def test_the_hadamard_closed_form_lands_on_the_tropical_grassmannian(
    fixture_name: str,
) -> None:
    # The exact spectrum of eq:hadamard, inverted, gives a weight per split
    # -- including every split the tree does not have, which must come back
    # zero. Summing those weights over the splits that separate two taxa is
    # a route to the metric that shares no algebra with a walk over the
    # tree, so the four-point condition holding on it is a statement about
    # the model. This is the second oracle of the ticket, and it needs no
    # fit and no alignment.
    params = load_fixture(fixture_name)
    names, spectrum = edge_spectrum(params.tau)
    recovered = hadamard_conjugation(expected_spectrum(spectrum))
    metric = metric_from_split_weights(split_weights(recovered, names), names)

    _, exact = tree_distances(params.tau)
    assert_allclose(metric, exact, atol=1e-12)
    assert four_point_violation(metric) < 1e-12


@pytest.mark.oracle
@pytest.mark.parametrize("fixture_name", [FIVE_TAXA, SIX_TAXA])
def test_the_hadamard_metric_resolves_every_quartet_as_the_tree_does(
    fixture_name: str,
) -> None:
    # The relaxation reads a topology out of a metric by the argmin of the
    # pairing sums; on the closed form's metric that argmin must be the
    # generating tree's own resolution at every quartet. Where the test
    # above pins the coordinates, this pins what the relaxation makes of
    # them.
    params = load_fixture(fixture_name)
    names, spectrum = edge_spectrum(params.tau)
    recovered = hadamard_conjugation(expected_spectrum(spectrum))
    metric = condensed(
        metric_from_split_weights(split_weights(recovered, names), names)
    )
    positions = pairing_positions(len(names))
    sums = metric[positions[..., 0]] + metric[positions[..., 1]]

    assert_allclose(
        sums.argmin(axis=1),
        resolutions(quartet_indices(len(names)), names, params.tau),
    )


@pytest.mark.oracle
@pytest.mark.simulated_truth
def test_the_two_state_recoding_resolves_every_quartet_as_the_tree_does(
    five_taxon: Instance,
) -> None:
    # A four-state Jukes-Cantor alignment reduces to the two-state model at
    # `recoding_scale` of every branch length, so the metric the conjugation
    # returns from the recoded alignment is a scale change of the true one
    # that the model produced rather than a constant multiplied in by hand.
    # What is asserted is what data at this site count supports: the
    # resolutions, and the topology. The scale itself is an estimate here --
    # its largest deviation from 2/3 of the truth is 0.049 at 1,200 sites --
    # and is pinned exactly on the closed form's own spectrum by the two
    # tests above instead.
    params = five_taxon.params
    recoded = dict(binary_recoding(five_taxon.alignment, five_taxon.k))
    names, spectrum = sequence_spectrum(recoded)
    weights = split_weights(hadamard_conjugation(spectrum), names)
    metric = metric_from_split_weights(weights, names)

    assert (
        normalized_robinson_foulds(neighbor_joining(names, metric), params.tau) == 0.0
    )

    positions = pairing_positions(len(names))
    flat = condensed(metric)
    sums = flat[positions[..., 0]] + flat[positions[..., 1]]
    assert_allclose(
        sums.argmin(axis=1),
        resolutions(quartet_indices(len(names)), names, params.tau),
    )


# --- refusals ------------------------------------------------------------


@pytest.mark.edge_case
def test_a_temperature_below_the_floor_is_refused(five_taxon: Instance) -> None:
    with pytest.raises(ValueError, match="temperature must be"):
        relaxed_score(
            five_taxon.table,
            five_taxon.positions,
            torch.from_numpy(five_taxon.start()),
            MINIMUM_TEMPERATURE / 2,
        )


@pytest.mark.edge_case
def test_fewer_than_four_taxa_have_no_quartet() -> None:
    with pytest.raises(ValueError, match="at least 4 taxa"):
        quartet_indices(3)


@pytest.mark.edge_case
def test_a_degenerate_quartet_admits_no_temperature(five_taxon: Instance) -> None:
    # Every distance equal makes all three pairing sums equal, so no
    # temperature separates them. Refused rather than returning a
    # temperature that certifies nothing.
    flat = np.ones(five_taxon.start().shape[0])
    with pytest.raises(ValueError, match="no temperature resolves it"):
        temperature_for(five_taxon.table, five_taxon.positions, flat, 1e-11)


@pytest.mark.edge_case
def test_a_non_positive_tolerance_is_refused(five_taxon: Instance) -> None:
    with pytest.raises(ValueError, match="tolerance must be positive"):
        temperature_for(five_taxon.table, five_taxon.positions, five_taxon.start(), 0.0)


@pytest.mark.edge_case
def test_a_non_positive_starting_distance_is_refused(five_taxon: Instance) -> None:
    start = five_taxon.start()
    start[0] = 0.0
    with pytest.raises(ValueError, match="must be positive"):
        optimize(five_taxon.table, start)


@pytest.mark.edge_case
def test_a_non_positive_branch_length_has_no_corner(five_taxon: Instance) -> None:
    topology = next(enumerate_topologies(five_taxon.names))
    with pytest.raises(ValueError, match="branch_length must be positive"):
        corner(topology, 0.0)


@pytest.mark.edge_case
@pytest.mark.parametrize(
    ("start", "end", "steps", "message"),
    [
        (0.5, 0.0, 10, "endpoints must be"),
        (0.01, 0.5, 10, "end must not exceed start"),
        (0.5, 0.01, 0, "steps must be at least"),
    ],
)
def test_the_schedule_refuses_an_unusable_range(
    start: float, end: float, steps: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        anneal(start, end, steps, 0)


@pytest.mark.structural
def test_the_pairing_order_is_the_four_point_condition_s() -> None:
    # `four_point_violation` forms its three sums in one order and this
    # module indexes PAIRINGS in the same one; a silent disagreement would
    # make every resolution wrong by a permutation and nothing else would
    # notice. Pinned by construction on a metric whose quartet is known.
    assert PAIRINGS == (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))
    metric = np.array(
        [
            [0.0, 1.0, 4.0, 4.0],
            [1.0, 0.0, 4.0, 4.0],
            [4.0, 4.0, 0.0, 1.0],
            [4.0, 4.0, 1.0, 0.0],
        ]
    )
    flat = condensed(metric)
    positions = pairing_positions(4)
    sums = flat[positions[..., 0]] + flat[positions[..., 1]]
    assert int(sums.argmin(axis=1)[0]) == 0


# --- the sizes only the release gate pays for ----------------------------


@pytest.mark.oracle
@pytest.mark.release
def test_the_relaxation_holds_at_six_taxa(six_taxon: Instance) -> None:
    # 105 topologies, and the full-likelihood enumeration beside it: 45
    # four-taxon fits and 105 six-taxon fits, 29 s.
    worst = max(
        _relative_corner_gap(six_taxon, topology)
        for topology in enumerate_topologies(six_taxon.names)
    )
    assert worst < CORNER_RELATIVE_TOLERANCE

    likelihoods = {
        leaf_bipartitions(topology): score_topology(
            topology, six_taxon.alignment, six_taxon.k
        )
        for topology in enumerate_topologies(six_taxon.names)
    }
    assert max(likelihoods, key=lambda key: likelihoods[key]) == six_taxon.best_key

    rng = np.random.default_rng(20260909)
    start = six_taxon.start()
    reached = [
        _ascend(
            six_taxon, np.exp(rng.normal(np.log(start.mean()), 0.5, size=start.shape))
        )
        for _ in range(8)
    ]
    assert all(abs(value - six_taxon.best) < CONVERGENCE_TIE for value in reached)


@pytest.mark.oracle
@pytest.mark.release
def test_at_seven_taxa_the_top_two_of_the_quartet_surface_are_tied(
    seven_taxon: Instance,
) -> None:
    # The negative result, and it is pinned rather than described. This
    # fixture's internal branches are ~0.02 (issue #177), and the top two
    # quartet scores come out 0.0117 apart in 339,982 -- 3.4e-8 relative,
    # inside the convergence of the fits that produced them. So "the
    # enumerated argmax" is not a target any method can be held to here, and
    # neither the relaxation nor neighbor joining returns it. What both do
    # return is the generating topology, which is the claim that survives.
    ordered = sorted(seven_taxon.scores.values(), reverse=True)
    separation = (ordered[0] - ordered[1]) / abs(ordered[0])
    assert separation < CONVERGENCE_TIE, (
        f"the top two quartet scores differ by {separation:.3g} relative; "
        "this test is the record that they do not"
    )

    result = optimize(
        seven_taxon.table,
        seven_taxon.start(),
        temperature=ANNEAL_FROM,
        final_temperature=ANNEAL_TO,
        steps=STEPS,
        learning_rate=LEARNING_RATE,
    )
    _, matrix, _ = distance_matrix(seven_taxon.alignment, seven_taxon.k)
    joined = neighbor_joining(seven_taxon.names, matrix)

    assert normalized_robinson_foulds(result.topology, seven_taxon.params.tau) == 0.0
    assert normalized_robinson_foulds(joined, seven_taxon.params.tau) == 0.0


@pytest.mark.oracle
@pytest.mark.release
def test_the_relaxation_holds_at_eight_taxa(eight_taxon: Instance) -> None:
    # The largest size at which enumeration referees a topology: 10,395 of
    # them. Two-state, so the table is 210 fits rather than the same number
    # at four times the pruning cost, and so the Hadamard oracle covers the
    # same instance.
    rng = np.random.default_rng(20260909)
    topologies = list(enumerate_topologies(eight_taxon.names))
    sample = [topologies[index] for index in rng.choice(len(topologies), 200, False)]
    worst = max(
        _relative_corner_gap(eight_taxon, topology)
        for topology in [*sample, eight_taxon.params.tau]
    )
    assert worst < CORNER_RELATIVE_TOLERANCE

    start = eight_taxon.start()
    reached = [
        _ascend(
            eight_taxon,
            np.exp(rng.normal(np.log(start.mean()), 0.5, size=start.shape)),
        )
        for _ in range(8)
    ]
    hits = sum(abs(value - eight_taxon.best) < CONVERGENCE_TIE for value in reached)

    # 7 of 8, and the rate is asserted rather than assumed zero. The miss
    # stops 15.8 below the maximum in 302,287 -- 5.2e-5 relative, outside the
    # tie above -- so it is a local optimum of the ascent and not two
    # topologies the fits cannot separate. This is the first size at which
    # ascent loses one, and the margin is stated so a worse run is reported
    # rather than hidden by a threshold set to the observation.
    assert hits >= 7, f"best {eight_taxon.best:.4f}, reached {reached}"
