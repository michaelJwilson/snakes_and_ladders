"""The generic Gibbs sampler and annealer, held to every distribution they target.

One sweep over the factor graph serves four problems, and each is pinned
where an exact answer exists: the Potts lattice against enumeration and,
draw for draw, against the specialised sweep it copies the arithmetic of;
the chain against the enumerated path posterior, for the single-site and
the exact block move; a tree at one site against the marginals sum-product
gives; the coupled model against the enumerated label posterior. Annealing is
held to the frustrated instance whose ground state has a closed form, and
the topology move to the flat-prior weight over fitted likelihoods at
temperature one (issue #309).
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from snakes_and_ladders.likelihood.hmm_paths import (
    PathEnumeration,
    emission_log_density,
    enumerate_hidden_paths,
    path_log_probability,
)
from snakes_and_ladders.likelihood.message_passing import sum_product
from snakes_and_ladders.likelihood.potts import log_weights
from snakes_and_ladders.likelihood.spatio_sequential import enumerate_spatio_sequential
from snakes_and_ladders.opt.schedule import Constant, Exponential
from snakes_and_ladders.search import gibbs
from snakes_and_ladders.search.backend import Backend
from snakes_and_ladders.search.gibbs import (
    _Indexed,
    anneal_factor_graph,
    anneal_topology,
    chain_block_sweep,
    gibbs_sweep,
    sample_factor_graph,
)
from snakes_and_ladders.search.infer import score_topology
from snakes_and_ladders.search.potts_mcmc import (
    _adjacency,
    _single_site_sweep,
    anneal_potts,
    energies,
)
from snakes_and_ladders.search.statistics import chi_square_p_value
from snakes_and_ladders.search.topology import (
    enumerate_topologies,
    leaf_bipartitions,
    nni_neighbours,
    random_topology,
)
from snakes_and_ladders.sim.canonical import (
    minimum_frustrated_edges,
)
from snakes_and_ladders.sim.factor_graph import (
    Factor,
    FactorGraph,
    Variable,
    from_hmm,
    from_potts,
    from_tree,
)
from snakes_and_ladders.sim.fixtures import fixture
from snakes_and_ladders.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from snakes_and_ladders.sim.hmm import simulate_sequences
from snakes_and_ladders.sim.jc import jc_transition_probabilities
from snakes_and_ladders.sim.params import load_simulation_params
from snakes_and_ladders.sim.simulate import simulate_alignment
from snakes_and_ladders.sim.spatio_sequential import (
    coupled_factor_graph,
    simulate_spatio_sequential,
)
from snakes_and_ladders.sim.tree import preorder

from tests._fixtures import FOUR_TAXA, fixture_path, load_fixture

SIGNIFICANCE = 0.001
FIELD = np.array([0.6, -0.4])


def _potts_pair() -> tuple[PottsGraph, FactorGraph]:
    graph = lattice_graph((2, 2), BoundaryCondition.OPEN, 0.8)
    return graph, from_potts(graph, FIELD)


@pytest.mark.simulated_truth
def test_the_generic_sweep_samples_the_potts_boltzmann_distribution() -> None:
    graph, factor_graph = _potts_pair()
    configurations = np.array(list(product(range(2), repeat=4)))
    weights = log_weights(graph, FIELD, configurations)
    expected = np.exp(weights - weights.max())
    expected /= expected.sum()

    chain = sample_factor_graph(
        factor_graph, np.random.default_rng(1), 10_000, burn_in=100, thin=5
    )

    assert chain.variables == ("s0", "s1", "s2", "s3")
    counts = np.zeros(len(configurations))
    for state in chain.states:
        counts[int(np.flatnonzero((configurations == state).all(axis=1))[0])] += 1
    assert chi_square_p_value(counts, len(chain.states) * expected) > SIGNIFICANCE


@pytest.mark.oracle
def test_the_generic_sweep_reproduces_the_potts_sweep_draw_for_draw() -> None:
    # Same uniforms, same site order, same cumulative search: the only way
    # the two could differ is a uniform within rounding of a boundary, and
    # over 2,000 sweeps of four sites none did. Asserted at 99 percent so a
    # single such draw does not fail the suite; realized at 100.
    graph, factor_graph = _potts_pair()
    adjacency = _adjacency(graph)
    indexed = _Indexed(factor_graph)
    generic, specialised = np.random.default_rng(5), np.random.default_rng(5)
    state_a = np.zeros(4, dtype=np.int64)
    state_b = state_a.copy()
    agreed = 0
    for _ in range(2000):
        gibbs_sweep(indexed, state_a, generic)
        _single_site_sweep(state_b, FIELD, adjacency, specialised)
        agreed += int(np.array_equal(state_a, state_b))
        state_b[:] = state_a

    assert agreed >= 1980, agreed


def _lattice_graph(extent: int) -> FactorGraph:
    lattice = lattice_graph((extent, extent), BoundaryCondition.OPEN, 0.6)
    return from_potts(lattice, np.array([0.3, -0.7, 0.15]))


@pytest.mark.oracle
@pytest.mark.parametrize("extent", [16, 32])
@pytest.mark.parametrize("seed", range(4))
def test_the_compiled_sweep_reproduces_the_numpy_one_bitwise(
    extent: int, seed: int
) -> None:
    # What lets the kernels be the default (#561, #563): the same uniforms in
    # the same order give the same states, exactly, not to a tolerance. The
    # sweep kernel gathers the conditional from one array of tables and NumPy
    # from a slice per factor, in the same order, so the sums are identical;
    # the exponential is where the two could part, and the kernel declines any
    # site whose draw comes within the last place of a cumulative boundary.
    # The density kernel reads the same array and takes no exponential, so
    # order is all it needs: it sums factors left to right in graph order, as
    # the dictionary oracle does. Realized: 8 of 8 runs agree on every state
    # and every log-density, 15,360 draws at 16x16 and 61,440 at 32x32, and
    # every compiled density equals the dictionary one to the last bit.
    graph = _lattice_graph(extent)

    numpy_chain = sample_factor_graph(
        graph, np.random.default_rng(seed), 15, backend=Backend.PYTHON
    )
    compiled = sample_factor_graph(
        graph, np.random.default_rng(seed), 15, backend=Backend.NUMBA
    )

    assert np.array_equal(numpy_chain.states, compiled.states)
    assert np.array_equal(numpy_chain.log_densities, compiled.log_densities)
    oracle = [
        graph.log_density(dict(zip(compiled.variables, map(int, state), strict=True)))
        for state in compiled.states
    ]
    assert np.array_equal(compiled.log_densities, np.array(oracle))


@pytest.mark.edge_case
def test_the_log_density_has_no_rust_backend() -> None:
    graph = _lattice_graph(4)
    indexed = _Indexed(graph)

    with pytest.raises(ValueError, match="no rust backend"):
        indexed.log_density(np.zeros(len(indexed.names), dtype=np.int64), Backend.RUST)


@pytest.mark.structural
def test_a_site_the_kernel_declines_is_decided_by_numpy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The handing back is what makes the pin exact rather than probable, and
    # no realistic draw reaches it -- so it is driven here instead, by a
    # guard wide enough that every site falls to NumPy. The chain must be the
    # same chain, which is a statement about resuming at the right position
    # with the right draw, not about arithmetic.
    graph = _lattice_graph(8)
    monkeypatch.setattr(gibbs, "_GUARD", 1e12)

    fallen_back = sample_factor_graph(
        graph, np.random.default_rng(11), 10, backend=Backend.NUMBA
    )

    expected = sample_factor_graph(
        graph, np.random.default_rng(11), 10, backend=Backend.PYTHON
    )
    assert np.array_equal(fallen_back.states, expected.states)


@pytest.mark.edge_case
def test_the_sweep_has_no_rust_backend() -> None:
    graph = _lattice_graph(4)

    with pytest.raises(ValueError, match="no rust backend"):
        sample_factor_graph(graph, np.random.default_rng(0), 2, backend=Backend.RUST)


def _chain() -> tuple[FactorGraph, list[tuple[int, ...]], np.ndarray]:
    rng = np.random.default_rng(2)
    n_states, length = 2, 5
    initial = rng.dirichlet(np.ones(n_states))
    transition = rng.dirichlet(np.ones(n_states), size=n_states)
    density = np.log(rng.dirichlet(np.ones(n_states), size=length))
    graph = from_hmm(np.log(initial), np.log(transition), density)
    paths = list(product(range(n_states), repeat=length))
    log_posterior = np.array(
        [
            np.log(initial[path[0]])
            + density[0, path[0]]
            + sum(
                np.log(transition[path[t - 1], path[t]]) + density[t, path[t]]
                for t in range(1, length)
            )
            for path in paths
        ]
    )
    posterior = np.exp(log_posterior - log_posterior.max())
    return graph, paths, posterior / posterior.sum()


@pytest.mark.simulated_truth
def test_the_generic_sweep_samples_the_hidden_path_posterior() -> None:
    graph, paths, posterior = _chain()

    chain = sample_factor_graph(
        graph, np.random.default_rng(3), 8000, burn_in=100, thin=4
    )

    counts = np.zeros(len(paths))
    for state in chain.states:
        counts[paths.index(tuple(int(v) for v in state))] += 1
    assert chi_square_p_value(counts, len(chain.states) * posterior) > SIGNIFICANCE


BLOCK_LENGTH = 8
BLOCK_DRAWS = 4000
#: Paths are lumped, most probable first, until a cell expects at least this
#: many draws: a chi-square over all 3**8 cells is invalid at any sample size
#: the CI budget holds, since almost every cell would expect far under one.
CELL_FLOOR = 25.0


def _enumerated_chain() -> tuple[FactorGraph, np.ndarray, PathEnumeration, np.ndarray]:
    """The declared HMM instance, its factor graph, and both enumerations of it.

    The observations are the fixture's own first sequence, truncated to a
    length the path enumeration reaches; the posterior the block move is held
    to is :mod:`snakes_and_ladders.likelihood.hmm_paths`', which shares no
    code with the sampler.
    """
    params = fixture("hmm", "ci").params
    observations = simulate_sequences(params).observations[0][:BLOCK_LENGTH]
    density = emission_log_density(params, observations)
    graph = from_hmm(np.log(params.initial), np.log(params.transition), density)
    joint = np.array(
        [
            path_log_probability(params, np.array(path), observations)
            for path in product(range(params.n_states), repeat=BLOCK_LENGTH)
        ]
    )
    posterior = np.exp(joint - joint.max())
    return (
        graph,
        observations,
        enumerate_hidden_paths(params, observations),
        posterior / posterior.sum(),
    )


def _lumped(posterior: np.ndarray, n_draws: int) -> list[list[int]]:
    """Paths in descending posterior order, cut wherever a cell expects ``CELL_FLOOR``."""
    cells: list[list[int]] = []
    current: list[int] = []
    mass = 0.0
    for index in np.argsort(-posterior):
        current.append(int(index))
        mass += float(posterior[index])
        if mass * n_draws >= CELL_FLOOR:
            cells.append(current)
            current, mass = [], 0.0
    cells[-1].extend(current)
    return cells


@pytest.mark.oracle
def test_the_block_move_draws_the_whole_chain_from_the_enumerated_path_posterior() -> (
    None
):
    # Every block draw is an independent sample from the posterior, so no
    # thinning is needed: that is what "exact" buys. Held to the 3**8 = 6,561
    # enumerated paths of the declared instance, both marginally and jointly.
    # Over 4,000 draws the largest per-site deviation from the enumerated
    # marginal is 0.0146, the smallest per-site chi-square p-value 0.062, and
    # the p-value of the joint over 36 lumped cells 0.251; over seeds 0 to 5
    # the smallest of either was 0.062.
    graph, _, enumerated, posterior = _enumerated_chain()
    names = [f"z{t}" for t in range(BLOCK_LENGTH)]
    paths = list(product(range(enumerated.posterior.shape[1]), repeat=BLOCK_LENGTH))
    index = {path: position for position, path in enumerate(paths)}
    state = np.zeros(BLOCK_LENGTH, dtype=np.int64)
    rng = np.random.default_rng(0)

    drawn = np.zeros(len(paths))
    marginal = np.zeros_like(enumerated.posterior)
    for _ in range(BLOCK_DRAWS):
        chain_block_sweep(graph, state, rng, names)
        drawn[index[tuple(int(value) for value in state)]] += 1
        marginal[np.arange(BLOCK_LENGTH), state] += 1

    assert np.abs(marginal / BLOCK_DRAWS - enumerated.posterior).max() < 0.03
    for site in range(BLOCK_LENGTH):
        assert (
            chi_square_p_value(marginal[site], BLOCK_DRAWS * enumerated.posterior[site])
            > SIGNIFICANCE
        )
    cells = _lumped(posterior, BLOCK_DRAWS)
    assert (
        chi_square_p_value(
            np.array([drawn[cell].sum() for cell in cells]),
            np.array([posterior[cell].sum() for cell in cells]) * BLOCK_DRAWS,
        )
        > SIGNIFICANCE
    )


@pytest.mark.oracle
def test_the_generic_sweep_recovers_the_exact_marginals_on_a_tree() -> None:
    params = load_fixture(FOUR_TAXA)
    alignment = dict(
        simulate_alignment(
            params.tau, params.k, params.pi, np.random.default_rng(1), 5
        ).alignment
    )
    site = {name: int(states[0]) for name, states in alignment.items()}
    transitions = {
        node.name: jc_transition_probabilities(node.branch_length, params.k)
        for node in preorder(params.tau)
        if node.branch_length is not None
    }
    graph = from_tree(params.tau, params.k, params.pi, site, transitions)
    exact = sum_product(graph).variable

    chain = sample_factor_graph(
        graph, np.random.default_rng(2), 6000, burn_in=100, thin=3
    )

    for column, name in enumerate(chain.variables):
        if name in site:
            assert (chain.states[:, column] == site[name]).all()
            continue
        counts = np.bincount(chain.states[:, column], minlength=params.k).astype(float)
        assert (
            chi_square_p_value(counts, len(chain.states) * exact[name]) > SIGNIFICANCE
        )


@pytest.mark.simulated_truth
def test_the_generic_sweep_samples_the_coupled_model_s_label_posterior() -> None:
    params = fixture("spatio_sequential", "ci").params
    data = simulate_spatio_sequential(params, np.random.default_rng(1))
    graph = coupled_factor_graph(params, data.observations)
    exact = enumerate_spatio_sequential(params, data.observations)

    chain = sample_factor_graph(
        graph, np.random.default_rng(3), 6000, burn_in=200, thin=3
    )

    for node in range(params.graph.n_nodes):
        column = chain.variables.index(f"l{node}")
        counts = np.bincount(chain.states[:, column], minlength=2).astype(float)
        p_value = chi_square_p_value(
            counts, len(chain.states) * exact.label_posterior[node]
        )
        assert p_value > SIGNIFICANCE, (node, p_value)


@pytest.mark.oracle
def test_annealing_reaches_the_closed_form_ground_state_as_the_potts_annealer_does() -> (
    None
):
    # The triangular antiferromagnet's ground state has exactly N agreeing
    # edges, so the minimum energy is known at every size. Six seeds, 200
    # sweeps each: the generic annealer and anneal_potts both reach it on 6.
    graph = fixture("frustrated_lattice", "ci").params.lattice()
    target = float(minimum_frustrated_edges(graph))
    factor_graph = from_potts(graph, np.zeros(2))
    schedule = Exponential(2.0, 0.05, 200)
    generic = specialised = 0
    for seed in range(6):
        annealed = anneal_factor_graph(
            factor_graph, schedule, np.random.default_rng(seed)
        )
        energy = float(energies(graph, np.zeros(2), annealed.state[None])[0])
        generic += int(abs(energy - target) < 1e-9)
        assert annealed.trajectory.shape == (201,)
        names = [variable.name for variable in factor_graph.variables]
        recomputed = factor_graph.log_density(
            dict(zip(names, annealed.state.tolist(), strict=True))
        )
        assert abs(annealed.log_density - recomputed) < 1e-12
        specialised += int(
            abs(
                anneal_potts(
                    graph, np.zeros(2), schedule, np.random.default_rng(seed)
                ).energy
                - target
            )
            < 1e-9
        )

    assert generic >= 5, generic
    assert specialised >= 5, specialised


@pytest.mark.mathematical
def test_the_temperature_scales_every_table_so_a_hot_chain_is_nearly_uniform() -> None:
    _, factor_graph = _potts_pair()

    hot = sample_factor_graph(
        factor_graph, np.random.default_rng(9), 4000, thin=2, temperature=1e3
    )

    counts = np.bincount(hot.states[:, 0], minlength=2).astype(float)
    assert chi_square_p_value(counts, np.full(2, len(hot.states) / 2)) > SIGNIFICANCE


def _five_taxa(n_sites: int) -> tuple[dict[str, np.ndarray], int]:
    params = load_simulation_params(fixture_path("tree_search/ci.yaml"))
    dataset = simulate_alignment(
        params.tau, params.k, params.pi, np.random.default_rng(2), n_sites
    )
    return dict(dataset.alignment), params.k


@pytest.mark.oracle
def test_the_topology_move_at_temperature_one_samples_the_enumerated_flat_prior_weight() -> (
    None
):
    # Twenty sites, so the weight over the 15 topologies is spread rather
    # than a point mass; every topology is fitted once and cached, so the
    # chain's cost is the fits and not its length.
    alignment, k = _five_taxa(20)
    topologies = list(enumerate_topologies(sorted(alignment)))
    keys = [leaf_bipartitions(topology) for topology in topologies]
    cache = {
        key: score_topology(topology, alignment, k)
        for key, topology in zip(keys, topologies, strict=True)
    }
    scores = np.array([cache[key] for key in keys])
    weights = np.exp(scores - scores.max())
    weights /= weights.sum()

    run = anneal_topology(
        alignment,
        k,
        Constant(1.0, 6000),
        np.random.default_rng(7),
        topologies[0],
        scores=cache,
    )
    # The run records values, not topologies; the visited topologies are
    # re-walked with the same generator and cache, which is the same chain.
    current, value = topologies[0], cache[keys[0]]
    visits = np.zeros(len(keys))
    rng = np.random.default_rng(7)
    for step in range(6000):
        options = list(nni_neighbours(current))
        proposal = options[int(rng.integers(len(options)))]
        proposed = cache[leaf_bipartitions(proposal)]
        if proposed - value >= 0.0 or rng.random() < np.exp(proposed - value):
            current, value = proposal, proposed
        if step % 5 == 0:
            visits[keys.index(leaf_bipartitions(current))] += 1

    assert len(run.scores) == 15
    assert run.trajectory.shape == (6001,)
    keep = weights * visits.sum() >= 5.0  # pool the rare topologies for the test
    pooled_counts = np.append(visits[keep], visits[~keep].sum())
    pooled_expected = np.append(weights[keep], weights[~keep].sum()) * visits.sum()
    assert chi_square_p_value(pooled_counts, pooled_expected) > SIGNIFICANCE


@pytest.mark.oracle
def test_the_annealed_topology_move_reaches_the_enumerated_best() -> None:
    alignment, k = _five_taxa(300)
    topologies = list(enumerate_topologies(sorted(alignment)))
    cache = {leaf_bipartitions(t): score_topology(t, alignment, k) for t in topologies}
    best = max(cache, key=lambda key: cache[key])
    hits = 0
    for seed in range(6):
        start = random_topology(sorted(alignment), np.random.default_rng(100 + seed))
        run = anneal_topology(
            alignment,
            k,
            Exponential(2.0, 0.02, 60),
            np.random.default_rng(seed),
            start,
            scores=cache,
        )
        hits += int(leaf_bipartitions(run.topology) == best)
        assert run.log_likelihood == cache[leaf_bipartitions(run.topology)]

    assert hits >= 5, hits


@pytest.mark.edge_case
def test_refusals() -> None:
    _, factor_graph = _potts_pair()
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="temperature must be positive"):
        sample_factor_graph(factor_graph, rng, 10, temperature=0.0)
    with pytest.raises(ValueError, match="n_sweeps"):
        sample_factor_graph(factor_graph, rng, 0)
    with pytest.raises(ValueError, match="inside its cardinality"):
        sample_factor_graph(factor_graph, rng, 2, start=np.array([0, 0, 0, 2]))
    with pytest.raises(ValueError, match="not consecutive"):
        chain_block_sweep(
            factor_graph, np.zeros(4, dtype=np.int64), rng, ["s0", "s3", "s1"]
        )
    triangle = FactorGraph(
        [Variable("a", 2), Variable("b", 2), Variable("c", 2)],
        [Factor("abc", ("a", "b", "c"), np.zeros((2, 2, 2)))],
    )
    with pytest.raises(ValueError, match="not consecutive"):
        chain_block_sweep(triangle, np.zeros(3, dtype=np.int64), rng, ["a", "b", "c"])
