"""Message passing on a factor graph, pinned against the three evaluators it unifies.

The claim `sim/factor_graph.py` makes is that pruning, the forward recursion
and belief propagation are one algorithm on three shapes of one object. A
claim of that form is tested by running the one algorithm on each shape and
holding it to the implementation that predates it, which shares no code with
it: `likelihood.pruning` on a tree, `hmm_paths.enumerate_hidden_paths` on a
chain, `belief_propagation` and `potts.enumerate_potts` on a graph. Where the
graph is a tree the reference is exact and so is the assertion; where it is
loopy both are the Bethe approximation and the assertion is that they are the
*same* approximation.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable

import numpy as np
import pytest
import torch
from snakes_and_ladders.backend import Backend
from snakes_and_ladders.enumeration import configurations
from snakes_and_ladders.likelihood import message_passing_reference as reference
from snakes_and_ladders.likelihood.belief_propagation import belief_propagation
from snakes_and_ladders.likelihood.forward_backward import forward_backward
from snakes_and_ladders.likelihood.hmm_paths import (
    emission_log_density,
    enumerate_hidden_paths,
)
from snakes_and_ladders.likelihood.message_passing import (
    ConvergenceError,
    Marginals,
    MessageScheduleName,
    max_product,
    sum_product,
)
from snakes_and_ladders.likelihood.potts import enumerate_potts, log_weights
from snakes_and_ladders.likelihood.pruning import log_likelihood
from snakes_and_ladders.likelihood.schedule import Guarantee
from snakes_and_ladders.likelihood.spatio_sequential import (
    class_log_density,
    class_posteriors,
)
from snakes_and_ladders.numerics import logsumexp
from snakes_and_ladders.opt.hmm import forward_log_likelihood
from snakes_and_ladders.sim.factor_graph import (
    Factor,
    FactorGraph,
    Variable,
    from_coupled,
    from_hmm,
    from_potts,
    from_tree,
)
from snakes_and_ladders.sim.fixtures import fixture
from snakes_and_ladders.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from snakes_and_ladders.sim.jc import jc_transition_probabilities
from snakes_and_ladders.sim.simulator import simulate_tree
from snakes_and_ladders.sim.spatio_sequential import simulate_spatio_sequential
from snakes_and_ladders.sim.tree import Node, preorder

from tests._fixtures import SMALL_SITES, load_fixture
from tests._rows import every_row, every_value
from tests.regression.likelihood.conftest import (
    CHAIN_CASES,
    FIELD,
    TREE,
    random_hmm,
)

RTOL = 1e-11
ATOL = 1e-12


# The tree `test_belief_propagation.py` uses: six nodes, five edges, couplings
# of mixed sign so a symmetric mistake cannot pass.

LOOPY = lattice_graph((3, 3), coupling=0.4, boundary=BoundaryCondition.OPEN)


# --- the Potts shape -----------------------------------------------------------


@pytest.mark.critical
@pytest.mark.oracle
def test_the_potts_log_density_is_log_weights_on_every_configuration() -> None:
    graph = from_potts(TREE, FIELD)
    configurations = np.array(
        list(itertools.product(range(len(FIELD)), repeat=TREE.n_nodes))
    )

    reference = log_weights(TREE, FIELD, configurations)
    realized = np.array(
        [
            graph.log_density({f"s{i}": int(s) for i, s in enumerate(row)})
            for row in configurations
        ]
    )

    np.testing.assert_allclose(realized, reference, rtol=0.0, atol=1e-13)


@pytest.mark.critical
@pytest.mark.oracle
def test_sum_product_on_the_potts_tree_is_the_enumeration() -> None:
    exact = enumerate_potts(TREE, FIELD)

    result = sum_product(from_potts(TREE, FIELD))

    assert result.exact
    assert result.iterations == 2
    assert math.isclose(result.log_partition, exact.log_partition, rel_tol=1e-14)
    single = np.stack([result.variable[f"s{i}"] for i in range(TREE.n_nodes)])
    np.testing.assert_allclose(single, exact.single_site, rtol=RTOL, atol=ATOL)
    pairwise = np.stack([result.factor[f"J{e}"] for e in range(len(TREE.edges))])
    np.testing.assert_allclose(pairwise, exact.pairwise, rtol=RTOL, atol=ATOL)


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.parametrize(
    "schedule",
    [
        MessageScheduleName.FLOODING,
        MessageScheduleName.SEQUENTIAL,
        MessageScheduleName.RESIDUAL,
    ],
    ids=str,
)
def test_an_iterative_schedule_on_the_loopy_lattice_is_belief_propagation(
    schedule: MessageScheduleName,
) -> None:
    # Both are the Bethe approximation, so agreement between codes is asserted;
    # the schedules share stationary points (#825) and meet the reference flooding.
    reference = belief_propagation(LOOPY, FIELD, damping=0.5, tolerance=1e-12)

    result = sum_product(from_potts(LOOPY, FIELD), schedule=schedule, tolerance=1e-12)

    assert not result.exact
    assert math.isclose(
        result.log_partition, reference.bethe_log_partition, rel_tol=1e-9
    )
    single = np.stack([result.variable[f"s{i}"] for i in range(LOOPY.n_nodes)])
    np.testing.assert_allclose(single, reference.single_site, rtol=1e-8, atol=1e-10)


@pytest.mark.smoke
def test_the_tree_schedule_refuses_a_loopy_graph() -> None:
    graph = from_potts(LOOPY, FIELD)
    assert not graph.is_tree()

    with pytest.raises(ValueError, match="tree schedule"):
        sum_product(graph, schedule=MessageScheduleName.TREE)


@pytest.mark.smoke
def test_flooding_refuses_when_it_has_not_converged() -> None:
    with pytest.raises(ConvergenceError):
        sum_product(
            from_potts(LOOPY, FIELD),
            schedule=MessageScheduleName.FLOODING,
            max_iterations=1,
        )


@pytest.mark.smoke
def test_damping_outside_the_unit_interval_is_refused() -> None:
    def check(damping: float) -> None:
        with pytest.raises(ValueError, match="damping"):
            sum_product(
                from_potts(LOOPY, FIELD),
                schedule=MessageScheduleName.FLOODING,
                damping=damping,
            )

    every_value([-0.1, 1.0], check)


# --- the chain shape ------------------------------------------------------------


@pytest.mark.critical
@pytest.mark.oracle
def test_every_chain_evaluator_is_the_path_enumeration() -> None:
    # One table, one referee (issue #982 merged five tests over these four
    # chains): `hmm_paths.enumerate_hidden_paths` sums every path, and each
    # evaluator below shares no code with it.
    def check(n_states: int, n_symbols: int, length: int, seed: int) -> None:
        params = random_hmm(n_states, n_symbols, length, seed)
        observations = np.random.default_rng(seed).integers(0, n_symbols, size=length)
        enumerated = enumerate_hidden_paths(params, observations)
        log_density = emission_log_density(params, observations)
        log_initial, log_transition = np.log(params.initial), np.log(params.transition)

        # `opt.hmm`'s forward recursion: the check that the enumeration sums the
        # model it claims to.
        forward = forward_log_likelihood(
            torch.from_numpy(observations[None, :]),
            torch.log(torch.from_numpy(params.initial)),
            torch.log(torch.from_numpy(params.transition)),
            torch.log(torch.from_numpy(params.emission)),
        )
        assert enumerated.log_likelihood == pytest.approx(float(forward), rel=1e-12)

        # Forward--backward: evidence, posterior, and the pairwise marginals
        # summing to the posterior on either side.
        run = forward_backward(log_density, log_initial, log_transition)
        assert abs(run.log_evidence - enumerated.log_likelihood) < 1e-12 * abs(
            enumerated.log_likelihood
        )
        np.testing.assert_allclose(
            run.posterior, enumerated.posterior, rtol=1e-11, atol=1e-13
        )
        np.testing.assert_allclose(run.pairwise.sum(axis=(1, 2)), 1.0, rtol=1e-12)
        np.testing.assert_allclose(
            run.pairwise.sum(axis=2), run.posterior[:-1], rtol=1e-11
        )
        np.testing.assert_allclose(
            run.pairwise.sum(axis=1), run.posterior[1:], rtol=1e-11
        )

        # Sum- and max-product on the chain's factor graph, on the default route,
        # which is the Rust kernel: the marginals and `log Z`, and Viterbi.
        graph = from_hmm(log_initial, log_transition, log_density)
        result = sum_product(graph)
        assert result.exact
        assert math.isclose(
            result.log_partition, enumerated.log_likelihood, rel_tol=1e-13
        )
        posterior = np.stack([result.variable[f"z{t}"] for t in range(length)])
        np.testing.assert_allclose(
            posterior, enumerated.posterior, rtol=RTOL, atol=ATOL
        )
        assignment, marginals = max_product(graph)
        path = np.array([assignment[f"z{t}"] for t in range(length)])
        np.testing.assert_array_equal(path, enumerated.viterbi)
        assert math.isclose(
            marginals.log_partition, enumerated.viterbi_log_probability, rel_tol=1e-13
        )

        # The NumPy route against the dictionary reference, bitwise (issue #341).
        _assert_same_marginals(
            sum_product(graph, backend=Backend.PYTHON), reference.sum_product(graph)
        )
        assignment, marginals = max_product(graph, backend=Backend.PYTHON)
        expected_assignment, expected = reference.max_product(graph)
        assert assignment == expected_assignment
        _assert_same_marginals(marginals, expected)

    every_row(CHAIN_CASES, check)


# --- the tree shape -------------------------------------------------------------


def _transitions(tau: Node, k: int) -> dict[str, np.ndarray]:
    return {
        node.name: jc_transition_probabilities(node.branch_length, k)
        for node in preorder(tau)
        if node.branch_length is not None
    }


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.parametrize(
    ("schedule", "guarantee"),
    [("tree", Guarantee.EXACT), ("upward", Guarantee.PARTIAL)],
)
def test_sum_product_per_site_sums_to_pruning(
    schedule: str, guarantee: Guarantee
) -> None:
    # `likelihood.pruning` is the leaf-to-root recursion written on the tree;
    # the upward schedule is that recursion as half a message-passing plan,
    # exact at the root only (issue #592, merged from `test_schedule.py`).
    params = load_fixture(SMALL_SITES)
    dataset = simulate_tree(params, np.random.default_rng(params.seed), n_sites=7)
    alignment = dict(dataset.alignment)
    transitions = _transitions(params.tau, params.k)

    total = 0.0
    for s in range(7):
        site = {name: int(states[s]) for name, states in alignment.items()}
        result = sum_product(
            from_tree(params.tau, params.k, params.pi, site, transitions),
            schedule=schedule,
        )
        assert result.guarantee is guarantee
        total += result.log_partition

    reference = log_likelihood(params.tau, params.k, params.pi, alignment)
    assert math.isclose(total, reference, rel_tol=1e-13)


@pytest.mark.analytic
def test_the_leaf_marginals_on_the_tree_are_the_observed_indicators() -> None:
    # An observed leaf has a hard indicator factor, so its marginal is a delta
    # at the observation whatever the branch lengths say.
    params = load_fixture(SMALL_SITES)
    dataset = simulate_tree(params, np.random.default_rng(params.seed), n_sites=3)
    site = {name: int(states[0]) for name, states in dict(dataset.alignment).items()}

    result = sum_product(
        from_tree(
            params.tau, params.k, params.pi, site, _transitions(params.tau, params.k)
        )
    )

    for name, state in site.items():
        expected = np.zeros(params.k)
        expected[state] = 1.0
        np.testing.assert_allclose(result.variable[name], expected, atol=ATOL)


# --- the coupled shape -----------------------------------------------------------


@pytest.mark.critical
@pytest.mark.oracle
def test_the_coupled_log_density_is_the_joint_written_out() -> None:
    # eq:joint (textbook, sec:factor-graph): beta * sum_edges J delta(l_i, l_j)
    # + sum_m [log pi_m(k_{0,m}) + sum_s log A_m(k_{s-1,m}, k_{s,m})]
    # + sum_n sum_s log P(x_{sn} | k_{s,l_n}, theta_{l_n}).
    rng = np.random.default_rng(7)
    spatial = PottsGraph(n_nodes=3, edges=((0, 1), (1, 2)), coupling=(0.6, -0.3))
    beta, length, n_classes, n_states = 0.8, 2, 2, 2
    log_initial = [np.log(rng.dirichlet(np.ones(n_states))) for _ in range(n_classes)]
    log_transition = [
        np.log(rng.dirichlet(np.ones(n_states), size=n_states))
        for _ in range(n_classes)
    ]
    gated = rng.normal(size=(spatial.n_nodes, length, n_classes, n_states))
    graph = from_coupled(spatial, beta, log_initial, log_transition, gated)
    assert not graph.is_tree()

    labels_space = itertools.product(range(n_classes), repeat=spatial.n_nodes)
    chains_space = list(itertools.product(range(n_states), repeat=length * n_classes))
    for labels in labels_space:
        for chains in chains_space:
            k = np.array(chains).reshape(n_classes, length)
            expected = beta * sum(
                J * (labels[i] == labels[j]) for (i, j), J in spatial.weighted_edges()
            )
            for m in range(n_classes):
                expected += log_initial[m][k[m, 0]]
                expected += sum(
                    log_transition[m][k[m, s - 1], k[m, s]] for s in range(1, length)
                )
            for n in range(spatial.n_nodes):
                for s in range(length):
                    expected += gated[n, s, labels[n], k[labels[n], s]]
            assignment = {f"l{n}": labels[n] for n in range(spatial.n_nodes)}
            assignment.update(
                {
                    f"k{s},{m}": int(k[m, s])
                    for m in range(n_classes)
                    for s in range(length)
                }
            )
            assert math.isclose(graph.log_density(assignment), expected, rel_tol=1e-13)


#: Generator seeds the coupled instance is simulated at. Twelve draws, two
#: classes each: 24 chains, enough that the tie below is met rather than
#: avoided.
COUPLED_SEEDS = 12

#: Separation in log mass a path law's mode is called unique above: the
#: float64 noise floor, below which two paths are tied and the decode is a
#: tie rule's.
MODE_MARGIN = 1e-12

#: Separation, in nats, the unique modes are held to. The smallest realized
#: over the 22 of them is 0.385, so a change that flattened the law towards
#: its runner-up would fail here before it reached the tie.
UNIQUE_MARGIN = 0.3


def _path_law(
    posterior: np.ndarray, pairwise: np.ndarray, paths: np.ndarray
) -> np.ndarray:
    """``log Q(k_1..k_S)`` for every path, from one class's E-step posteriors.

    ``Q(k) = Q(k_1) prod_s Q(k_{s-1}, k_s) / Q(k_{s-1})``: the E step's own numbers.
    """
    log_q = np.log(posterior[0])[paths[:, 0]]
    for position in range(1, paths.shape[1]):
        log_q = (
            log_q
            + np.log(pairwise[position - 1])[paths[:, position - 1], paths[:, position]]
            - np.log(posterior[position - 1])[paths[:, position - 1]]
        )
    return np.asarray(log_q, dtype=float)


@pytest.mark.critical
@pytest.mark.oracle
def test_max_product_decodes_the_mode_of_the_coupled_e_step_s_path_law() -> None:
    # The rung below (#734): the coupled E step's pair posteriors determine the
    # law over paths; max-product decodes its mode with mass
    # `log_partition - E == log Q(decoded path)`. Over 12 seeds, both classes,
    # 24 chains of 2**6 paths: normalized to 9.2e-15, mass identity 8.9e-15
    # (1e-12 declared), argmax on 22 of 24. The other two tie (1.1e-16,
    # 8.9e-16 apart): only the mass identity survives, and per-site decoding
    # differs there at one position each, which `hmm_paths` keeps apart.
    params = fixture("spatio_sequential", "ci").params
    paths = configurations(params.n_states, params.n_positions)
    modes = ties = disagreements = 0
    for seed in range(COUPLED_SEEDS):
        data = simulate_spatio_sequential(params, np.random.default_rng(seed))
        step = class_posteriors(params, data.observations, data.labels)
        density = class_log_density(params, data.observations, data.labels)
        for m in range(params.n_classes):
            graph = from_hmm(
                np.log(params.initial[m]), np.log(params.transition), density[m]
            )
            assignment, marginals = max_product(graph)
            decoded = np.array([assignment[f"z{s}"] for s in range(params.n_positions)])

            log_q = _path_law(step.posterior[m], step.pairwise[m], paths)
            assert abs(float(logsumexp(log_q, axis=0))) < 1e-13
            at_decoded = float(log_q[np.flatnonzero((paths == decoded).all(axis=1))[0]])
            assert at_decoded == pytest.approx(
                marginals.log_partition - step.log_evidence[m], abs=1e-12
            )

            ranked = np.argsort(-log_q)
            margin = float(log_q[ranked[0]] - log_q[ranked[1]])
            if margin > MODE_MARGIN:
                assert np.array_equal(paths[ranked[0]], decoded)
                assert margin > UNIQUE_MARGIN
                modes += 1
            else:
                ties += 1
                assert abs(float(log_q[ranked[0]]) - at_decoded) < 1e-14
            disagreements += int(
                not np.array_equal(step.posterior[m].argmax(axis=1), decoded)
            )

    assert (modes, ties) == (22, 2)
    assert disagreements == 2


# --- the Forney form -------------------------------------------------------------


@pytest.mark.analytic
def test_the_forney_form_has_every_variable_on_exactly_two_factors_or_fewer() -> None:
    forney = from_potts(TREE, FIELD).forney()

    assert all(forney.degree(v.name) <= 2 for v in forney.variables)
    assert forney.is_tree()


@pytest.mark.critical
@pytest.mark.oracle
def test_message_passing_on_the_forney_form_gives_the_same_marginals() -> None:
    graph = from_potts(TREE, FIELD)
    reference = sum_product(graph)

    result = sum_product(graph.forney())

    assert math.isclose(result.log_partition, reference.log_partition, rel_tol=1e-13)
    for variable in graph.variables:
        copies = [
            name for name in result.variable if name.split("=")[0] == variable.name
        ]
        assert copies, variable.name
        for name in copies:
            np.testing.assert_allclose(
                result.variable[name],
                reference.variable[variable.name],
                rtol=RTOL,
                atol=ATOL,
            )


@pytest.mark.smoke
def test_a_graph_with_no_variable_above_degree_two_is_its_own_forney_form() -> None:
    # A chain of pairwise factors alone. With emissions each interior state
    # would sit on three factors and need an equality node.
    graph = FactorGraph(
        [Variable("a", 2), Variable("b", 2), Variable("c", 2)],
        [
            Factor("ab", ("a", "b"), np.zeros((2, 2))),
            Factor("bc", ("b", "c"), np.zeros((2, 2))),
        ],
    )

    forney = graph.forney()

    assert [v.name for v in forney.variables] == ["a", "b", "c"]
    assert [f.name for f in forney.factors] == ["ab", "bc"]


# --- construction refusals ---------------------------------------------------------


@pytest.mark.smoke
def test_a_malformed_graph_is_refused_at_construction() -> None:
    refusals: list[tuple[str, Callable[[], object]]] = [
        (
            "unknown variable",
            lambda: FactorGraph(
                [Variable("a", 2)], [Factor("f", ("a", "b"), np.zeros((2, 2)))]
            ),
        ),
        (
            "table shape",
            lambda: FactorGraph([Variable("a", 3)], [Factor("f", ("a",), np.zeros(2))]),
        ),
        (
            "in no factor",
            lambda: FactorGraph(
                [Variable("a", 2), Variable("b", 2)],
                [Factor("f", ("a",), np.zeros(2))],
            ),
        ),
        ("domain", lambda: Variable("a", 0)),
    ]
    for message, construct in refusals:
        with pytest.raises(ValueError, match=message):
            construct()


# --- the edge-array layout against the dictionary oracle (issue #341) ------------
#
# Same arithmetic per message in the same order: bitwise marginals on every
# schedule; ``log_partition`` reorders its Bethe terms, 1e-12 relative. The
# NumPy route is named (#754): the Rust default differs in `exp`/`log` last
# places and is pinned in `test_message_passing_rust.py`.


def _assert_same_marginals(realized: Marginals, expected: Marginals) -> None:
    assert realized.iterations == expected.iterations
    assert realized.exact == expected.exact
    assert set(realized.variable) == set(expected.variable)
    assert set(realized.factor) == set(expected.factor)
    for name, values in expected.variable.items():
        np.testing.assert_array_equal(realized.variable[name], values, err_msg=name)
    for name, values in expected.factor.items():
        np.testing.assert_array_equal(realized.factor[name], values, err_msg=name)
    if math.isnan(expected.log_partition):
        assert math.isnan(realized.log_partition)
    else:
        assert math.isclose(
            realized.log_partition, expected.log_partition, rel_tol=1e-12
        )


def _coupled_mixed_cardinality() -> FactorGraph:
    # Three classes over two-state chains: the labels have cardinality 3 and
    # the chain states 2, so the edge rows are padded and the padding is what
    # this instance exercises. Loopy, so flooding.
    rng = np.random.default_rng(11)
    spatial = PottsGraph(n_nodes=3, edges=((0, 1), (1, 2)), coupling=(0.6, -0.3))
    n_classes, n_states, length = 3, 2, 2
    return from_coupled(
        spatial,
        0.8,
        [np.log(rng.dirichlet(np.ones(n_states))) for _ in range(n_classes)],
        [
            np.log(rng.dirichlet(np.ones(n_states), size=n_states))
            for _ in range(n_classes)
        ],
        rng.normal(size=(spatial.n_nodes, length, n_classes, n_states)),
    )


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.parametrize(
    "graph",
    [
        pytest.param(from_potts(TREE, FIELD), id="potts-tree"),
        pytest.param(from_potts(TREE, FIELD).forney(), id="forney-form"),
    ],
)
def test_the_tree_schedule_reproduces_the_dictionary_oracle_bitwise(
    graph: FactorGraph,
) -> None:
    _assert_same_marginals(
        sum_product(graph, backend=Backend.PYTHON), reference.sum_product(graph)
    )


@pytest.mark.critical
@pytest.mark.oracle
def test_the_tree_site_reproduces_the_dictionary_oracle_bitwise() -> None:
    # Hard zeros: the leaf indicators put ``-inf`` in the tables.
    params = load_fixture(SMALL_SITES)
    dataset = simulate_tree(params, np.random.default_rng(params.seed), n_sites=1)
    site = {name: int(states[0]) for name, states in dict(dataset.alignment).items()}
    graph = from_tree(
        params.tau, params.k, params.pi, site, _transitions(params.tau, params.k)
    )

    _assert_same_marginals(
        sum_product(graph, backend=Backend.PYTHON), reference.sum_product(graph)
    )


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.parametrize(
    "graph",
    [
        pytest.param(from_potts(LOOPY, FIELD), id="3x3-lattice"),
        pytest.param(_coupled_mixed_cardinality(), id="coupled-mixed-cardinality"),
    ],
)
def test_flooding_reproduces_the_dictionary_oracle_bitwise(graph: FactorGraph) -> None:
    assert not graph.is_tree()

    _assert_same_marginals(
        sum_product(graph, schedule=MessageScheduleName.FLOODING),
        reference.sum_product(graph, schedule=MessageScheduleName.FLOODING),
    )
    assignment, marginals = max_product(graph, schedule=MessageScheduleName.FLOODING)
    expected_assignment, expected = reference.max_product(
        graph, schedule=MessageScheduleName.FLOODING
    )
    assert assignment == expected_assignment
    _assert_same_marginals(marginals, expected)


@pytest.mark.critical
@pytest.mark.oracle
def test_the_tree_schedule_on_a_deep_chain_is_the_forward_recursion() -> None:
    # 2,000 positions: past the interpreter's recursion limit for the
    # reference's depth-first order, and the levelled schedule is
    # breadth-first. The forward recursion is the oracle, so this is a claim
    # about the evidence and not only about not raising.
    from snakes_and_ladders.opt.hmm import forward_log_likelihood_from_density

    rng = np.random.default_rng(5)
    log_initial = np.log(rng.dirichlet(np.ones(3)))
    log_transition = np.log(rng.dirichlet(np.ones(3), size=3))
    log_density = rng.normal(size=(2_000, 3))

    result = sum_product(from_hmm(log_initial, log_transition, log_density))

    expected = float(
        forward_log_likelihood_from_density(
            torch.as_tensor(log_density)[None],
            torch.as_tensor(log_initial),
            torch.as_tensor(log_transition),
        )
    )
    assert result.exact
    assert math.isclose(result.log_partition, expected, rel_tol=1e-12)
