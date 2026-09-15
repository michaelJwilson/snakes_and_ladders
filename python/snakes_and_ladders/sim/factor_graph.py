"""A factor graph, the one structure the three problem classes share.

A tree under a substitution model, a Potts model on a lattice and a hidden
Markov chain are one object: variables with finite domains, and factors, each
a non-negative function of a few of them, whose product is the unnormalized
distribution. Pruning, the forward recursion and belief propagation are then
one algorithm on three shapes of this object (`docs/tex/textbook.tex`,
``sec:factor-graph``), and the model issue #290 asks for -- spatial labels, a
hidden chain per class, and a gated emission joining one label to one chain
state -- is a fourth shape rather than a fourth code path.

**Forney-style.** In a normal (Forney) factor graph the variables are the
edges and every node is a factor, so a variable shared by more than two
factors passes through an equality node. :meth:`FactorGraph.forney` builds
that form from the bipartite one held here: it is what the textbook's figures
draw, and message passing on it is message passing here with identity factors
inserted, which a test asserts. The bipartite form is kept as the working
representation because an adapter writes it directly and the equality nodes
carry no information of their own.

Every factor holds its function as a **log table** over its variables' domains
in the order it names them, so a factor of degree ``d`` is a ``d``-dimensional
array. Observations are folded into the tables by the adapters -- a leaf's
state becomes an indicator factor, an emission becomes a unary factor over the
hidden state -- so the graph is over latent variables only, and the same
message passing serves a density and a probability alike.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from snakes_and_ladders.sim.convolutional import IMPOSSIBLE_EDGE, Trellis
from snakes_and_ladders.sim.graph import PottsGraph
from snakes_and_ladders.sim.ldpc import ParityCheck
from snakes_and_ladders.sim.tree import Node, edges, preorder


@dataclass(frozen=True)
class Variable:
    """A discrete latent variable.

    Parameters
    ----------
    name : str
        Unique within the graph.
    cardinality : int
        Size of the domain, ``>= 1``.
    """

    name: str
    cardinality: int

    def __post_init__(self) -> None:
        if self.cardinality < 1:
            msg = f"variable {self.name!r} needs a domain of at least 1, got {self.cardinality}"
            raise ValueError(msg)


@dataclass(frozen=True)
class Factor:
    """A non-negative function of a few variables, held as a log table.

    Parameters
    ----------
    name : str
        Unique within the graph.
    variables : tuple[str, ...]
        The variables it depends on, in the order of ``log_table``'s axes.
    log_table : np.ndarray
        ``log psi``, shape ``tuple(cardinality of each variable)``. ``-inf``
        is a hard zero.
    """

    name: str
    variables: tuple[str, ...]
    log_table: np.ndarray

    def __post_init__(self) -> None:
        if len(set(self.variables)) != len(self.variables):
            msg = f"factor {self.name!r} names a variable twice: {self.variables}"
            raise ValueError(msg)
        if self.log_table.ndim != len(self.variables):
            msg = (
                f"factor {self.name!r} has a {self.log_table.ndim}-D table for "
                f"{len(self.variables)} variables"
            )
            raise ValueError(msg)


@dataclass(frozen=True)
class FactorGraph:
    """Variables, factors, and the bipartite structure between them.

    Parameters
    ----------
    variables : Sequence[Variable]
    factors : Sequence[Factor]

    Raises
    ------
    ValueError
        If a name repeats, a factor names a variable the graph lacks, a table's
        shape disagrees with its variables' domains, or a variable sits in no
        factor -- a variable nothing constrains is uniform and almost always a
        typo in an adapter.
    """

    variables: tuple[Variable, ...]
    factors: tuple[Factor, ...]
    _index: dict[str, Variable] = field(init=False, repr=False, compare=False)

    def __init__(
        self, variables: Sequence[Variable], factors: Sequence[Factor]
    ) -> None:
        object.__setattr__(self, "variables", tuple(variables))
        object.__setattr__(self, "factors", tuple(factors))
        index = {variable.name: variable for variable in self.variables}
        if len(index) != len(self.variables):
            msg = "variable names must be unique"
            raise ValueError(msg)
        if len({factor.name for factor in self.factors}) != len(self.factors):
            msg = "factor names must be unique"
            raise ValueError(msg)
        touched: set[str] = set()
        for factor in self.factors:
            for name in factor.variables:
                if name not in index:
                    msg = f"factor {factor.name!r} names unknown variable {name!r}"
                    raise ValueError(msg)
            expected = tuple(index[name].cardinality for name in factor.variables)
            if factor.log_table.shape != expected:
                msg = (
                    f"factor {factor.name!r} has table shape {factor.log_table.shape}, "
                    f"its variables have domains {expected}"
                )
                raise ValueError(msg)
            touched.update(factor.variables)
        lonely = sorted(set(index) - touched)
        if lonely:
            msg = f"variables in no factor: {lonely}"
            raise ValueError(msg)
        object.__setattr__(self, "_index", index)

    def variable(self, name: str) -> Variable:
        """The variable called ``name``."""
        return self._index[name]

    def degree(self, name: str) -> int:
        """How many factors ``name`` appears in."""
        return sum(name in factor.variables for factor in self.factors)

    def neighbours(self, name: str) -> Iterator[Factor]:
        """The factors ``name`` appears in, in factor order."""
        return (factor for factor in self.factors if name in factor.variables)

    def is_tree(self) -> bool:
        """Whether the bipartite graph is acyclic and connected.

        Message passing is exact exactly here; elsewhere it is the Bethe
        approximation, and the caller must say which it wants.
        """
        n_nodes = len(self.variables) + len(self.factors)
        n_edges = sum(len(factor.variables) for factor in self.factors)
        if n_edges != n_nodes - 1:
            return False
        parent: dict[str, str] = {}

        def find(node: str) -> str:
            while parent.get(node, node) != node:
                node = parent[node]
            return node

        for factor in self.factors:
            for name in factor.variables:
                a, b = find(f"f:{factor.name}"), find(f"v:{name}")
                if a == b:
                    return False
                parent[a] = b
        return True

    def log_density(self, assignment: Mapping[str, int]) -> float:
        """``sum_f log psi_f`` at one full assignment: the unnormalized log-density.

        The definition every adapter is pinned against, by enumeration, and
        the oracle of the compiled path a sampler takes over the edge layout
        (:meth:`snakes_and_ladders.search.gibbs._Indexed.log_density`, issue
        #563), which sums the same terms in the same order and so reproduces
        this bitwise.
        """
        total = 0.0
        for factor in self.factors:
            total += float(
                factor.log_table[tuple(assignment[name] for name in factor.variables)]
            )
        return total

    def forney(self) -> FactorGraph:
        """The normal (Forney) form: every variable of degree above two passes
        through an equality factor, so each variable is an edge between exactly
        two factors.

        A variable of degree ``d > 2`` becomes ``d`` copies joined by one
        ``d``-ary equality factor whose table is ``0`` on the diagonal and
        ``-inf`` elsewhere. The distribution is unchanged, which the message
        passing test asserts by computing the same marginals on both forms.
        """
        variables: list[Variable] = []
        factors: list[Factor] = []
        for variable in self.variables:
            degree = self.degree(variable.name)
            if degree <= 2:
                variables.append(variable)
                continue
            copies = [
                Variable(f"{variable.name}={i}", variable.cardinality)
                for i in range(degree)
            ]
            variables.extend(copies)
            table = np.full((variable.cardinality,) * degree, -np.inf)
            for value in range(variable.cardinality):
                table[(value,) * degree] = 0.0
            factors.append(
                Factor(f"={variable.name}", tuple(copy.name for copy in copies), table)
            )
        for factor in self.factors:
            renamed = []
            for name in factor.variables:
                if self.degree(name) <= 2:
                    renamed.append(name)
                    continue
                copy_index = sum(
                    1
                    for other in self.factors[: self.factors.index(factor)]
                    if name in other.variables
                )
                renamed.append(f"{name}={copy_index}")
            factors.append(Factor(factor.name, tuple(renamed), factor.log_table))
        return FactorGraph(variables, factors)


# --- adapters: the problem classes, and the coupled model ---------------


def from_potts(graph: PottsGraph, field_values: np.ndarray) -> FactorGraph:
    """The Potts model of :func:`snakes_and_ladders.likelihood.potts.log_weights`.

    One variable per node, a unary factor ``h`` per node, and a pairwise factor
    ``J_ij delta(s_i, s_j)`` per edge, so ``log_density`` is ``log_weights``.

    Parameters
    ----------
    graph : PottsGraph
    field_values : np.ndarray
        Shape ``(q,)``, or ``(n_nodes, q)`` for a per-node field.
    """
    values = np.asarray(field_values, dtype=float)
    if values.ndim == 1:
        values = np.tile(values, (graph.n_nodes, 1))
    q = int(values.shape[1])
    variables = [Variable(f"s{i}", q) for i in range(graph.n_nodes)]
    factors = [
        Factor(f"h{i}", (f"s{i}",), values[i].copy()) for i in range(graph.n_nodes)
    ]
    identity = np.eye(q)
    for position, ((first, second), coupling) in enumerate(graph.weighted_edges()):
        factors.append(
            Factor(f"J{position}", (f"s{first}", f"s{second}"), coupling * identity)
        )
    return FactorGraph(variables, factors)


def from_hmm(
    log_initial: np.ndarray, log_transition: np.ndarray, log_density: np.ndarray
) -> FactorGraph:
    """A hidden Markov chain with its observations folded in.

    Parameters
    ----------
    log_initial : np.ndarray
        Shape ``(m,)``.
    log_transition : np.ndarray
        Shape ``(m, m)``, row ``i`` the distribution leaving state ``i``.
    log_density : np.ndarray
        Shape ``(T, m)``: the emission score of the observation at each
        position under each state, as :func:`snakes_and_ladders.opt.hmm.forward_log_likelihood_from_density`
        takes it -- a log-probability for a discrete family, a log-density
        otherwise.
    """
    length, m = log_density.shape
    variables = [Variable(f"z{t}", m) for t in range(length)]
    factors = [
        Factor("pi", ("z0",), np.asarray(log_initial, dtype=float) + log_density[0])
    ]
    for t in range(1, length):
        factors.append(
            Factor(
                f"A{t}", (f"z{t - 1}", f"z{t}"), np.asarray(log_transition, dtype=float)
            )
        )
        factors.append(
            Factor(f"e{t}", (f"z{t}",), np.asarray(log_density[t], dtype=float))
        )
    return FactorGraph(variables, factors)


def from_tree(
    tau: Node,
    k: int,
    pi: np.ndarray,
    site: Mapping[str, int],
    transitions: Mapping[str, np.ndarray],
) -> FactorGraph:
    """One site of an alignment on a tree: pruning's own factor graph.

    A variable per node's state; the root prior as a unary factor; ``P(t)``
    on each branch as a pairwise factor from parent to child; each leaf's
    observed state as an indicator factor. Summing it out is
    :func:`snakes_and_ladders.likelihood.pruning.log_likelihood` at that site (``eq:pruning``,
    ``eq:root``), and a test asserts so.

    Parameters
    ----------
    tau : Node
        The rooted topology.
    k : int
        States.
    pi : np.ndarray
        Root distribution, shape ``(k,)``.
    site : Mapping[str, int]
        Leaf name to observed state at this site.
    transitions : Mapping[str, np.ndarray]
        Child name to its branch's ``P(t)``, shape ``(k, k)``.
    """
    variables = [Variable(node.name, k) for node in preorder(tau)]
    factors = [Factor("pi", (tau.name,), np.log(np.asarray(pi, dtype=float)))]
    with np.errstate(divide="ignore"):
        for parent, child in edges(tau):
            factors.append(
                Factor(
                    f"P:{child.name}",
                    (parent.name, child.name),
                    np.log(np.asarray(transitions[child.name], dtype=float)),
                )
            )
        for node in preorder(tau):
            if node.is_leaf:
                indicator = np.full(k, -np.inf)
                indicator[site[node.name]] = 0.0
                factors.append(Factor(f"x:{node.name}", (node.name,), indicator))
    return FactorGraph(variables, factors)


def from_coupled(
    graph: PottsGraph,
    beta: float,
    log_initial: Sequence[np.ndarray],
    log_transition: Sequence[np.ndarray],
    gated_log_density: np.ndarray,
) -> FactorGraph:
    """The coupled spatio-sequential model of issue #290.

    Spatial labels ``l_n`` in ``{0..M-1}`` under a Potts prior with couplings
    ``beta * J``; one hidden chain ``k_{s,m}`` per class with its own initial
    distribution and transition; and for every node, position and class a
    **gated emission** factor over ``(l_n, k_{s,m})`` that contributes
    ``log P(x_{sn} | k_{s,m}, theta_m)`` when ``l_n = m`` and nothing
    otherwise. Its ``log_density`` is the joint of ``eq:joint`` in the
    textbook, up to the Potts normalizer, and a test pins it.

    Parameters
    ----------
    graph : PottsGraph
        The spatial graph; its couplings are ``J``.
    beta : float
        Inverse temperature on the spatial prior.
    log_initial, log_transition : Sequence[np.ndarray]
        Per class: shape ``(K,)`` and ``(K, K)``.
    gated_log_density : np.ndarray
        Shape ``(n_nodes, S, M, K)``: ``log P(x_{sn} | k, theta_m)``.
    """
    n_nodes, length, n_classes, n_states = gated_log_density.shape
    variables = [Variable(f"l{n}", n_classes) for n in range(n_nodes)]
    factors: list[Factor] = []
    identity = np.eye(n_classes)
    for position, ((first, second), coupling) in enumerate(graph.weighted_edges()):
        factors.append(
            Factor(
                f"J{position}", (f"l{first}", f"l{second}"), beta * coupling * identity
            )
        )
    for m in range(n_classes):
        for s in range(length):
            variables.append(Variable(f"k{s},{m}", n_states))
        factors.append(
            Factor(f"pi{m}", (f"k0,{m}",), np.asarray(log_initial[m], dtype=float))
        )
        for s in range(1, length):
            factors.append(
                Factor(
                    f"A{s},{m}",
                    (f"k{s - 1},{m}", f"k{s},{m}"),
                    np.asarray(log_transition[m], dtype=float),
                )
            )
    for n in range(n_nodes):
        for s in range(length):
            for m in range(n_classes):
                table = np.zeros((n_classes, n_states))
                table[m, :] = gated_log_density[n, s, m]
                factors.append(Factor(f"x{n},{s},{m}", (f"l{n}", f"k{s},{m}"), table))
    return FactorGraph(variables, factors)


def from_parity_check(code: ParityCheck, llr: np.ndarray) -> FactorGraph:
    """A parity-check code with its channel output folded in (``sec:ldpc``).

    One binary variable ``x{i}`` per bit; one unary factor ``y{i}`` per bit
    with log table ``(0, -L_i)``, the channel's log-likelihood ratio in the
    convention of ``eq:ldpc-llr``; one parity factor ``c{j}`` per row of
    ``H`` over its bits, zero where they sum to zero mod 2 and ``-inf``
    elsewhere. ``log_density`` is then ``-c . L`` on a codeword and ``-inf``
    off it, which is ``eq:ldpc-code`` up to the constant a test states, and
    :func:`snakes_and_ladders.likelihood.message_passing.sum_product` on it is
    the oracle the vectorized decoder is held to.

    Parameters
    ----------
    code : ParityCheck
    llr : np.ndarray
        Shape ``(n_bits,)``.
    """
    ratios = np.asarray(llr, dtype=float)
    if ratios.shape != (code.n_bits,):
        msg = f"llr has shape {ratios.shape}, the code has {code.n_bits} bits"
        raise ValueError(msg)
    variables = [Variable(f"x{i}", 2) for i in range(code.n_bits)]
    factors = [
        Factor(f"y{i}", (f"x{i}",), np.array([0.0, -ratios[i]]))
        for i in range(code.n_bits)
    ]
    members = code.edge_variable[code.check_order]
    for j in range(code.n_checks):
        bits = members[code.check_offsets[j] : code.check_offsets[j + 1]]
        parity = np.indices((2,) * bits.size).sum(axis=0) % 2
        table = np.where(parity == 0, 0.0, -np.inf)
        factors.append(Factor(f"c{j}", tuple(f"x{i}" for i in bits), table))
    return FactorGraph(variables, factors)


def from_trellis(
    trellis: Trellis,
    systematic_llr: np.ndarray,
    parity_llr: np.ndarray,
    apriori_llr: np.ndarray | None = None,
    *,
    terminated: bool = True,
) -> FactorGraph:
    """A terminated convolutional code with its channel output folded in.

    The chain :func:`from_hmm` builds carries its observation on the *state*;
    a trellis carries it on the *edge*, since the bits transmitted at a step
    are a function of the transition taken. So the variables here are the
    ``T + 1`` register states, and one factor per step holds the whole of
    that step: the transition it allows and the channel's score for the two
    bits that transition transmits (``sec:turbo``). Summing it out is
    :func:`snakes_and_ladders.likelihood.convolutional.bcjr`, and the graph
    is a chain, so :func:`snakes_and_ladders.likelihood.message_passing.sum_product`
    on the tree schedule is exact on it and a test asserts the two agree.

    The input bit is recoverable from the state pair rather than carried as a
    variable of its own: it is the ``u`` for which ``next_state[s, u] = s'``,
    unique because the two edges leaving a state enter different states, so a
    variable per input bit would add a degree-2 equality node and nothing
    else.

    Parameters
    ----------
    trellis : Trellis
    systematic_llr, parity_llr : np.ndarray
        Shape ``(T,)`` each, in the convention of ``eq:ldpc-llr``.
    apriori_llr : np.ndarray | None
        Shape ``(T,)``: an a priori ratio on the input bit, as a turbo
        half-iteration supplies. ``None`` is the uninformative zero.
    terminated : bool
        Whether the final state is pinned to zero by an indicator factor.

    Raises
    ------
    ValueError
        If the ratio arrays disagree in shape.
    """
    systematic = np.asarray(systematic_llr, dtype=float)
    parity = np.asarray(parity_llr, dtype=float)
    apriori = (
        np.zeros_like(systematic)
        if apriori_llr is None
        else np.asarray(apriori_llr, dtype=float)
    )
    if systematic.ndim != 1 or not (systematic.shape == parity.shape == apriori.shape):
        msg = (
            f"systematic {systematic.shape}, parity {parity.shape} and a priori "
            f"{apriori.shape} must be the same one-dimensional shape"
        )
        raise ValueError(msg)
    length, n_states = systematic.size, trellis.n_states
    variables = [Variable(f"s{t}", n_states) for t in range(length + 1)]
    start = np.full(n_states, IMPOSSIBLE_EDGE)
    start[0] = 0.0
    factors = [Factor("start", ("s0",), start)]
    inputs = np.array([0.0, 1.0])
    for t in range(length):
        table = np.full((n_states, n_states), IMPOSSIBLE_EDGE)
        weight = -inputs[None, :] * (systematic[t] + apriori[t]) - (
            trellis.parity * parity[t]
        )
        for u in (0, 1):
            table[np.arange(n_states), trellis.next_state[:, u]] = weight[:, u]
        factors.append(Factor(f"T{t}", (f"s{t}", f"s{t + 1}"), table))
    if terminated:
        end = np.full(n_states, IMPOSSIBLE_EDGE)
        end[0] = 0.0
        factors.append(Factor("end", (f"s{length}",), end))
    return FactorGraph(variables, factors)
