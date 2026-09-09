"""``rustworkx`` fronting the Swendsen-Wang cluster labelling (issues #322, #389).

The front that was measured and declined. A Swendsen-Wang sweep activates
bonds, labels the connected components of the bond graph, and recolours each
component; the labelling is the term issue #389 proposed giving to a library.
:func:`cluster_labels` computes it with ``rustworkx.connected_components``
over a ``PyGraph`` rebuilt from the active bonds, and
:func:`swendsen_wang_sweep` is the whole sweep with that labelling in place of
:func:`~snakes_and_ladders.search.potts_mcmc._roots`, which is the unit the
decline was measured in.

**Declined twice over**, both records in `STATUS.md` and
`docs/experiments/008-frameworks-on-three-hot-paths.md`.

On the clock: 1.12x to 1.17x slower per sweep than the pointer-doubling
labelling this branch adopted --- 0.229 against 0.195 ms at extent 8, 0.725
against 0.650 at 16, 1.568 against 1.386 at 24, 5.846 against 5.220 at 48.
``rustworkx`` beats the union-find *walk* it would have replaced (28.5 against
54.6 µs at extent 8, 1158.4 against 1699.8 at 48), and loses to the vectorized
roots that replaced the walk instead. One thread on a shared four-core Linux
x86-64 host under the exclusive lock, best of 5.

On the chain, which is the reason that stands whatever a later version does to
the clock: ``rustworkx`` **renumbers the clusters**. The adopted labelling
gives every site its union-find root, so the clusters are recoloured in
ascending root order; ``connected_components`` numbers them by discovery in
its own scan, and the two orders disagree --- at extent 8, seed 8, from the
sixteenth cluster on. The recolouring then takes the same draws in a different
order, so the sampler composes a *different* Markov chain from the same seed,
and experiment 001's autocorrelation times and every seeded Potts figure move
with it. The partition is identical; the sequence of states is not.

`tests/regression/search/test_potts_mcmc_rustworkx.py` pins both halves: the
partition against the adopted labelling, and the renumbering as a measured
difference rather than one sorted away.

Imports ``rustworkx`` at module scope: it is the ``frameworks`` extra, and a
caller without it gets an ``ImportError`` here rather than a silent fallback
to the labelling this exists to referee.
"""

from __future__ import annotations

import numpy as np
import rustworkx

from snakes_and_ladders.search.potts_mcmc import _Bonds, _recolour


def bond_graph(n_nodes: int, bonds: _Bonds, active: np.ndarray) -> rustworkx.PyGraph:
    """The active bonds as a ``rustworkx.PyGraph`` on ``n_nodes`` nodes.

    Rebuilt per call rather than mutated: the active set is redrawn every
    sweep, and the ``PyGraph`` API reached here has no cheaper way to replace
    an edge set than to build the graph again. That is what the timing above
    measures, and a graph held across sweeps and mutated is a different
    measurement nobody has taken.

    Parameters
    ----------
    n_nodes : int
        Sites, which are the graph's nodes whether or not a bond reaches them:
        an isolated site is a cluster of one and must be recoloured.
    bonds : _Bonds
        The edge endpoints, as
        :func:`~snakes_and_ladders.search.potts_mcmc._bonds_of` builds them.
    active : np.ndarray
        Boolean per edge: the bonds this sweep opened.

    Returns
    -------
    rustworkx.PyGraph
        Simple rather than a multigraph, since a repeated pair contributes
        nothing to the components.
    """
    graph = rustworkx.PyGraph(multigraph=False)
    graph.add_nodes_from(range(n_nodes))
    graph.add_edges_from_no_data(
        [
            (int(bonds.first[edge]), int(bonds.second[edge]))
            for edge in np.flatnonzero(active)
        ]
    )
    return graph


def cluster_labels(n_nodes: int, bonds: _Bonds, active: np.ndarray) -> np.ndarray:
    """One label per site, numbered by ``rustworkx``'s component discovery order.

    The array
    :func:`~snakes_and_ladders.search.potts_mcmc._roots` returns holds a site
    index --- the component's union-find root --- where this holds a component
    counter from ``0``. Both label the same partition; neither ordering is
    the other's, and this one is what makes the sweep below a different chain.

    Parameters
    ----------
    n_nodes : int
        Sites.
    bonds : _Bonds
        The edge endpoints.
    active : np.ndarray
        Boolean per edge.

    Returns
    -------
    np.ndarray
        Shape ``(n_nodes,)``, values in ``[0, n_clusters)``.
    """
    labels = np.empty(n_nodes, dtype=np.int64)
    components = rustworkx.connected_components(bond_graph(n_nodes, bonds, active))
    for index, component in enumerate(components):
        labels[list(component)] = index
    return labels


def swendsen_wang_sweep(
    state: np.ndarray, bonds: _Bonds, field: np.ndarray, rng: np.random.Generator
) -> None:
    """One Swendsen-Wang sweep with :func:`cluster_labels` doing the labelling.

    Signature for signature what
    :func:`~snakes_and_ladders.search.potts_mcmc._swendsen_wang_sweep` takes,
    and identical to it up to the labelling: the same bond activation from the
    same draws, the same stable sort grouping the members of a cluster in
    ascending site order, the same recolouring. What differs is the order the
    *clusters* are visited in, and that is enough to make the two sweeps
    different chains --- see this module's docstring.

    Mutates ``state`` in place.

    Parameters
    ----------
    state : np.ndarray
        The current colouring, one entry per site.
    bonds : _Bonds
        The edge endpoints and per-edge bond probabilities.
    field : np.ndarray
        The per-colour external field the recolouring accepts on.
    rng : np.random.Generator
        The chain's only source of randomness, read in the order the adopted
        sweep reads it: the bond draws first, then one proposal and one
        uniform per cluster.
    """
    like = state[bonds.first] == state[bonds.second]
    active = like & (rng.random(bonds.first.shape[0]) < bonds.activation)

    n_nodes = state.shape[0]
    labels = cluster_labels(n_nodes, bonds, active)
    order = np.argsort(labels, kind="stable")
    grouped = labels[order]
    starts = np.flatnonzero(np.r_[True, grouped[1:] != grouped[:-1]])
    for begin, end in zip(starts, np.r_[starts[1:], n_nodes], strict=True):
        _recolour(state, order[begin:end], field, rng)


__all__ = ["bond_graph", "cluster_labels", "swendsen_wang_sweep"]
