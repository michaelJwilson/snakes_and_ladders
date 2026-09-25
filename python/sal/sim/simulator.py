"""One way to simulate a problem: ``(params, rng) -> dataset``, registered beside its loader.

The calling audit (`docs/reviews/2026-09-20-calling.md`) found *simulate*
spelled five ways across the problems: the params unpacked (the tree), a
generator beside the params (the mixtures, the coupled model), the seed inside
the params (the HMM, the count pairs), a graph and a field (the Potts models),
and an encoder plus a channel with no simulator at all (the codes). A caller
who has drawn one problem's data cannot draw the next without reading its
module. This module is the one shape --- :class:`Simulator` --- and the
registry :data:`SIMULATORS`, keyed by the declared model exactly as
:data:`sal.sim.fixtures.LOADERS` is, so the loader and the
simulator of a model are found under one name.

**Referential, and bitwise.** Every entry calls the function its problem has
always drawn with, in the same order, from the generator it is handed; where a
simulator built its generator from ``params.seed`` it still does when handed
none, so a fixture's draw is the same bits either way and the tests say so
draw for draw. ``sim/CLAUDE.md``'s rule --- a generator, never a seed --- is
what the shape states: the seed is the fixture's and the caller derives the
stream from it once (issue #829).

**What is not here, and why**, one reason each in :data:`NOT_SIMULATED`: the
count pairs draw a keyed stream per vertex so the Rust twin can be checked
vertex by vertex (#671), which one generator cannot reproduce; the codes have
an encoder and a channel and no dataset record yet; a frustrated lattice or a
glass is an instance rather than a draw; a test function has no data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

import numpy as np

from sal.sim.emission_mixture import simulate_emission_mixture
from sal.sim.graph import lattice_graph
from sal.sim.hmm import simulate_sequences
from sal.sim.mixture import simulate_mixture
from sal.sim.potts import SimulatedPottsDataset, simulate_potts
from sal.sim.simulate import SimulatedDataset, simulate_alignment
from sal.sim.spatio_sequential import simulate_spatio_sequential

if TYPE_CHECKING:
    from sal.sim.params import SimulationParams
    from sal.sim.potts import PottsLatticeParams, SpatioOnlyParams
    from sal.sim.potts_chain import PottsParams

P_contra = TypeVar("P_contra", contravariant=True)
D_co = TypeVar("D_co", covariant=True)


@runtime_checkable
class Simulator(Protocol[P_contra, D_co]):
    """Data drawn from a declared model under a declared generator, with the truth beside it."""

    def __call__(self, params: P_contra, rng: np.random.Generator) -> D_co:
        """Draw one dataset from ``params`` using ``rng``, and nothing else."""
        ...  # pragma: no cover - protocol


def simulate_tree(
    params: SimulationParams, rng: np.random.Generator, n_sites: int | None = None
) -> SimulatedDataset:
    """:func:`~sal.sim.simulate.simulate_alignment` on the fixture's tree, states and root.

    ``n_sites`` overrides the fixture's site count: the one argument a caller
    varies without changing the model (issue #1010), so no call site spells
    out the fixture's other three fields.
    """
    return simulate_alignment(
        params.tau,
        params.k,
        params.pi,
        rng,
        params.n_sites if n_sites is None else n_sites,
    )


def simulate_potts_lattice(
    params: PottsLatticeParams, rng: np.random.Generator
) -> SimulatedPottsDataset:
    """:func:`~sal.sim.potts.simulate_potts` on the lattice the fixture declares."""
    graph = lattice_graph(params.shape, params.boundary, params.coupling)
    return simulate_potts(graph, params.field, rng, params.n_samples, params.burn_in)


def simulate_spatio_only(
    params: SpatioOnlyParams, rng: np.random.Generator
) -> SimulatedPottsDataset:
    """:func:`~sal.sim.potts.simulate_potts` on the fixture's own graph and field."""
    return simulate_potts(
        params.graph, params.field, rng, params.n_samples, params.burn_in
    )


def simulate_potts_chain(params: PottsParams, rng: np.random.Generator) -> np.ndarray:
    """The exact open-chain draw, as :func:`sal.opt.potts.simulate_chains` makes it.

    The same recursion through the same call, so the states are the bits that
    function returns; it is written here rather than imported so ``sim`` does
    not reach into ``opt`` for a draw (#830).
    """
    from sal.sim.graph import BoundaryCondition

    graph = lattice_graph(
        (params.chain_length,), BoundaryCondition.OPEN, params.coupling
    )
    return simulate_potts(graph, params.field, rng, params.n_chains).configurations


#: The simulator per declared model, under the name `LOADERS` reads it by.
SIMULATORS: dict[str, Simulator[Any, Any]] = {
    "jukes-cantor": simulate_tree,
    "potts-chain": simulate_potts_chain,
    "potts-lattice": simulate_potts_lattice,
    "spatio-only": simulate_spatio_only,
    "hidden-markov": simulate_sequences,
    "gaussian-mixture": simulate_mixture,
    "emission-mixture": simulate_emission_mixture,
    "spatio-sequential": simulate_spatio_sequential,
}

#: The declared models with no simulator, each against the reason.
NOT_SIMULATED: dict[str, str] = {
    "spatio-sequential-counts": (
        "the draw is a keyed stream per vertex, `default_rng([seed, node])`, so "
        "the Rust twin is checked vertex by vertex (#671); one generator cannot "
        "reproduce it and the module states the exception to `sim/CLAUDE.md`"
    ),
    "ldpc": "an encoder and a channel exist and no dataset record does: a transmission is drawn by the test that needs it",
    "bicycle": "as `ldpc`",
    "bicycle-css": "as `ldpc`; the error draw is `sim.css.sample_x_error`",
    "turbo": "as `ldpc`; the noise scale is set by `likelihood.turbo` from a declared `E_b/N_0`",
    "polar": "as `ldpc`; the frozen set is the construction's (#826) and a transmission is drawn by the test that needs it",
    "spatio-tiling": "an instance and not a draw: the planted states are the truth a ground state is read against",
    "frustrated-lattice": "an instance and not a draw: the graph, or the glass at a frustration, is the data",
    "test-functions": "a surface has no data",
}


__all__ = [
    "NOT_SIMULATED",
    "SIMULATORS",
    "Simulator",
    "simulate_potts_chain",
    "simulate_potts_lattice",
    "simulate_spatio_only",
    "simulate_tree",
]
