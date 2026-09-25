"""The Potts chain in a field as a declared instance: its parameters, its loader, its exact draw.

The chain's parameters and the recursion that draws from them exactly lived in
``opt/potts.py`` beside the objectives that fit them, which put a problem's
definition and its simulator in the fitter and made the fixture registry reach
into ``opt`` for a fixture (issue #830). They are here now; the objectives stay
in ``opt`` and import the parameters from here, and nothing in either moved a
number: the draw is :func:`sal.sim.potts.simulate_potts` on an
open chain, bitwise what the copy in ``opt`` returned (#813).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np

from sal.sim.graph import BoundaryCondition, lattice_graph
from sal.sim.potts import simulate_potts

_REQUIRED_FIELDS = frozenset(
    {"seed", "n_chains", "chain_length", "n_states", "coupling", "field"}
)


@dataclass(frozen=True)
class PottsParams:
    """Fully-specified truth for a Potts-chain fixture.

    Parameters
    ----------
    n_states : int
        Number of states per site, >= 2.
    chain_length : int
        Sites per chain, >= 2. A length-1 chain has no coupling term and
        would leave ``J`` unidentifiable.
    n_chains : int
        Independent chains simulated from the truth.
    coupling : float
        The true ``J``.
    field : np.ndarray
        The true ``h``, shape ``(n_states,)``, canonicalized on load to
        ``logsumexp(h) == 0`` so it is comparable with a fitted field.
    seed : int
        Seed for ``np.random.default_rng``.
    """

    n_states: int
    chain_length: int
    n_chains: int
    coupling: float
    field: np.ndarray
    seed: int

    #: The fields :func:`sal.fixtures.load_params` checks are present before
    #: calling :meth:`from_declared`.
    required_fields: ClassVar[frozenset[str]] = _REQUIRED_FIELDS

    @classmethod
    def from_declared(cls, declared: Mapping[str, Any], path: Path, /) -> Self:
        """Build the truth from a Potts fixture's declared mapping.

        ``declared`` is the mapping
        :func:`sal.fixtures.load_params` read from ``path``
        with :attr:`required_fields` present; ``path`` names the file in
        every error.

        Parsed truth, with ``field`` canonicalized to ``logsumexp(h) == 0``.

        Raises
        ------
        ValueError
            If a required field is missing, ``field`` does not have shape
            ``(n_states,)``, or a size is too small to identify the parameters.
        """
        n_states = int(declared["n_states"])
        chain_length = int(declared["chain_length"])
        if n_states < 2:
            msg = f"{path}: n_states must be >= 2, got {n_states}"
            raise ValueError(msg)
        if chain_length < 2:
            msg = f"{path}: chain_length must be >= 2, got {chain_length}"
            raise ValueError(msg)

        field = np.asarray(declared["field"], dtype=np.float64)
        if field.shape != (n_states,):
            msg = f"{path}: field has shape {field.shape}, expected ({n_states},)"
            raise ValueError(msg)
        # Canonicalize the gauge here rather than demanding the yaml be written
        # in it: h and h + c are the same model, and a hand-written fixture
        # should not have to solve for c.
        field = field - float(np.log(np.exp(field).sum()))

        return cls(
            n_states=n_states,
            chain_length=chain_length,
            n_chains=int(declared["n_chains"]),
            coupling=float(declared["coupling"]),
            field=field,
            seed=int(declared["seed"]),
        )


def simulate_chains(
    params: PottsParams, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Draw ``n_chains`` exact samples from the truth in ``params``.

    The draw is :func:`sal.sim.potts.simulate_potts` on an
    open chain, which is where the exact backward-message recursion lives;
    the copy that stood here drew the same states, bit for bit, at the
    declared instance (#813), so this is the call and not a second
    simulator. Exact, not MCMC: the fixture carries no equilibration
    assumption (root ``CLAUDE.md``, "Simulate Component-Wise").

    Parameters
    ----------
    params : PottsParams
        The generating truth.
    rng : np.random.Generator | None
        Generator to draw from. ``None`` builds one from ``params.seed``, the
        stream every fixture was drawn on (issue #829).

    Returns
    -------
    np.ndarray
        Integer states, shape ``(n_chains, chain_length)``.
    """
    graph = lattice_graph(
        (params.chain_length,), BoundaryCondition.OPEN, params.coupling
    )
    rng = np.random.default_rng(params.seed) if rng is None else rng
    return simulate_potts(graph, params.field, rng, params.n_chains).configurations
