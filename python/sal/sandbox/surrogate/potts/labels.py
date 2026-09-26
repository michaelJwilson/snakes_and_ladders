"""The labellers the curriculum trains on: planted, ICM and alpha-expansion (issue #1067).

A planted draw is labelled by its planted states, which is the ground state
by :func:`~sal.search.potts_starts.recovery_bound`. Any other field is
labelled by a solver: ICM, the lower of two descents to convergence, or
alpha-expansion. The trial trained on ICM labels first and found the model
reproduced ICM; the gain over ICM appeared once the labels were
alpha-expansion's (#1067).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from sal.backend import Backend
from sal.search.alpha_expansion import alpha_expansion
from sal.search.icm import iterated_conditional_modes
from sal.sim.graph import PottsGraph
from sal.sim.potts import SpatioTilingParams

#: The descent's sweep cap, standing in for "until a sweep changes nothing".
ICM_SWEEPS = 100_000


class Labeller(StrEnum):
    """Which solver labels a field a planted labelling does not."""

    ICM = "icm"
    EXPANSION = "expansion"


@dataclass(frozen=True)
class Label:
    """A labelling and where it came from.

    Parameters
    ----------
    labelling : np.ndarray
        ``int64``, shape ``(n_nodes,)``.
    source : str
        ``"planted"``, ``"icm-argmax"``, ``"icm-random"`` or ``"expansion"``.
    """

    labelling: np.ndarray
    source: str


def planted_label(params: SpatioTilingParams) -> Label:
    """Each site labelled its tile's favoured state.

    Returns
    -------
    Label
    """
    return Label(np.asarray(params.states[params.tiles], dtype=np.int64), "planted")


def icm_label(graph: PottsGraph, field: np.ndarray, seed: int) -> Label:
    """ICM to convergence from the field argmax and from a random start, the lower kept.

    Returns
    -------
    Label
        The argmax descent on a tie.
    """
    n_states = field.shape[1]
    rng = np.random.default_rng(seed)
    argmax = iterated_conditional_modes(
        graph,
        field,
        n_states=n_states,
        rng=rng,
        start=field.argmax(1),
        max_iterations=ICM_SWEEPS,
    )
    random = iterated_conditional_modes(
        graph, field, n_states=n_states, rng=rng, max_iterations=ICM_SWEEPS
    )
    if argmax.energy <= random.energy:
        return Label(np.asarray(argmax.labelling, dtype=np.int64), "icm-argmax")
    return Label(np.asarray(random.labelling, dtype=np.int64), "icm-random")


def expansion_label(graph: PottsGraph, field: np.ndarray) -> Label:
    """Alpha-expansion from the field argmax, on the Rust cut.

    Returns
    -------
    Label
    """
    result = alpha_expansion(
        graph, field, n_states=field.shape[1], backend=Backend.RUST
    )
    return Label(np.asarray(result.labelling, dtype=np.int64), "expansion")


def solver_label(
    graph: PottsGraph, field: np.ndarray, labeller: Labeller, seed: int
) -> Label:
    """The label ``labeller`` gives ``field``; ``seed`` drives ICM's random start.

    Returns
    -------
    Label
    """
    if labeller is Labeller.EXPANSION:
        return expansion_label(graph, field)
    return icm_label(graph, field, seed)
