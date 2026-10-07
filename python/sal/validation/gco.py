"""gco as the timing reference for alpha expansion (issue #974).

gco is Veksler and Delong's C++ expansion (gco-v3.0), wrapped by
``gco-wrapper``; neither shares code with
:mod:`sal.search.alpha_expansion`. Its terms are research-use,
and it runs only in ``scripts/gco.py``, in a subprocess.

**Not an oracle.** Expansion stops at a local minimum, gco starts at label 0
where the package starts at the field's argmax, and gco cuts on integer
costs: its float mode scales every term by 1e5 and truncates. The two
labellings may differ and both be correct, so what is compared is the energy
of each under :func:`sal.sim.potts.energy`, and the properties
a correct expansion has whatever it started from.

**The model.** The package's energy is ``-sum_i h_i[s_i] - sum J_ij
[s_i == s_j]``, which is ``sum_i D_i(s_i) + sum J_ij [s_i != s_j]`` less a
constant, with ``D_i(a) = -h_i[a]`` shifted by its row minimum so every data
cost is non-negative; that is gco's form with a Potts smooth cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sal.external.potts_inputs import expansion_inputs
from sal.external.runner import run
from sal.sim.graph import PottsGraph
from sal.sim.potts import energy, site_field

#: The script this adapter runs.
SCRIPT = "gco"


@dataclass(frozen=True)
class Expansion:
    """gco's labelling, its energy in the package's terms, and its times."""

    labelling: np.ndarray
    #: The labelling's energy under :func:`sal.sim.potts.energy`.
    energy: float
    #: gco's own figure, on its scaled integer costs, in their units.
    gco_energy: float
    #: Wall seconds of ``expansion()`` alone.
    seconds: float
    #: Wall seconds building the graph in gco.
    build_seconds: float
    #: Peak resident bytes the build and the expansion added.
    peak_bytes: int


def alpha_expansion(
    graph: PottsGraph,
    field: np.ndarray,
    n_states: int,
    *,
    start: np.ndarray | None = None,
    move: str = "expansion",
) -> Expansion:
    """gco's expansion to convergence on the package's Potts model.

    ``move="swap"`` runs gco's alpha-beta swap to convergence instead
    (issue #997). The inputs are :func:`sal.external.potts_inputs.expansion_inputs`,
    the bytes :func:`sal.external.potts.ground_state` sends (#1282).
    """
    values = site_field(np.asarray(field, dtype=float), graph.n_nodes)
    if values.shape[1] != n_states:
        msg = f"the field has {values.shape[1]} states, not {n_states}"
        raise ValueError(msg)
    inputs = expansion_inputs(graph, values, start=start, swap=move == "swap")
    result = run(SCRIPT, inputs)
    labelling = result.outputs["labels"]
    return Expansion(
        labelling=labelling,
        energy=energy(graph, values, labelling),
        gco_energy=float(result.outputs["gco_energy"]),
        seconds=result.seconds,
        build_seconds=float(result.outputs["build_seconds"]),
        peak_bytes=result.peak_bytes,
    )
