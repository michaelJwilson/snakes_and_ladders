"""Simulated bifurcation: a Potts ground state read off a relaxed oscillator per site and label.

Every ground-state route here is sequential over sites or over moves --- the
cut augments paths, alpha expansion cycles labels, a cluster move grows a
component --- and the Gumbel-softmax relaxation of :mod:`sal.learn.relaxed`
ascends by autodiff. Simulated bifurcation (Goto, Tatsumura and Dixon 2019;
the ballistic form of Goto et al. 2021) is the one route whose inner loop is
an elementwise update over every site at once: each label of each site is an
oscillator ``x`` in ``[-1, 1]`` with momentum ``y``, the field and the
neighbours push it, a ramp ``a(t)`` from zero pins it to a corner, and the
labelling is the arg max over labels at the end. That shape is what root
``CLAUDE.md``'s GPU rule was written for, and issue #823 admits this module as
the first kernel that rule can be read against.

**The relaxation and what it maximizes.** With one-hot rows the energy
``E(s) = -sum_i h_i[s_i] - sum_(ij) J_ij [s_i == s_j]`` is ``-S(x)`` for
``S(x) = sum_i sum_a h[i, a] x[i, a] + sum_(ij) J_ij sum_a x[i, a] x[j, a]``, and
``S`` is read on the box ``x in [-1, 1]^(n_nodes x n_states)`` --- multilinear,
so its maximum over the box sits at a corner (the argument
:mod:`sal.learn.relaxed` makes for the simplex holds for the box,
term by term). The dynamics ascend ``S`` while the ramp pulls ``x`` to
``{-1, +1}``; the readout is the label with the largest ``x`` at each site,
and its energy is :func:`sal.sim.potts.energy` on that labelling,
never the relaxed ``S``. At two labels the two oscillators of a site carry one
sign between them, and the readout is the sign.

**One arithmetic, two array libraries.** The update is written once over an
array namespace and run on NumPy or on ``torch``, so the oracle and the device
path cannot drift: the torch path is pinned to the NumPy one to a stated
tolerance --- the scatter-add over edges reduces in an order neither library
promises --- and the labelling is pinned equal. A heuristic with no guarantee
is refereed by what has one: the cut at two labels at any size, enumeration at
nine sites, and the dual bound's certificate where neither reaches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from sal import oxisal
from sal.backend import Backend, refuse_backend
from sal.opt.termination import Termination
from sal.sim.graph import CompressedAdjacency, PottsGraph
from sal.sim.potts import energy, site_field, states_of

if TYPE_CHECKING:
    import torch

#: Ramp end and integration step of the ballistic dynamics, Goto et al. (2021)'s
#: values; the ramp ``a(t)`` runs from zero to ``A_END`` over the declared steps.
A_END = 1.0
DEFAULT_DT = 0.5
DEFAULT_STEPS = 1000

#: Amplitude of the uniform start, small so no site begins committed.
START_SCALE = 0.1


@dataclass(frozen=True)
class BifurcationResult:
    """A labelling read off the oscillators, and what it cost.

    Parameters
    ----------
    labelling : np.ndarray
        ``(n_nodes,)`` integer labels, the arg max over each site's oscillators
        of the best replica.
    energy : float
        :func:`sal.sim.potts.energy` of that labelling --- the
        discrete energy, never the relaxed score.
    steps : int
        Integration steps each replica ran.
    n_replicas : int
        Independent starts run; the labelling is the best of them.
    termination : Termination | None
        The integration's: it runs its declared steps and tests no criterion,
        so it ends on the count (issue #860).
    """

    labelling: np.ndarray
    energy: float
    steps: int
    n_replicas: int
    termination: Termination | None = None


def _coupling_scale(graph: PottsGraph, rows: np.ndarray) -> float:
    """``c_0`` so the largest drive any site can see is one half.

    Goto's ``0.5 / (sigma_J sqrt(N))`` is set for a dense coupling matrix in
    zero field; on a sparse graph in a per-site field it leaves the field as
    the whole drive and every site saturates in its own field's direction on
    the first steps, which is the field-only labelling and not a search (issue
    #823 measured a mean gap of 3.24 against the cut at that scale, 0.13 at
    this one). The scale here is read from the instance: the largest sum over
    a site of ``|J|`` on its edges and ``|h|`` over its labels, so the
    restoring term and the drive start comparable and the ramp decides.
    """
    total = np.zeros(graph.n_nodes)
    index = graph.edge_index
    magnitude = np.abs(graph.edge_coupling)
    np.add.at(total, index[:, 0], magnitude)
    np.add.at(total, index[:, 1], magnitude)
    largest = float(np.max(total + np.abs(rows).max(axis=1)))
    return 0.5 / largest if largest > 0.0 else 1.0


def _integrate_numpy(
    rows: np.ndarray,
    adjacency: CompressedAdjacency,
    x: np.ndarray,
    *,
    steps: int,
    dt: float,
    c0: float,
    discrete: bool,
) -> np.ndarray:
    """The ballistic update over NumPy arrays; ``x`` is ``(n_nodes, n_states)`` and returned.

    The coupling sum is read over the graph's own compressed rows
    (:attr:`~sal.sim.graph.PottsGraph.incidence`), one
    neighbour slot at a time across every node, so each node's sum is formed
    in row order from zero: the order ``bifurcation_integrate`` follows, and
    stated here rather than left to a reduction whose order NumPy does not
    promise (issue #997). Two `np.add.at` scatters were 8.1 s of a 16.0 s
    run at 142^2.
    """
    degrees = np.diff(adjacency.offsets)
    slots = [
        (nodes, adjacency.offsets[nodes] + slot)
        for slot in range(int(degrees.max()) if degrees.size else 0)
        for nodes in [np.flatnonzero(degrees > slot)]
    ]
    y = np.zeros_like(x)
    summed = np.empty_like(x)
    for step in range(steps):
        ramp = A_END * step / steps
        drive = np.sign(x) if discrete else x
        summed.fill(0.0)
        for nodes, entries in slots:
            summed[nodes] += (
                adjacency.couplings[entries, None]
                * drive[adjacency.neighbours[entries]]
            )
        force = rows + summed
        y += dt * (-(A_END - ramp) * x + c0 * force)
        x += dt * A_END * y
        outside = np.abs(x) > 1.0
        x[outside] = np.sign(x[outside])
        y[outside] = 0.0
    return x


def _integrate_torch(
    rows: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
    couplings: torch.Tensor,
    x: torch.Tensor,
    *,
    steps: int,
    dt: float,
    c0: float,
    discrete: bool,
) -> torch.Tensor:
    """The same update over tensors, operation for operation, on ``x.device``."""
    import torch

    y = torch.zeros_like(x)
    weights = couplings[:, None]
    for step in range(steps):
        ramp = A_END * step / steps
        force = rows.clone()
        drive = torch.sign(x) if discrete else x
        force.index_add_(0, first, weights * drive[second])
        force.index_add_(0, second, weights * drive[first])
        y += dt * (-(A_END - ramp) * x + c0 * force)
        x += dt * A_END * y
        outside = x.abs() > 1.0
        x = torch.where(outside, torch.sign(x), x)
        y = torch.where(outside, torch.zeros_like(y), y)
    return x


def simulated_bifurcation(
    graph: PottsGraph,
    field: np.ndarray,
    rng: np.random.Generator,
    *,
    n_states: int | None = None,
    steps: int = DEFAULT_STEPS,
    dt: float = DEFAULT_DT,
    n_replicas: int = 1,
    coupling_scale: float | None = None,
    discrete: bool = False,
    backend: Backend = Backend.RUST,
    device: torch.device | str | None = None,
) -> BifurcationResult:
    """A low-energy labelling by ballistic simulated bifurcation.

    Parameters
    ----------
    graph : PottsGraph
        The instance. Couplings of either sign: a repulsive coupling is where
        the cut does not apply and this still runs, refereed there by the dual
        bound rather than by an optimum.
    field : np.ndarray
        ``(n_states,)`` or ``(n_nodes, n_states)``, broadcast by
        :func:`sal.sim.potts.site_field`.
    rng : np.random.Generator
        Draws every replica's start, and nothing else; the dynamics are
        deterministic from it.
    n_states : int | None
        ``q >= 2``, read from the field's state axis; given, it is checked
        against it (issue #1091).
    steps : int
        Integration steps per replica; the ramp reaches ``A_END`` at the last.
    dt : float
        Integration step.
    n_replicas : int
        Independent starts; the best labelling by discrete energy is returned.
    coupling_scale : float | None
        ``c_0``; ``None`` reads it from the instance (:func:`_coupling_scale`).
    discrete : bool
        ``False`` is the ballistic variant, the default: on the declared glass
        and the three-state lattice it reached the enumerated optimum where
        the discrete one (``True``, the neighbours entering by their signs,
        Goto et al. 2021) did not, and the reverse was measured nowhere.
    backend : Backend
        :data:`~sal.backend.Backend.RUST`, the default since
        issue #997, runs the whole integration in
        ``oxisal.bifurcation_integrate`` over the graph's
        compressed rows, bit for bit the NumPy route on a lattice: 1.3 s
        against 10.2 s at 142^2 and q = 10.
        :data:`~sal.backend.Backend.PYTHON` runs NumPy, the
        oracle; :data:`~sal.backend.Backend.TORCH` the same
        arithmetic over tensors on ``device``.
    device : torch.device | str | None
        Where the torch path runs; ignored by the NumPy one.

    Returns
    -------
    BifurcationResult

    Raises
    ------
    ValueError
        If ``n_states < 2``, ``steps < 1``, ``n_replicas < 1``, ``dt <= 0``, or
        the backend is neither PYTHON nor TORCH.
    """
    n_states = states_of(field, graph.n_nodes, n_states)
    if n_states < 2:
        msg = f"n_states must be >= 2, got {n_states}"
        raise ValueError(msg)
    if steps < 1 or n_replicas < 1:
        msg = f"steps and n_replicas must be >= 1, got {steps}, {n_replicas}"
        raise ValueError(msg)
    if dt <= 0.0:
        msg = f"dt must be > 0, got {dt}"
        raise ValueError(msg)
    refuse_backend(
        "simulated_bifurcation", backend, (Backend.PYTHON, Backend.TORCH, Backend.RUST)
    )
    rows = site_field(
        np.asarray(field, dtype=np.float64), graph.n_nodes, n_states=n_states
    )
    c0 = _coupling_scale(graph, rows) if coupling_scale is None else coupling_scale
    index = graph.edge_index.reshape(-1, 2)
    first, second = index[:, 0], index[:, 1]
    couplings = np.asarray(graph.edge_coupling, dtype=np.float64)

    best_labelling: np.ndarray | None = None
    best_energy = np.inf
    for _ in range(n_replicas):
        start = rng.uniform(-START_SCALE, START_SCALE, size=rows.shape)
        if backend is Backend.RUST:
            adjacency = graph.incidence
            final = start.copy()
            oxisal.bifurcation_integrate(
                np.ascontiguousarray(rows).reshape(-1),
                adjacency.offsets,
                adjacency.neighbours,
                adjacency.couplings,
                final.reshape(-1),
                n_states,
                steps,
                dt,
                c0,
                A_END,
                discrete,
            )
        elif backend is Backend.PYTHON:
            final = _integrate_numpy(
                rows,
                graph.incidence,
                start.copy(),
                steps=steps,
                dt=dt,
                c0=c0,
                discrete=discrete,
            )
        else:
            import torch

            final = (
                _integrate_torch(
                    torch.as_tensor(rows, device=device),
                    torch.as_tensor(np.array(first), device=device),
                    torch.as_tensor(np.array(second), device=device),
                    torch.as_tensor(np.array(couplings), device=device),
                    torch.as_tensor(start, device=device),
                    steps=steps,
                    dt=dt,
                    c0=c0,
                    discrete=discrete,
                )
                .cpu()
                .numpy()
            )
        labelling = np.asarray(np.argmax(final, axis=1), dtype=np.int64)
        value = energy(graph, rows, labelling)
        if value < best_energy:
            best_energy, best_labelling = value, labelling
    assert best_labelling is not None
    return BifurcationResult(
        labelling=best_labelling,
        energy=best_energy,
        steps=steps,
        n_replicas=n_replicas,
        # The integration runs its declared steps and tests nothing, so it
        # ends on the count, never on a criterion (issue #860).
        termination=Termination.after(steps, converged=False),
    )


__all__ = [
    "A_END",
    "DEFAULT_DT",
    "DEFAULT_STEPS",
    "BifurcationResult",
    "simulated_bifurcation",
]
