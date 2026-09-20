"""A 1-D Potts chain in an external field: reference instance of ``Objective``.

Not phylogenetics, and that is its job (issue #63): an interface justified by
a single model is shaped by it, so the abstraction is tested against something
whose only similarity to a tree is that it factorizes over discrete states.

The model, for ``q`` states on a chain of length ``L``::

    P(s) proportional to exp( J * sum_i delta(s_i, s_{i+1}) + sum_i h[s_i] )

with ``J`` a scalar coupling and ``h`` an external field. The normalizer is
computed exactly by transfer matrix in log space, which is the same
sum-product recursion Felsenstein pruning performs on a tree (Mezard &
Montanari, ch. 2; Koller & Friedman for the general framing) -- the reason
one optimizer is expected to serve both.

**Gauge.** The likelihood is invariant to adding a constant to every entry of
``h``, so ``h`` is fixed to ``logsumexp(h) == 0`` via
:func:`snakes_and_ladders.opt.constrain.log_simplex`. Without that the model is
unidentifiable and a fitted field has no value to compare against truth.

**Relationship to** :mod:`snakes_and_ladders.sim.potts`. That module
generalizes ``simulate_chains`` below to an arbitrary graph and its exact
open-chain sampler is the general-per-edge-coupling form of the one here. They
are not consolidated: ``opt/CLAUDE.md``'s "no application imports" rule,
enforced by ``tests/regression/opt/test_opt_objective.py``, forbids importing
``snakes_and_ladders.sim``. Issue #186 tracks the duplication, by moving
``PottsParams``/``load_potts_params`` out of ``snakes_and_ladders.opt`` as
issue #171 moved ``snakes_and_ladders.opt.hmm``'s truth type into
``snakes_and_ladders.sim.hmm``.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence

import numpy as np
import torch

from snakes_and_ladders.opt.constrain import free_from_log_simplex, log_simplex
from snakes_and_ladders.opt.objective import Objective


def _log_transfer(coupling: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
    """``T[i, j] = J * delta(i, j) + h[j]``, the one matrix every site carries."""
    return coupling * torch.eye(
        field.shape[0], dtype=field.dtype, device=field.device
    ) + field.unsqueeze(0)


def squaring_is_cheaper(n_states: int, length: int) -> bool:
    """Whether :func:`log_partition_by_squaring` does no more work than the recursion.

    Counted in ``q ** 2``-element `logsumexp` calls, which is one step of the
    recursion. The recursion takes ``n = length - 1`` of them. Squaring takes
    ``floor(log2 n)`` matrix squarings, each ``q ** 3`` and so ``q`` of those
    units, plus one vector-matrix product per set bit of ``n``; it is taken
    when the total does not exceed ``n``, where it touches no more elements
    *and* makes no more calls and so cannot lose on either term.

    **The route reads the alphabet, not the chain length** (root
    ``CLAUDE.md``, "Cost depends on the data"): the cube in ``q`` is what a
    reassociation buys its shorter product with. Measured over ``q`` in 2..64
    and ``length`` in 2..1,024, squaring is 4.51x at ``q = 3, length = 64``
    and 0.04x at ``q = 64, length = 4`` -- a rule reading the length alone
    would take both (issue #754, ``docs/experiments/028``).

    The rule is conservative by construction: it declines wins it cannot
    bound, 1.25x at ``q = 3, length = 8`` and 2.53x at ``q = 32,
    length = 64``. The recursion is what this function always did, so the
    branch it declines to is never a regression.

    Parameters
    ----------
    n_states : int
        States per site, ``q``.
    length : int
        Chain length.

    Returns
    -------
    bool
        True where squaring is taken.
    """
    exponent = length - 1
    if exponent <= 1:
        return True
    return n_states * (exponent.bit_length() - 1) + exponent.bit_count() <= exponent


def log_partition_by_recursion(
    coupling: torch.Tensor, field: torch.Tensor, length: int
) -> torch.Tensor:
    """``log Z`` by carrying one message across the chain, site by site.

    The reference: ``length - 1`` vector-matrix products in log space, in the
    order the chain is written. :func:`log_partition_by_squaring` is pinned
    against it.

    Parameters
    ----------
    coupling : torch.Tensor
        Scalar ``J``.
    field : torch.Tensor
        ``h``, shape ``(q,)``.
    length : int
        Chain length, >= 1.

    Returns
    -------
    torch.Tensor
        Scalar ``log Z``, differentiable with respect to both parameters.
    """
    log_transfer = _log_transfer(coupling, field)
    alpha = field
    for _ in range(length - 1):
        alpha = torch.logsumexp(alpha.unsqueeze(1) + log_transfer, dim=0)
    return torch.logsumexp(alpha, dim=0)


def log_partition_by_squaring(
    coupling: torch.Tensor, field: torch.Tensor, length: int
) -> torch.Tensor:
    """``log Z`` with the transfer product reassociated by repeated squaring.

    The chain is homogeneous -- one ``J`` and one ``h`` for every site -- so
    ``T ** (length - 1)`` is a power of a single matrix, and a power
    reassociates: ``T ** 63`` is five squarings and six vector-matrix
    products rather than 63 products in sequence. The count goes from
    ``O(length)`` `logsumexp` calls to ``O(log length)``, which is what the
    autograd tape is built over as well as what the forward pass runs.

    Exact, not an approximation: the reassociation changes the *order* of the
    log-space sums and nothing else. It is bitwise with
    :func:`log_partition_by_recursion` where the order happens to coincide
    and within
    :data:`~snakes_and_ladders.likelihood.device.CROSS_DEVICE_RTOL_FLOAT64`
    where it does not; the realized difference is 6.815e-16 relative at the
    stress instance (``q = 3``, ``length = 64``), and 6.13e-14 is the largest
    over ``q`` in 2..8 and ``length`` in 1..129 where the route is taken.

    A site-dependent field would make the product inhomogeneous and this
    route wrong. ``field`` is one vector, so the case cannot arise here; it
    is what would have to change first.

    Parameters
    ----------
    coupling : torch.Tensor
        Scalar ``J``.
    field : torch.Tensor
        ``h``, shape ``(q,)``, shared by every site.
    length : int
        Chain length, >= 1.

    Returns
    -------
    torch.Tensor
        Scalar ``log Z``, differentiable with respect to both parameters.
    """
    alpha = field
    exponent = length - 1
    if exponent > 0:
        power = _log_transfer(coupling, field)
        while exponent:
            if exponent & 1:
                alpha = torch.logsumexp(alpha.unsqueeze(1) + power, dim=0)
            exponent >>= 1
            if exponent:
                power = torch.logsumexp(power.unsqueeze(2) + power.unsqueeze(0), dim=1)
    return torch.logsumexp(alpha, dim=0)


def log_partition(
    coupling: torch.Tensor, field: torch.Tensor, length: int
) -> torch.Tensor:
    """Exact ``log Z`` for a chain of ``length`` sites, by transfer matrix.

    Two routes compute the same number and :func:`squaring_is_cheaper` picks
    between them on ``q`` and ``length`` together. Either is exact; they
    differ in the order the log-space sums are taken, by 6.815e-16 relative at
    the stress instance.

    Parameters
    ----------
    coupling : torch.Tensor
        Scalar ``J``.
    field : torch.Tensor
        ``h``, shape ``(q,)``.
    length : int
        Chain length, >= 1.

    Returns
    -------
    torch.Tensor
        Scalar ``log Z``, differentiable with respect to both parameters.
    """
    if squaring_is_cheaper(int(field.shape[0]), length):
        return log_partition_by_squaring(coupling, field, length)
    return log_partition_by_recursion(coupling, field, length)


def graph_statistics(
    n_states: int, edges: Sequence[tuple[int, int]], n_nodes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Enumerate every configuration's sufficient statistics for a Potts graph.

    The energy is ``J * agreement(s) + sum_i h[s_i]``, so a configuration
    enters the partition function through two numbers: agreeing edges, and
    sites per state. Enumerating those once turns ``log Z`` into a `logsumexp`
    over precomputed rows, so a gradient step does not re-walk the
    configuration space.

    **This is exact and exponential, and the caller owns that trade.**
    ``n_states ** n_nodes`` rows: 19,683 for a 3-state 3x3 lattice, the size
    #170's simulator validates against. A 4x4 lattice at 3 states is 43
    million and is not this function's business.

    Parameters
    ----------
    n_states : int
        States per site, ``q``.
    edges : Sequence[tuple[int, int]]
        Undirected edges as ``(i, j)`` node indices. Passed rather than taken
        from a `PottsGraph`: ``opt/CLAUDE.md`` forbids importing
        ``snakes_and_ladders.sim``, so a caller unpacks the graph.
    n_nodes : int
        Sites in the graph.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Agreeing-edge count per configuration, shape ``(q ** n_nodes,)``, and
        state counts per configuration, shape ``(q ** n_nodes, q)``.
    """
    configurations = torch.tensor(
        list(itertools.product(range(n_states), repeat=n_nodes)), dtype=torch.long
    )
    agreements = torch.zeros(configurations.shape[0], dtype=torch.float64)
    for first, second in edges:
        agreements += (configurations[:, first] == configurations[:, second]).to(
            torch.float64
        )
    counts = torch.zeros(
        (configurations.shape[0], n_states), dtype=torch.float64
    ).scatter_add_(
        1, configurations, torch.ones_like(configurations, dtype=torch.float64)
    )
    return agreements, counts


def log_partition_graph(
    coupling: torch.Tensor,
    field: torch.Tensor,
    agreements: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    """Exact ``log Z`` for a Potts graph, from enumerated statistics.

    Differentiable in ``coupling`` and ``field``: the normalizer is where
    every gradient of the likelihood comes from.

    Parameters
    ----------
    coupling : torch.Tensor
        Scalar ``J``.
    field : torch.Tensor
        ``h``, shape ``(n_states,)``.
    agreements, counts : torch.Tensor
        As returned by :func:`graph_statistics`, for the graph in question.

    Returns
    -------
    torch.Tensor
        Scalar ``log Z``.
    """
    return torch.logsumexp(coupling * agreements + counts @ field, dim=0)


class PottsLatticeObjective(Objective):
    """Negative log-likelihood of Potts configurations on a graph.

    The lattice counterpart of :class:`PottsObjective`, the same shape: an
    unconstrained vector, a differentiable scalar, and a map back to named
    parameters. Nothing in ``snakes_and_ladders.opt.fit`` changes to carry it,
    which is `opt/CLAUDE.md`'s claim for the interface.

    The normalizer is **exact**, by enumeration, so a fitted optimum can be
    checked against a brute-force scan rather than against the optimizer's own
    convergence. That bounds the sizes this is for; see
    :func:`graph_statistics`.

    Parameters
    ----------
    configurations : np.ndarray
        Observed states, shape ``(n_samples, n_nodes)``.
    n_states : int
        Number of states, ``q``.
    edges : Sequence[tuple[int, int]]
        The graph's undirected edges, as node-index pairs.
    dtype : torch.dtype
        Precision; ``float64`` by default, since a finite-difference check is
        meaningless in ``float32``.

    Raises
    ------
    ValueError
        If ``configurations`` is not 2-D, or an edge names a node outside it.
    """

    def __init__(
        self,
        configurations: np.ndarray,
        n_states: int,
        edges: Sequence[tuple[int, int]],
        dtype: torch.dtype = torch.float64,
    ) -> None:
        observed = torch.as_tensor(configurations, dtype=torch.long)
        if observed.ndim != 2:
            msg = (
                f"expected configurations of shape (n_samples, n_nodes), got "
                f"{tuple(observed.shape)}"
            )
            raise ValueError(msg)
        n_nodes = int(observed.shape[1])
        for edge in edges:
            if not (0 <= edge[0] < n_nodes and 0 <= edge[1] < n_nodes):
                msg = f"edge {edge} names a node outside [0, {n_nodes})"
                raise ValueError(msg)

        self._n_states = n_states
        self._dtype = dtype
        self._n_samples = int(observed.shape[0])
        # The data enters only through these two, exactly as for the chain.
        agreement = torch.zeros(self._n_samples, dtype=dtype)
        for first, second in edges:
            agreement += (observed[:, first] == observed[:, second]).to(dtype)
        self._agreement_total = agreement.sum()
        self._counts = torch.zeros(n_states, dtype=dtype)
        self._counts.scatter_add_(
            0, observed.reshape(-1), torch.ones(observed.numel(), dtype=dtype)
        )
        self._agreements, self._configuration_counts = graph_statistics(
            n_states, edges, n_nodes
        )

    def initial(self) -> torch.Tensor:
        """A deliberately uninformative start: zero coupling, uniform field."""
        return torch.zeros(self._n_states, dtype=self._dtype)

    def constrain(self, theta: torch.Tensor) -> Mapping[str, torch.Tensor]:
        """Split ``theta`` into the coupling and the gauge-fixed field.

        The gauge is the chain's: adding a constant to every entry of ``h``
        shifts the energy and ``log Z`` by the same amount, so without it the
        model is unidentifiable and no fitted field has a value to compare
        against.
        """
        return {"coupling": theta[0], "field": log_simplex(theta[1:])}

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood of every observed configuration."""
        constrained = self.constrain(theta)
        coupling, field = constrained["coupling"], constrained["field"]
        log_z = log_partition_graph(
            coupling, field, self._agreements, self._configuration_counts
        )
        unnormalized = coupling * self._agreement_total + (self._counts * field).sum()
        return -(unnormalized - self._n_samples * log_z)

    def theta_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """The unconstrained vector whose :meth:`constrain` is ``named``.

        The inverse of the constraint map, keyed exactly as :meth:`constrain`
        returns. It is what lets a fit produced by *any* optimizer be given an
        interval: the observed information is a property of the objective at a
        point, and this is how a point stated in the model's own parameters
        becomes one the Hessian can be taken at (issue #268).

        Parameters
        ----------
        named : Mapping[str, torch.Tensor]
            Constrained parameters, under :meth:`constrain`'s own keys.

        Returns
        -------
        torch.Tensor
            ``theta`` such that ``constrain(theta)`` returns ``named``.
        """
        return torch.cat(
            [
                named["coupling"].reshape(1).to(self._dtype),
                free_from_log_simplex(named["field"].to(self._dtype)),
            ]
        )

    def theta_from_truth(self, coupling: float, field: np.ndarray) -> torch.Tensor:
        """Place a known truth in the unconstrained coordinates.

        Returns
        -------
        torch.Tensor
            ``theta`` such that ``constrain(theta)`` returns this truth.
        """
        as_tensor = torch.as_tensor(field, dtype=self._dtype)
        return torch.cat(
            [
                torch.tensor([coupling], dtype=self._dtype),
                free_from_log_simplex(as_tensor),
            ]
        )


class PottsObjective(Objective):
    """Negative log-likelihood of Potts chains, as an :class:`~snakes_and_ladders.opt.objective.Objective`.

    Parameters
    ----------
    chains : np.ndarray
        Observed states, shape ``(n_chains, chain_length)``.
    n_states : int
        Number of states, ``q``.
    dtype : torch.dtype
        Precision of the computation; ``float64`` by default, since a
        finite-difference derivative check is meaningless in ``float32``.
    """

    def __init__(
        self,
        chains: np.ndarray,
        n_states: int,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self._chains = torch.as_tensor(chains, dtype=torch.long)
        self._n_states = n_states
        self._dtype = dtype
        self._length = int(self._chains.shape[1])
        # Number of adjacent-pair agreements, per chain -- the only statistic
        # of the data the coupling sees, so it is computed once.
        self._agreements = (self._chains[:, :-1] == self._chains[:, 1:]).sum(dim=1)
        self._counts = torch.zeros(n_states, dtype=dtype)
        self._counts.scatter_add_(
            0, self._chains.reshape(-1), torch.ones(self._chains.numel(), dtype=dtype)
        )

    def initial(self) -> torch.Tensor:
        """A deliberately uninformative start: zero coupling, uniform field."""
        return torch.zeros(self._n_states, dtype=self._dtype)

    def constrain(self, theta: torch.Tensor) -> Mapping[str, torch.Tensor]:
        """Split ``theta`` into the coupling and the gauge-fixed field."""
        return {"coupling": theta[0], "field": log_simplex(theta[1:])}

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood of every observed chain."""
        constrained = self.constrain(theta)
        coupling, field = constrained["coupling"], constrained["field"]
        log_z = log_partition(coupling, field, self._length)
        unnormalized = (
            coupling * self._agreements.to(self._dtype).sum()
            + (self._counts * field).sum()
        )
        return -(unnormalized - self._chains.shape[0] * log_z)

    def theta_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """The unconstrained vector whose :meth:`constrain` is ``named``.

        The inverse of the constraint map, keyed exactly as :meth:`constrain`
        returns. It is what lets a fit produced by *any* optimizer be given an
        interval: the observed information is a property of the objective at a
        point, and this is how a point stated in the model's own parameters
        becomes one the Hessian can be taken at (issue #268).

        Parameters
        ----------
        named : Mapping[str, torch.Tensor]
            Constrained parameters, under :meth:`constrain`'s own keys.

        Returns
        -------
        torch.Tensor
            ``theta`` such that ``constrain(theta)`` returns ``named``.
        """
        return torch.cat(
            [
                named["coupling"].reshape(1).to(self._dtype),
                free_from_log_simplex(named["field"].to(self._dtype)),
            ]
        )

    def theta_from_truth(self, coupling: float, field: np.ndarray) -> torch.Tensor:
        """Place a known truth in the unconstrained coordinates.

        Parameters
        ----------
        coupling : float
            True ``J``.
        field : np.ndarray
            True ``h``, already gauge-fixed to ``logsumexp(h) == 0``.

        Returns
        -------
        torch.Tensor
            ``theta`` such that ``constrain(theta)`` returns this truth.
        """
        as_tensor = torch.as_tensor(field, dtype=self._dtype)
        return torch.cat(
            [
                torch.tensor([coupling], dtype=self._dtype),
                free_from_log_simplex(as_tensor),
            ]
        )
