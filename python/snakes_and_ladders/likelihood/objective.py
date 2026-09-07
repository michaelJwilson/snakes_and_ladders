"""The phylogenetic instance of ``snakes_and_ladders.opt.objective.Objective``.

It lives here rather than in ``snakes_and_ladders.opt`` because ``opt/CLAUDE.md`` forbids
that package from importing any application module, and a test enforces it
(issue #63). The dependency runs one way -- an application knows the
optimization vocabulary, the optimizer knows no models -- and this file is
that direction made concrete: it imports ``snakes_and_ladders.opt.constrain`` and is
imported by nothing in ``snakes_and_ladders.opt``.

**Not every branch length is identifiable, and which ones are depends on the
root.** Under a reversible model the likelihood is invariant to where the
root sits along the branch it subdivides (Felsenstein, *Inferring
Phylogenies*, ch. 16, the "pulley principle"), so on a **rooted binary** tree
the two branches below the root are confounded: only their sum is estimable.
Measured on the 8-taxon fixture, shifting mass between them across a 9:1
range moves the log-likelihood by at most 3.6e-12 -- floating-point noise --
while the same shift between two non-root siblings moves it by 14.7. Fitting
them separately would leave the observed information singular and every
confidence interval undefined.

So the pair is fitted as **one** parameter and reported as its sum. On a tree
in the trifurcating-root convention there is no such pair and every branch is
estimable, which is the usual reason phylogenetic inference is done on
unrooted topologies.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from snakes_and_ladders.likelihood import pruning_torch
from snakes_and_ladders.opt.constrain import (
    free_from_log_simplex,
    free_from_positive,
    log_simplex,
    positive,
)
from snakes_and_ladders.sim.gtr import n_exchangeabilities
from snakes_and_ladders.sim.tree import Node

# Starting length for every branch, in expected substitutions per site. Small
# and uninformative rather than fitted from the data: a data-dependent start
# would make the fit's own convergence part of what the recovery test
# measures.
_INITIAL_BRANCH_LENGTH = 0.1


class BranchLengthObjective:
    """Negative log-likelihood of an alignment as a function of branch lengths.

    Parameters
    ----------
    tau : Node
        The topology, held fixed. Branch lengths on the ``Node`` tree are
        ignored -- ``pruning_torch`` takes them as a tensor, per
        ``likelihood/CLAUDE.md``.
    k : int
        Number of states.
    pi : np.ndarray
        Root state distribution, shape ``(k,)``.
    alignment : Mapping[str, np.ndarray]
        Observed states per taxon.
    dtype : torch.dtype
        Precision; ``float64`` by default, since a finite-difference
        derivative check is meaningless in ``float32``.
    device : torch.device | str | None
        Where to run. ``None`` leaves the tensor on the default device.

    Raises
    ------
    ValueError
        If ``tau``'s root has fewer than two children, which is not a tree
        this likelihood is defined on.
    """

    def __init__(
        self,
        tau: Node,
        k: int,
        pi: np.ndarray,
        alignment: Mapping[str, np.ndarray],
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        if len(tau.children) < 2:
            msg = (
                f"root {tau.name!r} has {len(tau.children)} children; a tree "
                f"needs a root with at least 2"
            )
            raise ValueError(msg)

        self._tau = tau
        self._k = k
        self._pi = pi
        self._alignment = dict(alignment)
        self._dtype = dtype
        self._device = device

        self._branch_order = pruning_torch.branch_order(tau)
        root_children = [child.name for child in tau.children]
        # The confounded pair, if there is one. Two children means a rooted
        # binary tree and a flat direction; three means the trifurcating-root
        # convention, where every branch stands on its own.
        self._merged: tuple[int, int] | None = (
            (
                self._branch_order.index(root_children[0]),
                self._branch_order.index(root_children[1]),
            )
            if len(root_children) == 2
            else None
        )

    @property
    def parameter_names(self) -> list[str]:
        """Names of the *estimable* parameters, in ``theta`` order.

        One per branch, except that on a rooted binary tree the two branches
        below the root appear once, as ``"a+b"``, because only their sum is
        estimable.
        """
        if self._merged is None:
            return list(self._branch_order)
        first, second = self._merged
        names = [
            name for index, name in enumerate(self._branch_order) if index != second
        ]
        names[names.index(self._branch_order[first])] = (
            f"{self._branch_order[first]}+{self._branch_order[second]}"
        )
        return names

    @property
    def n_parameters(self) -> int:
        """Number of estimable parameters."""
        return len(self._branch_order) - (0 if self._merged is None else 1)

    def initial(self) -> torch.Tensor:
        """A short, uninformative branch length everywhere."""
        return torch.full(
            (self.n_parameters,),
            float(np.log(_INITIAL_BRANCH_LENGTH)),
            dtype=self._dtype,
            device=self._device,
        )

    def constrain(self, theta: torch.Tensor) -> Mapping[str, torch.Tensor]:
        """The estimable branch lengths ``theta`` encodes, in ``parameter_names`` order.

        Parameters
        ----------
        theta : torch.Tensor
            Unconstrained parameters.

        Returns
        -------
        Mapping[str, torch.Tensor]
            ``{"branch_lengths": ...}``, positive, one entry per estimable
            parameter -- so on a rooted binary tree the root pair appears
            once, as its sum.
        """
        return {"branch_lengths": positive(theta)}

    def branch_lengths(self, theta: torch.Tensor) -> torch.Tensor:
        """Expand ``theta`` to one length per branch, in ``branch_order``.

        The estimable sum is split evenly between the two root branches.
        Any split gives the same likelihood -- that is what "confounded"
        means -- so halving is a reporting convention, not an estimate, and
        neither half should be quoted as one.

        Parameters
        ----------
        theta : torch.Tensor
            Unconstrained parameters.

        Returns
        -------
        torch.Tensor
            One positive length per branch, ordered for ``pruning_torch``.
        """
        lengths = positive(theta)
        if self._merged is None:
            return lengths
        first, second = self._merged
        head = lengths[:second]
        tail = lengths[second:]
        halved = head.clone()
        halved[first] = head[first] / 2.0
        return torch.cat([halved, halved[first : first + 1], tail])

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood of the alignment at these branch lengths."""
        return -pruning_torch.log_likelihood(
            self._tau,
            self._k,
            self._pi,
            self._alignment,
            self.branch_lengths(theta),
        )

    def fitted_tree(self, theta: torch.Tensor) -> Node:
        """The topology with ``theta``'s branch lengths attached.

        The inverse of :meth:`theta_from_truth`, and the form anything that
        draws or serializes a fitted tree needs --- ``pruning_torch`` keeps
        lengths out of the ``Node`` structure, which is right for
        differentiation and wrong for display.

        On a rooted binary tree the estimable sum is split evenly between the
        two root branches, per :meth:`branch_lengths`. That is a drawing
        convention, which is exactly what this method is for; neither half is
        an estimate.

        Parameters
        ----------
        theta : torch.Tensor
            Unconstrained parameters.

        Returns
        -------
        Node
            A copy of the topology carrying the fitted lengths.
        """
        lengths = dict(
            zip(
                self._branch_order,
                self.branch_lengths(theta).detach().tolist(),
                strict=True,
            )
        )

        def rebuild(node: Node) -> Node:
            children = tuple(rebuild(child) for child in node.children)
            length = lengths.get(node.name)
            return Node(name=node.name, branch_length=length, children=children)

        return rebuild(self._tau)

    def theta_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """The unconstrained vector whose :meth:`constrain` is ``named``.

        The inverse of the constraint map, keyed exactly as :meth:`constrain`
        returns (issue #268). Only the estimable branch lengths appear, as in
        :meth:`constrain`: a confounded pair is reported as the combination
        that is identified, and there is nothing to invert for the parts that
        are not.

        Parameters
        ----------
        named : Mapping[str, torch.Tensor]
            Constrained parameters, under :meth:`constrain`'s own keys.

        Returns
        -------
        torch.Tensor
            ``theta`` such that ``constrain(theta)`` returns ``named``.
        """
        return free_from_positive(named["branch_lengths"])

    def theta_from_truth(self, tau: Node) -> torch.Tensor:
        """Place a tree's own branch lengths in the unconstrained coordinates.

        Parameters
        ----------
        tau : Node
            A tree with the same topology as this objective's, carrying the
            lengths to encode.

        Returns
        -------
        torch.Tensor
            ``theta`` whose estimable parameters are those lengths, with the
            root pair replaced by its sum.
        """
        lengths = pruning_torch.branch_lengths_from_tree(tau, dtype=self._dtype)
        if self._merged is not None:
            first, second = self._merged
            summed = lengths[first] + lengths[second]
            keep = [index for index in range(lengths.numel()) if index != second]
            lengths = lengths[keep].clone()
            lengths[keep.index(first)] = summed
        return free_from_positive(lengths)


class SubstitutionModelObjective:
    """Fits branch lengths, GTR exchangeabilities and ``pi`` together.

    The follow-up to :class:`BranchLengthObjective`, and a modelling change
    rather than an optimizer one: Jukes-Cantor has no free rate parameters,
    so fitting ``Q`` and ``pi`` needs a model that has some
    (:mod:`snakes_and_ladders.sim.gtr`).

    **Three gauges, all load-bearing.** Each removes a direction along which
    the likelihood is exactly flat, and a flat direction makes the observed
    information singular and every interval undefined --- not merely wide.

    * ``Q`` is normalized to one expected substitution per unit time, or it
      trades off against every branch length at once.
    * One exchangeability is held at 1, or the whole vector can be scaled and
      the rate normalization undoes it.
    * ``pi`` goes through :func:`snakes_and_ladders.opt.constrain.log_simplex`, which pins
      its first logit, or a constant can be added to all of them.

    Parameters
    ----------
    tau : Node
        The topology, held fixed.
    k : int
        Number of states.
    alignment : Mapping[str, np.ndarray]
        Observed states per taxon.
    dtype : torch.dtype
        Precision; ``float64`` by default.
    device : torch.device | str | None
        Where to run.
    """

    def __init__(
        self,
        tau: Node,
        k: int,
        alignment: Mapping[str, np.ndarray],
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        # The branch-length block, including the confounded-root-pair merge,
        # is exactly BranchLengthObjective's. Reused rather than restated so
        # the two cannot drift; pi is a placeholder here because this class
        # fits its own.
        self._branches = BranchLengthObjective(
            tau, k, np.full(k, 1.0 / k), alignment, dtype=dtype, device=device
        )
        self._tau = tau
        self._k = k
        self._alignment = dict(alignment)
        self._dtype = dtype
        self._device = device

        self._n_free_exchangeabilities = n_exchangeabilities(k) - 1
        rows, columns = np.triu_indices(k, k=1)
        self._rows = torch.as_tensor(rows, device=device)
        self._columns = torch.as_tensor(columns, device=device)

    @property
    def n_parameters(self) -> int:
        """Branch lengths, free exchangeabilities, and free ``pi`` entries."""
        return (
            self._branches.n_parameters + self._n_free_exchangeabilities + (self._k - 1)
        )

    @property
    def parameter_names(self) -> list[str]:
        """Names in ``theta`` order, branch lengths first."""
        return [
            *self._branches.parameter_names,
            *(f"s{index}" for index in range(self._n_free_exchangeabilities)),
            *(f"pi{index}" for index in range(1, self._k)),
        ]

    def _split(self, theta: torch.Tensor) -> tuple[torch.Tensor, ...]:
        first = self._branches.n_parameters
        second = first + self._n_free_exchangeabilities
        return theta[:first], theta[first:second], theta[second:]

    def rate_matrix(self, theta: torch.Tensor) -> torch.Tensor:
        """The normalized GTR rate matrix ``theta`` encodes.

        Parameters
        ----------
        theta : torch.Tensor
            Unconstrained parameters.

        Returns
        -------
        torch.Tensor
            Rate matrix of shape ``(k, k)``, differentiable throughout.
        """
        _, free_exchangeabilities, free_pi = self._split(theta)
        pi = torch.exp(log_simplex(free_pi))
        values = torch.cat(
            [
                positive(free_exchangeabilities),
                torch.ones(1, dtype=self._dtype, device=self._device),
            ]
        )
        upper = torch.zeros(
            (self._k, self._k), dtype=self._dtype, device=self._device
        ).index_put((self._rows, self._columns), values)
        symmetric = upper + upper.T
        rate = symmetric * pi.unsqueeze(0)
        rate = rate - torch.diag(rate.sum(dim=1))
        scale = -(pi * torch.diagonal(rate)).sum()
        return rate / scale

    def initial(self) -> torch.Tensor:
        """Uninformative: short branches, equal exchangeabilities, uniform ``pi``.

        That start is Jukes-Cantor exactly, which is a deliberate choice: the
        fit begins at the model the fixtures' simpler tests use, so any
        departure it reaches is something the data asked for.
        """
        return torch.cat(
            [
                self._branches.initial(),
                torch.zeros(
                    self._n_free_exchangeabilities,
                    dtype=self._dtype,
                    device=self._device,
                ),
                torch.zeros(self._k - 1, dtype=self._dtype, device=self._device),
            ]
        )

    def constrain(self, theta: torch.Tensor) -> Mapping[str, torch.Tensor]:
        """Branch lengths, the full exchangeability vector, and ``pi``.

        Parameters
        ----------
        theta : torch.Tensor
            Unconstrained parameters.

        Returns
        -------
        Mapping[str, torch.Tensor]
            ``branch_lengths`` (estimable entries only, as for
            :class:`BranchLengthObjective`), ``exchangeabilities`` (all of
            them, the last pinned at 1), and ``pi``.
        """
        branches, free_exchangeabilities, free_pi = self._split(theta)
        return {
            "branch_lengths": positive(branches),
            "exchangeabilities": torch.cat(
                [
                    positive(free_exchangeabilities),
                    torch.ones(1, dtype=self._dtype, device=self._device),
                ]
            ),
            "pi": torch.exp(log_simplex(free_pi)),
        }

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood under the GTR model ``theta`` encodes."""
        branches, _, free_pi = self._split(theta)
        return -pruning_torch.log_likelihood(
            self._tau,
            self._k,
            torch.exp(log_simplex(free_pi)),
            self._alignment,
            self._branches.branch_lengths(branches),
            rate_matrix=self.rate_matrix(theta),
        )

    def theta_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """The unconstrained vector whose :meth:`constrain` is ``named``.

        The inverse of the constraint map, keyed exactly as :meth:`constrain`
        returns (issue #268). The last exchangeability is pinned at 1 by the
        gauge and so is dropped rather than inverted — it is not a free
        parameter and has no coordinate to return to.

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
                free_from_positive(named["branch_lengths"]),
                free_from_positive(named["exchangeabilities"][:-1]),
                free_from_log_simplex(torch.log(named["pi"])),
            ]
        )

    def theta_from_truth(
        self, tau: Node, exchangeabilities: np.ndarray, pi: np.ndarray
    ) -> torch.Tensor:
        """Place a known truth in the unconstrained coordinates.

        Parameters
        ----------
        tau : Node
            A tree carrying the true branch lengths.
        exchangeabilities : np.ndarray
            The true exchangeabilities, all of them. Rescaled here so the
            last is 1, which is the same model: scaling them is undone by the
            rate normalization.
        pi : np.ndarray
            The true stationary distribution.

        Returns
        -------
        torch.Tensor
            ``theta`` whose constrained parameters are that truth.
        """
        scaled = torch.as_tensor(exchangeabilities, dtype=self._dtype)
        scaled = scaled / scaled[-1]
        return torch.cat(
            [
                self._branches.theta_from_truth(tau),
                free_from_positive(scaled[:-1]),
                free_from_log_simplex(
                    torch.log(torch.as_tensor(pi, dtype=self._dtype))
                ),
            ]
        )
