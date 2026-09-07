"""Topology search: the outer loop that the optimizer deliberately excludes.

`opt/CLAUDE.md` records the seam this module is the first user of. A discrete
move changes the *structure* being fitted, so it changes what the parameter
vector means and how long it is; it cannot be a step inside a fit over a
fixed-length vector. It constructs a **new** objective, and something outside
the optimizer has to own that construction. This is that something.

The consequence is that nothing here needed to change `snakes_and_ladders.opt`. The same
``fit`` that fits a Potts chain scores every candidate topology, which is the
claim issue #63 made and could not itself test.

**Budget is counted in candidates scored, not in seconds.** ``DEV.md`` forbids
ranking performance on CI hardware, and a wall-clock budget would make a
result depend on the machine that produced it, so a run would not be
reproducible from the generator it was given. What scoring a candidate costs
is reported beside it -- fits, and likelihood evaluations -- because since
issue #289 the two are no longer one number: a fit warm-started from the
parent's lengths converges in a fraction of the evaluations a cold one takes,
and a lazily scored candidate costs one.

**What carries from a topology to its neighbour.** A neighbour shares every
branch but the few a move touched, and a branch is the split it induces
(:func:`snakes_and_ladders.search.topology.branch_splits`), so the parent's fitted
lengths start the neighbour's fit on every branch that persists. The same
identity lets an unfitted evaluation of the neighbour reuse the parent's
subtree partials (:class:`~snakes_and_ladders.likelihood.pruning_torch.PartialCache`),
which is what lazy scoring -- rank every neighbour cheaply, fit only the top
few, always fit the accepted move in full -- rests on. Warm starts are on by
default because they change where a fit *starts* and not where it ends;
lazy scoring is opt-in because it changes which candidates are fitted.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import torch

from snakes_and_ladders.likelihood.objective import (
    BranchLengthObjective,
    SubstitutionModelObjective,
)
from snakes_and_ladders.likelihood.pruning_torch import (
    PartialCache,
    log_likelihood_cached,
)
from snakes_and_ladders.opt.fit import fit
from snakes_and_ladders.search.topology import (
    Topology,
    branch_splits,
    leaf_bipartitions,
    nni_neighbours,
    random_topology,
    spr_neighbours,
)


class MoveSet(StrEnum):
    """Which neighbourhood the search proposes from."""

    NNI = "nni"
    SPR = "spr"


class Model(StrEnum):
    """Which substitution model the continuous fit uses.

    ``JC`` fits branch lengths alone: Jukes-Cantor has no free rate
    parameters and its stationary distribution is uniform by construction, so
    there is nothing else in it to fit. ``GTR`` additionally fits the
    exchangeabilities and the stationary distribution.
    """

    JC = "jc"
    GTR = "gtr"


@dataclass(frozen=True)
class Inference:
    """The outcome of a search.

    Parameters
    ----------
    topology : Topology
        The best topology found.
    log_likelihood : float
        Its maximized log-likelihood -- a full fit's, whatever ``lazy_top``.
    parameters : Mapping[str, np.ndarray]
        The fitted continuous parameters, under the names the model uses.
    evaluations : int
        Candidates scored, which is what the budget counts. Each is a full
        fit when ``lazy_top`` is ``None`` and one likelihood evaluation
        otherwise, with the best ``lazy_top`` of them then fitted.
    trace : tuple[float, ...]
        Log-likelihood after each accepted move, starting with the initial
        topology's. Its length minus one is the number of moves taken.
    converged : bool
        Whether the search stopped because no neighbour improved, rather
        than because the budget ran out. A search that ran out of budget has
        not finished, and reporting its result as an optimum would be wrong.
        A zero budget --- a fit with no search --- reports ``True``, since
        there was no search to leave unfinished.
    fits : int
        Candidates fitted in full, the initial topology included.
    likelihood_evaluations : int
        Forward passes spent, inside fits and on lazy scores together: the
        unit a budget-matched comparison between two ways of searching is
        stated in (issue #289).
    """

    topology: Topology
    log_likelihood: float
    parameters: Mapping[str, np.ndarray]
    evaluations: int
    trace: tuple[float, ...]
    converged: bool
    fits: int = 0
    likelihood_evaluations: int = 0


@dataclass(frozen=True)
class _Fitted:
    """A scored topology, with what the next neighbour can start from."""

    value: float
    parameters: Mapping[str, np.ndarray]
    named: Mapping[str, torch.Tensor]
    lengths_by_split: Mapping[frozenset[str], float]
    default_length: float
    evaluations: int


class _Counting:
    """An objective that counts its evaluations; the protocol otherwise."""

    def __init__(
        self, inner: BranchLengthObjective | SubstitutionModelObjective
    ) -> None:
        self.inner = inner
        self.calls = 0

    def initial(self) -> torch.Tensor:
        return self.inner.initial()

    def constrain(self, theta: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return self.inner.constrain(theta)

    def theta_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.inner.theta_from(named)

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.inner(theta)


def _objective(
    model: Model, topology: Topology, k: int, alignment: Mapping[str, np.ndarray]
) -> BranchLengthObjective | SubstitutionModelObjective:
    if model is Model.JC:
        return BranchLengthObjective(topology, k, np.full(k, 1.0 / k), alignment)
    return SubstitutionModelObjective(topology, k, alignment)


def _warm_lengths(topology: Topology, warm: _Fitted) -> torch.Tensor:
    """The parent's length on every branch the neighbour kept, the default elsewhere."""
    return torch.tensor(
        [
            warm.lengths_by_split.get(split, warm.default_length)
            for split in branch_splits(topology)
        ],
        dtype=torch.float64,
    )


def _warm_theta(
    objective: BranchLengthObjective | SubstitutionModelObjective,
    topology: Topology,
    warm: _Fitted,
) -> torch.Tensor:
    named = dict(warm.named)
    named["branch_lengths"] = _warm_lengths(topology, warm)
    return objective.theta_from(named)


def _score(
    model: Model,
    topology: Topology,
    k: int,
    alignment: Mapping[str, np.ndarray],
    warm: _Fitted | None = None,
) -> _Fitted:
    """Fit a candidate, from the parent's lengths when given, and record the cost."""
    objective = _objective(model, topology, k, alignment)
    counting = _Counting(objective)
    theta0 = None if warm is None else _warm_theta(objective, topology, warm)
    result = fit(counting, theta0=theta0)
    named = {
        name: value.detach()
        for name, value in objective.constrain(result.theta).items()
    }
    lengths = named["branch_lengths"].tolist()
    return _Fitted(
        value=-result.value,
        parameters={name: value.numpy() for name, value in named.items()},
        named=named,
        lengths_by_split=dict(zip(branch_splits(topology), lengths, strict=True)),
        default_length=float(torch.exp(objective.initial()[0])),
        evaluations=counting.calls,
    )


def _lazy_score(
    model: Model,
    topology: Topology,
    k: int,
    alignment: Mapping[str, np.ndarray],
    warm: _Fitted,
    cache: PartialCache,
) -> float:
    """One likelihood evaluation at the parent's parameters, no fit."""
    lengths = _warm_lengths(topology, warm)
    if model is Model.JC:
        return log_likelihood_cached(
            topology, k, np.full(k, 1.0 / k), alignment, lengths, cache
        )
    objective = SubstitutionModelObjective(topology, k, alignment)
    theta = _warm_theta(objective, topology, warm)
    return log_likelihood_cached(
        topology,
        k,
        warm.named["pi"],
        alignment,
        lengths,
        cache,
        rate_matrix=objective.rate_matrix(theta).detach(),
    )


def infer(
    alignment: Mapping[str, np.ndarray],
    k: int,
    *,
    topology: Topology | None = None,
    model: Model = Model.JC,
    moves: MoveSet = MoveSet.NNI,
    max_evaluations: int = 200,
    rng: np.random.Generator | None = None,
    warm_start: bool = True,
    lazy_top: int | None = None,
) -> Inference:
    """Hill-climb over topologies, fitting continuous parameters per candidate.

    Parameters
    ----------
    alignment : Mapping[str, np.ndarray]
        Observed states per taxon, each of shape ``(n_sites,)``.
    k : int
        Number of states.
    topology : Topology | None
        Where to start. ``None`` draws a random topology from ``rng``.
    model : Model
        Substitution model for the continuous fit.
    moves : MoveSet
        Neighbourhood the search proposes from.
    max_evaluations : int
        Maximum candidates scored. The initial topology's own fit is not
        counted against it.
    rng : np.random.Generator | None
        Source of the starting topology, required when ``topology`` is
        ``None`` and unused otherwise. Passed in rather than seeded here, so
        a caller running an ensemble gets independent starts rather than the
        same one repeatedly (`sim/CLAUDE.md`, issue #240). There is no
        default: a generator made here would be either unseeded, and the run
        irreproducible, or seeded from a constant nobody declared.
    warm_start : bool
        Start each candidate's fit from the parent's fitted lengths on the
        branches it kept (issue #289). The optimum is the same; what changes
        is the evaluations spent reaching it, which the result reports.
    lazy_top : int | None
        ``None`` fits every candidate. An integer scores every candidate with
        one likelihood evaluation at the parent's parameters, reusing the
        parent's subtree partials, then fits only the best that many; the
        accepted move is always a full fit. Which candidates get fitted
        changes, so this is opt-in, and its cost in missed optima is the
        measurement the regression suite reports.

    Returns
    -------
    Inference
        The best topology found and its fitted parameters.

    Raises
    ------
    ValueError
        If the alignment has fewer than 4 taxa, below which no unrooted
        topology has a neighbour to move to, or ``lazy_top`` is not positive.
    """
    if len(alignment) < 4:
        msg = f"need at least 4 taxa to search, got {len(alignment)}"
        raise ValueError(msg)
    if lazy_top is not None and lazy_top < 1:
        msg = f"lazy_top must be at least 1 when given, got {lazy_top}"
        raise ValueError(msg)

    if topology is not None:
        current = topology
    elif rng is None:
        msg = "searching for a topology needs an rng to draw the start from"
        raise ValueError(msg)
    else:
        current = random_topology(sorted(alignment), rng)
    best = _score(model, current, k, alignment)
    trace = [best.value]
    seen = {leaf_bipartitions(current)}
    neighbourhood = nni_neighbours if moves is MoveSet.NNI else spr_neighbours
    cache = PartialCache()

    evaluations, fits, likelihood_evaluations = 0, 1, best.evaluations
    # A zero budget is a request to fit without searching, and that request
    # is complete as soon as the initial topology is scored. Reporting it as
    # unconverged would describe a search that was never asked for.
    converged = max_evaluations == 0
    while evaluations < max_evaluations:
        warm = best if warm_start else None
        candidate: Topology | None = None
        candidate_fit = best
        fresh: list[Topology] = []
        for neighbour in neighbourhood(current):
            key = leaf_bipartitions(neighbour)
            if key in seen:
                continue
            if evaluations >= max_evaluations:
                # Marked seen only once actually scored: a candidate skipped
                # for want of budget has not been ruled out, and recording it
                # would hide it from a later, larger budget.
                break
            seen.add(key)
            evaluations += 1
            fresh.append(neighbour)
        if lazy_top is None:
            to_fit = fresh
        else:
            ranked = sorted(
                fresh,
                key=lambda neighbour: _lazy_score(
                    model, neighbour, k, alignment, best, cache
                ),
                reverse=True,
            )
            likelihood_evaluations += len(fresh)
            to_fit = ranked[:lazy_top]
        for neighbour in to_fit:
            fitted = _score(model, neighbour, k, alignment, warm)
            fits += 1
            likelihood_evaluations += fitted.evaluations
            if fitted.value > candidate_fit.value:
                candidate, candidate_fit = neighbour, fitted
        if candidate is None:
            converged = True
            break
        current, best = candidate, candidate_fit
        trace.append(best.value)

    return Inference(
        topology=current,
        log_likelihood=best.value,
        parameters=best.parameters,
        evaluations=evaluations,
        trace=tuple(trace),
        converged=converged,
        fits=fits,
        likelihood_evaluations=likelihood_evaluations,
    )


def score_topology(
    topology: Topology,
    alignment: Mapping[str, np.ndarray],
    k: int,
    model: Model = Model.JC,
) -> float:
    """Maximized log-likelihood of one topology, with no search.

    Exposed because exhaustive enumeration needs exactly this and should not
    have to reach into a private helper to get it.

    Parameters
    ----------
    topology : Topology
        The topology to fit.
    alignment : Mapping[str, np.ndarray]
        Observed states per taxon.
    k : int
        Number of states.
    model : Model
        Substitution model for the continuous fit.

    Returns
    -------
    float
        The maximized log-likelihood.
    """
    return _score(model, topology, k, alignment).value
