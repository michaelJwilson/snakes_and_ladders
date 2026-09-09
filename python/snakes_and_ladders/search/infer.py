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

**The same climb over a parsimony score.** :func:`parsimony_search` walks
the same neighbourhoods with :func:`~snakes_and_ladders.likelihood.parsimony.fitch_score`
or :func:`~snakes_and_ladders.likelihood.parsimony.sankoff_score` in place of
the fitted likelihood --- large parsimony (issue #335). There is nothing
continuous to fit, so a candidate costs one pass and no warm start or lazy
rank applies; what is kept is the accounting, a budget in candidates scored
and each topology scored at most once, and the oracle, which is enumeration.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import torch

from snakes_and_ladders.bound import Surrogate
from snakes_and_ladders.likelihood.objective import (
    BranchLengthObjective,
    SubstitutionModelObjective,
)
from snakes_and_ladders.likelihood.parsimony import fitch_score, sankoff_score
from snakes_and_ladders.likelihood.pruning_torch import (
    PartialCache,
    log_likelihood_cached,
)
from snakes_and_ladders.opt.fit import fit
from snakes_and_ladders.opt.objective import Objective
from snakes_and_ladders.search.topology import (
    Topology,
    branch_splits,
    leaf_bipartitions,
    nni_neighbours,
    random_topology,
    spr_neighbours,
)

_log = logging.getLogger(__name__)


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

    def __init__(self, inner: Objective) -> None:
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


class _Restricted:
    """``inner`` on the coordinates ``free``, every other one held at ``base``.

    Partial re-optimization (issue #408): a move changes a handful of
    branches and leaves the rest of the tree alone, so a fit that varies
    every coordinate spends most of its evaluations re-deriving lengths the
    move did not touch. Holding those at the parent's fitted values makes
    the fit ``len(free)``-dimensional, and the value it reaches is a lower
    bound on the full fit's rather than equal to it --- which is why the
    accepted move is refitted in full before it is reported.

    The protocol is satisfied on the reduced vector: ``initial`` is
    ``base`` restricted to ``free``, ``constrain`` scatters back before
    delegating, so a caller reads the same named parameters it always did.
    """

    def __init__(
        self, inner: Objective, base: torch.Tensor, free: torch.Tensor
    ) -> None:
        self.inner = inner
        self._base = base
        self._free = free

    def _full(self, theta: torch.Tensor) -> torch.Tensor:
        return self._base.index_copy(0, self._free, theta)

    def initial(self) -> torch.Tensor:
        return self._base[self._free]

    def constrain(self, theta: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return self.inner.constrain(self._full(theta))

    def theta_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.inner.theta_from(named)[self._free]

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        return self.inner(self._full(theta))


def _disturbed(topology: Topology, warm: _Fitted) -> torch.Tensor:
    """Indices into ``theta`` of the branches the move to ``topology`` created.

    A branch is the split it induces, so a branch the move left alone is one
    whose split the parent also had and whose fitted length therefore
    carries over (:func:`_warm_lengths`). What is left is the path between
    the pruning point and the regraft point, which is where an SPR move
    actually changes the tree, and it is one edge for an NNI.
    """
    return torch.tensor(
        [
            index
            for index, split in enumerate(branch_splits(topology))
            if split not in warm.lengths_by_split
        ],
        dtype=torch.long,
    )


def _objective(
    model: Model, topology: Topology, k: int, alignment: Mapping[str, np.ndarray]
) -> BranchLengthObjective | SubstitutionModelObjective:
    if model is Model.JC:
        return BranchLengthObjective(topology, k, np.full(k, 1.0 / k), alignment)
    return SubstitutionModelObjective(topology, k, alignment)


def _start(
    alignment: Mapping[str, np.ndarray],
    topology: Topology | None,
    rng: np.random.Generator | None,
) -> Topology:
    """The topology a search begins from, drawn from ``rng`` when none is given."""
    if len(alignment) < 4:
        msg = f"need at least 4 taxa to search, got {len(alignment)}"
        raise ValueError(msg)
    if topology is not None:
        return topology
    if rng is None:
        msg = "searching for a topology needs an rng to draw the start from"
        raise ValueError(msg)
    return random_topology(sorted(alignment), rng)


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
    *,
    partial: bool = False,
) -> _Fitted:
    """Fit a candidate, from the parent's lengths when given, and record the cost.

    ``partial`` varies only the branches the move created
    (:func:`_disturbed`), holding every other coordinate at the parent's
    fitted value; it needs ``warm`` to have those values and is ignored
    without one.
    """
    objective = _objective(model, topology, k, alignment)
    theta0 = None if warm is None else _warm_theta(objective, topology, warm)
    scored: Objective = objective
    if partial and warm is not None and theta0 is not None:
        free = _disturbed(topology, warm)
        if free.numel():
            scored = _Restricted(objective, theta0, free)
            theta0 = scored.initial()
    counting = _Counting(scored)
    result = fit(counting, theta0=theta0)
    named = {
        name: value.detach() for name, value in scored.constrain(result.theta).items()
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


def _neighbourhood(
    moves: MoveSet, radius: int | None
) -> Callable[[Topology], Iterator[Topology]]:
    """The move generator, with the SPR radius bound applied when asked.

    Raises
    ------
    ValueError
        If ``radius`` is given with ``MoveSet.NNI``, whose neighbourhood is
        the internal edges and has no pruning point to measure from ---
        silently ignoring it would report a bounded search that was not one.
    """
    if moves is MoveSet.NNI:
        if radius is not None:
            msg = "radius bounds an SPR regraft; MoveSet.NNI has no pruning point"
            raise ValueError(msg)
        return nni_neighbours

    def bounded(topology: Topology) -> Iterator[Topology]:
        return spr_neighbours(topology, radius=radius)

    return bounded


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
    surrogate: Surrogate | None = None,
    radius: int | None = None,
    partial_reoptimization: bool = False,
) -> Inference:
    """Hill-climb over topologies, fitting continuous parameters per candidate.

    Parameters
    ----------
    alignment : Mapping[str, np.ndarray]
        Observed states per taxon, each of shape ``(n_sites,)``.
    k : int
        Number of states.
    topology : Topology | None
        Where to start. ``None`` draws a random topology from ``rng``. A
        parsimony start is this argument and not a mode of its own:
        ``parsimony_search(alignment, k, rng=rng).topology`` is the tree the
        Fitch climb reaches, and it costs one post-order pass per candidate
        against this loop's fit. What it buys is measured in
        ``tests/benchmarks/test_search_infer_bench.py``.
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
    surrogate : Surrogate | None
        What ranks the neighbourhood when ``lazy_top`` is given: ``None``
        ranks by the one-evaluation lazy score, a surrogate ranks by its own
        value (issue #308), and a bound or a learned predictor from
        ``likelihood.surrogate`` or ``search.surrogate`` fits here. Its
        evaluations are not likelihood evaluations and are not counted as
        such; the fits it saves or costs are what ``fits`` reports.
    radius : int | None
        Bound an SPR regraft to within ``radius`` of the pruning point
        (:func:`~snakes_and_ladders.search.topology.spr_neighbours`), which
        makes the neighbourhood ``O(n * radius)`` rather than ``O(n ** 2)``.
        ``None`` is unbounded, and so is any radius from the leaf count up:
        the two are the same search, candidate for candidate. Rejected with
        ``MoveSet.NNI``, which has no pruning point.
    partial_reoptimization : bool
        Fit a candidate over only the branches the move created, holding
        every other length at the parent's fitted value (:class:`_Restricted`,
        issue #408, extending the warm starts of issue #289). A partial fit
        reaches a value no higher than the full fit's, so it screens
        candidates rather than scoring them: the accepted move is refitted in
        full, and ``log_likelihood`` is a full fit's whatever this is set to.
        Which candidate wins can change, so it is opt-in, and what it costs
        in missed optima is measured rather than assumed zero. Needs
        ``warm_start``, which is where the lengths it holds fixed come from.

    Returns
    -------
    Inference
        The best topology found and its fitted parameters.

    Raises
    ------
    ValueError
        If the alignment has fewer than 4 taxa, below which no unrooted
        topology has a neighbour to move to, ``lazy_top`` is not positive,
        a surrogate is given without ``lazy_top`` to apply it to, ``radius``
        is given with ``MoveSet.NNI`` or is below 1, or
        ``partial_reoptimization`` is set without ``warm_start``.
    """
    if lazy_top is not None and lazy_top < 1:
        msg = f"lazy_top must be at least 1 when given, got {lazy_top}"
        raise ValueError(msg)
    if surrogate is not None and lazy_top is None:
        msg = "a surrogate ranks the candidates lazy_top selects; give both"
        raise ValueError(msg)
    if partial_reoptimization and not warm_start:
        msg = (
            "partial_reoptimization holds the parent's fitted lengths on the "
            "branches the move kept; warm_start is where they come from"
        )
        raise ValueError(msg)

    neighbourhood = _neighbourhood(moves, radius)
    current = _start(alignment, topology, rng)
    best = _score(model, current, k, alignment)
    trace = [best.value]
    seen = {leaf_bipartitions(current)}
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
        elif surrogate is not None:
            ranked = sorted(
                fresh,
                key=lambda neighbour: float(surrogate(neighbour, alignment)),
                reverse=True,
            )
            to_fit = ranked[:lazy_top]
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
            fitted = _score(
                model, neighbour, k, alignment, warm, partial=partial_reoptimization
            )
            fits += 1
            likelihood_evaluations += fitted.evaluations
            if fitted.value > candidate_fit.value:
                candidate, candidate_fit = neighbour, fitted
        if candidate is None:
            converged = True
            break
        if partial_reoptimization:
            # A partial fit is a lower bound on the full one, so the winner is
            # refitted over every branch before it is accepted: the reported
            # log-likelihood, and the trace, stay maximized values.
            candidate_fit = _score(model, candidate, k, alignment, candidate_fit)
            fits += 1
            likelihood_evaluations += candidate_fit.evaluations
        current, best = candidate, candidate_fit
        trace.append(best.value)
        _log.debug(
            "accepted move %d: log-likelihood %.6f after %d evaluations",
            len(trace) - 1,
            best.value,
            evaluations,
        )

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


@dataclass(frozen=True)
class ParsimonyInference:
    """The outcome of a large-parsimony search.

    Parameters
    ----------
    topology : Topology
        The most parsimonious topology found.
    score : float
        Its parsimony score: an integer-valued float under the unit step
        matrix, where it is the Fitch score.
    evaluations : int
        Candidates scored, which is what the budget counts. Each is one
        post-order pass; there is no fit to count beside it.
    trace : tuple[float, ...]
        Score after each accepted move, starting with the initial
        topology's, non-increasing by construction.
    converged : bool
        Whether the search stopped because no neighbour improved rather than
        because the budget ran out, on the same terms as :class:`Inference`.
    """

    topology: Topology
    score: float
    evaluations: int
    trace: tuple[float, ...]
    converged: bool


def _metric_step_matrix(step_matrix: np.ndarray, k: int) -> np.ndarray:
    """``step_matrix`` if the unrooted score is well defined under it, else raise.

    A search walks unrooted topologies and keys them on their bipartitions,
    so every rooting of a candidate must score the same. That holds when the
    matrix is a metric: symmetric with a zero diagonal, so an edge costs the
    same read either way, and satisfying the triangle inequality, so a
    degree-2 root cannot be labelled with an intermediate state cheaper than
    the direct change on the edge it splits.
    """
    step = np.asarray(step_matrix, dtype=np.float64)
    if step.shape != (k, k):
        msg = f"step_matrix must have shape {(k, k)}, got {step.shape}"
        raise ValueError(msg)
    if not np.array_equal(step, step.T) or bool(np.any(np.diag(step) != 0.0)):
        msg = (
            "step_matrix must be symmetric with a zero diagonal: an unrooted "
            "topology has no direction to read an asymmetric cost along"
        )
        raise ValueError(msg)
    via = np.min(step[:, :, None] + step[None, :, :], axis=1)
    if bool(np.any(step > via + 1e-12)):
        msg = (
            "step_matrix must satisfy the triangle inequality, or a rooting "
            "changes the score of the same unrooted topology"
        )
        raise ValueError(msg)
    return step


def parsimony_search(
    alignment: Mapping[str, np.ndarray],
    k: int,
    *,
    step_matrix: np.ndarray | None = None,
    topology: Topology | None = None,
    moves: MoveSet = MoveSet.NNI,
    max_evaluations: int = 200,
    rng: np.random.Generator | None = None,
) -> ParsimonyInference:
    """Hill-climb over topologies on the parsimony score: large parsimony (``eq:large-parsimony``).

    The loop of :func:`infer` with the fit replaced by one parsimony pass, and
    the same accounting: a budget in candidates scored, each topology scored
    at most once, keyed on its bipartitions, and a converged flag that means
    no neighbour improved. Below eight taxa
    :func:`~snakes_and_ladders.search.topology.enumerate_topologies` referees it,
    which is how the regression suite pins it.

    Parameters
    ----------
    alignment : Mapping[str, np.ndarray]
        Observed states per taxon, each of shape ``(n_sites,)``.
    k : int
        Number of states.
    step_matrix : np.ndarray | None
        ``None`` scores by :func:`~snakes_and_ladders.likelihood.parsimony.fitch_score`.
        A ``(k, k)`` matrix scores by
        :func:`~snakes_and_ladders.likelihood.parsimony.sankoff_score`, and must
        be a metric --- symmetric, zero on the diagonal, and satisfying the
        triangle inequality --- because an unrooted topology has one score
        only when every rooting of it scores the same.
    topology : Topology | None
        Where to start. ``None`` draws a random topology from ``rng``.
    moves : MoveSet
        Neighbourhood the search proposes from.
    max_evaluations : int
        Maximum candidates scored. The initial topology's own score is not
        counted against it.
    rng : np.random.Generator | None
        Source of the starting topology, required when ``topology`` is
        ``None`` and unused otherwise, on the terms :func:`infer` states.

    Returns
    -------
    ParsimonyInference
        The most parsimonious topology found and its score.

    Raises
    ------
    ValueError
        If the alignment has fewer than 4 taxa, if ``topology`` is ``None``
        and no ``rng`` is given, or if ``step_matrix`` is not a metric of
        shape ``(k, k)``.
    """
    if step_matrix is None:

        def score(candidate: Topology) -> float:
            return float(fitch_score(candidate, alignment, k))

    else:
        step = _metric_step_matrix(step_matrix, k)

        def score(candidate: Topology) -> float:
            return sankoff_score(candidate, alignment, step)

    current = _start(alignment, topology, rng)
    best = score(current)
    trace = [best]
    seen = {leaf_bipartitions(current)}
    neighbourhood = nni_neighbours if moves is MoveSet.NNI else spr_neighbours

    evaluations = 0
    converged = max_evaluations == 0
    while evaluations < max_evaluations:
        candidate: Topology | None = None
        candidate_score = best
        for neighbour in neighbourhood(current):
            key = leaf_bipartitions(neighbour)
            if key in seen:
                continue
            if evaluations >= max_evaluations:
                break
            seen.add(key)
            evaluations += 1
            value = score(neighbour)
            if value < candidate_score:
                candidate, candidate_score = neighbour, value
        if candidate is None:
            converged = True
            break
        current, best = candidate, candidate_score
        trace.append(best)
        _log.debug(
            "accepted move %d: parsimony score %g after %d evaluations",
            len(trace) - 1,
            best,
            evaluations,
        )

    return ParsimonyInference(
        topology=current,
        score=best,
        evaluations=evaluations,
        trace=tuple(trace),
        converged=converged,
    )
