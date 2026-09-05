"""Topology search: the outer loop that the optimizer deliberately excludes.

`opt/CLAUDE.md` records the seam this module is the first user of. A discrete
move changes the *structure* being fitted, so it changes what the parameter
vector means and how long it is; it cannot be a step inside a fit over a
fixed-length vector. It constructs a **new** objective, and something outside
the optimizer has to own that construction. This is that something.

The consequence is that nothing here needed to change `snakes_and_ladders.opt`. The same
``fit`` that fits a Potts chain scores every candidate topology, which is the
claim issue #63 made and could not itself test.

**Budget is counted in candidate fits, not in seconds.** ``DEV.md`` forbids
ranking performance on CI hardware, and a wall-clock budget would make a
result depend on the machine that produced it, so a run would not be
reproducible from its seed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from snakes_and_ladders.likelihood.objective import (
    BranchLengthObjective,
    SubstitutionModelObjective,
)
from snakes_and_ladders.opt.fit import fit
from snakes_and_ladders.search.topology import (
    Topology,
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
        Its maximized log-likelihood.
    parameters : Mapping[str, np.ndarray]
        The fitted continuous parameters, under the names the model uses.
    evaluations : int
        Candidate fits performed, which is what the budget counts.
    trace : tuple[float, ...]
        Log-likelihood after each accepted move, starting with the initial
        topology's. Its length minus one is the number of moves taken.
    converged : bool
        Whether the search stopped because no neighbour improved, rather
        than because the budget ran out. A search that ran out of budget has
        not finished, and reporting its result as an optimum would be wrong.
        A zero budget --- a fit with no search --- reports ``True``, since
        there was no search to leave unfinished.
    """

    topology: Topology
    log_likelihood: float
    parameters: Mapping[str, np.ndarray]
    evaluations: int
    trace: tuple[float, ...]
    converged: bool


def _objective(
    model: Model, topology: Topology, k: int, alignment: Mapping[str, np.ndarray]
) -> BranchLengthObjective | SubstitutionModelObjective:
    if model is Model.JC:
        return BranchLengthObjective(topology, k, np.full(k, 1.0 / k), alignment)
    return SubstitutionModelObjective(topology, k, alignment)


def _score(
    model: Model, topology: Topology, k: int, alignment: Mapping[str, np.ndarray]
) -> tuple[float, Mapping[str, np.ndarray]]:
    """Fit a candidate and return its log-likelihood and fitted parameters."""
    objective = _objective(model, topology, k, alignment)
    result = fit(objective)
    parameters = {
        name: value.detach().numpy()
        for name, value in objective.constrain(result.theta).items()
    }
    return -result.value, parameters


def infer(
    alignment: Mapping[str, np.ndarray],
    k: int,
    topology: Topology | None = None,
    model: Model = Model.JC,
    moves: MoveSet = MoveSet.NNI,
    max_evaluations: int = 200,
    seed: int = 0,
) -> Inference:
    """Fit an alignment, searching for the topology unless one is given.

    With ``topology`` supplied the topology is held fixed and this reduces to
    the continuous fit, which is the same call with the search switched off
    rather than a separate code path.

    Hill climbing: score the current topology, score every neighbour, move to
    the best strict improvement, stop when none improves or the budget is
    spent. Topologies already scored are skipped --- an SPR neighbourhood
    overlaps its predecessor heavily, and refitting a topology already fitted
    is the dominant avoidable cost.

    Parameters
    ----------
    alignment : Mapping[str, np.ndarray]
        Observed states per taxon, each of shape ``(n_sites,)``.
    k : int
        Number of states.
    topology : Topology | None
        Starting topology; ``None`` draws one at random from ``seed``.
        Supplying one and leaving the budget at zero fits it without moving.
    model : Model
        Substitution model for the continuous fit.
    moves : MoveSet
        Neighbourhood to propose from.
    max_evaluations : int
        Maximum candidate fits. The initial topology's own fit is not
        counted against it.
    seed : int
        Seed for the starting topology.

    Returns
    -------
    Inference
        The best topology found and its fitted parameters.

    Raises
    ------
    ValueError
        If the alignment has fewer than 4 taxa, below which no unrooted
        topology has a neighbour to move to.
    """
    if len(alignment) < 4:
        msg = f"need at least 4 taxa to search, got {len(alignment)}"
        raise ValueError(msg)

    current = (
        random_topology(sorted(alignment), np.random.default_rng(seed))
        if topology is None
        else topology
    )
    best_value, best_parameters = _score(model, current, k, alignment)
    trace = [best_value]
    seen = {leaf_bipartitions(current)}
    neighbourhood = nni_neighbours if moves is MoveSet.NNI else spr_neighbours

    evaluations = 0
    # A zero budget is a request to fit without searching, and that request
    # is complete as soon as the initial topology is scored. Reporting it as
    # unconverged would describe a search that was never asked for.
    converged = max_evaluations == 0
    while evaluations < max_evaluations:
        candidate: Topology | None = None
        candidate_value = best_value
        candidate_parameters = best_parameters
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
            value, parameters = _score(model, neighbour, k, alignment)
            evaluations += 1
            if value > candidate_value:
                candidate, candidate_value = neighbour, value
                candidate_parameters = parameters
        if candidate is None:
            converged = True
            break
        current, best_value, best_parameters = (
            candidate,
            candidate_value,
            candidate_parameters,
        )
        trace.append(best_value)

    return Inference(
        topology=current,
        log_likelihood=best_value,
        parameters=best_parameters,
        evaluations=evaluations,
        trace=tuple(trace),
        converged=converged,
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
    value, _ = _score(model, topology, k, alignment)
    return value
