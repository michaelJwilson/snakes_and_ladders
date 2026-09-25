"""Fitting the coupled spatio-sequential model: block ascent, three label solvers, and the annealed start (issue #306).

The block structure of ``eq:joint`` makes maximum likelihood two alternating
problems. Given the labels, each class is a hidden Markov chain over its
members' observations and expectation--maximization fits its emissions, its
initial distribution and the shared self-transition -- the E step is
:func:`~sal.likelihood.spatio_sequential.class_posteriors`
and the M step is each family's own ``reestimate`` (the M-step identity in the textbook).
Given the parameters, the posterior defines a per-node, per-class external
field ``H`` (the external-field equation of the textbook) and the labels are the ground state of a
Potts model in that field (the label ground-state equation of the textbook): exactly the problem
:func:`~sal.search.alpha_expansion.alpha_expansion` solves
with a bound, :func:`~sal.search.icm.iterated_conditional_modes`
descends, and a Wolff cluster move in a field samples.

Every block is held to one property: ``log p(x, l | theta)`` with the chains
marginalized does not decrease. The two exact solvers give that by
construction; the annealed Wolff move is a *proposer* here, its best visited
labelling taken only when it improves the joint (open question 1 of #306).
Whether a cluster move escapes what single-site descent freezes into is a
measurement, made past enumeration on a lattice with planted labels.

The annealed start, ``Graph_BurnIn++`` (the textbook's burn-in algorithm), runs the
same blocks while the inverse temperature rises from zero, so the labels are
nearly free while the chains and emissions are fitted to what the data alone
supports, and the spatial prior tightens as the classes separate. Its
seeding, ``Emission_Mixture++``, is k-means++ under the family's own
Bregman divergence, in :mod:`sal.opt.mixture`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment

from sal.backend import Backend
from sal.emissions import (
    BetaBinomialEmission,
    CategoricalEmission,
    EmissionFamily,
    GaussianEmission,
    NegativeBinomialEmission,
)
from sal.likelihood.forward_backward import sample_path
from sal.likelihood.spatio_sequential import (
    ClassPosteriors,
    class_log_density,
    class_posteriors,
    covariate_columns,
    external_field,
    labelled_log_likelihood,
)
from sal.opt.emission_mixture import plus_plus_start
from sal.opt.mixture import emission_mixture_plus_plus
from sal.opt.termination import Termination
from sal.sample.potts_mcmc import adjacency_lists, wolff_sweep
from sal.sample.schedule import TempSchedule
from sal.search.alpha_expansion import alpha_expansion
from sal.search.icm import SweepOrder, iterated_conditional_modes
from sal.sim.count_pairs import (
    IndependentCountPair,
    IndependentCountPairSeeding,
    rate_space,
)
from sal.sim.potts import SiteField, energy
from sal.sim.spatio_sequential import SpatioSequentialParams
from sal.track import current as current_tracked

if TYPE_CHECKING:
    from sal.likelihood.spatio_sequential_rust import CovariateRows, ObservationRows


class LabelSolver(StrEnum):
    """How the label block is solved."""

    ALPHA_EXPANSION = "alpha_expansion"
    """Exact two-state expansions cycled to a local minimum within the bound."""

    ICM = "icm"
    """Single-site descent to a local minimum."""

    WOLFF = "wolff"
    """Cluster moves in the field on an annealing schedule, best labelling kept."""


@dataclass(frozen=True)
class SpatioSequentialFit:
    """What block ascent returns.

    Parameters
    ----------
    params : SpatioSequentialParams
        The fitted truth: emissions, initial distributions and self-transition.
    labels : np.ndarray
        The fitted labels, shape ``(n_nodes,)``.
    log_likelihoods : np.ndarray
        ``log p(x, l | theta)`` after every block, shape ``(2 * n_blocks + 1,)``:
        the start, then after each EM half and each label half. Non-decreasing.
    field : np.ndarray
        The external field at the fitted labels and parameters, ``(n_nodes, M)``.
    termination : Termination | None
        The block loop's: every block runs, so it ends on the count it was
        given rather than on a criterion (issue #860).
    """

    params: SpatioSequentialParams
    labels: np.ndarray
    log_likelihoods: np.ndarray
    field: np.ndarray
    termination: Termination | None = None


def m_step(
    params: SpatioSequentialParams,
    observations: np.ndarray,
    labels: np.ndarray,
    posteriors: ClassPosteriors,
) -> SpatioSequentialParams:
    """Re-estimate every class's emissions, ``Pi_m`` and the shared ``t`` from the E step.

    A class with no members keeps its emissions: there is nothing to
    re-estimate them from, and its chain posterior is its prior. Under
    ``params.shared_emissions`` one family is re-estimated on every class's
    block, and every class takes it. A per-step
    ``self_transition`` is kept too, for the reason stated at the assignment.

    ``params.covariate`` is selected by the same members and moved by the same
    axis as the observations (issue #652). That pairing is the one place this
    module can go quietly wrong: a covariate sliced differently from the block
    it accompanies is a fit that conditions on the wrong exposures and
    converges anyway, which is why the slice is
    :func:`~sal.likelihood.spatio_sequential.covariate_columns`'s,
    the slice ``covariate_block`` wraps, and not this module's own (issue
    #670).
    """
    labels = np.asarray(labels, dtype=np.int64)
    # Arrays throughout: an M step takes no derivative, and each family's
    # `reestimate` converts what it is handed, the observations in its own
    # `observation_dtype` (issue #1011).
    blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]] = []
    for m in range(len(params.emissions)):
        members = np.flatnonzero(labels == m)
        if members.size == 0:
            blocks.append((np.empty(0), np.empty(0), None))
            continue
        block = np.moveaxis(
            observations[:, members], 1, 0
        )  # (n_m, S), plus any channel axes the family's observation carries
        block_covariate = covariate_columns(params, members)
        # `covariate_columns` returns the block position-major, as the
        # observations are held; the family wants it member-major, as the block
        # above is. Moving the axis is all this does -- appending the singleton
        # here was the defect of issue #670, since a covariate carrying the
        # family's own channel axis already has one inside each channel.
        exposure = (
            None if block_covariate is None else np.moveaxis(block_covariate, 1, 0)
        )  # (n_m, S, 1), or (n_m, S, ...) for a family with axes of its own
        weights = np.repeat(
            posteriors.posterior[m][None], members.size, axis=0
        )  # (n_m, S, K)
        blocks.append((block, weights, exposure))
    emissions: list[EmissionFamily]
    if params.shared_emissions:
        # One family for every class (issue #933): its M step is the family's
        # own on the classes' blocks stacked along the member axis, which is
        # the sum over classes of each class's expected log-likelihood.
        filled = [one for one in blocks if one[0].size > 0]
        if not filled:
            emissions = list(params.emissions)
        else:
            covariates = [one[2] for one in filled]
            pooled = (
                params.emissions[0]
                .reestimate(
                    np.concatenate([one[0] for one in filled]),
                    np.concatenate([one[1] for one in filled]),
                    covariate=None
                    if covariates[0] is None
                    else np.concatenate([c for c in covariates if c is not None]),
                )
                .emissions
            )
            emissions = [pooled] * params.n_classes
    else:
        emissions = [
            family
            if block.size == 0
            else family.reestimate(block, weights, covariate=exposure).emissions
            for family, (block, weights, exposure) in zip(
                params.emissions, blocks, strict=True
            )
        ]
    initial = np.maximum(posteriors.posterior[:, 0, :], 1e-12)
    initial = initial / initial.sum(axis=1, keepdims=True)
    self_transition: float | np.ndarray = params.self_transition
    # A per-step rate is conditioned on, not fitted (issue #658), the same
    # standing `baum_welch_family` gives a per-step kernel and for the same
    # reason: it carries one free value per transition against the transitions
    # of `M` chains, and pooling them into the scalar below would answer a
    # question nobody asked -- what single stickiness best explains a chain
    # whose stickiness the caller said varies.
    if params.n_positions > 1 and np.asarray(params.self_transition).ndim == 0:
        stays = sum(
            float(np.trace(posteriors.pairwise[m, s]))
            for m in range(params.n_classes)
            for s in range(params.n_positions - 1)
        )
        total = params.n_classes * (params.n_positions - 1)
        self_transition = min(max(stays / total, 1e-6), 1 - 1e-6)
    return replace(
        params,
        initial=initial,
        self_transition=self_transition,
        emissions=tuple(emissions),
    )


def label_step(
    params: SpatioSequentialParams,
    observations: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    solver: LabelSolver,
    *,
    field: np.ndarray | None = None,
    wolff_schedule: TempSchedule | None = None,
    wolff_moves_per_step: int = 4,
    min_sites: int = 0,
) -> np.ndarray:
    """One label block: the ground state of the Potts model in the field, by ``solver``.

    The energy minimized is ``-beta sum J delta(l, l') + sum_n H[n, l_n]``,
    so the solvers see couplings ``beta * J`` and a per-node field ``-H``.
    The two exact solvers start from ``labels``; Wolff anneals from them on
    ``wolff_schedule`` (temperatures multiply ``beta``'s inverse) and returns
    the lowest-energy labelling it visited.

    ``min_sites`` is the ICM descent's floor
    (:func:`~sal.search.icm.iterated_conditional_modes`,
    issue #1055): after each sweep a class holding fewer sites is dissolved
    into those at or above it. ``0``, the default, is the step before the
    floor existed, bitwise; the other solvers have no floor and refuse one.
    """
    if min_sites != 0 and solver is not LabelSolver.ICM:
        msg = (
            f"the {solver} label step has no min_sites floor; it applies to "
            f"{LabelSolver.ICM}"
        )
        raise ValueError(msg)
    labels = np.asarray(labels, dtype=np.int64).copy()
    if field is None:
        field = external_field(params, observations, labels)
    graph = params.scaled_graph()
    potential = -field
    if solver is LabelSolver.ALPHA_EXPANSION:
        return np.asarray(
            alpha_expansion(graph, potential, params.n_classes, start=labels).labelling
        )
    if solver is LabelSolver.ICM:
        # The sweep `search.icm` runs, started from `labels` in a random site
        # order (issue #858). It reads the same argmin off local deltas rather
        # than off a full energy per candidate, which is `O(k * degree)` per
        # site against `O(k * n_edges)`; the labelling is pinned against the
        # recomputing loop this replaced. The Python sweep, since a random
        # order that stops on a clean sweep draws each permutation as it
        # starts, which the compiled sweep does not run.
        return iterated_conditional_modes(
            graph,
            potential,
            params.n_classes,
            rng,
            start=labels,
            sweep_order=SweepOrder.RANDOM,
            min_sites=min_sites,
            backend=Backend.PYTHON,
        ).labelling
    if wolff_schedule is None:
        msg = "the Wolff solver needs a schedule"
        raise ValueError(msg)
    best_labels = labels.copy()
    best_value = energy(graph, potential, labels)
    # The field declared as the energy it is: the move reads -H (#921).
    declared = SiteField.from_energy(field)
    offsets, neighbours, couplings = params.graph.compressed_adjacency()
    lists = adjacency_lists(offsets, neighbours, couplings)
    for step in range(wolff_schedule.n_steps):
        temperature = wolff_schedule(step)
        for _ in range(wolff_moves_per_step):
            wolff_sweep(
                labels,
                declared,
                offsets,
                neighbours,
                couplings,
                rng,
                beta=params.beta / temperature,
                lists=lists,
            )
        value = energy(graph, potential, labels)
        if value < best_value:
            best_value, best_labels = value, labels.copy()
    return best_labels


def redraw_small_labels(
    labels: np.ndarray,
    n_classes: int,
    min_label_sites: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Every site of a class holding fewer than ``min_label_sites`` sites, relabelled uniformly (issue #933, R11).

    A class the label step leaves with a handful of sites fits its emissions
    to noise. Each site of a class holding at least one and fewer than
    ``min_label_sites`` sites takes a label drawn uniformly from the classes
    that are not so small, empty ones included, from ``rng``. At
    ``min_label_sites <= 0``, or with no class that small, the labels are
    returned as given and ``rng`` is not drawn from, so the default leaves a
    fit bitwise what it was. Where every occupied class is that small there
    is nowhere to move a site to, and the labels are returned as given.

    Returns
    -------
    np.ndarray
    """
    if min_label_sites <= 0:
        return labels
    counts = np.bincount(labels, minlength=n_classes)
    small = (counts > 0) & (counts < min_label_sites)
    if not bool(small.any()):
        return labels
    kept = np.flatnonzero(~small)
    if kept.size == 0:
        return labels
    moved = small[labels]
    redrawn = labels.copy()
    redrawn[moved] = kept[rng.integers(0, kept.size, size=int(moved.sum()))]
    return redrawn


def fit_spatio_sequential(
    params: SpatioSequentialParams,
    observations: np.ndarray,
    rng: np.random.Generator,
    *,
    solver: LabelSolver = LabelSolver.ALPHA_EXPANSION,
    n_blocks: int = 10,
    labels: np.ndarray | None = None,
    fit_parameters: bool = True,
    wolff_schedule: TempSchedule | None = None,
    backend: Backend = Backend.PYTHON,
    min_label_sites: int = 0,
    covariate_rows: CovariateRows = "range",
) -> SpatioSequentialFit:
    """Block-coordinate ascent on ``log p(x, l | theta)``.

    Inside a :func:`sal.track.track` block it records
    ``log_likelihood`` once per block, and ``objective`` as its negative: the
    entries of :attr:`SpatioSequentialFit.log_likelihoods` the blocks end on,
    so the series is that field's history and not a second definition of it
    (issue #887).

    Parameters
    ----------
    params : SpatioSequentialParams
        The starting parameters; the fitted ones replace the emissions,
        ``initial`` and ``self_transition``.
    observations : np.ndarray
        Shape ``(S, n_nodes)``.
    rng : np.random.Generator
        Draws the starting labels when none are given, and the Wolff moves.
    solver : LabelSolver
        The label block.
    n_blocks : int
        EM-then-label rounds; at least one.
    labels : np.ndarray | None
        Starting labels; ``None`` draws them uniformly.
    fit_parameters : bool
        ``False`` holds ``params`` fixed and runs the label block alone, the
        setting in which the label step is pinned against enumeration.
    wolff_schedule : TempSchedule | None
        Required by the Wolff solver.
    backend : Backend
        Which kernel runs the E step, the field and the labelled log-likelihood:
        :data:`~sal.backend.Backend.PYTHON` is the NumPy oracle
        and :data:`~sal.backend.Backend.RUST` the tabulated
        kernel of :mod:`sal.likelihood.spatio_sequential_rust`,
        chosen inside :mod:`sal.likelihood.spatio_sequential`
        so the three cannot be mixed (#828).
    min_label_sites : int
        At the start of every block, the sites of a class holding fewer than
        this many are relabelled uniformly from the other classes
        (:func:`redraw_small_labels`), so the block's M step fits no class to
        a handful of sites. ``0``, the default, redraws nothing and draws
        nothing from ``rng``.
    covariate_rows : CovariateRows
        The Rust backend's layout for the trial count's table
        (:func:`sal.likelihood.spatio_sequential_rust.observation_rows`, issue
        #1064). The rows depend on the observations and the covariate alone,
        so on the Rust backend they are built once here and every E step, field
        and labelled log-likelihood of the fit reuses them. ``"range"``, the
        default, is the only value the NumPy backend takes: it builds no table.

    Raises
    ------
    ValueError
        If ``n_blocks < 1``, a re-estimated family did not converge, or
        ``covariate_rows`` is not the default on the NumPy backend.
    """
    if n_blocks < 1:
        msg = f"at least one block, got {n_blocks}"
        raise ValueError(msg)
    n_nodes = params.graph.n_nodes
    current = (
        rng.integers(0, params.n_classes, size=n_nodes)
        if labels is None
        else np.asarray(labels, dtype=np.int64).copy()
    )
    rows: CovariateRows | ObservationRows = covariate_rows
    if backend is Backend.RUST:
        # Imported here, as `sal.backend.twin` imports it, so the NumPy
        # backend never loads the extension.
        from sal.likelihood.spatio_sequential_rust import observation_rows

        rows = observation_rows(
            observations, params.covariate, covariate_rows=covariate_rows
        )
    posteriors_of = partial(class_posteriors, backend=backend, covariate_rows=rows)
    field_of = partial(external_field, backend=backend, covariate_rows=rows)
    log_likelihood_of = partial(
        labelled_log_likelihood, backend=backend, covariate_rows=rows
    )
    values = [log_likelihood_of(params, observations, current)]
    tracked = current_tracked()
    tracked.record(0, objective=-values[0], log_likelihood=values[0])
    for block in range(n_blocks):
        current = redraw_small_labels(current, params.n_classes, min_label_sites, rng)
        posteriors = posteriors_of(params, observations, current)
        if fit_parameters:
            params = m_step(params, observations, current, posteriors)
            posteriors = posteriors_of(params, observations, current)
        values.append(log_likelihood_of(params, observations, current))
        field = field_of(params, observations, current, posteriors.posterior)
        proposed = label_step(
            params,
            observations,
            current,
            rng,
            solver,
            field=field,
            wolff_schedule=wolff_schedule,
        )
        candidate = log_likelihood_of(params, observations, proposed)
        if candidate >= values[-1]:
            current = proposed
            values.append(candidate)
        else:
            values.append(values[-1])
        tracked.record(block + 1, objective=-values[-1], log_likelihood=values[-1])
    tracked.record_cost(n_blocks, current.nbytes)
    field = field_of(params, observations, current, None)
    # Every block runs: the loop tests nothing and ends on the count it was
    # given (issue #860).
    return SpatioSequentialFit(
        params,
        current,
        np.array(values),
        field,
        Termination.after(n_blocks, converged=False),
    )


@dataclass(frozen=True)
class Merge:
    """What one :func:`merge_step` tried and what it kept.

    Parameters
    ----------
    params : SpatioSequentialParams
        The parameters after the kept merge, or the input where none was.
    labels : np.ndarray
        The labels after the kept merge, or the input.
    criterion : float
        :func:`~sal.likelihood.spatio_sequential.labelled_log_likelihood`
        less ``penalty`` per occupied class, at what was kept.
    merged : tuple[int, int] | None
        ``(kept, emptied)``: the class that absorbed the other, or ``None``.
    tried : tuple[tuple[int, int, float], ...]
        Every pair tried and its criterion, in the order tried.
    """

    params: SpatioSequentialParams
    labels: np.ndarray
    criterion: float
    merged: tuple[int, int] | None
    tried: tuple[tuple[int, int, float], ...]


def merge_step(
    params: SpatioSequentialParams,
    observations: np.ndarray,
    labels: np.ndarray,
    *,
    penalty: float = 0.0,
) -> Merge:
    """Merge the pair of classes that most raises the labelled joint, if any does (issue #933).

    No other move lowers the number of classes a labelling occupies: the
    label solvers keep every class they are given, and ``opt.split_merge``
    acts on mixture components. For every pair ``a < b`` of occupied
    classes, ``b``'s sites take ``a`` and one :func:`m_step` refits the
    parameters on the merged labels; the criterion is the labelled joint
    less ``penalty`` per occupied class. The best merge is kept when its
    criterion is at least the input's, so a block ascent that interleaves
    this step stays non-decreasing in the criterion. The emptied class keeps
    its parameters, as :func:`m_step` keeps an empty class's, and a later
    label step may repopulate it.

    Parameters
    ----------
    params : SpatioSequentialParams
        The current parameters.
    observations : np.ndarray
        ``(S, n_nodes, ...)``.
    labels : np.ndarray
        The current class per node.
    penalty : float
        The criterion's cost per occupied class, non-negative; ``0``, the
        default, merges only where the joint itself does not fall.

    Returns
    -------
    Merge

    Raises
    ------
    ValueError
        If ``penalty`` is negative.
    """
    if penalty < 0.0:
        msg = f"a class penalty is non-negative, got {penalty}"
        raise ValueError(msg)
    labels = np.asarray(labels, dtype=np.int64)

    def criterion(candidate: SpatioSequentialParams, labelled: np.ndarray) -> float:
        occupied = int(np.unique(labelled).size)
        return (
            labelled_log_likelihood(candidate, observations, labelled)
            - penalty * occupied
        )

    current = criterion(params, labels)
    best: Merge | None = None
    tried: list[tuple[int, int, float]] = []
    occupied = [int(m) for m in np.unique(labels)]
    for index, kept in enumerate(occupied):
        for emptied in occupied[index + 1 :]:
            merged = np.where(labels == emptied, kept, labels)
            refit = m_step(
                params,
                observations,
                merged,
                class_posteriors(params, observations, merged),
            )
            value = criterion(refit, merged)
            tried.append((kept, emptied, value))
            if value >= current and (best is None or value > best.criterion):
                best = Merge(refit, merged, value, (kept, emptied), ())
    if best is None:
        return Merge(params, labels, current, None, tuple(tried))
    return replace(best, tried=tuple(tried))


def label_accuracy(fitted: np.ndarray, planted: np.ndarray, n_classes: int) -> float:
    """The fraction of nodes labelled as planted, up to the best permutation of classes.

    A class label is a name and not a quantity, so a labelling is right when
    some renaming of it is: the comparison is over permutations of the ``M``
    names. Searching them is a linear assignment on the contingency table of
    fitted against planted counts --- ``M!`` is 3.6 million at the ``M = 10``
    the declared 5,041-vertex instance carries (issue #399), and the
    assignment is cubic --- so `scipy.optimize.linear_sum_assignment` finds
    the best renaming and the answer is the same one enumerating the
    permutations gives.

    Parameters
    ----------
    fitted, planted : np.ndarray
        One class per node, entries in ``[0, n_classes)``.
    n_classes : int
        ``M``.

    Returns
    -------
    float
        The fraction agreeing under the best renaming, in ``[0, 1]``.
    """
    fitted = np.asarray(fitted)
    planted = np.asarray(planted)
    agreements = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(agreements, (fitted, planted), 1)
    rows, columns = linear_sum_assignment(agreements, maximize=True)
    return float(agreements[rows, columns].sum()) / float(fitted.size)


def seed_emissions(
    params: SpatioSequentialParams,
    observations: np.ndarray,
    rng: np.random.Generator,
    *,
    seed_counts: bool = False,
) -> SpatioSequentialParams:
    """``Emission_Mixture++`` for every class: seeds under the family's own divergence.

    A categorical family is seeded from symbols (a smoothed one-hot row per
    seed); a Gaussian one from values (the seed as the mean, the pooled scale).
    With ``seed_counts`` a count family is seeded too (issue #933):
    :func:`_seed_count_family` says how. Without it, and for any other
    family, the parameters are kept, which the notebook records; the default
    stays off so a caller's start does not move under it. Under
    ``params.shared_emissions`` one family is seeded and every class takes it.

    The Gaussian score is the Bregman divergence exactly. The categorical's
    carries the smoothing constant its seeded row scores at the seed itself,
    ``-log 0.9``, on top of the divergence; by #560's argument that does not
    cancel, and it is left here because removing it moves the coupled model's
    own start (issue #570).
    """
    emissions: list[EmissionFamily] = []
    flat = observations.reshape(-1)
    families = params.emissions[:1] if params.shared_emissions else params.emissions
    for family in families:
        seeded = (
            _seed_count_family(family, params, observations, rng)
            if seed_counts
            else None
        )
        if seeded is not None:
            emissions.append(seeded)
        elif isinstance(family, CategoricalEmission):
            n_symbols = int(family.matrix.shape[1])

            def score(
                seed: float, values: np.ndarray, n_symbols: int = n_symbols
            ) -> np.ndarray:
                row = np.full(n_symbols, 0.1 / (n_symbols - 1))
                row[int(seed)] = 0.9
                return -np.log(row[values.astype(np.int64)])

            seeds = emission_mixture_plus_plus(flat, params.n_states, score, rng)
            matrix = np.full((params.n_states, n_symbols), 0.1 / (n_symbols - 1))
            for k, seed in enumerate(seeds):
                matrix[k, int(seed)] = 0.9
            emissions.append(CategoricalEmission(matrix))
        elif isinstance(family, GaussianEmission):
            pooled = float(np.std(flat)) or 1.0

            def gaussian_score(
                seed: float, values: np.ndarray, pooled: float = pooled
            ) -> np.ndarray:
                return 0.5 * ((values - seed) / pooled) ** 2

            seeds = emission_mixture_plus_plus(
                flat, params.n_states, gaussian_score, rng
            )
            emissions.append(
                GaussianEmission(
                    np.sort(np.asarray(seeds, dtype=float)),
                    np.full(params.n_states, pooled),
                    float(family.variance_floor),
                )
            )
        else:
            emissions.append(family)
    if params.shared_emissions:
        emissions = emissions * params.n_classes
    return replace(params, emissions=tuple(emissions))


def _seed_count_family(
    family: EmissionFamily,
    params: SpatioSequentialParams,
    observations: np.ndarray,
    rng: np.random.Generator,
) -> EmissionFamily | None:
    """A count family seeded by ``Emission_Mixture++`` under its divergence, or ``None`` (issue #933).

    Every (position, node) observation is a candidate, placed in rate space
    where the params carry a covariate: a total over its exposure, successes
    as the fraction of their own trials, over the family's declared count
    (:func:`~sal.sim.count_pairs.rate_space`). The seeds set
    the per-state means and rates; the dispersion and concentration start at
    the family's own, averaged over its states, since one seed carries no
    shape. ``None`` for a family this does not seed.
    """
    covariate = (
        None if params.covariate is None else np.asarray(params.covariate, dtype=float)
    )
    k = params.n_states
    if isinstance(family, IndependentCountPair):
        trials = float(family.successes.trials[0])
        rows = observations.reshape(-1, 2).astype(np.float64)
        if covariate is not None:
            rows = rate_space(rows, covariate.reshape(-1, 2), trials)
        seeding = IndependentCountPairSeeding(
            dispersion=float(family.total.dispersion.mean()),
            concentration=float(family.successes.concentration.mean()),
            trials=trials,
        )
        return plus_plus_start(rows, k, seeding, rng)
    if isinstance(family, NegativeBinomialEmission):
        values = observations.reshape(-1).astype(np.float64)
        if covariate is not None:
            values = values / covariate.reshape(-1)
        dispersion = float(family.dispersion.mean())
        return plus_plus_start(
            values,
            k,
            lambda rows: NegativeBinomialEmission(
                np.full(len(rows), dispersion), np.maximum(rows.reshape(-1), 1e-6)
            ),
            rng,
        )
    if isinstance(family, BetaBinomialEmission):
        trials = float(family.trials[0])
        values = observations.reshape(-1).astype(np.float64)
        if covariate is not None:
            values = values / covariate.reshape(-1) * trials
        concentration = float(family.concentration.mean())

        def at(rows: np.ndarray) -> BetaBinomialEmission:
            rate = (rows.reshape(-1) + 0.5) / (trials + 1.0)
            return BetaBinomialEmission(
                np.full(len(rate), trials),
                rate * concentration,
                (1.0 - rate) * concentration,
            )

        return plus_plus_start(values, k, at, rng)
    return None


def graph_burn_in(
    params: SpatioSequentialParams,
    observations: np.ndarray,
    rng: np.random.Generator,
    schedule: TempSchedule,
    *,
    wolff_moves_per_step: int = 4,
    seed_counts: bool = False,
) -> SpatioSequentialFit:
    """``Graph_BurnIn++`` (the textbook's burn-in algorithm): the blocks while the inverse temperature rises.

    ``schedule`` gives a *temperature* per step; the inverse temperature used
    is ``params.beta / T``, so a schedule ending at ``T = 1`` ends at the
    model's own ``beta``. Each step: one Wolff pass on the labels in the
    current field, a block Gibbs draw of every class's chain, an E step and an
    M step, then the field. Emissions are seeded once by ``Emission_Mixture++``
    and labels uniformly; the returned fit is the state at the schedule's end,
    to be polished by :func:`fit_spatio_sequential`. ``seed_counts`` is
    :func:`seed_emissions`'s.
    """
    params = seed_emissions(params, observations, rng, seed_counts=seed_counts)
    labels = rng.integers(0, params.n_classes, size=params.graph.n_nodes)
    field = external_field(params, observations, labels)
    values = [labelled_log_likelihood(params, observations, labels)]
    offsets, neighbours, couplings = params.graph.compressed_adjacency()
    lists = adjacency_lists(offsets, neighbours, couplings)
    for step in range(schedule.n_steps):
        beta = params.beta / schedule(step)
        declared = SiteField.from_energy(field)
        for _ in range(wolff_moves_per_step):
            wolff_sweep(
                labels,
                declared,
                offsets,
                neighbours,
                couplings,
                rng,
                beta=beta,
                lists=lists,
            )
        density = class_log_density(params, observations, labels)
        for m in range(
            params.n_classes
        ):  # block Gibbs over the chains, recorded nowhere
            sample_path(
                density[m], np.log(params.initial[m]), np.log(params.transition), rng
            )
        posteriors = class_posteriors(params, observations, labels)
        params = m_step(params, observations, labels, posteriors)
        posteriors = class_posteriors(params, observations, labels)
        field = external_field(params, observations, labels, posteriors.posterior)
        values.append(labelled_log_likelihood(params, observations, labels))
    return SpatioSequentialFit(
        params,
        labels,
        np.array(values),
        field,
        Termination.after(schedule.n_steps, converged=False),
    )
