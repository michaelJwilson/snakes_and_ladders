"""A mixture of count emissions: the Gaussian mixture with its family swapped.

:mod:`sal.opt.mixture`'s mixture machinery --- the
log-likelihood, the responsibilities, the weights as their mean --- asks its
components only for a log-density and an M step, so it is reused here and
this module adds the loop around them and the start that seeds it.

**The M step is the family's, and nothing here reimplements one.** A count
family's is an optimization rather than a formula
(:class:`sal.emissions.CountPairEmission` solves for a
dispersion and for a beta-binomial's two shape parameters) and reports whether
it settled. This loop propagates that report: an unconverged inner solve
reaching an outer likelihood is the fault ``likelihood/CLAUDE.md`` forbids.

**And it is where ``Emission_Mixture++`` finally has a model.** Issue #306
built the seeding rule --- k-means++ with a family's own Bregman divergence as
the distance --- and recorded that no problem used it.

Ground truth and data generation live in
:mod:`sal.sim.emission_mixture`; this module draws no data.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from sal.emissions import (
    BetaBinomialEmission,
    BinomialEmission,
    CountPairEmission,
    EmissionFamily,
    NegativeBinomialEmission,
    PoissonEmission,
)
from sal.enumeration import refuse_oversized
from sal.opt.constrain import (
    free_from_log_simplex,
    free_from_positive,
    log_simplex,
    positive,
)
from sal.opt.em import em_loop
from sal.opt.mixture import (
    e_step,
    emission_mixture_plus_plus,
    mixture_log_likelihood,
    responsibilities,
    uniform_seeds,
)
from sal.opt.objective import Objective
from sal.opt.termination import Termination

#: Builds a ``k``-state family centred on ``k`` observations, one per row. The
#: seam an initializer needs from a model whose parameters it cannot otherwise
#: interpret --- the counterpart of
#: :meth:`sal.opt.mixture.GaussianMixtureObjective.theta_from_centres`
#: for a family that has no ``theta``.
type ComponentsAt = Callable[[np.ndarray], EmissionFamily]


@dataclass(frozen=True)
class EmissionMixtureFit:
    """What one expectation-maximization run produced.

    Parameters
    ----------
    weights : torch.Tensor
        Fitted mixing weights, shape ``(n_components,)``.
    components : EmissionFamily
        The fitted component family.
    responsibilities : torch.Tensor
        ``P(component | observation)`` at the fitted parameters, shape
        ``(n_samples, n_components)``. Carried on the result because it is the
        E step the last M step consumed; recomputing it from the returned
        parameters gives the next iteration's.
    log_likelihood : float
        The log-likelihood at the returned parameters. A probability, since
        every count family is discrete, so it is at most zero.
    iterations : int
        EM iterations run.
    at_boundary : bool
        Whether a component's M step reached the edge of the range this data
        identifies its parameter over --- a flat likelihood in a dispersion or
        a concentration, reported rather than treated as an error (issue
        #122).
    termination : Termination | None
        Whether the loop met its relative tolerance or ran out of iterations,
        in the form every result states it in (issue #860). ``iterations``
        stays: it is what this result has always been read by.
    """

    weights: torch.Tensor
    components: EmissionFamily
    responsibilities: torch.Tensor
    log_likelihood: float
    iterations: int
    at_boundary: bool
    termination: Termination | None = None


def expectation_maximization(
    observations: np.ndarray | torch.Tensor,
    weights: np.ndarray | torch.Tensor,
    components: EmissionFamily,
    max_iterations: int = 200,
    tolerance: float = 1e-10,
    *,
    covariate: np.ndarray | torch.Tensor | None = None,
) -> EmissionMixtureFit:
    """Fit a mixture of count emissions by EM.

    The E step is :func:`sal.opt.mixture.responsibilities` and
    the M step is the family's own :meth:`reestimate`: independent
    observations carry no message between them, and the family receives the
    posterior an HMM's forward--backward pass would hand it. The alternation
    around the two is :func:`sal.opt.em.em_loop`'s, which is
    the rung this one shares with
    :func:`sal.opt.mixture.expectation_maximization` and its
    only one: the defaults, the result type and the oracle this is pinned
    against are here (issue #859).

    Parameters
    ----------
    observations : np.ndarray | torch.Tensor
        Observations, shape ``(n_samples,)`` or ``(n_samples, channels)``.
    weights : np.ndarray | torch.Tensor
        Starting mixing weights, shape ``(n_components,)``; an array is read
        as a tensor of its own dtype, and a tensor is used as given.
    components : EmissionFamily
        Starting components.
    max_iterations : int
        Maximum EM iterations.
    tolerance : float
        Stop when the log-likelihood improves by less than this *relative* to
        its magnitude --- absolute would not transfer across data sizes
        (``DEV.md``, issue #111).
    covariate : np.ndarray | torch.Tensor | None
        Per-observation covariate, scored in the E step and conditioned on in
        the M step alike (issue #933): an exposure, a trial count, or one of
        each per channel for a pair. ``None`` fits as before, bitwise.

    Returns
    -------
    EmissionMixtureFit
        The fitted parameters, the responsibilities at them, and the
        log-likelihood.

    Raises
    ------
    ValueError
        If a component's M step did not converge. A number read off an inner
        solve that never settled is not an estimate, and returning it here
        would surface several iterations later as a non-monotone likelihood.

    Notes
    -----
    Integer counts in one channel, under a Poisson, binomial, negative
    binomial or beta-binomial family, are fitted on their distinct values
    (issue #997): a responsibility is a function of the value alone, so the
    E step scores each distinct count once and the M step reads the counts
    weighted by how many observations hold them. The responsibilities are
    gathered back to every observation once, after the last step. At 10^6
    draws of a three-component negative binomial, 20 iterations took 2.1 s
    per observation, 43% of it the `torch.unique` each log-density repeated.
    A float array of the same counts takes the per-observation route, which
    is the oracle.
    """
    weights = torch.as_tensor(weights)
    distinct = _distinct_counts(observations, components)
    if distinct is not None:
        return _cell_expectation_maximization(
            *distinct,
            weights,
            components,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
    values = torch.as_tensor(observations, dtype=torch.float64)
    conditioned = (
        None if covariate is None else torch.as_tensor(covariate, dtype=torch.float64)
    )
    boundary = False
    attempt = 0

    def step(
        state: tuple[torch.Tensor, EmissionFamily, torch.Tensor],
    ) -> tuple[tuple[torch.Tensor, EmissionFamily, torch.Tensor], float]:
        """One E step, one M step, and the log-likelihood at the state given."""
        nonlocal boundary, attempt
        attempt += 1
        current, family, _ = state
        log_weight = torch.log(current)
        evidence, posterior = e_step(values, log_weight, family, covariate=conditioned)
        log_likelihood = float(evidence)
        reestimated = (
            family.reestimate(values, posterior)
            if conditioned is None
            else family.reestimate(values, posterior, conditioned)
        )
        if not reestimated.converged:
            msg = (
                f"a component's M step did not settle at EM iteration "
                f"{attempt}: residual {reestimated.residual:.3e} after "
                f"{reestimated.iterations} inner iterations"
            )
            raise ValueError(msg)
        boundary = boundary or reestimated.at_boundary
        advanced = (posterior.mean(dim=0), reestimated.emissions, posterior)
        return advanced, log_likelihood

    # The responsibilities of the last E step are carried out of the loop, so
    # the result holds the posterior its final M step consumed. Before the
    # first step there is none, and an empty budget returns that.
    start = (
        weights,
        components,
        torch.empty((values.shape[0], components.n_states), dtype=torch.float64),
    )
    (weights, components, posterior), log_likelihood, termination = em_loop(
        step, start, tolerance=tolerance, max_iterations=max_iterations
    )
    return EmissionMixtureFit(
        weights=weights,
        components=components,
        responsibilities=posterior,
        log_likelihood=log_likelihood,
        iterations=termination.iterations,
        at_boundary=boundary,
        termination=termination,
    )


def log_densities(observations: np.ndarray, components: EmissionFamily) -> np.ndarray:
    """``log p_k(x_i)`` for every observation and component, shape ``(n_samples, K)``, as an array.

    The family's own :meth:`log_density` on the observations as float64, read
    out once: the E step's input for a consumer that holds arrays and takes no
    derivative (issue #1011).
    """
    values = torch.as_tensor(observations, dtype=torch.float64)
    return components.log_density(values).detach().numpy()


def responsibilities_at(
    observations: np.ndarray, weights: np.ndarray, components: EmissionFamily
) -> np.ndarray:
    """The E step at ``weights`` and ``components``, shape ``(n_samples, K)``, as an array.

    :func:`sal.opt.mixture.responsibilities` on the observations
    as float64 and ``log(weights)``, read out once (issue #1011).
    """
    values = torch.as_tensor(observations, dtype=torch.float64)
    log_weight = torch.log(torch.as_tensor(weights, dtype=torch.float64))
    return responsibilities(values, log_weight, components).detach().numpy()


def partial_expectation_maximization(
    observations: np.ndarray,
    posterior: np.ndarray,
    components: EmissionFamily,
    affected: Sequence[int],
    mass: np.ndarray,
    iterations: int,
) -> tuple[np.ndarray, EmissionFamily]:
    """EM on the ``affected`` components alone, their total claim on each observation held at ``mass``.

    The partial EM of Ueda, Nakano, Ghahramani & Hinton (2000): one M step
    realizes ``posterior``, then each of ``iterations`` steps redistributes
    ``mass`` among the affected components by their joint, softmax-normalized
    among them, and M-steps again. Every other column is never touched, so
    the unaffected components are refitted to the responsibilities they
    started with.

    Parameters
    ----------
    observations : np.ndarray
        Observations, shape ``(n_samples,)`` or ``(n_samples, channels)``.
    posterior : np.ndarray
        The responsibilities the first M step realizes, shape
        ``(n_samples, K)``.
    components : EmissionFamily
        The family whose ``reestimate`` realizes them.
    affected : Sequence[int]
        The components partial EM redistributes among.
    mass : np.ndarray
        Their total responsibility on each observation, shape
        ``(n_samples, 1)``.
    iterations : int
        Partial E steps after the first M step, each followed by an M step.

    Returns
    -------
    tuple[np.ndarray, EmissionFamily]
        The mixing weights, the mean of the last responsibilities, and the
        family its last M step fitted.

    Raises
    ------
    ValueError
        If a component's M step did not converge.
    """
    values = torch.as_tensor(observations, dtype=torch.float64)
    moved = torch.from_numpy(posterior)
    held = torch.from_numpy(mass)
    columns = list(affected)
    family = components
    for iteration in range(iterations + 1):
        reestimated = family.reestimate(values, moved)
        if not reestimated.converged:
            msg = (
                f"a component's M step did not settle at partial EM iteration "
                f"{iteration}"
            )
            raise ValueError(msg)
        family = reestimated.emissions
        weights = moved.mean(dim=0)
        if iteration == iterations:
            break
        joint = torch.log(weights) + family.log_density(values)
        local = joint[:, columns]
        moved = moved.clone()
        moved[:, columns] = held * torch.softmax(local, dim=-1)
    return weights.detach().numpy(), family


#: The count families whose M step reads only weighted counts, so a fit on
#: the distinct values with their multiplicities is the per-observation fit.
_COUNTED = (
    PoissonEmission,
    BinomialEmission,
    NegativeBinomialEmission,
    BetaBinomialEmission,
)


def _distinct_counts(
    observations: np.ndarray | torch.Tensor, components: EmissionFamily
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """The distinct counts, their multiplicities and each observation's index, or ``None``."""
    if type(components) not in _COUNTED:
        return None
    values = (
        observations.detach().numpy()
        if isinstance(observations, torch.Tensor)
        else np.asarray(observations)
    )
    if (
        values.ndim != 1
        or values.size == 0
        or not np.issubdtype(values.dtype, np.integer)
    ):
        return None
    if int(values.min()) < 0:
        return None
    cells, inverse, multiplicity = np.unique(
        values, return_inverse=True, return_counts=True
    )
    return cells, multiplicity, inverse


def _cell_expectation_maximization(
    cells: np.ndarray,
    multiplicity: np.ndarray,
    inverse: np.ndarray,
    weights: torch.Tensor,
    components: EmissionFamily,
    *,
    max_iterations: int,
    tolerance: float,
) -> EmissionMixtureFit:
    """:func:`expectation_maximization` on the distinct counts (issue #997)."""
    support = torch.from_numpy(cells.astype(np.float64))
    held = torch.from_numpy(multiplicity.astype(np.float64))
    n_samples = float(multiplicity.sum())
    boundary = False
    attempt = 0

    def step(
        state: tuple[torch.Tensor, EmissionFamily, torch.Tensor],
    ) -> tuple[tuple[torch.Tensor, EmissionFamily, torch.Tensor], float]:
        nonlocal boundary, attempt
        attempt += 1
        current, family, _ = state
        joint = torch.log(current) + family.log_density(support)
        normalizer = torch.logsumexp(joint, dim=1, keepdim=True)
        log_likelihood = float((held * normalizer[:, 0]).sum())
        posterior = torch.exp(joint - normalizer)
        weighted = posterior * held[:, None]
        reestimated = family.reestimate(support, weighted)
        if not reestimated.converged:
            msg = (
                f"a component's M step did not settle at EM iteration "
                f"{attempt}: residual {reestimated.residual:.3e} after "
                f"{reestimated.iterations} inner iterations"
            )
            raise ValueError(msg)
        boundary = boundary or reestimated.at_boundary
        return (weighted.sum(dim=0) / n_samples, reestimated.emissions, posterior), (
            log_likelihood
        )

    start = (
        weights,
        components,
        torch.empty((0, components.n_states), dtype=torch.float64),
    )
    (weights, components, posterior), log_likelihood, termination = em_loop(
        step, start, tolerance=tolerance, max_iterations=max_iterations
    )
    responsibilities = (
        posterior[torch.from_numpy(inverse)]
        if posterior.shape[0]
        else torch.empty((inverse.size, components.n_states), dtype=torch.float64)
    )
    return EmissionMixtureFit(
        weights=weights,
        components=components,
        responsibilities=responsibilities,
        log_likelihood=log_likelihood,
        iterations=termination.iterations,
        at_boundary=boundary,
        termination=termination,
    )


def enumerated_posterior(
    observations: np.ndarray | torch.Tensor,
    log_weight: torch.Tensor,
    components: EmissionFamily,
) -> torch.Tensor:
    """``P(component | observations)`` summed over every joint labelling.

    The independent answer :func:`sal.opt.mixture.responsibilities`
    is refereed against, sharing no line with it: the responsibilities
    normalize each observation's row on its own, while this scores each of the
    ``K ** N`` labellings of the whole dataset, normalizes over all of them,
    and marginalizes back to one row per observation. Their agreement is what
    says the mixture's posterior factorizes across observations.

    Exponential in the number of observations, so affordable only over a
    handful. The observations being independent, the marginal at each of a
    handful is the quantity the whole dataset's carries, which is what makes a
    prefix a legitimate oracle.

    Parameters
    ----------
    observations : np.ndarray | torch.Tensor
        Observations, shape ``(n_samples,)`` or ``(n_samples, channels)``.
    log_weight : torch.Tensor
        Log mixing weights, shape ``(n_components,)``.
    components : EmissionFamily
        The component densities.

    Returns
    -------
    torch.Tensor
        Shape ``(n_samples, n_components)``, each row summing to one.

    Raises
    ------
    ValueError
        If ``n_components ** n_samples`` is past
        :data:`sal.enumeration.MAX_ENUMERABLE_CONFIGURATIONS`.
    """
    values = torch.as_tensor(observations, dtype=torch.float64)
    n_samples = int(values.shape[0])
    n_components = int(log_weight.shape[0])
    refuse_oversized(
        n_components**n_samples,
        what=f"{n_components} ** {n_samples} component labellings",
    )
    scored = (components.log_density(values) + log_weight).numpy()

    labellings = list(itertools.product(range(n_components), repeat=n_samples))
    joint = np.array(
        [
            sum(scored[position, state] for position, state in enumerate(labelling))
            for labelling in labellings
        ]
    )
    probability = np.exp(joint - joint.max())
    probability /= probability.sum()

    marginal = np.zeros((n_samples, n_components))
    for weight, labelling in zip(probability, labellings, strict=True):
        for position, state in enumerate(labelling):
            marginal[position, state] += weight
    return torch.as_tensor(marginal)


@dataclass(frozen=True)
class CountPairSeeding:
    """Places a :class:`CountPairEmission` component on an observed pair.

    The pair says where a component starts: its depth is the component's
    negative-binomial mean, and its allele fraction, smoothed by the Jeffreys
    prior's half-count so ``(n, 0)`` does not seed a rate of exactly zero, is
    the beta-binomial rate. One observation carries no *shape*, so the
    dispersion and concentration start at declared values shared by every
    component.

    Parameters
    ----------
    dispersion : float
        Starting negative-binomial ``r`` for every component.
    concentration : float
        Starting beta-binomial ``a + b`` for every component.
    joint : bool
        Which form of the family to build. No default, for the reason
        :class:`CountPairEmission` gives.
    trials : float | None
        The independent form's fixed trial count, one number shared by every
        component: a property of the assay rather than of a component, and a
        per-component vector is unreadable when the seeding scores one
        candidate component against the observations. ``None`` in the joint
        form.

    Raises
    ------
    ValueError
        If the trial count is given in the joint form or omitted in the
        independent one, on the terms :class:`CountPairEmission` states.
    """

    dispersion: float
    concentration: float
    joint: bool
    trials: float | None = None

    def __post_init__(self) -> None:
        if self.joint and self.trials is not None:
            msg = "the joint form's trial count is the observed total"
            raise ValueError(msg)
        if not self.joint and self.trials is None:
            msg = "the independent form needs a fixed trial count"
            raise ValueError(msg)

    def __call__(self, rows: np.ndarray) -> CountPairEmission:
        """The family seeded on these pairs, one component per row.

        Parameters
        ----------
        rows : np.ndarray
            Observed pairs, shape ``(n_components, 2)``.

        Returns
        -------
        CountPairEmission
            A family with one state per row.

        """
        pairs = np.asarray(rows, dtype=np.float64).reshape(-1, 2)
        totals, successes = pairs[:, 0], pairs[:, 1]
        trials = None if self.trials is None else np.full(pairs.shape[0], self.trials)
        over = np.maximum(totals, 1.0) if trials is None else trials
        rate = (successes + 0.5) / (over + 1.0)
        return CountPairEmission(
            np.full(pairs.shape[0], self.dispersion),
            np.maximum(totals, 1.0),
            rate * self.concentration,
            (1.0 - rate) * self.concentration,
            trials,
            joint=self.joint,
        )


def _seed_scores(
    observations: np.ndarray, at: ComponentsAt
) -> Callable[..., np.ndarray]:
    """``(index, indices) -> D_phi(y, the component seeded at that index)``.

    :func:`sal.opt.mixture.emission_mixture_plus_plus` draws its
    seeds from the array it is given, so the array here is of *indices*: an
    observation is a pair, and a draw from a flattened array of pairs would
    seed a component on half of one.

    **The score is the family's Bregman divergence, not its negative log
    density** (issue #560). The two differ by ``log b_phi(y)``, the log density
    at the family's best member for that observation, which depends on the
    observation and not on the seed; D-squared sampling normalizes its scores
    rather than shifting them, so that term does not cancel --- it dilutes the
    rule toward uniform in proportion to its size against a typical divergence.
    Measured on a Gaussian, where the divergence is the squared Euclidean
    distance exactly: seeding under the divergence costs **1.8496** times the
    exact optimum, under the negative log density **3.9111**, and uniformly
    **4.3470** (``tests/regression/opt/test_opt_mixture_seeding.py``,
    ``docs/experiments/010``). The divergence is non-negative, as the sampling
    rule needs, and zero at the seed's own observation.
    """
    rows = np.asarray(observations, dtype=np.float64)

    def score(seed: float, candidates: np.ndarray) -> np.ndarray:
        family = at(rows[[int(seed)]])
        scored = family.bregman_divergence(
            torch.as_tensor(rows[candidates.astype(np.int64)], dtype=torch.float64)
        )
        return np.asarray(scored[:, 0].numpy())

    return score


def plus_plus_start(
    observations: np.ndarray,
    n_components: int,
    at: ComponentsAt,
    rng: np.random.Generator,
) -> EmissionFamily:
    """Seed the components by ``Emission_Mixture++`` (issue #306, ``eq:kmeanspp``).

    Parameters
    ----------
    observations : np.ndarray
        Observations, shape ``(n_samples,)`` or ``(n_samples, channels)``.
    n_components : int
        Components to seed.
    at : ComponentsAt
        Builds a family from the chosen observations.
    rng : np.random.Generator
        Generator, passed in rather than seeded here (``sim/CLAUDE.md``).

    Returns
    -------
    EmissionFamily
        The seeded components.
    """
    rows = np.asarray(observations, dtype=np.float64)
    indices = np.arange(rows.shape[0], dtype=np.float64)
    chosen = emission_mixture_plus_plus(
        indices, n_components, _seed_scores(rows, at), rng
    )
    return at(rows[chosen.astype(np.int64)])


def uniform_start(
    observations: np.ndarray,
    n_components: int,
    at: ComponentsAt,
    rng: np.random.Generator,
) -> EmissionFamily:
    """Seed the components on observations drawn uniformly, without replacement.

    The baseline :func:`plus_plus_start` is measured against, kept beside it
    on the terms :func:`sal.opt.mixture.uniform_seeds` states.

    Parameters
    ----------
    observations : np.ndarray
        Observations, shape ``(n_samples,)`` or ``(n_samples, channels)``.
    n_components : int
        Components to seed.
    at : ComponentsAt
        Builds a family from the chosen observations.
    rng : np.random.Generator
        Generator, passed in.

    Returns
    -------
    EmissionFamily
        The seeded components.
    """
    rows = np.asarray(observations, dtype=np.float64)
    indices = np.arange(rows.shape[0], dtype=np.float64)
    chosen = uniform_seeds(indices, n_components, rng)
    return at(rows[chosen.astype(np.int64)])


class EmissionMixtureObjective(Objective):
    """Negative log-likelihood of a mixture of any family whose parameters are positive (issue #964).

    ``theta`` is ``K - 1`` free weights, then for each parameter name the
    family states, ``K`` log values: the weights through
    :func:`~sal.opt.constrain.log_simplex` and every
    parameter through :func:`~sal.opt.constrain.positive`.
    ``build`` turns the named parameters back into a family, so the
    objective is differentiable in ``theta`` wherever the family's
    ``log_density`` is, and a Hamiltonian or Langevin chain can sample a
    count mixture the Gaussian :class:`~sal.opt.mixture.GaussianMixtureObjective`
    cannot express.

    Parameters
    ----------
    observations : np.ndarray
        Shape ``(n,)`` or ``(n, channels)``, in the family's dtype.
    start : EmissionFamily
        The family :meth:`initial` starts at, with uniform weights; its
        :meth:`~sal.emissions.EmissionFamily.named_parameters`
        name the blocks of ``theta``, every one positive.
    build : Callable[[Mapping[str, torch.Tensor]], EmissionFamily]
        The family at named parameters of shape ``(K,)`` each; constants the
        family carries (a trial count, the joint form) are the closure's.

    Raises
    ------
    ValueError
        If ``start`` has fewer than two states or a parameter is not positive.
    """

    def __init__(
        self,
        observations: np.ndarray,
        start: EmissionFamily,
        build: Callable[[Mapping[str, torch.Tensor]], EmissionFamily],
    ) -> None:
        if start.n_states < 2:
            msg = f"a mixture has at least two components, got {start.n_states}"
            raise ValueError(msg)
        named = {
            name: torch.as_tensor(value, dtype=torch.float64).reshape(-1)
            for name, value in start.named_parameters().items()
        }
        if any(bool((value <= 0).any()) for value in named.values()):
            msg = "every parameter of the family must be positive"
            raise ValueError(msg)
        self._observations = torch.as_tensor(
            observations, dtype=start.observation_dtype
        )
        self._start = named
        self._names = tuple(named)
        self._k = start.n_states
        self._build = build

    @property
    def observations(self) -> torch.Tensor:
        """The observations being fitted."""
        return self._observations

    @property
    def n_components(self) -> int:
        """``K``."""
        return self._k

    @property
    def n_parameters(self) -> int:
        """``K - 1`` free weights and ``K`` per named parameter."""
        return self._k - 1 + self._k * len(self._names)

    def _blocks(self, theta: torch.Tensor) -> dict[str, torch.Tensor]:
        offset = self._k - 1
        blocks: dict[str, torch.Tensor] = {}
        for name in self._names:
            blocks[name] = positive(theta[offset : offset + self._k])
            offset += self._k
        return blocks

    def components(self, theta: torch.Tensor) -> EmissionFamily:
        """The family ``theta`` encodes, differentiable in ``theta``."""
        return self._build(self._blocks(theta))

    def initial(self) -> torch.Tensor:
        """Uniform weights at ``start``'s parameters."""
        uniform = torch.full((self._k,), -math.log(self._k), dtype=torch.float64)
        return self.theta_from({"log_weight": uniform, **self._start})

    def constrain(self, theta: torch.Tensor) -> Mapping[str, torch.Tensor]:
        """The log weights and every named parameter."""
        return {"log_weight": log_simplex(theta[: self._k - 1]), **self._blocks(theta)}

    def theta_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """The unconstrained vector whose :meth:`constrain` is ``named``."""
        parts = [free_from_log_simplex(torch.as_tensor(named["log_weight"]))]
        parts += [
            free_from_positive(torch.as_tensor(named[name], dtype=torch.float64))
            for name in self._names
        ]
        return torch.cat(parts)

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        """The negative mixture log-likelihood at ``theta``."""
        return -mixture_log_likelihood(
            self._observations,
            log_simplex(theta[: self._k - 1]),
            self.components(theta),
        )
