"""Gumbel-softmax over discrete chains: gradients where the answer is known.

``ROADMAP.md`` Stage 3 asks for differentiable search over discrete structure
in two places --- tree topologies via the tropical Grassmannian, and the
Gumbel-softmax relaxation of Potts and HMM states. Only the second has an
oracle: the spaces it relaxes are enumerable, so "does gradient ascent on a
relaxation find what discrete search finds" is a *falsifiable* question here
and is not one for topologies at any interesting size. This module takes that
half and nothing else.

**The relaxation is an extension, not a second model.** A configuration
becomes an ``(L, k)`` row-stochastic matrix and each objective below is the
same score its discrete counterpart computes, read on the simplex. At a
one-hot the two agree to float64, pinned over every configuration of an
enumerable chain. A relaxation that is not an extension of what it relaxes is
optimizing a different problem, and nothing measured against it transfers.

**One identity does a lot of work, and its boundary is multilinearity ---
not the chain.** Both objectives are *multilinear*: every term involves at
most one factor from each site. Under a factorized ``q`` distinct sites are
independent, so each term's expectation is that term at the marginals, and::

    E_q[score(s)] = score(q)

exactly. The bilinear form at the *marginals* is the expected discrete score,
not an approximation of it. Three things follow.

* The deterministic relaxation (:func:`optimize` with ``stochastic=False``)
  needs no sampling and approximates nothing.
* The estimators' question becomes sharp: the relaxation is exact, so
  whatever bias :func:`estimate_gradient` measures belongs to Gumbel-softmax
  alone.
* A multilinear function on a product of simplices attains its maximum at a
  vertex, so the relaxation introduces **no new global optimum**. Everything
  a relaxed search can lose is lost to local optima of the ascent, never to
  the relaxation itself.

The boundary is easy to state wrongly, and two plausible statements of it are
false. It is *not* that the model must be a chain: a lattice, or any graph, is
equally multilinear. It is *not* that the terms must be pairwise either --- a
term coupling three **distinct** sites is still one factor per site, and the
identity still holds, which a test checks.

What breaks it is a term using the *same* site twice, since ``E[X**2]`` is
``E[X]`` for an indicator and not ``E[X]**2``. That is not hypothetical here:
:class:`phylo.sim.graph.PottsGraph` deliberately permits a doubled bond,
because a periodic lattice of extent 2 produces one legitimately, and such an
edge would join a node to itself after wrapping. A test pins the failure on
exactly that shape --- measured, the two sides read 1.000 against 0.557 --- so
the limit is a checked boundary rather than a sentence.

Two estimators are provided because the choice between them is empirical.
:data:`RelaxationMode.SOFT` is differentiable throughout and optimizes a
smoothed objective at any ``temperature > 0``;
:data:`RelaxationMode.STRAIGHT_THROUGH` takes a discrete forward pass and a
soft backward one, so its forward value is always a real configuration's score
and its gradient is biased. Both are measured against the exact gradient
rather than chosen on principle.

See Jang, Gu & Poole (2017); Maddison, Mnih & Teh (2017).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

import numpy as np
import torch

from phylo.learn.potts import Configuration, PottsLandscape

#: Below this the softmax saturates in float64 and the gradient underflows, so
#: a smaller temperature is refused rather than silently returning a stalled
#: optimization.
MINIMUM_TEMPERATURE = 1e-3


class RelaxationMode(StrEnum):
    """Which Gumbel-softmax estimator to use.

    A ``StrEnum`` rather than a bare string for the reason
    :class:`phylo.sim.graph.BoundaryCondition` is one: an unrecognized mode is
    a type error at the call site, not a branch that silently does the wrong
    thing.
    """

    #: Softmax of the perturbed logits, used as-is. Differentiable
    #: throughout; the forward value is a simplex point, not a configuration.
    SOFT = "soft"
    #: One-hot forward, soft backward. The forward value is always a real
    #: configuration's score; the gradient is the soft one and is biased.
    STRAIGHT_THROUGH = "straight-through"


@runtime_checkable
class RelaxedObjective(Protocol):
    """A discrete chain score and its extension to the simplex.

    The seam this module is built on: everything below --- the estimators, the
    exact gradient, the optimizer --- is written against this and never
    against a Potts chain or an HMM in particular. Adding a third discrete
    space means writing one of these, not a second optimizer.
    """

    @property
    def n_sites(self) -> int:
        """Sites in the chain."""
        ...

    @property
    def n_states(self) -> int:
        """States available at each site."""
        ...

    def relaxed(self, probabilities: torch.Tensor) -> torch.Tensor:
        """The score on the simplex, differentiable in ``probabilities``."""
        ...

    def discrete(self, configuration: Configuration) -> float:
        """The score of one configuration, computed without the relaxation."""
        ...


@dataclass(frozen=True)
class RelaxedPotts:
    """The Potts chain's score, extended to the simplex.

    ``J * sum_t sum_a P[t, a] P[t+1, a] + sum_t sum_a P[t, a] h[a]``, which at
    a one-hot is :meth:`phylo.learn.potts.PottsLandscape.energy` exactly. The
    parameters are read off the landscape rather than re-declared, so the
    extension cannot drift from what it extends.
    """

    landscape: PottsLandscape

    @property
    def n_sites(self) -> int:
        return self.landscape.chain_length

    @property
    def n_states(self) -> int:
        return self.landscape.n_states

    def relaxed(self, probabilities: torch.Tensor) -> torch.Tensor:
        field = torch.from_numpy(self.landscape.field)
        agreement = (probabilities[:-1] * probabilities[1:]).sum()
        return self.landscape.coupling * agreement + (probabilities * field).sum()

    def discrete(self, configuration: Configuration) -> float:
        return self.landscape.energy(configuration)


@dataclass(frozen=True)
class RelaxedHmmPath:
    """An HMM hidden path's joint log-probability, extended to the simplex.

    ``log P(path, observations)`` read on the simplex: the initial and
    emission terms are linear in the per-site probabilities and the transition
    term is bilinear in adjacent pairs. At a one-hot it is
    :func:`phylo.likelihood.hmm_paths.path_log_probability` exactly, which a
    test pins --- across the module boundary, since ``phylo.learn`` may not
    import ``phylo.likelihood`` and the two implementations are therefore
    genuinely independent.

    Parameters are taken as *log* matrices rather than as an ``HmmParams``
    because ``learn/CLAUDE.md`` forbids importing ``phylo.sim``. That is a
    constraint doing real work here: the objective ends up depending on three
    arrays rather than on a fixture type.

    Parameters
    ----------
    log_initial : torch.Tensor
        Shape ``(k,)``.
    log_transition : torch.Tensor
        Shape ``(k, k)``.
    log_emission : torch.Tensor
        Shape ``(k, m)``.
    observations : torch.Tensor
        Integer symbols, shape ``(T,)``.
    """

    log_initial: torch.Tensor
    log_transition: torch.Tensor
    log_emission: torch.Tensor
    observations: torch.Tensor

    @property
    def n_sites(self) -> int:
        return int(self.observations.shape[0])

    @property
    def n_states(self) -> int:
        return int(self.log_initial.shape[0])

    def relaxed(self, probabilities: torch.Tensor) -> torch.Tensor:
        emitted = self.log_emission[:, self.observations].T
        start = (probabilities[0] * self.log_initial).sum()
        transitions = torch.einsum(
            "ta,ab,tb->", probabilities[:-1], self.log_transition, probabilities[1:]
        )
        return start + transitions + (probabilities * emitted).sum()

    def discrete(self, configuration: Configuration) -> float:
        total = float(self.log_initial[configuration[0]])
        for step, state in enumerate(configuration):
            total += float(self.log_emission[state, int(self.observations[step])])
            if step:
                total += float(self.log_transition[configuration[step - 1], state])
        return total


@dataclass(frozen=True)
class RelaxedOptimum:
    """The result of gradient ascent on the relaxed objective.

    Parameters
    ----------
    configuration : Configuration
        The discrete configuration read off the final logits by ``argmax``.
    score : float
        Its *discrete* score. Reported rather than the relaxed value, because
        the relaxed value at ``temperature > 0`` is not attained by any
        configuration and comparing it against an enumerated optimum would
        flatter the method.
    relaxed_score : float
        The relaxed objective at the final logits, which is what
        :func:`temperature_sweep` reports the gap of.
    steps : int
        Gradient steps taken.
    """

    configuration: Configuration
    score: float
    relaxed_score: float
    steps: int


def one_hot(configuration: Configuration, n_states: int) -> torch.Tensor:
    """A configuration as a ``(L, k)`` corner of the simplex."""
    matrix = torch.zeros((len(configuration), n_states), dtype=torch.float64)
    matrix[torch.arange(len(configuration)), list(configuration)] = 1.0
    return matrix


def gumbel_softmax(
    logits: torch.Tensor,
    temperature: float,
    generator: torch.Generator,
    *,
    mode: RelaxationMode = RelaxationMode.SOFT,
) -> torch.Tensor:
    """One Gumbel-softmax sample, per row.

    Parameters
    ----------
    logits : torch.Tensor
        Shape ``(L, k)``. Unnormalized; only differences within a row matter.
    temperature : float
        The ``tau`` of the relaxation, ``>= MINIMUM_TEMPERATURE``.
    generator : torch.Generator
        Passed in rather than seeded here, so a caller drawing a *batch* gets
        independent samples --- the mistake a ``seed`` parameter invites, and
        which this repository has made before.
    mode : RelaxationMode
        ``STRAIGHT_THROUGH`` rounds the forward pass to a one-hot while leaving
        the backward pass soft, by the standard ``hard - soft.detach() + soft``
        identity.

    Raises
    ------
    ValueError
        If ``temperature`` is below :data:`MINIMUM_TEMPERATURE`.
    """
    if temperature < MINIMUM_TEMPERATURE:
        msg = (
            f"temperature must be >= {MINIMUM_TEMPERATURE}, got {temperature}: "
            "below it the softmax saturates and the gradient underflows"
        )
        raise ValueError(msg)

    uniform = torch.rand(logits.shape, generator=generator, dtype=torch.float64)
    gumbel = -torch.log(-torch.log(uniform))
    soft = torch.softmax((logits + gumbel) / temperature, dim=1)
    if mode is RelaxationMode.SOFT:
        return soft
    hard = torch.zeros_like(soft).scatter_(1, soft.argmax(dim=1, keepdim=True), 1.0)
    return hard - soft.detach() + soft


def enumerate_optimum(objective: RelaxedObjective) -> tuple[Configuration, float]:
    """The best configuration, by trying every one.

    The oracle. Exponential, and affordable only because these instances are
    deliberately small --- the same role
    :func:`phylo.learn.potts.optimum` plays for the chain and exhaustive
    topology enumeration plays for tree search.
    """
    best, best_score = None, -np.inf
    for candidate in itertools.product(
        range(objective.n_states), repeat=objective.n_sites
    ):
        score = objective.discrete(candidate)
        if score > best_score:
            best, best_score = candidate, score
    assert best is not None
    return best, best_score


def exact_expected_score(objective: RelaxedObjective, logits: torch.Tensor) -> float:
    """``E_q[score(s)]`` by enumerating every configuration.

    The oracle the identity in the module docstring is checked against. Shares
    no algebra with :meth:`RelaxedObjective.relaxed`, which is what makes
    agreement between them evidence rather than a tautology.
    """
    probabilities = torch.softmax(logits, dim=1)
    index = torch.arange(objective.n_sites)
    total = 0.0
    for candidate in itertools.product(
        range(objective.n_states), repeat=objective.n_sites
    ):
        weight = float(torch.prod(probabilities[index, list(candidate)]))
        total += weight * objective.discrete(candidate)
    return total


def exact_expected_gradient(
    objective: RelaxedObjective, logits: torch.Tensor
) -> np.ndarray:
    """``d E_q[score] / d logits``, by autodiff through the enumeration.

    The gradient every estimator is a biased approximation of. Differentiates
    the explicit sum over configurations rather than the closed form, so the
    two routes stay independent.
    """
    parameters = logits.detach().clone().requires_grad_(True)
    probabilities = torch.softmax(parameters, dim=1)
    index = torch.arange(objective.n_sites)

    total = torch.zeros((), dtype=torch.float64)
    for candidate in itertools.product(
        range(objective.n_states), repeat=objective.n_sites
    ):
        weight = torch.prod(probabilities[index, list(candidate)])
        total = total + weight * objective.discrete(candidate)
    total.backward()  # type: ignore[no-untyped-call]

    assert parameters.grad is not None
    return parameters.grad.numpy().copy()


def estimate_gradient(
    objective: RelaxedObjective,
    logits: torch.Tensor,
    temperature: float,
    generator: torch.Generator,
    *,
    mode: RelaxationMode = RelaxationMode.SOFT,
    n_samples: int = 1,
) -> np.ndarray:
    """The Gumbel-softmax gradient estimate, averaged over ``n_samples`` draws.

    Biased at every ``temperature > 0`` under both modes. The bias is the
    question, so it is measured against :func:`exact_expected_gradient` rather
    than assumed small. Averaging more samples reduces variance and leaves the
    bias untouched, which is why the two are reported separately.
    """
    parameters = logits.detach().clone().requires_grad_(True)
    total = torch.zeros((), dtype=torch.float64)
    for _ in range(n_samples):
        sample = gumbel_softmax(parameters, temperature, generator, mode=mode)
        total = total + objective.relaxed(sample)
    (total / n_samples).backward()  # type: ignore[no-untyped-call]

    assert parameters.grad is not None
    return parameters.grad.numpy().copy()


def anneal(start: float, end: float, steps: int, step: int) -> float:
    """Geometric temperature schedule, evaluated at one step.

    Geometric rather than linear because the relaxation's behaviour is set by
    the *ratio* of logit gaps to ``tau``, so equal multiplicative steps are
    equal steps in the thing that matters.

    Raises
    ------
    ValueError
        If either endpoint is below :data:`MINIMUM_TEMPERATURE`, ``end``
        exceeds ``start``, or ``steps`` is below 1.
    """
    if min(start, end) < MINIMUM_TEMPERATURE:
        msg = f"both endpoints must be >= {MINIMUM_TEMPERATURE}, got ({start}, {end})"
        raise ValueError(msg)
    if end > start:
        msg = f"end must not exceed start, got start={start}, end={end}"
        raise ValueError(msg)
    if steps < 1:
        msg = f"steps must be at least 1, got {steps}"
        raise ValueError(msg)
    if steps == 1:
        return start
    fraction = min(step, steps - 1) / (steps - 1)
    return float(start * (end / start) ** fraction)


def optimize(
    objective: RelaxedObjective,
    seed: int,
    *,
    temperature: float = 0.5,
    final_temperature: float | None = None,
    mode: RelaxationMode = RelaxationMode.SOFT,
    steps: int = 400,
    learning_rate: float = 0.1,
    n_samples: int = 1,
    stochastic: bool = True,
) -> RelaxedOptimum:
    """Gradient ascent on the logits, then read the configuration off.

    Parameters
    ----------
    objective : RelaxedObjective
        The discrete problem being relaxed.
    seed : int
        Seeds the initial logits and the Gumbel draws.
    temperature : float
        Fixed ``tau``, or the *starting* ``tau`` when ``final_temperature`` is
        given.
    final_temperature : float | None
        ``None`` holds ``temperature`` fixed. A value anneals geometrically to
        it over ``steps``, per :func:`anneal`. Both are supported because the
        fixed-``tau`` sweep is the measurement and annealing is the practice,
        and they do not always agree.
    mode : RelaxationMode
        Ignored when ``stochastic`` is ``False``.
    steps, learning_rate : int, float
        Adam budget.
    n_samples : int
        Gumbel draws averaged per step.
    stochastic : bool
        ``False`` runs the *deterministic* relaxation: ascend the score at
        ``softmax(logits / tau)`` with no sampling. Legitimate for any
        multilinear objective, by the identity in the module docstring --- the
        score at the marginals *is* the expected discrete score, so nothing is
        approximated away. It would not be for a higher-order interaction.

    Returns
    -------
    RelaxedOptimum
        Carrying the discrete score of the ``argmax`` configuration, which is
        the number to compare against an enumerated optimum.
    """
    generator = torch.Generator().manual_seed(seed)
    logits = 0.01 * torch.randn(
        (objective.n_sites, objective.n_states),
        generator=generator,
        dtype=torch.float64,
    )
    logits.requires_grad_(True)
    optimizer = torch.optim.Adam([logits], lr=learning_rate)

    current = temperature
    for step in range(steps):
        current = (
            temperature
            if final_temperature is None
            else anneal(temperature, final_temperature, steps, step)
        )
        optimizer.zero_grad()
        if stochastic:
            total = torch.zeros((), dtype=torch.float64)
            for _ in range(n_samples):
                sample = gumbel_softmax(logits, current, generator, mode=mode)
                total = total + objective.relaxed(sample)
            loss = -total / n_samples
        else:
            loss = -objective.relaxed(torch.softmax(logits / current, dim=1))
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()

    with torch.no_grad():
        configuration: Configuration = tuple(
            int(value) for value in logits.argmax(dim=1)
        )
        final = float(objective.relaxed(torch.softmax(logits / current, dim=1)))

    return RelaxedOptimum(
        configuration=configuration,
        score=objective.discrete(configuration),
        relaxed_score=final,
        steps=steps,
    )
