"""Initializers built from a sampler: a chain, an annealing run, a tempering run.

Three of :mod:`snakes_and_ladders.opt.initialize`'s starting points are a
sampler's output --- the last draw of an HMC chain, the best state of an
annealing run, the cold replica of a tempering run --- and they lived beside
the ones that are not, which put a sampler import in ``opt/`` against its
own sentence (issue #830). They implement the same
:class:`~snakes_and_ladders.opt.initialize.Initializer` seam and are reached
by the same callers; only the directory moved, and every start is bitwise
what it was.
"""

from __future__ import annotations

import torch

from snakes_and_ladders.opt.initialize import Initializer
from snakes_and_ladders.opt.objective import Objective
from snakes_and_ladders.sample import hmc
from snakes_and_ladders.sample.schedule import TempSchedule


class FromChain(Initializer):
    """A start drawn from a short Hamiltonian chain on the objective itself.

    The three above propose a point; this one samples the surface. Where the
    objective is a negative log-likelihood the chain targets its posterior
    under a flat prior, which `opt/CLAUDE.md` records is improper for most
    models --- so what is taken from it is a *start*, never a posterior
    summary.

    **Label switching is the hazard, not the step size.** A mixture
    likelihood is invariant under permuting its components, so a chain that
    crosses between modes returns the same fit under arbitrary names.
    :meth:`starts` therefore returns the draws unchanged and the caller
    canonicalizes; :meth:`chain` hands back the diagnostics, because a seed
    drawn from a chain that has not mixed is a random restart with a longer
    bill.

    Parameters
    ----------
    n_samples : int
        Draws kept, at least 1. The last is the start.
    step_size : float
        Leapfrog step, positive.
    generator : torch.Generator
        The stream, passed in rather than seeded here (issue #337).
    n_steps : int
        Leapfrog steps per proposal.
    burn_in : int
        Proposals discarded before the kept draws.

    Raises
    ------
    ValueError
        If fewer than one draw is asked for.
    """

    def __init__(
        self,
        n_samples: int,
        step_size: float,
        generator: torch.Generator,
        n_steps: int = hmc.DEFAULT_STEPS,
        burn_in: int = 0,
    ) -> None:
        if n_samples < 1:
            msg = f"n_samples must be at least 1, got {n_samples}"
            raise ValueError(msg)
        self.n_samples = n_samples
        self.step_size = step_size
        self.generator = generator
        self.n_steps = n_steps
        self.burn_in = burn_in

    def chain(self, objective: Objective) -> hmc.HmcChain:
        """The chain itself: the draws, the acceptance rate and what it cost.

        Returns
        -------
        HmcChain
        """
        return hmc.sample(
            objective,
            self.generator,
            self.n_samples,
            step_size=self.step_size,
            n_steps=self.n_steps,
            burn_in=self.burn_in,
        )

    def starts(self, objective: Objective) -> list[torch.Tensor]:
        """The chain's draws, in the order drawn.

        Returns
        -------
        list[torch.Tensor]
            ``n_samples`` points.
        """
        return list(self.chain(objective).theta)


class FromAnnealing(Initializer):
    """The best point a Hamiltonian chain visits while its temperature falls.

    The single-chain control :class:`FromTempering` is measured against:
    both spend gradients to leave a local optimum, and only a comparison at
    equal `force_evaluations` says whether the ladder earns its replicas.

    Parameters
    ----------
    schedule : TempSchedule
        Temperature per proposal; its length is the budget in proposals.
    step_size : float
        Leapfrog step, positive.
    generator : torch.Generator
        The stream, passed in.
    n_steps : int
        Leapfrog steps per proposal.
    """

    def __init__(
        self,
        schedule: TempSchedule,
        step_size: float,
        generator: torch.Generator,
        n_steps: int = hmc.DEFAULT_STEPS,
    ) -> None:
        self.schedule = schedule
        self.step_size = step_size
        self.generator = generator
        self.n_steps = n_steps

    def run(self, objective: Objective) -> hmc.Annealed:
        """The annealing run: its best point, the acceptance rate and its cost.

        Returns
        -------
        snakes_and_ladders.sample.hmc.Annealed
        """
        return hmc.anneal(
            objective,
            self.schedule,
            self.generator,
            step_size=self.step_size,
            n_steps=self.n_steps,
        )

    def starts(self, objective: Objective) -> list[torch.Tensor]:
        """The lowest-valued point visited.

        Returns
        -------
        list[torch.Tensor]
            Exactly one start.
        """
        return [self.run(objective).theta]


class FromTempering(Initializer):
    """The best point a temperature ladder visits, exchanging positions.

    The hot replicas cross barriers the cold one cannot, so this escapes a
    local optimum by construction where the starts above escape it by a
    spread. It is also the most expensive initializer here: its cost is
    ``n_rounds`` times the ladder's length in proposals, and a caller that
    does not report that against the fit it seeds has hidden the price.

    Parameters
    ----------
    temperatures : tuple[float, ...]
        The ladder, coldest first, as
        :func:`snakes_and_ladders.sample.hmc.parallel_tempering` requires.
    n_rounds : int
        Transitions per replica, at least one.
    step_size : float
        Leapfrog step, positive.
    generator : torch.Generator
        The parent stream, passed in.
    n_steps : int
        Leapfrog steps per proposal.
    """

    def __init__(
        self,
        temperatures: tuple[float, ...],
        n_rounds: int,
        step_size: float,
        generator: torch.Generator,
        n_steps: int = hmc.DEFAULT_STEPS,
    ) -> None:
        self.temperatures = temperatures
        self.n_rounds = n_rounds
        self.step_size = step_size
        self.generator = generator
        self.n_steps = n_steps

    def run(self, objective: Objective) -> hmc.Tempered:
        """The tempering run: its best point, both acceptance rates and its cost.

        Returns
        -------
        Tempered
        """
        return hmc.parallel_tempering(
            objective,
            self.temperatures,
            self.generator,
            self.n_rounds,
            step_size=self.step_size,
            n_steps=self.n_steps,
        )

    def starts(self, objective: Objective) -> list[torch.Tensor]:
        """The lowest-valued point visited at any temperature.

        Returns
        -------
        list[torch.Tensor]
            Exactly one start.
        """
        return [self.run(objective).theta]
