"""Where an optimization starts, as something the caller can choose.

Issue #251. `Objective.initial()` and `fit`'s `theta0` override already made
the *seam*; what every implementation put through it was one fixed constant.

**The repository has been bitten by this once.** `opt/hmm.py`'s `initial()`
records that the uniform point is a *stationary point*: with every hidden state
identical the gradient in the initial and transition parameters is exactly
zero, so an optimizer started there never moves. The fix was a fixed tilt in
that one file; `perturbed` here is that idea with the model taken out.

**And one start is known not to be enough.** `opt/testfunctions.py` keeps
Himmelblau — four equal minima — and Rastrigin, where a fit from a single start
lands in whichever cell it began in. `search/` has restarted all along:
`STATUS.md` records hill climbing reaching the enumerated optimum from 12 of 12
starts.

**Randomness enters as a generator.** `opt/hmm.py` rejected a jitter because "a
seeded jitter would make the fit depend on a second seed nobody declared" —
an argument against *seeding inside a call*, not against randomness. An
initializer taking a generator is reproducible; one seeding itself is the
defect `sim/CLAUDE.md` forbids (issue #240).

**What is not here.** An initializer that reads the data cannot live in this
module: `opt/` may import no application module, and `test_opt_objective.py`
asserts it. Those belong beside the objective they initialize ---
`opt/mixture.py`'s k-means++ (issue #262), `search/initialize.py`'s
neighbor-joining and Hadamard starts (issue #364).

**Three of these sample rather than propose.** `FromChain`, `FromAnnealing`
and `FromTempering` run `opt/hmc.py` on the objective and take a start from
where the chain went (issue #541). They read the objective and no data, so
they stay here; each reports what it spent in gradients, because a start
that costs as much as the fit it seeds is a different proposition from one
that costs 2% of it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import torch

from snakes_and_ladders.opt import hmc
from snakes_and_ladders.opt.objective import Objective
from snakes_and_ladders.opt.schedule import Schedule


@runtime_checkable
class Initializer(Protocol):
    """A source of starting points for one objective.

    One start is the degenerate case of many, which is why this returns a
    sequence rather than a tensor: a caller that wants today's behaviour asks
    for one, and the shape of the answer does not change.
    """

    def starts(self, objective: Objective) -> list[torch.Tensor]:
        """Candidate starting points, in unconstrained coordinates.

        Returns
        -------
        list[torch.Tensor]
            At least one point, each of the objective's own dimension.
        """
        ...  # pragma: no cover


class FromObjective:
    """The objective's own `initial()`, unchanged.

    The default, and today's behaviour exactly: every number `STATUS.md` pins
    was produced this way, so this has to stay reachable and stay first.
    """

    def starts(self, objective: Objective) -> list[torch.Tensor]:
        """The single point the objective nominates.

        Returns
        -------
        list[torch.Tensor]
            Exactly one start.
        """
        return [objective.initial()]


class Perturbed:
    """The objective's start, tilted by a fixed amount along each coordinate.

    Deterministic by design. It exists for a surface whose nominal start is
    *stationary* — the HMM's uniform point — where a reproducible nudge off the
    symmetry is enough and randomness would only cost a declared generator.

    The tilt alternates in sign so the perturbation does not translate every
    coordinate the same way, which on a symmetric objective would land on
    another point of the same symmetry.

    Parameters
    ----------
    magnitude : float
        Size of the tilt in unconstrained coordinates.
    """

    def __init__(self, magnitude: float = 0.1) -> None:
        if magnitude <= 0.0:
            msg = f"magnitude must be positive, got {magnitude}"
            raise ValueError(msg)
        self.magnitude = magnitude

    def starts(self, objective: Objective) -> list[torch.Tensor]:
        """One start, off the objective's own by a fixed alternating tilt.

        Returns
        -------
        list[torch.Tensor]
            Exactly one start.
        """
        base = objective.initial()
        signs = torch.tensor(
            [1.0 if index % 2 == 0 else -1.0 for index in range(base.shape[0])],
            dtype=base.dtype,
            device=base.device,
        )
        return [base + self.magnitude * signs]


class RandomRestart:
    """Several starts, drawn around the objective's own.

    For a surface with more than one local optimum, where the answer a single
    fit returns is a property of where it began. `search/CLAUDE.md`'s budget
    rule applies: `n_starts` restarts cost `n_starts` fits, so the count is
    the caller's to justify.

    Parameters
    ----------
    n_starts : int
        Points to draw, at least 1.
    scale : float
        Standard deviation of the Gaussian displacement, in unconstrained
        coordinates.
    rng : np.random.Generator
        Passed in rather than seeded here, so an ensemble gets independent
        restarts and a declared seed still determines the run
        (`sim/CLAUDE.md`, issue #240).
    include_nominal : bool
        Whether the objective's own start is the first of them. Default
        ``True``: a restart set that cannot reproduce the single-start answer
        can be worse than one fit.
    """

    def __init__(
        self,
        n_starts: int,
        scale: float,
        rng: np.random.Generator,
        include_nominal: bool = True,
    ) -> None:
        if n_starts < 1:
            msg = f"n_starts must be at least 1, got {n_starts}"
            raise ValueError(msg)
        if scale <= 0.0:
            msg = f"scale must be positive, got {scale}"
            raise ValueError(msg)
        self.n_starts = n_starts
        self.scale = scale
        self.rng = rng
        self.include_nominal = include_nominal

    def starts(self, objective: Objective) -> list[torch.Tensor]:
        """``n_starts`` points, the first being the nominal one if asked.

        Returns
        -------
        list[torch.Tensor]
            Exactly ``n_starts`` starts.
        """
        base = objective.initial()
        drawn: list[torch.Tensor] = []
        if self.include_nominal:
            drawn.append(base)
        while len(drawn) < self.n_starts:
            displacement = torch.tensor(
                self.rng.normal(scale=self.scale, size=int(base.shape[0])),
                dtype=base.dtype,
                device=base.device,
            )
            drawn.append(base + displacement)
        return drawn


class FromChain:
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


class FromAnnealing:
    """The best point a Hamiltonian chain visits while its temperature falls.

    The single-chain control :class:`FromTempering` is measured against:
    both spend gradients to leave a local optimum, and only a comparison at
    equal `force_evaluations` says whether the ladder earns its replicas.

    Parameters
    ----------
    schedule : Schedule
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
        schedule: Schedule,
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
        snakes_and_ladders.opt.hmc.Annealed
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


class FromTempering:
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
        :func:`snakes_and_ladders.opt.hmc.parallel_tempering` requires.
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
