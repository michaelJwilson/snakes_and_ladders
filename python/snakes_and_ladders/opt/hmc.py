"""Hamiltonian Monte Carlo over an :class:`~snakes_and_ladders.opt.objective.Objective`.

Every continuous result in this repository is a point estimate plus an
interval built from the observed information at the optimum --- a Gaussian
approximation to the posterior, evaluated at one point. This samples the
posterior instead, so an interval becomes a quantile rather than a curvature
estimate, and the two can be compared: where the Laplace approximation is good
they agree, and where it is not the disagreement is the finding.

The objective is read as an unnormalized negative log density. That is a
choice, not an identity, and it is the caller's to justify: for a
log-likelihood plus a proper log prior it is the posterior; for a bare
log-likelihood it is a posterior under an improper flat prior, which may not
be normalizable at all. Nothing here can check that, so nothing here pretends
to --- :func:`sample` reports the chain and the diagnostics, and what the
chain is a sample *of* is stated by whoever built the objective.

`snakes_and_ladders.opt` may import no application module and this needs none: an
`Objective` supplies an unconstrained vector and a differentiable scalar,
which is exactly the interface a gradient-based sampler wants.

**The integrator is a composition, not a procedure.** Every method here is a
sequence of second-order kick-drift-kick sub-steps differing only in their
lengths, so :class:`Integrator` carries the weights and one driver runs them
all. A higher order costs more force evaluations per step, which is why the
choice between them is a measurement at equal *evaluations* and never at
equal steps --- `search/CLAUDE.md`'s budget rule, and the reason
:meth:`Integrator.force_evaluations` exists.

**Temperature is the momentum's variance.** The tempered target
``exp(-U / T)`` is the marginal of ``exp(-(U + K) / T)``, whose momentum is
``N(0, T)``; and Hamilton's equations for ``(U + K) / T`` are the untempered
ones in rescaled time, so the *same* integrator with the *same* step serves
every temperature and only the momentum draw and the acceptance ratio change
(issue #267). That is what makes :func:`anneal` the sampler on a schedule
rather than a second sampler: at ``T = 1`` every operation is the identity
bitwise, and every chain drawn before the temperature existed is unchanged.
A quasi-Newton fit has no acceptance ratio to temper, which is why ``fit``
takes no schedule and annealing a continuous objective enters here.

See Neal (2011), "MCMC using Hamiltonian dynamics"; Yoshida (1990) for the
fourth-order composition and Suzuki (1991) for why its middle coefficient
must be negative; Nocedal & Wright for the symplectic structure; Kirkpatrick,
Gelatt & Vecchi (1983) for annealing.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from snakes_and_ladders.opt.objective import Objective
from snakes_and_ladders.opt.schedule import Schedule

DEFAULT_STEPS = 20


@dataclass(frozen=True)
class WithGaussianPrior:
    """An objective plus an isotropic Gaussian log prior, so it has a posterior.

    A bare negative log-likelihood read as a log density is a posterior under
    an improper flat prior, which for most models is not normalizable --- so
    a chain against it is sampling from nothing well defined, and no
    diagnostic in :func:`sample` can tell. Adding a proper prior fixes that,
    and stating it here rather than inside the sampler keeps it the caller's
    declaration: the sampler still just minimizes what it is handed.

    Isotropic and centred on zero *in unconstrained coordinates*, which is
    weakly informative rather than uninformative --- on a log-simplex
    coordinate it pulls toward the uniform distribution, and on a coupling
    toward zero. That is a modelling choice, and any interval reported from a
    chain against it inherits it.

    Parameters
    ----------
    objective : Objective
        The negative log-likelihood being given a prior.
    scale : float
        Prior standard deviation on every unconstrained coordinate.
    """

    objective: Objective
    scale: float

    def __post_init__(self) -> None:
        if self.scale <= 0.0:
            msg = f"prior scale must be positive, got {self.scale}"
            raise ValueError(msg)

    def initial(self) -> torch.Tensor:
        return self.objective.initial()

    def constrain(self, theta: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return self.objective.constrain(theta)

    def theta_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.objective.theta_from(named)

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        penalty = (theta * theta).sum() / (2.0 * self.scale**2)
        return self.objective(theta) + penalty


@dataclass(frozen=True)
class HmcChain:
    """A chain, and what it cost to get it.

    Parameters
    ----------
    theta : torch.Tensor
        Draws in unconstrained coordinates, shape ``(n_samples, dimension)``.
    acceptance_rate : float
        Fraction of proposals accepted. Hamiltonian dynamics conserves energy
        exactly, so a correct implementation with a small step size accepts
        nearly everything; a rate near zero means the integrator is diverging
        rather than that the target is hard.
    energy_error : torch.Tensor
        ``|H(proposal) - H(current)|`` per proposal. The diagnostic that
        distinguishes a step size too large from a bug: the first grows
        smoothly with the step size, the second does not.
    """

    theta: torch.Tensor
    acceptance_rate: float
    energy_error: torch.Tensor


#: The cube root that Yoshida's fourth-order composition is built from.
_CUBE_ROOT_OF_TWO = 2.0 ** (1.0 / 3.0)

#: Yoshida's (1990) fourth-order weights: three second-order sub-steps of
#: lengths ``w1``, ``w0``, ``w1`` with ``2 * w1 + w0 == 1``.
#:
#: **``w0`` is negative --- the middle sub-step runs backwards in time.** That
#: is not a sign error and not avoidable: no composition of a second-order
#: symmetric method reaches fourth order with positive coefficients (Suzuki,
#: 1991). The trajectory is therefore non-monotone in time, which is worth
#: knowing before plotting one and concluding the integrator is broken.
YOSHIDA_WEIGHTS = (
    1.0 / (2.0 - _CUBE_ROOT_OF_TWO),
    -_CUBE_ROOT_OF_TWO / (2.0 - _CUBE_ROOT_OF_TWO),
    1.0 / (2.0 - _CUBE_ROOT_OF_TWO),
)

#: The trivial composition: one second-order sub-step of full length.
LEAPFROG_WEIGHTS = (1.0,)


@dataclass(frozen=True)
class Integrator:
    """A symplectic integrator, as the composition of sub-steps it is.

    Every method here is a composition of the second-order kick-drift-kick
    step, differing only in the sub-step lengths. Carrying the weights rather
    than a procedure buys three things a bare function does not:

    * **one implementation.** The kicks between adjacent sub-steps merge, so a
      hand-written composition is the same arithmetic with more places to put
      a wrong coefficient. :func:`leapfrog` is this object at
      ``(1.0,)`` and reproduces the previous implementation exactly;
    * **a declared cost.** A comparison between integrators is only meaningful
      at equal force evaluations, and :meth:`force_evaluations` is where that
      count comes from rather than a caller's arithmetic;
    * **a declared order**, so a test can assert the one it was built for
      instead of the one it happens to achieve.

    Parameters
    ----------
    name : str
        For error messages and reports.
    weights : tuple[float, ...]
        Sub-step lengths, summing to 1. Reversibility requires the sequence be
        a palindrome; both compositions here are.
    order : int
        The order of accuracy of the energy error in the step size.

    Raises
    ------
    ValueError
        If the weights do not sum to 1, or are not a palindrome. The first
        integrates the wrong amount of time while leaving reversibility
        intact, which is the one arithmetic slip a reversibility check does
        not catch.
    """

    name: str
    weights: tuple[float, ...]
    order: int

    def __post_init__(self) -> None:
        total = math.fsum(self.weights)
        if abs(total - 1.0) > 1e-12:
            msg = (
                f"{self.name}: sub-step weights sum to {total!r}, expected 1.0; "
                f"a composition that does not integrates the wrong interval "
                f"while remaining perfectly reversible"
            )
            raise ValueError(msg)
        if self.weights != tuple(reversed(self.weights)):
            msg = f"{self.name}: weights must be a palindrome to be reversible"
            raise ValueError(msg)

    def force_evaluations(self, n_steps: int) -> int:
        """Gradient evaluations one trajectory of ``n_steps`` costs.

        ``len(weights) * n_steps + 1``: the kicks at the join between two
        sub-steps merge into one, so a composition of ``s`` sub-steps costs
        ``s`` gradients per step rather than ``2 s``.
        """
        return len(self.weights) * n_steps + 1

    def __call__(
        self,
        objective: Objective,
        theta: torch.Tensor,
        momentum: torch.Tensor,
        step_size: float,
        n_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Integrate Hamiltonian dynamics over ``n_steps`` steps.

        Separated from :func:`sample` because its two defining properties ---
        reversibility and its order of accuracy --- are exact statements
        testable without any sampling, and they are where an error actually
        localizes. A distributional test says the chain is wrong; these say
        which half.

        Parameters
        ----------
        objective : Objective
            Read as a negative log density, so its gradient is the force.
        theta, momentum : torch.Tensor
            Position and momentum, both 1-D of the same length.
        step_size : float
            Integrator step. Energy error grows as its ``order`` power.
        n_steps : int
            Steps per trajectory.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Position and momentum after ``n_steps``.
        """
        position = theta.detach().clone()
        velocity = momentum.detach().clone()
        kicks, drifts = _coefficients(self.weights, n_steps)

        velocity = velocity - kicks[0] * step_size * _gradient(objective, position)
        for drift, kick in zip(drifts, kicks[1:], strict=True):
            position = position + drift * step_size * velocity
            velocity = velocity - kick * step_size * _gradient(objective, position)
        return position, velocity


def _coefficients(
    weights: tuple[float, ...], n_steps: int
) -> tuple[list[float], list[float]]:
    """Kick and drift coefficients for ``n_steps`` of a composition.

    Each sub-step is a kick-drift-kick of half, full, half its length, and the
    trailing half-kick of one sub-step sits at the same position as the
    leading half-kick of the next, so the two merge. That leaves one more kick
    than there are sub-steps, which is where
    :meth:`Integrator.force_evaluations` comes from.
    """
    sub_steps = list(weights) * n_steps
    kicks = [0.5 * sub_steps[0]]
    drifts = []
    for current, following in itertools.pairwise(sub_steps):
        drifts.append(current)
        kicks.append(0.5 * (current + following))
    drifts.append(sub_steps[-1])
    kicks.append(0.5 * sub_steps[-1])
    return kicks, drifts


#: The second-order kick-drift-kick method. The default everywhere, and the
#: reference every other integrator is checked against.
leapfrog = Integrator(name="leapfrog", weights=LEAPFROG_WEIGHTS, order=2)

#: Yoshida's fourth-order triple jump. Three force evaluations per step
#: against leapfrog's one, so whether it pays is a measurement at equal
#: evaluations rather than a consequence of the higher order.
yoshida = Integrator(name="yoshida", weights=YOSHIDA_WEIGHTS, order=4)


def hamiltonian(
    objective: Objective, theta: torch.Tensor, momentum: torch.Tensor
) -> float:
    """``U(theta) + K(momentum)``, the quantity the integrator conserves."""
    potential = float(objective(theta.detach()))
    kinetic = 0.5 * float((momentum * momentum).sum())
    return potential + kinetic


def sample(
    objective: Objective,
    generator: torch.Generator,
    n_samples: int,
    *,
    step_size: float,
    n_steps: int = DEFAULT_STEPS,
    theta0: torch.Tensor | None = None,
    burn_in: int = 0,
    integrator: Integrator = leapfrog,
    temperature: float = 1.0,
) -> HmcChain:
    """Draw ``n_samples`` from the density ``exp(-objective / temperature)``.

    Parameters
    ----------
    objective : Objective
        Read as an unnormalized negative log density.
    generator : torch.Generator
        The stream every momentum and acceptance draw comes from, passed in
        rather than seeded here (`sim/CLAUDE.md`): a chain is reproducible
        from ``torch.Generator().manual_seed(seed)`` at the call site, and two
        chains drawn from one generator are two chains.
    n_samples : int
        Draws recorded after burn-in.
    step_size : float
        Leapfrog step. Required rather than defaulted: it is the one
        parameter whose right value depends on the target's scale, and a
        default would be wrong silently. Adaptation is deliberately absent
        --- see the module note below.
    n_steps : int
        Leapfrog steps per proposal.
    theta0 : torch.Tensor | None
        Starting point; ``objective.initial()`` when omitted.
    burn_in : int
        Draws discarded before recording.
    integrator : Integrator
        The symplectic method. ``leapfrog`` by default, which is what every
        chain in this repository was drawn with. A higher-order method takes
        more force evaluations per step, so it is worth choosing only against
        a comparison at equal evaluations rather than at equal steps ---
        :meth:`Integrator.force_evaluations` is what makes that comparison
        possible.
    temperature : float
        The chain targets ``exp(-objective / temperature)``; 1 is the
        objective as declared. Whether that is a tempered *energy* or a power
        posterior is the caller's to say (`snakes_and_ladders.opt.schedule`).

    Returns
    -------
    HmcChain
        The draws, the acceptance rate, and the per-proposal energy error.

    Raises
    ------
    ValueError
        If ``step_size`` or ``temperature`` is not positive, or ``n_steps``
        is below 1. A zero-length trajectory proposes the current point every
        time, which accepts at rate 1 and samples nothing --- a chain that
        looks healthy by every diagnostic and has not moved.
    """
    _check_trajectory(step_size, n_steps)
    if not temperature > 0.0:
        msg = f"temperature must be positive, got {temperature}"
        raise ValueError(msg)

    position = _start(objective, theta0)

    draws = torch.empty((n_samples, position.shape[0]), dtype=torch.float64)
    errors = torch.empty(n_samples + burn_in, dtype=torch.float64)
    accepted = 0

    for index in range(n_samples + burn_in):
        position, error, was_accepted = _transition(
            objective, position, temperature, generator, step_size, n_steps, integrator
        )
        errors[index] = error
        if index >= burn_in:
            accepted += was_accepted
            draws[index - burn_in] = position

    return HmcChain(
        theta=draws,
        acceptance_rate=accepted / n_samples if n_samples else 0.0,
        energy_error=errors[burn_in:],
    )


@dataclass(frozen=True)
class Annealed:
    """What one annealing run found, and what it cost.

    Parameters
    ----------
    theta : torch.Tensor
        The lowest-valued point visited, in unconstrained coordinates. The
        *best* rather than the last: the final proposals run cold but not at
        zero, so the chain can leave the best point it found.
    value : float
        The objective there.
    final : torch.Tensor
        Where the chain ended.
    acceptance_rate : float
        Over the whole schedule. Near zero at the cold end is the symptom of
        a step too large for the final temperature.
    force_evaluations : int
        Gradients spent, so the run is comparable to any other optimizer at
        equal evaluations.
    """

    theta: torch.Tensor
    value: float
    final: torch.Tensor
    acceptance_rate: float
    force_evaluations: int


def anneal(
    objective: Objective,
    schedule: Schedule,
    generator: torch.Generator,
    *,
    step_size: float,
    n_steps: int = DEFAULT_STEPS,
    theta0: torch.Tensor | None = None,
    integrator: Integrator = leapfrog,
) -> Annealed:
    """Simulated annealing with Hamiltonian proposals: :func:`sample` on a schedule.

    One proposal per schedule step at that step's temperature, tracking the
    lowest objective seen. The transition at each step is exactly the one
    :func:`sample` runs at a constant temperature, so a constant schedule
    reproduces a chain draw for draw from generators seeded alike; what
    annealing adds is that the temperature falls, and what it buys is measured
    against the alternatives at equal force evaluations and never assumed.

    Parameters
    ----------
    objective : Objective
        What to minimize. Read as an energy, so ``T`` is physical; a negative
        log-likelihood here is a power posterior and the caller should know
        which they meant.
    schedule : Schedule
        Temperature per proposal. Its length is the budget in proposals;
        ``force_evaluations`` on the result is the budget in gradients.
    generator : torch.Generator
        As :func:`sample`.
    step_size, n_steps, theta0, integrator
        As :func:`sample`. The step needs no rescaling with temperature ---
        see the module note --- but a step that is stable at the hot end can
        still reject at the cold end, which the acceptance rate reports.

    Returns
    -------
    Annealed
    """
    _check_trajectory(step_size, n_steps)
    position = _start(objective, theta0)

    best, best_value = position.clone(), float(objective(position))
    accepted = 0
    for step in range(schedule.n_steps):
        position, _, was_accepted = _transition(
            objective,
            position,
            schedule(step),
            generator,
            step_size,
            n_steps,
            integrator,
        )
        accepted += was_accepted
        value = float(objective(position))
        if value < best_value:
            best, best_value = position.clone(), value
    return Annealed(
        theta=best,
        value=best_value,
        final=position,
        acceptance_rate=accepted / schedule.n_steps,
        force_evaluations=schedule.n_steps * integrator.force_evaluations(n_steps),
    )


@dataclass(frozen=True)
class Tempered:
    """What one parallel-tempering run found, and what it cost.

    Parameters
    ----------
    theta : torch.Tensor
        The lowest-valued point visited at any temperature.
    value : float
        The objective there.
    positions : torch.Tensor
        Every replica after every round, shape ``(n_rounds, n_replicas,
        dimension)``; replica ``r`` sits at ``temperatures[r]`` throughout,
        because an exchange swaps *positions* between temperatures rather
        than moving a chain along the ladder. Recorded so a replica's
        marginal can be checked against the tempered target, as the discrete
        version's are.
    acceptance_rate : torch.Tensor
        Fraction of Hamiltonian proposals accepted per replica, shape
        ``(n_replicas,)``.
    swap_acceptance : torch.Tensor
        Fraction of proposed exchanges accepted per adjacent pair, shape
        ``(n_replicas - 1,)``. Near zero means the ladder has a gap no
        position crosses and the replicas are independent chains; near one
        means two temperatures are close enough that one is redundant.
    force_evaluations : int
        Gradients spent over every replica, so the run is comparable to any
        other optimizer at equal evaluations.
    """

    theta: torch.Tensor
    value: float
    positions: torch.Tensor
    acceptance_rate: torch.Tensor
    swap_acceptance: torch.Tensor
    force_evaluations: int


def _swap_log_ratio(
    temperature_cold: float, temperature_hot: float, value_cold: float, value_hot: float
) -> float:
    """Log acceptance of exchanging the positions at two temperatures.

    The joint target is the product of the tempered marginals, so the ratio
    is ``(1/T_cold - 1/T_hot)(U_cold - U_hot)``: an exchange that hands the
    colder replica the lower value is always accepted. The same expression
    :func:`snakes_and_ladders.search.potts_mcmc.parallel_tempering` accepts
    on, with the objective where that has an energy.
    """
    return (1.0 / temperature_cold - 1.0 / temperature_hot) * (value_cold - value_hot)


def parallel_tempering(
    objective: Objective,
    temperatures: tuple[float, ...],
    generator: torch.Generator,
    n_rounds: int,
    *,
    step_size: float,
    n_steps: int = DEFAULT_STEPS,
    theta0: torch.Tensor | None = None,
    integrator: Integrator = leapfrog,
) -> Tempered:
    """Replicas at fixed temperatures, exchanging positions by Metropolis.

    Each round is one :func:`sample` transition per replica at its own
    temperature, then every adjacent pair proposes to exchange positions and
    accepts on :func:`_swap_log_ratio`. The hot replicas cross barriers the
    cold one cannot, and an exchange carries what they find down the ladder
    (Swendsen & Wang, 1986; Geyer, 1991; Earl & Deem, 2005). The continuous
    counterpart of :func:`snakes_and_ladders.search.potts_mcmc.parallel_tempering`,
    which ``opt`` cannot import and which moves spins rather than a vector.

    **The replicas must not share a stream and must be reproducible from one
    generator.** The caller's ``generator`` draws a seed per replica and then
    only the exchange uniforms, so the replicas are independent streams and
    one generator state reproduces the run. Sharing one stream would correlate the
    replicas, which is the whole point lost while every diagnostic looks
    healthy.

    Parameters
    ----------
    objective : Objective
        What to minimize. Read as an energy, so ``T`` is physical; a negative
        log-likelihood here is a power posterior and the caller should know
        which they meant.
    temperatures : tuple[float, ...]
        The ladder, coldest first; at least two, all positive, strictly
        increasing so that adjacent pairs are the ones that exchange.
    generator : torch.Generator
        The parent stream, seeded by the caller (issue #337); it draws the
        replicas' seeds and the exchange uniforms.
    n_rounds : int
        Transitions per replica, at least one. The budget in proposals is
        ``n_rounds * len(temperatures)``; ``force_evaluations`` on the result
        is the budget in gradients.
    step_size, n_steps, theta0, integrator
        As :func:`sample`; every replica starts at ``theta0``.

    Returns
    -------
    Tempered

    Raises
    ------
    ValueError
        If fewer than two temperatures are given --- a ladder of one has
        nothing to exchange and is :func:`sample` --- if any is not positive
        or the ladder is not increasing, or if ``n_rounds`` is below one.
    """
    _check_trajectory(step_size, n_steps)
    if len(temperatures) < 2:
        msg = (
            f"parallel tempering needs at least two temperatures, got "
            f"{len(temperatures)}: a ladder of one has nothing to exchange"
        )
        raise ValueError(msg)
    for cold, hot in itertools.pairwise(temperatures):
        if not 0.0 < cold < hot:
            msg = (
                f"temperatures must be positive and increasing, coldest first, "
                f"got {temperatures}"
            )
            raise ValueError(msg)
    if n_rounds < 1:
        msg = f"n_rounds must be at least 1, got {n_rounds}"
        raise ValueError(msg)

    parent = generator
    n_replicas = len(temperatures)
    children = [
        torch.Generator().manual_seed(int(child))
        for child in torch.randint(0, 2**31 - 1, (n_replicas,), generator=parent)
    ]
    start = _start(objective, theta0)
    positions = [start.clone() for _ in range(n_replicas)]
    values = [float(objective(start))] * n_replicas
    best, best_value = start.clone(), values[0]
    accepted = torch.zeros(n_replicas, dtype=torch.float64)
    swapped = torch.zeros(n_replicas - 1, dtype=torch.float64)
    recorded = torch.empty((n_rounds, n_replicas, start.shape[0]), dtype=torch.float64)

    for round_index in range(n_rounds):
        for replica in range(n_replicas):
            positions[replica], _, was_accepted = _transition(
                objective,
                positions[replica],
                temperatures[replica],
                children[replica],
                step_size,
                n_steps,
                integrator,
            )
            accepted[replica] += was_accepted
            values[replica] = float(objective(positions[replica]))
        for pair in range(n_replicas - 1):
            log_ratio = _swap_log_ratio(
                temperatures[pair],
                temperatures[pair + 1],
                values[pair],
                values[pair + 1],
            )
            uniform = float(torch.rand(1, generator=parent))
            if log_ratio >= 0.0 or uniform < math.exp(log_ratio):
                swapped[pair] += 1
                positions[pair], positions[pair + 1] = (
                    positions[pair + 1],
                    positions[pair],
                )
                values[pair], values[pair + 1] = values[pair + 1], values[pair]
        lowest = min(range(n_replicas), key=values.__getitem__)
        if values[lowest] < best_value:
            best, best_value = positions[lowest].clone(), values[lowest]
        recorded[round_index] = torch.stack(positions)

    return Tempered(
        theta=best,
        value=best_value,
        positions=recorded,
        acceptance_rate=accepted / n_rounds,
        swap_acceptance=swapped / n_rounds,
        force_evaluations=n_rounds * n_replicas * integrator.force_evaluations(n_steps),
    )


def _check_trajectory(step_size: float, n_steps: int) -> None:
    if step_size <= 0.0:
        msg = f"step_size must be positive, got {step_size}"
        raise ValueError(msg)
    if n_steps < 1:
        msg = (
            f"n_steps must be at least 1, got {n_steps}: a zero-length "
            "trajectory proposes the current point and accepts at rate 1, "
            "which looks healthy and samples nothing"
        )
        raise ValueError(msg)


def _start(objective: Objective, theta0: torch.Tensor | None) -> torch.Tensor:
    return (
        objective.initial().detach().clone()
        if theta0 is None
        else theta0.detach().clone()
    ).to(torch.float64)


def _transition(
    objective: Objective,
    position: torch.Tensor,
    temperature: float,
    generator: torch.Generator,
    step_size: float,
    n_steps: int,
    integrator: Integrator,
) -> tuple[torch.Tensor, float, int]:
    """One Metropolis step with a Hamiltonian proposal at ``temperature``.

    Momentum is drawn with variance ``temperature`` and the acceptance ratio
    divides the energy difference by it; the integrator itself is untempered.
    At ``temperature = 1.0`` both are the identity bitwise, so this *is* the
    untempered transition and not an approximation of it.

    Returns
    -------
    tuple[torch.Tensor, float, int]
        The new position, the absolute energy error of the proposal, and 1
        if it was accepted.
    """
    momentum = torch.randn(
        position.shape, generator=generator, dtype=torch.float64
    ) * math.sqrt(temperature)
    current = hamiltonian(objective, position, momentum)

    proposal, proposed_momentum = integrator(
        objective, position, momentum, step_size, n_steps
    )
    # Negating the momentum makes the proposal symmetric, which is what
    # leaves the acceptance ratio as the energy difference alone. It has
    # no effect on the next iteration, where the momentum is redrawn.
    proposed = hamiltonian(objective, proposal, -proposed_momentum)

    error = abs(proposed - current)
    uniform = float(torch.rand(1, generator=generator))
    if uniform < float(torch.exp(torch.tensor((current - proposed) / temperature))):
        return proposal, error, 1
    return position, error, 0


def _gradient(objective: Objective, theta: torch.Tensor) -> torch.Tensor:
    """``dU/dtheta``, by autograd through the objective."""
    point = theta.detach().clone().requires_grad_(True)
    value = objective(point)
    (grad,) = torch.autograd.grad(value, point)
    return grad.detach()
