"""The emission M step as a hook, and one optimizer that fills it (issue #1171).

Both EM drivers --- :func:`sal.opt.hmm.baum_welch_family` and
:func:`sal.opt.emission_mixture.expectation_maximization` --- hand the
family's :meth:`~sal.emissions.EmissionFamily.reestimate` the observations
and the E step's posterior, and read back a
:class:`~sal.emissions.Reestimate`. An :class:`MStep` takes the same
arguments and returns the same record, so a family with no closed form, or a
fit holding some parameters fixed, enters either loop without a second
driver.

**The objective is the expected complete-data log-likelihood**, the emission
term of EM's ``Q``: ``sum_i sum_k posterior[i, k] log p_k(x_i)``. The initial
distribution, the transitions and the mixing weights are closed-form in
every model here and stay with the driver. :class:`LbfgsMStep` maximizes the
term by :func:`sal.opt.fit.fit`, over free coordinates read through each
parameter's declared :class:`~sal.emissions.Domain` (issue #1164), and holds
named parameters at their current values through
:class:`~sal.opt.objective.Restricted` (issue #1168).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol

import torch

from sal.emissions import EmissionFamily, Reestimate
from sal.opt.constrain import domain_blocks
from sal.opt.fit import fit
from sal.opt.objective import Restricted, coordinates, value_and_gradient


class MStep(Protocol):
    """An emission M step the EM drivers take in place of the family's own (issue #1171).

    Called once per iteration with the current family, the observations as
    the driver holds them, the E step's posterior as probabilities --- shape
    ``(n_sequences, length, n_states)`` from Baum-Welch, zero at padded
    positions, and ``(n_samples, n_states)`` from a mixture --- and the
    covariate the E step scored against, or ``None``. It returns what
    :meth:`~sal.emissions.EmissionFamily.reestimate` returns. ``None`` in a
    driver's ``m_step`` is the family's ``reestimate``, unchanged.
    """

    def __call__(
        self,
        components: EmissionFamily,
        observations: torch.Tensor,
        posterior: torch.Tensor,
        covariate: torch.Tensor | None,
    ) -> Reestimate[EmissionFamily]:
        """The re-estimated family and what its solve reports."""
        ...


def expected_complete_log_likelihood(
    components: EmissionFamily,
    observations: torch.Tensor,
    posterior: torch.Tensor,
    covariate: torch.Tensor | None = None,
) -> float:
    """``sum_i sum_k posterior[i, k] log p_k(x_i)``: the emission term an M step maximizes.

    A position with zero posterior weight on a state adds nothing, whatever
    that state scores there: a padded position, or a state the E step left
    no data.

    Returns
    -------
    float
    """
    with torch.no_grad():
        return float(_expected(components, observations, posterior, covariate))


def _expected(
    components: EmissionFamily,
    observations: torch.Tensor,
    posterior: torch.Tensor,
    covariate: torch.Tensor | None,
) -> torch.Tensor:
    """:func:`expected_complete_log_likelihood` as a tensor, differentiable in the family."""
    scored = components.log_density(observations, covariate=covariate)
    weighted = torch.where(
        posterior > 0.0, posterior * scored, torch.zeros_like(scored)
    )
    return weighted.sum()


class ExpectedCompleteObjective:
    """The negative expected complete-data log-likelihood over a family's free coordinates.

    An :class:`~sal.opt.objective.Objective` with declared blocks: ``theta``
    is each named parameter's free coordinates under its declared domain,
    laid end to end in :meth:`~sal.emissions.EmissionFamily.named_parameters`
    order (:func:`~sal.opt.constrain.domain_blocks`), and
    :meth:`initial` is the family as given.

    Parameters
    ----------
    components : EmissionFamily
        The family whose parameters vary; its constants are kept.
    observations, posterior, covariate : torch.Tensor
        What the M step is handed, as :class:`MStep` states it.
    """

    def __init__(
        self,
        components: EmissionFamily,
        observations: torch.Tensor,
        posterior: torch.Tensor,
        covariate: torch.Tensor | None,
    ) -> None:
        self.components = components
        self._observations = observations
        self._posterior = posterior
        self._covariate = covariate
        named = components.named_parameters()
        self._layout = domain_blocks(named, components.parameter_domains(), 0)
        self._start = torch.cat(
            [block.free_of(named[name]) for name, block in self._layout.items()]
        )

    @property
    def blocks(self) -> Mapping[str, slice]:
        """Each named parameter's coordinates in ``theta``."""
        return {
            name: slice(block.offset, block.stop)
            for name, block in self._layout.items()
        }

    def initial(self) -> torch.Tensor:
        """The free coordinates of the family as given."""
        return self._start.clone()

    def constrain(self, theta: torch.Tensor) -> Mapping[str, torch.Tensor]:
        """Each named parameter, read through its declared domain."""
        return {name: block.read(theta) for name, block in self._layout.items()}

    def theta_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """The free coordinates of ``named``."""
        return torch.cat(
            [block.free_of(named[name]) for name, block in self._layout.items()]
        )

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        """``-Q``, to minimize."""
        family = self.components.with_parameters(self.constrain(theta))
        return -_expected(family, self._observations, self._posterior, self._covariate)


@dataclass(frozen=True)
class LbfgsMStep:
    """An :class:`MStep` that maximizes the expected complete-data log-likelihood by L-BFGS.

    Starts at the family it is handed, so a strong-Wolfe line search accepts
    only steps that do not lower the objective, and finishes with at most
    ``polish`` Newton steps on the varied coordinates, where the line search
    can no longer see a gain. A solve that ends below its start returns the
    start instead. On a
    closed-form family it reaches the closed-form M step
    (``tests/regression/opt/test_opt_m_step.py`` pins it within 1e-8).

    Parameters
    ----------
    held : tuple[str, ...]
        Names of :meth:`~sal.emissions.EmissionFamily.named_parameters` held
        at their current values; ``()`` varies every parameter.
    max_iterations : int
        :func:`~sal.opt.fit.fit`'s budget of outer L-BFGS steps.
    tolerance : float
        The threshold on ``max|grad| / max(1, |Q|)``, :func:`~sal.opt.fit.fit`'s
        measure. An unconverged solve is reported in ``converged``, and the
        drivers refuse it.
    handoff : float
        Where L-BFGS hands over to Newton steps, on the same measure; at
        1e-8 a line search still sees the gain of a step.
    polish : int
        Most Newton steps taken after L-BFGS where it stops short of
        ``tolerance``; each is kept only if it shrinks the gradient.
    """

    held: tuple[str, ...] = ()
    max_iterations: int = 100
    tolerance: float = 1e-10
    handoff: float = 1e-8
    polish: int = 5

    def __call__(
        self,
        components: EmissionFamily,
        observations: torch.Tensor,
        posterior: torch.Tensor,
        covariate: torch.Tensor | None,
    ) -> Reestimate[EmissionFamily]:
        """One L-BFGS solve from ``components``, then Newton steps; see the class docstring."""
        objective = ExpectedCompleteObjective(
            components, observations, posterior, covariate
        )
        varied = _varied(objective.blocks, self.held)
        if not varied:
            return Reestimate(components)
        start = objective.initial()
        restricted = Restricted(objective, start, coordinates(objective, varied))
        solved = fit(
            restricted,
            max_iterations=self.max_iterations,
            tolerance=max(self.handoff, self.tolerance),
        )
        theta = solved.theta
        iterations = solved.iterations
        value, gradient = value_and_gradient(restricted, theta)
        converged = _relative(value, gradient) <= self.tolerance
        # The line search compares values, and within sqrt(eps) of the optimum
        # a step's gain is below the value's rounding, so L-BFGS stalls there:
        # at 1e-9 of the closed-form mean on 200 Gaussian draws. A Newton step
        # reads only derivatives, so it goes on to the gradient's own rounding.
        for _ in range(self.polish if not converged else 0):
            hessian = torch.autograd.functional.hessian(restricted, theta)  # type: ignore[no-untyped-call]
            stepped = theta - torch.linalg.solve(hessian, gradient)
            stepped_value, stepped_gradient = value_and_gradient(restricted, stepped)
            if not bool(stepped_gradient.abs().max() < gradient.abs().max()):
                break
            theta, value, gradient = stepped, stepped_value, stepped_gradient
            iterations += 1
            if _relative(value, gradient) <= self.tolerance:
                converged = True
                break
        full = restricted.embed(theta)
        with torch.no_grad():
            if bool(objective(full) > objective(start)):
                full = start
        named = {
            name: value.detach() for name, value in objective.constrain(full).items()
        }
        mass = posterior.reshape(-1, posterior.shape[-1]).sum(dim=0)
        return Reestimate(
            components.with_parameters(named),
            converged=converged,
            iterations=iterations,
            residual=float(gradient.abs().max()) / max(float(posterior.sum()), 1.0),
            frozen=tuple(int(k) for k in torch.nonzero(mass == 0.0).reshape(-1)),
        )


def _relative(value: torch.Tensor, gradient: torch.Tensor) -> float:
    """``max|grad| / max(1, |value|)``, :func:`~sal.opt.fit.fit`'s convergence measure."""
    return float(gradient.abs().max()) / max(1.0, abs(float(value)))


def _varied(blocks: Mapping[str, slice], held: Iterable[str]) -> list[str]:
    """The block names not ``held``, in ``theta`` order; an unknown name is refused."""
    held = tuple(held)
    unknown = sorted(set(held) - set(blocks))
    if unknown:
        msg = f"held names {unknown} are not parameters; the parameters are {list(blocks)}"
        raise ValueError(msg)
    return [name for name in blocks if name not in held]
