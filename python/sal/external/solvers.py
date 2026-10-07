"""The external solvers, what each can be asked, and the licence its answer carries (issue #1282).

A :class:`Solver` is one framework's one method: gco's alpha expansion and its
alpha-beta swap are two members of one framework. Each declares the
:class:`Capability` set it offers in :data:`DECLARED`, beside the
:data:`~sal.external.frameworks.FRAMEWORKS` entry it runs. :func:`require`
refuses a call that asks for a capability the solver lacks, naming what is
missing; :func:`invoke` asks it before anything else, so a refused call
starts no subprocess. An absent framework raises :class:`ExternalUnavailable`
naming its extra, and never falls back to the package's own solver.

:func:`provenance` reads the framework, its installed version, its licence
and whether that licence is OSI-approved; the version comes from the
distribution's metadata, so the framework is not imported to read it.

This module is step 1 of #1282. The calls that pose a problem in sal's types
and return sal's result types (``ground_state``, ``lower_bound``,
``hmm_fit``, ``viterbi``, ``hmc_sample``) and the persistent session are
later steps; each reaches its framework through :func:`invoke`.
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from sal.external.frameworks import FRAMEWORKS, Framework
from sal.external.runner import Run, installed, run


class Capability(StrEnum):
    """What a solver can be asked: a task, or a kind of input it accepts."""

    #: Returns a labelling that minimizes :func:`sal.sim.potts.energy`.
    GROUND_STATE = "ground_state"
    #: Returns a lower bound on the minimum energy.
    LOWER_BOUND = "lower_bound"
    #: The labelling it returns is a minimum, not a local one.
    EXACT = "exact"
    #: Accepts more than two states per site; a solver without it takes q = 2 only.
    MULTI_LABEL = "multi_label"
    #: Accepts a forbidden label, as a stated large finite unary (#1139).
    FORBIDDEN_LABELS = "forbidden_labels"
    #: Fits an HMM's parameters by Baum-Welch.
    HMM_FIT = "hmm_fit"
    #: Decodes an HMM's most probable path.
    VITERBI = "viterbi"
    #: Scores an HMM's log-likelihood of the observations.
    LOG_LIKELIHOOD = "log_likelihood"
    #: Accepts a one-channel Gaussian emission.
    GAUSSIAN_EMISSIONS = "gaussian_emissions"
    #: Accepts a Poisson emission.
    POISSON_EMISSIONS = "poisson_emissions"
    #: Accepts a categorical emission.
    CATEGORICAL_EMISSIONS = "categorical_emissions"
    #: Draws a Hamiltonian Monte Carlo chain from a log-density.
    HMC_SAMPLE = "hmc_sample"


class Solver(StrEnum):
    """One external framework's one method."""

    GCO_EXPANSION = "gco_expansion"
    GCO_SWAP = "gco_swap"
    PYMAXFLOW_EXACT = "pymaxflow_exact"
    HIGHS_LP = "highs_lp"
    HMMLEARN = "hmmlearn"
    BLACKJAX_HMC = "blackjax_hmc"

    @property
    def framework(self) -> Framework:
        """The registered framework this solver runs."""
        return FRAMEWORKS[DECLARED[self].framework]

    @property
    def capabilities(self) -> frozenset[Capability]:
        """What this solver can be asked."""
        return DECLARED[self].capabilities


@dataclass(frozen=True)
class Declaration:
    """A solver's framework, by its :data:`FRAMEWORKS` name, and its capabilities."""

    framework: str
    capabilities: frozenset[Capability]


_POTTS_MOVES = frozenset(
    {Capability.GROUND_STATE, Capability.MULTI_LABEL, Capability.FORBIDDEN_LABELS}
)

#: Every solver's declaration. PyMaxflow's cut is exact at q = 2 alone, so it
#: lacks :attr:`Capability.MULTI_LABEL`; HiGHS solves the local-polytope LP,
#: a bound and no labelling (#1063); hmmlearn fits the three families its
#: adapter writes (#975, #997).
DECLARED: Mapping[Solver, Declaration] = {
    Solver.GCO_EXPANSION: Declaration("gco", _POTTS_MOVES),
    Solver.GCO_SWAP: Declaration("gco", _POTTS_MOVES),
    Solver.PYMAXFLOW_EXACT: Declaration(
        "pymaxflow", frozenset({Capability.GROUND_STATE, Capability.EXACT})
    ),
    Solver.HIGHS_LP: Declaration(
        "highs", frozenset({Capability.LOWER_BOUND, Capability.MULTI_LABEL})
    ),
    Solver.HMMLEARN: Declaration(
        "hmmlearn",
        frozenset(
            {
                Capability.HMM_FIT,
                Capability.VITERBI,
                Capability.LOG_LIKELIHOOD,
                Capability.GAUSSIAN_EMISSIONS,
                Capability.POISSON_EMISSIONS,
                Capability.CATEGORICAL_EMISSIONS,
            }
        ),
    ),
    Solver.BLACKJAX_HMC: Declaration("blackjax", frozenset({Capability.HMC_SAMPLE})),
}


@dataclass(frozen=True)
class Provenance:
    """Where an external answer came from, and the terms it carries."""

    #: The framework's registered name, a key of :data:`FRAMEWORKS`.
    framework: str
    #: The installed distribution's version, as its metadata states it.
    version: str
    #: The licence, as the distribution declares it.
    licence: str
    #: Whether every part of the licence is OSI-approved.
    osi: bool


class ExternalUnavailable(ImportError):
    """A solver's framework is not installed; the message names its extra."""

    def __init__(self, solver: Solver) -> None:
        framework = solver.framework
        self.solver = solver
        self.extra = framework.extra
        message = (
            f"{solver} needs {framework.distribution}, which is not installed: "
            f"install the `{self.extra}` extra"
        )
        super().__init__(message)


class CapabilityRefused(ValueError):
    """A call asked a solver for capabilities it does not declare."""

    def __init__(self, solver: Solver, missing: frozenset[Capability]) -> None:
        self.solver = solver
        self.missing = missing
        named = ", ".join(sorted(missing))
        super().__init__(f"{solver} does not offer: {named}")


def available(solver: Solver) -> bool:
    """Whether ``solver``'s framework is installed, found without being imported."""
    return installed(solver.framework.module)


def provenance(solver: Solver) -> Provenance:
    """The framework, installed version and licence an answer from ``solver`` carries.

    Raises :class:`ExternalUnavailable` where the framework is absent.
    """
    if not available(solver):
        raise ExternalUnavailable(solver)
    framework = solver.framework
    return Provenance(
        framework=framework.name,
        version=importlib.metadata.version(framework.distribution),
        licence=framework.licence,
        osi=framework.osi,
    )


def require(solver: Solver, needs: Iterable[Capability]) -> None:
    """Refuse a call that asks ``solver`` for what it does not declare.

    Raises :class:`CapabilityRefused` naming every missing capability.
    """
    missing = frozenset(needs) - solver.capabilities
    if missing:
        raise CapabilityRefused(solver, missing)


def invoke(
    solver: Solver,
    needs: Iterable[Capability],
    inputs: Mapping[str, np.ndarray],
    *,
    timeout: float = 600.0,
) -> Run:
    """Run ``solver``'s script on ``inputs``, once its capabilities and framework are checked.

    The two checks run in this process, in this order, before any subprocess
    starts: :func:`require`, then :func:`available`. The script is the
    framework's, as :func:`sal.external.runner.run` names it.
    """
    require(solver, needs)
    if not available(solver):
        raise ExternalUnavailable(solver)
    return run(solver.framework.name, inputs, timeout=timeout)
