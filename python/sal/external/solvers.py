"""The external solvers, what each can be asked, and the licence its answer carries (issue #1282).

A :class:`Solver` is one framework's one method: gco's alpha expansion and its
alpha-beta swap are two members of one framework. Each declares the
:class:`Capability` set it offers in :data:`DECLARED`, beside the
:data:`~sal.external.frameworks.FRAMEWORKS` entry it runs. :func:`require`
refuses a call that asks for a capability the solver lacks, naming what is
missing; :func:`invoke` asks it before anything else, so a refused call
starts no subprocess. An absent framework raises :class:`ExternalUnavailable`
naming its extra, or the script that builds it, and never falls back to the
package's own solver.

:func:`provenance` reads the framework, its installed version, its licence
and whether that licence is OSI-approved; the version comes from the
distribution's metadata, so the framework is not imported to read it, or
for a source build from the commit the build recorded (#1279).

This module is step 1 of #1282. The calls that pose a problem in sal's types
and return sal's result types reach their framework through :func:`invoke`:
:mod:`sal.external.potts`'s ``ground_state`` (step 3) and ``lower_bound``
(step 4), :mod:`sal.external.hmm`'s ``fit``, ``viterbi`` and
``forward_log_likelihood`` (step 5), and :mod:`sal.external.hmc`'s
``sample`` (step 6).
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from sal.external.frameworks import FRAMEWORKS, Framework, built, built_version
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
    #: Accepts a Gaussian emission over more than one channel, or with a flat one.
    MULTI_CHANNEL_EMISSIONS = "multi_channel_emissions"
    #: Accepts a binomial emission.
    BINOMIAL_EMISSIONS = "binomial_emissions"
    #: Accepts a negative binomial emission.
    NEGATIVE_BINOMIAL_EMISSIONS = "negative_binomial_emissions"
    #: Accepts a beta-binomial emission, in either parameterization.
    BETA_BINOMIAL_EMISSIONS = "beta_binomial_emissions"
    #: Accepts an emission over a pair of counts.
    COUNT_PAIR_EMISSIONS = "count_pair_emissions"
    #: Accepts an emission family no capability above names, such as a caller's own.
    UNNAMED_EMISSIONS = "unnamed_emissions"
    #: Draws a Hamiltonian Monte Carlo chain from a log-density.
    HMC_SAMPLE = "hmc_sample"
    #: Adapts the step size and a diagonal mass in a warm-up before the draws.
    WINDOW_ADAPTATION = "window_adaptation"
    #: Draws each proposal's step from a band about the adapted one.
    STEP_JITTER = "step_jitter"
    #: Samples the zero-mean Gaussian of a precision, dense or diagonal.
    GAUSSIAN_TARGET = "gaussian_target"
    #: Samples ``exp(-U)`` for Rosenbrock's function ``U``.
    ROSENBROCK_TARGET = "rosenbrock_target"
    #: Samples a one-channel Gaussian mixture's likelihood in its ``theta``.
    GAUSSIAN_MIXTURE_TARGET = "gaussian_mixture_target"
    #: Samples a Gaussian HMM's likelihood of equal-length segments in its ``theta``.
    GAUSSIAN_HMM_TARGET = "gaussian_hmm_target"
    #: Samples an HMM's likelihood of segments of unequal lengths.
    RAGGED_SEGMENTS = "ragged_segments"
    #: Samples a count mixture's likelihood.
    COUNT_MIXTURE_TARGET = "count_mixture_target"
    #: Samples a count HMM's likelihood.
    COUNT_HMM_TARGET = "count_hmm_target"
    #: Samples an objective that declares no kernel, a Python closure.
    UNDECLARED_TARGET = "undeclared_target"


class Solver(StrEnum):
    """One external framework's one method."""

    GCO_EXPANSION = "gco_expansion"
    GCO_SWAP = "gco_swap"
    PYMAXFLOW_EXACT = "pymaxflow_exact"
    HIGHS_LP = "highs_lp"
    HMMLEARN = "hmmlearn"
    BLACKJAX_HMC = "blackjax_hmc"
    OPENGM_ICM = "opengm_icm"
    OPENGM_LBP = "opengm_lbp"
    OPENGM_ASTAR = "opengm_astar"
    OPENGM_TRWS = "opengm_trws"
    OPENGM_DD = "opengm_dd"
    OPENGM_EXPANSION = "opengm_expansion"
    OPENGM_SWAP = "opengm_swap"

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
_POTTS_BOUND = frozenset(
    {Capability.LOWER_BOUND, Capability.MULTI_LABEL, Capability.FORBIDDEN_LABELS}
)

#: Every solver's declaration. PyMaxflow's cut is exact at q = 2 alone, so it
#: lacks :attr:`Capability.MULTI_LABEL`; HiGHS solves the local-polytope LP,
#: a bound (#1063), and its ILP, the minimum (#1274), so it is exact and
#: no ground-state solver; hmmlearn fits the three families its
#: adapter writes (#975, #997), and declares none of the other families'
#: capabilities, so :mod:`sal.external.hmm` refuses them; BlackJAX samples
#: the targets its script rebuilds in JAX, and adapts without a jittered
#: step, so :mod:`sal.external.hmc` refuses the rest. OpenGM's A* is exact,
#: its ICM, loopy BP and moves local; its TRW-S and dual decomposition bound
#: the minimum, and neither is exact (#1279).
DECLARED: Mapping[Solver, Declaration] = {
    Solver.GCO_EXPANSION: Declaration("gco", _POTTS_MOVES),
    Solver.GCO_SWAP: Declaration("gco", _POTTS_MOVES),
    Solver.PYMAXFLOW_EXACT: Declaration(
        "pymaxflow", frozenset({Capability.GROUND_STATE, Capability.EXACT})
    ),
    Solver.HIGHS_LP: Declaration(
        "highs",
        frozenset(
            {
                Capability.LOWER_BOUND,
                Capability.MULTI_LABEL,
                Capability.FORBIDDEN_LABELS,
                Capability.EXACT,
            }
        ),
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
    Solver.BLACKJAX_HMC: Declaration(
        "blackjax",
        frozenset(
            {
                Capability.HMC_SAMPLE,
                Capability.WINDOW_ADAPTATION,
                Capability.GAUSSIAN_TARGET,
                Capability.ROSENBROCK_TARGET,
                Capability.GAUSSIAN_MIXTURE_TARGET,
                Capability.GAUSSIAN_HMM_TARGET,
            }
        ),
    ),
    Solver.OPENGM_ICM: Declaration("opengm", _POTTS_MOVES),
    Solver.OPENGM_LBP: Declaration("opengm", _POTTS_MOVES),
    Solver.OPENGM_ASTAR: Declaration("opengm", _POTTS_MOVES | {Capability.EXACT}),
    Solver.OPENGM_TRWS: Declaration("opengm", _POTTS_BOUND),
    Solver.OPENGM_DD: Declaration("opengm", _POTTS_BOUND),
    Solver.OPENGM_EXPANSION: Declaration("opengm", _POTTS_MOVES),
    Solver.OPENGM_SWAP: Declaration("opengm", _POTTS_MOVES),
}


@dataclass(frozen=True)
class Provenance:
    """Where an external answer came from, and the terms it carries."""

    #: The framework's registered name, a key of :data:`FRAMEWORKS`.
    framework: str
    #: The installed distribution's version, as its metadata states it, or
    #: the commit a source build pinned.
    version: str
    #: The licence, as the distribution declares it.
    licence: str
    #: Whether every part of the licence is OSI-approved.
    osi: bool


class ExternalUnavailable(ImportError):
    """A solver's framework is not installed; the message names its extra, or its build."""

    def __init__(self, solver: Solver) -> None:
        framework = solver.framework
        self.solver = solver
        self.extra = framework.extra
        message = (
            f"{solver} needs {framework.distribution}, which is not installed: "
            f"{framework.remedy}"
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
    """Whether ``solver``'s framework is installed, found without being imported or loaded."""
    framework = solver.framework
    if framework.build is not None:
        return built(framework) is not None
    return installed(framework.module)


def provenance(solver: Solver) -> Provenance:
    """The framework, installed version and licence an answer from ``solver`` carries.

    Raises :class:`ExternalUnavailable` where the framework is absent.
    """
    if not available(solver):
        raise ExternalUnavailable(solver)
    framework = solver.framework
    return Provenance(
        framework=framework.name,
        version=(
            built_version(framework)
            if framework.build is not None
            else importlib.metadata.version(framework.distribution)
        ),
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
