"""What a hidden state emits, separated from how the model is fitted.

The emission is the only part of an HMM that knows what an observation
*is*. The recursions need three things from it: draw an observation given a
state, score one given a state, re-estimate from posterior state weights.
This package is that interface and its implementations;
``snakes_and_ladders.opt.hmm`` holds the recursions that consume it.

**Why it is here rather than in a module that owns a model.** ``opt/CLAUDE.md``
forbids ``snakes_and_ladders.opt`` from importing ``snakes_and_ladders.sim``, and
``tests/regression/opt/test_opt_objective.py`` asserts it, so a family defined
in ``sim`` would be unreachable from the objective that scores with it. A
family names no tree, no alignment and no lattice, so it sits beside
:mod:`snakes_and_ladders.numerics` on the same terms: importable from
anywhere, inverting no layering.

**A density is not a probability, and the difference is load-bearing.**
:meth:`EmissionFamily.log_density` returns a log-probability for a family over
a countable alphabet and a log *density* for one over the reals. Only the
first is bounded above by zero, so the evidence ``log P(observations)`` of a
continuous-emission HMM may be positive and an assertion that it is not fails
on correct code. :attr:`EmissionFamily.is_discrete` states which case a family
is, so a test can assert the bound exactly where it holds.

**An unbounded likelihood is a property of the model, not a bug in the fit.**
A Gaussian emission's likelihood has no maximum: put one state's mean on a
single observation and let its variance go to zero (Bishop, *Pattern
Recognition and Machine Learning*, section 9.2.1).
:class:`GaussianEmission` therefore carries an explicit variance floor and
**refuses** rather than clamps when a re-estimate reaches it, because a
clamped fit returns normally and its intervals mean nothing (issue #122).

**A covariate arrives shaped for the states, and that rule is stated here
because this is what enforces it.** :func:`trial_count` and :func:`exposure`
refuse a covariate whose last axis is not a singleton, so what reaches a
single-channel family carries one: the singleton is what broadcasts the
per-observation value along the states at the ``+``. A family whose
observation carries axes of its own --- the two-channel count pair --- takes
one covariate *per* axis instead, and the singleton belongs inside each
channel rather than after them. So a seam that slices a covariate adds the
singleton **only** where the covariate has no axes of its own; appending it to
a ``(S, V, 2)`` covariate makes ``(S, V, 2, 1)``, whose last axis names no
channel and broadcasts against the wrong one. Five functions across ``sim``
and ``likelihood`` slice a covariate and each obeys this; #670 is what a
fifth copy of the rule costs when one of them drifts, and issue #677 is why
they now point here rather than restate it.

**A draw and an M step take arrays; a score takes a tensor.**
:meth:`EmissionFamily.sample` and :meth:`EmissionFamily.reestimate` take no
derivative, so each takes a NumPy array or a tensor and converts once at entry
through :func:`~snakes_and_ladders.emissions.base.as_tensor`, reproducing the
tensor call bitwise (issue #1011). :meth:`EmissionFamily.log_density` is
differentiated through by the objectives and stays a tensor in, tensor out, and
a family's parameters stay tensors, so importing this package imports torch.

Parameterization for an unconstrained optimizer belongs to
``snakes_and_ladders.opt.constrain``; here it would make
:mod:`snakes_and_ladders.sim` import ``snakes_and_ladders.opt`` transitively
to draw a sequence.

**The package is four modules, and each imports only those before it.**
:mod:`~snakes_and_ladders.emissions.base` declares the interface,
:mod:`~snakes_and_ladders.emissions.families` the categorical and Gaussian
families, :mod:`~snakes_and_ladders.emissions.mstep` the count families' M-step
solves and the backend they run on, and
:mod:`~snakes_and_ladders.emissions.counts` the count families, which call
those solves. Every name this module exported before the split (issue #1010)
is re-exported here, so ``from snakes_and_ladders.emissions import X`` is
unchanged; :data:`M_STEP_BACKEND` is read from and written to
:mod:`~snakes_and_ladders.emissions.mstep`, the one module that reads it.

**A private name stays in the module that defines it.** A helper two
submodules share is public in the module that owns it --- the M-step solves
:mod:`~snakes_and_ladders.emissions.counts` calls are public in
:mod:`~snakes_and_ladders.emissions.mstep` --- and one that belongs to neither
goes in an underscore-prefixed submodule, ``_common.py``, under a public name;
none does today. No module imports another module's underscore-prefixed name,
and ``tests/regression/test_duplication_guards.py`` enforces it.
"""

from __future__ import annotations

import sys
import types

from snakes_and_ladders.backend import Backend
from snakes_and_ladders.emissions import mstep
from snakes_and_ladders.emissions.base import (
    CountEmissionFamily,
    CovariateNotSupportedError,
    EmissionFamily,
    FamilyT_co,
    ParameterDomainError,
    Reestimate,
    Values,
    exposure,
    refuse_covariate,
    trial_count,
    validated_exposure,
    validated_trials,
)
from snakes_and_ladders.emissions.counts import (
    BetaBinomialEmission,
    BinomialEmission,
    CountPairEmission,
    NegativeBinomialEmission,
    PoissonEmission,
)
from snakes_and_ladders.emissions.families import (
    COLLAPSE_EXPONENT,
    CategoricalEmission,
    GaussianEmission,
    pooled_variance_floor,
    refuse_collapsed,
)
from snakes_and_ladders.emissions.mstep import (
    EXPOSED_TAIL_RATIO,
    identifiable_concentration_bound,
    identifiable_dispersion_bound,
)

#: Where the count families' M-step solves run: a view of
#: :data:`snakes_and_ladders.emissions.mstep.M_STEP_BACKEND`. Declared, not
#: bound, so that the module's property below answers both a read and a write.
M_STEP_BACKEND: Backend


class _Package(types.ModuleType):
    """This module, with :data:`M_STEP_BACKEND` forwarded to :mod:`.mstep`.

    A re-exported constant is a copy: setting it here would leave the solves
    reading the old value, and a test switching the backend would compare the
    compiled route with itself.
    """

    @property
    def M_STEP_BACKEND(self) -> Backend:
        """The backend the count families' M-step solves run on."""
        return mstep.M_STEP_BACKEND

    @M_STEP_BACKEND.setter
    def M_STEP_BACKEND(self, value: Backend) -> None:
        mstep.M_STEP_BACKEND = value


sys.modules[__name__].__class__ = _Package

__all__ = [
    "COLLAPSE_EXPONENT",
    "EXPOSED_TAIL_RATIO",
    "M_STEP_BACKEND",
    "BetaBinomialEmission",
    "BinomialEmission",
    "CategoricalEmission",
    "CountEmissionFamily",
    "CountPairEmission",
    "CovariateNotSupportedError",
    "EmissionFamily",
    "FamilyT_co",
    "GaussianEmission",
    "NegativeBinomialEmission",
    "ParameterDomainError",
    "PoissonEmission",
    "Reestimate",
    "Values",
    "exposure",
    "identifiable_concentration_bound",
    "identifiable_dispersion_bound",
    "pooled_variance_floor",
    "refuse_collapsed",
    "refuse_covariate",
    "trial_count",
    "validated_exposure",
    "validated_trials",
]
