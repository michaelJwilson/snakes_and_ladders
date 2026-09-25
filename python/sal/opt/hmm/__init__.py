"""A discrete HMM: the second reference instance of ``Objective``.

One of issue #63's named cases, and unlike the Potts chain it has an
*independent fitting algorithm* -- Baum-Welch -- so a gradient fit can be
checked against something other than itself.

The forward recursion here and Felsenstein pruning are the same sum-product
computation on different graphs: a caterpillar tree carrying one observed
leaf per internal node *is* an HMM (``eq:forward`` of ``docs/tex/textbook.tex``,
derived in ``app:forward-backward``; Durbin et al., ch. 3; Koller & Friedman
for the general framing).

**Label switching.** The likelihood is invariant to permuting the hidden
states, so a fitted parameter set matches truth only up to a permutation. The
model is otherwise identifiable; every row is gauge-fixed by
:func:`sal.opt.constrain.log_simplex`. A recovery test must align the
permutation before comparing.

Ground truth and data generation live in :mod:`sal.sim.hmm`; this
module holds the fitting objective, its EM oracle, and the state-alignment
helper a recovery test needs.

**The package is three modules, and each imports only those before it.**
:mod:`~sal.opt.hmm.forward` holds the kernels --- the forward
recursion and the state alignment --- that the other two share;
:mod:`~sal.opt.hmm.objectives` the gradient-fit objectives, one
per emission family; and :mod:`~sal.opt.hmm.estimation` the EM
drivers, with the route each takes between the torch recursion and the
compiled steps. Viterbi and the scored evidence are evaluators and live in
:mod:`sal.likelihood.hmm` (issue #1059). Every public name this module defined
before the split (issue #1010) is re-exported here, so
``from sal.opt.hmm import X`` is unchanged.

**A private name stays in the module that defines it**, and no submodule needs
another's. The one exception is ``_HmmObjective``, the base the objectives
share, which :mod:`sal.opt.hmm.jax` reads from here until issue
#1004 gives it a public seam; it is re-exported so that import is unchanged,
and ``tests/regression/test_duplication_guards.py`` admits both crossings.
"""

from __future__ import annotations

from sal.opt.hmm.estimation import (
    CategoricalFit,
    CovariateUpdate,
    EmFit,
    ExpectedRateNormalizer,
    baum_welch,
    baum_welch_family,
    compiled_family,
)
from sal.opt.hmm.forward import (
    align_by_key,
    align_families,
    align_states,
    forward_log_likelihood,
    forward_log_likelihood_from_density,
)
from sal.opt.hmm.objectives import (
    BetaBinomialHmmObjective,
    BinomialHmmObjective,
    GaussianHmmObjective,
    HmmMetrics,
    HmmObjective,
    NegativeBinomialHmmObjective,
    PoissonHmmObjective,
)

# Re-exported for `opt.hmm.jax`, whose import of it from here predates the
# split and is admitted until issue #1004; the alias is what declares the
# re-export to `mypy --strict`.
from sal.opt.hmm.objectives import (
    _HmmObjective as _HmmObjective,  # noqa: PLC0414
)

__all__ = [
    "BetaBinomialHmmObjective",
    "BinomialHmmObjective",
    "CategoricalFit",
    "CovariateUpdate",
    "EmFit",
    "ExpectedRateNormalizer",
    "GaussianHmmObjective",
    "HmmMetrics",
    "HmmObjective",
    "NegativeBinomialHmmObjective",
    "PoissonHmmObjective",
    "align_by_key",
    "align_families",
    "align_states",
    "baum_welch",
    "baum_welch_family",
    "compiled_family",
    "forward_log_likelihood",
    "forward_log_likelihood_from_density",
]
