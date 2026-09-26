"""Learned surrogates for a solver, declined and conserved (issue #1067).

``potts`` is the one here: a multigrid network trained on ground-state
labellings of ``spatio_tiling`` fields. Its module docstring states the
question, the trial's answer and why the route was declined.
"""

from sal import _submodules

__getattr__ = _submodules(__name__)
