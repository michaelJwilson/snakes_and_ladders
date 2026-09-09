"""The home of the second implementation, whichever side won (issue #322).

A framework and the hand-rolled code it was measured against are two
implementations of one equation, and one of them belongs here: the
implementation a framework replaced on a hot path, kept as the referee root
``CLAUDE.md``'s oracle rule requires, or the framework front that lost, kept
so the decline stays re-checkable when the library's next version moves the
numbers. Only ``tests/`` and ``snakes_and_ladders.qa`` import from here, never
``sim``, ``likelihood``, ``opt``, ``search`` or ``learn``; nothing is deleted
from it; and nothing is re-exported from the package root. ``CLAUDE.md`` in
this directory states the rules and ``tests/regression/test_sandbox.py``
asserts them.

Three framework fronts so far, all declined on their numbers (issues #391,
#392), each importing the ``frameworks`` extra at module scope so a caller
without it fails rather than silently falling back to what the front
referees: :mod:`~snakes_and_ladders.sandbox.gym_vector` batches the rollout
through ``gymnasium.vector``,
:mod:`~snakes_and_ladders.sandbox.torchrl_advantage` computes the generalized
advantage through TorchRL's ``GAE``, and
:mod:`~snakes_and_ladders.sandbox.torchrl_clip` computes the clipped
surrogate through its ``ClipPPOLoss``. Each module's docstring carries the
ratio it was declined at and the host that produced it.
"""
