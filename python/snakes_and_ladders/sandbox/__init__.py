"""The conserved home: implementations kept for what they referee, not for a hot path (issue #322).

Two kinds move in here and neither is deleted. A hand-rolled implementation a
framework replaced on a hot path stays as the referee the framework is pinned
against, per root ``CLAUDE.md``'s oracle rule. An implementation a
measurement declined -- built, measured, found to buy nothing -- stays with
the tests that measured it, because a measurement whose subject was removed
is a claim with no way back to it.

Only ``tests/`` and ``snakes_and_ladders.qa`` import from here, never ``sim``,
``likelihood``, ``opt``, ``search`` or ``learn``; nothing is deleted from it;
and nothing is re-exported from the package root. ``CLAUDE.md`` in this
directory states the rules and ``tests/regression/test_sandbox.py`` asserts
them.

Three framework fronts are here on the second rule, all declined on their
numbers (issues #391, #392), each importing the ``frameworks`` extra at
module scope so a caller without it fails rather than silently falling back
to what the front referees: :mod:`~snakes_and_ladders.sandbox.gym_vector`
batches the rollout through ``gymnasium.vector``,
:mod:`~snakes_and_ladders.sandbox.torchrl_advantage` computes the generalized
advantage through TorchRL's ``GAE``, and
:mod:`~snakes_and_ladders.sandbox.torchrl_clip` computes the clipped
surrogate through its ``ClipPPOLoss``. Each module's docstring carries the
ratio it was declined at and the host that produced it.

``tropical`` is here on the second rule: the tropical Grassmannian relaxation
of topology search (issue #408), which neighbor joining matched at no
gradient steps at every size enumeration referees.
"""
