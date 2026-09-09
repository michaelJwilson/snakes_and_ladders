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

``pruning_problem`` is here on the second rule: crossing the FFI boundary once
per search rather than once per pass (issue #444), which cut bytes per pass
1,881x and left wall time within 8.4% at every size measured.

``tropical`` is here on the second rule: the tropical Grassmannian relaxation
of topology search (issue #408), which neighbor joining matched at no
gradient steps at every size enumeration referees.
"""
