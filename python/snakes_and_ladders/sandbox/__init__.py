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

``tropical`` is here on the second rule: the tropical Grassmannian relaxation
of topology search (issue #408), which neighbor joining matched at no
gradient steps at every size enumeration referees. ``pruning_burn`` is here on
the same rule: `burn`'s taped gradient over the Felsenstein recursion (issue
#449), which cost 46.80 ms against the route that replaced it at 16.36. Its
Rust half is behind the ``sandbox`` Cargo feature and absent from the default
build, so the module imports either way and refuses to run without it.
"""
