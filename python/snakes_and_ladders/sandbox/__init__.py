"""The oracle home: hand-rolled implementations a framework has replaced on a hot path (issue #322).

An implementation moves in here when a measured adoption lands -- a framework
fronts the hot path, and the implementation it replaced stays as the referee
the framework is pinned against, per root ``CLAUDE.md``'s oracle rule. Only
``tests/`` and ``snakes_and_ladders.qa`` import from here, never ``sim``,
``likelihood``, ``opt``, ``search`` or ``learn``; nothing is deleted from it;
and nothing is re-exported from the package root. ``CLAUDE.md`` in this
directory states the rules and ``tests/regression/test_sandbox.py`` asserts
them.

Empty so far: no adoption has yet been measured to beat the implementation it
would replace, so nothing has been moved.
"""
