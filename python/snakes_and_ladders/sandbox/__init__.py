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
gradient steps at every size enumeration referees. ``compiled_pruning`` is
here on the same rule and is thinner: ``likelihood.pruning_torch``'s
recursion through ``torch.compile``, declined at 1.21x and 1.09x eager
(issue #443). A compiler is not a second implementation, so what it carries
beyond the one ``torch.compile`` call is the reading of Dynamo's compile
state that makes "compiled, and not silently fallen back to eager" an
assertion; its docstring says so first, and names the experiment, the ratio
and the host.
"""
