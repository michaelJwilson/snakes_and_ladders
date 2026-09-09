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

One front so far, and a thin one:
:mod:`~snakes_and_ladders.sandbox.compiled_pruning` is
``likelihood.pruning_torch``'s recursion through ``torch.compile``, declined
at 1.21x and 1.09x eager (issue #443). A compiler is not a second
implementation, so what the module carries beyond the one ``torch.compile``
call is the reading of Dynamo's compile state that makes "compiled, and not
silently fallen back to eager" an assertion; its docstring says so first, and
names the experiment, the ratio and the host.
"""
