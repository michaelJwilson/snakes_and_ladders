"""The home for clean solutions to posed problems that are off the hot path (issue #322).

Two things arrive here and they are one thing seen from either side: an
implementation a framework replaced, which stays as the referee the framework
is pinned against, per root ``CLAUDE.md``'s oracle rule; and a candidate a
measurement declined, which stays because the question it answers was well
posed and the answer is a result. Scaffolding is neither and is deleted, its
measurement surviving in the pull request and in ``STATUS.md``. Only
``tests/`` and ``snakes_and_ladders.qa`` import from here, never ``sim``,
``likelihood``, ``opt``, ``search`` or ``learn``; nothing that moved in is
deleted; and nothing is re-exported from the package root. ``CLAUDE.md`` in
this directory states the rules and ``tests/regression/test_sandbox.py``
asserts them.

Holding one module: ``pruning_problem``, the declined answer to crossing the
FFI boundary once per search rather than once per pass (issue #444). No
adoption has yet been measured to beat the implementation it would replace.
"""
