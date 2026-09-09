"""The home of the second implementation, whichever side won (issue #322).

A framework and the hand-rolled code it was measured against are two
implementations of one equation, and one of them belongs here: the
implementation a framework replaced on a hot path, kept as the referee root
``CLAUDE.md``'s oracle rule requires, or the framework front that lost, kept
so the decline stays re-checkable when the library's next version moves the
numbers. The two need not be a library and our own code -- a method declined
against a simpler method is the same case. Only ``tests/`` and
``snakes_and_ladders.qa`` import from here, never ``sim``, ``likelihood``,
``opt``, ``search`` or ``learn``; nothing is deleted from it; and nothing is
re-exported from the package root. ``CLAUDE.md`` in this directory states the
rules and ``tests/regression/test_sandbox.py`` asserts them.

Three framework fronts, all declined on their numbers (issues #388, #389,
#390): :mod:`~snakes_and_ladders.sandbox.pyg_surrogate` computes
``GraphSurrogate``'s forward through PyTorch Geometric's ``GINConv``,
:mod:`~snakes_and_ladders.sandbox.rustworkx_clusters` labels the
Swendsen-Wang bond graph through ``rustworkx.connected_components``, and
:mod:`~snakes_and_ladders.sandbox.scipy_mincut` solves the ground state's
minimum cut through ``scipy.sparse.csgraph.maximum_flow``. Each module's
docstring carries the ratio it was declined at and the host that produced it.
The first two import the ``frameworks`` extra at module scope, so a caller
without it fails rather than silently falling back to what the front referees;
scipy is a core dependency and the third imports it directly.

Three further framework fronts, all declined on their numbers (issues #391,
#392) and each importing the ``frameworks`` extra at module scope for the same
reason: :mod:`~snakes_and_ladders.sandbox.gym_vector` batches the rollout
through ``gymnasium.vector``,
:mod:`~snakes_and_ladders.sandbox.torchrl_advantage` computes the generalized
advantage through TorchRL's ``GAE``, and
:mod:`~snakes_and_ladders.sandbox.torchrl_clip` computes the clipped surrogate
through its ``ClipPPOLoss``.

:mod:`~snakes_and_ladders.sandbox.tropical` is a decline of the same kind
against no framework at all: the tropical Grassmannian relaxation of topology
search (issue #408), which neighbor joining matched at no gradient steps at
every size enumeration referees.
:mod:`~snakes_and_ladders.sandbox.compiled_pruning` is here on the same rule
and is thinner: ``likelihood.pruning_torch``'s recursion through
``torch.compile``, declined at 1.21x and 1.09x eager (issue #443). A compiler
is not a second implementation, so what it carries beyond the one
``torch.compile`` call is the reading of Dynamo's compile state that makes
"compiled, and not silently fallen back to eager" an assertion; its docstring
says so first, and names the experiment, the ratio and the host.
:mod:`~snakes_and_ladders.sandbox.pruning_problem` is here on the same rule
again: crossing the FFI boundary once per search rather than once per pass
(issue #444), which cut bytes per pass 1,881x and left wall time within 8.4%
at every size measured.
:mod:`~snakes_and_ladders.sandbox.pruning_burn` is here on the same rule once
more: ``burn``'s taped gradient over the Felsenstein recursion (issue #449),
at 46.80 ms against the route that replaced it at 16.36. Its Rust half is
behind the ``sandbox`` Cargo feature and absent from the default build, so the
module imports either way and refuses to run without it.
"""
