"""Which implementation runs a kernel, chosen by the caller and never silently.

Root ``CLAUDE.md`` keeps every accelerated kernel beside its pure Python or
NumPy oracle, and its backend rule admits a compiled path only against a
measurement on an existing hot path. The enum is the seam that makes the
choice explicit at the call site: a test that pins a kernel against its
oracle names both, and a caller that wants the oracle can say so.

Two compiled backends exist here for two different reasons, and the split is
deliberate rather than a stack. **Rust** carries the sampling sweep, where
the loop is over an adjacency structure and there is no array arithmetic for
NumPy to vectorize. **Numba** carries the kernels whose arithmetic is
NumPy's operation for operation -- descent, energy evaluation, the Gibbs
sweep.

Both are defaults, and on the same terms. ``f64::exp`` and NumPy's differ in
the last place, and ``searchsorted`` is a threshold, so a sampling port is
distributional unless something bounds that: each compiled sweep decides a
site only where the draw clears every cumulative boundary by more than the
two exponentials can move it, and hands the rest back. The pin is then exact
and the compiled path is the default without a committed number moving
(issues #561, #599).

**Which way each module's default points** (issue #860). A default decides
what a caller who does not ask gets, and it is a per-module decision; stated
here so it is read once rather than off 46 signatures.

``RUST``
    `numerics.sample_rows`, `likelihood.convolutional`,
    `likelihood.message_passing`, `likelihood.turbo`,
    `likelihood.ragged`, and the sweeps of `sample.potts_mcmc`,
    `sample.annealed` and `sample.tempered`.
``NUMBA``
    `sample.gibbs` and
    `search.icm.iterated_conditional_modes`: arithmetic that is
    NumPy's operation for operation.
``JAX``
    the gradient of `opt.hmm`'s objectives (issue #1000), pinned to the
    ``TORCH`` autograd route at 1e-10.
``PYTHON``
    everything else that takes the enum --- `likelihood.pruning`,
    `likelihood.spatio_sequential`, `search.maxflow`,
    `search.alpha_expansion`'s cut moves, `search.bifurcation`,
    `search.spatio_sequential`, `sim.count_pairs`, `sample.potts_keyed`,
    `learn.ranking`.

The rule the split follows is the one above: a compiled default is taken
where the pin against the oracle is *exact* and a measurement earned it, and
left at ``PYTHON`` otherwise --- which is every entry point whose return is a
number a document quotes, an oracle's own answer or a fixture's draw. There
the compiled route is asked for by name and the pin says the answer is the
same.

**Where a twin lives, and who reaches it** (issue #1059). An algorithm with
a twin is a package: its reference is ``<sub>.<algorithm>``, the package's
``__init__``, and each twin is ``<sub>.<algorithm>.<backend>`` beside it,
``<backend>`` a member's value: ``numba``, ``rust``, ``torch`` or ``jax``. The
reference is the gateway: it takes ``backend=`` and dispatches, and outside
the algorithm's own package only ``tests/`` imports a twin, which
``tests/regression/test_duplication_guards.py`` reads from the imports.

:func:`refuse_backend` is how a module declines a member it has no
implementation for. The sentence was written inline at five sites and is one
here, so a refusal reads the same wherever a caller meets it.
:func:`twin` is how a module with a Rust twin reaches it: the refusal, then
the twin imported at call time (issue #1010).
"""

from __future__ import annotations

import importlib
from enum import StrEnum
from types import ModuleType


class Backend(StrEnum):
    """Where a kernel runs."""

    PYTHON = "python"
    """The pure Python or NumPy oracle."""

    NUMBA = "numba"
    """A ``numba.njit`` kernel over the compressed-row adjacency."""

    RUST = "rust"
    """The ``oxisal`` extension."""

    TORCH = "torch"
    """A ``torch`` kernel: the same arithmetic as the NumPy oracle over a tensor,
    which is what puts it on a device. Admitted for a hot path that is
    elementwise over sites and earns the GPU rule, or is measured against it
    (issue #823); a kernel whose loop is over an adjacency belongs to Rust."""

    JAX = "jax"
    """A ``jax`` route: an objective's value and gradient under
    ``jit(value_and_grad)``. The HMM objectives' default gradient since issue
    #1000 measured it at 0.07x--0.18x PyTorch autograd's runtime at
    10^4--10^5 positions; ``TORCH`` there is the autograd oracle it is pinned
    to."""


def refuse_backend(name: str, backend: Backend, allowed: tuple[Backend, ...]) -> None:
    """Refuse a member ``name`` has no implementation for, in the one sentence.

    Parameters
    ----------
    name : str
        What was asked for, as the message names it: the callable, or the
        phrase its module wrote --- ``"message passing"``, ``"the coupled
        model"``. The caller passes what the site printed, so no message
        changed when the guards moved here (issue #860).
    backend : Backend
        What the caller asked for.
    allowed : tuple[Backend, ...]
        What does run it, listed in the message in this order.

    Raises
    ------
    ValueError
        If ``backend`` is not among ``allowed``.
    """
    if backend in allowed:
        return
    runs_on = " or ".join(str(one) for one in allowed)
    msg = f"{name} runs on {runs_on}, not {backend}"
    raise ValueError(msg)


def twin(name: str, backend: Backend, oracle: str) -> ModuleType | None:
    """The Rust twin of module ``oracle`` where ``backend`` asks for it, else ``None``.

    An algorithm's reference lives at ``<sub>.<algorithm>`` and its Rust twin
    at ``<sub>.<algorithm>.rust`` (issue #1059). The gateway refuses every
    member but ``PYTHON`` and ``RUST`` with :func:`refuse_backend`, and imports
    the twin inside the call rather than at module level: each twin imports a
    type or a helper from its oracle, so the module-level import is a cycle,
    and it would also put the extension behind every import of the oracle.
    Six modules wrote those three steps and the comment out by hand, at eight
    sites; the
    caller passes its ``__name__``, so the twin is found by the layout rather
    than by a second spelling of the path.

    What the call site gives up is the type of the kernel: an attribute of a
    module imported by name is ``Any`` to ``mypy``, so a site returning the
    kernel's value states the type it returns with ``cast``.

    Parameters
    ----------
    name : str
        As :func:`refuse_backend`.
    backend : Backend
        What the caller asked for.
    oracle : str
        The calling module's ``__name__``.

    Returns
    -------
    ModuleType | None
        ``<sub>.<algorithm>.rust`` for :data:`Backend.RUST`; ``None`` for
        :data:`Backend.PYTHON`, where the caller runs its own oracle.

    Raises
    ------
    ValueError
        If ``backend`` is neither ``PYTHON`` nor ``RUST``.
    """
    refuse_backend(name, backend, (Backend.PYTHON, Backend.RUST))
    if backend is not Backend.RUST:
        return None
    return importlib.import_module(f"{oracle}.rust")
