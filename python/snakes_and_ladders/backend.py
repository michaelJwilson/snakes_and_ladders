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
    `likelihood.ragged_rust`, and the sweeps of `sample.potts_mcmc`,
    `sample.annealed` and `sample.tempered`.
``NUMBA``
    `sample.gibbs` and
    `search.alpha_expansion.iterated_conditional_modes`: arithmetic that is
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

:func:`refuse_backend` is how a module declines a member it has no
implementation for. The sentence was written inline at five sites and is one
here, so a refusal reads the same wherever a caller meets it.
"""

from __future__ import annotations

from enum import StrEnum


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
