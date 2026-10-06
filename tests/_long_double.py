"""Forward-backward in 80-bit `np.longdouble`, the referee of the float64 kernels on long segments.

Issues #1253 and #1262. On a segment of thousands of positions the log-space
recursions' own rounding reaches 4e-11, so a float64 oracle stops being the
finer referee; this one carries about three more decimal digits and is
written independently of every kernel it judges: scaled, in probabilities,
one segment at a time.
"""

from __future__ import annotations

import numpy as np


def long_double_posteriors(
    density: np.ndarray,
    lengths: tuple[int, ...],
    initial: np.ndarray,
    transition: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scaled forward-backward in `np.longdouble`, one segment at a time, written independently of the kernel.

    `transition` is one log matrix, or a `(total, n, n)` stack whose row `t`
    is the step into position `t`, as `step_transitions` writes it.
    """
    wide = np.longdouble
    n_states = density.shape[1]
    stack = np.exp(transition.astype(wide))
    if stack.ndim == 2:
        stack = np.broadcast_to(stack, (density.shape[0], n_states, n_states))
    prior = np.exp(initial.astype(wide))
    posterior = np.empty(density.shape, dtype=wide)
    pairs = np.zeros((n_states, n_states), dtype=wide)
    evidence = []
    start = 0
    for n in lengths:
        x = density[start : start + n].astype(wide)
        steps = stack[start : start + n]
        high = x.max(axis=1, keepdims=True)
        b = np.exp(x - high)
        alpha = np.empty((n, n_states), dtype=wide)
        scale = np.empty(n, dtype=wide)
        step = prior * b[0]
        scale[0] = step.sum()
        alpha[0] = step / scale[0]
        for t in range(1, n):
            step = (alpha[t - 1] @ steps[t]) * b[t]
            scale[t] = step.sum()
            alpha[t] = step / scale[t]
        beta = np.ones(n_states, dtype=wide)
        posterior[start + n - 1] = alpha[n - 1]
        for t in range(n - 1, 0, -1):
            onward = b[t] * beta / scale[t]
            pairs += alpha[t - 1][:, None] * steps[t] * onward[None, :]
            beta = steps[t] @ onward
            posterior[start + t - 1] = alpha[t - 1] * beta
        evidence.append(np.log(scale).sum() + high.sum())
        start += n
    return posterior, pairs, np.array(evidence)


def long_double_filter(
    density: np.ndarray,
    initial: np.ndarray,
    transition: np.ndarray,
) -> np.ndarray:
    """One segment's forward filter ``p(z_t | y_1..t)`` in `np.longdouble`, rows summing to one.

    Issue #1266: the referee of the forward-filter backward-sample draws.
    The recursion of :func:`long_double_posteriors`, scaled at every step;
    `transition` is one log matrix or a `(T, n, n)` stack, as there.
    """
    wide = np.longdouble
    length, n_states = density.shape
    stack = np.exp(transition.astype(wide))
    if stack.ndim == 2:
        stack = np.broadcast_to(stack, (length, n_states, n_states))
    x = density.astype(wide)
    b = np.exp(x - x.max(axis=1, keepdims=True))
    alpha = np.empty((length, n_states), dtype=wide)
    step = np.exp(initial.astype(wide)) * b[0]
    alpha[0] = step / step.sum()
    for t in range(1, length):
        step = (alpha[t - 1] @ stack[t]) * b[t]
        alpha[t] = step / step.sum()
    return alpha
