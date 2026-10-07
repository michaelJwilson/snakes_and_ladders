"""The bytes BlackJAX receives for a Hamiltonian chain (issue #1282, step 6).

One construction, read by :func:`sal.external.hmc.sample` and by the adapter
:mod:`sal.validation.blackjax`, so a chain from either path is the other's
bitwise on one key. NumPy alone, so the adapter and the script import no
torch through it.

The mode and the target travel as ``int64`` codes, an index into
:data:`MODES` and :data:`TARGETS`: a session sends ``float64``, ``int64`` or
``bool`` arrays only (:func:`sal.external.transport.admit`), and a string is
none of them.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

#: The script's modes, by code.
MODES = ("integrate", "sample", "langevin", "random_walk", "replay", "adapted")

#: The log-densities the script builds in JAX, by code; each is the negated
#: energy of the :mod:`sal.sample.declared` kernel of the same name.
TARGETS = ("gaussian", "rosenbrock", "gaussian_mixture", "gaussian_hmm")


def target_inputs(
    precision: np.ndarray | None = None,
    rosenbrock: tuple[float, float] | None = None,
    mixture: tuple[int, np.ndarray] | None = None,
    hmm: tuple[int, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """The target's inputs: a Gaussian's precision, Rosenbrock's ``(a, b)``, a mixture's ``(k, values)`` or a Gaussian HMM's ``(m, sequences)``.

    The first given of ``hmm``, ``mixture`` and ``rosenbrock`` is the
    target; with none of them, the zero-mean Gaussian of ``precision``,
    ``(d, d)`` or a ``(d,)`` diagonal. ``sequences`` are equal-length rows.
    """
    if hmm is not None:
        return {
            "target": np.asarray(TARGETS.index("gaussian_hmm"), dtype=np.int64),
            "n_states": np.asarray(hmm[0], dtype=np.int64),
            "values": np.ascontiguousarray(hmm[1], dtype=np.float64),
        }
    if mixture is not None:
        return {
            "target": np.asarray(TARGETS.index("gaussian_mixture"), dtype=np.int64),
            "n_components": np.asarray(mixture[0], dtype=np.int64),
            "values": np.ascontiguousarray(mixture[1], dtype=np.float64),
        }
    if rosenbrock is not None:
        return {
            "target": np.asarray(TARGETS.index("rosenbrock"), dtype=np.int64),
            "constants": np.asarray(rosenbrock, dtype=np.float64),
        }
    return {
        "target": np.asarray(TARGETS.index("gaussian"), dtype=np.int64),
        "precision": np.ascontiguousarray(precision, dtype=np.float64),
    }


def chain_inputs(
    target: Mapping[str, np.ndarray],
    position: np.ndarray,
    step_size: float,
    n_steps: int,
    n_draws: int,
    key: int,
    *,
    store_chain: bool = True,
    warmup: int | None = None,
    target_acceptance: float | None = None,
    initial_step_size: float | None = None,
) -> dict[str, np.ndarray]:
    """The script's inputs for ``n_draws`` HMC transitions, after a window adaptation where ``warmup`` is given.

    ``key`` is the integer JAX builds its PRNG key from in the subprocess.
    With ``warmup``, the mode is ``adapted`` and ``target_acceptance`` is
    required; ``initial_step_size`` is the warm-up's first step, and without
    it BlackJAX starts from its own default, 1.
    """
    adapted = warmup is not None
    inputs = {
        "mode": np.asarray(MODES.index("adapted" if adapted else "sample"), np.int64),
        **target,
        "position": np.ascontiguousarray(position, dtype=np.float64),
        "step_size": np.asarray(step_size, dtype=np.float64),
        "n_steps": np.asarray(n_steps, dtype=np.int64),
        "n_draws": np.asarray(n_draws, dtype=np.int64),
        "key": np.asarray(key, dtype=np.int64),
        "store_chain": np.asarray(store_chain),
    }
    if adapted:
        inputs["warmup"] = np.asarray(warmup, dtype=np.int64)
        inputs["target_acceptance"] = np.asarray(target_acceptance, dtype=np.float64)
    if initial_step_size is not None:
        inputs["initial_step_size"] = np.asarray(initial_step_size, dtype=np.float64)
    return inputs
