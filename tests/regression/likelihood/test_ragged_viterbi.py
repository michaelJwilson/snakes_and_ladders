"""The ragged Viterbi against enumeration, the rectangular decoder, its oracle and the truth.

Issue #1138. Four referees: every path of every segment enumerated, at three
states (four for the Kronecker kinds, ``K = 2``) and lengths 2 to 6, ties
included; `hmm.viterbi` on a rectangular batch whose log-density is a
categorical family's; the NumPy oracle, which the compiled kernel matches path
for path; and the simulated hidden path at stress size.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
import torch
from sal.backend import Backend
from sal.emissions import CategoricalEmission, GaussianEmission
from sal.likelihood.hmm import viterbi as rectangular_viterbi
from sal.likelihood.ragged import (
    Paths,
    SwitchKind,
    step_transitions,
    viterbi,
    viterbi_oracle,
)
from sal.ragged import Ragged
from sal.sim.hmm import HmmParams, simulate_sequences

from tests._rows import every_value

#: Rust against NumPy on the log-joint: the kernel's step under a Kronecker
#: kind is `(delta + ln A) + ln S` where the oracle's is `delta + ln(A S)`,
#: so the two may differ in the last bits.
KERNEL_RTOL = 1e-12

KINDS = list(SwitchKind)


def _case(
    kind: SwitchKind, lengths: tuple[int, ...], seed: int
) -> tuple[Ragged, np.ndarray, np.ndarray, np.ndarray | None]:
    """Random scores and parameters, three states or `2 K` with `K = 2`."""
    rng = np.random.default_rng(seed)
    slow = 3 if kind is SwitchKind.STAY_OR_MOVE else 2
    n_states = slow if kind is SwitchKind.STAY_OR_MOVE else 2 * slow
    density = np.log(rng.random((sum(lengths), n_states)))
    initial = np.log(rng.dirichlet(np.ones(n_states)))
    transition = np.log(rng.dirichlet(np.ones(slow), size=slow))
    # The stay-or-move kind is checked with a switch here; without one in
    # the rectangular test.
    switch = rng.uniform(0.05, 0.95, size=sum(lengths))
    return Ragged(density, lengths), initial, transition, switch


def _enumerated(
    density: Ragged,
    initial: np.ndarray,
    transition: np.ndarray,
    switch: np.ndarray | None,
    kind: SwitchKind,
) -> Paths:
    """Every path of every segment scored, the best kept.

    A tie goes to the path whose last state is lowest, then the one before
    it, and so on: what a lower-state rule at every back-pointer selects. Each
    path is scored in the recursion's order, so a symmetric tie is an exact one.
    """
    n_states = density.values.shape[1]
    paths: list[int] = []
    joints: list[float] = []
    at = 0
    for segment in density.segments():
        length = len(segment)
        steps = (
            np.broadcast_to(transition, (length - 1, n_states, n_states))
            if switch is None
            else step_transitions(transition, switch[at + 1 : at + length], kind)
        )
        best: tuple[float, tuple[int, ...]] = (-np.inf, ())
        for path in product(range(n_states), repeat=length):
            score = initial[path[0]] + segment[0, path[0]]
            for t in range(1, length):
                score = score + steps[t - 1, path[t - 1], path[t]] + segment[t, path[t]]
            reverse = path[::-1]
            if (
                not best[1]
                or score > best[0]
                or (score == best[0] and reverse < best[1][::-1])
            ):
                best = (float(score), path)
        paths.extend(best[1])
        joints.append(best[0])
        at += length
    return Paths(np.asarray(paths, dtype=np.int64), np.asarray(joints))


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("backend", [Backend.RUST, Backend.PYTHON])
def test_viterbi_is_the_enumerated_best_path(
    kind: SwitchKind, backend: Backend
) -> None:
    """Lengths 2 to 6, every segment's paths enumerated: path exact, log-joint to 1e-12."""

    def check(lengths: tuple[int, ...]) -> None:
        density, initial, transition, switch = _case(kind, lengths, seed=sum(lengths))
        got = viterbi(
            density,
            initial,
            transition,
            switch=switch,
            switch_kind=kind,
            backend=backend,
        )
        want = _enumerated(density, initial, transition, switch, kind)
        np.testing.assert_array_equal(got.path, want.path)
        np.testing.assert_allclose(got.log_joint, want.log_joint, rtol=1e-12)

    every_value([(2, 3, 4, 5, 6), (6, 2), (5,)], check)


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("backend", [Backend.RUST, Backend.PYTHON])
def test_viterbi_breaks_a_tie_to_the_lower_state(
    kind: SwitchKind, backend: Backend
) -> None:
    """Scores symmetric under a swap of two states, so every best path has an exact twin.

    Stay-or-move swaps states 0 and 1 with a transition commuting with the
    swap; the Kronecker kinds swap the fast layer, which `S` commutes with
    for every `A`. The enumeration picks the twin a lower-state rule picks.
    """
    rng = np.random.default_rng(11)
    lengths = (2, 4, 6, 3)
    total = sum(lengths)
    switch = rng.uniform(0.05, 0.95, size=total)
    if kind is SwitchKind.STAY_OR_MOVE:
        # `P A P = A` for `P` the swap of states 0 and 1.
        transition = np.log(
            np.array([[0.6, 0.1, 0.3], [0.1, 0.6, 0.3], [0.2, 0.2, 0.6]])
        )
        initial = np.log(np.array([0.3, 0.3, 0.4]))
        half = np.log(rng.random((total, 2)))
        density = np.column_stack([half[:, 0], half[:, 0], half[:, 1]])
    else:
        transition = np.log(rng.dirichlet(np.ones(2), size=2))
        initial = np.log(np.repeat(rng.dirichlet(np.ones(2)), 2) / 2)
        density = np.repeat(np.log(rng.random((total, 2))), 2, axis=1)
    ragged = Ragged(density, lengths)
    got = viterbi(
        ragged, initial, transition, switch=switch, switch_kind=kind, backend=backend
    )
    want = _enumerated(ragged, initial, transition, switch, kind)
    np.testing.assert_array_equal(got.path, want.path)
    np.testing.assert_allclose(got.log_joint, want.log_joint, rtol=1e-12)
    # The tie is exercised: each segment's last state and its twin's score the
    # same, and the lower one is kept --- never state 1 under the swap of 0
    # and 1, never the odd layer under the layer swap.
    last = got.path[np.cumsum(lengths) - 1]
    if kind is SwitchKind.STAY_OR_MOVE:
        assert np.all(last != 1), last
    else:
        assert np.all(last % 2 == 0), last


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.parametrize("backend", [Backend.RUST, Backend.PYTHON])
def test_viterbi_is_the_rectangular_decoder_bitwise(backend: Backend) -> None:
    """Eight sequences of 40, a categorical family's log-density: path and total bitwise."""
    rng = np.random.default_rng(5)
    n_states, n_symbols, n_sequences, length = 3, 4, 8, 40
    log_initial = torch.log(torch.as_tensor(rng.dirichlet(np.ones(n_states))))
    log_transition = torch.log(
        torch.as_tensor(rng.dirichlet(np.ones(n_states), size=n_states))
    )
    family = CategoricalEmission(rng.dirichlet(np.ones(n_symbols), size=n_states))
    observations = rng.integers(n_symbols, size=(n_sequences, length))
    log_density = family.log_density(torch.as_tensor(observations)).numpy()

    paths = viterbi(
        Ragged(log_density.reshape(-1, n_states), (length,) * n_sequences),
        log_initial.numpy(),
        log_transition.numpy(),
        backend=backend,
    )
    for route in (Backend.PYTHON, Backend.RUST):
        states, total = rectangular_viterbi(
            observations, log_initial, log_transition, family, backend=route
        )
        np.testing.assert_array_equal(paths.path, states.reshape(-1))
        assert float(paths.log_joint.sum()) == total


@pytest.mark.critical
@pytest.mark.backend
@pytest.mark.parametrize("kind", KINDS)
def test_the_compiled_kernel_matches_the_oracle(kind: SwitchKind) -> None:
    """Uneven lengths at ten states (`K = 5` Kronecker): path bitwise, log-joint to 1e-12."""
    rng = np.random.default_rng(23)
    lengths = (2, 300, 7, 41, 1000, 2, 64)
    slow = 10 if kind is SwitchKind.STAY_OR_MOVE else 5
    n_states = slow if kind is SwitchKind.STAY_OR_MOVE else 2 * slow
    density = Ragged(np.log(rng.random((sum(lengths), n_states))), lengths)
    initial = np.log(rng.dirichlet(np.ones(n_states)))
    transition = np.log(rng.dirichlet(np.ones(slow), size=slow))
    for switch in (
        rng.uniform(size=sum(lengths)),
        None if kind is SwitchKind.STAY_OR_MOVE else np.full(sum(lengths), 0.5),
    ):
        ours = viterbi(density, initial, transition, switch=switch, switch_kind=kind)
        theirs = viterbi_oracle(density, initial, transition, switch, kind)
        np.testing.assert_array_equal(ours.path, theirs.path)
        np.testing.assert_allclose(ours.log_joint, theirs.log_joint, rtol=KERNEL_RTOL)


@pytest.mark.critical
@pytest.mark.analytic
def test_a_kronecker_kind_without_a_switch_is_refused() -> None:
    density, initial, transition, _ = _case(SwitchKind.KRONECKER, (3, 2), seed=1)
    with pytest.raises(ValueError, match="switch"):
        viterbi(density, initial, transition, switch_kind=SwitchKind.KRONECKER)
    with pytest.raises(ValueError, match="switch"):
        viterbi_oracle(density, initial, transition, None, SwitchKind.KRONECKER)


@pytest.mark.end2end
def test_viterbi_recovers_a_simulated_path_at_stress_size() -> None:
    """131,060 positions over 10 Gaussian states, decoded against the hidden path.

    Means one apart at scale 0.6 overlap: the per-position argmax of the
    density recovers 63.5% of states. A sticky chain (0.95 to stay) is what
    the recursion adds, and the decoded path recovers 97.6%; the floors are
    0.60 and 0.95, and Viterbi must beat the argmax by 0.3.
    """

    n_states = 10
    transition = np.full((n_states, n_states), 0.05 / (n_states - 1))
    np.fill_diagonal(transition, 0.95)
    lengths = (2, 3, 50, 200, 1000, 3000) * 12 + (2000,) * 40
    params = HmmParams(
        n_states=n_states,
        lengths=lengths,
        initial=np.full(n_states, 1.0 / n_states),
        transition=transition,
        emissions=GaussianEmission(
            np.arange(n_states, dtype=float),
            np.full(n_states, 0.6),
            # Read only by a refit, which this test does not make.
            variance_floor=1e-6,
        ),
        seed=1138,
        tolerance=0.0,
    )
    data = simulate_sequences(params)
    log_density = params.emissions.log_density(
        torch.as_tensor(data.observations, dtype=torch.float64)
    ).numpy()
    paths = viterbi(
        Ragged(log_density, lengths), np.log(params.initial), np.log(transition)
    )
    decoded = float(np.mean(paths.path == data.states))
    pointwise = float(np.mean(np.argmax(log_density, axis=1) == data.states))
    assert pointwise > 0.60, pointwise
    assert decoded > 0.95, decoded
    assert decoded - pointwise > 0.3, (decoded, pointwise)
