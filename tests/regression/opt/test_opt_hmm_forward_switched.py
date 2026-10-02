"""The switched differentiable forward over ragged segments, `forward_log_likelihood_ragged(..., switch, switch_kind)` (issue #1186).

Referees, under every `SwitchKind`: the summed per-segment evidence of
`sal.likelihood.ragged.posteriors` (the compiled kernel) within 1e-12; the
gradient with respect to the scores is that kernel's posterior marginal
within 1e-12 (the Fisher identity); the gradient with respect to the switch is
the central difference within the measured truncation error, and zero at a
segment's first entry, which no step reads.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sal.likelihood import ragged
from sal.likelihood.ragged import posteriors
from sal.opt.hmm import SwitchKind, forward_log_likelihood_ragged
from sal.ragged import Ragged

#: Absolute tolerance against the compiled ragged kernel (issue #1167).
ORACLE_TOLERANCE = 1e-12

#: Central-difference step, and the tolerance on the derivative it gives:
#: truncation ``h^2 f'''/6`` and rounding ``eps |f| / h`` balance near
#: h = 1e-5, where the realized maximum error over the three kinds is 1.4e-10,
#: against 1.5e-8 at h = 1e-4 and 1.0e-9 at h = 1e-6 (issue #1186).
STEP = 1e-5
DIFFERENCE_TOLERANCE = 1e-9

#: Slow states: the Kronecker kinds run over ``2 K``, stay-or-move over ``K``.
K = 3
#: Unequal segments, each at least `MINIMUM_LENGTH` so `Ragged` admits them.
LENGTHS = (7, 2, 12, 5, 9, 3)

Instance = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _instance(kind: SwitchKind, lengths: tuple[int, ...], seed: int) -> Instance:
    """Seeded scores, initial distribution, transition and switch for ``kind``."""
    generator = torch.Generator().manual_seed(seed)
    n_states = K if kind is SwitchKind.STAY_OR_MOVE else 2 * K
    total = sum(lengths)

    def draw(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=torch.float64)

    density = draw(total, n_states)
    initial = torch.log_softmax(draw(n_states), dim=0)
    transition = torch.log_softmax(draw(K, K), dim=1)
    switch = torch.rand(total, generator=generator, dtype=torch.float64)
    return density, initial, transition, switch


def _compiled(
    kind: SwitchKind, lengths: tuple[int, ...], instance: Instance
) -> ragged.Posteriors:
    """The compiled kernel's posteriors on the same instance."""
    density, initial, transition, switch = (one.detach() for one in instance)
    return posteriors(
        Ragged(values=density.numpy(), lengths=lengths),
        initial.numpy(),
        transition.numpy(),
        switch=switch.numpy(),
        switch_kind=kind,
    )


KINDS = pytest.mark.parametrize("kind", list(SwitchKind), ids=str)


@pytest.mark.critical
@pytest.mark.oracle
@KINDS
@pytest.mark.parametrize("lengths", [LENGTHS, (6,) * 5], ids=["unequal", "equal"])
def test_evidence_matches_the_compiled_kernel(
    kind: SwitchKind, lengths: tuple[int, ...]
) -> None:
    # Each segment restarts at the initial distribution and each step into
    # position t is built from switch[t]; the compiled kernel's summed
    # evidence is the referee, on the padded and on the reshaped block.
    instance = _instance(kind, lengths, 1186)
    density, initial, transition, switch = instance
    total = forward_log_likelihood_ragged(
        density, lengths, initial, transition, switch=switch, switch_kind=kind
    )
    expected = _compiled(kind, lengths, instance).log_evidence.sum()
    assert abs(float(total) - float(expected)) <= ORACLE_TOLERANCE


@pytest.mark.critical
@pytest.mark.oracle
@KINDS
def test_score_gradient_is_the_posterior_marginal(kind: SwitchKind) -> None:
    # Fisher identity: d log Z / d log_density[t, i] = P(z_t = i | x), read
    # from the compiled kernel's log posterior.
    instance = _instance(kind, LENGTHS, 1187)
    density, initial, transition, switch = instance
    density = density.clone().requires_grad_(True)
    forward_log_likelihood_ragged(
        density, LENGTHS, initial, transition, switch=switch, switch_kind=kind
    ).backward()  # type: ignore[no-untyped-call]
    assert density.grad is not None
    marginal = np.exp(_compiled(kind, LENGTHS, instance).log_posterior)
    np.testing.assert_allclose(
        density.grad.numpy(), marginal, rtol=0.0, atol=ORACLE_TOLERANCE
    )


@pytest.mark.analytic
@KINDS
def test_switch_gradient_matches_central_differences(kind: SwitchKind) -> None:
    # Every interior switch entry is moved by +-h; a segment's first entry is
    # read by no step, so its gradient is exactly zero.
    density, initial, transition, switch = _instance(kind, LENGTHS, 1188)
    variable = switch.clone().requires_grad_(True)
    forward_log_likelihood_ragged(
        density, LENGTHS, initial, transition, switch=variable, switch_kind=kind
    ).backward()  # type: ignore[no-untyped-call]
    assert variable.grad is not None
    starts = np.cumsum((0, *LENGTHS[:-1]))
    assert torch.equal(variable.grad[starts], torch.zeros(len(LENGTHS)))
    numeric = torch.zeros_like(switch)
    for t in range(switch.shape[0]):
        shift = torch.zeros_like(switch)
        shift[t] = STEP
        up, down = (
            forward_log_likelihood_ragged(
                density, LENGTHS, initial, transition, switch=moved, switch_kind=kind
            )
            for moved in (switch + shift, switch - shift)
        )
        numeric[t] = (up - down) / (2 * STEP)
    error = float((variable.grad - numeric).abs().max())
    assert error <= DIFFERENCE_TOLERANCE


@pytest.mark.analytic
def test_a_switch_of_one_is_the_unswitched_chain() -> None:
    # (1 - s) I + s A at s = 1 is A: the switched recursion and the
    # unswitched one compute the same evidence by different products.
    density, initial, _, switch = _instance(SwitchKind.STAY_OR_MOVE, LENGTHS, 1189)
    transition = torch.log_softmax(
        torch.randn(K, K, generator=torch.Generator().manual_seed(1189)), dim=1
    ).double()
    switched = forward_log_likelihood_ragged(
        density, LENGTHS, initial, transition, switch=torch.ones_like(switch)
    )
    plain = forward_log_likelihood_ragged(density, LENGTHS, initial, transition)
    assert abs(float(switched) - float(plain)) <= ORACLE_TOLERANCE


@pytest.mark.smoke
def test_switch_kind_is_one_type_in_opt_and_likelihood() -> None:
    # Defined in `opt.hmm.forward`, re-exported from `likelihood.ragged`.
    assert ragged.SwitchKind is SwitchKind


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("kind", "states", "switch_shape", "fill"),
    [
        (SwitchKind.KRONECKER, 2 * K, None, 0.5),
        (SwitchKind.KRONECKER_DIAGONAL, 2 * K + 1, (sum(LENGTHS),), 0.5),
        (SwitchKind.STAY_OR_MOVE, K, (sum(LENGTHS) - 1,), 0.5),
        (SwitchKind.STAY_OR_MOVE, K, (sum(LENGTHS),), 1.5),
    ],
    ids=["no-switch", "odd-states", "short-switch", "not-a-probability"],
)
def test_refuses_what_the_compiled_kernel_refuses(
    kind: SwitchKind,
    states: int,
    switch_shape: tuple[int, ...] | None,
    fill: float,
) -> None:
    width = K if kind is SwitchKind.STAY_OR_MOVE else states // 2
    switch = (
        None
        if switch_shape is None
        else torch.full(switch_shape, fill, dtype=torch.float64)
    )
    with pytest.raises(ValueError, match="switch"):
        forward_log_likelihood_ragged(
            torch.zeros(sum(LENGTHS), states, dtype=torch.float64),
            LENGTHS,
            torch.zeros(states, dtype=torch.float64),
            torch.zeros(width, width, dtype=torch.float64),
            switch=switch,
            switch_kind=kind,
        )
