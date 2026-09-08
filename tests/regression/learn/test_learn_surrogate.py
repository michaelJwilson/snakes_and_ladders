"""The learned surrogates on data whose truth is known (issue #308).

Nothing here is a topology or a lattice: the targets are synthetic
functions of synthetic features and tokens, so every claim is about the
models and the training loop rather than about an application. A linear
target must be recovered by the linear model; a permutation-invariant target
must be recovered by the token models and their value must not move when
the tokens are permuted; a calibrated bound must hold at its coverage on
data it was not calibrated on.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from snakes_and_ladders.bound import Bound
from snakes_and_ladders.learn.surrogate import (
    AttentionSurrogate,
    Examples,
    GraphSurrogate,
    LinearSurrogate,
    MLPSurrogate,
    SetSurrogate,
    Stage,
    argmax_agreement,
    augment,
    calibrate,
    concatenate,
    curriculum,
    fit_surrogate,
    r_squared,
    split_by_group,
)

N_FEATURES, N_TOKEN_FEATURES = 3, 2


def _synthetic(
    rng: np.random.Generator, n_groups: int, per_group: int, noise: float
) -> Examples:
    """Targets ``2 x0 - x1 + 0.5 sum_tokens(t0 t1) + offset + noise``, grouped."""
    features, targets, groups, tokens, adjacency, offsets = [], [], [], [], [], []
    for group in range(n_groups):
        for _ in range(per_group):
            x = rng.normal(size=N_FEATURES)
            n_tokens = int(rng.integers(3, 7))
            t = rng.normal(size=(n_tokens, N_TOKEN_FEATURES))
            edges = np.array([(i, i + 1) for i in range(n_tokens - 1)], dtype=np.int64)
            offset = 10.0 * group
            y = 2.0 * x[0] - x[1] + 0.5 * float(np.sum(t[:, 0] * t[:, 1])) + offset
            features.append(x)
            targets.append(y + noise * rng.normal())
            groups.append(group)
            tokens.append(torch.as_tensor(t))
            adjacency.append(edges)
            offsets.append(offset)
    return Examples(
        torch.as_tensor(np.array(features)),
        torch.as_tensor(np.array(targets)),
        np.array(groups),
        tuple(tokens),
        tuple(adjacency),
        offset=torch.as_tensor(np.array(offsets)),
    )


@pytest.mark.structural
def test_split_by_group_keeps_groups_whole_and_covers_everything() -> None:
    groups = np.repeat(np.arange(10), 4)
    split = split_by_group(groups)
    assert not np.any(split.train & split.validation)
    assert not np.any(split.validation & split.test)
    assert np.all(split.train | split.validation | split.test)
    for mask in (split.train, split.validation, split.test):
        for group in np.unique(groups[mask]):
            assert np.all(mask[groups == group])
    assert len(np.unique(groups[split.train])) == 6
    with pytest.raises(ValueError, match="sum to one"):
        split_by_group(groups, (0.5, 0.5, 0.5))


@pytest.mark.simulated_truth
def test_linear_surrogate_recovers_a_linear_target_on_held_out_groups() -> None:
    # Without the token term the target is affine in the features; the
    # standardization and the offset are undone on the way out, so the
    # held-out prediction is the target to the noise.
    rng = np.random.default_rng(0)
    examples = _synthetic(rng, 12, 20, 0.0)
    plain = Examples(
        examples.features,
        examples.targets
        - torch.stack([t[:, 0] @ t[:, 1] * 0.5 for t in examples.tokens]),
        examples.groups,
        offset=examples.offset,
    )
    split = split_by_group(plain.groups)
    fitted = fit_surrogate(
        LinearSurrogate(N_FEATURES),
        plain.subset(split.train),
        plain.subset(split.validation),
        generator=torch.Generator().manual_seed(1),
        max_epochs=2000,
        patience=200,
        weight_decay=0.0,
    )
    test = plain.subset(split.test)
    assert r_squared(fitted.predict(test), test.targets) > 0.999
    assert fitted.kind is Bound.POINT


@pytest.mark.simulated_truth
@pytest.mark.parametrize(
    "make",
    [
        lambda: SetSurrogate(N_FEATURES, N_TOKEN_FEATURES),
        lambda: AttentionSurrogate(N_FEATURES, N_TOKEN_FEATURES),
        lambda: GraphSurrogate(N_FEATURES, N_TOKEN_FEATURES),
        lambda: MLPSurrogate(N_FEATURES),
    ],
    ids=["set", "attention", "graph", "mlp"],
)
def test_models_explain_the_target_on_held_out_groups(make: object) -> None:
    # The token models see the bilinear term the MLP cannot, so their
    # held-out R^2 clears 0.9 while the MLP, which reads only the feature
    # vector, is held to what the features explain.
    rng = np.random.default_rng(2)
    examples = _synthetic(rng, 12, 15, 0.05)
    split = split_by_group(examples.groups)
    fitted = fit_surrogate(
        make(),  # type: ignore[operator]
        examples.subset(split.train),
        examples.subset(split.validation),
        generator=torch.Generator().manual_seed(3),
        max_epochs=200,
        patience=40,
    )
    test = examples.subset(split.test)
    score = r_squared(fitted.predict(test), test.targets)
    assert score > (0.9 if fitted.model.reads_tokens else 0.5), score
    assert fitted.epochs <= 200


@pytest.mark.mathematical
@pytest.mark.parametrize(
    "make",
    [
        lambda: SetSurrogate(N_FEATURES, N_TOKEN_FEATURES),
        lambda: AttentionSurrogate(N_FEATURES, N_TOKEN_FEATURES),
        lambda: GraphSurrogate(N_FEATURES, N_TOKEN_FEATURES),
    ],
    ids=["set", "attention", "graph"],
)
def test_token_models_are_invariant_to_token_order(make: object) -> None:
    # Permuting an example's tokens, and relabelling its edges to match, is
    # the same example; the architectures are built so the value cannot move.
    rng = np.random.default_rng(4)
    examples = _synthetic(rng, 2, 3, 0.0)
    fitted = fit_surrogate(
        make(),  # type: ignore[operator]
        examples,
        examples,
        generator=torch.Generator().manual_seed(5),
        max_epochs=5,
    )
    before = fitted.predict(examples)
    permuted_tokens, permuted_edges = [], []
    for tokens, edges in zip(examples.tokens, examples.adjacency, strict=True):
        order = rng.permutation(tokens.shape[0])
        inverse = np.argsort(order)
        permuted_tokens.append(tokens[order])
        permuted_edges.append(inverse[edges])
    permuted = Examples(
        examples.features,
        examples.targets,
        examples.groups,
        tuple(permuted_tokens),
        tuple(permuted_edges),
        offset=examples.offset,
    )
    torch.testing.assert_close(fitted.predict(permuted), before, rtol=0.0, atol=1e-10)


@pytest.mark.simulated_truth
def test_calibrated_bound_holds_at_its_coverage_on_fresh_groups() -> None:
    # Calibrated at 0.9 on 8 groups, the lower bound is above the truth on
    # no more than a fifth of 400 fresh examples: the nominal 10% plus the
    # sampling margin a rate claim on 200 calibration points carries.
    rng = np.random.default_rng(6)
    examples = _synthetic(rng, 24, 25, 1.0)
    split = split_by_group(examples.groups, (1 / 3, 1 / 3, 1 / 3))
    fitted = fit_surrogate(
        LinearSurrogate(N_FEATURES),
        examples.subset(split.train),
        examples.subset(split.validation),
        generator=torch.Generator().manual_seed(7),
    )
    test = examples.subset(split.test)
    for kind in (Bound.LOWER, Bound.UPPER):
        bound = calibrate(fitted, examples.subset(split.validation), kind, 0.9)
        predicted = bound.predict(test)
        violated = (
            predicted > test.targets
            if kind is Bound.LOWER
            else predicted < test.targets
        )
        rate = float(violated.double().mean())
        assert rate <= 0.2, (kind, rate)
        assert bound.margin > 0.0
    with pytest.raises(ValueError, match="lower or an upper"):
        calibrate(fitted, test, Bound.POINT, 0.9)
    with pytest.raises(ValueError, match="strictly between"):
        calibrate(fitted, test, Bound.LOWER, 1.0)


@pytest.mark.mathematical
def test_argmax_agreement_and_r_squared_score_what_they_say() -> None:
    target = torch.tensor([1.0, 3.0, 2.0, 5.0, 4.0, 6.0])
    groups = np.array([0, 0, 0, 1, 1, 1])
    assert argmax_agreement(target, target, groups) == 1.0
    assert argmax_agreement(-target, target, groups) == 0.0
    assert r_squared(target, target) == 1.0
    assert r_squared(torch.full_like(target, float(target.mean())), target) == 0.0


@pytest.mark.structural
def test_curriculum_carries_weights_and_standardization_forward() -> None:
    rng = np.random.default_rng(8)
    first, second = _synthetic(rng, 4, 10, 0.1), _synthetic(rng, 4, 10, 0.1)
    fits = curriculum(
        lambda: MLPSurrogate(N_FEATURES),
        [Stage("a", first, first), Stage("b", second, second)],
        generator=torch.Generator().manual_seed(9),
        max_epochs=3,
    )
    assert fits.stages == ("a", "b")
    torch.testing.assert_close(fits.fits[1].features.mean, fits.fits[0].features.mean)
    torch.testing.assert_close(fits.fits[1].target.scale, fits.fits[0].target.scale)
    joined = concatenate([first, second])
    assert len(joined) == 80
    assert len(np.unique(joined.groups)) == 8


@pytest.mark.structural
def test_augment_keeps_the_original_and_adds_the_copies() -> None:
    rng = np.random.default_rng(10)
    augmented = augment([1, 2], lambda x, r: x + int(r.integers(10, 20)), rng, 2)
    assert augmented[0] == 1
    assert augmented[3] == 2
    assert len(augmented) == 6


@pytest.mark.edge_case
def test_examples_refuse_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="need 2 targets"):
        Examples(torch.zeros((2, 3)), torch.zeros(3), np.zeros(2, dtype=np.int64))
    with pytest.raises(ValueError, match="token tensors"):
        Examples(
            torch.zeros((2, 3)),
            torch.zeros(2),
            np.zeros(2, dtype=np.int64),
            (torch.zeros((1, 2)),),
        )
    with pytest.raises(ValueError, match="offsets"):
        Examples(
            torch.zeros((2, 3)),
            torch.zeros(2),
            np.zeros(2, dtype=np.int64),
            offset=torch.zeros(3),
        )
