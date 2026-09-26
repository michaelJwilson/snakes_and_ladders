"""Learned surrogates on the `spatio_only` Potts lattices (issue #365).

`log Z` is exact by enumeration at `ci` (3**9) and by transfer matrix at
`stress` (72 sites); at `release` (5,041 sites, 10 classes) the target is the
alpha-expansion energy. Every prediction is bracketed: mean-field and
spanning-tree bounds on `log Z` below `release`, the decoupled energy and the
per-site data optimum's at every size. The offset alone ranks the held-out
best first at `ci` (0.667 at `stress`, 0.333 at `release`), so each fit is
held to R^2 on the gap above it: 0.32 to 0.49 nats at 9 sites, 2,066 to
2,200 at 5,041.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import pytest
import torch
from sal.backend import Backend
from sal.learn.ranking import (
    enumerated_log_partition_target,
    ground_state_offset,
    ground_state_target,
    lattice_examples,
    lattice_instances,
    strip_log_partition_target,
)
from sal.learn.surrogate import (
    AttentionSurrogate,
    Examples,
    GraphSurrogate,
    LinearSurrogate,
    MLPSurrogate,
    Stage,
    argmax_agreement,
    curriculum,
    fit_surrogate,
    r_squared,
    split_by_group,
)
from sal.likelihood.potts import enumerate_potts, log_weights
from sal.likelihood.surrogate import (
    decoupled_ground_energy,
    decoupled_log_partition,
    mean_field_log_partition,
    saturated_log_partition,
    spanning_tree_log_partition,
)
from sal.sim.fixtures import fixture
from sal.sim.graph import PottsGraph
from sal.sim.potts import SpatioOnlyParams, energy

#: Loaded once each: a lattice and a field per rung, from which every instance
#: below is redrawn.
CI: SpatioOnlyParams = fixture("spatio_only", "ci").params
STRESS: SpatioOnlyParams = fixture("spatio_only", "stress").params
RELEASE: SpatioOnlyParams = fixture("spatio_only", "release").params

#: Adam for at most this many epochs, stopped this long after the validation
#: error last improved. Both are below `fit_surrogate`'s own defaults, and the
#: fits below stop on patience rather than on the cap.
MAX_EPOCHS = 150
PATIENCE = 25


def _split(examples: Examples) -> tuple[Examples, Examples, Examples]:
    """Whole groups to train, validate and hold out, at one half and two quarters."""
    split = split_by_group(examples.groups, (0.5, 0.25, 0.25))
    return (
        examples.subset(split.train),
        examples.subset(split.validation),
        examples.subset(split.test),
    )


def _gap_r_squared(predicted: torch.Tensor, examples: Examples) -> float:
    """How much of the gap above the offset the prediction explains, on held-out groups."""
    offset = examples.offset
    assert offset is not None
    return r_squared(predicted - offset, examples.residual)


def _log_partition_bracket(
    graphs: Sequence[PottsGraph], fields: Sequence[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Mean field below and the spanning-tree bound above, per instance."""
    lower = [
        float(mean_field_log_partition(graph, torch.as_tensor(field)))
        for graph, field in zip(graphs, fields, strict=True)
    ]
    upper = [
        float(spanning_tree_log_partition(graph, torch.as_tensor(field)))
        for graph, field in zip(graphs, fields, strict=True)
    ]
    return np.array(lower), np.array(upper)


def _energy_bracket(
    graphs: Sequence[PottsGraph], fields: Sequence[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """The decoupled energy below, the per-site data optimum's above: any size."""
    lower = [
        float(decoupled_ground_energy(graph, torch.as_tensor(field)))
        for graph, field in zip(graphs, fields, strict=True)
    ]
    upper = [
        energy(graph, field, np.asarray(field).argmax(axis=1))
        for graph, field in zip(graphs, fields, strict=True)
    ]
    return np.array(lower), np.array(upper)


# --- the bounds -------------------------------------------------------------


@pytest.mark.oracle
def test_four_bounds_bracket_the_enumerated_log_partition() -> None:
    # Measured: exact 13.5439; mean field 13.1100, spanning tree 13.8358;
    # decoupled 10.9492 and 21.8499, 15x looser at one pass, so they reach `release`.
    field = torch.as_tensor(CI.field)
    exact = enumerate_potts(CI.graph, CI.field).log_partition

    mean_field = float(mean_field_log_partition(CI.graph, field))
    spanning = float(spanning_tree_log_partition(CI.graph, field))
    decoupled = float(decoupled_log_partition(CI.graph, field))
    saturated = float(saturated_log_partition(CI.graph, field))

    assert decoupled <= mean_field <= exact <= spanning <= saturated
    assert spanning - mean_field == pytest.approx(0.7258, abs=5e-4)
    assert saturated - decoupled == pytest.approx(10.9007, abs=5e-4)


@pytest.mark.analytic
def test_the_decoupled_energy_bound_lies_below_every_configuration() -> None:
    # The bound is the sum of each term's own optimum, so no configuration
    # can beat it; a labelling attaining all of them at once need not exist,
    # which is the slack the ground-state target sits in. Checked against
    # every one of the 19,683 configurations rather than against a sample.
    bound = float(decoupled_ground_energy(CI.graph, torch.as_tensor(CI.field)))
    configurations = np.array(
        list(itertools.product(range(CI.n_classes), repeat=CI.graph.n_nodes)),
        dtype=np.int64,
    )
    minimum = -float(log_weights(CI.graph, CI.field, configurations).max())

    assert bound <= minimum
    assert minimum - bound == pytest.approx(4.0260, abs=5e-4)


# --- the fits ---------------------------------------------------------------


@pytest.mark.oracle
def test_a_surrogate_learns_the_gap_above_the_mean_field_bound_at_nine_sites() -> None:
    # Six groups of four 3x3 instances, whole groups split three ways; both
    # models learn the 0.32-0.49 nat gap above mean field inside the tree
    # bracket. The bound alone ranks every held-out best first.
    graphs, fields, groups = lattice_instances(CI, 6, 4)
    examples = lattice_examples(
        graphs, fields, enumerated_log_partition_target(), groups=groups
    )
    train, validation, test = _split(examples)
    assert bool(torch.all(examples.residual >= 0.0))

    lower, upper = _log_partition_bracket(graphs, fields)
    held_out = np.flatnonzero(split_by_group(examples.groups, (0.5, 0.25, 0.25)).test)
    n_features = examples.features.shape[1]
    n_token_features = examples.tokens[0].shape[1]
    for model in (
        LinearSurrogate(n_features),
        GraphSurrogate(n_features, n_token_features),
    ):
        fitted = fit_surrogate(
            model,
            train,
            validation,
            rng=torch.Generator().manual_seed(0),
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
        )
        predicted = fitted.predict(test)
        assert _gap_r_squared(predicted, test) > 0.7
        assert np.all(predicted.numpy() >= lower[held_out])
        assert np.all(predicted.numpy() <= upper[held_out])

    assert test.offset is not None
    assert argmax_agreement(test.offset, test.targets, test.groups) == 1.0


@pytest.mark.oracle
@pytest.mark.release
def test_a_ci_fit_collapses_at_the_transfer_matrix_rung_and_recovers_on_transfer() -> (
    None
):
    # ROADMAP 2.2 (#414), 9 sites to 72. Over 12 held-out instances of 48:
    # zero-shot R^2 -338.6 (graph) and -360.8 (attention), 0.722 and 0.522
    # fine-tuned; from scratch at `stress` 0.091 and 0.958. Transfer helps the
    # graph form and costs the attention one.
    small = lattice_instances(CI, 12, 4)
    large = lattice_instances(STRESS, 12, 4)
    at_ci = lattice_examples(
        small[0], small[1], enumerated_log_partition_target(), groups=small[2]
    )
    at_stress = lattice_examples(
        large[0], large[1], strip_log_partition_target(), groups=large[2]
    )
    ci_train, ci_validation, _ = _split(at_ci)
    stress_train, stress_validation, stress_test = _split(at_stress)

    lower, upper = _log_partition_bracket(large[0], large[1])
    held_out = np.flatnonzero(split_by_group(at_stress.groups, (0.5, 0.25, 0.25)).test)
    n_features = at_ci.features.shape[1]
    n_token_features = at_ci.tokens[0].shape[1]
    for make, floor in (
        (lambda: GraphSurrogate(n_features, n_token_features), 0.6),
        (lambda: AttentionSurrogate(n_features, n_token_features), 0.4),
    ):
        fitted = curriculum(
            make,
            [
                Stage("ci", ci_train, ci_validation),
                Stage("stress", stress_train, stress_validation),
            ],
            rng=torch.Generator().manual_seed(0),
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
        )
        zero_shot = _gap_r_squared(fitted.fits[0].predict(stress_test), stress_test)
        transferred = fitted.fits[1].predict(stress_test)

        assert zero_shot < 0.0
        assert _gap_r_squared(transferred, stress_test) > floor
        assert np.all(transferred.numpy() >= lower[held_out])
        assert np.all(transferred.numpy() <= upper[held_out])


@pytest.mark.analytic
@pytest.mark.release
def test_the_surrogates_predict_the_ground_state_energy_at_five_thousand_sites() -> (
    None
):
    # 71x71 triangular, 5,041 sites, 10 classes; target the alpha-expansion
    # energy, offset the decoupled bound. Over 12 held-out of 48: deep MLP
    # 0.515, linear 0.332, all inside a 1.03-per-site bracket. `GraphSurrogate`
    # diverges (R^2 -47.5 at 150 epochs, -976.7 at 30: `_Batch.pool` sums over
    # nodes); `AttentionSurrogate` needs 203.3 MB per matrix, 9.76 GB for 24
    # examples, and was killed at 9.96 GB, twice. Both ticketed; the diverged
    # fit stays inside the bracket, predicting a gap above a bound.
    graphs, fields, groups = lattice_instances(RELEASE, 12, 4)
    examples = lattice_examples(
        graphs,
        fields,
        ground_state_target(RELEASE.n_classes, backend=Backend.RUST),
        groups=groups,
    ).with_offset(ground_state_offset(graphs, fields))
    train, validation, test = _split(examples)
    assert bool(torch.all(examples.residual >= 0.0))

    lower, upper = _energy_bracket(graphs, fields)
    held_out = np.flatnonzero(split_by_group(examples.groups, (0.5, 0.25, 0.25)).test)
    n_features = examples.features.shape[1]
    n_token_features = examples.tokens[0].shape[1]

    for model, floor in (
        (LinearSurrogate(n_features), 0.25),
        (MLPSurrogate(n_features), 0.45),
    ):
        fitted = fit_surrogate(
            model,
            train,
            validation,
            rng=torch.Generator().manual_seed(0),
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
        )
        predicted = fitted.predict(test)
        assert _gap_r_squared(predicted, test) > floor
        assert np.all(predicted.numpy() >= lower[held_out])
        assert np.all(predicted.numpy() <= upper[held_out])

    diverged = fit_surrogate(
        GraphSurrogate(n_features, n_token_features),
        train,
        validation,
        rng=torch.Generator().manual_seed(0),
        max_epochs=30,
        patience=PATIENCE,
    ).predict(test)
    assert _gap_r_squared(diverged, test) < 0.0
    assert np.all(diverged.numpy() >= lower[held_out])
    assert np.all(diverged.numpy() <= upper[held_out])
