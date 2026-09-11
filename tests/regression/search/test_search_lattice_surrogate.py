"""Learned surrogates on the `spatio_only` Potts lattices (issue #365).

Three rungs of one fixture, and what referees a fit changes at each. At `ci`
enumeration gives `log Z` over 3**9 configurations; at `stress` the column
transfer matrix gives it at 72 sites; at `release` neither reaches 5,041
sites and 10 classes, so the target is the energy alpha-expansion reaches,
which the discrete solvers do compute there.

Every prediction is bracketed before it is scored. `mean_field_log_partition`
and `spanning_tree_log_partition` bracket `log Z` at the two lower rungs;
`decoupled_ground_energy` and the energy of the per-site data optimum bracket
the ground state at every size, the spanning-tree bound being one exact tree
pass per edge and so out of reach at 14,840 edges. A fit outside its bracket
is wrong whatever its held-out error.

**What is reported is the gap, and the control says why.** At `ci` the
analytic offset alone already ranks every held-out group's best first, so
`argmax_agreement` there measures the instance and not the model; at
`stress` it ranks 0.667 of them and at `release` 0.333. What a model adds at
every rung is the gap above its offset --- 0.32 to 0.49 nats over 9 sites,
2,066 to 2,200 over 5,041 --- which the offset itself says nothing about, so
the number each fit is held to is the coefficient of determination on that
gap over held-out groups.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import pytest
import torch
from snakes_and_ladders.learn.surrogate import (
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
from snakes_and_ladders.likelihood.potts import enumerate_potts, log_weights
from snakes_and_ladders.likelihood.surrogate import (
    decoupled_ground_energy,
    decoupled_log_partition,
    mean_field_log_partition,
    saturated_log_partition,
    spanning_tree_log_partition,
)
from snakes_and_ladders.search.alpha_expansion import energy
from snakes_and_ladders.search.backend import Backend
from snakes_and_ladders.search.surrogate import (
    enumerated_log_partition_target,
    ground_state_offset,
    ground_state_target,
    lattice_examples,
    lattice_instances,
    strip_log_partition_target,
)
from snakes_and_ladders.sim.fixtures import fixture
from snakes_and_ladders.sim.graph import PottsGraph
from snakes_and_ladders.sim.potts import PottsSpotsParams

#: Loaded once each: a lattice and a field per rung, from which every instance
#: below is redrawn.
CI: PottsSpotsParams = fixture("potts_spots", "ci").params
STRESS: PottsSpotsParams = fixture("potts_spots", "stress").params
RELEASE: PottsSpotsParams = fixture("potts_spots", "release").params

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
    """The decoupled energy below and the per-site data optimum's energy above.

    Both are one pass over the sites and one over the edges, so unlike the
    `log Z` bracket they exist at 5,041 sites.
    """
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
    # Enumeration over 3**9 configurations of the declared instance, against
    # the two brackets the fits below are held inside. Measured: the exact
    # value is 13.5439, mean field 13.1100 and the spanning-tree bound
    # 13.8358 either side of it, and the two decoupled bounds 10.9492 and
    # 21.8499 outside those. The decoupled pair is looser by a factor of 15
    # and costs one pass rather than one exact tree pass per edge, which is
    # why it is the pair that reaches `release`.
    field = torch.as_tensor(CI.field)
    exact = enumerate_potts(CI.graph, CI.field).log_partition

    mean_field = float(mean_field_log_partition(CI.graph, field))
    spanning = float(spanning_tree_log_partition(CI.graph, field))
    decoupled = float(decoupled_log_partition(CI.graph, field))
    saturated = float(saturated_log_partition(CI.graph, field))

    assert decoupled <= mean_field <= exact <= spanning <= saturated
    assert spanning - mean_field == pytest.approx(0.7258, abs=5e-4)
    assert saturated - decoupled == pytest.approx(10.9007, abs=5e-4)


@pytest.mark.mathematical
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
    # Six groups of four instances of the declared 3x3 family, `log Z` exact
    # by enumeration, split whole groups three ways. Both models learn the
    # gap above the mean-field bound, which spans 0.32 to 0.49 nats and which
    # the bound itself says nothing about, and both stay inside the
    # spanning-tree bracket. The control is the bound alone: it ranks every
    # held-out group's best first, so the ranking is the instance's and the
    # coefficient of determination on the gap is the model's.
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
            generator=torch.Generator().manual_seed(0),
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
    # The claim ROADMAP 2.2 asks for and issue #414 records for 3x3 to 4x6
    # lattices, on this fixture's own two square rungs: 9 sites to 72, `log Z`
    # exact by enumeration below and by the column transfer matrix above. The
    # features and the tokens are the same width at both, so one model reads
    # both without an embedding --- and it still collapses. Measured on 12
    # held-out instances of 48: zero-shot at `stress` the graph model explains
    # -338.6 of the gap and the attention model -360.8, recovering to 0.722
    # and 0.522 after training there from those weights. A `stress` fit from
    # scratch reaches 0.958 for attention and 0.091 for the graph model, so
    # the transfer helps the graph form and costs the attention one.
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
            generator=torch.Generator().manual_seed(0),
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
        )
        zero_shot = _gap_r_squared(fitted.fits[0].predict(stress_test), stress_test)
        transferred = fitted.fits[1].predict(stress_test)

        assert zero_shot < 0.0
        assert _gap_r_squared(transferred, stress_test) > floor
        assert np.all(transferred.numpy() >= lower[held_out])
        assert np.all(transferred.numpy() <= upper[held_out])


@pytest.mark.mathematical
@pytest.mark.release
def test_the_surrogates_predict_the_ground_state_energy_at_five_thousand_sites() -> (
    None
):
    # The release rung, and the first fit run at this size: the triangular
    # 71x71 lattice, 5,041 sites and 10 classes. `oracle: none` there, so the
    # target is the energy alpha expansion reaches --- what the discrete
    # solvers do compute at this size --- and the offset the decoupled energy
    # bound below it. Measured over 12 held-out instances of 48: the deep MLP
    # explains 0.515 of the gap and the linear model 0.332, and every
    # prediction of both lies inside a bracket 1.03 per site wide.
    #
    # The two token models are the finding, and it inverts what the ticket
    # expected. `GraphSurrogate` diverges here --- R^2 -47.5 over 150 epochs
    # in 646 s, -976.7 over 30 --- because `_Batch.pool` sums over nodes, so
    # the vector its decoder reads is three orders of magnitude larger at
    # 5,041 sites than at the 9 the same architecture was fitted on.
    # `AttentionSurrogate` does not run at all: one attention matrix is
    # 5,041**2 float64 = 203.3 MB and 24 training examples over two heads is
    # 9.76 GB, which the kernel killed at 9.96 GB resident, twice. Both are
    # ticketed. What holds is the fallback: the diverged graph fit is still
    # inside the bracket, because it predicts a gap above a bound rather than
    # the energy itself.
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
            generator=torch.Generator().manual_seed(0),
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
        generator=torch.Generator().manual_seed(0),
        max_epochs=30,
        patience=PATIENCE,
    ).predict(test)
    assert _gap_r_squared(diverged, test) < 0.0
    assert np.all(diverged.numpy() >= lower[held_out])
    assert np.all(diverged.numpy() <= upper[held_out])
