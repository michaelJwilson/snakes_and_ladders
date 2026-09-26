"""The Potts surrogate trial, conserved: what its network, fixtures and evaluation are pinned to (issue #1067).

Referees: relabelling the states permutes the logits bitwise, the symmetry
the architecture is built on; the noisy field is reproduced from its seed to
a recorded digest, and its smoothing and coupling equal a per-edge
recomputation at the declared ``m``, ``sigma`` and ``k``; a planted draw is
expansion's labelling, as ``recovery_bound`` proves; the gap is
``sim.potts.energy`` minus ``search.trws``'s bound. A short training run on
11x11 lowers the loss. All at 11x11, one thread for the network.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator

import numpy as np
import pytest
import torch
from sal.sandbox.surrogate.potts import fields
from sal.sandbox.surrogate.potts.evaluate import evaluate_field, gaps
from sal.sandbox.surrogate.potts.labels import (
    Labeller,
    expansion_label,
    planted_label,
)
from sal.sandbox.surrogate.potts.model import Surrogate, hierarchy, predict
from sal.sandbox.surrogate.potts.train import Stage, mixed_buffer, train_stage
from sal.search.trws import trws
from sal.sim.potts import energy, tiling_field

#: The side every test runs at: 121 sites, three coarse grids of 12, 2 and 2.
SIDE = 11

#: SHA-256 of the noisy field at ``NOISY_VALIDATION.seed(11, 0)``, rounded
#: to 1e-9 so a last-bit difference in ``exp`` across platforms does not move
#: it; recorded at this commit.
NOISY_DIGEST = "28dc858078da5634bbc89d008e567d263042d6ac8c83cf7fff35c4ef04210c0d"

#: Agreement asked of two float64 routes to one smoothed noise: the same sums
#: in a different order, over four rounds.
SMOOTHING_TOLERANCE = 1e-12


@pytest.fixture
def one_thread() -> Iterator[None]:
    """Torch on one thread, restored after: the run's time and its sums are the host's otherwise."""
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.mark.analytic
@pytest.mark.usefixtures("one_thread")
@pytest.mark.parametrize("noisy", [False, True], ids=["mixed", "noisy"])
def test_relabelling_the_states_permutes_the_logits_bitwise(noisy: bool) -> None:
    # The weights act on channels and are shared across states; the one
    # state-mixing term, the mean, is summed in sorted order. So the network
    # is equivariant under every permutation of the ten states, exactly.
    torch.manual_seed(0)
    model = Surrogate(fields.BASE_COUPLING)
    model.eval()
    if noisy:
        graph, values = fields.noisy_field(SIDE, fields.NOISY_VALIDATION.seed(SIDE, 0))
    else:
        graph, values = fields.mixed_field(SIDE, fields.MIXED_VALIDATION.seed(SIDE, 0))
    grids = hierarchy(graph)
    batch = torch.as_tensor(
        np.stack([values, values[::-1].copy()]), dtype=torch.float32
    )
    rng = np.random.default_rng(1067)
    with torch.no_grad():
        logits = model(batch, grids)
        for _ in range(4):
            order = torch.as_tensor(rng.permutation(fields.N_STATES))
            assert torch.equal(model(batch[:, :, order], grids), logits[:, :, order])


@pytest.mark.smoke
@pytest.mark.snapshot
def test_the_noisy_field_is_reproduced_from_its_seed() -> None:
    seed = fields.NOISY_VALIDATION.seed(SIDE, 0)
    _, first = fields.noisy_field(SIDE, seed)
    _, again = fields.noisy_field(SIDE, seed)

    assert np.array_equal(first, again)
    digest = hashlib.sha256(np.round(first, 9).astype("<f8").tobytes()).hexdigest()
    assert digest == NOISY_DIGEST


@pytest.mark.oracle
def test_the_smoothing_and_coupling_are_the_declared_ones() -> None:
    # The chosen point, restated as the trial's params.json records it.
    assert (fields.COUPLING_SCALE, fields.NOISE_SCALE, fields.SMOOTHING_ROUNDS) == (
        2.0,
        1.5,
        4,
    )
    seed = fields.NOISY_VALIDATION.seed(SIDE, 3)
    graph, values = fields.noisy_field(SIDE, seed)
    assert np.all(graph.edge_coupling == fields.BASE_COUPLING * fields.COUPLING_SCALE)

    # The noise recomputed without the edge accumulation: a Python list of
    # each site's neighbours, each row averaged, SMOOTHING_ROUNDS times.
    params = fields.draw_tiling(graph, seed, planted=False)
    noise = np.random.default_rng([seed, 1]).standard_normal(values.shape)
    neighbours: list[list[int]] = [[i] for i in range(graph.n_nodes)]
    for i, j in graph.edges:
        neighbours[i].append(j)
        neighbours[j].append(i)
    for _ in range(fields.SMOOTHING_ROUNDS):
        noise = np.array([noise[row].mean(axis=0) for row in neighbours])
    noise /= noise.std()
    planted = tiling_field(params.tiles, params.states, params.strengths, 10)

    assert np.array_equal(planted, params.field)
    np.testing.assert_allclose(
        values, planted + fields.NOISE_SCALE * noise, rtol=0, atol=SMOOTHING_TOLERANCE
    )
    # The strengths span J0, not m J0 (the fields module's docstring).
    assert params.strengths.min() >= 0.1 * fields.BASE_COUPLING
    assert params.strengths.max() <= 10 * fields.BASE_COUPLING


@pytest.mark.analytic
def test_a_planted_draw_is_expansions_labelling() -> None:
    # Every strength clears its tile's recovery bound, so no expansion move
    # lowers a labelling that leaves a tile off its state: the planted
    # labelling is the only one alpha-expansion can stop at.
    graph = fields.lattice(SIDE)
    for index in range(3):
        params = fields.draw_tiling(
            graph, fields.MIXED_TRAIN.seed(SIDE, index), planted=True
        )
        planted = planted_label(params).labelling
        assert np.array_equal(expansion_label(graph, params.field).labelling, planted)


@pytest.mark.smoke
@pytest.mark.usefixtures("one_thread")
def test_a_short_training_run_lowers_the_loss() -> None:
    # Twenty steps of sixteen 11x11 fields, about two seconds: the first five
    # steps' loss against the last five's, from a seeded initialization.
    buffer = mixed_buffer(SIDE, 16, Labeller.EXPANSION)
    torch.manual_seed(0)
    model = Surrogate(fields.BASE_COUPLING)
    log = train_stage(model, Stage({SIDE: buffer}, [1.0], seconds=60, max_steps=20), 0)

    assert len(log.loss) == 20
    assert np.mean(log.loss[-5:]) < 0.5 * np.mean(log.loss[:5])
    assert {example.source for example in buffer} == {"planted", "expansion"}


@pytest.mark.smoke
@pytest.mark.analytic
@pytest.mark.usefixtures("one_thread")
def test_the_gap_is_energy_minus_the_trws_bound() -> None:
    torch.manual_seed(0)
    model = Surrogate(fields.BASE_COUPLING)
    model.eval()
    seed = fields.MIXED_VALIDATION.seed(SIDE, 1)
    graph, values = fields.mixed_field(SIDE, seed)
    grids = hierarchy(graph)
    record = evaluate_field(model, grids, graph, values, seed, arms=("a", "d", "f"))
    gap = gaps([record])

    bound = trws(graph, values).bound
    assert gap["a"][0] == energy(graph, values, predict(model, grids, values)) - bound
    assert gap["d"][0] == record["d"].value - bound
    # TRW-S bounds the ground state from below, so no labelling is under it.
    assert min(gap["a"][0], gap["d"][0]) >= -1e-9
