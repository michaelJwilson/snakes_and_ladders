"""Gated curriculum training over staged labellers, under a wall budget (issue #1067).

**Data.** A buffer is a list of :class:`Example`: a field, its label and a
loss weight. Mixed-draw buffers alternate planted and solver-labelled draws
from :data:`~sal.sandbox.surrogate.potts.fields.MIXED_TRAIN`, the solver's
examples down-weighted by ``solver_weight`` against the planted ones, whose
labels are exact. Noisy buffers are expansion-labelled throughout. Buffers are
built in :func:`sal.parallel.map_tasks` workers.

**A stage** trains from the previous weights on its buffers, a side drawn per
step with the stage's probabilities, the learning rate cosine-annealed from
``lr_start`` to ``lr_end``, until its wall budget or step cap is spent. The
loss is the per-site softmax cross-entropy, averaged per field and weighted
per example.

**The gate.** A stage's weights are kept only if surrogate-plus-ICM energy on
held-out gate draws is lower than the previous weights' by more than two
standard errors, paired per field; otherwise the curriculum carries the
previous weights forward.

**Regenerating the weights.** No weights are committed. The trial's four
stages ran 150, 150, 180 and 420 s: 11x11 then 21x21 on ICM labels, then
21x21 and 41x41 on expansion labels, from buffers of 20,000, 10,000, 8,000
and 6,000 examples at batches of 16, 8, 8 and 4, on four threads. Its learning
rates were not recorded; :data:`LR_START` and :data:`LR_END` are this
module's. Build the buffers with :func:`mixed_buffer`, run :func:`curriculum`
over the four :class:`Stage` s, and save the result with
``torch.save(model.state_dict(), path)``.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from sal.parallel import map_tasks
from sal.sandbox.surrogate.potts.evaluate import FieldOf, Verdict, paired
from sal.sandbox.surrogate.potts.fields import (
    MIXED_TRAIN,
    NOISY_TRAIN,
    SeedRange,
    draw_tiling,
    lattice,
    noisy_field,
)
from sal.sandbox.surrogate.potts.labels import (
    Labeller,
    expansion_label,
    planted_label,
    solver_label,
)
from sal.sandbox.surrogate.potts.model import Grid, Surrogate, hierarchy, predict
from sal.search.icm import iterated_conditional_modes

#: The learning rate a stage starts at and anneals to.
LR_START, LR_END = 2e-3, 2e-4

#: The loss weight of a solver-labelled example against a planted one.
SOLVER_WEIGHT = 0.7

#: Batch size per lattice side: fewer fields as each grows.
BATCH = {11: 16, 21: 8, 41: 4, 71: 2, 75: 2}


@dataclass(frozen=True)
class Example:
    """One training field and its label.

    Parameters
    ----------
    field : np.ndarray
        ``float32``, ``(n_nodes, q)``.
    labelling : np.ndarray
        ``int64``, ``(n_nodes,)``.
    weight : float
        The example's loss weight.
    source : str
        The labeller, from :class:`~sal.sandbox.surrogate.potts.labels.Label`.
    """

    field: np.ndarray
    labelling: np.ndarray
    weight: float
    source: str


def mixed_example(item: tuple[int, int, bool, Labeller]) -> Example:
    """``(side, seed, planted, labeller)``: a planted draw's exact label, or a mixed draw's solver label.

    Returns
    -------
    Example
    """
    side, seed, planted, labeller = item
    graph = lattice(side)
    params = draw_tiling(graph, seed, planted=planted)
    if planted:
        label, weight = planted_label(params), 1.0
    else:
        label = solver_label(graph, params.field, labeller, seed)
        weight = SOLVER_WEIGHT
    return Example(
        params.field.astype(np.float32), label.labelling, weight, label.source
    )


def noisy_example(item: tuple[int, int]) -> Example:
    """``(side, seed)``: a noisy field and its expansion label.

    Returns
    -------
    Example
    """
    side, seed = item
    graph, values = noisy_field(side, seed)
    label = expansion_label(graph, values)
    return Example(values.astype(np.float32), label.labelling, 1.0, label.source)


def mixed_buffer(
    side: int,
    n_examples: int,
    labeller: Labeller,
    *,
    seeds: SeedRange = MIXED_TRAIN,
    workers: int = 1,
) -> list[Example]:
    """``n_examples`` draws at ``side``, the even-indexed planted and the odd solver-labelled.

    Returns
    -------
    list[Example]
    """
    items = [
        (side, seeds.seed(side, index), index % 2 == 0, labeller)
        for index in range(n_examples)
    ]
    return map_tasks(
        mixed_example,
        items,
        workers=workers,
        backend="processes" if workers > 1 else "serial",
        intra_op_threads=1,
    )


def noisy_buffer(
    side: int, n_examples: int, *, seeds: SeedRange = NOISY_TRAIN, workers: int = 1
) -> list[Example]:
    """``n_examples`` noisy fields at ``side``, expansion-labelled.

    Returns
    -------
    list[Example]
    """
    items = [(side, seeds.seed(side, index)) for index in range(n_examples)]
    return map_tasks(
        noisy_example,
        items,
        workers=workers,
        backend="processes" if workers > 1 else "serial",
        intra_op_threads=1,
    )


@dataclass(frozen=True, eq=False)
class Stage:
    """One curriculum stage.

    Parameters
    ----------
    buffers : Mapping[int, Sequence[Example]]
        Training examples per lattice side.
    probabilities : Sequence[float]
        The chance each step draws each side, in ``buffers``' order.
    seconds : float
        The wall budget.
    max_steps : int | None
        A step cap beside the budget; when given, the learning rate anneals
        over steps rather than seconds, so a capped run is reproducible.
    lr_start, lr_end : float
        The cosine schedule's ends.
    coupling_scale : float
        The buffers' lattice coupling in units of ``J0``: one for mixed
        draws, :data:`~sal.sandbox.surrogate.potts.fields.COUPLING_SCALE` for
        noisy ones.
    """

    buffers: Mapping[int, Sequence[Example]]
    probabilities: Sequence[float]
    seconds: float
    max_steps: int | None = None
    lr_start: float = LR_START
    lr_end: float = LR_END
    coupling_scale: float = 1.0


@dataclass
class TrainLog:
    """Per step: wall clock, loss and site accuracy against the label."""

    seconds: list[float] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    accuracy: list[float] = field(default_factory=list)


def _batch(
    buffer: Sequence[Example], index: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fields, labels and weights of ``buffer[index]`` as tensors."""
    fields = torch.as_tensor(np.stack([buffer[i].field for i in index]))
    labels = torch.as_tensor(np.stack([buffer[i].labelling for i in index]))
    weights = torch.as_tensor([buffer[i].weight for i in index], dtype=torch.float32)
    return fields, labels, weights


def train_stage(model: Surrogate, stage: Stage, seed: int) -> TrainLog:
    """Train ``model`` in place for one stage.

    Returns
    -------
    TrainLog
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sides = list(stage.buffers)
    grids = {side: hierarchy(lattice(side, stage.coupling_scale)) for side in sides}
    probabilities = np.asarray(stage.probabilities, dtype=float)
    optimizer = torch.optim.Adam(model.parameters(), lr=stage.lr_start)
    model.train()
    log = TrainLog()
    start = time.perf_counter()
    step = 0
    while (elapsed := time.perf_counter() - start) < stage.seconds and (
        stage.max_steps is None or step < stage.max_steps
    ):
        done = (
            elapsed / stage.seconds
            if stage.max_steps is None
            else step / stage.max_steps
        )
        rate = stage.lr_end + (stage.lr_start - stage.lr_end) * 0.5 * (
            1 + np.cos(np.pi * done)
        )
        for group in optimizer.param_groups:
            group["lr"] = rate
        side = int(rng.choice(sides, p=probabilities))
        buffer = stage.buffers[side]
        index = rng.integers(0, len(buffer), size=BATCH.get(side, 2))
        fields, labels, weights = _batch(buffer, index)
        logits = model(fields, grids[side])
        n_states = logits.shape[-1]
        per_site = nn.functional.cross_entropy(
            logits.reshape(-1, n_states), labels.reshape(-1), reduction="none"
        )
        per_field = per_site.reshape(index.size, -1).mean(dim=1)
        loss = (per_field * weights).sum() / weights.sum()
        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        step += 1
        log.seconds.append(time.perf_counter() - start)
        log.loss.append(float(loss.item()))
        log.accuracy.append(float((logits.argmax(-1) == labels).float().mean().item()))
    return log


def _polished(
    model: Surrogate, grids: list[Grid], side: int, seed: int, field_of: FieldOf
) -> float:
    """Surrogate-plus-ICM energy, arm (b), on one gate draw."""
    graph, values = field_of(side, seed)
    return iterated_conditional_modes(
        graph,
        values,
        values.shape[1],
        np.random.default_rng(0),
        start=predict(model, grids, values),
        max_sweeps=100_000,
    ).energy


def gate(
    reference: Surrogate,
    candidate: Surrogate,
    side: int,
    seeds: SeedRange,
    n_fields: int,
    field_of: FieldOf,
) -> Verdict:
    """Candidate minus reference surrogate-plus-ICM energy on ``n_fields`` gate draws.

    Returns
    -------
    Verdict
        ``"lower"`` advances the candidate.
    """
    graph, _ = field_of(side, seeds.seed(side, 0))
    grids = hierarchy(graph)
    reference.eval()
    candidate.eval()
    draws = [seeds.seed(side, index) for index in range(n_fields)]
    ref = [_polished(reference, grids, side, s, field_of) for s in draws]
    cand = [_polished(candidate, grids, side, s, field_of) for s in draws]
    return paired(np.array(cand), np.array(ref))


@dataclass(frozen=True)
class GatedStage:
    """A stage, its gate and its outcome.

    Parameters
    ----------
    stage : Stage
    gate_side : int
        The side the gate draws at.
    log : TrainLog
    verdict : Verdict
    advanced : bool
        Whether the curriculum kept the stage's weights.
    """

    stage: Stage
    gate_side: int
    log: TrainLog
    verdict: Verdict
    advanced: bool


def curriculum(
    model: Surrogate,
    stages: Sequence[tuple[Stage, int]],
    gate_seeds: SeedRange,
    n_gate_fields: int,
    field_of: FieldOf,
    seed: int = 0,
) -> tuple[Surrogate, list[GatedStage]]:
    """Train each ``(stage, gate_side)`` from the weights kept so far, keeping it only past its gate.

    Returns
    -------
    tuple[Surrogate, list[GatedStage]]
        The weights kept, and each stage's record.
    """
    kept = model
    record = []
    for offset, (stage, gate_side) in enumerate(stages):
        candidate = copy.deepcopy(kept)
        log = train_stage(candidate, stage, seed + offset)
        verdict = gate(kept, candidate, gate_side, gate_seeds, n_gate_fields, field_of)
        advanced = verdict.outcome == "lower"
        if advanced:
            kept = candidate
        record.append(GatedStage(stage, gate_side, log, verdict, advanced))
    return kept, record
