"""Learned surrogates: point predictors, and the calibrated bounds made from them (issue #308).

An analytic bound carries a proof; a learned one carries a measurement. The
predictors here read the features and tokens an application module
assembles -- this module sees tensors and never a topology or a lattice, per
``learn/CLAUDE.md`` -- and are held to three numbers on data they were not
fitted to: the coefficient of determination, the fraction of neighbourhoods
whose best member they rank first, and the cost of a prediction against the
exact evaluation it stands in for. A :class:`CalibratedBound` shifts a point
predictor by the quantile of its held-out residuals, so it claims a bound at
a stated coverage, and ``certify`` in ``likelihood.surrogate`` checks the
claim at that coverage.

Four architectures, in rising order of what they can see. A
:class:`LinearSurrogate` and an :class:`MLPSurrogate` read one vector per
structure. A :class:`SetSurrogate`, an :class:`AttentionSurrogate` and a
:class:`GraphSurrogate` read one token per branch or per node, and are
built so no ordering of the tokens can change the value: the set model sums
an encoding of each token, the attention model has no positional signal to
break its permutation equivariance, and the graph model aggregates a node's
neighbours by a sum. That is the symmetry of the problem -- a tree is the
same tree with its children swapped or its root moved -- obeyed by the
architecture, and :func:`augment` uses the same symmetry to multiply the
training data, so a model that only reads a vector is trained on every
spelling it might meet.

Training follows the regime of ``learn/CLAUDE.md``: ``float64``, Adam,
inputs and targets standardized on the training split alone, and early
stopping on a validation split that shares no generating seed with the
training one. :func:`curriculum` fits one model through a sequence of
stages, so a predictor trained on the sizes the exact oracle can afford is
measured on the sizes it cannot, both zero-shot and after transfer.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import torch

from snakes_and_ladders.bound import Bound


@dataclass(frozen=True)
class Examples:
    """Structures as tensors: one row of ``features`` and one target each, with optional tokens.

    Parameters
    ----------
    features : torch.Tensor
        Shape ``(n, n_features)``.
    targets : torch.Tensor
        Shape ``(n,)``, the exact values being learned.
    groups : np.ndarray
        Shape ``(n,)``, an integer per example naming the neighbourhood or
        seed it came from: what a split must not straddle, and what
        :func:`argmax_agreement` ranks within.
    tokens : tuple[torch.Tensor, ...]
        One ``(n_tokens_i, n_token_features)`` tensor per example, for the
        token models; empty for a dataset that carries none.
    adjacency : tuple[np.ndarray, ...]
        One ``(n_edges_i, 2)`` edge list per example, indexing its tokens,
        for the graph model; empty otherwise.
    offset : torch.Tensor | None
        Shape ``(n,)``: an analytic value the model predicts the target's
        distance from -- a lower bound, typically -- so what is learned is
        the gap and a poor fit falls back to the bound rather than to
        nothing. ``None`` learns the target itself.
    """

    features: torch.Tensor
    targets: torch.Tensor
    groups: np.ndarray
    tokens: tuple[torch.Tensor, ...] = ()
    adjacency: tuple[np.ndarray, ...] = ()
    offset: torch.Tensor | None = None

    def __post_init__(self) -> None:
        n = self.features.shape[0]
        if self.targets.shape != (n,) or self.groups.shape != (n,):
            msg = f"{n} feature rows need {n} targets and {n} groups"
            raise ValueError(msg)
        if self.offset is not None and self.offset.shape != (n,):
            msg = f"{n} examples need {n} offsets, got {tuple(self.offset.shape)}"
            raise ValueError(msg)
        if self.tokens and len(self.tokens) != n:
            msg = f"{n} examples but {len(self.tokens)} token tensors"
            raise ValueError(msg)
        if self.adjacency and len(self.adjacency) != n:
            msg = f"{n} examples but {len(self.adjacency)} adjacency lists"
            raise ValueError(msg)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def subset(self, mask: np.ndarray) -> Examples:
        """The examples where ``mask`` is true, in order."""
        chosen = np.flatnonzero(mask)
        return Examples(
            self.features[chosen],
            self.targets[chosen],
            self.groups[chosen],
            tuple(self.tokens[i] for i in chosen) if self.tokens else (),
            tuple(self.adjacency[i] for i in chosen) if self.adjacency else (),
            None if self.offset is None else self.offset[chosen],
        )

    @property
    def residual(self) -> torch.Tensor:
        """The targets less the offset: what a model is fitted to."""
        return self.targets if self.offset is None else self.targets - self.offset

    def with_offset(self, offset: torch.Tensor) -> Examples:
        """The same examples, predicted relative to ``offset``."""
        return replace(self, offset=offset)


def concatenate(parts: Sequence[Examples]) -> Examples:
    """Examples from several sources as one dataset, groups kept distinct by offsetting."""
    offset, groups = 0, []
    for part in parts:
        groups.append(part.groups + offset)
        offset += int(part.groups.max()) + 1 if len(part) else 0
    return Examples(
        torch.cat([part.features for part in parts]),
        torch.cat([part.targets for part in parts]),
        np.concatenate(groups),
        tuple(token for part in parts for token in part.tokens),
        tuple(edges for part in parts for edges in part.adjacency),
        None
        if any(part.offset is None for part in parts)
        else torch.cat([part.offset for part in parts if part.offset is not None]),
    )


@dataclass(frozen=True)
class Split:
    """Boolean masks over one :class:`Examples`, disjoint and covering it."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def split_by_group(
    groups: np.ndarray, fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)
) -> Split:
    """Assign whole groups to train, validation and test, in group order, at the given fractions.

    A group is a generating seed or a neighbourhood; assigning it whole is
    what makes the held-out numbers about structures the model never saw
    rather than about siblings of ones it did.
    """
    if abs(sum(fractions) - 1.0) > 1e-9 or min(fractions) < 0.0:
        msg = f"fractions must be non-negative and sum to one, got {fractions}"
        raise ValueError(msg)
    unique = np.unique(groups)
    first = int(round(fractions[0] * len(unique)))
    second = first + int(round(fractions[1] * len(unique)))
    train = np.isin(groups, unique[:first])
    validation = np.isin(groups, unique[first:second])
    return Split(train, validation, ~(train | validation))


# --- models ---------------------------------------------------------------


class _Batch:
    """Token tensors of many examples flattened into one, with the example each row belongs to."""

    def __init__(self, examples: Examples) -> None:
        self.features = examples.features
        self.n = len(examples)
        if examples.tokens:
            self.tokens = torch.cat(examples.tokens)
            self.owner = torch.as_tensor(
                np.repeat(np.arange(self.n), [t.shape[0] for t in examples.tokens])
            )
            offsets = np.cumsum([0, *[t.shape[0] for t in examples.tokens[:-1]]])
            self.edges = torch.as_tensor(
                np.concatenate(
                    [
                        edges + offset
                        for edges, offset in zip(
                            examples.adjacency, offsets, strict=True
                        )
                    ]
                )
                if examples.adjacency
                else np.zeros((0, 2), dtype=np.int64)
            )
        else:
            self.tokens = torch.zeros((0, 0), dtype=torch.float64)
            self.owner = torch.zeros((0,), dtype=torch.int64)
            self.edges = torch.zeros((0, 2), dtype=torch.int64)

    def pool(self, encoded: torch.Tensor) -> torch.Tensor:
        """Sum each example's rows: the order-free reduction every token model ends with."""
        pooled = torch.zeros((self.n, encoded.shape[1]), dtype=encoded.dtype)
        return pooled.index_add(0, self.owner, encoded)

    def padded(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokens as ``(n, max_tokens, d)`` with a mask of the padding, for attention."""
        counts = torch.bincount(self.owner, minlength=self.n)
        width = int(counts.max()) if self.n else 0
        padded = torch.zeros(
            (self.n, width, self.tokens.shape[1]), dtype=self.tokens.dtype
        )
        mask = torch.ones((self.n, width), dtype=torch.bool)
        position = (
            torch.cat([torch.arange(int(c)) for c in counts])
            if self.n
            else torch.zeros(0, dtype=torch.int64)
        )
        padded[self.owner, position] = self.tokens
        mask[self.owner, position] = False
        return padded, mask


def _mlp(width_in: int, hidden: int, depth: int, width_out: int) -> torch.nn.Sequential:
    layers: list[torch.nn.Module] = []
    width = width_in
    for _ in range(depth):
        layers += [torch.nn.Linear(width, hidden), torch.nn.SiLU()]
        width = hidden
    layers.append(torch.nn.Linear(width, width_out))
    return torch.nn.Sequential(*layers).to(torch.float64)


class LinearSurrogate(torch.nn.Module):
    """Affine in the feature vector: the baseline every other model must beat."""

    reads_tokens = False

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1).to(torch.float64)

    def forward(self, batch: _Batch) -> torch.Tensor:
        out: torch.Tensor = self.linear(batch.features)[:, 0]
        return out


class MLPSurrogate(torch.nn.Module):
    """A deep MLP on the feature vector."""

    reads_tokens = False

    def __init__(self, n_features: int, *, hidden: int = 32, depth: int = 3) -> None:
        super().__init__()
        self.net = _mlp(n_features, hidden, depth, 1)

    def forward(self, batch: _Batch) -> torch.Tensor:
        out: torch.Tensor = self.net(batch.features)[:, 0]
        return out


class SetSurrogate(torch.nn.Module):
    """Deep Sets: encode each token, sum, decode, with the feature vector joined before decoding."""

    reads_tokens = True

    def __init__(
        self, n_features: int, n_token_features: int, *, hidden: int = 32
    ) -> None:
        super().__init__()
        self.encode = _mlp(n_token_features, hidden, 2, hidden)
        self.decode = _mlp(hidden + n_features, hidden, 2, 1)

    def forward(self, batch: _Batch) -> torch.Tensor:
        pooled = batch.pool(self.encode(batch.tokens))
        out: torch.Tensor = self.decode(torch.cat([pooled, batch.features], dim=1))[
            :, 0
        ]
        return out


class AttentionSurrogate(torch.nn.Module):
    """One self-attention block over the tokens, mean-pooled: permutation-equivariant because it has no positions."""

    reads_tokens = True

    def __init__(
        self,
        n_features: int,
        n_token_features: int,
        *,
        hidden: int = 32,
        heads: int = 2,
    ) -> None:
        super().__init__()
        self.embed = torch.nn.Linear(n_token_features, hidden).to(torch.float64)
        self.attention = torch.nn.MultiheadAttention(
            hidden, heads, batch_first=True, dtype=torch.float64
        )
        self.norm = torch.nn.LayerNorm(hidden, dtype=torch.float64)
        self.decode = _mlp(hidden + n_features, hidden, 2, 1)

    def forward(self, batch: _Batch) -> torch.Tensor:
        padded, mask = batch.padded()
        embedded = self.embed(padded)
        attended, _ = self.attention(
            embedded, embedded, embedded, key_padding_mask=mask
        )
        tokens = self.norm(embedded + attended)
        keep = (~mask).to(tokens.dtype)[:, :, None]
        pooled = (tokens * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1.0)
        out: torch.Tensor = self.decode(torch.cat([pooled, batch.features], dim=1))[
            :, 0
        ]
        return out


class GraphSurrogate(torch.nn.Module):
    """Message passing over the structure's own graph, neighbours summed, then pooled."""

    reads_tokens = True

    def __init__(
        self,
        n_features: int,
        n_token_features: int,
        *,
        hidden: int = 32,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.embed = torch.nn.Linear(n_token_features, hidden).to(torch.float64)
        self.layers = torch.nn.ModuleList(
            [_mlp(2 * hidden, hidden, 1, hidden) for _ in range(n_layers)]
        )
        self.decode = _mlp(hidden + n_features, hidden, 2, 1)

    def forward(self, batch: _Batch) -> torch.Tensor:
        state = torch.nn.functional.silu(self.embed(batch.tokens))
        source = torch.cat([batch.edges[:, 0], batch.edges[:, 1]])
        target = torch.cat([batch.edges[:, 1], batch.edges[:, 0]])
        for layer in self.layers:
            gathered = torch.zeros_like(state).index_add(0, target, state[source])
            state = state + layer(torch.cat([state, gathered], dim=1))
        out: torch.Tensor = self.decode(
            torch.cat([batch.pool(state), batch.features], dim=1)
        )[:, 0]
        return out


# --- training -------------------------------------------------------------


@dataclass(frozen=True)
class Standardizer:
    """Per-feature mean and scale from the training split, applied everywhere else."""

    mean: torch.Tensor
    scale: torch.Tensor

    @staticmethod
    def of(values: torch.Tensor) -> Standardizer:
        mean = values.mean(dim=0)
        scale = values.std(dim=0, unbiased=False)
        return Standardizer(mean, torch.where(scale > 0, scale, torch.ones_like(scale)))

    def apply(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.scale

    def undo(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.scale + self.mean


@dataclass(frozen=True)
class Fitted:
    """A trained predictor with the standardization it was trained under."""

    model: torch.nn.Module
    features: Standardizer
    tokens: Standardizer | None
    target: Standardizer
    train_loss: tuple[float, ...]
    validation_loss: tuple[float, ...]
    epochs: int

    kind = Bound.POINT

    def _standardized(self, examples: Examples) -> _Batch:
        tokens = (
            tuple(self.tokens.apply(t) for t in examples.tokens)
            if self.tokens is not None and examples.tokens
            else examples.tokens
        )
        return _Batch(
            replace(
                examples, features=self.features.apply(examples.features), tokens=tokens
            )
        )

    def predict(self, examples: Examples) -> torch.Tensor:
        """Predictions in the target's own units, the offset restored, no gradient."""
        with torch.no_grad():
            gap = self.target.undo(self.model(self._standardized(examples)))
        return gap if examples.offset is None else gap + examples.offset


def fit_surrogate(
    model: torch.nn.Module,
    train: Examples,
    validation: Examples,
    *,
    generator: torch.Generator,
    learning_rate: float = 1e-2,
    max_epochs: int = 300,
    patience: int = 30,
    weight_decay: float = 1e-4,
    warm: Fitted | None = None,
) -> Fitted:
    """Adam on the squared error, stopped when the validation error has not improved for ``patience`` epochs.

    Initial weights come from ``generator``; passing ``warm`` continues from
    that fit's weights and standardization instead, which is what transfer
    along a curriculum means.
    """
    if warm is None:
        for parameter in model.parameters():
            if parameter.dim() > 1:
                torch.nn.init.xavier_uniform_(parameter, generator=generator)
            else:
                torch.nn.init.zeros_(parameter)
        features = Standardizer.of(train.features)
        tokens = Standardizer.of(torch.cat(train.tokens)) if train.tokens else None
        target = Standardizer.of(train.residual)
    else:
        model.load_state_dict(warm.model.state_dict())
        features, tokens, target = warm.features, warm.tokens, warm.target
    fitted = Fitted(model, features, tokens, target, (), (), 0)
    train_batch, validation_batch = (
        fitted._standardized(train),
        fitted._standardized(validation),
    )
    train_target, validation_target = (
        target.apply(train.residual),
        target.apply(validation.residual),
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_state = {name: value.clone() for name, value in model.state_dict().items()}
    best, since_best = float("inf"), 0
    train_losses: list[float] = []
    validation_losses: list[float] = []
    for _ in range(max_epochs):
        optimizer.zero_grad()
        loss = torch.mean((model(train_batch) - train_target) ** 2)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        with torch.no_grad():
            held_out = float(
                torch.mean((model(validation_batch) - validation_target) ** 2)
            )
        train_losses.append(float(loss.detach()))
        validation_losses.append(held_out)
        if held_out < best - 1e-12:
            best, since_best = held_out, 0
            best_state = {
                name: value.clone() for name, value in model.state_dict().items()
            }
        else:
            since_best += 1
            if since_best >= patience:
                break
    model.load_state_dict(best_state)
    return Fitted(
        model,
        features,
        tokens,
        target,
        tuple(train_losses),
        tuple(validation_losses),
        len(train_losses),
    )


def r_squared(predicted: torch.Tensor, target: torch.Tensor) -> float:
    """``1 - SS_res / SS_tot``: the fraction of the target's variance the prediction explains."""
    residual = float(torch.sum((target - predicted) ** 2))
    total = float(torch.sum((target - target.mean()) ** 2))
    return 1.0 - residual / total if total > 0 else float("nan")


def argmax_agreement(
    predicted: torch.Tensor, target: torch.Tensor, groups: np.ndarray
) -> float:
    """Fraction of groups whose largest target the prediction also ranks first."""
    hits, total = 0, 0
    for group in np.unique(groups):
        members = np.flatnonzero(groups == group)
        if len(members) < 2:
            continue
        total += 1
        hits += int(torch.argmax(predicted[members]) == torch.argmax(target[members]))
    return hits / total if total else float("nan")


# --- non-analytic bounds ----------------------------------------------------


@dataclass(frozen=True)
class CalibratedBound:
    """A point predictor shifted by a residual quantile, claiming a bound at ``coverage``.

    For a lower bound the shift is the ``coverage`` quantile of
    ``predicted - exact`` on the calibration split, so the shifted value
    falls below the exact one on that fraction of calibration examples;
    exchangeability with the test examples is what carries the rate over,
    and ``certify`` at ``allowed_violation_rate = 1 - coverage`` is the
    check. It is a claim about a rate, never about one structure.
    """

    fitted: Fitted
    kind: Bound
    margin: float
    coverage: float

    def predict(self, examples: Examples) -> torch.Tensor:
        point = self.fitted.predict(examples)
        return point - self.margin if self.kind is Bound.LOWER else point + self.margin


def calibrate(
    fitted: Fitted, calibration: Examples, kind: Bound, coverage: float
) -> CalibratedBound:
    """Shift ``fitted`` into a bound holding on a ``coverage`` fraction of ``calibration``."""
    if kind is Bound.POINT:
        msg = "a calibrated bound is a lower or an upper bound"
        raise ValueError(msg)
    if not 0.0 < coverage < 1.0:
        msg = f"coverage must be strictly between 0 and 1, got {coverage}"
        raise ValueError(msg)
    residual = fitted.predict(calibration) - calibration.targets
    signed = residual if kind is Bound.LOWER else -residual
    return CalibratedBound(
        fitted, kind, float(torch.quantile(signed, coverage)), coverage
    )


# --- symmetry and curriculum ----------------------------------------------


def augment[S](
    structures: Sequence[S],
    transform: Callable[[S, np.random.Generator], S],
    rng: np.random.Generator,
    copies: int,
) -> list[S]:
    """Each structure plus ``copies`` transformed spellings of it, for training a model that is not itself invariant."""
    augmented = []
    for structure in structures:
        augmented.append(structure)
        augmented.extend(transform(structure, rng) for _ in range(copies))
    return augmented


@dataclass(frozen=True)
class Stage:
    """One rung of a curriculum: what to train on and what to validate on at that size."""

    name: str
    train: Examples
    validation: Examples


@dataclass(frozen=True)
class Curriculum:
    """The fit after each stage, so a later size can be scored zero-shot and after transfer."""

    stages: tuple[str, ...]
    fits: tuple[Fitted, ...] = field(default_factory=tuple)


def curriculum(
    make_model: Callable[[], torch.nn.Module],
    stages: Sequence[Stage],
    *,
    generator: torch.Generator,
    **options: float | int,
) -> Curriculum:
    """Fit through the stages in order, each continuing from the previous fit's weights."""
    fits: list[Fitted] = []
    for stage in stages:
        fits.append(
            fit_surrogate(
                make_model(),
                stage.train,
                stage.validation,
                generator=generator,
                warm=fits[-1] if fits else None,
                **options,  # type: ignore[arg-type]
            )
        )
    return Curriculum(tuple(stage.name for stage in stages), tuple(fits))


__all__ = [
    "AttentionSurrogate",
    "CalibratedBound",
    "Curriculum",
    "Examples",
    "Fitted",
    "GraphSurrogate",
    "LinearSurrogate",
    "MLPSurrogate",
    "SetSurrogate",
    "Split",
    "Stage",
    "Standardizer",
    "argmax_agreement",
    "augment",
    "calibrate",
    "concatenate",
    "curriculum",
    "fit_surrogate",
    "r_squared",
    "split_by_group",
]
