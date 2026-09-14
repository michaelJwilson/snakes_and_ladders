"""Two-channel count observations for the coupled model, and the 5K instance (issue #399).

The coupled spatio-sequential model of
:mod:`snakes_and_ladders.sim.spatio_sequential` at the size the roadmap asks
for: a triangular lattice of 5,041 vertices, ``M`` classes, ``K`` hidden
states per class and ``S`` sequential positions, each vertex emitting a
*pair* of counts per position --- a total, negative binomial, and a number of
successes, beta-binomial.

**The seam.** The plan's step 2 (issue #399, second amendment) makes the pair
one family, ``emissions.CountPairEmission``, in its independent and its joint
form. Until that lands :class:`IndependentCountPair` computes the independent
form here: two existing families side by side, their log-densities added. Every
consumer already takes an
:class:`~snakes_and_ladders.emissions.EmissionFamily`, so the swap is the
constructor in :func:`load_spatio_sequential_counts_params` and nothing else.

**One simulation, three instances.** The fixture is drawn once at the fine
resolution and the coarse instances are *binned* from it --- counts summed
over equal blocks along the sequential direction, in both channels --- so the
three are consistent by construction rather than by three seeds agreeing
(issue #399, third and fourth amendments). Aggregation is exact for the
negative binomial: a sum of ``f`` independent ``NB(r, p)`` counts is
``NB(f r, p)``, which scales both the mean and the dispersion by ``f``. It is
*not* exact for the beta-binomial --- a sum of beta-binomials is not
beta-binomial --- so a coarse instance's beta-binomial channel is a declared
misspecification, and what may be asserted there is label recovery and not a
parameter.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from snakes_and_ladders.emissions import (
    BetaBinomialEmission,
    EmissionFamily,
    NegativeBinomialEmission,
    Reestimate,
)
from snakes_and_ladders.fixtures import load_declared
from snakes_and_ladders.sim.graph import BoundaryCondition, triangular_lattice_graph
from snakes_and_ladders.sim.spatio_sequential import (
    SpatioSequentialParams,
)

#: Channel index of the total count, the negative binomial one.
TOTAL = 0

#: Channel index of the success count, the beta-binomial one.
SUCCESSES = 1


class IndependentCountPair:
    """A total and a number of successes, drawn independently, per hidden state.

    The independent form of the two-channel emission: the negative binomial's
    count and the beta-binomial's successes are drawn from separate streams and
    their densities multiply, with the beta-binomial's trials fixed by the state
    rather than taken from the total. The joint form --- trials equal to the
    drawn total, a coverage and its allele count --- is
    ``emissions.CountPairEmission``'s.

    An observation is a pair, so every array this family sees carries a trailing
    axis of length two: ``(..., 2)`` in, ``(..., n_states)`` out. That is the
    one shape difference from a single-channel family, and why
    :func:`snakes_and_ladders.sim.spatio_sequential.gated_log_density` reads the
    leading two axes rather than unpacking all of them.

    Parameters
    ----------
    total : NegativeBinomialEmission
        The first channel, over the non-negative integers.
    successes : BetaBinomialEmission
        The second channel, over ``{0, ..., trials[state]}``.

    Raises
    ------
    ValueError
        If the two channels carry different numbers of states: they are two
        views of one hidden state.
    """

    def __init__(
        self, total: NegativeBinomialEmission, successes: BetaBinomialEmission
    ) -> None:
        if total.n_states != successes.n_states:
            msg = (
                f"the total channel has {total.n_states} states and the success "
                f"channel {successes.n_states}; they are one hidden state"
            )
            raise ValueError(msg)
        self._total = total
        self._successes = successes

    @property
    def total(self) -> NegativeBinomialEmission:
        """The negative-binomial channel."""
        return self._total

    @property
    def successes(self) -> BetaBinomialEmission:
        """The beta-binomial channel."""
        return self._successes

    @property
    def n_states(self) -> int:
        """Hidden states this family emits from."""
        return self._total.n_states

    @property
    def is_discrete(self) -> bool:
        """True: both channels are counts."""
        return True

    @property
    def observation_dtype(self) -> torch.dtype:
        """Floating point, as both channels' ``lgamma`` terms require."""
        return torch.float64

    def sample(self, states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draw one pair per entry of ``states``, shape ``states.shape + (2,)``."""
        return np.stack(
            [self._total.sample(states, rng), self._successes.sample(states, rng)],
            axis=-1,
        )

    def log_density(self, observations: torch.Tensor) -> torch.Tensor:
        """The pair's log-density under every state: the two channels' sum."""
        return self._total.log_density(
            observations[..., TOTAL]
        ) + self._successes.log_density(observations[..., SUCCESSES])

    def bregman_divergence(self, observations: torch.Tensor) -> torch.Tensor:
        """The pair's divergence under every state: the two channels' sum.

        The channels are independent given the state, so the divergences add
        exactly as the log-densities do.

        Returns
        -------
        torch.Tensor
            Shape ``(..., n_states)``.
        """
        return self._total.bregman_divergence(
            observations[..., TOTAL]
        ) + self._successes.bregman_divergence(observations[..., SUCCESSES])

    def validate(self, observations: np.ndarray) -> None:
        """Raise unless both channels are valid counts for their own family."""
        observations = np.asarray(observations)
        if observations.shape[-1] != 2:
            msg = f"a pair has two channels, got {observations.shape[-1]}"
            raise ValueError(msg)
        self._total.validate(observations[..., TOTAL])
        self._successes.validate(observations[..., SUCCESSES])

    def reestimate(
        self, observations: torch.Tensor, posterior: torch.Tensor
    ) -> Reestimate[IndependentCountPair]:
        """Each channel's own M step, reported together.

        The channels are independent given the state, so the joint M step
        separates; the report does not, and a caller must see one refusal from
        either channel. ``converged`` is therefore the conjunction and
        ``at_boundary`` the disjunction, with the larger iteration count and
        residual of the two.
        """
        first = self._total.reestimate(observations[..., TOTAL], posterior)
        second = self._successes.reestimate(observations[..., SUCCESSES], posterior)
        return Reestimate(
            emissions=IndependentCountPair(first.emissions, second.emissions),
            converged=first.converged and second.converged,
            at_boundary=first.at_boundary or second.at_boundary,
            iterations=max(first.iterations, second.iterations),
            residual=max(first.residual, second.residual),
        )

    def alignment_key(self) -> torch.Tensor:
        """Both channels' means, shape ``(n_states, 2)``: two states emitting the same pair are one state."""
        return torch.stack([self._total.mean, self._successes.mean], dim=-1)

    def named_parameters(self) -> Mapping[str, torch.Tensor]:
        """Both channels' parameters, each prefixed by its channel."""
        named: dict[str, torch.Tensor] = {}
        for prefix, family in (
            ("total", self._total.named_parameters()),
            ("successes", self._successes.named_parameters()),
        ):
            named.update({f"{prefix}.{name}": value for name, value in family.items()})
        return named


def aggregate(family: IndependentCountPair, factor: int) -> IndependentCountPair:
    """The family a sum of ``factor`` consecutive draws is distributed under.

    Exact for the negative-binomial channel and not for the other, which is the
    point of separating them: a sum of ``f`` independent ``NB(r, mu)`` counts is
    ``NB(f r, f mu)``, while a sum of ``f`` beta-binomials is not beta-binomial.
    The second channel is returned with its trials multiplied by ``f`` and its
    ``(a, b)`` unchanged --- the binomial part of the aggregation, the closest
    member of the family to the truth and the declared misspecification a coarse
    instance carries.

    Parameters
    ----------
    family : IndependentCountPair
        The fine instance's family.
    factor : int
        Positions per bin, ``>= 1``.

    Returns
    -------
    IndependentCountPair
        The aggregated family. ``factor == 1`` returns an equal family.

    Raises
    ------
    ValueError
        If ``factor`` is below one.
    """
    if factor < 1:
        msg = f"a bin holds at least one position, got {factor}"
        raise ValueError(msg)
    return IndependentCountPair(
        NegativeBinomialEmission(
            dispersion=family.total.dispersion * factor,
            mean=family.total.mean * factor,
        ),
        BetaBinomialEmission(
            trials=family.successes.trials * factor,
            alpha=family.successes.alpha,
            beta=family.successes.beta,
        ),
    )


def binned_model(params: SpatioSequentialParams, factor: int) -> SpatioSequentialParams:
    """The model a bin factor puts the data under, without drawing the data.

    :func:`coarsen` needs the fine draw --- 384.6 MiB and 33 s at the declared
    5,041-vertex instance --- and a caller that wants only the parameters
    should not pay for it. :func:`aggregate` per class, and the position count
    divided.

    Parameters
    ----------
    params : SpatioSequentialParams
        The fine model, its emissions :class:`IndependentCountPair`.
    factor : int
        Positions per bin, dividing ``n_positions``.

    Returns
    -------
    SpatioSequentialParams
        The model at this factor.

    Raises
    ------
    ValueError
        If the factor does not divide the positions, or a family is not the
        two-channel count emission :func:`aggregate` is defined for.
    """
    if factor < 1 or params.n_positions % factor:
        msg = f"bin factor {factor} does not divide {params.n_positions} positions"
        raise ValueError(msg)
    for family in params.emissions:
        if not isinstance(family, IndependentCountPair):
            msg = (
                f"binning is defined for the two-channel count emission, not for "
                f"{type(family).__name__}"
            )
            raise ValueError(msg)
    return replace(
        params,
        n_positions=params.n_positions // factor,
        emissions=tuple(
            aggregate(family, factor)
            for family in params.emissions
            if isinstance(family, IndependentCountPair)
        ),
    )


@dataclass(frozen=True)
class BinInstance:
    """One of the instances a count-pair fixture declares.

    Parameters
    ----------
    factor : int
        Positions summed into each bin. ``1`` is the fine instance the file's
        seed draws; the others are binned from it.
    marker : str
        The tier the full test at this factor runs in --- one of
        :data:`BIN_MARKERS`. Stated in the file rather than in the test,
        because which factor fits which budget is a measurement of the
        instance (`DEV.md`, CI & Performance Budget).
    """

    factor: int
    marker: str


#: The tiers a bin instance may declare. ``key`` names the key instance: the
#: largest whose full test fits the key budget, and the one every downstream
#: study of this problem defaults to (issue #399, fifth amendment). At most
#: one instance in a file carries it, and a file whose instances all fit the
#: per-pull-request budget carries none.
BIN_MARKERS = ("ci", "release", "stress", "key")


@dataclass(frozen=True)
class SpatioSequentialCountsParams:
    """A declared count-pair instance: the model, its seed, and its bin factors.

    Parameters
    ----------
    model : SpatioSequentialParams
        The coupled model at the fine resolution, its emissions the
        two-channel families.
    seed : int
        Seeds every stream of the draw. Held here rather than passed by a
        caller because the instance *is* the draw from this seed: two callers
        drawing it differently would be two instances under one name.
    bins : tuple[BinInstance, ...]
        The declared bin factors, coarsest last.
    counts_digest : str | None
        Digest of the fine instance's counts, ``None`` in a file that has not
        recorded one. A changed simulator changes it, which is what makes the
        change visible in review rather than in a recovery number.

    Raises
    ------
    ValueError
        If a factor does not divide ``n_positions`` --- a partial last bin
        would be a different distribution from every other bin --- if a
        marker is unknown, or if the file names more than one key instance.
    """

    model: SpatioSequentialParams
    seed: int
    bins: tuple[BinInstance, ...]
    counts_digest: str | None

    def __post_init__(self) -> None:
        if not self.bins:
            msg = "a count-pair fixture declares at least one bin factor"
            raise ValueError(msg)
        for entry in self.bins:
            if entry.factor < 1 or self.model.n_positions % entry.factor:
                msg = (
                    f"bin factor {entry.factor} does not divide "
                    f"{self.model.n_positions} positions"
                )
                raise ValueError(msg)
            if entry.marker not in BIN_MARKERS:
                msg = f"bin marker {entry.marker!r} is not one of {list(BIN_MARKERS)}"
                raise ValueError(msg)
        keys = [entry.factor for entry in self.bins if entry.marker == "key"]
        if len(keys) > 1:
            msg = f"at most one bin instance is the key fixture, got {keys}"
            raise ValueError(msg)

    @property
    def factors(self) -> tuple[int, ...]:
        """The declared bin factors, in file order."""
        return tuple(entry.factor for entry in self.bins)

    @property
    def key_factor(self) -> int:
        """The bin factor of the key instance.

        Returns
        -------
        int

        Raises
        ------
        ValueError
            If the file declares no key instance. Refused rather than
            answered with the largest factor: which instance a study defaults
            to is a claim the fixture makes, and a file that makes none has
            not measured it.
        """
        for entry in self.bins:
            if entry.marker == "key":
                return entry.factor
        msg = "this fixture declares no key instance"
        raise ValueError(msg)

    def marker(self, factor: int) -> str:
        """The tier the full test at ``factor`` runs in.

        Returns
        -------
        str
            One of :data:`BIN_MARKERS`.

        Raises
        ------
        KeyError
            If the file declares no such factor.
        """
        for entry in self.bins:
            if entry.factor == factor:
                return entry.marker
        msg = f"{factor} is not a declared bin factor; {self.factors} are"
        raise KeyError(msg)


@dataclass(frozen=True)
class CountPairInstance:
    """One bin factor's data, with the model it is distributed under.

    Parameters
    ----------
    factor : int
        Positions per bin; ``1`` is the fine instance.
    params : SpatioSequentialParams
        The model at this factor: :func:`aggregate`'s families and
        ``n_positions // factor`` positions.
    labels : np.ndarray
        The planted class of every vertex, shape ``(n_nodes,)``.
    states : np.ndarray
        The *fine* hidden path of every class, shape ``(M, S)``. A bin is a
        block of it and has no single state of its own, which is why this
        stays at the fine resolution at every factor.
    observations : np.ndarray
        Counts, shape ``(S // factor, n_nodes, 2)``: channel
        :data:`TOTAL` then channel :data:`SUCCESSES`.
    """

    factor: int
    params: SpatioSequentialParams
    labels: np.ndarray
    states: np.ndarray
    observations: np.ndarray

    @property
    def nbytes(self) -> int:
        """Bytes the observations occupy, the term that decides whether an instance fits."""
        return int(self.observations.nbytes)


def planted_labels(params: SpatioSequentialParams, n_classes: int) -> np.ndarray:
    """Contiguous bands of rows, one class per band, shape ``(n_nodes,)``.

    The Potts prior is ferromagnetic, so the labelling a recovery test plants
    has to be spatially smooth or the prior fights the truth and the test
    measures that fight. Bands are the smooth labelling with the fewest
    boundary edges at a given number of classes, and they leave every class
    the same number of vertices to within one row.

    Parameters
    ----------
    params : SpatioSequentialParams
        Carries the lattice, whose first extent the bands run across.
    n_classes : int
        ``M``.

    Returns
    -------
    np.ndarray
        Entries in ``[0, M)``, ``int64``.

    Raises
    ------
    ValueError
        If the graph is not a 2-D lattice, which has no rows to band.
    """
    shape = params.graph.shape
    if shape is None or len(shape) != 2:
        msg = f"bands need a 2-D lattice, got shape {shape}"
        raise ValueError(msg)
    rows, columns = shape
    row_of = np.arange(rows * columns) // columns
    return np.asarray(np.minimum(row_of * n_classes // rows, n_classes - 1))


def chain_states(
    params: SpatioSequentialParams, rng: np.random.Generator
) -> np.ndarray:
    """Every class's hidden path, shape ``(M, S)``, as a walk on ``Z_K``.

    The shared transition is circulant --- ``t`` on the diagonal and the rest
    spread evenly --- so a step is ``+0`` with probability ``t`` and otherwise
    uniform on ``{1, ..., K - 1}``, and the path is the running sum of those
    steps modulo ``K``. That is the same chain the per-position categorical
    draw of
    :func:`~snakes_and_ladders.sim.spatio_sequential.simulate_spatio_sequential`
    produces and costs one vectorized pass instead of ``M x S`` calls, which
    at ``S = 20,000`` is the difference between a fixture that loads and one
    that does not.
    """
    n_classes, n_states, n_positions = (
        params.n_classes,
        params.n_states,
        params.n_positions,
    )
    start = np.array(
        [rng.choice(n_states, p=params.initial[m]) for m in range(n_classes)]
    )
    steps = np.where(
        rng.random((n_classes, n_positions - 1)) < params.self_transition,
        0,
        rng.integers(1, n_states, size=(n_classes, n_positions - 1)),
    )
    walk = np.concatenate([start[:, None], steps], axis=1).cumsum(axis=1)
    return np.asarray(walk % n_states, dtype=np.int64)


def simulate_count_pairs(
    declared: SpatioSequentialCountsParams,
) -> CountPairInstance:
    """Draw the fine instance: the planted labels, every class's chain, and every vertex's counts.

    **A stream per vertex.** Vertex ``v``'s counts come from
    ``default_rng([seed, v])`` and the chains from ``default_rng([seed, V])``,
    so the draw is a function of the seed and the vertex alone --- not of the
    order vertices are visited in, nor of how many threads visit them. That is
    what lets the Rust simulator
    (:mod:`snakes_and_ladders.sim.count_pairs_rust`) be checked against this
    one vertex by vertex, and it is the exception `sim/CLAUDE.md`'s "a
    generator, never a seed" rule is stated against: the seed is the
    fixture's, declared in its file, and every stream derived from it here is
    derived once.

    Parameters
    ----------
    declared : SpatioSequentialCountsParams
        The loaded fixture.

    Returns
    -------
    CountPairInstance
        The fine instance, ``factor = 1``.

    Raises
    ------
    ValueError
        If a drawn count does not fit ``uint16``. The counts are held in the
        narrow type because the fine instance is ``S x V`` of them and the
        wide one doubles a fixture that already runs to hundreds of
        megabytes; it is also the type the Rust kernels index their emission
        tables by, so the range the table covers and the range the dtype
        holds are one statement. A draw that overflows it is a fixture whose
        parameters moved, not a type to widen silently.
    """
    params = declared.model
    n_nodes = params.graph.n_nodes
    labels = planted_labels(params, params.n_classes)
    states = chain_states(params, np.random.default_rng([declared.seed, n_nodes]))

    observations = np.empty((params.n_positions, n_nodes, 2), dtype=np.uint16)
    for node in range(n_nodes):
        family = params.emissions[int(labels[node])]
        drawn = family.sample(
            states[int(labels[node])], np.random.default_rng([declared.seed, node])
        )
        if drawn.max() > np.iinfo(np.uint16).max:
            msg = (
                f"vertex {node} drew {drawn.max()}, past uint16; the declared "
                f"emission parameters have outgrown the fixture's dtype"
            )
            raise ValueError(msg)
        observations[:, node, :] = drawn
    return CountPairInstance(
        factor=1,
        params=params,
        labels=labels,
        states=states,
        observations=observations,
    )


def coarsen(fine: CountPairInstance, factor: int) -> CountPairInstance:
    """Sum both channels over blocks of ``factor`` consecutive positions.

    Parameters
    ----------
    fine : CountPairInstance
        The fine instance, ``factor = 1``.
    factor : int
        Positions per bin, dividing ``n_positions``.

    Returns
    -------
    CountPairInstance
        The binned instance, its model :func:`aggregate`'s.

    Raises
    ------
    ValueError
        If ``fine`` is not the fine instance, or ``factor`` does not divide
        its positions.
    """
    if fine.factor != 1:
        msg = f"bin from the fine instance, not from factor {fine.factor}"
        raise ValueError(msg)
    n_positions = fine.params.n_positions
    if factor < 1 or n_positions % factor:
        msg = f"bin factor {factor} does not divide {n_positions} positions"
        raise ValueError(msg)
    if factor == 1:
        return fine
    binned = fine.observations.reshape(
        n_positions // factor, factor, *fine.observations.shape[1:]
    ).sum(axis=1, dtype=np.int32)
    for family in fine.params.emissions:
        if not isinstance(family, IndependentCountPair):
            msg = (
                f"binning is defined for the two-channel count emission, not for "
                f"{type(family).__name__}"
            )
            raise ValueError(msg)
    families = tuple(
        aggregate(family, factor)
        for family in fine.params.emissions
        if isinstance(family, IndependentCountPair)
    )
    return CountPairInstance(
        factor=factor,
        params=replace(
            fine.params,
            n_positions=n_positions // factor,
            emissions=families,
        ),
        labels=fine.labels,
        states=fine.states,
        observations=binned,
    )


def counts_digest(instance: CountPairInstance) -> str:
    """A digest of an instance's counts: 16 hex characters of their SHA-256.

    Recorded in the fixture file so a changed simulator is visible in review
    rather than in a recovery number several tests later.

    Returns
    -------
    str
    """
    contiguous = np.ascontiguousarray(instance.observations)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()[:16]


_REQUIRED_FIELDS = frozenset(
    {
        "seed",
        "shape",
        "boundary",
        "coupling",
        "n_classes",
        "n_states",
        "n_positions",
        "beta",
        "self_transition",
        "initial",
        "emissions",
        "bin",
    }
)

_REQUIRED_LADDERS = ("mean", "dispersion", "trials", "rate", "concentration")


def _ladder(declared: Mapping[str, object], name: str, n_states: int) -> np.ndarray:
    """One per-state ladder from the file, checked for length.

    ``trials`` is checked for more than that: it must be the same in every
    state. The beta-binomial's *support* is ``{0, ..., n}``, so a ladder that
    varies makes a count drawn in one state impossible in another --- and the
    log-density there is not a small number but a ``lgamma`` of a negative
    argument, which is ``NaN``. The gated model evaluates every vertex under
    every state, so one such count takes a whole class's evidence to ``NaN``
    and the recovery it feeds to chance. Measured on the 5K instance before
    this check existed: two of ten classes lost their evidence and the field
    argmin fell to 0.113, the fraction of vertices in the first class.

    The number of trials is a property of the observation rather than of the
    hidden state, so declaring one value per state was the error; the classes
    are separated in this channel by ``class_rate_shift`` instead.

    Raises
    ------
    ValueError
        If the ladder is the wrong length, or --- for ``trials`` --- if it is
        not constant across the states.
    """
    values = np.asarray(declared[name], dtype=np.float64)
    if values.shape != (n_states,):
        msg = f"emissions.{name} has {values.shape} entries, expected ({n_states},)"
        raise ValueError(msg)
    if name == "trials" and not bool((values == values[0]).all()):
        msg = (
            f"emissions.trials must be the same in every state, got "
            f"{values.tolist()}: a count drawn under one state is outside "
            f"another's support, where the log-density is NaN rather than small"
        )
        raise ValueError(msg)
    return values


def load_spatio_sequential_counts_params(path: Path) -> SpatioSequentialCountsParams:
    """Load and validate a count-pair coupled fixture yaml.

    **The emissions are declared as ladders, not as a table.** ``M x K``
    families carrying five parameters each is 500 numbers at the declared 5K
    size, which no reader checks and no reviewer reads. What the file states
    instead is one ladder per parameter over the ``K`` states, shared by every
    class, and two per-class vectors that separate the classes: the negative
    binomial's mean is scaled by ``class_mean_scale[m]`` and the
    beta-binomial's rate shifted by ``class_rate_shift[m]``. Everything is
    still a literal in the file, and the number of them is ``5 K + 2 M``.

    Parameters
    ----------
    path : Path
        Path to the yaml file.

    Returns
    -------
    SpatioSequentialCountsParams
        The parsed, validated instance.

    Raises
    ------
    ValueError
        If a required field is absent, a ladder is the wrong length, a
        declared beta-binomial rate leaves ``(0, 1)`` once shifted, or the
        emission form is not one this module implements.
    """
    raw = load_declared(path, _REQUIRED_FIELDS)
    n_classes, n_states = int(raw["n_classes"]), int(raw["n_states"])

    declared_emissions = raw["emissions"]
    form = str(declared_emissions["form"])
    if form != "independent":
        msg = (
            f"{path}: emission form {form!r} is not implemented here; the joint "
            f"form arrives with emissions.CountPairEmission (issue #399)"
        )
        raise ValueError(msg)
    missing = set(_REQUIRED_LADDERS) - declared_emissions.keys()
    if missing:
        msg = f"{path}: emissions is missing {sorted(missing)}"
        raise ValueError(msg)
    ladders = {
        name: _ladder(declared_emissions, name, n_states) for name in _REQUIRED_LADDERS
    }
    scale = np.asarray(declared_emissions["class_mean_scale"], dtype=np.float64)
    shift = np.asarray(declared_emissions["class_rate_shift"], dtype=np.float64)
    for name, values in (("class_mean_scale", scale), ("class_rate_shift", shift)):
        if values.shape != (n_classes,):
            msg = (
                f"{path}: emissions.{name} has {values.shape}, expected ({n_classes},)"
            )
            raise ValueError(msg)

    families: list[EmissionFamily] = []
    for m in range(n_classes):
        rate = ladders["rate"] + shift[m]
        if bool(((rate <= 0.0) | (rate >= 1.0)).any()):
            msg = (
                f"{path}: class {m}'s beta-binomial rate leaves (0, 1) once "
                f"shifted by {shift[m]}: {rate.tolist()}"
            )
            raise ValueError(msg)
        families.append(
            IndependentCountPair(
                NegativeBinomialEmission(
                    dispersion=ladders["dispersion"],
                    mean=ladders["mean"] * scale[m],
                ),
                BetaBinomialEmission(
                    trials=ladders["trials"],
                    alpha=ladders["concentration"] * rate,
                    beta=ladders["concentration"] * (1.0 - rate),
                ),
            )
        )

    model = SpatioSequentialParams(
        graph=triangular_lattice_graph(
            (int(raw["shape"][0]), int(raw["shape"][1])),
            BoundaryCondition(str(raw["boundary"])),
            float(raw["coupling"]),
        ),
        n_classes=n_classes,
        n_states=n_states,
        n_positions=int(raw["n_positions"]),
        beta=float(raw["beta"]),
        self_transition=float(raw["self_transition"]),
        initial=np.asarray(raw["initial"], dtype=np.float64),
        emissions=tuple(families),
    )
    digest = raw.get("counts_digest")
    return SpatioSequentialCountsParams(
        model=model,
        seed=int(raw["seed"]),
        bins=tuple(
            BinInstance(factor=int(entry["factor"]), marker=str(entry["marker"]))
            for entry in raw["bin"]
        ),
        counts_digest=None if digest is None else str(digest),
    )
