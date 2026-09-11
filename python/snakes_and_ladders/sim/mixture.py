"""Ground truth and data generation for a finite Gaussian mixture.

The problem class that is an HMM with the Markov chain removed: observations
are independent, each drawn from one of ``k`` components chosen by a weight
vector. Truth ships with the data on the footing
:mod:`snakes_and_ladders.sim.hmm` established --- the component label that
produced each observation is retained rather than discarded, because a dataset
without it cannot referee a clustering.

**The components are an emission family, not a second Gaussian.**
:class:`snakes_and_ladders.emissions.GaussianEmission` already carries a
per-state mean and scale, already knows how to draw from them, and already
re-estimates itself from posterior weights. A mixture needs exactly those
three things, so it uses that family rather than a copy of it (issue #262).
Whether the seam extracted from an HMM drops into a model that is not one is
the question this answers.

Fitting lives in :mod:`snakes_and_ladders.opt.mixture`, which imports the truth
type from here but draws no data itself.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from snakes_and_ladders.emissions import GaussianEmission
from snakes_and_ladders.fixtures import load_declared


@dataclass(frozen=True)
class BinInstance:
    """One declared instance of a fixture: a decimation factor and its tier.

    The counterpart of :class:`snakes_and_ladders.sim.count_pairs.BinInstance`,
    and it is the same declaration the registry reads to find a problem's key
    instance. **What a coarse instance is differs with the model.** A
    spatio-sequential fixture *sums* ``factor`` consecutive positions, because
    the positions are coupled and a sum is the coarser observation. A mixture's
    observations are independent, so there is nothing to sum: the coarse
    instance is ``n_samples // factor`` draws from the same generating
    parameters, which is distributed exactly as a subsample of the fine draw
    and costs a fraction of it.

    Parameters
    ----------
    factor : int
        Draws of the fine instance per draw of this one, ``>= 1``. ``1`` is
        the fine instance the file declares.
    marker : str
        The tier the full test at this factor runs in, measured rather than
        assumed (``DEV.md``, CI & Performance Budget).

    Raises
    ------
    ValueError
        If the factor is below one.
    """

    factor: int
    marker: str

    def __post_init__(self) -> None:
        if self.factor < 1:
            msg = f"a bin holds at least one draw, got {self.factor}"
            raise ValueError(msg)


@dataclass(frozen=True)
class MixtureParams:
    """Fully-specified truth for a Gaussian mixture fixture.

    Parameters
    ----------
    weights : np.ndarray
        Mixing weights, shape ``(n_components,)``, summing to 1 and all
        strictly positive --- a component with zero weight is not a component
        of the model, and leaving it in would make the fitted parameter for it
        undefined rather than merely uncertain.
    components : GaussianEmission
        The per-component mean and scale. A family carrying a channel axis
        makes every observation a vector, and the mixture is then over that
        many dimensions (issue #548).
    n_samples : int
        Observations to draw.
    seed : int
        Seed for ``np.random.default_rng``.
    tolerance : float
        Absolute tolerance a validation test checks simulated frequencies
        against their analytic counterpart within.
    bins : tuple[BinInstance, ...]
        The coarser instances the file declares, coarsest last; empty where
        the file declares one size only.

    Raises
    ------
    ValueError
        If the weights do not match the components, do not sum to 1, or are
        not strictly positive.
    """

    weights: np.ndarray
    components: GaussianEmission
    n_samples: int
    seed: int
    tolerance: float
    bins: tuple[BinInstance, ...] = ()

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float64)
        if weights.shape != (self.components.n_states,):
            msg = (
                f"weights have shape {weights.shape}, expected "
                f"({self.components.n_states},)"
            )
            raise ValueError(msg)
        if not np.isclose(weights.sum(), 1.0):
            msg = f"weights sum to {weights.sum()}, expected 1.0"
            raise ValueError(msg)
        if bool((weights <= 0.0).any()):
            msg = f"every weight must be positive, got {weights.tolist()}"
            raise ValueError(msg)

    @property
    def n_components(self) -> int:
        """Components in the mixture."""
        return self.components.n_states

    @property
    def n_channels(self) -> int:
        """Entries an observation carries; ``1`` for a scalar observation."""
        return self.components.n_channels

    def at(self, marker: str) -> MixtureParams:
        """The declared instance carrying ``marker``, at its own sample count.

        Parameters
        ----------
        marker : str
            The tier the instance was declared for.

        Returns
        -------
        MixtureParams
            The same generating parameters over ``n_samples // factor`` draws.

        Raises
        ------
        KeyError
            If no declared instance carries the marker.
        """
        for instance in self.bins:
            if instance.marker == marker:
                return replace(
                    self, n_samples=self.n_samples // instance.factor, bins=()
                )
        declared = [instance.marker for instance in self.bins]
        msg = f"no instance marked {marker!r}; the file declares {declared}"
        raise KeyError(msg)


@dataclass(frozen=True)
class SimulatedMixtureDataset:
    """A simulated mixture dataset together with the labels and its truth.

    Parameters
    ----------
    labels : np.ndarray
        Component that produced each observation, shape ``(n_samples,)``.
    observations : np.ndarray
        The observations, shape ``(n_samples,)``, or ``(n_samples, channels)``
        where the components carry a channel axis.
    weights : np.ndarray
        The generating mixing weights.
    components : GaussianEmission
        The generating components.
    seed : int
        Seed used.
    """

    labels: np.ndarray
    observations: np.ndarray
    weights: np.ndarray
    components: GaussianEmission
    seed: int


def simulate_mixture(
    params: MixtureParams, rng: np.random.Generator | None = None
) -> SimulatedMixtureDataset:
    """Draw component labels and observations by ancestral sampling.

    Parameters
    ----------
    params : MixtureParams
        The generating truth.
    rng : np.random.Generator | None
        Generator to draw from. ``None`` builds one from ``params.seed``,
        which is what a single-dataset fixture wants; an ensemble passes its
        own, since seeding inside the call would make every draw identical
        (``sim/CLAUDE.md``).

    Returns
    -------
    SimulatedMixtureDataset
        The labels, the observations, and the generating truth.
    """
    generator = np.random.default_rng(params.seed) if rng is None else rng
    labels = generator.choice(
        params.n_components, size=params.n_samples, p=params.weights
    )
    return SimulatedMixtureDataset(
        labels=labels,
        observations=params.components.sample(labels, generator),
        weights=params.weights,
        components=params.components,
        seed=params.seed,
    )


_REQUIRED_FIELDS = frozenset(
    {"seed", "n_samples", "tolerance", "weights", "means", "scales", "variance_floor"}
)


def load_mixture_params(path: Path) -> MixtureParams:
    """Load and validate a Gaussian-mixture fixture yaml.

    Parameters
    ----------
    path : Path
        Path to the yaml file.

    Returns
    -------
    MixtureParams
        The parsed, validated truth. The weight and component checks are
        :class:`MixtureParams`'s own, so a file and an instance built in code
        are refused on the same terms.

    Raises
    ------
    ValueError
        If a required field is missing, or ``means`` and ``scales`` differ in
        length.
    """
    raw = load_declared(path, _REQUIRED_FIELDS)

    means = np.asarray(raw["means"], dtype=np.float64)
    scales = np.asarray(raw["scales"], dtype=np.float64)
    if means.shape != scales.shape:
        msg = f"{path}: means have shape {means.shape}, scales {scales.shape}"
        raise ValueError(msg)
    bins = tuple(
        BinInstance(factor=int(entry["factor"]), marker=str(entry["marker"]))
        for entry in raw.get("bin", ())
    )

    return MixtureParams(
        weights=np.asarray(raw["weights"], dtype=np.float64),
        components=GaussianEmission(means, scales, float(raw["variance_floor"])),
        n_samples=int(raw["n_samples"]),
        seed=int(raw["seed"]),
        tolerance=float(raw["tolerance"]),
        bins=bins,
    )
