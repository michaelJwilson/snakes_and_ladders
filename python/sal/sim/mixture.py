"""Ground truth and data generation for a finite Gaussian mixture.

The problem class that is an HMM with the Markov chain removed: observations
are independent, each drawn from one of ``k`` components chosen by a weight
vector. Truth ships with the data on the footing
:mod:`sal.sim.hmm` established --- the component label that
produced each observation is retained rather than discarded, because a dataset
without it cannot referee a clustering.

**The components are an emission family, not a second Gaussian.**
:class:`sal.emissions.GaussianEmission` already carries a
per-state mean and scale, already knows how to draw from them, and already
re-estimates itself from posterior weights. A mixture needs exactly those
three things, so it uses that family rather than a copy of it (issue #262).
Whether the seam extracted from an HMM drops into a model that is not one is
the question this answers.

Fitting lives in :mod:`sal.opt.mixture`, which imports the truth
type from here but draws no data itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np

from sal.emissions import EmissionFamily, GaussianEmission
from sal.fixtures import BinInstance

#: What one bin of a mixture fixture holds. A mixture's observations are
#: independent, so a coarse instance is a subsample rather than a sum, and
#: :class:`~sal.fixtures.BinInstance` carries the word.
BIN_UNIT = "draw"


_REQUIRED_FIELDS = frozenset(
    {"seed", "n_samples", "tolerance", "weights", "means", "scales", "variance_floor"}
)


@dataclass(frozen=True)
class MixtureParamsBase[F: EmissionFamily]:
    """The mixture a component family is a parameter of: weights, a family, a size.

    Two fixtures declare a mixture --- this module's Gaussian one and
    :mod:`sal.sim.emission_mixture`'s count one --- and the
    mixture itself never asks what its components are, so the fields and
    the weight check are here and the family is the type parameter (issue
    #862). What is *not* shared is the dataset each draw returns: a rung's
    declared output is its own type.

    Parameters
    ----------
    weights : np.ndarray
        Mixing weights, shape ``(n_components,)``, summing to 1 and all
        strictly positive --- a component with zero weight is not a component
        of the model, and leaving it in would make the fitted parameter for it
        undefined rather than merely uncertain.
    components : F
        The per-component emission, one state per component.
    n_samples : int
        Observations to draw.
    seed : int
        Seed for ``np.random.default_rng``.
    tolerance : float
        Tolerance a validation test reads a recovered quantity within; which
        quantity, and whether the tolerance is absolute or relative, is the
        fixture's own statement.

    Raises
    ------
    ValueError
        If the weights do not match the components, do not sum to 1, or are
        not strictly positive.
    """

    weights: np.ndarray
    components: F
    n_samples: int
    seed: int
    tolerance: float

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


def draw_mixture(
    params: MixtureParamsBase[Any], rng: np.random.Generator | None
) -> tuple[np.ndarray, np.ndarray]:
    """Ancestral sampling from a mixture: a component per observation, then the observation.

    The body both simulators ran (issue #862). Each wraps it in the dataset
    type its rung declares.

    Parameters
    ----------
    params : MixtureParamsBase[Any]
        The generating truth.
    rng : np.random.Generator | None
        Generator to draw from. ``None`` builds one from ``params.seed``,
        which is what a single-dataset fixture wants; an ensemble passes its
        own, since seeding inside the call would make every draw identical
        (``sim/CLAUDE.md``).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The component labels, shape ``(n_samples,)``, and the observations
        the components drew for them.
    """
    generator = np.random.default_rng(params.seed) if rng is None else rng
    labels = generator.choice(
        params.n_components, size=params.n_samples, p=params.weights
    )
    return labels, params.components.sample(labels, generator)


@dataclass(frozen=True)
class MixtureParams(MixtureParamsBase[GaussianEmission]):
    """Fully-specified truth for a Gaussian mixture fixture.

    :class:`MixtureParamsBase`'s fields, with the component family the
    Gaussian one and the declared coarser instances beside them.

    Parameters
    ----------
    components : GaussianEmission
        The per-component mean and scale. A family carrying a channel axis
        makes every observation a vector, and the mixture is then over that
        many dimensions (issue #548).
    tolerance : float
        Absolute tolerance a validation test checks simulated frequencies
        against their analytic counterpart within.
    bins : tuple[BinInstance, ...]
        The coarser instances the file declares, coarsest last; empty where
        the file declares one size only.
    """

    bins: tuple[BinInstance, ...] = ()

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

    #: The fields :func:`sal.fixtures.load_params` checks are present before
    #: calling :meth:`from_declared`.
    required_fields: ClassVar[frozenset[str]] = _REQUIRED_FIELDS

    @classmethod
    def from_declared(cls, declared: Mapping[str, Any], path: Path, /) -> Self:
        """Build the truth from a Gaussian-mixture fixture's declared mapping.

        ``declared`` is the mapping
        :func:`sal.fixtures.load_params` read from ``path``
        with :attr:`required_fields` present; ``path`` names the file in
        every error.

        The weight and component checks are :class:`MixtureParams`'s own, so
        a file and an instance built in code are refused on the same terms.

        Raises
        ------
        ValueError
            If a required field is missing, or ``means`` and ``scales`` differ in
            length.
        """
        means = np.asarray(declared["means"], dtype=np.float64)
        scales = np.asarray(declared["scales"], dtype=np.float64)
        if means.shape != scales.shape:
            msg = f"{path}: means have shape {means.shape}, scales {scales.shape}"
            raise ValueError(msg)
        bins = tuple(
            BinInstance(
                factor=int(entry["factor"]),
                unit=BIN_UNIT,
                marker=str(entry["marker"]),
            )
            for entry in declared.get("bin", ())
        )

        return cls(
            weights=np.asarray(declared["weights"], dtype=np.float64),
            components=GaussianEmission(
                means, scales, float(declared["variance_floor"])
            ),
            n_samples=int(declared["n_samples"]),
            seed=int(declared["seed"]),
            tolerance=float(declared["tolerance"]),
            bins=bins,
        )


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
    labels, observations = draw_mixture(params, rng)
    return SimulatedMixtureDataset(
        labels=labels,
        observations=observations,
        weights=params.weights,
        components=params.components,
        seed=params.seed,
    )
