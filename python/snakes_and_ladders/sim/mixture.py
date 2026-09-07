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

from dataclasses import dataclass

import numpy as np

from snakes_and_ladders.emissions import GaussianEmission


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
        The per-component mean and scale.
    n_samples : int
        Observations to draw.
    seed : int
        Seed for ``np.random.default_rng``.
    tolerance : float
        Absolute tolerance a validation test checks simulated frequencies
        against their analytic counterpart within.

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


@dataclass(frozen=True)
class SimulatedMixtureDataset:
    """A simulated mixture dataset together with the labels and its truth.

    Parameters
    ----------
    labels : np.ndarray
        Component that produced each observation, shape ``(n_samples,)``.
    observations : np.ndarray
        The observations, shape ``(n_samples,)``.
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
