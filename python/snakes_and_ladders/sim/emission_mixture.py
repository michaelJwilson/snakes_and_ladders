"""Ground truth and data generation for a mixture of count emissions.

:mod:`snakes_and_ladders.sim.mixture` is this model with a Gaussian family.
What changes here is the observation: a component emits a
:class:`snakes_and_ladders.emissions.CountPairEmission` pair --- a sequencing
depth and the allele count within it --- so the mixture is over count families
whose M step is itself an optimization rather than a formula, and whose
observation is not a scalar. Everything the mixture itself does is unchanged,
which is the claim: the component family is a parameter of the problem and not
part of it.

**Truth ships with the data**, on the footing
:mod:`snakes_and_ladders.sim.hmm` established: the component that produced
each observation is retained, because a dataset without it cannot referee a
clustering.

Fitting lives in :mod:`snakes_and_ladders.opt.emission_mixture`, which imports
the truth type from here and draws no data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from snakes_and_ladders.emissions import CountPairEmission, EmissionFamily
from snakes_and_ladders.fixtures import load_declared

#: The component families a fixture may declare, and the constructor
#: arguments each reads. Both are :class:`CountPairEmission`, one per form:
#: the form is part of the file's name for the family rather than a separate
#: flag, because the two are different generative models and a file that
#: declared the family without the form would state half a model. A count
#: family with a scalar observation is added here when a fixture needs one;
#: the mixture itself never asks what its components are.
FAMILIES = ("count-pair-joint", "count-pair-independent")


@dataclass(frozen=True)
class EmissionMixtureParams:
    """Fully-specified truth for a count-emission mixture fixture.

    Parameters
    ----------
    weights : np.ndarray
        Mixing weights, shape ``(n_components,)``, summing to 1 and all
        strictly positive --- a component with zero weight is not a component
        of the model, and leaving it in would make the fitted parameter for it
        undefined rather than merely uncertain.
    components : EmissionFamily
        The per-component emission, one state per component.
    n_samples : int
        Observations to draw.
    seed : int
        Seed for ``np.random.default_rng``.
    tolerance : float
        Relative tolerance a validation test checks a recovered weight or
        component parameter against its planted value within.

    Raises
    ------
    ValueError
        If the weights do not match the components, do not sum to 1, or are
        not strictly positive.
    """

    weights: np.ndarray
    components: EmissionFamily
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
class SimulatedEmissionMixtureDataset:
    """A simulated dataset together with the labels and its truth.

    Parameters
    ----------
    labels : np.ndarray
        Component that produced each observation, shape ``(n_samples,)``.
    observations : np.ndarray
        The observations, shape ``(n_samples,)`` for a family with a scalar
        observation and ``(n_samples, channels)`` for one whose observation is
        a tuple.
    weights : np.ndarray
        The generating mixing weights.
    components : EmissionFamily
        The generating components.
    seed : int
        Seed used.
    """

    labels: np.ndarray
    observations: np.ndarray
    weights: np.ndarray
    components: EmissionFamily
    seed: int


def simulate_emission_mixture(
    params: EmissionMixtureParams, rng: np.random.Generator | None = None
) -> SimulatedEmissionMixtureDataset:
    """Draw component labels and observations by ancestral sampling.

    Parameters
    ----------
    params : EmissionMixtureParams
        The generating truth.
    rng : np.random.Generator | None
        Generator to draw from. ``None`` builds one from ``params.seed``,
        which is what a single-dataset fixture wants; an ensemble passes its
        own, since seeding inside the call would make every draw identical
        (``sim/CLAUDE.md``).

    Returns
    -------
    SimulatedEmissionMixtureDataset
        The labels, the observations, and the generating truth.
    """
    generator = np.random.default_rng(params.seed) if rng is None else rng
    labels = generator.choice(
        params.n_components, size=params.n_samples, p=params.weights
    )
    return SimulatedEmissionMixtureDataset(
        labels=labels,
        observations=params.components.sample(labels, generator),
        weights=params.weights,
        components=params.components,
        seed=params.seed,
    )


_REQUIRED_FIELDS = frozenset(
    {"seed", "n_samples", "tolerance", "weights", "family", "dispersion", "mean"}
)


def _count_pair(raw: Any, path: Path, *, joint: bool) -> CountPairEmission:
    """Build the declared count-pair family, refusing a field the form cannot use."""
    for field in ("alpha", "beta"):
        if field not in raw:
            msg = f"{path}: a count-pair family needs {field!r}"
            raise ValueError(msg)
    if joint and "trials" in raw:
        msg = (
            f"{path}: the joint form's trial count is the drawn total; "
            f"'trials' belongs to 'count-pair-independent'"
        )
        raise ValueError(msg)
    if not joint and "trials" not in raw:
        msg = f"{path}: the independent form needs 'trials'"
        raise ValueError(msg)
    return CountPairEmission(
        np.asarray(raw["dispersion"], dtype=np.float64),
        np.asarray(raw["mean"], dtype=np.float64),
        np.asarray(raw["alpha"], dtype=np.float64),
        np.asarray(raw["beta"], dtype=np.float64),
        None if joint else np.asarray(raw["trials"], dtype=np.float64),
        joint=joint,
    )


def load_emission_mixture_params(path: Path) -> EmissionMixtureParams:
    """Load and validate a count-emission-mixture fixture yaml.

    Parameters
    ----------
    path : Path
        Path to the yaml file.

    Returns
    -------
    EmissionMixtureParams
        The parsed, validated truth. The weight and component checks are
        :class:`EmissionMixtureParams`'s and the family's own, so a file and
        an instance built in code are refused on the same terms.

    Raises
    ------
    ValueError
        If a required field is missing, or the declared family is not one of
        :data:`FAMILIES`.
    """
    raw = load_declared(path, _REQUIRED_FIELDS)
    family = str(raw["family"])
    if family not in FAMILIES:
        msg = f"{path}: family {family!r} is not one of {list(FAMILIES)}"
        raise ValueError(msg)

    return EmissionMixtureParams(
        weights=np.asarray(raw["weights"], dtype=np.float64),
        components=_count_pair(raw, path, joint=family == "count-pair-joint"),
        n_samples=int(raw["n_samples"]),
        seed=int(raw["seed"]),
        tolerance=float(raw["tolerance"]),
    )
