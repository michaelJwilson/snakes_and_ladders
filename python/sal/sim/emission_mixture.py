"""Ground truth and data generation for a mixture of count emissions.

:mod:`sal.sim.mixture` is this model with a Gaussian family.
What changes here is the observation: a component emits a
:class:`sal.emissions.CountPairEmission` pair --- a sequencing
depth and the allele count within it --- so the mixture is over count families
whose M step is itself an optimization rather than a formula, and whose
observation is not a scalar. Everything the mixture itself does is unchanged,
which is the claim: the component family is a parameter of the problem and not
part of it.

**Truth ships with the data**, on the footing
:mod:`sal.sim.hmm` established: the component that produced
each observation is retained, because a dataset without it cannot referee a
clustering.

Fitting lives in :mod:`sal.opt.emission_mixture`, which imports
the truth type from here and draws no data.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np

from sal.emissions import CountPairEmission, EmissionFamily
from sal.sim.mixture import MixtureParamsBase, draw_mixture

#: The component families a fixture may declare, and the constructor
#: arguments each reads. Both are :class:`CountPairEmission`, one per form:
#: the form is part of the file's name for the family rather than a separate
#: flag, because the two are different generative models and a file that
#: declared the family without the form would state half a model. A count
#: family with a scalar observation is added here when a fixture needs one;
#: the mixture itself never asks what its components are.
FAMILIES = ("count-pair-joint", "count-pair-independent")


_REQUIRED_FIELDS = frozenset(
    {"seed", "n_samples", "tolerance", "weights", "family", "dispersion", "mean"}
)


@dataclass(frozen=True)
class EmissionMixtureParams(MixtureParamsBase[EmissionFamily]):
    """Fully-specified truth for a count-emission mixture fixture.

    :class:`~sal.sim.mixture.MixtureParamsBase`'s fields,
    with the component family left open: the mixture never asks what its
    components are, which is this fixture's claim.

    Parameters
    ----------
    components : EmissionFamily
        The per-component emission, one state per component.
    tolerance : float
        Relative tolerance a validation test checks a recovered weight or
        component parameter against its planted value within.
    """

    #: The fields :func:`sal.fixtures.load_params` checks are present before
    #: calling :meth:`from_declared`.
    required_fields: ClassVar[frozenset[str]] = _REQUIRED_FIELDS

    @classmethod
    def from_declared(cls, declared: Mapping[str, Any], path: Path, /) -> Self:
        """Build the truth from a count-emission-mixture fixture's declared mapping.

        ``declared`` is the mapping
        :func:`sal.fixtures.load_params` read from ``path``
        with :attr:`required_fields` present; ``path`` names the file in
        every error.

        The weight and component checks are :class:`EmissionMixtureParams`'s
        and the family's own, so a file and an instance built in code are
        refused on the same terms.

        Raises
        ------
        ValueError
            If a required field is missing, or the declared family is not one of
            :data:`FAMILIES`.
        """
        family = str(declared["family"])
        if family not in FAMILIES:
            msg = f"{path}: family {family!r} is not one of {list(FAMILIES)}"
            raise ValueError(msg)

        return cls(
            weights=np.asarray(declared["weights"], dtype=np.float64),
            components=_count_pair(declared, path, joint=family == "count-pair-joint"),
            n_samples=int(declared["n_samples"]),
            seed=int(declared["seed"]),
            tolerance=float(declared["tolerance"]),
        )


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
    labels, observations = draw_mixture(params, rng)
    return SimulatedEmissionMixtureDataset(
        labels=labels,
        observations=observations,
        weights=params.weights,
        components=params.components,
        seed=params.seed,
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
