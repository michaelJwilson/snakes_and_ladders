"""The count-pair HMM reference cell: generating levels, and the states a fit gets (issue #1390).

A hidden Markov chain over ``L`` *generating levels*, each emitting a count
pair (:class:`~sal.emissions.CountPairEmission`, independent form): a
negative-binomial total scaled by a per-position exposure, and beta-binomial
successes over a per-position trial count. Level ``l`` has total log mean
``log base_mean + delta_log_mean * ladder[l]``, success rate ``rate[l]``, and
its own dispersion and concentration. A fit is handed ``n_fit_states``
states, which need not be ``L``: the downstream shape is ``K = 7`` states
fitted against 8 to 16 generating levels.

**Levels are declared, never derived.** Their count, their shares, and each
channel's value per level are fixture fields, so the reading of "one level
holds ~98%, rare levels at 1--2%, level pairs equal in one channel only" is
the file's and can be changed there. Two levels sharing a ladder rung differ
in rate, and two sharing a rate differ in rung.

**The chain** is reversible with stationary law ``shares``: off the diagonal
``P[k, j] = c * shares[j]``, so leaving ``k`` costs ``c * (1 - shares[k])``,
and ``c`` is set so the most occupied level stays with probability ``stay``.
The positions are cut into segments of ``segment_length``, each started from
``shares``; at ``stay = 1 - 1e-7`` a segment is one level almost surely, so
the segment count is what puts the rare levels in the data.

**Draw order,** from ``default_rng(seed)``: the paths' uniforms, the
exposures, the trial counts, then the emissions. Variants
(:meth:`CountHmmReferenceParams.variant`) are one factor from ``stress``, as
:mod:`sal.sim.potts_cell` states for the Potts cell.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np

from sal.emissions import CountPairEmission

_REQUIRED_FIELDS = frozenset(
    {
        "seed",
        "n_positions",
        "segment_length",
        "n_fit_states",
        "shares",
        "ladder",
        "rate",
        "dispersion",
        "concentration",
        "base_mean",
        "delta_log_mean",
        "stay",
        "exposure_spread",
        "trials_mean",
        "observations_digest",
    }
)

#: The per-level vectors a fixture declares, one entry per generating level.
_PER_LEVEL = ("shares", "ladder", "rate", "dispersion", "concentration")


@dataclass(frozen=True)
class CountHmm:
    """One drawn cell: the paths, the pairs, the covariate and the truth.

    Parameters
    ----------
    states : np.ndarray
        ``(n,)`` generating level per position, segments end to end.
    observations : np.ndarray
        ``(n, 2)`` total, then successes.
    covariate : np.ndarray
        ``(n, 2)`` exposure, then trial count, as
        :class:`~sal.emissions.CountPairEmission` reads a covariate.
    lengths : tuple[int, ...]
        The segment lengths.
    initial, transition : np.ndarray
        The chain's truth over the generating levels.
    components : CountPairEmission
        The emission truth, one state per generating level.
    """

    states: np.ndarray
    observations: np.ndarray
    covariate: np.ndarray
    lengths: tuple[int, ...]
    initial: np.ndarray
    transition: np.ndarray
    components: CountPairEmission


def observations_digest(cell: CountHmm) -> str:
    """16 hex characters of the SHA-256 of the paths, the pairs, then the covariate.

    Returns
    -------
    str
    """
    digest = hashlib.sha256(np.ascontiguousarray(cell.states, dtype=np.int64))
    digest.update(np.ascontiguousarray(cell.observations, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(cell.covariate, dtype=np.float64).tobytes())
    return digest.hexdigest()[:16]


@dataclass(frozen=True)
class CountHmmReferenceParams:
    """The declared cell, drawn by :meth:`instance`.

    Parameters
    ----------
    seed : int
        The seed every draw is made from.
    n_positions : int
        Positions, over every segment.
    segment_length : int
        Positions per segment; divides ``n_positions``.
    n_fit_states : int
        ``K``, the states a fit of the cell is given.
    shares : np.ndarray
        ``(L,)`` stationary law and initial distribution over the levels.
    ladder : np.ndarray
        ``(L,)`` integer rung of each level's total log mean.
    rate : np.ndarray
        ``(L,)`` success rate of each level.
    dispersion : np.ndarray
        ``(L,)`` the negative binomial's ``r`` per level.
    concentration : np.ndarray
        ``(L,)`` the beta-binomial's ``alpha + beta`` per level.
    base_mean : float
        The total's mean at rung 0 and unit exposure.
    delta_log_mean : float
        The log-mean step between adjacent rungs.
    stay : float
        The most occupied level's self-transition probability.
    exposure_spread : float
        The log-normal scale of the per-position exposure, at log-mean 0.
    trials_mean : float
        The trial count is ``1 + Poisson(trials_mean - 1)`` per position.
    observations_digest : str
        :func:`observations_digest` of :meth:`instance`, as recorded.
    variants : Mapping[str, Mapping[str, Any]]
        One-factor variants by name, as the file declares them.
    """

    seed: int
    n_positions: int
    segment_length: int
    n_fit_states: int
    shares: np.ndarray
    ladder: np.ndarray
    rate: np.ndarray
    dispersion: np.ndarray
    concentration: np.ndarray
    base_mean: float
    delta_log_mean: float
    stay: float
    exposure_spread: float
    trials_mean: float
    observations_digest: str
    variants: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    declared: Mapping[str, Any] = field(default_factory=dict, compare=False)
    path: Path = field(default=Path(), compare=False)

    #: The fields :func:`sal.fixtures.load_params` checks are present.
    required_fields: ClassVar[frozenset[str]] = _REQUIRED_FIELDS

    @classmethod
    def from_declared(cls, declared: Mapping[str, Any], path: Path, /) -> Self:
        """Build the declared cell, refusing a value no chain can carry.

        Raises
        ------
        ValueError
            If the per-level vectors disagree in length, the shares are not a
            distribution, a rate or ``stay`` lies outside ``(0, 1)``, the
            segments do not tile the positions, or ``trials_mean < 1``.
        """
        vectors = {
            name: np.asarray(declared[name], dtype=np.float64) for name in _PER_LEVEL
        }
        levels = vectors["shares"].shape[0]
        if levels < 2 or any(v.shape != (levels,) for v in vectors.values()):
            shapes = {name: v.shape for name, v in vectors.items()}
            msg = (
                f"{path}: {list(_PER_LEVEL)} must be one entry per level, got {shapes}"
            )
            raise ValueError(msg)
        shares = vectors["shares"]
        if shares.min() <= 0.0 or abs(shares.sum() - 1.0) > 1e-9:
            msg = f"{path}: shares must be positive and sum to 1, got {shares}"
            raise ValueError(msg)
        stay = float(declared["stay"])
        rate = vectors["rate"]
        if not (rate.min() > 0.0 and rate.max() < 1.0 and 0.0 < stay < 1.0):
            msg = f"{path}: rates and stay must lie in (0, 1)"
            raise ValueError(msg)
        n_positions = int(declared["n_positions"])
        segment_length = int(declared["segment_length"])
        if segment_length < 2 or n_positions % segment_length:
            msg = (
                f"{path}: segments of {segment_length} do not tile "
                f"{n_positions} positions"
            )
            raise ValueError(msg)
        trials_mean = float(declared["trials_mean"])
        if trials_mean < 1.0:
            msg = f"{path}: trials_mean must be >= 1, got {trials_mean}"
            raise ValueError(msg)
        variants = declared.get("variants") or {}
        return cls(
            seed=int(declared["seed"]),
            n_positions=n_positions,
            segment_length=segment_length,
            n_fit_states=int(declared["n_fit_states"]),
            shares=shares,
            ladder=np.asarray(declared["ladder"], dtype=np.int64),
            rate=rate,
            dispersion=vectors["dispersion"],
            concentration=vectors["concentration"],
            base_mean=float(declared["base_mean"]),
            delta_log_mean=float(declared["delta_log_mean"]),
            stay=stay,
            exposure_spread=float(declared["exposure_spread"]),
            trials_mean=trials_mean,
            observations_digest=str(declared["observations_digest"]),
            variants={str(name): dict(entry) for name, entry in variants.items()},
            declared=dict(declared),
            path=path,
        )

    @property
    def n_levels(self) -> int:
        """``L``, the generating levels."""
        return int(self.shares.shape[0])

    def variant(self, name: str) -> CountHmmReferenceParams:
        """The declared variant ``name``: this cell with its one factor changed.

        Raises
        ------
        KeyError
            If no variant of that name is declared; the message lists those
            that are.
        """
        if name not in self.variants:
            msg = f"{self.path}: no variant {name!r}; declares {sorted(self.variants)}"
            raise KeyError(msg)
        changed = {
            **self.declared,
            "observations_digest": "",
            **self.variants[name],
            "variants": {},
        }
        return type(self).from_declared(changed, self.path)

    @property
    def transition(self) -> np.ndarray:
        """The reversible chain the module states, ``(L, L)``."""
        top = float(self.shares.max())
        rate = (1.0 - self.stay) / (1.0 - top)
        matrix = rate * np.broadcast_to(self.shares, (self.n_levels,) * 2).copy()
        np.fill_diagonal(matrix, 1.0 - rate * (1.0 - self.shares))
        return matrix

    @property
    def components(self) -> CountPairEmission:
        """The emission truth; ``trials_mean`` stands where no covariate is given."""
        return CountPairEmission(
            self.dispersion,
            self.base_mean * np.exp(self.delta_log_mean * self.ladder),
            self.concentration * self.rate,
            self.concentration * (1.0 - self.rate),
            np.full(self.n_levels, round(self.trials_mean)),
            joint=False,
        )

    def instance(self) -> CountHmm:
        """Draw the cell from :attr:`seed`, in the order the module states.

        Returns
        -------
        CountHmm
        """
        rng = np.random.default_rng(self.seed)
        n, last = self.n_positions, self.n_levels - 1
        transition = self.transition
        uniforms = rng.random(n)
        initial = np.cumsum(self.shares)
        cumulative = np.cumsum(transition, axis=1)
        states = np.empty(n, dtype=np.int64)
        for t in range(n):
            row = initial if t % self.segment_length == 0 else cumulative[states[t - 1]]
            states[t] = min(int(np.searchsorted(row, uniforms[t])), last)
        exposure = rng.lognormal(0.0, self.exposure_spread, size=n)
        trials = 1.0 + rng.poisson(self.trials_mean - 1.0, size=n)
        covariate = np.stack([exposure, trials], axis=1)
        family = self.components
        observations = np.asarray(
            family.sample(states, rng, covariate), dtype=np.float64
        )
        lengths = (self.segment_length,) * (n // self.segment_length)
        return CountHmm(
            states, observations, covariate, lengths, self.shares, transition, family
        )
