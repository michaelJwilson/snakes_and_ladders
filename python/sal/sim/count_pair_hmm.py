"""A joint count-pair HMM with rare states on their own levels (issues #1378, #1416).

A hidden Markov chain over ``K`` states, each emitting a count pair
(:class:`~sal.emissions.CountPairEmission`, joint form): a negative-binomial
total and beta-binomial successes over that total. Every field is declared:
the stationary law ``occupancy``, the per-state total ``mean`` and success
``rate``, one ``dispersion`` (the negative binomial's ``r``) and one
``concentration`` (``alpha + beta``) shared by the states.

**The chain** is reversible with stationary law ``occupancy``: state ``k``
leaves at ``c (1 - pi_k)`` and lands on ``j`` in proportion to ``pi_j``, ``c``
set so a state other than the first stays ``rare_dwell`` positions on
average. One segment of ``n_positions``, started from ``occupancy``.

**Draw order** is :func:`sal.sim.hmm.simulate_sequences`' from
``default_rng(seed)``, so the instance is the one ``sal.qa.count_pair_hmm_anneal``
built in place before the fixture declared it, bitwise.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np

from sal.emissions import CountPairEmission
from sal.sim.hmm import HmmParams, SimulatedHmmDataset, simulate_sequences

_REQUIRED_FIELDS = frozenset(
    {
        "seed",
        "n_positions",
        "occupancy",
        "mean",
        "rate",
        "dispersion",
        "concentration",
        "rare_dwell",
        "observations_digest",
    }
)


def observations_digest(simulated: SimulatedHmmDataset) -> str:
    """16 hex characters of the SHA-256 of the path, then the ``(n, 2)`` pairs.

    Returns
    -------
    str
    """
    states = np.asarray(simulated.states).reshape(-1)
    pairs = np.asarray(simulated.observations, dtype=np.float64).reshape(-1, 2)
    digest = hashlib.sha256(np.ascontiguousarray(states, dtype=np.int64))
    digest.update(np.ascontiguousarray(pairs).tobytes())
    return digest.hexdigest()[:16]


@dataclass(frozen=True)
class CountPairHmmParams:
    """The declared chain, drawn by :meth:`instance`.

    Parameters
    ----------
    seed : int
        The seed every draw is made from.
    n_positions : int
        Positions of the one segment.
    occupancy : np.ndarray
        ``(K,)`` stationary law and initial distribution.
    mean : np.ndarray
        ``(K,)`` the total's mean per state.
    rate : np.ndarray
        ``(K,)`` the success rate per state.
    dispersion : float
        The negative binomial's ``r``, every state.
    concentration : float
        The beta-binomial's ``alpha + beta``, every state.
    rare_dwell : float
        Mean dwell of a state other than the first, in positions.
    observations_digest : str
        :func:`observations_digest` of :meth:`instance`, as recorded.
    """

    seed: int
    n_positions: int
    occupancy: np.ndarray
    mean: np.ndarray
    rate: np.ndarray
    dispersion: float
    concentration: float
    rare_dwell: float
    observations_digest: str

    #: The fields :func:`sal.fixtures.load_params` checks are present.
    required_fields: ClassVar[frozenset[str]] = _REQUIRED_FIELDS

    @classmethod
    def from_declared(cls, declared: Mapping[str, Any], path: Path, /) -> Self:
        """Build the declared chain, refusing a value no chain can carry.

        Raises
        ------
        ValueError
            If the per-state vectors disagree in length, ``occupancy`` is not
            a distribution, a rate lies outside ``(0, 1)``, or there are fewer
            than two positions.
        """
        occupancy = np.asarray(declared["occupancy"], dtype=np.float64)
        mean = np.asarray(declared["mean"], dtype=np.float64)
        rate = np.asarray(declared["rate"], dtype=np.float64)
        k = occupancy.shape[0]
        if k < 2 or mean.shape != (k,) or rate.shape != (k,):
            msg = f"{path}: occupancy, mean and rate must be one entry per state"
            raise ValueError(msg)
        if occupancy.min() <= 0.0 or abs(occupancy.sum() - 1.0) > 1e-9:
            msg = f"{path}: occupancy must be positive and sum to 1, got {occupancy}"
            raise ValueError(msg)
        if not (rate.min() > 0.0 and rate.max() < 1.0):
            msg = f"{path}: rates must lie in (0, 1), got {rate}"
            raise ValueError(msg)
        n_positions = int(declared["n_positions"])
        if n_positions < 2:
            msg = f"{path}: n_positions must be >= 2, got {n_positions}"
            raise ValueError(msg)
        return cls(
            seed=int(declared["seed"]),
            n_positions=n_positions,
            occupancy=occupancy,
            mean=mean,
            rate=rate,
            dispersion=float(declared["dispersion"]),
            concentration=float(declared["concentration"]),
            rare_dwell=float(declared["rare_dwell"]),
            observations_digest=str(declared["observations_digest"]),
        )

    @property
    def n_states(self) -> int:
        """``K``."""
        return int(self.occupancy.shape[0])

    @property
    def transition(self) -> np.ndarray:
        """The reversible chain the module states, ``(K, K)``."""
        pi = self.occupancy
        leave = (1.0 - pi) / (self.rare_dwell * (1.0 - pi[1:]).mean())
        transition = leave[:, None] * pi[None, :] / (1.0 - pi[:, None])
        np.fill_diagonal(transition, 1.0 - leave)
        out: np.ndarray = transition
        return out

    @property
    def components(self) -> CountPairEmission:
        """The emission truth, joint form."""
        return CountPairEmission(
            np.full(self.n_states, self.dispersion),
            self.mean,
            self.concentration * self.rate,
            self.concentration * (1.0 - self.rate),
            None,
            joint=True,
        )

    def instance(self) -> SimulatedHmmDataset:
        """Draw the chain from :attr:`seed` by :func:`~sal.sim.hmm.simulate_sequences`.

        Returns
        -------
        SimulatedHmmDataset
        """
        return simulate_sequences(
            HmmParams(
                n_states=self.n_states,
                lengths=(self.n_positions,),
                initial=self.occupancy,
                transition=self.transition,
                emissions=self.components,
                seed=self.seed,
                tolerance=0.0,
            )
        )
