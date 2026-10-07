"""Exact results for the Potts model on the square lattice, as referees for a sampler at lattice scale.

A cluster move is checked by enumeration on a handful of sites (#1142,
#1158); these are the results that check it on ``64 x 64`` and larger
(#1276). Each is a closed form from outside this repository, written in the
convention :func:`sal.sim.potts.energies` declares,
``E(s) = -sum_(ij) J [s_i == s_j]`` with no field, so a chain's mean of
``energies / n_nodes`` is compared against the value returned here.

- :func:`onsager_energy`: the ``q = 2`` energy per site on the infinite
  lattice at every coupling (Onsager 1944).
- :func:`kaufman_energy`: the same on the ``L x L`` torus, exactly (Kaufman
  1949). Away from the transition it differs from Onsager's by a term that
  decays as ``exp(-L / xi)``; at it, by ``O(1 / L)``. A chain on the torus is
  refereed here, so no finite-size correction enters its tolerance.
- :func:`yang_magnetization`: the ``q = 2`` spontaneous magnetization on the
  infinite lattice (Onsager; Yang 1952).
- :func:`baxter_critical_energy`: the ``q``-state energy per site at
  :func:`sal.sim.potts.critical_coupling` on the infinite lattice, for
  ``q <= 4`` (Baxter 1973).

A ``q = 2`` Potts coupling ``J`` is the Ising coupling ``K = J / 2``, since
``[s_i == s_j] = (1 + sigma_i sigma_j) / 2``; the conversion is made here,
once, and every argument below is the Potts ``J``.
"""

from __future__ import annotations

import math

import numpy as np

from sal.sim.potts import critical_coupling

#: The relative gap at which the arithmetic-geometric mean stops.
_AGM_TOLERANCE = 1e-16


def _agm(first: float, second: float) -> float:
    """The arithmetic-geometric mean, which gives ``K(k) = pi / (2 agm(1, k'))``."""
    while abs(first - second) > _AGM_TOLERANCE * first:
        first, second = 0.5 * (first + second), math.sqrt(first * second)
    return first


def _ising_bond_correlation(ising: float) -> float:
    """``<sigma_i sigma_j>`` between nearest neighbours, infinite square lattice, at Ising ``K``.

    Onsager's ``(1/2) coth 2K [1 + (2/pi)(2 tanh^2 2K - 1) K(k)]`` with
    ``k = 2 sinh 2K / cosh^2 2K``. Written in ``s = sinh 2K``, the complement
    is ``k' = |1 - s^2| / (1 + s^2)`` and ``2 tanh^2 2K - 1 = (s^2 - 1) /
    (1 + s^2)``, so neither is a difference of near-equal numbers; at
    ``s = 1`` the product of the vanishing factor and the divergent integral
    is its limit, zero.
    """
    s2 = math.sinh(2.0 * ising) ** 2
    if s2 == 1.0:
        elliptic_term = 0.0
    else:
        complement = abs(1.0 - s2) / (1.0 + s2)
        elliptic_term = (s2 - 1.0) / ((1.0 + s2) * _agm(1.0, complement))
    return 0.5 / math.tanh(2.0 * ising) * (1.0 + elliptic_term)


def onsager_energy(coupling: float) -> float:
    """The ``q = 2`` Potts energy per site on the infinite square lattice.

    ``-2 J <[s_i == s_j]>``, two bonds per site, with the bond expectation
    from Onsager's nearest-neighbour correlation at ``K = J / 2``.

    Parameters
    ----------
    coupling : float
        ``J > 0``, already multiplied by ``beta``.

    Returns
    -------
    float

    Examples
    --------
    At the transition the like-bond fraction is ``(1 + 1 / sqrt 2) / 2``:

    >>> J = critical_coupling(2)
    >>> round(onsager_energy(J) / (-J), 12) == round(1 + 2**-0.5, 12)
    True
    """
    if not coupling > 0.0:
        msg = f"the coupling must be positive, got {coupling}"
        raise ValueError(msg)
    return -coupling * (1.0 + _ising_bond_correlation(0.5 * coupling))


def kaufman_energy(coupling: float, side: int) -> float:
    """The ``q = 2`` Potts energy per site on the periodic ``side x side`` lattice, exactly.

    Kaufman's four-term partition function on the torus (Kaufman 1949, in
    the form of Ferdinand and Fisher 1969, eq. 2.1--2.4), with
    ``n = m = side``:
    ``Z = (1/2) (2 sinh 2K)^(N/2) (Z_1 + Z_2 + Z_3 + Z_4)``, where
    ``Z_1, Z_2 = prod_r 2 cosh, 2 sinh (m gamma_(2r+1) / 2)`` and
    ``Z_3, Z_4`` the same over ``gamma_(2r)``, ``r = 0 .. n - 1``;
    ``cosh gamma_l = cosh 2K coth 2K - cos(pi l / n)`` for ``l >= 1`` and
    ``gamma_0 = 2K + ln tanh K``, which changes sign at the transition.
    The energy is ``d ln Z / dK``, differentiated term by term rather than
    by a difference, and each product is carried as a log magnitude and a
    sign: at ``side = 256`` a factor reaches ``exp(10^2)``.

    Parameters
    ----------
    coupling : float
        ``J > 0``, already multiplied by ``beta``.
    side : int
        ``L >= 3``; the lattice is the one
        ``lattice_graph((side, side), BoundaryCondition.PERIODIC, J)`` builds,
        ``2 L^2`` bonds. At ``L = 2`` that builder doubles every bond and the
        model is another one.

    Returns
    -------
    float
    """
    if not coupling > 0.0:
        msg = f"the coupling must be positive, got {coupling}"
        raise ValueError(msg)
    if side < 3:
        msg = f"the torus needs side >= 3 for single bonds, got {side}"
        raise ValueError(msg)
    ising = 0.5 * coupling
    sinh2, cosh2 = math.sinh(2.0 * ising), math.cosh(2.0 * ising)
    base = cosh2 * cosh2 / sinh2
    base_slope = 2.0 * cosh2 * (2.0 * sinh2 * sinh2 - cosh2 * cosh2) / (sinh2 * sinh2)

    index = np.arange(2 * side)
    cosh_gamma = base - np.cos(np.pi * index / side)
    # `arccosh` of `1 + small` loses the small; this form does not. The
    # `l = 0` entry is overwritten below, and clipped first: there the
    # excess is ``-(1 - cosh gamma_0)`` to rounding, and can fall below zero.
    excess = np.maximum(cosh_gamma - 1.0, 0.0)
    gamma = np.log1p(excess + np.sqrt(excess * (excess + 2.0)))
    slope = np.empty_like(gamma)
    slope[1:] = base_slope / np.sinh(gamma[1:])
    gamma[0] = 2.0 * ising + math.log(math.tanh(ising))
    slope[0] = 2.0 + 2.0 / sinh2
    half = 0.5 * side * gamma
    half_slope = 0.5 * side * slope

    log_terms, slope_terms = [], []
    for parity in (1, 0):
        arguments, rates = half[parity::2], half_slope[parity::2]
        magnitude = np.abs(arguments)
        decay = np.exp(-2.0 * magnitude)
        # 2 cosh a and its log derivative a' tanh a.
        log_cosh = magnitude + np.log1p(decay)
        log_terms.append((float(log_cosh.sum()), 1.0))
        slope_terms.append(float((rates * np.tanh(arguments)).sum()))
        # 2 sinh a: a sign, a log magnitude, and a' coth a. Only gamma_0 can
        # vanish, and only at the transition.
        sign = float(np.prod(np.sign(arguments)))
        if sign == 0.0:
            log_terms.append((-math.inf, 0.0))
            slope_terms.append(0.0)
            continue
        log_sinh = magnitude + np.log1p(-decay)
        log_terms.append((float(log_sinh.sum()), sign))
        slope_terms.append(float((rates / np.tanh(arguments)).sum()))

    peak = max(log for log, _ in log_terms)
    weights = np.array([sign * math.exp(log - peak) for log, sign in log_terms])
    log_slope = float(weights @ np.array(slope_terms)) / float(weights.sum())
    n_sites = side * side
    # d ln Z / dK = N coth 2K + the four products' share; one bond in two.
    correlation = (n_sites / math.tanh(2.0 * ising) + log_slope) / (2 * n_sites)
    return -coupling * (1.0 + correlation)


def yang_magnetization(coupling: float) -> float:
    """The ``q = 2`` spontaneous magnetization on the infinite square lattice.

    ``(1 - sinh^-4 2K)^(1/8)`` above the critical coupling and zero at or
    below it, at ``K = J / 2``; the magnetization is ``|2 n_0 / N - 1|``,
    the Ising ``|sum sigma| / N``.

    Parameters
    ----------
    coupling : float
        ``J > 0``, already multiplied by ``beta``.

    Returns
    -------
    float
    """
    if not coupling > 0.0:
        msg = f"the coupling must be positive, got {coupling}"
        raise ValueError(msg)
    inverse = math.sinh(coupling) ** -4
    return (1.0 - inverse) ** 0.125 if inverse < 1.0 else 0.0


def baxter_critical_energy(n_states: int) -> float:
    """The ``q``-state Potts energy per site at the transition, infinite square lattice.

    ``-J_c (1 + 1 / sqrt q)`` at ``J_c = ln(1 + sqrt q)`` (Baxter 1973;
    Baxter 1982, ch. 12), from self-duality: two bonds per site, each like
    with probability ``(1 + 1 / sqrt q) / 2``. At ``q = 2`` it is Onsager's
    value at :func:`~sal.sim.potts.critical_coupling`.

    Parameters
    ----------
    n_states : int
        ``q`` in ``2 .. 4``, where the transition is continuous. Above 4 the
        energy jumps at the transition and this value is the midpoint of the
        jump, which no chain converges to, so it is refused.

    Returns
    -------
    float

    Examples
    --------
    >>> round(baxter_critical_energy(4), 12) == round(-1.5 * math.log(3.0), 12)
    True
    """
    if not 2 <= n_states <= 4:
        msg = f"the transition is continuous for 2 <= q <= 4, got q = {n_states}"
        raise ValueError(msg)
    return -critical_coupling(n_states) * (1.0 + 1.0 / math.sqrt(n_states))
