"""Randomness inside a move, without giving up a deterministic ``step``.

Issue #706. ``Environment.step`` is deterministic by contract and
:mod:`snakes_and_ladders.learn.exact`'s enumeration depends on it: the
trajectory set is finite only because one action from one state has one
successor. That is why the k-armed bandit and slippery Frozen Lake are not of
this protocol (#597), and it is the obstacle a Potts environment whose moves
are Monte Carlo has to clear first.

**A move's randomness is keyed on the state and the action rather than drawn
from a stream.** ``sim/count_pairs.py`` settles the same question for a
simulator: vertex ``v``'s counts come from ``default_rng([seed, v])``, so the
draw depends on the key and not on the order or the thread that reaches it.
Applied here, a heat-bath sweep or a cluster growth from state ``s`` under
action ``a`` always consumes the same uniforms, so

* ``step`` is a pure function and the enumeration stays valid;
* a rollout is reproducible from the environment's seed alone, with no stream
  threaded through the caller;
* the same episode replays identically on any thread, which
  ``snakes_and_ladders.parallel`` needs and a shared generator would deny.

**The key is a digest, not ``hash``.** Python's ``hash`` is salted per process
for ``bytes`` and ``str``, so a key built from it would make a "reproducible"
episode reproducible only within one interpreter. ``blake2b`` over the
labelling's own bytes and a canonical action encoding is stable across
processes, platforms and orderings, which is the property being claimed.

:class:`KeyedMove` is how a move that needs more than a uniform --- a cluster
growth, a bond pass --- reaches an environment in this package without
`learn/` importing the module that implements it.

What this does **not** buy: an expectation. A keyed move is one successor, so
the environment is a deterministic MDP over a stochastic move's *realization*
at that key. Widening ``step`` to take a generator, and replacing
``exact_optimal_value``'s recursion with an expectation over successors, is the
other route and is #597's open question; it changes every implementer in the
tree, which is why it is not taken here.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

import numpy as np

#: Bytes of digest kept from ``blake2b``. Eight gives a 64-bit key, which is
#: what ``np.random.SeedSequence`` mixes without truncation.
DIGEST_BYTES = 8


def state_key(key: int, state: np.ndarray, action_code: int) -> int:
    """A 64-bit key for one ``(state, action)`` pair under one environment seed.

    Parameters
    ----------
    key : int
        The environment's own key material, drawn once from the caller's
        generator at construction --- `sim/CLAUDE.md`'s boundary rule, which is
        why no public signature here takes a seed. Two environments with
        different key material give different realizations of the same move at
        the same state.
    state : np.ndarray
        The labelling, as integers. Its bytes are digested, so the key depends
        on the configuration and not on how it was reached.
    action_code : int
        A canonical encoding of the action, from
        :meth:`~snakes_and_ladders.learn.potts_nd.PottsNDEnvironment.action_code`
        or its equivalent. Two distinct actions must not share a code, or two
        distinct moves would share a stream.

    Returns
    -------
    int
        In ``[0, 2**64)``.
    """
    digest = hashlib.blake2b(
        np.ascontiguousarray(state, dtype=np.int64).tobytes(),
        digest_size=DIGEST_BYTES,
        person=b"sal-706",
    )
    digest.update(int(key).to_bytes(8, "little", signed=False))
    digest.update(int(action_code).to_bytes(8, "little", signed=True))
    return int.from_bytes(digest.digest(), "little")


def keyed_generator(
    key: int, state: np.ndarray, action_code: int
) -> np.random.Generator:
    """The generator a move is entitled to at this state and action.

    A fresh generator per call rather than a cached one: the point is that the
    stream is a function of the key, so holding it would only hide that.
    """
    return np.random.default_rng(state_key(key, state, action_code))


@runtime_checkable
class KeyedMove(Protocol):
    """A Monte Carlo move a deterministic ``step`` can call.

    The implementations live in ``snakes_and_ladders.search``, which may import
    both halves; this package holds the protocol alone, which is what keeps
    `learn/CLAUDE.md`'s import rule green while a cluster move still reaches
    :class:`~snakes_and_ladders.learn.potts_nd.PottsNDEnvironment`.

    Under the seam rule: two consuming modules rather than three. It is kept
    because the alternative is not a third consumer but an import
    `tests/regression/learn/test_learn_environment.py` refuses --- the seam
    exists to carry a dependency in the direction the package allows, so its
    consumer count is bounded by the two sides it joins.

    A move holds the problem it moves on --- the graph and the field --- so
    ``propose`` is a function of the labelling, the action's parameters and the
    generator alone, and the layout it reads is built once rather than per
    call. The environment checks ``n_nodes`` and ``n_states`` against its own
    at construction: two callers scoring one problem is the property #706
    exists to establish, and a move built on a different lattice would break it
    silently.
    """

    @property
    def n_nodes(self) -> int:
        """Sites the move expects, checked against the environment's."""

    @property
    def n_states(self) -> int:
        """Labels the move expects, checked against the environment's."""

    @property
    def parametric(self) -> bool:
        """Whether an action of this move names a site and a target label.

        A Wolff step names its cluster root and the colour to recolour to; a
        Swendsen-Wang pass names neither, since it partitions the whole lattice
        and recolours every cluster. The environment reads this to decide how
        many candidates the move contributes.
        """

    def propose(
        self,
        state: np.ndarray,
        *,
        temperature: float,
        site: int,
        label: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, int]:
        """The successor labelling, and what the move cost in site visits.

        Parameters
        ----------
        state : np.ndarray
            The labelling, as ``int64``. Not mutated.
        temperature : float
            ``0.0`` is admitted and is not a small number: it is where the move
            becomes deterministic in its bonds, which is the limit an oracle
            can check exactly.
        site, label : int
            The action's parameters, or ``-1`` each where
            :attr:`parametric` is ``False``.
        rng : np.random.Generator
            Keyed on ``(state, action)`` by :func:`keyed_generator`, which is
            what makes the successor a function of the pair rather than of the
            order the move was reached in.

        Returns
        -------
        tuple[np.ndarray, int]
            The successor, and its charge in the site visits
            ``search.ground_state`` charges the same move --- realized rather
            than nominal, since a cluster move's cost is its cluster's size.
        """
