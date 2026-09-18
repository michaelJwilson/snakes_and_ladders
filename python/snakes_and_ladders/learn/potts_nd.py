"""A Potts lattice in N dimensions with a per-site field, searched by Monte Carlo moves.

Issue #706. Two things separate this from
:class:`~snakes_and_ladders.learn.potts.PottsEnvironment`, and both are the
point:

**The field is per site.** That environment's ``h`` is ``(n_states,)`` --- one
global tilt per label --- while ``search.ground_state`` and ``search.maxflow``
score a ``(n_nodes, n_states)`` field. So no learned policy has ever been
measured on the instance the classical ground-state methods are ranked on. A
uniform field also makes the ferromagnetic ground state trivial, for the reason
``maxflow.site_field`` states: every coupling favours agreement and every site
prefers the same label, so the answer is ``argmax(h)`` everywhere.

**The action carries a temperature.** A sweep action at ``T = 0`` is one
iterated-conditional-modes sweep --- each site taking its conditional mode in
index order --- and the same action at ``T > 0`` is a heat-bath pass. So the
schedule is *learned* rather than declared, where
``search.potts_mcmc.anneal_potts`` takes it as two constants, and **the
classical method is a point in the policy class rather than a separate
program**: a policy that always takes the sweep at rung zero reproduces
``search.alpha_expansion.iterated_conditional_modes``'s labelling exactly,
which a test pins.

The *fine* action at ``T = 0`` is a different classical baseline --- steepest
ascent, the best single flip anywhere --- and
:meth:`PottsNDEnvironment.steepest_action` returns it. The two are not the
same algorithm and the distinction is load-bearing: ICM is cheap per unit of
progress because one sweep updates every site, while steepest ascent pays a
whole-lattice scan to move one label.

**Two grains, and the reason both are here.** A fine action names a site and a
label, and costs about five site visits. A coarse action buys a whole sweep,
2,784 visits on the 24x24 lattice. At #704's matched budget of 200 sweeps a
fine-only episode is **1e5 decisions**, which is not a budget problem but a
learnability one --- nothing in ``learn`` trains over that horizon --- and a
coarse-only episode is 200. Carrying both leaves *which grain to spend on* to
the policy, and makes it a reported quantity rather than a design decision.

The move sets this module admits are declared per construction and never
joined here: #706 runs the single-site arm, the Wolff arm and the
Swendsen-Wang arm as controls for a mixed arm that comes later, because a
mixed arm alone cannot say whether the schedule or the move selection bought
the result.

Randomness is keyed, not streamed (:mod:`snakes_and_ladders.learn.keyed`), so
``step`` stays the deterministic function the protocol and
``learn.exact``'s enumeration require.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import NamedTuple

import numpy as np
import torch

from snakes_and_ladders.learn.environment import Environment
from snakes_and_ladders.learn.keyed import keyed_generator

#: One label per site.
Configuration = tuple[int, ...]

#: Candidates scored per decision away from a rung where the full neighbourhood
#: is affordable. Measured, not chosen: one candidate scores in 0.30 us, so 128
#: costs 39 us against a sweep's 4.7 ms of real work on the 24x24 lattice,
#: while the 13,836-candidate mixed set costs 4.2 ms --- 0.86 of a sweep
#: (issue #706).
DEFAULT_CANDIDATES = 128

#: Rungs of the geometric temperature ladder, and its ends. ``0.0`` is the
#: first rung and is exact rather than a small number: it is what makes a
#: single-site action an ICM update, and the classical control a point in this
#: action space.
DEFAULT_LADDER = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


class MoveKind(StrEnum):
    """Which move an action applies.

    Named here rather than imported from
    ``snakes_and_ladders.search.potts_mcmc.PottsMove`` because ``learn``
    imports nothing from ``search`` (`learn/CLAUDE.md`), and spelled the same
    so a reader is not asked to hold two vocabularies. The cluster kinds
    arrive with their arms.
    """

    FLIP = "flip"
    SWEEP = "sweep"


class PottsAction(NamedTuple):
    """One action: a move, where it applies, and how hot.

    Parameters
    ----------
    kind : MoveKind
        ``FLIP`` names a site and a label; ``SWEEP`` names neither and visits
        every site.
    site, label : int
        The flip's site and target label, or ``-1`` for a sweep.
    rung : int
        Index into the environment's temperature ladder.
    """

    kind: MoveKind
    site: int
    label: int
    rung: int


class PottsNDEnvironment(Environment[Configuration, PottsAction]):
    """Single-site search over an N-D Potts lattice at a known per-site field.

    Parameters
    ----------
    edges : Sequence[tuple[int, int]]
        Undirected edges. Passed rather than a graph object, as
        :meth:`~snakes_and_ladders.learn.potts.PottsEnvironment.on_graph`
        takes them, because ``learn`` may not import ``sim``.
    n_nodes : int
        Sites, at least two.
    coupling : float
        The true ``J``, uniform over edges and non-negative: the cluster arms
        to come need a non-negative coupling for their bond probability, and a
        sign that changes per arm would make the arms incomparable.
    field : np.ndarray
        The true ``h``, shape ``(n_nodes, n_states)``.
    ladder : Sequence[float]
        Temperatures, ascending, the first of which may be ``0.0``.
    kinds : Sequence[MoveKind]
        Which grains this arm admits.
    candidates : int
        Scored candidates per decision, or the whole neighbourhood when it is
        smaller.
    generator : np.random.Generator
        Drawn from once, at construction, for the key material every move's
        realization is derived from
        (:mod:`snakes_and_ladders.learn.keyed`). A generator rather than a
        seed, as `sim/CLAUDE.md` requires of every public signature, and
        required rather than defaulted: a default stream would make an
        environment whose episodes do not replay, which is the one thing this
        module is built to avoid.

    Raises
    ------
    ValueError
        If the field is not ``(n_nodes, n_states)`` with at least two of each,
        the coupling is negative, the ladder is empty, unsorted or carries a
        negative temperature, or ``candidates`` is under one.
    """

    def __init__(
        self,
        *,
        edges: Sequence[tuple[int, int]],
        n_nodes: int,
        coupling: float,
        field: np.ndarray,
        ladder: Sequence[float] = DEFAULT_LADDER,
        kinds: Sequence[MoveKind] = (MoveKind.FLIP, MoveKind.SWEEP),
        candidates: int = DEFAULT_CANDIDATES,
        generator: np.random.Generator,
    ) -> None:
        field = np.asarray(field, dtype=np.float64)
        if n_nodes < 2:
            msg = f"a lattice has at least two sites, got {n_nodes}"
            raise ValueError(msg)
        if field.shape[0] != n_nodes or field.ndim != 2 or field.shape[1] < 2:
            msg = (
                f"field must be ({n_nodes}, n_states >= 2), got shape {field.shape}: "
                "a per-site field is what makes the ground state non-trivial"
            )
            raise ValueError(msg)
        if coupling < 0.0:
            msg = f"coupling must be non-negative, got {coupling}"
            raise ValueError(msg)
        ladder = tuple(float(value) for value in ladder)
        if not ladder or any(value < 0.0 for value in ladder):
            msg = f"the ladder is non-empty and non-negative, got {ladder}"
            raise ValueError(msg)
        if list(ladder) != sorted(ladder):
            msg = f"the ladder is ascending, got {ladder}"
            raise ValueError(msg)
        if candidates < 1:
            msg = f"candidates must be >= 1, got {candidates}"
            raise ValueError(msg)
        if not kinds:
            msg = "an arm admits at least one move kind"
            raise ValueError(msg)

        self._coupling = float(coupling)
        self._field = field
        self._n_nodes = int(n_nodes)
        self._n_states = int(field.shape[1])
        self._ladder = ladder
        self._kinds = tuple(dict.fromkeys(kinds))
        self._candidates = int(candidates)
        self._key = int(generator.integers(1 << 62))
        self._set_neighbours(edges)

    def _set_neighbours(self, edges: Sequence[tuple[int, int]]) -> None:
        """Hold the adjacency as a padded table, plus the sweep's price.

        The padded gather is
        :class:`~snakes_and_ladders.learn.potts.PottsEnvironment`'s, for the
        reason issue #341 gives: one NumPy pass over every action rather than a
        Python-level delta per action.
        """
        rows: list[list[int]] = [[] for _ in range(self._n_nodes)]
        for first, second in edges:
            if not (0 <= first < self._n_nodes and 0 <= second < self._n_nodes):
                msg = f"edge ({first}, {second}) lies outside {self._n_nodes} sites"
                raise ValueError(msg)
            rows[first].append(int(second))
            rows[second].append(int(first))
        width = max((len(row) for row in rows), default=0)
        table = np.zeros((self._n_nodes, width), dtype=np.int64)
        mask = np.zeros((self._n_nodes, width), dtype=bool)
        for site, row in enumerate(rows):
            table[site, : len(row)] = row
            mask[site, : len(row)] = True
        self._neighbour_table = table
        self._neighbour_mask = mask
        self._degree = mask.sum(axis=1)
        self._n_edges = len(edges)
        # The edge order as given, so `score` can sum the coupling term the way
        # `sim.potts.energies` sums it -- one gather and a dot product, in that
        # order -- and land on its negation bitwise rather than to a tolerance.
        self._edge_ends = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        self._edge_coupling = np.full(len(edges), self._coupling, dtype=np.float64)
        # `search/ground_state.py`'s `visits_per_sweep`: a write per site and a
        # read per edge end. Charged identically so a comparison against that
        # module's methods is against its own unit.
        self._sweep_visits = self._n_nodes + 2 * self._n_edges

    @property
    def n_states(self) -> int:
        """``q``."""
        return self._n_states

    @property
    def n_nodes(self) -> int:
        """``V``."""
        return self._n_nodes

    @property
    def coupling(self) -> float:
        """The true ``J``."""
        return self._coupling

    @property
    def field(self) -> np.ndarray:
        """The true ``h``, as a copy, so a trained-against parameter cannot move."""
        return self._field.copy()

    @property
    def ladder(self) -> tuple[float, ...]:
        """The temperatures an action may name."""
        return self._ladder

    @property
    def sweep_visits(self) -> int:
        """Site visits one sweep costs, as ``search.ground_state`` charges it."""
        return self._sweep_visits

    def score(self, state: Configuration) -> float:
        """``J * agreements + sum_i h_i[s_i]``, which the agent maximizes.

        The negation of ``sim.potts.energies``, and named ``score`` rather than
        ``energy`` because it is the quantity that goes *up*:
        :class:`~snakes_and_ladders.learn.potts.PottsEnvironment` calls the
        same thing ``energy`` while returning the log weight, which is one
        vocabulary too many.

        The field term is summed first and the coupling term added as one
        gather and a dot product over the edges *in the order they were
        given* --- `sim.potts.energies`'s own arithmetic, so this is its
        negation **bitwise** and not to a tolerance. Reassociating it, by
        halving a doubled count off the adjacency table for instance, moves
        the last bit and a test pins that it does not.
        """
        labels = np.asarray(state, dtype=np.int64)
        total = self._field[np.arange(self._n_nodes), labels].sum()
        if self._n_edges:
            ends = self._edge_ends
            agree = labels[ends[:, 0]] == labels[ends[:, 1]]
            total = total + agree.astype(float) @ self._edge_coupling
        return float(total)

    def reset(self, rng: np.random.Generator) -> Configuration:
        """A uniform random labelling."""
        return tuple(
            int(value) for value in rng.integers(self._n_states, size=self._n_nodes)
        )

    def action_code(self, action: PottsAction) -> int:
        """A distinct integer per action, for :func:`~snakes_and_ladders.learn.keyed.state_key`.

        Distinctness is what the keying rests on: two actions sharing a code
        would share a stream and their realizations would be correlated in a
        way nothing reports.
        """
        kind = self._kinds.index(action.kind)
        return (
            ((action.site + 1) * self._n_states + (action.label + 1))
            * len(self._ladder)
            + action.rung
        ) * len(self._kinds) + kind

    def actions(self, state: Configuration) -> Sequence[PottsAction]:
        """The scored candidates at ``state``: every rung, a subset of the sites.

        Every sweep action is always offered --- there are only ``len(ladder)``
        of them --- and the flips are subset to ``candidates`` by the state's
        own key, so the same state offers the same actions on every visit and
        ``learn.exact`` enumerates what the policy scored. A trajectory-keyed
        subset would make the action set depend on how the state was reached,
        which the enumeration cannot follow.
        """
        offered: list[PottsAction] = []
        if MoveKind.SWEEP in self._kinds:
            offered.extend(
                PottsAction(MoveKind.SWEEP, -1, -1, rung)
                for rung in range(len(self._ladder))
            )
        if MoveKind.FLIP in self._kinds:
            offered.extend(self._flip_candidates(state))
        return offered

    def _flip_candidates(self, state: Configuration) -> list[PottsAction]:
        """Flips to score: all of them, or ``candidates`` drawn by the state's key."""
        labels = np.asarray(state, dtype=np.int64)
        whole = self._n_nodes * (self._n_states - 1) * len(self._ladder)
        sites = np.repeat(np.arange(self._n_nodes), self._n_states - 1)
        offsets = np.tile(np.arange(1, self._n_states), self._n_nodes)
        targets = (labels[sites] + offsets) % self._n_states
        pairs = np.stack([sites, targets], axis=1)
        if whole <= self._candidates:
            return [
                PottsAction(MoveKind.FLIP, int(site), int(label), rung)
                for site, label in pairs
                for rung in range(len(self._ladder))
            ]
        rng = keyed_generator(self._key, labels, -1)
        per_rung = max(1, self._candidates // len(self._ladder))
        chosen = rng.choice(pairs.shape[0], size=per_rung, replace=False)
        return [
            PottsAction(MoveKind.FLIP, int(pairs[index, 0]), int(pairs[index, 1]), rung)
            for index in chosen
            for rung in range(len(self._ladder))
        ]

    def visits(self, state: Configuration, action: PottsAction) -> int:
        """What ``action`` costs in site visits.

        A flip writes its own label and reads its neighbours'; a sweep is
        ``search.ground_state``'s ``n_nodes + 2 n_edges``. The two differ by
        about five hundred at 576 sites, which is why #706's budget carries
        the scored-candidate count beside this one: no single number makes them
        commensurable *and* prices the policy's own scoring.
        """
        del state
        if action.kind is MoveKind.SWEEP:
            return self._sweep_visits
        return 1 + int(self._degree[action.site])

    def step(
        self, state: Configuration, action: PottsAction
    ) -> tuple[Configuration, float]:
        """Apply ``action`` and return the successor and the change in score.

        The reward is ``score(s') - score(s)``, so an undiscounted return
        telescopes to the total improvement, which is the convention
        :class:`~snakes_and_ladders.learn.environment.Episode` states. It is
        computed from the affected bonds rather than by rescoring, and a test
        pins the two against each other.

        At rung ``0`` --- temperature zero --- a flip is accepted only if it
        does not lower the score, which is one ICM update. Above it the flip is
        a Metropolis accept at that temperature, and a sweep is a heat-bath
        pass over every site. Both draw from the keyed generator, so the
        successor is a function of ``(state, action)`` alone.
        """
        temperature = self._ladder[action.rung]
        if action.kind is MoveKind.SWEEP:
            return self._sweep(state, action, temperature)
        return self._flip(state, action, temperature)

    def _flip(
        self, state: Configuration, action: PottsAction, temperature: float
    ) -> tuple[Configuration, float]:
        """One site's Metropolis update, deterministic at ``T = 0``."""
        gain = self._flip_gain(np.asarray(state, dtype=np.int64), action)
        if gain >= 0.0:
            accepted = True
        elif temperature == 0.0:
            accepted = False
        else:
            rng = keyed_generator(
                self._key, np.asarray(state), self.action_code(action)
            )
            accepted = bool(rng.random() < np.exp(gain / temperature))
        if not accepted:
            return state, 0.0
        successor = list(state)
        successor[action.site] = action.label
        return tuple(successor), float(gain)

    def _sweep(
        self, state: Configuration, action: PottsAction, temperature: float
    ) -> tuple[Configuration, float]:
        """A heat-bath pass over every site, or a greedy pass at ``T = 0``.

        Sites are visited in index order and each update sees the labels the
        earlier ones left, which is what makes a sweep a sweep rather than a
        batch of independent flips --- and what
        ``search.potts_mcmc._single_site_sweep`` does.
        """
        labels = np.asarray(state, dtype=np.int64).copy()
        before = self.score(tuple(int(value) for value in labels))
        rng = keyed_generator(self._key, np.asarray(state), self.action_code(action))
        for site in range(self._n_nodes):
            neighbours = self._neighbour_table[site][self._neighbour_mask[site]]
            agreement = (
                labels[neighbours][None, :] == np.arange(self._n_states)[:, None]
            ).sum(axis=1)
            local = self._coupling * agreement + self._field[site]
            if temperature == 0.0:
                labels[site] = int(np.argmax(local))
                continue
            weights = np.exp((local - local.max()) / temperature)
            labels[site] = int(rng.choice(self._n_states, p=weights / weights.sum()))
        successor = tuple(int(value) for value in labels)
        return successor, self.score(successor) - before

    def _flip_gain(self, labels: np.ndarray, action: PottsAction) -> float:
        """``score`` after the flip less before it, from the affected terms alone."""
        neighbours = self._neighbour_table[action.site][
            self._neighbour_mask[action.site]
        ]
        agreement = float(
            (labels[neighbours] == action.label).sum()
            - (labels[neighbours] == labels[action.site]).sum()
        )
        field = float(
            self._field[action.site, action.label]
            - self._field[action.site, labels[action.site]]
        )
        return self._coupling * agreement + field

    def features(
        self, state: Configuration, actions: Sequence[PottsAction]
    ) -> torch.Tensor:
        """``(len(actions), 3)``: the move's gain, its temperature, its log price.

        None is constant across a state's actions where an arm admits both
        grains: the gain varies by site, the temperature by rung, and the price
        by kind. A column that were constant would sit in the direction the
        softmax cancels, which is the gauge
        :class:`~snakes_and_ladders.learn.environment.Environment` names.

        A sweep's *gain* is not knowable before it runs, so its column carries
        the sum of the positive single-site gains --- what a greedy pass could
        collect at most, in one vectorized read. It is a feature and not a
        promise: the policy learns what it is worth.
        """
        if not actions:
            return torch.empty((0, 3), dtype=torch.float64)
        labels = np.asarray(state, dtype=np.int64)
        bound = self._greedy_bound(labels)
        gains = np.fromiter(
            (
                bound
                if action.kind is MoveKind.SWEEP
                else self._flip_gain(labels, action)
                for action in actions
            ),
            dtype=np.float64,
            count=len(actions),
        )
        temperatures = np.fromiter(
            (self._ladder[action.rung] for action in actions),
            dtype=np.float64,
            count=len(actions),
        )
        prices = np.fromiter(
            (self.visits(state, action) for action in actions),
            dtype=np.float64,
            count=len(actions),
        )
        return torch.from_numpy(np.stack([gains, temperatures, np.log(prices)], axis=1))

    def _greedy_bound(self, labels: np.ndarray) -> float:
        """The sum of every site's best single-flip gain, taken independently.

        An upper bound on one greedy sweep's collection, since the flips
        interact: two neighbours both preferring to agree cannot both be
        credited. Used as a sweep action's feature for that reason --- it is
        the cheapest quantity that moves with how much a sweep has left to do.
        """
        neighbour_labels = labels[self._neighbour_table]
        agreement = (
            (neighbour_labels[:, :, None] == np.arange(self._n_states)[None, None, :])
            & self._neighbour_mask[:, :, None]
        ).sum(axis=1)
        local = self._coupling * agreement + self._field
        current = local[np.arange(self._n_nodes), labels]
        return float(np.maximum(local.max(axis=1) - current, 0.0).sum())

    def n_features(self) -> int:
        """Three: the gain, the temperature and the log price."""
        return 3

    def is_terminal(self, state: Configuration) -> bool:
        """Never: a sampler has no local-optimum stopping rule.

        :class:`~snakes_and_ladders.learn.potts.PottsEnvironment` ends an
        episode where no flip improves, which is the greedy searcher's own
        stopping rule and is what makes the two comparable there. Here an
        action at ``T > 0`` may lower the score by design, so "no move
        improves" is not a state this environment stops at. The episode is
        bounded by the caller's budget instead, and
        :class:`~snakes_and_ladders.learn.environment.Episode` reports the
        truncation.
        """
        del state
        return False

    def icm_action(self) -> PottsAction:
        """The action one iterated-conditional-modes sweep is: a sweep at ``T = 0``.

        A policy that always takes this *is* ICM, so the classical baseline is
        a point in this action space rather than a separate program to trust.
        That is what makes #705's **match** outcome provable, and it is the
        property the Potts chain had analytically as the weight vector
        proportional to ``(J, 1)``.
        """
        return PottsAction(MoveKind.SWEEP, -1, -1, 0)

    def steepest_action(self, state: Configuration) -> PottsAction:
        """The best single flip anywhere at ``T = 0``: steepest ascent, not ICM.

        A different classical baseline, and the difference is not cosmetic:
        ICM's sweep updates every site for one lattice of work, while this
        scans the lattice to move one label. Ties break towards the lower site
        and label, as a deterministic baseline must.
        """
        labels = np.asarray(state, dtype=np.int64)
        neighbour_labels = labels[self._neighbour_table]
        agreement = (
            (neighbour_labels[:, :, None] == np.arange(self._n_states)[None, None, :])
            & self._neighbour_mask[:, :, None]
        ).sum(axis=1)
        local = self._coupling * agreement + self._field
        gains = local - local[np.arange(self._n_nodes), labels][:, None]
        gains[np.arange(self._n_nodes), labels] = -np.inf
        site, label = np.unravel_index(int(np.argmax(gains)), gains.shape)
        return PottsAction(MoveKind.FLIP, int(site), int(label), 0)
