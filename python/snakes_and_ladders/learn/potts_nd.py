"""A Potts lattice in N dimensions with a per-site field, searched by Monte Carlo moves.

Issue #706. Two things separate this from
:class:`~snakes_and_ladders.learn.potts.PottsEnvironment`, and both are the
point:

**The field is per site.** That environment's ``h`` is ``(n_states,)`` --- one
global tilt per label --- while ``search.ground_state`` and ``search.maxflow``
score a ``(n_nodes, n_states)`` field. So no learned policy has ever been
measured on the instance the classical ground-state methods are ranked on. A
uniform field also makes the ferromagnetic ground state trivial, for the reason
``search.maxflow.ising_ground_state`` states: every coupling favours agreement and every site
prefers the same label, so the answer is ``argmax(h)`` everywhere.

**The action carries a temperature.** A sweep action at ``T = 0`` is one
iterated-conditional-modes sweep --- each site taking its conditional mode in
index order --- and the same action at ``T > 0`` is a heat-bath pass. So the
schedule is *learned* rather than declared, where
``sample.potts_mcmc.anneal_potts`` takes it as two constants, and **the
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

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from snakes_and_ladders.learn.environment import Environment
from snakes_and_ladders.learn.keyed import KeyedMove, keyed_generator
from snakes_and_ladders.sample.accept import accept_at_temperature
from snakes_and_ladders.sample.potts_mcmc import MoveKind

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


#: Kinds whose move is supplied through
#: :class:`~snakes_and_ladders.learn.keyed.KeyedMove` rather than implemented
#: here, since a cluster growth needs the graph this package will not import.
CLUSTER_KINDS = (MoveKind.WOLFF, MoveKind.SWENDSEN_WANG, MoveKind.NIEDERMAYER)

#: Kinds whose action names a site and a target label. The rest name a
#: temperature and nothing else, and visit every site.
PARAMETRIC_KINDS = (MoveKind.FLIP, MoveKind.WOLFF, MoveKind.NIEDERMAYER)

#: Kinds whose charge is the cluster they build and so is known only by
#: building it. :meth:`PottsNDEnvironment.visits` runs the move for these;
#: for the rest the charge is read in constant time.
GROWN_KINDS = (MoveKind.WOLFF, MoveKind.NIEDERMAYER)


class FeatureColumn(StrEnum):
    """What a feature column reads, and the arms it can vary in.

    The feature map's *width* is a property of the arm, not of the package.
    :class:`~snakes_and_ladders.learn.environment.Environment`'s gauge rule
    forbids a column constant across a state's actions --- a softmax over
    scored actions cancels it, so it carries no information and only adds a
    direction in which the weights are unidentifiable. A Swendsen-Wang arm's
    actions differ in their rung alone, so a gain column would be exactly that
    constant, while the single-site arm's gain is its most informative feature.
    Rather than pick one width and let an arm carry a dead column,
    :meth:`PottsNDEnvironment._surviving_columns` keeps the columns that can
    vary in the arm it is given, and the constructor refuses an arm where none
    can.
    """

    GAIN = "gain"
    TEMPERATURE = "temperature"
    BOND = "bond"
    PRICE = "log-price"


class PottsAction(NamedTuple):
    """One action: a move, where it applies, and how hot.

    Parameters
    ----------
    kind : MoveKind
        ``FLIP``, ``WOLFF`` and ``NIEDERMAYER`` name a site and a label --- the
        flipped site, or the cluster's root and the colour it is recoloured to
        or transposed with. ``SWEEP`` and ``SWENDSEN_WANG`` name neither and
        visit every site.
    site, label : int
        The site and target label, or ``-1`` each where the kind names
        neither.
    rung : int
        Index into the environment's temperature ladder.
    """

    kind: MoveKind
    site: int
    label: int
    rung: int


class PottsNDEnvironment(Environment[Configuration, PottsAction]):
    """Search over an N-D Potts lattice at a known per-site field.

    One arm is one declared move set. ``kinds`` chooses it, and #706 measures
    four: single-site (both grains), Wolff, Swendsen-Wang, and the union of
    them. The three single-move arms are the controls --- each asks only
    whether a *learned* schedule beats a declared one for its move --- and the
    union arm asks the second question, whether choosing among moves beats the
    best single move, which is only answerable against those controls.

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
    moves : Mapping[MoveKind, KeyedMove] | None
        The cluster moves, one per kind in :data:`CLUSTER_KINDS` this arm
        admits, from :func:`snakes_and_ladders.sample.potts_keyed.cluster_moves`
        or another implementer. Supplied rather than imported because ``learn``
        imports nothing from ``search``; checked against this environment's own
        ``n_nodes`` and ``n_states``, since a move built on a different lattice
        would make the two callers score different problems without saying so.
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
        negative temperature, ``candidates`` is under one, a cluster kind
        arrives without its move or on a different lattice, or the arm's
        feature map would carry no column that varies.
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
        moves: Mapping[MoveKind, KeyedMove] | None = None,
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
        self._set_moves(moves or {})
        # Hoisted out of `features`: a rung's temperature and its bond
        # probability are the same two numbers at every decision, and the
        # alternative is a Python-level call per candidate (issue #706).
        self._temperatures = np.asarray(self._ladder, dtype=np.float64)
        self._bonds = np.asarray(
            [self.bond_probability(rung) for rung in range(len(self._ladder))],
            dtype=np.float64,
        )
        self._columns = self._surviving_columns()

    def _set_moves(self, moves: Mapping[MoveKind, KeyedMove]) -> None:
        """Hold each cluster kind's move, having checked it is this lattice's.

        A kind admitted without its move is refused here rather than at the
        first ``step``: an arm that cannot take half its actions is a
        misconfigured experiment, and the cheapest place to say so is where it
        was configured.
        """
        for kind in self._kinds:
            if kind not in CLUSTER_KINDS:
                continue
            move = moves.get(kind)
            if move is None:
                msg = (
                    f"{kind} needs its move passed as moves[{kind!r}]: a "
                    "cluster growth reads the graph, which `learn` does not "
                    "import (issue #706)"
                )
                raise ValueError(msg)
            if move.n_nodes != self._n_nodes or move.n_states != self._n_states:
                msg = (
                    f"{kind}'s move is on a ({move.n_nodes}, {move.n_states}) "
                    f"lattice and this one is ({self._n_nodes}, "
                    f"{self._n_states}): the two would score different problems"
                )
                raise ValueError(msg)
        self._moves = {
            kind: moves[kind] for kind in self._kinds if kind in CLUSTER_KINDS
        }

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
        for kind in self._kinds:
            if kind in PARAMETRIC_KINDS:
                offered.extend(self._sited_candidates(state, kind))
            else:
                offered.extend(
                    PottsAction(kind, -1, -1, rung) for rung in range(len(self._ladder))
                )
        return offered

    def _sited_candidates(
        self, state: Configuration, kind: MoveKind
    ) -> list[PottsAction]:
        """Sited actions to score: all, or ``candidates`` drawn by the state's key.

        One rule for both parametric kinds. A flip's ``(site, label)`` is where
        to move and to what; a Wolff step's is the cluster's root and the colour
        the cluster is recoloured to, so the candidate set has the same shape
        and is subset the same way. A target label equal to the site's current
        one is excluded: for a flip it is a no-op, and for Wolff it is the
        recolouring ``potts_mcmc.sweeps._recolour`` reports as never proposed.
        """
        labels = np.asarray(state, dtype=np.int64)
        whole = self._n_nodes * (self._n_states - 1) * len(self._ladder)
        sites = np.repeat(np.arange(self._n_nodes), self._n_states - 1)
        offsets = np.tile(np.arange(1, self._n_states), self._n_nodes)
        targets = (labels[sites] + offsets) % self._n_states
        pairs = np.stack([sites, targets], axis=1)
        if whole <= self._candidates:
            return [
                PottsAction(kind, int(site), int(label), rung)
                for site, label in pairs
                for rung in range(len(self._ladder))
            ]
        rng = keyed_generator(self._key, labels, -1 - self._kinds.index(kind))
        per_rung = max(1, self._candidates // len(self._ladder))
        chosen = rng.choice(pairs.shape[0], size=per_rung, replace=False)
        return [
            PottsAction(kind, int(pairs[index, 0]), int(pairs[index, 1]), rung)
            for index in chosen
            for rung in range(len(self._ladder))
        ]

    def visits(self, state: Configuration, action: PottsAction) -> int:
        """What ``action`` costs in site visits, realized rather than nominal.

        A flip writes its own label and reads its neighbours'; a sweep and a
        Swendsen-Wang pass are ``search.ground_state``'s ``n_nodes + 2
        n_edges``. The first two differ by about five hundred at 576 sites,
        which is why #706's budget carries the scored-candidate count beside
        this one: no single number makes them commensurable *and* prices the
        policy's own scoring.

        **A grown cluster's charge is its size**, which is not known
        until the cluster is grown, so this runs the move to find out. It is
        exact rather than an average because the growth is keyed on
        ``(state, action)``: the cluster counted here is the cluster
        :meth:`step` will build. The cost of the answer is one cluster growth,
        which is why :meth:`features` prices a *candidate* by the nominal
        charge below and not by this --- 128 candidates would otherwise cost
        128 growths, or some thirty sweeps of work per decision.
        """
        if action.kind in GROWN_KINDS:
            _, spent = self._moves[action.kind].propose(
                np.asarray(state, dtype=np.int64),
                temperature=self._ladder[action.rung],
                site=action.site,
                label=action.label,
                rng=self._move_generator(state, action),
            )
            return spent
        if action.kind in PARAMETRIC_KINDS:
            return 1 + int(self._degree[action.site])
        return self._sweep_visits

    def _nominal_visits(self, action: PottsAction) -> int:
        """A lower bound on ``action``'s charge, readable without running it.

        Exact for every kind but Wolff, where it is the root's own visit count
        --- a cluster is at least its root, so the bound holds and is what a
        candidate can be priced by in constant time. :meth:`visits` is the
        realized charge and is what a budget is debited by.
        """
        if action.kind in PARAMETRIC_KINDS:
            return 1 + int(self._degree[action.site])
        return self._sweep_visits

    def _move_generator(
        self, state: Configuration, action: PottsAction
    ) -> np.random.Generator:
        """The generator this ``(state, action)`` pair is entitled to."""
        return keyed_generator(
            self._key, np.asarray(state, dtype=np.int64), self.action_code(action)
        )

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
        if action.kind in CLUSTER_KINDS:
            return self._cluster(state, action, temperature)
        if action.kind is MoveKind.SWEEP:
            return self._sweep(state, action, temperature)
        return self._flip(state, action, temperature)

    def _cluster(
        self, state: Configuration, action: PottsAction, temperature: float
    ) -> tuple[Configuration, float]:
        """One cluster move, through the seam that carries it.

        The reward is read by rescoring rather than from the affected bonds:
        the move reports its successor and its charge, not which sites it
        touched, and a cluster can be the whole lattice. Rescoring is
        ``O(n_nodes + n_edges)`` against the move's own cost, which is the same
        order, so nothing is saved by tracking the delta and the score is the
        one :meth:`score` defines.
        """
        successor, _ = self._moves[action.kind].propose(
            np.asarray(state, dtype=np.int64),
            temperature=temperature,
            site=action.site,
            label=action.label,
            rng=self._move_generator(state, action),
        )
        after = tuple(int(value) for value in successor)
        return after, self.score(after) - self.score(state)

    def _flip(
        self, state: Configuration, action: PottsAction, temperature: float
    ) -> tuple[Configuration, float]:
        """One site's Metropolis update, deterministic at ``T = 0``."""
        gain = self._flip_gain(np.asarray(state, dtype=np.int64), action)
        accepted = accept_at_temperature(
            temperature, gain, lambda: self._move_generator(state, action).random()
        )
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
        ``sample.potts_mcmc.sweeps.single_site_sweep`` does.
        """
        labels = np.asarray(state, dtype=np.int64).copy()
        before = self.score(tuple(int(value) for value in labels))
        rng = self._move_generator(state, action)
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
    ) -> NDArray[np.float64]:
        """``(len(actions), n_features())``: one column per reading that varies.

        The columns are the arm's, chosen at construction by
        :meth:`_surviving_columns` and named by :class:`FeatureColumn`:

        * **gain** --- the score a single flip at the action's site would buy,
          or for a whole-lattice move the sum of every site's best single-flip
          gain. Neither is what the move will actually collect: a sweep's
          sites interact, and a Wolff step recolours a cluster rather than its
          root. Both are features and not promises, read in one vectorized
          pass, and what they are worth is what the policy learns.
        * **temperature** --- the rung's ``T``, the coordinate every action
          carries.
        * **bond** --- ``1 - exp(-J / T)``, the probability a like-coloured
          bond is active, which is the whole of what a rung means to a cluster
          move and is ``0`` for a single-site one. At ``T = 0`` it is ``1``
          exactly, which is why the cluster is the monochromatic component
          there.
        * **log-price** --- ``log`` of :meth:`_nominal_visits`, so a candidate
          costs constant time to price. A Wolff candidate is priced by its
          root alone; the realized charge is :meth:`visits`.

        **A gather per column, not a call per candidate.** The first draft read
        the gain through :meth:`_flip_gain` inside a generator, and `cProfile`
        put 70% of a decision there --- 0.442 s of 0.634 s over 300 decisions
        --- for 12.4 us a candidate. `learn/CLAUDE.md`'s inlining rule names
        that shape exactly: no Python-level call per node, hoist it or
        vectorize it. Vectorized and with the bound read only where an action
        needs it, a Wolff decision fell from 1,562 us to 171 us over 126
        candidates, a 9.1x cut, and the mixed arm's from 2,670 us to 393 us.
        What remains is the tuple reads, and folding them into one structured
        pass measured *slower* on the arms that keep the bound --- 293 us
        against 279 us --- so it was not kept.
        """
        width = len(self._columns)
        if not actions:
            return np.empty((0, width), dtype=np.float64)
        labels = np.asarray(state, dtype=np.int64)
        sited = np.fromiter(
            (action.kind in PARAMETRIC_KINDS for action in actions),
            dtype=bool,
            count=len(actions),
        )
        clustered = np.fromiter(
            (action.kind in CLUSTER_KINDS for action in actions),
            dtype=bool,
            count=len(actions),
        )
        rungs = np.fromiter(
            (action.rung for action in actions), dtype=np.int64, count=len(actions)
        )
        readings: dict[FeatureColumn, np.ndarray] = {}
        if FeatureColumn.GAIN in self._columns or FeatureColumn.PRICE in self._columns:
            sites = np.where(
                sited,
                np.fromiter(
                    (action.site for action in actions),
                    dtype=np.int64,
                    count=len(actions),
                ),
                0,
            )
        if FeatureColumn.GAIN in self._columns:
            targets = np.fromiter(
                (action.label for action in actions),
                dtype=np.int64,
                count=len(actions),
            )
            gains = self._flip_gains(labels, sites, np.where(sited, targets, 0))
            # The bound is `O(n_nodes * degree * n_states)` and was measured at
            # 137 us of a 268 us decision, so it is read only where an action
            # needs it --- never in a Wolff-only or flip-only arm.
            readings[FeatureColumn.GAIN] = (
                gains
                if sited.all()
                else np.where(sited, gains, self._greedy_bound(labels))
            )
        if FeatureColumn.TEMPERATURE in self._columns:
            readings[FeatureColumn.TEMPERATURE] = self._temperatures[rungs]
        if FeatureColumn.BOND in self._columns:
            readings[FeatureColumn.BOND] = np.where(clustered, self._bonds[rungs], 0.0)
        if FeatureColumn.PRICE in self._columns:
            readings[FeatureColumn.PRICE] = np.log(
                np.where(sited, 1.0 + self._degree[sites], float(self._sweep_visits))
            )
        return np.stack([readings[column] for column in self._columns], axis=1)

    def _flip_gains(
        self, labels: np.ndarray, sites: np.ndarray, targets: np.ndarray
    ) -> np.ndarray:
        """:meth:`_flip_gain` over many ``(site, target)`` pairs, in one gather.

        Bitwise the scalar reading and not merely close to it: the agreement
        term is a count, so summing it over the padded row under the mask gives
        the same integer as summing over the compacted row, and the two terms
        are combined in the same order. A test pins the pair rather than
        trusting the argument.
        """
        neighbours = self._neighbour_table[sites]
        mask = self._neighbour_mask[sites]
        theirs = labels[neighbours]
        current = labels[sites]
        agreement = ((theirs == targets[:, None]) & mask).sum(axis=1) - (
            (theirs == current[:, None]) & mask
        ).sum(axis=1)
        field = self._field[sites, targets] - self._field[sites, current]
        gains: np.ndarray = self._coupling * agreement + field
        return gains

    def bond_probability(self, rung: int) -> float:
        """``1 - exp(-J / T)`` at ``rung``: one at ``T = 0``, exactly.

        ``potts_mcmc``'s bond probability, read here so a feature and the move
        that consumes it cannot disagree about what a rung means. The limit is
        taken rather than divided: ``1 / 0`` is not a float, and the value it
        would stand for is ``1.0``.
        """
        temperature = self._ladder[rung]
        if temperature == 0.0:
            return 1.0
        return float(-np.expm1(-self._coupling / temperature))

    def _surviving_columns(self) -> tuple[FeatureColumn, ...]:
        """The columns that can vary in this arm, in :class:`FeatureColumn` order.

        Each test is structural --- a property of ``kinds``, the ladder and the
        lattice's degrees, not of any one state --- so the width is fixed for
        the environment's life, which a policy needs. A column that cannot vary
        is dropped rather than carried: it sits in the direction a softmax over
        scored actions cancels, so it is a weight nothing identifies.

        Raises
        ------
        ValueError
            If no column survives, which is an arm offering one action --- a
            single move at a single rung --- and so nothing to learn.
        """
        prices = {
            self._nominal_visits(PottsAction(kind, site, 0, 0))
            for kind in self._kinds
            for site in (range(self._n_nodes) if kind in PARAMETRIC_KINDS else (-1,))
        }
        bonds = {
            self.bond_probability(rung) if kind in CLUSTER_KINDS else 0.0
            for kind in self._kinds
            for rung in range(len(self._ladder))
        }
        survives = {
            FeatureColumn.GAIN: any(kind in PARAMETRIC_KINDS for kind in self._kinds),
            FeatureColumn.TEMPERATURE: len(set(self._ladder)) > 1,
            FeatureColumn.BOND: len(bonds) > 1,
            FeatureColumn.PRICE: len(prices) > 1,
        }
        columns = tuple(column for column in FeatureColumn if survives[column])
        if not columns:
            msg = (
                f"kinds={[str(kind) for kind in self._kinds]} on a "
                f"{len(self._ladder)}-rung ladder leaves no feature column "
                "that varies: every action would score alike and there is "
                "nothing for a policy to prefer"
            )
            raise ValueError(msg)
        return columns

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
        """One per column this arm's move set and ladder let vary."""
        return len(self._columns)

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
