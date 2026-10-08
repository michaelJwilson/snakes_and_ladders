"""The Rust step loop against the enumerated law and the Python loop's accounting (issue #1368).

``oxisal.potts_loop`` runs a whole anneal or chain on a ChaCha8 stream seeded
by one draw of the run's generator, so its chain is of the Python loop's law
and is not its chain: bitwise agreement with the Python loop is not expected
and is not asserted. The referees:

* :func:`tests._chains.enumerated_law` on #1322's 2x3 instance (a per-site
  field with one forbidden label), against which every move the loop runs,
  alone and composed with a Gibbs sweep, is fitted by chi-square, at the
  significance and pooling `test_potts_recolour.py` declares;
* the Python loop's own accounting: ``spent``, the schedule index per step
  and the stop, equal where the move costs are fixed, and replayed through
  :func:`~sal.sample.loop.anneal_spent` on the charges the Rust loop
  returns where they are not;
* a move set the loop does not run, which runs the Python loop bitwise.
"""

from __future__ import annotations

import numpy as np
import pytest
from sal.backend import Backend
from sal.opt.budget import Budget
from sal.cost import Cost
from sal.sample.loop import Moved, anneal_spent
from sal.sample.potts_mcmc import PottsMove, Recolour, anneal_potts, sample_potts
from sal.sample.potts_mcmc.chains import loop_codes, run_loop
from sal.sample.schedule import ramp
from sal.sim.potts import site_field

from tests._chains import cell_counts, enumerated_law
from tests.regression.sample.test_potts_recolour import (
    FIELD,
    RECORDED,
    SIGNIFICANCE,
    THINNING,
    _graph,
    _pooled_p_value,
)

SEED = 1368


@pytest.mark.oracle
@pytest.mark.parametrize(
    "moves",
    [
        (PottsMove.SINGLE_SITE,),
        (PottsMove.SWENDSEN_WANG,),
        (PottsMove.SWENDSEN_WANG, PottsMove.SINGLE_SITE),
        (PottsMove.WOLFF,),
        (PottsMove.WOLFF, PottsMove.SINGLE_SITE),
        (PottsMove.WOLFF_HEAT_BATH,),
        (PottsMove.WOLFF_HEAT_BATH, PottsMove.SINGLE_SITE),
    ],
    ids=[
        "gibbs",
        "sw",
        "sw+gibbs",
        "wolff",
        "wolff+gibbs",
        "wolff-heat-bath",
        "wolff-heat-bath+gibbs",
    ],
)
@pytest.mark.parametrize("budget", [False, True], ids=["sweeps", "budget"])
def test_the_rust_loop_leaves_the_boltzmann_law_invariant(
    moves: tuple[PottsMove, ...], budget: bool
) -> None:
    """Chi-square of the thinned Rust-loop chain against the enumerated law, with no visit off its support."""
    graph = _graph()
    assert loop_codes(moves, Backend.RUST) is not None
    index, probability = enumerated_law(graph, FIELD)
    # Under a budget, the visits the fixed-length chain's recorded steps cost
    # where every step were one sweep; a Wolff chain records more steps.
    per_sweep = graph.n_nodes + 2 * len(graph.edges)
    size = Budget(Cost.SITE_VISITS, RECORDED * THINNING * per_sweep)
    chain = sample_potts(
        graph,
        FIELD,
        moves,
        np.random.default_rng(SEED),
        size if budget else RECORDED,
        burn_in=RECORDED // 10,
        thin=THINNING,
        recolour=Recolour.UNIFORM,
        loop_backend=Backend.RUST,
    )
    counts = cell_counts(index, chain.states)
    support = probability > 0
    # A composition with a sweep costs at least one sweep a step.
    assert len(chain.states) >= RECORDED // len(moves)
    assert counts[~support].sum() == 0
    assert _pooled_p_value(probability[support], counts[support]) > SIGNIFICANCE


@pytest.mark.analytic
@pytest.mark.parametrize(
    "moves",
    [(PottsMove.SINGLE_SITE,), (PottsMove.SWENDSEN_WANG, PottsMove.SINGLE_SITE)],
    ids=["gibbs", "sw+gibbs"],
)
@pytest.mark.parametrize("budget", [None, 1_000], ids=["schedule", "budget"])
def test_a_fixed_cost_anneal_spends_and_stops_as_the_python_loop(
    moves: tuple[PottsMove, ...], budget: int | None
) -> None:
    """``spent``, the steps run and the termination equal the Python loop's on fixed-cost moves."""
    graph = _graph()
    runs = {
        backend: anneal_potts(
            graph,
            FIELD,
            ramp.linear(2.0, 0.5, 20),
            np.random.default_rng(SEED),
            move=moves,
            recolour=Recolour.UNIFORM,
            budget=None if budget is None else Budget(Cost.SITE_VISITS, budget),
            loop_backend=backend,
        )
        for backend in (Backend.PYTHON, Backend.RUST)
    }
    python, rust = runs[Backend.PYTHON], runs[Backend.RUST]
    assert (rust.spent, rust.n_sweeps) == (python.spent, python.n_sweeps)
    assert rust.termination == python.termination


@pytest.mark.analytic
@pytest.mark.parametrize("move", [PottsMove.WOLFF, PottsMove.WOLFF_HEAT_BATH])
def test_a_wolff_budget_replays_on_the_python_loop(move: PottsMove) -> None:
    """The Rust loop's charges, replayed through ``anneal_spent``, give its schedule indices, ``spent`` and stop."""
    graph = _graph()
    schedule = ramp.linear(3.0, 0.5, 25)
    rows = site_field(FIELD, graph.n_nodes)
    state = np.zeros(graph.n_nodes, dtype=np.int64)
    state[0] = 1  # off the forbidden label at site 0
    codes = loop_codes((move,), Backend.RUST)
    assert codes is not None
    budget = 400
    ran = run_loop(
        state,
        rows,
        graph.compressed_adjacency(),
        codes,
        np.array([schedule(i) for i in range(schedule.n_steps)]),
        np.random.default_rng(SEED),
        budget=budget,
        n_main=schedule.n_steps,
        track_best=True,
    )
    charges = [int(each) for each in ran["charges"]]
    temperatures: list[float] = []

    def replay(
        s: np.ndarray, energy: float, carried: None, temperature: float, rng: None
    ) -> Moved[np.ndarray, None]:
        temperatures.append(temperature)
        return Moved(s, energy, None, charges[len(temperatures) - 1])

    walked = anneal_spent(
        replay, schedule, Moved(state, 0.0, None, 0), None, np.copy, budget=budget
    )
    assert temperatures == [schedule(int(i)) for i in ran["indices"]]
    assert walked.spent == int(ran["spent_total"]) == sum(charges)
    assert walked.termination.iterations == int(ran["n_main"]) == len(charges)


@pytest.mark.analytic
def test_a_move_without_a_rust_kernel_runs_the_python_loop_bitwise() -> None:
    """Heat-bath Swendsen-Wang (#1364) falls back to the Python loop for the whole run."""
    graph = _graph()
    assert loop_codes((PottsMove.SWENDSEN_WANG_HEAT_BATH,), Backend.RUST) is None
    runs = [
        anneal_potts(
            graph,
            FIELD,
            ramp.linear(2.0, 0.5, 10),
            np.random.default_rng(SEED),
            move=[PottsMove.SWENDSEN_WANG_HEAT_BATH],
            loop_backend=backend,
        )
        for backend in (Backend.PYTHON, Backend.RUST)
    ]
    assert np.array_equal(runs[0].best, runs[1].best)
    assert runs[0].energy == runs[1].energy
    assert runs[0].spent == runs[1].spent
