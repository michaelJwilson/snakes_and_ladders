"""The run tracker: it records what the run returns, and it changes nothing (issue #778).

Two claims, and the first is the one a sampler's caller cares about. **A
tracked run is the untracked run**, bitwise: the same seed gives the same
draws, the same labelling and the same counters under the default
:class:`NullTracker` as it does outside any context, so a hook cannot be a
source of scientific difference.

**And a recorded series ends at the field the result reports.** Every hook
records a number its dataclass already returns --- the acceptance rate, the
gradients spent, the best energy, the fitted value --- so the last entry of a
series equals that field by ``==`` rather than to a tolerance. That is what
makes the series readable as the field's history instead of as a second
definition to keep in step.

The referee is :class:`MemoryTracker`, and the Aim round trip is refereed by
the ``MemoryTracker`` record of the same call: what Aim reads back must be
what the run recorded, or the optional store is not the same store.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pytest
import torch
from matplotlib.figure import Figure
from snakes_and_ladders.backend import Backend
from snakes_and_ladders.opt import hmc
from snakes_and_ladders.opt.fit import fit
from snakes_and_ladders.opt.schedule import ExponentialTempSchedule
from snakes_and_ladders.parallel import map_tasks
from snakes_and_ladders.qa.figure import write_qa_figure
from snakes_and_ladders.search import potts_mcmc
from snakes_and_ladders.sim.graph import BoundaryCondition, lattice_graph
from snakes_and_ladders.track import MemoryTracker, NullTracker, current, track

from tests._objective_checks import AnalyticGaussian

mpl.use("Agg")

#: The gate-size chain: long enough that a series has a shape, short enough
#: that three of them run in a second.
N_SAMPLES = 40
BURN_IN = 5
STEP_SIZE = 0.3
SEED = 4242

#: The gate-size Potts instance: the 2x2 two-state lattice the sampler's own
#: tests enumerate, so nothing here has to declare a new one.
SHAPE = (2, 2)
COUPLING = 0.8
FIELD = np.zeros(2)
SWEEPS = 30
TEMPERATURES = (0.5, 1.0, 2.0)


def _objective() -> AnalyticGaussian:
    return AnalyticGaussian([0.4, -0.7], [[1.0, 0.3], [0.3, 2.0]])


def _chain() -> hmc.HmcChain:
    return hmc.sample(
        _objective(),
        torch.Generator().manual_seed(SEED),
        N_SAMPLES,
        step_size=STEP_SIZE,
        burn_in=BURN_IN,
    )


def _annealed() -> potts_mcmc.AnnealedPotts:
    graph = lattice_graph(SHAPE, BoundaryCondition.OPEN, COUPLING)
    return potts_mcmc.anneal_potts(
        graph,
        FIELD,
        ExponentialTempSchedule(start=2.0, end=0.1, n_steps=SWEEPS),
        np.random.default_rng(SEED),
        backend=Backend.PYTHON,
    )


def _tempered() -> potts_mcmc.TemperedChains:
    graph = lattice_graph(SHAPE, BoundaryCondition.OPEN, COUPLING)
    return potts_mcmc.parallel_tempering(
        graph,
        FIELD,
        TEMPERATURES,
        np.random.default_rng(SEED),
        SWEEPS,
        backend=Backend.PYTHON,
    )


@pytest.mark.patch
@pytest.mark.smoke
def test_the_null_tracker_leaves_a_chain_bitwise_what_it_was() -> None:
    outside = _chain()
    with track("hmc", tracker=NullTracker()):
        inside = _chain()

    assert torch.equal(inside.theta, outside.theta)
    assert torch.equal(inside.energy_error, outside.energy_error)
    assert inside.acceptance_rate == outside.acceptance_rate
    assert inside.force_evaluations == outside.force_evaluations


@pytest.mark.patch
@pytest.mark.smoke
def test_the_null_tracker_leaves_an_annealed_and_a_tempered_run_bitwise() -> None:
    outside_annealed, outside_tempered = _annealed(), _tempered()
    with track("potts", tracker=NullTracker()):
        inside_annealed, inside_tempered = _annealed(), _tempered()

    assert np.array_equal(inside_annealed.labelling, outside_annealed.labelling)
    assert np.array_equal(inside_annealed.final, outside_annealed.final)
    assert inside_annealed.energy == outside_annealed.energy
    assert inside_annealed.site_visits == outside_annealed.site_visits
    assert np.array_equal(inside_tempered.states, outside_tempered.states)
    assert np.array_equal(
        inside_tempered.swap_acceptance, outside_tempered.swap_acceptance
    )
    assert inside_tempered.best_energy == outside_tempered.best_energy


@pytest.mark.smoke
def test_the_chain_records_the_counters_the_chain_returns() -> None:
    with track("hmc") as tracker:
        chain = _chain()

    assert isinstance(tracker, MemoryTracker)
    assert tracker.parameters["name"] == "hmc"
    assert tracker.last("acceptance_so_far") == chain.acceptance_rate
    assert tracker.last("force_evaluations") == chain.force_evaluations
    assert len(tracker.scalars["energy_error"]) == N_SAMPLES
    assert [step for step, _ in tracker.scalars["energy_error"]] == list(
        range(N_SAMPLES)
    )
    # The error recorded per draw is the error the chain reports for it, and
    # the burn-in's errors are not recorded: `HmcChain.energy_error` drops
    # them too.
    assert [value for _, value in tracker.scalars["energy_error"]] == [
        float(error) for error in chain.energy_error
    ]
    assert tracker.last("state_bytes") == float(chain.theta.nbytes)
    assert tracker.last("peak_rss_bytes") > 0.0


@pytest.mark.smoke
def test_the_annealer_records_one_energy_per_sweep_ending_at_the_result() -> None:
    with track("anneal_potts") as tracker:
        annealed = _annealed()

    assert isinstance(tracker, MemoryTracker)
    assert len(tracker.scalars["energy"]) == annealed.n_sweeps == SWEEPS
    assert tracker.last("energy") == annealed.energy
    # The series is the best so far, so it never rises.
    energies = [value for _, value in tracker.scalars["energy"]]
    assert energies == sorted(energies, reverse=True)
    assert tracker.last("temperature") == pytest.approx(0.1)
    assert tracker.last("state_bytes") == float(annealed.final.nbytes)


@pytest.mark.smoke
def test_the_tempered_run_records_the_swap_acceptance_it_returns() -> None:
    with track("potts_parallel_tempering") as tracker:
        chains = _tempered()

    assert isinstance(tracker, MemoryTracker)
    assert len(tracker.scalars["swap_acceptance"]) == SWEEPS
    assert tracker.last("swap_acceptance") == float(np.mean(chains.swap_acceptance))
    assert tracker.last("state_bytes") == float(chains.states[0].nbytes)


@pytest.mark.smoke
def test_the_fit_records_the_value_and_the_norm_the_result_reports() -> None:
    objective = _objective()
    with track("fit") as tracker:
        result = fit(objective, torch.tensor([2.0, -3.0], dtype=torch.float64))

    assert isinstance(tracker, MemoryTracker)
    assert result.converged
    assert tracker.last("objective") == result.value
    assert tracker.last("relative_gradient_norm") == result.gradient_norm
    # One record per iteration, and one closing record carrying the result.
    assert len(tracker.scalars["objective"]) == result.iterations + 1
    assert tracker.last("wall_s") > 0.0


@pytest.mark.smoke
def test_two_contexts_record_only_their_own_and_outside_records_nothing() -> None:
    # `map_tasks` on the serial backend runs the task in the calling context,
    # which is the context the tracker lives in.
    def task(item: int) -> int:
        current().scalar("item", float(item), item)
        return item

    with track("first") as first:
        map_tasks(task, [0, 1], workers=1, backend="serial", intra_op_threads=None)
    with track("second") as second:
        map_tasks(task, [2], workers=1, backend="serial", intra_op_threads=None)

    assert isinstance(first, MemoryTracker)
    assert isinstance(second, MemoryTracker)
    assert first.scalars["item"] == [(0, 0.0), (1, 1.0)]
    assert second.scalars["item"] == [(2, 2.0)]
    assert first.parameters["name"] == "first"
    assert second.parameters["name"] == "second"
    assert isinstance(current(), NullTracker)


@pytest.mark.smoke
def test_a_pool_worker_starts_outside_the_context_and_records_nothing() -> None:
    # A `ContextVar` is not carried into a pool worker, which is why
    # `track.py` says a per-task run opens its own context. Pinned rather than
    # left to be discovered from an empty series.
    with track("threads") as tracker:
        map_tasks(
            _record_in_worker, [0, 1], workers=2, backend="threads", intra_op_threads=1
        )

    assert isinstance(tracker, MemoryTracker)
    assert "item" not in tracker.scalars


def _record_in_worker(item: int) -> int:
    current().scalar("item", float(item), item)
    return item


@pytest.mark.smoke
def test_a_written_qa_figure_is_recorded_under_its_stem(tmp_path: Path) -> None:
    figure = Figure()
    figure.add_subplot().plot([0.0, 1.0], [1.0, 0.0])

    with track("qa") as tracker:
        written = write_qa_figure(tmp_path, "track_probe", figure, "A probe.")

    assert isinstance(tracker, MemoryTracker)
    assert tracker.figures["track_probe"] is figure
    assert written.figure_path.exists()


@pytest.mark.release
@pytest.mark.smoke
def test_an_aim_run_reads_back_what_the_memory_tracker_recorded(
    tmp_path: Path,
) -> None:
    aim = pytest.importorskip("aim")
    from snakes_and_ladders.track import AimTracker

    with track("hmc") as expected:
        _chain()

    written = AimTracker(repo=tmp_path, experiment="track_probe")
    run_hash = written.run_hash
    with track("hmc", tracker=written):
        _chain()

    # Aim serves a read from its index, which a writing run does not update;
    # `aim storage --repo <path> reindex` is the CLI for it, and this is the
    # call that command makes.
    aim.Repo.from_path(str(tmp_path))._recreate_index()
    back = aim.Run(run_hash=run_hash, repo=str(tmp_path), read_only=True)
    sequences = {metric.name: metric for metric in back.metrics()}

    assert isinstance(expected, MemoryTracker)
    assert back["name"] == "hmc"
    for name in ("acceptance_so_far", "energy_error", "force_evaluations"):
        recorded = [value for _, value in expected.scalars[name]]
        sequence = sequences[name]
        # The values are compared as a multiset and the steps by their range:
        # aim 3.29's ``values_list`` returns a sequence in the order its
        # storage holds it rather than in step order, and reading a value at
        # a step raises. What is refereed is that every number the run
        # recorded reached the store, once, over the steps it recorded them
        # at --- the claim the extra is carried for.
        assert sorted(sequence.values.values_list()) == sorted(recorded)
        assert sequence.first_step() == 0
        assert sequence.last_step() == len(recorded) - 1
