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

**A context is a key, not a label.** One quantity recorded for each of a
ladder's rungs is one series under a context per rung, so the record of rung
``k`` is read back with ``{"rung": k}`` and ends at entry ``k`` of the vector
the result returns --- ``occupation[k]``, ``swap_acceptance[k]``,
``up_fraction[k]``. A context-free call under the same name is a third
series and is what :meth:`MemoryTracker.last` reads without one.

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
from snakes_and_ladders.opt.langevin import LangevinChain, mala
from snakes_and_ladders.opt.schedule import (
    ExponentialTempSchedule,
    adapt_ladder_by_round_trips,
)
from snakes_and_ladders.opt.slice import SliceChain, slice_sample
from snakes_and_ladders.parallel import map_tasks
from snakes_and_ladders.qa.figure import write_qa_figure
from snakes_and_ladders.search import annealed, potts_mcmc
from snakes_and_ladders.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from snakes_and_ladders.track import (
    NULL_TRACKER,
    MemoryTracker,
    NullTracker,
    current,
    track,
)

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


#: The gate-size ladder for the three annealed estimators: four rungs from
#: the exact `beta = 0` law to the target, eight replicas. Small enough that
#: the three run in a second on the Python sweep, long enough that a per-rung
#: series has three entries.
BETAS = (0.0, 0.25, 0.5, 1.0)
N_REPLICAS = 8
TEMPERING_SWEEPS = 40

#: The slice chain's width and stepping-out budget, as
#: `tests/regression/opt/test_opt_slice.py` sets them.
WIDTH = 2.0
MAX_STEPS_OUT = 10


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


def _mala() -> LangevinChain:
    return mala(
        _objective(),
        torch.Generator().manual_seed(SEED),
        N_SAMPLES,
        step_size=STEP_SIZE,
        burn_in=BURN_IN,
    )


def _slice() -> SliceChain:
    return slice_sample(
        _objective(),
        torch.Generator().manual_seed(SEED),
        N_SAMPLES,
        width=WIDTH,
        max_steps_out=MAX_STEPS_OUT,
        burn_in=BURN_IN,
    )


def _lattice() -> PottsGraph:
    return lattice_graph(SHAPE, BoundaryCondition.OPEN, COUPLING)


def _ais() -> annealed.LogPartition:
    return annealed.annealed_importance_sampling(
        _lattice(),
        FIELD,
        BETAS,
        np.random.default_rng(SEED),
        N_REPLICAS,
        backend=Backend.PYTHON,
    )


def _population() -> annealed.LogPartition:
    return annealed.population_annealing(
        _lattice(),
        FIELD,
        BETAS,
        np.random.default_rng(SEED),
        N_REPLICAS,
        backend=Backend.PYTHON,
    )


def _simulated(weights: np.ndarray) -> annealed.SimulatedTempered:
    return annealed.simulated_tempering(
        _lattice(),
        FIELD,
        BETAS,
        weights,
        np.random.default_rng(SEED),
        TEMPERING_SWEEPS,
        backend=Backend.PYTHON,
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


@pytest.mark.patch
@pytest.mark.smoke
def test_the_null_tracker_leaves_the_two_new_chains_bitwise() -> None:
    outside_mala, outside_slice = _mala(), _slice()
    with track("samplers", tracker=NullTracker()):
        inside_mala, inside_slice = _mala(), _slice()

    assert torch.equal(inside_mala.theta, outside_mala.theta)
    assert torch.equal(inside_mala.energy_error, outside_mala.energy_error)
    assert inside_mala.acceptance_rate == outside_mala.acceptance_rate
    assert inside_mala.force_evaluations == outside_mala.force_evaluations
    assert torch.equal(inside_slice.theta, outside_slice.theta)
    assert inside_slice.objective_evaluations == outside_slice.objective_evaluations
    assert inside_slice.shrinkages_per_draw == outside_slice.shrinkages_per_draw


@pytest.mark.patch
@pytest.mark.smoke
def test_the_null_tracker_leaves_the_three_annealed_runs_bitwise() -> None:
    weights = annealed.rung_weights(_ais())
    outside_ais, outside_population = _ais(), _population()
    outside_tempered = _simulated(weights)
    with track("annealed", tracker=NullTracker()):
        inside_ais, inside_population = _ais(), _population()
        inside_tempered = _simulated(weights)

    for inside, outside in (
        (inside_ais, outside_ais),
        (inside_population, outside_population),
    ):
        assert inside.log_z == outside.log_z
        assert inside.stderr == outside.stderr
        assert inside.ess == outside.ess
        assert np.array_equal(inside.rung_log_z, outside.rung_log_z)
        assert np.array_equal(inside.log_weights, outside.log_weights)
        assert inside.family_entropy == outside.family_entropy
    assert np.array_equal(inside_tempered.rungs, outside_tempered.rungs)
    assert np.array_equal(inside_tempered.states, outside_tempered.states)
    assert np.array_equal(inside_tempered.occupation, outside_tempered.occupation)
    assert inside_tempered.acceptance == outside_tempered.acceptance


@pytest.mark.smoke
def test_the_langevin_chain_records_the_counters_the_chain_returns() -> None:
    with track("mala") as tracker:
        chain = _mala()

    assert isinstance(tracker, MemoryTracker)
    # MALA runs `hmc._run_chain` with its own kernel, so the series are that
    # loop's and they end at `LangevinChain`'s fields rather than at
    # `HmcChain`'s.
    assert tracker.last("acceptance_so_far") == chain.acceptance_rate
    assert tracker.last("force_evaluations") == chain.force_evaluations
    assert len(tracker.scalars["energy_error"]) == N_SAMPLES
    assert [value for _, value in tracker.scalars["energy_error"]] == [
        float(error) for error in chain.energy_error
    ]
    assert tracker.last("state_bytes") == float(chain.theta.nbytes)


@pytest.mark.smoke
def test_the_slice_chain_records_the_evaluations_it_reports() -> None:
    with track("slice_sample") as tracker:
        chain = _slice()

    assert isinstance(tracker, MemoryTracker)
    # One record per sweep, the burn-in's included: the three per-draw means
    # the result reports divide by every sweep, so a series over the recorded
    # draws alone would end on a different divisor.
    sweeps = N_SAMPLES + BURN_IN
    assert len(tracker.scalars["objective_evaluations"]) == sweeps
    assert tracker.last("objective_evaluations") == float(chain.objective_evaluations)
    assert tracker.last("evaluations_per_draw") == chain.evaluations_per_draw
    assert tracker.last("expansions_per_draw") == chain.expansions_per_draw
    assert tracker.last("shrinkages_per_draw") == chain.shrinkages_per_draw
    # The cumulative count never falls, and the evaluations of the first
    # sweep are the first entry.
    counts = [value for _, value in tracker.scalars["objective_evaluations"]]
    assert counts == sorted(counts)
    assert tracker.last("state_bytes") == float(chain.theta.nbytes)


@pytest.mark.smoke
def test_the_two_log_z_estimators_record_the_estimate_and_its_ess() -> None:
    with track("annealed_importance_sampling") as importance:
        estimate = _ais()
    with track("population_annealing") as population:
        resampled = _population()

    assert isinstance(importance, MemoryTracker)
    assert isinstance(population, MemoryTracker)
    rungs = len(BETAS) - 1
    assert len(importance.scalars["log_z"]) == rungs
    assert importance.last("log_z") == estimate.log_z
    assert importance.last("ess") == estimate.ess
    # The series is the ladder's own `rung_log_z`, read as it is formed.
    assert [value for _, value in importance.scalars["log_z"]] == [
        float(value) for value in estimate.rung_log_z[1:]
    ]
    assert len(population.scalars["log_z"]) == rungs
    assert population.last("log_z") == resampled.log_z
    assert population.last("ess") == resampled.ess
    assert population.last("stderr") == resampled.stderr
    assert population.last("family_entropy") == resampled.family_entropy
    assert [value for _, value in population.scalars["log_z"]] == [
        float(value) for value in resampled.rung_log_z[1:]
    ]


@pytest.mark.smoke
def test_the_tempering_walker_records_one_occupation_series_per_rung() -> None:
    weights = annealed.rung_weights(_ais())
    with track("simulated_tempering") as tracker:
        walker = _simulated(weights)

    assert isinstance(tracker, MemoryTracker)
    assert len(tracker.scalars["acceptance"]) == TEMPERING_SWEEPS
    assert tracker.last("acceptance") == walker.acceptance
    # One series under a context per rung, each ending at that rung's entry
    # of the occupation the result reports.
    for rung in range(len(BETAS)):
        context = {"rung": rung}
        assert len(tracker.series("occupation", context)) == TEMPERING_SWEEPS
        assert tracker.last("occupation", context) == walker.occupation[rung]
    assert (
        sum(tracker.last("occupation", {"rung": k}) for k in range(len(BETAS))) == 1.0
    )
    # The walker crossed the ladder under the pilot's weights, so the
    # per-rung series are not one series and three zeros.
    assert (
        sum(tracker.last("occupation", {"rung": k}) > 0.0 for k in range(len(BETAS)))
        >= 2
    )
    # `occupation` was never recorded outside a context, and reading it
    # without one is a read of a series nobody wrote.
    with pytest.raises(KeyError):
        tracker.last("occupation")


@pytest.mark.smoke
def test_the_tempered_run_records_the_acceptance_of_each_pair() -> None:
    with track("potts_parallel_tempering") as tracker:
        chains = _tempered()

    assert isinstance(tracker, MemoryTracker)
    # The mean stays the context-free series part 1 wrote; the vector behind
    # it is one series per adjacent pair, each ending at its own entry.
    assert tracker.last("swap_acceptance") == float(np.mean(chains.swap_acceptance))
    for pair in range(len(TEMPERATURES) - 1):
        context = {"rung": pair}
        assert len(tracker.series("swap_acceptance", context)) == SWEEPS
        assert tracker.last("swap_acceptance", context) == chains.swap_acceptance[pair]


@pytest.mark.smoke
def test_the_round_trip_warm_up_records_what_each_round_placed() -> None:
    # The closed-form profile `tests/regression/opt/test_opt_schedule.py`
    # places from: `f` linear in temperature, whose placement is the ladder
    # uniform in `T`. Two rounds, ten replicas measured.
    def measure(ladder: tuple[float, ...]) -> list[float]:
        values = np.array(ladder)
        return list((values - values[-1]) / (values[0] - values[-1]))

    start = (8.0, 4.0, 2.0, 1.0, 0.5)
    with track("adapt_ladder_by_round_trips") as tracker:
        placed = adapt_ladder_by_round_trips(measure, start, 1e-9, 5)

    assert isinstance(tracker, MemoryTracker)
    assert placed.rounds == 2
    assert len(tracker.scalars["replicas_measured"]) == placed.rounds
    assert tracker.last("replicas_measured") == float(placed.replicas_measured)
    for rung in range(len(start)):
        context = {"rung": rung}
        assert len(tracker.series("temperature", context)) == placed.rounds
        assert tracker.last("temperature", context) == placed.temperatures[rung]
        assert tracker.last("up_fraction", context) == placed.up_fraction[rung]


@pytest.mark.smoke
def test_the_default_tracker_is_the_instance_a_guarded_hook_tests_for() -> None:
    # A hook whose inner loop is one call per rung runs that loop only under
    # `tracker is not NULL_TRACKER`, so the default has to be that one shared
    # instance rather than an equal one, inside a block it must not be, and
    # the block's tracker is what `current()` returns.
    assert current() is NULL_TRACKER
    with track("guarded") as tracker:
        assert current() is tracker
        assert current() is not NULL_TRACKER
    assert current() is NULL_TRACKER
    # A `NullTracker` the caller builds is a different object, so a guarded
    # loop does run and its no-ops discard what it records: that run is still
    # bitwise the untracked one, which the two `patch` tests above pin.
    with track("its own null", tracker=NullTracker()):
        assert current() is not NULL_TRACKER


@pytest.mark.smoke
def test_a_context_keys_a_series_and_the_context_free_call_is_its_own() -> None:
    with track("contexts") as tracker:
        current().scalar("occupation", 0.25, 0)
        current().scalar("occupation", 0.5, 0, context={"rung": 0})
        current().scalar("occupation", 0.75, 0, context={"rung": 1})
        # The same context written twice is one series, and the order its
        # items were written in is not part of the key.
        current().scalar("occupation", 0.125, 1, context={"rung": 1, "chain": 2})
        current().scalar("occupation", 0.25, 2, context={"chain": 2, "rung": 1})

    assert isinstance(tracker, MemoryTracker)
    assert len(tracker.keyed) == 3
    assert tracker.scalars["occupation"] == [(0, 0.25)]
    assert tracker.last("occupation") == 0.25
    assert tracker.series("occupation", {"rung": 0}) == [(0, 0.5)]
    assert tracker.series("occupation", {"rung": 1}) == [(0, 0.75)]
    assert tracker.series("occupation", {"rung": 1, "chain": 2}) == [
        (1, 0.125),
        (2, 0.25),
    ]
    with pytest.raises(KeyError):
        tracker.series("occupation", {"rung": 2})


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


@pytest.mark.release
@pytest.mark.smoke
def test_an_aim_run_reads_back_one_sequence_per_context(tmp_path: Path) -> None:
    aim = pytest.importorskip("aim")
    from snakes_and_ladders.track import AimTracker

    weights = annealed.rung_weights(_ais())
    with track("simulated_tempering") as expected:
        _simulated(weights)

    written = AimTracker(repo=tmp_path, experiment="track_probe")
    run_hash = written.run_hash
    with track("simulated_tempering", tracker=written):
        _simulated(weights)

    aim.Repo.from_path(str(tmp_path))._recreate_index()
    back = aim.Run(run_hash=run_hash, repo=str(tmp_path), read_only=True)
    # A context is Aim's own per-series key, so one name under four rung
    # contexts is four sequences there as it is four series here.
    sequences = {
        (metric.name, tuple(sorted(metric.context.to_dict().items()))): metric
        for metric in back.metrics()
    }

    assert isinstance(expected, MemoryTracker)
    assert len([name for name, _ in sequences if name == "occupation"]) == len(BETAS)
    for rung in range(len(BETAS)):
        recorded = [value for _, value in expected.series("occupation", {"rung": rung})]
        sequence = sequences["occupation", (("rung", rung),)]
        # Compared as a multiset over the steps recorded, for the reason the
        # test above states.
        assert sorted(sequence.values.values_list()) == sorted(recorded)
        assert sequence.last_step() == len(recorded) - 1
    acceptance = sequences["acceptance", ()]
    assert sorted(acceptance.values.values_list()) == sorted(
        value for _, value in expected.scalars["acceptance"]
    )
