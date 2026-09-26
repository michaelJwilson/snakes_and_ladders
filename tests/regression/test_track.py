"""The run seam: it records what the run returns, it scores what the state means, and it changes nothing (issue #778).

**A tracked run is the untracked run**, bitwise: same seed, same draws,
labelling and counters under :data:`NULL_RUN`. **A recorded series ends at the
field the result reports**, by ``==``. **A metric is a function the package
already has**: each :class:`~sal.track.Metrics` is pinned to
the function its docstring cites and to the truth where the problem has one
(zero split distance, zero bit errors, zero distance at a known minimizer).
The referee is :class:`~sal.track.MemoryRun`, which also
referees the Aim round trip.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pytest
import torch
from matplotlib.figure import Figure
from sal.backend import Backend
from sal.emissions import CategoricalEmission
from sal.likelihood import parsimony, pruning
from sal.likelihood.hmm_paths import enumerate_hidden_paths
from sal.likelihood.ldpc import CodeMetrics
from sal.likelihood.mixture_assignments import (
    enumerate_mixture_assignments,
)
from sal.likelihood.objective import TreeMetrics
from sal.opt import testfunctions
from sal.opt.fit import fit
from sal.opt.hmm import HmmMetrics, HmmObjective, forward_log_likelihood
from sal.opt.mixture import (
    GaussianMixtureObjective,
    MixtureMetrics,
    clustering_cost,
    mixture_log_likelihood,
)
from sal.opt.testfunctions import Himmelblau, Rastrigin, Rosenbrock
from sal.parallel import map_tasks
from sal.qa.figure import write_qa_figure
from sal.sample import (
    annealed,
    hmc,
    langevin,
    potts_mcmc,
    slice,
    tempered,
)
from sal.sample.schedule import ExponentialTempSchedule
from sal.sim.elementary_codes import hamming_code
from sal.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from sal.sim.hmm import HmmParams
from sal.sim.potts import PottsMetrics, energy
from sal.sim.topology import robinson_foulds
from sal.sim.tree import Node
from sal.track import (
    NULL,
    NULL_RUN,
    MemoryRun,
    Run,
    current,
    track,
)

from tests._fixtures import ZONE_INTERNAL, balanced_four_taxa
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

#: A labelling of that lattice, and its energy by hand. The open 2x2 lattice
#: has four edges --- (0,1), (0,2), (1,3), (2,3) --- and this labelling agrees
#: across two of them, so ``E = -2 * 0.8``. The field is zero, so no site term
#: enters.
LABELLING = np.array([0, 1, 0, 1])
LABELLING_ENERGY = -2 * COUPLING

#: The four-taxon alignment the tree metrics are read on. Column 0 splits
#: ``AB`` from ``CD`` and the other three are constant, so ``((A,B),(C,D))``
#: explains it with one change and ``((A,C),(B,D))`` needs two.
TREE_ALIGNMENT = {
    "A": np.array([0, 1, 2, 3]),
    "B": np.array([0, 1, 2, 3]),
    "C": np.array([1, 1, 2, 3]),
    "D": np.array([1, 1, 2, 3]),
}
TREE_STATES = 4
TREE_PI = np.full(TREE_STATES, 1.0 / TREE_STATES)


class _Exploding:
    """A metrics set that fails when it is called: the null path must not reach it."""

    names: tuple[str, ...] = ("never",)

    def __call__(self, state: object) -> dict[str, float]:  # noqa: ARG002
        """Fail, whatever the state is."""
        msg = "a metric was computed on the null run"
        raise AssertionError(msg)


def _objective() -> AnalyticGaussian:
    return AnalyticGaussian([0.4, -0.7], [[1.0, 0.3], [0.3, 2.0]])


def _graph() -> PottsGraph:
    return lattice_graph(SHAPE, BoundaryCondition.OPEN, COUPLING)


def _chain() -> hmc.HmcChain:
    return hmc.sample(
        _objective(),
        torch.Generator().manual_seed(SEED),
        N_SAMPLES,
        step_size=STEP_SIZE,
        burn_in=BURN_IN,
    )


def _annealed() -> potts_mcmc.AnnealedPotts:
    return potts_mcmc.anneal_potts(
        _graph(),
        FIELD,
        ExponentialTempSchedule(start=2.0, end=0.1, n_steps=SWEEPS),
        np.random.default_rng(SEED),
        backend=Backend.PYTHON,
    )


def _tempered() -> potts_mcmc.TemperedChains:
    return potts_mcmc.parallel_tempering(
        _graph(),
        FIELD,
        TEMPERATURES,
        np.random.default_rng(SEED),
        SWEEPS,
        backend=Backend.PYTHON,
    )


def _memory(run: Run) -> MemoryRun:
    """The run as the referee it is, so an assertion reads it rather than writes it."""
    assert isinstance(run, MemoryRun)
    return run


@pytest.mark.patch
@pytest.mark.smoke
def test_the_null_run_leaves_a_chain_bitwise_what_it_was() -> None:
    outside = _chain()
    with track(NULL_RUN):
        inside = _chain()

    assert torch.equal(inside.draws, outside.draws)
    assert torch.equal(inside.energy_error, outside.energy_error)
    assert inside.acceptance_rate == outside.acceptance_rate
    assert inside.force_evaluations == outside.force_evaluations


@pytest.mark.patch
@pytest.mark.smoke
def test_the_null_run_leaves_an_annealed_and_a_tempered_run_bitwise() -> None:
    outside_annealed, outside_tempered = _annealed(), _tempered()
    with track(NULL_RUN):
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
    with track(name="hmc") as tracked:
        chain = _chain()

    run = _memory(tracked.run)
    assert run.params["name"] == "hmc"
    assert run.last("acceptance_so_far") == chain.acceptance_rate
    assert run.last("force_evaluations") == chain.force_evaluations
    assert len(run.series("energy_error")) == N_SAMPLES
    assert [step for step, _ in run.series("energy_error")] == list(range(N_SAMPLES))
    # The error recorded per draw is the error the chain reports for it, and
    # the burn-in's errors are not recorded: `HmcChain.energy_error` drops
    # them too.
    assert [value for _, value in run.series("energy_error")] == [
        float(error) for error in chain.energy_error
    ]
    assert run.last("state_bytes") == float(chain.draws.nbytes)
    assert run.last("peak_rss_bytes") > 0.0


@pytest.mark.smoke
def test_the_cost_record_carries_the_seconds_since_the_block_opened() -> None:
    # Issue #891: a fit's wall clock is read from the run, as its bytes are,
    # so a caller times a start and its fit by opening one block around both.
    # The clock is bracketed by the caller's own reading of the same counter.
    opened = time.perf_counter()
    with track() as tracked:
        _chain()
        closed = time.perf_counter()

    seconds = _memory(tracked.run).last("seconds")
    assert 0.0 < seconds <= closed - opened
    assert tracked.started >= opened


@pytest.mark.smoke
def test_every_entry_is_stamped_in_order_inside_the_block() -> None:
    # Issue #894: the per-iteration seconds `opt.starts` reads are the
    # store's stamps, one per entry and in the order the entries arrived,
    # bracketed by the caller's own reading of the same counter.
    with track() as tracked:
        _chain()
        closed = time.perf_counter()

    run = _memory(tracked.run)
    stamps = run.stamps("energy_error")
    assert len(stamps) == len(run.series("energy_error")) == N_SAMPLES
    assert stamps == sorted(stamps)
    assert tracked.started <= stamps[0]
    assert stamps[-1] <= closed


@pytest.mark.smoke
def test_every_record_carries_the_seconds_it_was_taken_at() -> None:
    # Issue #891: a curve against wall clock is read from the run, so every
    # sample carries its time, at its own step, non-decreasing, and one named
    # explicitly is kept rather than overwritten.
    with track() as tracked:
        _chain()
        tracked.record(99, seconds=-1.0)

    run = _memory(tracked.run)
    times = [value for _, value in run.series("seconds")]
    assert [step for step, _ in run.series("energy_error")] == [
        step for step, _ in run.series("seconds")[:N_SAMPLES]
    ]
    assert times[:-1] == sorted(times[:-1])
    assert times[-1] == -1.0


@pytest.mark.smoke
def test_the_annealer_records_one_energy_per_sweep_ending_at_the_result() -> None:
    with track() as tracked:
        annealed = _annealed()

    run = _memory(tracked.run)
    assert len(run.series("energy")) == annealed.n_sweeps == SWEEPS
    assert run.last("energy") == annealed.energy
    # The series is the best so far, so it never rises.
    energies = [value for _, value in run.series("energy")]
    assert energies == sorted(energies, reverse=True)
    assert run.last("temperature") == pytest.approx(0.1)
    assert run.last("state_bytes") == float(annealed.final.nbytes)


@pytest.mark.smoke
def test_the_tempered_run_records_the_swap_acceptance_it_returns() -> None:
    with track() as tracked:
        chains = _tempered()

    run = _memory(tracked.run)
    assert len(run.series("swap_acceptance")) == SWEEPS
    assert run.last("swap_acceptance") == float(np.mean(chains.swap_acceptance))
    assert run.last("state_bytes") == float(chains.states[0].nbytes)


@pytest.mark.smoke
def test_the_fit_records_the_value_and_the_norm_the_result_reports() -> None:
    objective = _objective()
    with track() as tracked:
        result = fit(objective, torch.tensor([2.0, -3.0], dtype=torch.float64))

    run = _memory(tracked.run)
    assert result.converged
    assert run.last("objective") == result.value
    assert run.last("relative_gradient_norm") == result.gradient_norm
    # One record per iteration, and one closing record carrying the result.
    assert len(run.series("objective")) == result.iterations + 1
    assert run.last("wall_s") > 0.0


@pytest.mark.smoke
def test_two_contexts_record_only_their_own_and_outside_records_nothing() -> None:
    # `map_tasks` on the serial pool runs the task in the calling context,
    # which is the context the run lives in.
    def task(item: int) -> int:
        current().record(item, item=float(item))
        return item

    with track(name="first") as first:
        map_tasks(task, [0, 1], workers=1, pool="serial", intra_op_threads=None)
    with track(name="second") as second:
        map_tasks(task, [2], workers=1, pool="serial", intra_op_threads=None)

    assert _memory(first.run).series("item") == [(0, 0.0), (1, 1.0)]
    assert _memory(second.run).series("item") == [(2, 2.0)]
    assert _memory(first.run).params["name"] == "first"
    assert _memory(second.run).params["name"] == "second"
    assert current() is NULL
    assert current().run is NULL_RUN


@pytest.mark.smoke
def test_a_series_under_a_context_is_not_the_series_of_the_same_name() -> None:
    # Aim keys a sequence by `(name, context)`, which is what a per-rung
    # series needs; `MemoryRun` keys it the same way, so a test that reads a
    # rung back reads the rung rather than a pooled series.
    with track() as tracked:
        tracked.record(0, energy=1.0, context={"rung": 0})
        tracked.record(0, energy=2.0, context={"rung": 1})
        tracked.record(0, energy=3.0)

    run = _memory(tracked.run)
    assert run.last("energy", {"rung": 0}) == 1.0
    assert run.last("energy", {"rung": 1}) == 2.0
    assert run.last("energy") == 3.0


@pytest.mark.smoke
def test_a_pool_worker_starts_outside_the_context_and_records_nothing() -> None:
    # A `ContextVar` is not carried into a pool worker, which is why
    # `track.py` says a per-task run opens its own context. Pinned rather than
    # left to be discovered from an empty series.
    with track() as tracked:
        map_tasks(
            _record_in_worker, [0, 1], workers=2, pool="threads", intra_op_threads=1
        )

    with pytest.raises(KeyError):
        _memory(tracked.run).series("item")


def _record_in_worker(item: int) -> int:
    current().record(item, item=float(item))
    return item


@pytest.mark.smoke
def test_a_metric_is_not_computed_on_the_null_run() -> None:
    # `record` returns on its first line when the run is `NULL_RUN`, so a
    # metrics set that would raise is never reached --- which is the claim
    # the overhead numbers rest on.
    with track(NULL_RUN, metrics=_Exploding()) as tracked:
        assert tracked.is_null
        _annealed()

    current().record(0, state=LABELLING, objective=1.0)
    assert current() is NULL
    assert NULL.metrics is None


@pytest.mark.smoke
def test_a_written_qa_figure_is_recorded_under_its_stem(tmp_path: Path) -> None:
    figure = Figure()
    figure.add_subplot().plot([0.0, 1.0], [1.0, 0.0])

    with track() as tracked:
        written = write_qa_figure(tmp_path, "track_probe", figure, "A probe.")

    recorded = _memory(tracked.run).last("track_probe")
    # `as_aim` hands the store the `Figure` where Aim is absent and an
    # `aim.Image` wrapping it where Aim is installed. Which one arrives is a
    # property of the environment, not of the call.
    assert recorded is figure or type(recorded).__name__ == "Image"
    assert written.figure_path.exists()


@pytest.mark.oracle
def test_potts_metrics_score_a_labelling_at_its_energy() -> None:
    graph = _graph()
    metrics = PottsMetrics(graph, FIELD)

    measured = metrics(LABELLING)

    assert metrics.names == ("state_energy",)
    # Against the hand count of agreeing edges, and against the function the
    # docstring cites.
    assert measured["state_energy"] == LABELLING_ENERGY
    assert measured["state_energy"] == energy(graph, FIELD, LABELLING)


@pytest.mark.oracle
@pytest.mark.patch
def test_hmm_metrics_report_the_forward_log_likelihood() -> None:
    observations = np.array([[0, 1, 0, 1], [1, 1, 0, 0]])
    objective = HmmObjective(observations, 2, 2)
    theta = objective.initial()
    named = objective.constrain(theta)
    truth = HmmParams(
        n_states=2,
        lengths=(4, 4),
        initial=np.exp(named["log_initial"].numpy()),
        transition=np.exp(named["log_transition"].numpy()),
        emissions=CategoricalEmission.from_log(named["log_emission"]),
        seed=0,
        tolerance=1e-9,
    )

    measured = HmmMetrics(objective)(theta)

    # The oracle sums all 2**4 paths per sequence term by term;
    # `forward_log_likelihood` sums by messages. The two orders agree to a
    # relative 1e-12 rather than bitwise.
    assert measured["log_likelihood"] == pytest.approx(
        sum(
            enumerate_hidden_paths(truth, sequence).log_evidence
            for sequence in observations
        ),
        rel=1e-12,
    )
    assert measured["log_likelihood"] == float(
        forward_log_likelihood(
            torch.as_tensor(observations),
            named["log_initial"],
            named["log_transition"],
            named["log_emission"],
        )
    )
    # The objective's own value with the sign the model states, and a
    # log-probability of eight symbols, so negative.
    assert measured["log_likelihood"] == -float(objective(theta))
    assert measured["log_likelihood"] < 0.0


@pytest.mark.oracle
def test_tree_metrics_score_the_truth_at_zero_split_distance() -> None:
    truth = balanced_four_taxa(0.1, 0.1, 0.1, 0.1)
    wrong = Node(
        "root",
        None,
        (
            Node("i1", ZONE_INTERNAL, (Node("A", 0.1), Node("C", 0.1))),
            Node("i2", ZONE_INTERNAL, (Node("B", 0.1), Node("D", 0.1))),
        ),
    )
    metrics = TreeMetrics(TREE_STATES, TREE_PI, TREE_ALIGNMENT, truth=truth)

    measured, other = metrics(truth), metrics(wrong)

    assert metrics.names == ("log_likelihood", "parsimony_score", "split_distance")
    assert measured["log_likelihood"] == pruning.log_likelihood(
        truth, TREE_STATES, TREE_PI, dict(TREE_ALIGNMENT)
    )
    assert measured["parsimony_score"] == float(
        parsimony.fitch_score(truth, TREE_ALIGNMENT, TREE_STATES)
    )
    # Zero on the tree that generated the alignment, and the Robinson-Foulds
    # distance on the one that did not: the two differ by one split each way.
    assert measured["split_distance"] == 0.0
    assert other["split_distance"] == float(robinson_foulds(truth, wrong)) == 2.0
    # The alignment supports `AB|CD`, so the truth wins on both criteria.
    assert other["log_likelihood"] < measured["log_likelihood"]
    assert other["parsimony_score"] > measured["parsimony_score"]


@pytest.mark.oracle
@pytest.mark.patch
def test_mixture_metrics_report_the_likelihood_and_the_clustering_cost() -> None:
    observations = np.array([0.1, 0.2, 3.0, 3.1])
    objective = GaussianMixtureObjective(observations, 2)
    theta = objective.initial()
    components = objective.components(theta)

    measured = MixtureMetrics(objective)(theta)

    # The oracle sums the joint over all 2**4 component assignments; the
    # metric takes the factorized form. They agree bitwise here.
    assert (
        measured["log_likelihood"]
        == enumerate_mixture_assignments(
            torch.exp(objective.constrain(theta)["log_weight"]).numpy(),
            components,
            observations,
        ).log_evidence
    )
    assert measured["log_likelihood"] == float(
        mixture_log_likelihood(
            objective.observations,
            objective.constrain(theta)["log_weight"],
            components,
        )
    )
    assert measured["log_likelihood"] == -float(objective(theta))
    assert measured["clustering_cost"] == clustering_cost(
        observations, components.mean.detach().numpy()
    )


@pytest.mark.oracle
def test_code_metrics_report_no_errors_on_the_sent_word() -> None:
    code = hamming_code(3)
    sent = np.zeros(code.n_bits, dtype=np.uint8)
    metrics = CodeMetrics(code, sent)
    flipped = sent.copy()
    flipped[0] = 1

    exact, one_error = metrics(sent), metrics(flipped)

    # The zero word is a codeword of every linear code: no errors, and no
    # unsatisfied check.
    assert exact == {"bit_error_rate": 0.0, "syndrome_weight": 0.0}
    # One flipped bit is one error in seven, and a Hamming code's columns are
    # the non-zero syndromes, so flipping bit 0 leaves a check unsatisfied.
    assert one_error["bit_error_rate"] == 1.0 / code.n_bits
    assert one_error["syndrome_weight"] == float(code.syndrome(flipped).sum()) > 0.0


@pytest.mark.oracle
@pytest.mark.parametrize(
    ("objective", "point"),
    [
        (Rosenbrock(dimension=2), None),
        (Rastrigin(dimension=2), None),
        (Himmelblau(), torch.tensor([3.0, 2.0], dtype=torch.float64)),
    ],
)
def test_test_function_metrics_report_zero_at_a_known_minimizer(
    objective: Rosenbrock | Rastrigin | Himmelblau, point: torch.Tensor | None
) -> None:
    minimizer = point
    if minimizer is None:
        assert not isinstance(objective, Himmelblau)
        minimizer = objective.minimizer()

    # Referenced through the module: pytest tries to collect a class whose
    # name starts with `Test`, and warns where it has a constructor.
    measured = testfunctions.TestFunctionMetrics(objective)(minimizer)

    # All three functions have value 0 at their global minima, which is what
    # makes the distance a fact rather than an estimate. `(3, 2)` is the
    # Himmelblau minimum the suite's fixture quotes first.
    assert measured["value"] == 0.0
    assert measured["distance_to_minimizer"] == 0.0


@pytest.mark.patch
@pytest.mark.smoke
def test_bound_potts_metrics_score_the_labelling_the_annealer_returns() -> None:
    graph = _graph()
    with track(metrics=PottsMetrics(graph, FIELD)) as tracked:
        annealed = _annealed()

    run = _memory(tracked.run)
    # The hook's own series ends at the field, and the metric at the energy
    # of the labelling the result carries: two names, one number, because the
    # annealer passes the best labelling it holds.
    assert run.last("energy") == annealed.energy
    assert run.last("state_energy") == energy(graph, FIELD, annealed.labelling)
    assert len(run.series("state_energy")) == SWEEPS


@pytest.mark.release
@pytest.mark.smoke
def test_an_aim_run_satisfies_the_run_protocol(tmp_path: Path) -> None:
    aim = pytest.importorskip("aim")

    run = aim.Run(repo=str(tmp_path), experiment="track_probe")
    try:
        # The Protocol is written from `aim.Run`'s own signatures, so this is
        # the assertion that there is no adapter to keep in step.
        assert isinstance(run, Run)
    finally:
        run.close()


@pytest.mark.release
@pytest.mark.smoke
def test_an_aim_run_reads_back_what_the_memory_run_recorded(tmp_path: Path) -> None:
    aim = pytest.importorskip("aim")

    with track(name="hmc") as expected:
        _chain()

    written = aim.Run(repo=str(tmp_path), experiment="track_probe")
    run_hash = str(written.hash)
    with track(written, name="hmc"):
        _chain()

    # Aim serves a read from its index, which a writing run does not update;
    # `aim storage --repo <path> reindex` is the CLI for it, and this is the
    # call that command makes.
    aim.Repo.from_path(str(tmp_path))._recreate_index()
    back = aim.Run(run_hash=run_hash, repo=str(tmp_path), read_only=True)
    sequences = {metric.name: metric for metric in back.metrics()}

    record = _memory(expected.run)
    assert back["name"] == "hmc"
    for name in ("acceptance_so_far", "energy_error", "force_evaluations"):
        recorded = [value for _, value in record.series(name)]
        sequence = sequences[name]
        # A multiset of values over the range of steps: aim 3.29's
        # ``values_list`` returns storage order, and reading a step raises.
        assert sorted(sequence.values.values_list()) == sorted(recorded)
        assert sequence.first_step() == 0
        assert sequence.last_step() == len(recorded) - 1


# --- The five loops #799 named, and the five metrics nothing recorded -------

#: A three-rung ladder from the uniform law, as the annealed estimators need.
BETAS = (0.0, 0.5, 1.0)


def _slice() -> slice.SliceChain:
    return slice.slice_sample(
        _objective(),
        torch.Generator().manual_seed(SEED),
        N_SAMPLES,
        width=1.0,
        max_steps_out=8,
        burn_in=BURN_IN,
    )


def _mala() -> langevin.LangevinChain:
    return langevin.mala(
        _objective(),
        torch.Generator().manual_seed(SEED),
        N_SAMPLES,
        step_size=STEP_SIZE,
        burn_in=BURN_IN,
    )


def _ais() -> annealed.LogPartition:
    return annealed.annealed_importance_sampling(
        _graph(), FIELD, BETAS, np.random.default_rng(SEED), 16, backend=Backend.PYTHON
    )


def _population() -> annealed.LogPartition:
    return annealed.population_annealing(
        _graph(), FIELD, BETAS, np.random.default_rng(SEED), 16, backend=Backend.PYTHON
    )


def _simulated_tempering() -> annealed.SimulatedTempered:
    return annealed.simulated_tempering(
        _graph(),
        FIELD,
        (0.5, 1.0, 2.0),
        np.zeros(3),
        np.random.default_rng(SEED),
        SWEEPS,
        backend=Backend.PYTHON,
    )


def _ensemble() -> tempered.TemperedEnsemble:
    return tempered.tempered_potts_pair(
        _graph(),
        FIELD,
        TEMPERATURES,
        np.random.default_rng(SEED),
        SWEEPS,
        backend=Backend.PYTHON,
    )


@pytest.mark.patch
@pytest.mark.smoke
def test_the_null_run_leaves_the_five_loops_bitwise_what_they_were() -> None:
    outside = (
        _slice(),
        _mala(),
        _ais(),
        _population(),
        _simulated_tempering(),
        _ensemble(),
    )
    with track(NULL_RUN):
        inside = (
            _slice(),
            _mala(),
            _ais(),
            _population(),
            _simulated_tempering(),
            _ensemble(),
        )
    assert torch.equal(inside[0].draws, outside[0].draws)
    assert inside[0].objective_evaluations == outside[0].objective_evaluations
    assert torch.equal(inside[1].draws, outside[1].draws)
    for one, other in ((inside[2], outside[2]), (inside[3], outside[3])):
        assert one.log_partition == other.log_partition
        assert one.stderr == other.stderr
        assert np.array_equal(one.rung_log_partition, other.rung_log_partition)
    assert np.array_equal(inside[4].states, outside[4].states)
    assert inside[4].acceptance == outside[4].acceptance
    assert np.array_equal(inside[5].walkers, outside[5].walkers)
    assert np.array_equal(inside[5].log_densities, outside[5].log_densities)


@pytest.mark.smoke
def test_the_slice_and_langevin_chains_record_the_unit_each_is_counted_in() -> None:
    with track() as tracked:
        sliced = _slice()
    run = _memory(tracked.run)
    assert len(run.series("objective_evaluations")) == N_SAMPLES
    assert run.last("objective_evaluations") == float(sliced.objective_evaluations)
    assert run.last("state_bytes") == float(sliced.draws.nbytes)
    with track() as tracked:
        chain = _mala()
    run = _memory(tracked.run)
    # `mala` runs `hmc.run_chain`, so the hook is the chain's and the unit is
    # gradients, one per proposal.
    assert run.last("acceptance_so_far") == chain.acceptance_rate
    assert run.last("force_evaluations") == float(chain.force_evaluations)
    assert len(run.series("energy_error")) == N_SAMPLES


@pytest.mark.smoke
def test_the_annealed_estimators_record_log_z_its_error_and_its_ess_per_rung() -> None:
    for estimator in (_ais, _population):
        with track() as tracked:
            estimate = estimator()
        run = _memory(tracked.run)
        assert [step for step, _ in run.series("log_z")] == [1, 2]
        assert run.last("log_z") == estimate.log_partition
        assert run.last("log_z_stderr") == estimate.stderr
        assert run.last("ess") == estimate.ess
        assert [value for _, value in run.series("log_z")] == list(
            estimate.rung_log_partition[1:]
        )
        assert run.last("state_bytes") > 0.0


@pytest.mark.smoke
def test_simulated_tempering_records_the_rung_the_acceptance_and_the_occupation() -> (
    None
):
    with track() as tracked:
        walker = _simulated_tempering()
    run = _memory(tracked.run)
    assert [value for _, value in run.series("rung")] == [
        float(r) for r in walker.rungs
    ]
    assert run.last("acceptance") == walker.acceptance
    assert run.last("sweeps_per_second") > 0.0
    for index, fraction in enumerate(walker.occupation):
        assert run.series("occupation", context={"rung": index}) == [
            (SWEEPS - 1, float(fraction))
        ]
    assert run.last("state_bytes") == float(walker.states.nbytes)


@pytest.mark.smoke
def test_the_ensemble_records_the_round_trips_its_walkers_carry() -> None:
    with track() as tracked:
        ensemble = _ensemble()
    run = _memory(tracked.run)
    assert run.last("round_trips") == float(
        tempered.round_trips(ensemble.walkers).sum()
    )
    assert run.last("up_fraction") == float(
        np.nanmean(tempered.up_fraction(ensemble.walkers))
    )
    assert len(run.series("sweeps_per_second")) == SWEEPS
    assert run.last("state_bytes") == float(
        ensemble.walkers.nbytes + ensemble.log_densities.nbytes
    )


@pytest.mark.smoke
def test_every_loop_records_what_it_cost_the_machine() -> None:
    # #799: `record_cost` was on three hooks of nine; the fit, the HMC annealer
    # and its tempering gain it, so every loop ends with its bytes beside its
    # last number.
    with track() as tracked:
        result = fit(_objective(), torch.tensor([2.0, -3.0], dtype=torch.float64))
    assert _memory(tracked.run).last("state_bytes") == float(result.theta.nbytes)
    with track() as tracked:
        annealed_hmc = hmc.anneal(
            _objective(),
            ExponentialTempSchedule(start=2.0, end=0.1, n_steps=SWEEPS),
            torch.Generator().manual_seed(SEED),
            step_size=STEP_SIZE,
            n_steps=3,
        )
    assert _memory(tracked.run).last("state_bytes") == float(annealed_hmc.theta.nbytes)
    with track() as tracked:
        tempered_hmc = hmc.parallel_tempering(
            _objective(),
            TEMPERATURES,
            torch.Generator().manual_seed(SEED),
            SWEEPS,
            step_size=STEP_SIZE,
            n_steps=3,
        )
    assert _memory(tracked.run).last("state_bytes") == float(
        tempered_hmc.positions.nbytes
    )
