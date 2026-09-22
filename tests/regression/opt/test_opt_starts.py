"""The seam of initialization against optimizer, `opt.starts` (issue #894).

What refereed what. The result's shape, the refusals and the curve's
bookkeeping are the seam's own and are `smoke`. The adapters are held to the
entry points they wrap as `patch`: a benchmark through the seam reproduces
the call it replaces. `describe` is held to `inspect.getdoc` of the object the
caller passed and `starts_latex` to `qa.figure.check_latex_safe`, as
`analytic`.
"""

from __future__ import annotations

import inspect
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from snakes_and_ladders.cost import Cost
from snakes_and_ladders.opt.budget import Budget
from snakes_and_ladders.opt.emission_mixture import expectation_maximization
from snakes_and_ladders.opt.fit import fit
from snakes_and_ladders.opt.hmm import HmmObjective, baum_welch_family
from snakes_and_ladders.opt.initialize import FromObjective, Perturbed, RandomRestart
from snakes_and_ladders.opt.mixture import GaussianMixtureObjective
from snakes_and_ladders.opt.starts import (
    StartsBenchmark,
    polish_by_baum_welch,
    polish_by_emission_em,
    polish_by_fit,
    refuse_start,
)
from snakes_and_ladders.opt.testfunctions import Himmelblau
from snakes_and_ladders.qa.figure import check_latex_safe
from snakes_and_ladders.qa.starts import GAP_FLOOR, starts_figure, starts_latex

#: L-BFGS iterations a Himmelblau start is polished at.
POLISH = Budget(Cost.ITERATIONS, 6)

#: Starts the seam may score per cell.
SEEDING = Budget(Cost.EVALUATIONS, 3)


def _benchmark(seeds: tuple[int, ...], workers: int = 1) -> StartsBenchmark:
    """Three starts of Himmelblau, whose four minima are known to be zero."""
    return StartsBenchmark(
        Himmelblau(),
        {
            "objective": FromObjective(),
            "perturbed": Perturbed(0.5),
            "restart": partial(RandomRestart, 3, 1.0),
        },
        polish_by_fit,
        seeding_budget=SEEDING,
        polish_budget=POLISH,
        seeds=seeds,
        workers=workers,
        reference=[0.0],
    )


@pytest.mark.smoke
def test_the_result_has_one_row_one_reading_and_one_curve_per_trial() -> None:
    result = _benchmark((0, 1)).run()

    assert result.names == ("objective", "perturbed", "restart")
    table = result.table()
    assert [row.start for row in table] == list(result.names)
    assert all(row.trials == 2 for row in table)
    assert all(row.spent <= POLISH.size for row in table)
    # A restart's three points share the six iterations, two each.
    assert all(trial.spent <= 6 for trial in result.trials("restart"))
    readings = result.readings(recovery=lambda _objective, theta: float(theta[0]))
    assert [reading.start for reading in readings] == list(result.names)
    assert all(reading.seeding_seconds >= 0.0 for reading in readings)
    assert all(reading.recovery is not None for reading in readings)
    curves = result.curves()
    for name in result.names:
        assert len(curves[name]) == 2
        for curve, trial in zip(curves[name], result.trials(name), strict=True):
            # The curve ends at the value the comparison scored, its seconds
            # never fall, and the gap is read against the supplied zero.
            assert curve.values[-1] == trial.value
            assert np.all(np.diff(curve.seconds) >= 0.0)
            assert np.array_equal(curve.gaps, curve.values)
            assert curve.values[curve.handover] >= trial.seeded_value
    restart = result.trials("restart")[0]
    assert restart.handover == 2


@pytest.mark.smoke
@pytest.mark.patch
def test_a_single_start_through_the_seam_is_the_fit_it_hands_to() -> None:
    # The polish phase of the curve is `opt.fit.fit`'s own record, bitwise:
    # the seam adds the seeding entry before it and the reported value after.
    objective = Himmelblau()
    result = StartsBenchmark(
        objective,
        {"objective": FromObjective()},
        polish_by_fit,
        seeding_budget=SEEDING,
        polish_budget=POLISH,
        seeds=(0,),
        workers=1,
    ).run()
    direct = fit(objective, objective.initial(), max_iterations=POLISH.size)
    trial = result.trials("objective")[0]
    assert trial.value == direct.value
    assert torch.equal(trial.theta, direct.theta)
    assert trial.termination == direct.termination
    assert trial.values[-1] == direct.value
    assert len(trial.values) == 1 + (direct.iterations + 1) + 1


@pytest.mark.smoke
@pytest.mark.backend
def test_a_pool_reproduces_the_serial_run_bitwise() -> None:
    # Each cell seeds its own generator, the factory draws from it, and a
    # pooled worker runs one torch thread: the values do not move with the
    # worker count.
    serial = _benchmark((0, 1, 2), workers=1).run()
    pooled = _benchmark((0, 1, 2), workers=2).run()
    assert np.array_equal(serial.comparison.best, pooled.comparison.best)
    for name in serial.names:
        assert [one.values for one in serial.trials(name)] == [
            one.values for one in pooled.trials(name)
        ]


@pytest.mark.smoke
def test_a_start_over_its_seeding_budget_is_refused_in_one_sentence() -> None:
    benchmark = StartsBenchmark(
        Himmelblau(),
        {"restart": partial(RandomRestart, 4, 1.0)},
        polish_by_fit,
        seeding_budget=SEEDING,
        polish_budget=POLISH,
        seeds=(0,),
        workers=1,
    )
    with pytest.raises(
        ValueError, match="offered 4 starts against a seeding budget of 3"
    ):
        benchmark.run()
    with pytest.raises(ValueError, match="refused on Himmelblau") as refused:
        refuse_start("chain", Himmelblau(), "no surrogate is declared")
    message = str(refused.value)
    assert (
        message
        == "the start 'chain' is refused on Himmelblau: no surrogate is declared"
    )
    assert message.count(". ") == 0
    with pytest.raises(ValueError, match="counts the starts the seam scores"):
        StartsBenchmark(
            Himmelblau(),
            {"objective": FromObjective()},
            polish_by_fit,
            seeding_budget=Budget(Cost.PASSES, 1),
            polish_budget=POLISH,
            seeds=(0,),
            workers=1,
        )


@pytest.mark.analytic
def test_describe_reads_the_docstring_and_the_fields_in_the_caller_s_order() -> None:
    result = _benchmark((0,)).run()
    described = result.describe()
    assert [one.start for one in described] == list(result.names)
    by_name = {one.start: one for one in described}
    for name, owner in (
        ("objective", FromObjective),
        ("perturbed", Perturbed),
        ("restart", RandomRestart),
    ):
        document = inspect.getdoc(owner)
        assert document is not None
        first = " ".join(document.split("\n\n", 1)[0].split())
        assert by_name[name].summary == first
    assert by_name["perturbed"].hyperparameters == (("magnitude", "0.5"),)
    # A factory's bound arguments are read in the class's own parameter order.
    assert by_name["restart"].hyperparameters == (("n_starts", "3"), ("scale", "1.0"))


@pytest.mark.analytic
def test_the_table_is_latex_safe_and_one_row_per_start() -> None:
    result = StartsBenchmark(
        Himmelblau(),
        {"from_objective": FromObjective(), "perturbed": Perturbed(0.5)},
        polish_by_fit,
        seeding_budget=SEEDING,
        polish_budget=POLISH,
        seeds=(0,),
        workers=1,
        reference=[0.0],
    ).run()
    body, caption = starts_latex(result)
    check_latex_safe(caption)
    lines = body.splitlines()
    assert lines[0] == r"\begin{tabular}{lcc}"
    assert lines[1:4] == [
        r"  \toprule",
        r"  start & gap (nats) & seconds \\",
        r"  \midrule",
    ]
    assert lines[-2:] == [r"  \bottomrule", r"\end{tabular}"]
    rows = lines[4:-2]
    assert len(rows) == 2
    # The name is escaped, and the gap is `problems_tables`' `:g`.
    assert rows[0].startswith(r"  from\_objective & ")
    gap = result.table()[0].gap
    assert rows[0].split(" & ")[1] == (
        str(int(gap)) if float(gap).is_integer() else f"{gap:g}"
    )


@pytest.mark.smoke
@pytest.mark.parametrize("n_seeds", [1, 3])
def test_the_standard_figure_builds_at_one_trial_and_at_three(n_seeds: int) -> None:
    result = _benchmark(tuple(range(n_seeds))).run()
    fig, caption = starts_figure(result, reference_label="the known minimum, zero")
    try:
        check_latex_safe(caption)
        assert ("One trial per start" in caption) == (n_seeds == 1)
        assert ("standard deviation" in caption) == (n_seeds > 1)
        left, right = fig.axes
        assert left.get_yscale() == "log"
        assert left.get_ylim()[0] <= GAP_FLOOR
        assert [label.get_text() for label in right.get_yticklabels()] == list(
            result.names
        )
        assert tuple(fig.get_size_inches()) == (10.0, 3.8)
    finally:
        plt.close(fig)


def _gaussian_mixture() -> GaussianMixtureObjective:
    """A two-component mixture on a seeded draw, small enough to fit in a second."""
    rng = np.random.default_rng(894)
    values = np.concatenate([rng.normal(-2.0, 1.0, 150), rng.normal(2.0, 1.0, 150)])
    return GaussianMixtureObjective(values, 2)


@pytest.mark.smoke
@pytest.mark.patch
def test_the_expectation_maximization_adapter_is_the_entry_point() -> None:
    objective = _gaussian_mixture()
    theta = objective.initial()
    polished = polish_by_emission_em(objective, theta, Budget(Cost.ITERATIONS, 25))
    direct = expectation_maximization(
        objective.observations,
        torch.exp(objective.constrain(theta)["log_weight"]),
        objective.components(theta),
        max_iterations=25,
    )
    assert polished.value == -direct.log_likelihood
    assert polished.termination == direct.termination
    # The point is EM's, through `theta_from` and back: the weights to the
    # rounding of one `log` and one `log_softmax`.
    assert torch.allclose(
        torch.exp(objective.constrain(polished.theta)["log_weight"]),
        direct.weights,
        rtol=1e-12,
        atol=0.0,
    )


@pytest.mark.smoke
@pytest.mark.patch
def test_the_baum_welch_adapter_is_the_entry_point() -> None:
    rng = np.random.default_rng(894)
    observations = rng.integers(0, 3, size=(4, 60))
    objective = HmmObjective(observations, 2, 3)
    theta = objective.initial()
    polished = polish_by_baum_welch(objective, theta, Budget(Cost.ITERATIONS, 30))
    named = objective.constrain(theta)
    direct = baum_welch_family(
        observations,
        named["log_initial"],
        named["log_transition"],
        objective.emissions(theta),
        max_iterations=30,
    )
    assert polished.value == -direct.log_likelihood
    assert polished.termination == direct.termination


@pytest.mark.smoke
def test_an_adapter_refuses_an_objective_it_cannot_read() -> None:
    with pytest.raises(ValueError, match="refused on Himmelblau"):
        polish_by_emission_em(Himmelblau(), Himmelblau().initial(), POLISH)
    with pytest.raises(ValueError, match="refused on Himmelblau"):
        polish_by_baum_welch(Himmelblau(), Himmelblau().initial(), POLISH)
