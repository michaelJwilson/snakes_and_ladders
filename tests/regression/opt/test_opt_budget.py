"""The budgeted comparison, checked on methods whose outcome is known exactly.

The utility is model-agnostic, so it is tested on methods that are arithmetic:
a run that returns a value drawn from its generator and spends what it was
given. What is asserted is the accounting -- the restart count derived from
the budget, the best over seeds, the reference, the refusals -- because the
accounting is what four hand-rolled studies each got slightly differently
(issue #281). The studies themselves are reproduced through it beside the
methods they compare.
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.opt.budget import (
    Budget,
    Comparison,
    Method,
    Outcome,
    OverspendError,
    compare,
    mcnemar,
    restarts,
)


def _draw(instance: float, budget: Budget, rng: np.random.Generator) -> Outcome:
    """A run whose value is uniform on ``[instance, instance + 1)`` and costs its budget."""
    return Outcome(instance + float(rng.random()), budget.size)


@pytest.mark.mathematical
def test_restarts_run_as_many_times_as_the_budget_allows_and_keep_the_minimum() -> None:
    budget = Budget("evaluations", 10)
    method = restarts(_draw, cost=3)

    outcome = method(0.0, budget, np.random.default_rng(1))

    draws = np.random.default_rng(1).random(3)
    assert outcome.spent == 9
    assert outcome.value == float(draws.min())


@pytest.mark.mathematical
def test_compare_scores_the_best_over_seeds_against_the_best_any_method_found() -> None:
    budget = Budget("evaluations", 4)
    instances = [0.0, 10.0]

    result = compare(
        {"single": _draw, "restarts": restarts(_draw, 1)},
        instances,
        budget,
        seeds=(0, 1),
        workers=1,
    )

    expected_single = np.array(
        [
            min(
                _draw(instance, budget, np.random.default_rng([seed, index])).value
                for seed in (0, 1)
            )
            for index, instance in enumerate(instances)
        ]
    )
    np.testing.assert_array_equal(result.best[0], expected_single)
    assert result.methods == ("single", "restarts")
    assert (result.best[1] <= result.best[0]).all()  # four draws against one
    np.testing.assert_array_equal(result.reference, result.best.min(axis=0))
    assert result.hits() == {"single": 0, "restarts": 2}
    assert result.spent.max() <= budget.size


@pytest.mark.mathematical
def test_a_known_optimum_is_the_reference_when_given() -> None:
    result = compare(
        {"single": _draw},
        [0.0, 1.0],
        Budget("evaluations", 1),
        seeds=(0,),
        workers=1,
        known=[0.0, 1.0],
    )

    np.testing.assert_array_equal(result.reference, [0.0, 1.0])
    assert result.hits() == {"single": 0}
    assert all(gap > 0 for gap in result.mean_gap().values())


@pytest.mark.structural
def test_the_table_names_the_unit_the_hits_and_the_spend() -> None:
    result = compare(
        {"single": _draw}, [0.0], Budget("sweeps", 5), seeds=(3,), workers=1
    )

    table = result.table()

    assert "| method | hits of 1 | mean gap | sweeps spent |" in table
    assert "| single | 1/1 | 0 | 5 of 5 |" in table
    assert isinstance(result, Comparison)


@pytest.mark.edge_case
def test_a_method_that_spends_past_its_budget_is_refused() -> None:
    def greedy(instance: float, budget: Budget, _rng: np.random.Generator) -> Outcome:
        return Outcome(instance, budget.size + 1)

    with pytest.raises(OverspendError, match="above the budget"):
        compare(
            {"greedy": greedy}, [0.0], Budget("evaluations", 3), seeds=(0,), workers=1
        )


@pytest.mark.edge_case
def test_a_restart_that_costs_more_than_the_whole_budget_is_refused() -> None:
    with pytest.raises(ValueError, match="above the budget"):
        restarts(_draw, cost=5)(0.0, Budget("evaluations", 4), np.random.default_rng(0))
    with pytest.raises(ValueError, match="at least one unit"):
        restarts(_draw, cost=0)


@pytest.mark.edge_case
@pytest.mark.parametrize(
    ("methods", "instances", "seeds", "known", "match"),
    [
        ({}, [0.0], (0,), None, "at least one method"),
        ({"single": _draw}, [], (0,), None, "at least one instance"),
        ({"single": _draw}, [0.0], (), None, "at least one seed"),
        ({"single": _draw}, [0.0, 1.0], (0,), [0.0], "known carries"),
    ],
)
def test_an_empty_or_inconsistent_comparison_is_refused(
    methods: dict[str, Method[float]],
    instances: list[float],
    seeds: tuple[int, ...],
    known: list[float] | None,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        compare(
            methods, instances, Budget("evaluations", 1), seeds, workers=1, known=known
        )


@pytest.mark.edge_case
def test_a_budget_without_a_unit_or_a_size_is_refused() -> None:
    with pytest.raises(ValueError, match="names its unit"):
        Budget("", 1)
    with pytest.raises(ValueError, match="at least one"):
        Budget("evaluations", 0)


@pytest.mark.mathematical
def test_mcnemar_is_the_two_sided_binomial_tail_on_the_discordant_pairs() -> None:
    # Ten discordant instances split 1 against 9: the tail is
    # (C(10,0) + C(10,1)) / 2**10 = 11 / 1024, doubled. Concordant instances
    # carry no evidence, so adding them changes nothing.
    first = np.array([True] * 1 + [False] * 9 + [True] * 5 + [False] * 5)
    second = np.array([False] * 1 + [True] * 9 + [True] * 5 + [False] * 5)

    assert mcnemar(first, second) == pytest.approx(2.0 * 11.0 / 1024.0, rel=1e-15)
    assert mcnemar(second, first) == pytest.approx(2.0 * 11.0 / 1024.0, rel=1e-15)
    assert mcnemar(first[10:], second[10:]) == 1.0  # no discordant pair
    # Five against zero: 2 / 2**5, the smallest p-value five starts can give.
    assert mcnemar(np.ones(5, dtype=bool), np.zeros(5, dtype=bool)) == 2.0 / 32.0
    # An even split is the null itself, and the doubled tail is capped at 1.
    assert mcnemar(np.array([True, False]), np.array([False, True])) == 1.0


@pytest.mark.edge_case
def test_mcnemar_refuses_hits_that_are_not_paired() -> None:
    with pytest.raises(ValueError, match="not paired"):
        mcnemar(np.ones(3, dtype=bool), np.ones(4, dtype=bool))


@pytest.mark.mathematical
def test_a_relative_tolerance_scales_with_the_reference_and_the_paired_p_reads_the_hits() -> (
    None
):
    # Three instances at references 1000, 1000 and 0. `first` misses the
    # first by 0.5, which an absolute 1e-9 refuses and a relative 1e-3
    # (margin 1.0) accepts; on the third the reference is 0 so the relative
    # margin is 0 and only an exact hit counts.
    def first(instance: float, _budget: Budget, _rng: np.random.Generator) -> Outcome:
        return Outcome(instance + (0.5 if instance == 1000.0 else 0.0), 1)

    def second(instance: float, _budget: Budget, _rng: np.random.Generator) -> Outcome:
        return Outcome(instance + 2.0, 1)

    result = compare(
        {"first": first, "second": second},
        [1000.0, 1000.0, 0.0],
        Budget("evaluations", 1),
        seeds=(0,),
        workers=1,
        known=[1000.0, 1000.0, 0.0],
    )

    assert result.hits() == {"first": 1, "second": 0}
    assert result.hits(1e-3, relative=True) == {"first": 3, "second": 0}
    assert result.reached(1e-3, relative=True)["first"].tolist() == [True] * 3
    assert result.paired_p("first", "second", 1e-3, relative=True) == 2.0 / 8.0
    assert result.paired_p("first", "second") == 1.0


@pytest.mark.structural
def test_four_workers_report_the_comparison_one_worker_reports() -> None:
    """The cells of a comparison are independent by construction (issue #344).

    Each seeds its own generator from ``[seed, index]``, so running them on a
    process pool changes when they run and nothing about what they draw; the
    two tables are equal bitwise, not within a tolerance. ``restarts`` is
    included because it has to cross the process boundary, which a closure
    could not.
    """
    methods = {"single": _draw, "restarts": restarts(_draw, 1)}
    instances = [0.0, 10.0, 20.0]
    budget = Budget("evaluations", 4)

    serial = compare(methods, instances, budget, seeds=(0, 1), workers=1)
    pooled = compare(methods, instances, budget, seeds=(0, 1), workers=4)

    np.testing.assert_array_equal(pooled.best, serial.best)
    np.testing.assert_array_equal(pooled.spent, serial.spent)
    np.testing.assert_array_equal(pooled.reference, serial.reference)
    assert pooled.methods == serial.methods
