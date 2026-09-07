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
        known=[0.0, 1.0],
    )

    np.testing.assert_array_equal(result.reference, [0.0, 1.0])
    assert result.hits() == {"single": 0}
    assert all(gap > 0 for gap in result.mean_gap().values())


@pytest.mark.structural
def test_the_table_names_the_unit_the_hits_and_the_spend() -> None:
    result = compare({"single": _draw}, [0.0], Budget("sweeps", 5), seeds=(3,))

    table = result.table()

    assert "| method | hits of 1 | mean gap | sweeps spent |" in table
    assert "| single | 1/1 | 0 | 5 of 5 |" in table
    assert isinstance(result, Comparison)


@pytest.mark.edge_case
def test_a_method_that_spends_past_its_budget_is_refused() -> None:
    def greedy(instance: float, budget: Budget, _rng: np.random.Generator) -> Outcome:
        return Outcome(instance, budget.size + 1)

    with pytest.raises(OverspendError, match="above the budget"):
        compare({"greedy": greedy}, [0.0], Budget("evaluations", 3), seeds=(0,))


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
        compare(methods, instances, Budget("evaluations", 1), seeds, known=known)


@pytest.mark.edge_case
def test_a_budget_without_a_unit_or_a_size_is_refused() -> None:
    with pytest.raises(ValueError, match="names its unit"):
        Budget("", 1)
    with pytest.raises(ValueError, match="at least one"):
        Budget("evaluations", 0)
