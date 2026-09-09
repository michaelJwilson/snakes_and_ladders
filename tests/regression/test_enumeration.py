"""The one weighted enumeration, against the eleven implementations it replaced.

`snakes_and_ladders.enumeration` gained a product table, a shift-and-normalize, a
per-site accumulation and an argmax under issue #387. Each replaced code that
was already committed and already pinned, so the acceptance criterion is not
that the seam is correct in the abstract but that it is *the same arithmetic*:
every reference below is the replaced expression copied verbatim from the
commit before the seam, and every assertion is bitwise or value-for-value
rather than within a tolerance. A tolerance would let a re-association through,
and a re-associated sum is exactly what moves an oracle's last digits under a
consolidation that claims to move nothing.

The sizes are the ones the ticket names: the 81 Potts configurations, the
`3**8` hidden paths, and the `2**16` assignments of the coupled and mixture
enumerations.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch
from snakes_and_ladders.enumeration import (
    MAX_ENUMERABLE_CONFIGURATIONS,
    accumulate,
    assignment_table,
    assignments,
    best_assignment,
    normalize,
)
from snakes_and_ladders.learn.hmm import StatePathLandscape, enumerate_paths
from snakes_and_ladders.learn.hmm import optimum as hmm_optimum
from snakes_and_ladders.learn.potts import (
    PottsLandscape,
    enumerate_configurations,
)
from snakes_and_ladders.learn.potts import optimum as potts_optimum

# The four shapes every reference below is checked at: the two the ticket
# names, and two that separate a transposed convention from a correct one.
SHAPES = [(3, 4), (3, 8), (2, 16), (4, 8)]


def _product_table(n_states: int, n_sites: int) -> np.ndarray:
    """`assignment_table`'s body before #387's vectorization."""
    return np.array(
        list(itertools.product(range(n_states), repeat=n_sites)), dtype=np.int64
    ).reshape(-1, n_sites)


@pytest.mark.oracle
@pytest.mark.parametrize(("n_states", "n_sites"), SHAPES)
def test_the_vectorized_table_is_the_itertools_product_bitwise(
    n_states: int, n_sites: int
) -> None:
    # Base-`n_states` digits of the assignment index against the materialized
    # product. Equality of *arrays* rather than of sets: the lexicographic
    # order is what makes an enumerated argmax reproducible, so an ordering
    # that differed would break the tie rule three callers depend on while
    # every marginal stayed right.
    realized = assignment_table(
        n_states, n_sites, what="table", limit=MAX_ENUMERABLE_CONFIGURATIONS
    )
    assert np.array_equal(realized, _product_table(n_states, n_sites))


@pytest.mark.oracle
@pytest.mark.parametrize(("n_states", "n_sites"), [(3, 4), (2, 8), (4, 5)])
def test_the_iterator_is_the_itertools_product_sequence(
    n_states: int, n_sites: int
) -> None:
    # `learn.potts.enumerate_configurations` and `learn.hmm.enumerate_paths`
    # both returned this generator directly and both promise a sequence, not
    # a set.
    realized = list(assignments(n_states, n_sites, what="assignments"))
    assert realized == list(itertools.product(range(n_states), repeat=n_sites))


@pytest.mark.oracle
@pytest.mark.parametrize(("n_states", "n_sites"), SHAPES)
def test_accumulate_is_bitwise_the_three_loops_it_replaced(
    n_states: int, n_sites: int
) -> None:
    # Three different open-coded loops summed into per-site bins: the
    # per-node `np.add.at` of `likelihood.potts`, the per-path
    # `posterior[arange, path] += weight` of `likelihood.hmm_paths`, and the
    # per-labelling form of `likelihood.spatio_sequential`. They visit the
    # summands in different *outer* orders; what has to hold is that each
    # (site, state) bin receives its terms in assignment order, because that
    # is what makes the sum reproduce to the last bit.
    table = assignment_table(n_states, n_sites, what="table")
    weight = np.asarray(np.random.default_rng(0).random(table.shape[0]))
    realized = accumulate(table, weight, n_states)

    per_site = np.zeros((n_sites, n_states))
    for site in range(n_sites):
        np.add.at(per_site[site], table[:, site], weight)

    per_row = np.zeros((n_sites, n_states))
    for row, value in zip(table, weight, strict=True):
        per_row[np.arange(n_sites), row] += value

    assert np.array_equal(realized, per_site)
    assert np.array_equal(realized, per_row)


@pytest.mark.oracle
@pytest.mark.parametrize(("n_states", "n_sites"), SHAPES)
def test_accumulate_agrees_bitwise_with_bincount(n_states: int, n_sites: int) -> None:
    # The form `likelihood.mixture_assignments` was written on (pull request
    # #420). Bitwise agreement is what lets that enumeration become a
    # consumer of this seam rather than a ninth implementation beside it,
    # without any of its committed responsibilities moving.
    table = assignment_table(n_states, n_sites, what="table")
    weight = np.asarray(np.random.default_rng(1).random(table.shape[0]))
    binned = np.stack(
        [
            np.bincount(table[:, site], weights=weight, minlength=n_states)
            for site in range(n_sites)
        ]
    )
    assert np.array_equal(accumulate(table, weight, n_states), binned)


@pytest.mark.oracle
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_normalize_is_bitwise_the_two_shifts_it_replaced(seed: int) -> None:
    # `likelihood.potts` divided by the sum and reported `log(total) + peak`;
    # `likelihood.hmm_paths` kept the unnormalized weights and reported
    # `shift + log(sum)`. Both forms are returned, and both must be the
    # expression they replaced rather than a rounding of the other.
    log_weight = np.random.default_rng(seed).normal(scale=40.0, size=6561)
    shifted, probability, log_total = normalize(log_weight)

    peak = log_weight.max()
    unnormalized = np.exp(log_weight - peak)
    total = unnormalized.sum()

    assert np.array_equal(shifted, unnormalized)
    assert np.array_equal(probability, unnormalized / total)
    assert log_total == float(np.log(total) + peak)
    assert log_total == float(log_weight.max()) + float(np.log(shifted.sum()))


def _hmm_landscape(seed: int) -> StatePathLandscape:
    rng = np.random.default_rng(seed)
    n_states, length, n_symbols = 3, 5, 3
    return StatePathLandscape(
        np.log(rng.dirichlet(np.ones(n_states))),
        np.log(rng.dirichlet(np.ones(n_states), size=n_states)),
        np.log(rng.dirichlet(np.ones(n_symbols), size=n_states)),
        rng.integers(0, n_symbols, size=length),
    )


def _potts_landscape(seed: int) -> PottsLandscape:
    rng = np.random.default_rng(seed)
    return PottsLandscape(float(rng.normal()), rng.normal(size=3), 4)


@pytest.mark.oracle
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_best_assignment_is_the_three_argmax_loops_it_replaced(seed: int) -> None:
    # `learn.potts.optimum`, `learn.hmm.optimum` and
    # `learn.relaxed.enumerate_optimum` each ran this loop. The strict `>` is
    # the whole content of the tie rule: with `>=` the answer would be the
    # lexicographically *last* maximum, which no committed number was taken
    # against and which nothing else in the suite would notice.
    landscapes: list[tuple[PottsLandscape | StatePathLandscape, int, int]] = [
        (_potts_landscape(seed), 3, 4),
        (_hmm_landscape(seed), 3, 5),
    ]
    for landscape, states, sites in landscapes:
        best, best_score = None, -np.inf
        for candidate in itertools.product(range(states), repeat=sites):
            value = landscape.energy(candidate)
            if value > best_score:
                best, best_score = candidate, value
        realized, realized_score = best_assignment(
            assignments(states, sites, what="candidates"), landscape.energy
        )
        assert realized == best
        assert realized_score == float(best_score)


@pytest.mark.oracle
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_the_two_optimum_adapters_return_what_their_loops_returned(seed: int) -> None:
    # The adapters end to end, not the helper: `optimum` in two modules,
    # against the loop each carried before #387, on the fixture shape the
    # committed baselines use.
    potts = _potts_landscape(seed)
    best, best_energy = None, -np.inf
    for candidate in itertools.product(
        range(potts.n_states), repeat=potts.chain_length
    ):
        energy = potts.energy(candidate)
        if energy > best_energy:
            best, best_energy = candidate, energy
    assert potts_optimum(potts) == (best, float(best_energy))

    hmm = _hmm_landscape(seed)
    best_path, best_value = None, -float("inf")
    for path in itertools.product(range(hmm.n_states), repeat=hmm.length):
        value = hmm.energy(path)
        if value > best_value:
            best_path, best_value = path, value
    assert hmm_optimum(hmm) == (best_path, best_value)


@pytest.mark.edge_case
def test_the_two_learn_enumerators_gained_the_cap_they_lacked() -> None:
    # Both returned an uncapped `itertools.product`, so an oversized call was
    # a machine that stopped responding rather than an error. The refusal is
    # the one #230 defined, in the units the call site names.
    with pytest.raises(ValueError, match="chain configurations"):
        list(enumerate_configurations(4, 20))
    with pytest.raises(ValueError, match="hidden paths"):
        list(enumerate_paths(4, 20))


@pytest.mark.edge_case
def test_the_refusal_precedes_the_first_assignment() -> None:
    # A caller that has already consumed half an enumeration has spent the
    # memory the refusal exists to save, so `assignments` must raise on the
    # call rather than on the first `next`.
    with pytest.raises(ValueError, match="too many"):
        assignments(4, 20, what="too many things")


@pytest.mark.edge_case
def test_a_caller_that_already_refused_may_opt_out() -> None:
    # `likelihood.brute_force` refuses the count once and then enumerates per
    # site; `likelihood.spatio_sequential` refuses on labellings times paths,
    # which dominates either factor. Repeating the check per site would only
    # restate a judgement under a less informative name.
    # 2**18 is past the 200,000 cap and 37 MB, which is a table this host
    # holds; 2**20 is 168 MB and would test the machine rather than the flag.
    assert assignment_table(2, 18, what="unchecked", limit=None).shape == (2**18, 18)
    assert next(assignments(4, 20, what="unchecked", limit=None)) == (0,) * 20


@pytest.mark.oracle
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_mixture_enumeration_is_bitwise_what_it_was_before_the_seam(
    seed: int,
) -> None:
    # The twelfth adapter, and the one that arrived while the seam was being
    # written (pull request #420). Its own tests hold it to 1.2e-16 of the
    # factorized evidence; this holds it to *bitwise* the expression it
    # carried, because a tolerance that loose would not see the seam
    # re-associating a sum of 65,536 terms.
    from snakes_and_ladders.emissions import GaussianEmission
    from snakes_and_ladders.likelihood.mixture_assignments import (
        enumerate_mixture_assignments,
    )

    rng = np.random.default_rng(seed)
    n_components, n_samples = 2, 12
    weights = rng.dirichlet(np.ones(n_components))
    components = GaussianEmission(
        torch.as_tensor(rng.normal(size=n_components)),
        torch.as_tensor(np.abs(rng.normal(size=n_components)) + 0.5),
        1e-6,
    )
    values = np.asarray(rng.normal(size=n_samples))

    log_weight = np.log(weights)
    scored = log_weight + components.log_density(
        torch.as_tensor(values, dtype=torch.float64)
    ).numpy().reshape(n_samples, n_components)
    place = n_components ** np.arange(n_samples - 1, -1, -1, dtype=np.int64)
    table = (
        np.arange(n_components**n_samples, dtype=np.int64)[:, None] // place
    ) % n_components
    log_joint = scored[np.arange(n_samples)[None, :], table].sum(axis=1)
    shift = float(log_joint.max())
    weight = np.exp(log_joint - shift)
    total = float(weight.sum())
    posterior = weight / total
    reference = np.stack(
        [
            np.bincount(table[:, site], weights=posterior, minlength=n_components)
            for site in range(n_samples)
        ]
    )

    realized = enumerate_mixture_assignments(weights, components, values)

    assert realized.log_evidence == shift + float(np.log(total))
    assert np.array_equal(realized.responsibilities, reference)
    assert np.array_equal(realized.assignment, table[int(log_joint.argmax())])
