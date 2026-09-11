"""The block-frequency bound, soundness first (issue #408).

An unsound bound silently returns a wrong tree and nothing else in the suite
would catch it: a search ranked by an interval that does not contain the exact
value discards the right candidate and reports a converged optimum. So the
containment check is the first test in this file and was written first --
before the bound existed, and failing.

Containment is checked wherever the exact value is computable: the `ci` and
`stress` tree fixtures, over a grid of block sizes and frequency cutoffs, and
on random alignments over several seeds. The random alignments matter more:
the bound must hold for an arbitrary column, not only one the generating
model is likely to produce.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
import torch
from snakes_and_ladders.bound import Bound, certify
from snakes_and_ladders.likelihood.blocks import (
    BlockFrequencyBound,
    Interval,
    block_frequency_interval,
    site_log_likelihood_extremes,
)
from snakes_and_ladders.likelihood.pruning_torch import (
    branch_lengths_from_tree,
    log_likelihood,
)
from snakes_and_ladders.search.infer import MoveSet, infer, score_topology
from snakes_and_ladders.search.topology import (
    enumerate_topologies,
    leaf_bipartitions,
)
from snakes_and_ladders.sim.params import SimulationParams
from snakes_and_ladders.sim.simulate import simulate_alignment
from snakes_and_ladders.sim.tree import preorder

from tests._fixtures import load_fixture

#: The fixtures with an exact value cheap enough to compute against every
#: cell of the grid below.
FIXTURES = (
    "tree_search/ci.yaml",
    "tree_jc/ci.yaml",
    "tree_search/stress.yaml",
    "tree_jc/stress.yaml",
)

#: Sites simulated per fixture. The claim is per-site, so a shorter
#: alignment checks it as well as a long one and leaves the grid affordable.
N_SITES = 600

#: The grid soundness is checked over: block size against frequency cutoff.
BLOCK_SIZES = (1, 2, 3, 5)
CUTOFFS = (1, 2, 4, 8, 32)


def _instance(
    name: str, n_sites: int = N_SITES
) -> tuple[SimulationParams, dict[str, np.ndarray], torch.Tensor]:
    params = load_fixture(name)
    dataset = simulate_alignment(
        params.tau, params.k, params.pi, np.random.default_rng(params.seed), n_sites
    )
    return params, dict(dataset.alignment), branch_lengths_from_tree(params.tau)


def _random_alignment(
    params: SimulationParams, seed: int, n_sites: int
) -> dict[str, np.ndarray]:
    """Uniform random columns: data the generating model did not produce."""
    rng = np.random.default_rng(seed)
    names = [node.name for node in preorder(params.tau) if node.is_leaf]
    return {
        name: rng.integers(0, params.k, size=n_sites, dtype=np.int64) for name in names
    }


@pytest.mark.mathematical
@pytest.mark.critical
def test_the_interval_contains_the_exact_log_likelihood() -> None:
    # Containment on every fixture, over the whole grid of block sizes and
    # cutoffs, and on random alignments over several seeds. The tolerance is
    # relative and one-sided: it admits the floating-point noise of two
    # summation orders, nothing more.
    for name in FIXTURES:
        params, alignment, lengths = _instance(name)
        pi = np.asarray(params.pi)
        cases = [alignment] + [
            _random_alignment(params, seed, N_SITES) for seed in range(4)
        ]
        for columns in cases:
            exact = float(log_likelihood(params.tau, params.k, pi, columns, lengths))
            for block_size in BLOCK_SIZES:
                for cutoff in CUTOFFS:
                    interval = block_frequency_interval(
                        params.tau,
                        params.k,
                        pi,
                        columns,
                        lengths,
                        block_size=block_size,
                        min_count=cutoff,
                    )
                    assert interval.contains(exact, tolerance=1e-9)


@pytest.mark.mathematical
def test_every_cutoff_of_one_evaluates_the_alignment_exactly() -> None:
    # A cutoff of one bounds nothing, so the interval collapses onto the
    # exact value at every block size: the partition is a partition of the
    # sites, and the exact half is the whole of it.
    for name in FIXTURES:
        params, alignment, lengths = _instance(name, 200)
        pi = np.asarray(params.pi)
        exact = float(log_likelihood(params.tau, params.k, pi, alignment, lengths))
        for block_size in BLOCK_SIZES:
            interval = block_frequency_interval(
                params.tau,
                params.k,
                pi,
                alignment,
                lengths,
                block_size=block_size,
                min_count=1,
            )
            assert interval.bounded_sites == 0
            assert interval.width == pytest.approx(0.0, abs=1e-9)
            assert float(interval.lower) == pytest.approx(exact, rel=1e-11)


@pytest.mark.mathematical
def test_the_site_extremes_bracket_every_column() -> None:
    # The tail bound rests on two numbers per tree, and they are checked
    # against every column a small alphabet admits: enumerate all k ** n
    # columns of the four-taxon fixture and none may fall outside.
    params, _, lengths = _instance("tree_jc/ci.yaml", 10)
    pi = np.asarray(params.pi)
    names = sorted(node.name for node in preorder(params.tau) if node.is_leaf)
    n_columns = params.k ** len(names)
    digits = np.arange(n_columns)
    every = {
        name: ((digits // params.k**index) % params.k).astype(np.int64)
        for index, name in enumerate(names)
    }
    lower, upper = site_log_likelihood_extremes(params.tau, params.k, pi, lengths)
    # One evaluation per column: the total over a single-column alignment is
    # that column's site log-likelihood.
    for column in range(n_columns):
        single = {name: every[name][column : column + 1] for name in names}
        value = float(log_likelihood(params.tau, params.k, pi, single, lengths))
        assert float(lower) <= value + 1e-12
        assert float(upper) >= value - 1e-12


@pytest.mark.structural
def test_the_containment_check_has_teeth() -> None:
    # The soundness test rests on Interval.contains, so an interval excluding
    # the value must be rejected; otherwise that test passes on an unsound
    # bound.
    exact = -1000.0
    assert Interval(
        lower=torch.tensor(-1100.0),
        upper=torch.tensor(-900.0),
        exact=torch.tensor(-1000.0),
        exact_sites=1,
        bounded_sites=0,
        exact_blocks=1,
        n_blocks=1,
        evaluated_columns=1,
    ).contains(exact)
    narrowed = Interval(
        lower=torch.tensor(-999.0),
        upper=torch.tensor(-900.0),
        exact=torch.tensor(-950.0),
        exact_sites=1,
        bounded_sites=0,
        exact_blocks=1,
        n_blocks=1,
        evaluated_columns=1,
    )
    assert not narrowed.contains(exact)
    assert not narrowed.contains(exact, tolerance=1e-6)


@pytest.mark.mathematical
def test_the_width_is_the_bounded_sites_times_the_per_site_range() -> None:
    # The interval's shape, not a number: the exact half cancels, so the width
    # is what the tail costs, which makes the width table below a statement
    # about the cutoff rather than about the tree.
    params, alignment, lengths = _instance("tree_search/ci.yaml", 400)
    pi = np.asarray(params.pi)
    lower, upper = site_log_likelihood_extremes(params.tau, params.k, pi, lengths)
    span = float(upper - lower)
    for block_size in BLOCK_SIZES:
        for cutoff in CUTOFFS:
            interval = block_frequency_interval(
                params.tau,
                params.k,
                pi,
                alignment,
                lengths,
                block_size=block_size,
                min_count=cutoff,
            )
            assert interval.exact_sites + interval.bounded_sites == 400
            assert interval.width == pytest.approx(
                interval.bounded_sites * span, rel=1e-9
            )


@pytest.mark.mathematical
def test_the_width_grows_with_the_cutoff_and_with_the_block_size() -> None:
    # Raising the cutoff moves blocks from the exact half to the tail and
    # never the other way, so the width is non-decreasing in it. The block
    # size is reported rather than asserted: a larger block repeats less
    # often, but by how much is the alignment's business.
    print("\ninterval width as a fraction of |log-likelihood|:")
    for name in ("tree_search/ci.yaml", "tree_jc/ci.yaml"):
        params, alignment, lengths = _instance(name, 2000)
        pi = np.asarray(params.pi)
        exact = abs(float(log_likelihood(params.tau, params.k, pi, alignment, lengths)))
        print(f"  {name} (|LL| = {exact:.0f}, 2000 sites)")
        for block_size in BLOCK_SIZES:
            widths, bounded, columns = [], [], []
            for cutoff in CUTOFFS:
                interval = block_frequency_interval(
                    params.tau,
                    params.k,
                    pi,
                    alignment,
                    lengths,
                    block_size=block_size,
                    min_count=cutoff,
                )
                widths.append(interval.width / exact)
                bounded.append(interval.bounded_sites)
                columns.append(interval.evaluated_columns)
            assert widths == sorted(widths)
            assert bounded == sorted(bounded)
            cells = " | ".join(
                f"c={cutoff}: {width:.3f} ({column} cols)"
                for cutoff, width, column in zip(CUTOFFS, widths, columns, strict=True)
            )
            print(f"    N={block_size}: {cells}")


@pytest.mark.oracle
def test_the_lower_end_is_a_bound_on_the_fitted_log_likelihood() -> None:
    # Certified in the sense bound.py defines, against a full fit of all 15
    # five-taxon topologies: the lower end is a value at feasible lengths, so
    # it cannot exceed the maximum over lengths, and one violation refuses the
    # certificate. The fits are computed once and looked up, since certify
    # calls the exact target per structure and six certificates would fit each
    # topology six times.
    params, alignment, _ = _instance("tree_search/ci.yaml", 400)
    pi = np.asarray(params.pi)
    topologies = list(enumerate_topologies(sorted(alignment)))
    fitted = {
        leaf_bipartitions(topology): score_topology(topology, alignment, params.k)
        for topology in topologies
    }

    def exact(topology: object, _: object) -> float:
        return fitted[leaf_bipartitions(topology)]  # type: ignore[arg-type]

    for block_size in (1, 2):
        for cutoff in (1, 4, 16):
            certificate = certify(
                BlockFrequencyBound(
                    params.k, pi, block_size=block_size, min_count=cutoff
                ),
                exact,
                topologies,
                alignment,
            )
            assert certificate.violations == 0
            assert certificate.n_structures == len(topologies)


@pytest.mark.oracle
def test_neither_end_ranks_and_the_frequent_half_does() -> None:
    # The tail term is (bounded sites) x (per-site extreme); the extreme
    # varies with the tree and exceeds the differences between neighbouring
    # topologies, so an ordering by either end is an ordering by the tail's
    # looseness. Asserting that it ranks would assert something false
    # (`likelihood/CLAUDE.md`); asserted instead is that the POINT claim,
    # which drops the tail, puts the fitted best first.
    params, alignment, _ = _instance("tree_search/ci.yaml", 2000)
    pi = np.asarray(params.pi)
    topologies = list(enumerate_topologies(sorted(alignment)))
    fitted = np.array([score_topology(t, alignment, params.k) for t in topologies])
    best = int(np.argmax(fitted))

    def order(claim: Bound, cutoff: int) -> np.ndarray:
        surrogate = BlockFrequencyBound(
            params.k, pi, block_size=1, min_count=cutoff, claim=claim
        )
        return np.array([float(surrogate(t, alignment)) for t in topologies])

    # At a cutoff of one nothing is bounded and every claim is the same
    # exact number, so all three orderings are the plug-in one.
    for claim in (Bound.LOWER, Bound.UPPER, Bound.POINT):
        assert int(np.argmax(order(claim, 1))) == best

    for cutoff in (2, 4, 8):
        assert int(np.argmax(order(Bound.POINT, cutoff))) == best
    # Reported, because it is what the ends are not for.
    ends = {
        claim: int(np.argmax(order(claim, 8))) for claim in (Bound.LOWER, Bound.UPPER)
    }
    print(f"\nfitted best {best}; ranked by the interval's ends at cutoff 8: {ends}")


def _ranked_against_exact(
    moves_and_seeds: Sequence[tuple[MoveSet, int]],
    fixture: str = "tree_search/ci.yaml",
    n_sites: int = 1200,
) -> list[tuple[MoveSet, int, int, int, int, int]]:
    """One row per start: the exact search's cost and the ranked search's.

    The search the ranking is for --- `infer`'s lazy seam, fitting one
    candidate per neighbourhood instead of all --- against the search that
    fits every candidate. Each row asserts the same topology and the same
    fitted log-likelihood, and records the fits and forward passes a
    budget-matched comparison counts (issue #289).

    The cutoff is a *count*, so what it means depends on the alignment's
    length: at 1200 sites a cutoff of 8 keeps 6 of 12 starts, at 2000 all 12.
    As a fraction of the sites retained, 4 at 1200 and 8 at 2000 are the same
    setting.
    """
    params, alignment, _ = _instance(fixture, n_sites)
    pi = np.asarray(params.pi)
    surrogate = BlockFrequencyBound(
        params.k, pi, block_size=1, min_count=4, claim=Bound.POINT
    )
    rows = []
    print(f"\n{fixture}, {n_sites} sites, cutoff 4:")
    for moves, seed in moves_and_seeds:
        full = infer(alignment, params.k, rng=np.random.default_rng(seed), moves=moves)
        ranked = infer(
            alignment,
            params.k,
            rng=np.random.default_rng(seed),
            moves=moves,
            lazy_top=1,
            surrogate=surrogate,
        )
        assert leaf_bipartitions(ranked.topology) == leaf_bipartitions(full.topology)
        assert ranked.log_likelihood == pytest.approx(full.log_likelihood, rel=1e-8)
        assert ranked.fits < full.fits
        assert ranked.likelihood_evaluations < full.likelihood_evaluations
        rows.append(
            (
                moves,
                seed,
                full.fits,
                ranked.fits,
                full.likelihood_evaluations,
                ranked.likelihood_evaluations,
            )
        )
    for moves, seed, fits, ranked_fits, passes, ranked_passes in rows:
        print(
            f"  {moves} seed {seed}: {ranked_fits} fits against {fits}, "
            f"{ranked_passes} passes against {passes}"
        )
    return rows


@pytest.mark.oracle
def test_a_ranked_search_reaches_what_the_exact_search_reaches() -> None:
    # One start under each move set, inside the per-pull-request tier. The
    # same claim over every start and at six taxa is the `release` test
    # below; this is the fast sibling DEV.md's duration rule asks a
    # re-tiered claim to keep.
    rows = _ranked_against_exact([(MoveSet.NNI, 0), (MoveSet.SPR, 0)])

    assert len(rows) == 2


@pytest.mark.oracle
@pytest.mark.release
def test_the_ranked_search_holds_over_starts_move_sets_and_taxa() -> None:
    # Six starts rather than two, and the six-taxon fixture as well as the
    # five: an agreement holding from one start is an agreement about that
    # start. Over the 10 s cap, so `release` per DEV.md's duration rule.
    rows = _ranked_against_exact(
        [(moves, seed) for moves in (MoveSet.NNI, MoveSet.SPR) for seed in range(3)]
    )
    rows += _ranked_against_exact(
        [(MoveSet.NNI, 0), (MoveSet.SPR, 0)], "tree_search/stress.yaml", 1500
    )

    assert len(rows) == 8


@pytest.mark.structural
def test_the_bound_is_a_surrogate_with_a_claim() -> None:
    params, alignment, _ = _instance("tree_search/ci.yaml", 200)
    pi = np.asarray(params.pi)
    claims = {
        claim: BlockFrequencyBound(params.k, pi, block_size=3, min_count=4, claim=claim)
        for claim in (Bound.LOWER, Bound.UPPER, Bound.POINT)
    }
    for claim, surrogate in claims.items():
        assert surrogate.kind is claim
    assert float(claims[Bound.LOWER](params.tau, alignment)) <= float(
        claims[Bound.UPPER](params.tau, alignment)
    )


@pytest.mark.edge_case
def test_a_malformed_partition_or_alignment_is_refused() -> None:
    params, alignment, lengths = _instance("tree_search/ci.yaml", 40)
    pi = np.asarray(params.pi)
    for block_size, min_count, message in (
        (0, 1, "block_size must be at least 1"),
        (1, 0, "min_count must be at least 1"),
    ):
        with pytest.raises(ValueError, match=message):
            block_frequency_interval(
                params.tau,
                params.k,
                pi,
                alignment,
                lengths,
                block_size=block_size,
                min_count=min_count,
            )
    with pytest.raises(ValueError, match="no taxa"):
        block_frequency_interval(
            params.tau, params.k, pi, {}, lengths, block_size=1, min_count=1
        )
    ragged = dict(alignment)
    first = next(iter(ragged))
    ragged[first] = ragged[first][:-1]
    with pytest.raises(ValueError, match="ragged"):
        block_frequency_interval(
            params.tau, params.k, pi, ragged, lengths, block_size=1, min_count=1
        )
    # A cutoff that empties the exact half leaves the point claim with a
    # zero to scale, which is above every log-likelihood and orders nothing;
    # it must refuse rather than rank by it. Block size 5 at 40 sites and a
    # cutoff of 2 is that case: no block of 5 columns repeats.
    empty = block_frequency_interval(
        params.tau, params.k, pi, alignment, lengths, block_size=5, min_count=2
    )
    assert empty.exact_sites == 0
    assert empty.contains(
        float(log_likelihood(params.tau, params.k, pi, alignment, lengths))
    )
    with pytest.raises(ValueError, match="no block of the .* reached the cutoff"):
        _ = empty.extrapolated
    with pytest.raises(ValueError, match="claim must be a Bound"):
        BlockFrequencyBound(params.k, pi, block_size=1, min_count=1, claim="middle")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="a topology and an alignment"):
        BlockFrequencyBound(params.k, pi, block_size=1, min_count=1)(3, alignment)
