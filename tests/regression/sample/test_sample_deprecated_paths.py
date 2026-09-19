"""Eleven old paths still import, warn once, and hand back the same objects (issue #777).

Issue #777 moved four modules out of ``opt`` and seven out of ``search`` into
``sample``, and left a shim at each old path for one release. What a shim owes
its caller is two things, and both are asserted here rather than read off the
source: the import emits one :class:`DeprecationWarning` naming the new path,
and every name it re-exports **is** the name the new module defines, so a
caller holding an old reference and one holding a new reference compare equal
by identity. The 95 names below are the contract; a shim that drops one
fails here rather than at a caller.
"""

from __future__ import annotations

import importlib
import warnings
from types import ModuleType

import pytest

#: Old path, new path, and the names the shim must re-export.
PAIRS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "snakes_and_ladders.opt.hmc",
        "snakes_and_ladders.sample.hmc",
        (
            "Adaptation",
            "Adapted",
            "Annealed",
            "HmcChain",
            "Integrator",
            "Kernel",
            "Tempered",
            "WithGaussianPrior",
            "anneal",
            "effective_sample_size",
            "hamiltonian",
            "leapfrog",
            "parallel_tempering",
            "sample",
            "yoshida",
        ),
    ),
    (
        "snakes_and_ladders.opt.langevin",
        "snakes_and_ladders.sample.langevin",
        (
            "LangevinChain",
            "mala",
        ),
    ),
    (
        "snakes_and_ladders.opt.slice",
        "snakes_and_ladders.sample.slice",
        (
            "SliceChain",
            "SliceDirection",
            "SliceUpdate",
            "slice_sample",
            "slice_update",
        ),
    ),
    (
        "snakes_and_ladders.opt.schedule",
        "snakes_and_ladders.sample.schedule",
        (
            "AdaptedLadder",
            "ConstantTempSchedule",
            "CosineTempSchedule",
            "ExponentialTempSchedule",
            "FeedbackLadder",
            "LinearTempSchedule",
            "TempSchedule",
            "adapt_ladder",
            "adapt_ladder_by_round_trips",
            "temperatures",
        ),
    ),
    (
        "snakes_and_ladders.search.potts_mcmc",
        "snakes_and_ladders.sample.potts_mcmc",
        (
            "AnnealedPotts",
            "ClusterCounter",
            "MoveKind",
            "PottsChain",
            "PottsMove",
            "Recolour",
            "TemperedChains",
            "adapt_ladder_potts",
            "anneal_potts",
            "autodiff_log_ratios",
            "energies",
            "houdayer_cluster",
            "niedermayer_threshold",
            "parallel_tempering",
            "sample_potts",
            "sample_potts_pair",
            "taylor_log_ratios",
            "tempered",
        ),
    ),
    (
        "snakes_and_ladders.search.gibbs",
        "snakes_and_ladders.sample.gibbs",
        (
            "Annealed",
            "AnnealedTopology",
            "GibbsChain",
            "GibbsMove",
            "anneal_factor_graph",
            "anneal_topology",
            "balanced_sweep",
            "cached_topology_score",
            "chain_block_sweep",
            "factor_autodiff_log_ratios",
            "factor_taylor_log_ratios",
            "gibbs_sweep",
            "sample_factor_graph",
            "topology_step",
        ),
    ),
    (
        "snakes_and_ladders.search.balanced",
        "snakes_and_ladders.sample.balanced",
        (
            "BalancingFunction",
            "draw_change",
            "log_balanced_weights",
            "log_metropolis_ratio",
            "log_normalizer",
            "log_ratios",
        ),
    ),
    (
        "snakes_and_ladders.search.potts_keyed",
        "snakes_and_ladders.sample.potts_keyed",
        (
            "NiedermayerMove",
            "SwendsenWangMove",
            "WolffMove",
            "cluster_moves",
            "monochrome_partition",
        ),
    ),
    (
        "snakes_and_ladders.search.tempered",
        "snakes_and_ladders.sample.tempered",
        (
            "S",
            "TemperedEnsemble",
            "adapt_ladder_round_trips",
            "round_trip_time",
            "round_trips",
            "tempered_factor_graph",
            "tempered_potts_pair",
            "tempered_topologies",
            "up_fraction",
        ),
    ),
    (
        "snakes_and_ladders.search.annealed",
        "snakes_and_ladders.sample.annealed",
        (
            "LogPartition",
            "Resampling",
            "SimulatedTempered",
            "annealed_importance_sampling",
            "geometric_betas",
            "population_annealing",
            "rung_weights",
            "simulated_tempering",
        ),
    ),
    (
        "snakes_and_ladders.search.statistics",
        "snakes_and_ladders.sample.statistics",
        (
            "chi_square_p_value",
            "integrated_autocorrelation_time",
            "sign_test_p_value",
        ),
    ),
)


def _loaded(name: str) -> ModuleType:
    """`name`, imported without recording whatever its first import emitted."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return importlib.import_module(name)


@pytest.mark.parametrize(
    ("old", "new"), [(old, new) for old, new, _ in PAIRS], ids=lambda value: value
)
@pytest.mark.smoke
@pytest.mark.infra
def test_the_old_path_warns_and_names_the_new_one(old: str, new: str) -> None:
    # `sys.modules` caches the first import and its warning with it, so one
    # module body is run here: `importlib.reload` runs it against the object
    # already loaded, which is what a first import in a fresh process does.
    module = _loaded(old)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(module)
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1
    message = str(deprecations[0].message)
    assert new in message
    assert old in message


@pytest.mark.parametrize(
    ("old", "new", "names"), PAIRS, ids=[old for old, _, _ in PAIRS]
)
@pytest.mark.smoke
@pytest.mark.infra
def test_the_old_path_re_exports_the_same_objects(
    old: str, new: str, names: tuple[str, ...]
) -> None:
    shim = _loaded(old)
    moved = importlib.import_module(new)
    assert set(shim.__all__) == set(names)
    for name in names:
        assert getattr(shim, name) is getattr(moved, name), name
