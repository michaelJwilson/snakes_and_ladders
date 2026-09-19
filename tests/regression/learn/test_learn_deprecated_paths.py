"""Three old ``search`` paths still import, warn once, and hand back the same objects (issue #779).

Issue #779 moved ``search/rl.py``, ``search/gym.py`` and ``search/surrogate.py``
to ``learn/tree.py``, ``learn/gym.py`` and ``learn/ranking.py``, and left a
shim at each old path for one release. What a shim owes its caller is two
things, and both are asserted here rather than read off the source: the import
emits one :class:`DeprecationWarning` naming the new path, and every name it
re-exports **is** the name the new module defines, so a caller holding an old
reference and a caller holding a new one compare equal by identity.

``gymnasium`` is the ``frameworks`` extra, so the ``gym`` pair skips without it;
the other two are core.
"""

from __future__ import annotations

import importlib
import warnings
from types import ModuleType

import pytest

#: Old path, new path, and the names the shim must re-export.
PAIRS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "snakes_and_ladders.search.rl",
        "snakes_and_ladders.learn.tree",
        (
            "FeatureSet",
            "RewardModel",
            "TreeEnvironment",
            "exchanged_subtrees",
            "standardize",
            "with_uniform_branch_lengths",
        ),
    ),
    (
        "snakes_and_ladders.search.surrogate",
        "snakes_and_ladders.learn.ranking",
        (
            "LatticeTarget",
            "LearnedTreeSurrogate",
            "TreeTarget",
            "enumerated_log_partition_target",
            "fixed_length_target",
            "ground_state_offset",
            "ground_state_target",
            "lattice_examples",
            "lattice_instances",
            "maximized_target",
            "mean_field_offset",
            "plug_in_offset",
            "shuffle_children",
            "strip_log_partition_target",
            "tree_examples",
            "tree_node_tokens",
        ),
    ),
    (
        "snakes_and_ladders.search.gym",
        "snakes_and_ladders.learn.gym",
        ("GymnasiumEnvironment", "Observation"),
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
    if old.endswith(".gym"):
        pytest.importorskip("gymnasium")
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
    if old.endswith(".gym"):
        pytest.importorskip("gymnasium")
    shim = _loaded(old)
    moved = importlib.import_module(new)
    assert set(shim.__all__) == set(names)
    for name in names:
        assert getattr(shim, name) is getattr(moved, name), name
