"""The mind map names every module once, and says what each is claimed against.

Issue #664. The map is generated, so what needs refereeing is the join it is
built on: that no module is missing, that no node names a module that is gone,
and that a module's milestone is one `ROADMAP.md` declares.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "infra"))

from mind_map import (  # noqa: E402
    APPLICATION,
    NODE_CAP,
    PACKAGE,
    modules,
    node_count,
    placed,
    roadmap_milestones,
    tree,
)


@pytest.mark.critical
@pytest.mark.structural
def test_every_module_is_walked_exactly_once() -> None:
    """The map's modules are the tree's modules, both ways."""
    walked = {
        ".".join(path.relative_to(PACKAGE).with_suffix("").parts)
        for path in PACKAGE.rglob("*.py")
        if path.name != "__init__.py" and "__pycache__" not in path.parts
    }
    named = [one.name for one in modules()]
    assert sorted(named) == sorted(walked)
    assert len(named) == len(set(named)), "a module is mapped twice"


@pytest.mark.critical
@pytest.mark.structural
def test_the_page_stays_under_its_node_cap() -> None:
    """Over the cap the guard fails rather than the page printing unreadably."""
    count = node_count()
    assert count <= NODE_CAP, (
        f"the map is {count} nodes against a {NODE_CAP} cap: collapse a package "
        "to a counted node and an importance sample, as `qa` is"
    )


@pytest.mark.critical
@pytest.mark.structural
def test_the_concern_split_is_the_rule_root_claude_md_states() -> None:
    """A module under an application package is an application, and no other."""
    for one in modules():
        expected = "application" if one.package in APPLICATION else "infrastructure"
        assert one.concern == expected, f"{one.name} is filed as {one.concern}"


@pytest.mark.critical
@pytest.mark.structural
def test_every_application_module_is_claimed_by_a_milestone() -> None:
    """`STATUS.md` records what every application module is for.

    This is the guard the map was built to make writable. 35 application
    modules were claimed by no milestone when the join was first run --- 28 of
    them named nowhere in `STATUS.md` at all, `sim.jc` and `opt.objective`
    among them --- so the roadmap recorded no purpose for a third of the
    application code (issue #664).
    """
    unclaimed = [
        one.name
        for one in modules()
        if one.concern == "application" and not one.milestones
    ]
    assert not unclaimed, (
        f"{len(unclaimed)} application modules are named under no `STATUS.md` "
        f"milestone: {unclaimed[:10]}. Name the module where its milestone "
        "records the work, rather than widening this guard."
    )


@pytest.mark.critical
@pytest.mark.structural
def test_every_claimed_milestone_is_one_the_roadmap_declares() -> None:
    """A label cites a milestone that exists, so the two documents cannot drift."""
    declared = set(roadmap_milestones())
    assert declared, "no milestones parsed from ROADMAP.md"
    cited = {number for one in modules() for number in one.milestones}
    assert cited <= declared, (
        f"claimed but not on the roadmap: {sorted(cited - declared)}"
    )


@pytest.mark.structural
def test_the_layout_is_a_tree_and_is_deterministic() -> None:
    """Every node but the two concerns has one parent, and two runs agree."""
    nodes = placed()
    roots = [index for index, (*_, parent) in enumerate(nodes) if parent < 0]
    assert len(roots) == 2, "one root per concern panel"
    for index, (*_, parent) in enumerate(nodes):
        assert parent < index, "a node precedes its parent"
    assert tree() == tree(), "two renderings disagree"
