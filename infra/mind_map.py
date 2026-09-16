"""A one-page mind map of the package: two concerns, the packages, and the modules.

Issue #664. `README.md` says where the documents live and `PROBLEMS.md` says
which problems are supported; nothing shows the *shape* of the package, and the
separation root `CLAUDE.md` insists on --- infrastructure against applications
--- is asserted in prose and nowhere drawn.

**A module's role is the roadmap claim it carries, not a restatement of its
docstring.** `STATUS.md` records what has landed against each milestone, so the
label comes from the join: the milestone whose section names the module. The
docstring supplies the wording where a reader needs more than a name, never the
attribution. `tests/regression/test_planning_documents_agree.py` already holds
`ROADMAP.md`, `STATUS.md` and `TICKETS.md` to the same milestone headings, so
that join has a guard rather than a convention.

**A module no `STATUS.md` section names is the interesting output**, not an
error: it is code in the tree that no roadmap milestone claims. Those are
listed rather than dropped, because finding them is half of what the map is
for.

Generated, never hand-drawn. A committed diagram of a tree is what `SEAMS.md`
was before issue #601 deleted it: a copy that goes stale in silence.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

#: The repository root, from `infra/`.
REPO_ROOT = Path(__file__).resolve().parents[1]

#: The package the map is of.
PACKAGE = REPO_ROOT / "python" / "snakes_and_ladders"

#: The packages whose modules are *applications*; everything else is
#: infrastructure. Root `CLAUDE.md` states the split and this is the whole of
#: it --- a list here rather than in the figure, so a module that moves between
#: them moves on the page with no edit to either.
APPLICATION = ("sim", "likelihood", "opt", "learn", "search")

#: ``## Milestone 1.1 — Simulation & Ground Truth Engine`` in `STATUS.md`.
MILESTONE = re.compile(r"^##\s+Milestone\s+([\d.]+)\s*[—-]\s*(.+?)\s*$", re.M)

#: ``- **Milestone 1.1: Simulation & Ground Truth Engine**`` in `ROADMAP.md`.
ROADMAP_MILESTONE = re.compile(r"^-\s+\*\*Milestone\s+([\d.]+):\s*(.+?)\*\*", re.M)

#: Any ``##`` heading, which is where a milestone's section stops.
SECTION = re.compile(r"^##\s+", re.M)

#: A backticked dotted name, as `STATUS.md` writes one.
CODE_SPAN = re.compile(r"`([A-Za-z_][\w.]*)`")

#: What a node's label may run to before it stops being readable on one page.
LABEL_WIDTH = 34

#: The page holds this many nodes. Over it, the guard fails rather than the
#: figure printing something unreadable (issue #664).
NODE_CAP = 200


@dataclass(frozen=True)
class Module:
    """One module, and where the documents place it."""

    #: Dotted name below the package, e.g. ``likelihood.pruning``.
    name: str
    #: ``sim``, ``likelihood``, ... or ``""`` for a top-level module.
    package: str
    #: ``application`` or ``infrastructure``.
    concern: str
    #: The milestones whose `STATUS.md` sections name it, sorted.
    milestones: tuple[str, ...]
    #: The docstring's first sentence, or ``""``.
    summary: str


def modules(package: Path = PACKAGE) -> tuple[Module, ...]:
    """Every module in the package, with its concern and its roadmap claims."""
    claims = status_claims()
    found = []
    for path in sorted(package.rglob("*.py")):
        if path.name == "__init__.py" or "__pycache__" in path.parts:
            continue
        name = ".".join(path.relative_to(package).with_suffix("").parts)
        head = name.split(".")[0] if "." in name else ""
        concern = "application" if head in APPLICATION else "infrastructure"
        found.append(
            Module(
                name=name,
                package=head,
                concern=concern,
                milestones=tuple(sorted(claims.get(name, ()))),
                summary=_summary(path),
            )
        )
    return tuple(found)


def _summary(path: Path) -> str:
    """The first sentence of a module's docstring, or ``""`` if it has none."""
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return ""
    text = ast.get_docstring(tree) or ""
    if not text:
        return ""
    sentence = text.split("\n\n", 1)[0].replace("\n", " ").strip()
    head = sentence.split(". ", 1)[0].rstrip(".")
    return " ".join(head.split())


def roadmap_milestones(root: Path = REPO_ROOT) -> dict[str, str]:
    """``number -> title``, from `ROADMAP.md`."""
    text = (root / "ROADMAP.md").read_text()
    return dict(ROADMAP_MILESTONE.findall(text))


def spellings(package: Path = PACKAGE) -> dict[str, str]:
    """``spelling -> module``, for every way a document names one.

    `STATUS.md` writes a module three ways, and a join reading only the first
    attributes a third of what it should: the dotted name
    (``likelihood.forward_backward``), the bare basename (``forward_backward``),
    and a symbol inside it (``device.available_device``), whose first component
    is the basename. A basename shared by two modules is **ambiguous and is not
    a spelling of either** --- attributing it to whichever was walked first
    would be a guess, and this map lists what it cannot attribute.

    A LaTeX label like ``app:pruning`` is not a candidate: `CODE_SPAN` admits
    only identifier characters and dots, so a colon or a hyphen excludes it.
    """
    names = [
        ".".join(path.relative_to(package).with_suffix("").parts)
        for path in sorted(package.rglob("*.py"))
        if path.name != "__init__.py" and "__pycache__" not in path.parts
    ]
    found: dict[str, str] = {name: name for name in names}
    seen: dict[str, list[str]] = {}
    for name in names:
        seen.setdefault(name.rsplit(".", 1)[-1], []).append(name)
    for base, owners in seen.items():
        if len(owners) == 1 and base not in found:
            found[base] = owners[0]
    return found


def status_claims(
    root: Path = REPO_ROOT, package: Path = PACKAGE
) -> dict[str, set[str]]:
    """``module -> milestones``, from which `STATUS.md` section names it.

    A section runs to the next milestone heading, so a module named anywhere
    inside one is claimed by it.
    """
    text = (root / "STATUS.md").read_text()
    known = spellings(package)
    # A milestone's section ends at the next `##` heading of **any** kind, not
    # at the next milestone. `STATUS.md` carries free-form sections after the
    # last milestone, and bounding on milestones alone swept every one of them
    # into Milestone 4.1 --- 40 modules attributed to a milestone that claims
    # none of them (issue #664).
    heads = [match.start() for match in SECTION.finditer(text)]
    claims: dict[str, set[str]] = {}
    for match in MILESTONE.finditer(text):
        number, start = match.group(1), match.start()
        later = [head for head in heads if head > start]
        end = later[0] if later else len(text)
        for span in CODE_SPAN.findall(text[start:end]):
            candidate = span.removeprefix("snakes_and_ladders.")
            # A symbol inside a module names the module: `device.available_device`.
            name = known.get(candidate) or known.get(candidate.rsplit(".", 1)[0])
            if name is not None:
                claims.setdefault(name, set()).add(number)
    return claims


#: How many of a collapsed package's modules the page names. The rest are a
#: count on the package's own node.
SAMPLE = 5

#: Packages collapsed to one node and a sample. `qa` is 29 modules of
#: infrastructure --- a fifth of the budget on the part a newcomer reads last.
COLLAPSED = ("qa",)


def sampled(carried: tuple[Module, ...], take: int = SAMPLE) -> tuple[Module, ...]:
    """The `take` modules carrying the most roadmap claims.

    An *importance* sample, not a random one: the weight is how much of the
    roadmap a module is named against, which `status_claims` already measures,
    and ties break by name. Deterministic by construction, so the page needs no
    seed and two runs cannot disagree.
    """
    ranked = sorted(carried, key=lambda one: (-len(one.milestones), one.name))
    return tuple(ranked[:take])


def _label(module: Module) -> str:
    """A node's text: the module, and the roadmap claim that is its role.

    Only an *application* module is marked when no milestone names it.
    Infrastructure is not roadmap work --- the build, the checks and the
    release process appear in no milestone by design --- so flagging it would
    report the separation root `CLAUDE.md` insists on as a defect.

    The milestone carries no font command: the node's own size is set by the
    document, and a `\footnotesize` nested inside it printed the tag larger
    than the name it qualifies.
    """
    leaf = module.name.rsplit(".", 1)[-1].replace("_", r"\_")
    if module.milestones:
        return rf"{leaf}~\textcolor{{claim}}{{{', '.join(module.milestones)}}}"
    if module.concern == "application":
        return rf"{leaf}~\textcolor{{unclaimed}}{{$\circ$}}"
    return leaf


#: Millimetres between a panel's columns, and between rows.
COLUMN, ROW = 30, 2.5

#: Where the second concern's panel starts. The two are laid side by side
#: rather than stacked: 127 rows in one column is 60 cm and two pages, and the
#: maintainer asked for one (issue #664).
PANEL = 120


def placed(
    found: tuple[Module, ...] | None = None,
) -> list[tuple[float, float, str, str, int]]:
    """``(x, y, kind, label, parent)`` in millimetres, laid out here.

    One panel per concern, side by side. Within a panel a leaf takes the next
    row and a package sits at the mean of its modules' rows, so the column is
    the depth and the row is the packing. Parentage is recorded as each node is
    made rather than inferred from geometry: a nearest-row guess reconnects the
    wrong package the moment two are adjacent.

    `forest` would pack this and was the first choice. It is not usable here:
    `forest.sty` ships in this TeX Live but `environ.sty`, `trimspaces.sty` and
    `elocalloc.sty` do not, so it cannot load without installing TeX packages,
    which this work is not permitted to do. TikZ's `graphdrawing` is the other
    automatic option and needs LuaLaTeX, where the build runs pdflatex.
    """
    found = modules() if found is None else found
    nodes: list[tuple[float, float, str, str, int]] = []
    for panel, concern in enumerate(("application", "infrastructure")):
        members = tuple(one for one in found if one.concern == concern)
        left = panel * PANEL
        concern_at = len(nodes)
        nodes.append((left, 0.0, "concern", concern, -1))
        row, package_rows = 1.4, []
        for package in sorted({one.package for one in members}):
            carried = tuple(one for one in members if one.package == package)
            shown = sampled(carried) if package in COLLAPSED else carried
            package_at = len(nodes)
            name = (package or "top level").replace("_", r"\_")
            head = (
                rf"{name}~{{\tiny {len(carried)} modules}}"
                if package in COLLAPSED
                else name
            )
            nodes.append((left + COLUMN, 0.0, "package", head, concern_at))
            leaves = []
            for module in shown:
                nodes.append(
                    (
                        left + 2 * COLUMN,
                        -row * ROW,
                        "module",
                        _label(module),
                        package_at,
                    )
                )
                leaves.append(-row * ROW)
                row += 1.0
            middle = sum(leaves) / len(leaves) if leaves else -row * ROW
            x, _, kind, label, parent = nodes[package_at]
            nodes[package_at] = (x, middle, kind, label, parent)
            package_rows.append(middle)
            row += 0.9
        centre = sum(package_rows) / len(package_rows) if package_rows else 0.0
        x, _, kind, label, parent = nodes[concern_at]
        nodes[concern_at] = (x, centre, kind, label, parent)
    return nodes


def tree(found: tuple[Module, ...] | None = None) -> str:
    """The TikZ body: one node per entry, one edge per parent."""
    nodes = placed(found)
    lines = [
        f"\\node[{kind}] (n{index}) at ({x:.2f}mm,{y:.2f}mm) {{{label}}};"
        for index, (x, y, kind, label, _) in enumerate(nodes)
    ]
    lines.extend(
        f"\\draw[edge] (n{parent}.east) -- ++(4mm,0) |- (n{index}.west);"
        for index, (_, _, _, _, parent) in enumerate(nodes)
        if parent >= 0
    )
    return "\n".join(lines)


def node_count(found: tuple[Module, ...] | None = None) -> int:
    """Nodes the page carries: root, concerns, packages, and modules shown."""
    found = modules() if found is None else found
    packages = {(one.concern, one.package) for one in found}
    shown = sum(
        len(sampled(tuple(one for one in found if one.package == package)))
        if package in COLLAPSED
        else sum(one.package == package for one in found)
        for _, package in packages
    )
    return 1 + 2 + len(packages) + shown


def write(destination: Path | None = None) -> Path:
    """Write the `forest` body, and return where it went."""
    target = destination or REPO_ROOT / "docs" / "tex" / "generated" / "mind_map.tex"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(tree() + "\n")
    return target


if __name__ == "__main__":
    written = write()
    print(f"{written}: {node_count()} nodes")
