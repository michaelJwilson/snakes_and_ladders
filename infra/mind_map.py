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
`ROADMAP.md` and `STATUS.md` to the same milestone headings, so
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
import math
import re
from dataclasses import dataclass
from pathlib import Path

import tex

#: The repository root, from `infra/`.
from _paths import REPO_ROOT

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


def split_modules(package: Path = PACKAGE) -> dict[str, tuple[str, ...]]:
    """``spelling -> modules`` for a package that was one module, as each is written.

    A package whose ``__init__`` imports from its own submodules is a module
    that was split and re-exports what it held --- ``emissions``,
    ``sample.potts_mcmc`` and ``opt.hmm`` since issue #1010 --- so `STATUS.md`
    naming it, or a symbol inside it, names each module it became. A directory
    package imports none of its submodules (its ``__init__`` defers to
    ``_submodules``) and is not one: naming ``sample`` claims nothing.
    """
    names = {
        ".".join(path.relative_to(package).with_suffix("").parts)
        for path in package.rglob("*.py")
        if path.name != "__init__.py" and "__pycache__" not in path.parts
    }
    # A basename a module also has is ambiguous, as `spellings` treats one.
    bases = {name.rsplit(".", 1)[-1] for name in names}
    found: dict[str, tuple[str, ...]] = {}
    for init in sorted(package.rglob("__init__.py")):
        if init.parent == package or "__pycache__" in init.parts:
            continue
        dotted = ".".join(init.parent.relative_to(package).parts)
        parts = {
            node.module.removeprefix(f"{package.name}.")
            for node in ast.parse(init.read_text()).body
            if isinstance(node, ast.ImportFrom) and node.module
        }
        modules = tuple(
            sorted(
                part
                for part in parts
                if part in names and part.startswith(f"{dotted}.")
            )
        )
        if modules:
            found[dotted] = modules
            if dotted.rsplit(".", 1)[-1] not in bases:
                found.setdefault(dotted.rsplit(".", 1)[-1], modules)
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
    split = split_modules(package)
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
            for one in (
                (name,)
                if name is not None
                else split.get(candidate) or split.get(candidate.rsplit(".", 1)[0], ())
            ):
                claims.setdefault(one, set()).add(number)
    return claims


#: How many of a collapsed package's modules the page names. The rest are a
#: count on the package's own node.
SAMPLE = 5

#: Packages collapsed to one node and a sample. `qa` is 29 modules of
#: infrastructure --- a fifth of the budget on the part a newcomer reads last
#: --- and `validation` is 28 more, the external frameworks' adapters and
#: scripts, collapsed when #1008's modules took the page past its cap.
COLLAPSED = ("qa", "validation")


def sampled(carried: tuple[Module, ...], take: int = SAMPLE) -> tuple[Module, ...]:
    """The `take` modules carrying the most roadmap claims.

    An *importance* sample, not a random one: the weight is how much of the
    roadmap a module is named against, which `status_claims` already measures,
    and ties break by name. Deterministic by construction, so the page needs no
    seed and two runs cannot disagree.
    """
    ranked = sorted(carried, key=lambda one: (-len(one.milestones), one.name))
    return tuple(ranked[:take])


#: What a tooltip carries before it is cut, in characters. A PDF viewer wraps
#: the text itself, but an unbounded docstring paragraph fills the screen.
TIP_WIDTH = 160


def _tip(module: Module) -> str:
    r"""The module's docstring summary, as text safe for LaTeX *and* for PDF.

    The tooltip crosses two escapes, which is what makes it fiddly: LaTeX reads
    it as a macro argument first, and only then is it written into a PDF
    literal string. A first attempt escaped for PDF alone and LaTeX read the
    `\(` as math mode.

    So the text is reduced to what is safe in both. Parentheses become brackets,
    which removes the only PDF escape that was needed; the LaTeX specials
    ``\ # $ % & { } ^ ~`` are dropped, since none of them carries meaning a
    tooltip needs; and ``_`` is emitted as ``\string_``, because module names
    are full of it and it is subscript to LaTeX in text mode. Non-ASCII goes:
    the annotation declares no encoding, so a viewer would print mojibake.

    So this is a strip and not `tex.escape`: an escape puts the character on
    the page, and a tooltip crossing into a PDF literal cannot carry one.
    """
    text = module.summary or "no module docstring"
    claim = f" ({', '.join(module.milestones)})" if module.milestones else ""
    text = f"{module.name}{claim} -- {text}"
    text = text.encode("ascii", "ignore").decode("ascii")
    if len(text) > TIP_WIDTH:
        text = text[: TIP_WIDTH - 3].rstrip() + "..."
    text = text.replace("(", "[").replace(")", "]")
    for character in "\\#$%&{}^~":
        text = text.replace(character, "")
    return text.replace("_", r"\string_")


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
    leaf = tex.escape(module.name.rsplit(".", 1)[-1])
    if module.milestones:
        return rf"{leaf}~\textcolor{{claim}}{{{', '.join(module.milestones)}}}"
    if module.concern == "application":
        return rf"{leaf}~\textcolor{{unclaimed}}{{$\circ$}}"
    return leaf


#: Radii, in millimetres: concern ring, package ring, leaf ring.
CONCERN_R, PACKAGE_R, LEAF_R = 34.0, 72.0, 104.0

#: Degrees left clear between the two concerns' wedges, each side, so the
#: branches read as two rather than one circle.
GAP = 7.0


def _wedges(
    weights: dict[str, int], gap: float = GAP
) -> dict[str, tuple[float, float]]:
    """``key -> (start, end)`` degrees, each wedge proportional to its weight.

    Splitting the circle evenly would waste it: `application` draws 95 leaves
    against `infrastructure`'s 20, so equal halves give one branch four times
    the room per leaf. Proportional wedges give **every leaf the same angle**,
    which is the quantity that decides whether labels collide (issue #664).
    """
    total = sum(weights.values()) or 1
    free = 360.0 - gap * len(weights)
    spans: dict[str, tuple[float, float]] = {}
    cursor = 0.0
    for key, weight in weights.items():
        span = free * weight / total
        spans[key] = (cursor + gap / 2, cursor + gap / 2 + span)
        cursor += span + gap
    return spans


def _polar(radius: float, degrees: float) -> tuple[float, float]:
    """Millimetre ``(x, y)`` at a radius and a bearing, zero to the right."""
    angle = math.radians(degrees)
    return radius * math.cos(angle), radius * math.sin(angle)


def placed(
    found: tuple[Module, ...] | None = None,
) -> list[tuple[float, float, str, str, float, int]]:
    """``(x, y, style, label, rotation, parent)`` for every node, radially.

    A mind map, not an indented tree: the package at the centre, the two
    concerns around it, the packages beyond them, and the modules on the outer
    ring. Each branch takes a wedge proportional to the leaves it carries, so
    every leaf gets the same angle.

    **A leaf's label is rotated to run radially outward**, and that is what
    makes 115 of them fit. Horizontal labels collide by the ratio of their
    width to the arc between them --- 3.1 degrees at a 96 mm radius is 5.2 mm
    of arc against a ~15 mm name --- while a radial label is constrained by its
    *height*, about 2 mm. Past the top of the circle a label would read upside
    down, so it is turned through 180 degrees and anchored on its other end.

    The layout is computed here; nothing in the document places a node.
    """
    found = modules() if found is None else found
    drawn = {
        concern: {
            package: (
                sampled(tuple(one for one in found if one.package == package))
                if package in COLLAPSED
                else tuple(one for one in found if one.package == package)
            )
            for package in sorted(
                {one.package for one in found if one.concern == concern}
            )
        }
        for concern in ("application", "infrastructure")
    }
    weights = {
        concern: sum(len(shown) for shown in packages.values())
        for concern, packages in drawn.items()
    }
    nodes: list[tuple[float, float, str, str, float, int]] = []
    nodes.append((0.0, 0.0, "centre", r"snakes\_and\_ladders", 0.0, -1))

    for concern, (low, high) in _wedges(weights).items():
        packages = drawn[concern]
        short = "app" if concern == "application" else "infra"
        concern_at = len(nodes)
        x, y = _polar(CONCERN_R, (low + high) / 2)
        nodes.append((x, y, f"concern, {short}", concern, 0.0, 0))
        inner = _wedges(
            {package: len(shown) for package, shown in packages.items()}, gap=2.0
        )
        for package, (start, stop) in inner.items():
            shown = packages[package]
            span = (stop - start) * (high - low) / 360.0
            first = low + (start / 360.0) * (high - low)
            package_at = len(nodes)
            key = (package or "toplevel").replace("_", "")
            name = tex.escape(package or "top level")
            carried = sum(
                one.package == package and one.concern == concern for one in found
            )
            head = rf"{name}~{{\tiny {carried}}}" if package in COLLAPSED else name
            middle = first + span / 2
            x, y = _polar(PACKAGE_R, middle)
            nodes.append((x, y, f"package, {key}", head, 0.0, concern_at))
            for index, module in enumerate(shown):
                bearing = first + span * (index + 0.5) / max(len(shown), 1)
                x, y = _polar(LEAF_R, bearing)
                flip = 90.0 < bearing % 360.0 < 270.0
                rotation = bearing + 180.0 if flip else bearing
                # The leaf takes its *package's* colour, not its concern's, so
                # a branch reads as one family at a glance.
                style = f"leaf, {key}leaf, {'flipped' if flip else 'plain'}"
                # The label is wrapped in `\tip`, which the document defines as
                # a `\pdfannot` over the text's own box: hovering a module in a
                # PDF reader shows its docstring (issue #664).
                text = rf"\tip{{{_label(module)}}}{{{_tip(module)}}}"
                nodes.append((x, y, style, text, rotation, package_at))
    return nodes


def tree(found: tuple[Module, ...] | None = None) -> str:
    """The TikZ body: one node per entry, one edge per parent."""
    nodes = placed(found)
    lines = [
        f"\\node[{style}, rotate={rotation:.2f}] (n{index}) "
        f"at ({x:.2f}mm,{y:.2f}mm) {{{label}}};"
        for index, (x, y, style, label, rotation, _) in enumerate(nodes)
    ]
    # An edge is drawn in its parent's colour and stops short of both nodes, so
    # it never runs under a bubble or into a label.
    lines.extend(
        f"\\draw[edge, {_hue(nodes[parent][2])}edge] (n{parent}) -- (n{index});"
        for index, (*_, parent) in enumerate(nodes)
        if parent >= 0
    )
    return "\n".join(lines)


def _hue(style: str) -> str:
    """The colour key a node's style carries, for an edge leaving it."""
    return style.rsplit(", ", 1)[-1].removesuffix("leaf") if ", " in style else style


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
