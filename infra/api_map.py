"""Typeset the package's Python surface: every module, class and public
function with the summary line from its own docstring (issue #576).

The descriptions already exist, so this writes no prose: it reads the first
paragraph of each docstring and lays it out. What it produces is
``docs/tex/generated/api_map.tex``, which ``docs/tex/api_map.tex`` inputs and
``infra/build_documents.sh`` builds into ``docs/api.pdf``. The generated file
is not committed and the PDF is, the arrangement ``infra/problems_tables.py``
already uses (issue #425).

**Read by ``ast``, never by importing.** ``infra/select_tests.py`` takes the
same route for the import graph and the reasons carry: no import side effects,
no compiled extension, and no module dropped from the map because its import
failed. A map that omits quietly is worse than no map, which is why
``tests/regression/docs/test_api_map.py`` counts the entries against a second
walk rather than checking that the generator ran.

Order is the package layout and not the alphabet: the root's modules, then
each subpackage, and within a module the order the source declares. The
structure is the content.

Two things the surface is reported against rather than trusted for. A public
top-level function with no summary line is a defect this document would
otherwise typeset as a blank, so ``--write`` fails on one; today there are
none. And a summary line over ``SUMMARY_WIDTH`` characters is evidence
against itself under root ``CLAUDE.md``'s writing style --- it is wrapped by
the layout and never truncated, and every one is listed on stderr so it can be
shortened where it is written.

Run::

    python infra/api_map.py --write   # regenerate
    python infra/api_map.py --check   # exit 1 if the tree's copy is stale
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "python" / "snakes_and_ladders"
GENERATED = REPO_ROOT / "docs" / "tex" / "generated" / "api_map.tex"

#: Characters a summary line may occupy before it is reported. Chosen against
#: the surface it describes rather than picked: the median is 68 and the
#: longest 166, so this separates a tail of twenty-odd from the body of the
#: distribution instead of flagging a third of it.
SUMMARY_WIDTH = 100

#: The Sphinx cross-reference roles the docstrings use. Each renders as the
#: code it names; a leading ``~`` means the last dotted component alone, as
#: Sphinx reads it.
_ROLES = "mod|class|func|meth|attr|data|obj|exc"

#: One span of inline code in a docstring: a cross-reference role, a double
#: backtick literal, or a single backtick literal. Everything outside is text.
_MARKUP = re.compile(
    rf":(?:{_ROLES}):`(?P<role>~?[^`]+)`"
    r"|``(?P<double>.+?)``"
    r"|(?<!`)`(?P<single>[^`]+)`(?!`)"
)

#: Characters LaTeX reads as syntax, and what each becomes in text mode.
_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "|": r"\textbar{}",
    "<": r"\textless{}",
    ">": r"\textgreater{}",
    "\u2013": "--",
    "\u2014": "---",
}


@dataclass(frozen=True)
class Entry:
    """One class, function or method the map carries.

    Parameters
    ----------
    kind : str
        ``"class"``, ``"function"`` or ``"method"``.
    name : str
        The name as it is written, a method qualified by its class.
    summary : str
        The first paragraph of its docstring, on one line; empty when it has
        no docstring.
    """

    kind: str
    name: str
    summary: str


@dataclass(frozen=True)
class Module:
    """One module, with the entries it declares in source order.

    Parameters
    ----------
    path : str
        Relative to ``python/snakes_and_ladders/`` and without the suffix, so
        ``sim/hmm`` and ``sim/__init__``.
    summary : str
        The module docstring's first paragraph, on one line.
    entries : tuple[Entry, ...]
        Its public classes, functions and methods, in declaration order.
    """

    path: str
    summary: str
    entries: tuple[Entry, ...]

    @property
    def package(self) -> str:
        """The directory it sits in, ``""`` for the package root."""
        return self.path.rpartition("/")[0]


def summary_of(node: ast.AST) -> str:
    """The first paragraph of ``node``'s docstring, whitespace collapsed.

    The paragraph and not the first physical line: two summaries in the
    package wrap onto a second line, and cutting them at the newline would
    print half a sentence as though it were the whole one.
    """
    if not isinstance(
        node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
    ):
        return ""
    docstring = ast.get_docstring(node)
    if docstring is None:
        return ""
    return " ".join(docstring.strip().split("\n\n")[0].split())


def is_overload(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Whether ``node`` is a ``typing.overload`` stub rather than a function.

    A stub carries no docstring by convention --- the implementation it
    precedes carries the summary --- so including one would report a missing
    summary that is correct as written.
    """
    return any(
        ast.unparse(decorator).split(".")[-1] == "overload"
        for decorator in node.decorator_list
    )


def entries_of(tree: ast.Module) -> tuple[Entry, ...]:
    """The public classes, functions and methods of one parsed module.

    Underscore-prefixed names are excluded, and so is every member of an
    underscore-prefixed class: a private class's methods are no more public
    than the class.
    """
    entries: list[Entry] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if node.name.startswith("_") or is_overload(node):
                continue
            entries.append(Entry("function", node.name, summary_of(node)))
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("_"):
                continue
            entries.append(Entry("class", node.name, summary_of(node)))
            for member in node.body:
                if not isinstance(member, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                if member.name.startswith("_") or is_overload(member):
                    continue
                entries.append(
                    Entry(
                        "method",
                        f"{node.name}.{member.name}",
                        summary_of(member),
                    )
                )
    return tuple(entries)


def module_files(directory: Path) -> list[Path]:
    """Every module under ``directory``, in package-layout order.

    A package's own ``__init__.py`` first, then its modules, then its
    subpackages --- the order a reader walks the tree in, which is what the
    map is ordered by.
    """
    files = sorted(
        path
        for path in directory.iterdir()
        if path.suffix == ".py" and path.name != "__init__.py"
    )
    initial = (
        [directory / "__init__.py"] if (directory / "__init__.py").is_file() else []
    )
    nested = [
        path
        for path in sorted(directory.iterdir())
        if path.is_dir() and path.name != "__pycache__"
    ]
    return initial + files + [file for child in nested for file in module_files(child)]


def modules(root: Path = PACKAGE_ROOT) -> tuple[Module, ...]:
    """Every module of the package, parsed, in package-layout order."""
    found: list[Module] = []
    for path in module_files(root):
        tree = ast.parse(path.read_text(), filename=str(path))
        found.append(
            Module(
                path=str(path.relative_to(root).with_suffix("")),
                summary=summary_of(tree),
                entries=entries_of(tree),
            )
        )
    return tuple(found)


def counts(found: tuple[Module, ...]) -> dict[str, int]:
    """How many entries of each kind the map carries."""
    tally = {"module": len(found), "class": 0, "function": 0, "method": 0}
    for module in found:
        for entry in module.entries:
            tally[entry.kind] += 1
    return tally


def without_summary(found: tuple[Module, ...]) -> list[str]:
    """Every public top-level function the map would typeset a blank for.

    The guard the ticket asks for: the count is zero today, and a function
    added without a docstring is what would change it.
    """
    return [
        f"{module.path}/{entry.name}"
        for module in found
        for entry in module.entries
        if entry.kind == "function" and not entry.summary
    ]


def overruns(found: tuple[Module, ...]) -> list[tuple[str, int]]:
    """Every summary over ``SUMMARY_WIDTH``, longest first, with its length."""
    over = [
        (f"{module.path}/{entry.name}", len(entry.summary))
        for module in found
        for entry in module.entries
        if len(entry.summary) > SUMMARY_WIDTH
    ]
    over += [
        (module.path, len(module.summary))
        for module in found
        if len(module.summary) > SUMMARY_WIDTH
    ]
    return sorted(over, key=lambda pair: (-pair[1], pair[0]))


def escape(text: str) -> str:
    """``text`` with every LaTeX-significant character neutralised."""
    return "".join(_ESCAPES.get(character, character) for character in text)


#: Inside ``\\texttt`` the five characters below are set from the typewriter
#: font itself, which carries them at their ASCII slots. ``\\textbar``,
#: ``\\textless``, ``\\textgreater`` and the brace commands ask for the OMS
#: symbol font in the typewriter family, which has no such shape, so LaTeX
#: substitutes ``cmsy`` and pdfTeX's font expansion refuses it at the shipout
#: of whichever page the line lands on (#770's build).
_CODE_ESCAPES = {
    **_ESCAPES,
    "{": r"\char`\{{}",
    "}": r"\char`\}{}",
    "|": r"\char`\|{}",
    "<": r"\char`\<{}",
    ">": r"\char`\>{}",
}


def escape_code(text: str) -> str:
    """``text`` as it is set inside ``\texttt``: as :func:`escape`, braces from the font."""
    return "".join(_CODE_ESCAPES.get(character, character) for character in text)


def tex_text(text: str) -> str:
    """One docstring summary as LaTeX: inline code set in ``\\texttt``.

    The docstrings mark code with double backticks and cross-reference it
    with Sphinx roles; both read as literal punctuation if passed through, so
    they are converted rather than escaped.
    """
    pieces: list[str] = []
    position = 0
    for match in _MARKUP.finditer(text):
        pieces.append(escape(text[position : match.start()]))
        role, double, single = match.group("role", "double", "single")
        token = role or double or single or ""
        if role is not None and role.startswith("~"):
            token = role.lstrip("~").rpartition(".")[2]
        pieces.append(rf"\texttt{{{escape_code(token)}}}")
        position = match.end()
    pieces.append(escape(text[position:]))
    return "".join(pieces)


def tex_name(name: str) -> str:
    """One entry or module name as LaTeX, breakable at its separators.

    A qualified method name reaches 48 characters, past the width of the
    column that holds it, and TeX will not hyphenate a typewriter word. The
    break opportunities are put where the name already reads as jointed.
    """
    escaped = escape(name)
    for separator in (r"\_", ".", "/"):
        escaped = escaped.replace(separator, separator + r"\allowbreak{}")
    return escaped


def _module_block(module: Module) -> list[str]:
    """One module: its heading, its summary, and a table of its entries."""
    lines = [
        rf"\subsection{{\texttt{{{tex_name(module.path)}}}}}",
        rf"\label{{sec:api:{module.path.replace('/', ':')}}}",
    ]
    if module.summary:
        lines.append(rf"\noindent {tex_text(module.summary)}")
    else:
        lines.append(r"\noindent \emph{No module docstring.}")
    if not module.entries:
        lines += ["", r"\noindent \emph{No public names.}", ""]
        return lines
    lines += [
        r"\begin{apitable}",
    ]
    for entry in module.entries:
        summary = tex_text(entry.summary) if entry.summary else r"\emph{No docstring.}"
        if entry.kind == "class":
            lines.append(rf"\apiclass{{{tex_name(entry.name)}}}{{{summary}}}")
        elif entry.kind == "method":
            lines.append(rf"\apimethod{{{tex_name(entry.name)}}}{{{summary}}}")
        else:
            lines.append(rf"\apifunction{{{tex_name(entry.name)}}}{{{summary}}}")
    lines += [r"\end{apitable}", ""]
    return lines


def _reading_section(found: tuple[Module, ...]) -> list[str]:
    """The note that says what the map covers and what it leaves out.

    Generated rather than written into the document, because every number in
    it is a count of the tree and a hand-written one is stale the next time a
    module lands.
    """
    tally = counts(found)
    over = overruns(found)
    blank = sum(
        1 for module in found for entry in module.entries if not entry.summary
    ) + sum(1 for module in found if not module.summary)
    return [
        r"\section{How to read this map}",
        r"\label{sec:api:reading}",
        r"\noindent",
        f"The package declares {tally['module']} modules, {tally['class']} public "
        f"classes, {tally['function']} public top-level functions and "
        f"{tally['method']} public methods: {sum(tally.values())} entries. Each is "
        "listed under the module that declares it, with the first paragraph of "
        "its own docstring. A path is relative to "
        r"\texttt{python/snakes\_and\_ladders/}, so the entry "
        r"\texttt{load\_hmm\_params} under \texttt{sim/hmm} is "
        r"\texttt{sim/hmm/load\_hmm\_params}; a method is written under its "
        "class, qualified by it.",
        "",
        r"\noindent",
        "The order is the package layout and not the alphabet --- the root's "
        r"modules, then each subpackage, and within a module the order the "
        "source declares --- because the structure is part of what is being "
        "shown.",
        "",
        r"\noindent",
        r"\textbf{What is not here.} Every underscore-prefixed name is "
        "excluded, and with an underscore-prefixed class go its methods: a "
        "private class's members are no more public than the class. So are "
        r"\texttt{typing.overload} stubs, which carry no docstring by "
        "convention and would be listed as missing one. Nothing else is "
        "filtered, and nothing is truncated.",
        "",
        r"\noindent",
        f"{blank} entries carry no docstring and are marked as such, none of "
        "them a public top-level function --- the generator fails rather than "
        f"typeset one. {len(over)} summaries exceed {SUMMARY_WIDTH} characters "
        "and are wrapped by the layout rather than cut; the generator lists "
        "them so they can be shortened where they are written.",
        "",
    ]


def render(found: tuple[Module, ...] | None = None) -> str:
    """The whole generated document body, as LaTeX."""
    found = modules() if found is None else found
    lines = [
        "% Generated from the package sources by infra/api_map.py. Do not",
        "% edit; regenerate with its --write flag.",
        *_reading_section(found),
    ]
    package = "\x00"
    for module in found:
        if module.package != package:
            package = module.package
            title = (
                rf"\texttt{{{tex_name(package)}/}}" if package else "The package root"
            )
            lines += [
                rf"\section{{{title}}}",
                rf"\label{{sec:api:package:{package.replace('/', ':') or 'root'}}}",
            ]
        lines += _module_block(module)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Regenerate or check the map. Returns 1 from ``--check`` when stale."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="regenerate the map")
    mode.add_argument("--check", action="store_true", help="exit 1 if stale")
    arguments = parser.parse_args(argv)
    found = modules()
    for path, length in overruns(found):
        print(f"over {SUMMARY_WIDTH} characters: {path} ({length})", file=sys.stderr)
    blank = without_summary(found)
    if blank:
        print(
            "public top-level functions with no summary line: " + ", ".join(blank),
            file=sys.stderr,
        )
        return 1
    text = render(found)
    if arguments.write:
        GENERATED.parent.mkdir(parents=True, exist_ok=True)
        GENERATED.write_text(text)
        print(f"wrote {GENERATED}")
        return 0
    if not GENERATED.is_file() or GENERATED.read_text() != text:
        print(
            f"{GENERATED} is stale; regenerate with: python infra/api_map.py --write",
            file=sys.stderr,
        )
        return 1
    print(f"{GENERATED} is current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
