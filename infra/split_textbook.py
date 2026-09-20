"""One PDF per textbook section, cut from the book's pages (issue #844).

The book is typeset once by `infra/build_documents.sh`; this reads the
``.toc`` that build wrote, takes each section's first page, and writes one
``pdfpages`` driver per section --- ``\\includepdf[pages=a-b]{../textbook.pdf}``
--- plus one for the front matter and contents, the pages before the first
section. The cuts are pages of the one book, so numbering, cross-references and
citations are what the book has, and the book is byte for byte what it was.
The source stays one file, which is what the document guards read.

Usage: ``infra/split_textbook.py [--toc docs/textbook.toc] [--out docs/tex/generated/sections]``
prints one driver path per line for the build script to typeset.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOC = REPO_ROOT / "docs" / "textbook.toc"
OUT = REPO_ROOT / "docs" / "tex" / "generated" / "sections"

#: A ``\contentsline`` for a section: its number, its title and its page.
SECTION = re.compile(
    r"\\contentsline \{section\}\{\\numberline \{(?P<number>[^}]*)\}(?P<title>.*)\}"
    r"\{(?P<page>\d+)\}\{[^}]*\}%?$"
)
#: What a title may keep in a file name.
UNSAFE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class Section:
    """One section of the book, and the page it starts on."""

    number: str
    title: str
    page: int

    @property
    def slug(self) -> str:
        """``NN-words`` for the file, from the number and the title's words.

        A numbered section is zero-padded to two digits so the files sort as
        the book reads; an appendix keeps its letter (``A-...``), which sorts
        after the digits.
        """
        words = UNSAFE.sub("-", _plain(self.title).lower()).strip("-")
        number = self.number.strip()
        label = f"{int(number):02d}" if number.isdigit() else number.lower()
        return f"{label}-{words}"


def _plain(title: str) -> str:
    """A TOC title with its macros removed: ``\\emph{x}`` is ``x``, ``$Z$`` is ``Z``."""
    text = re.sub(r"\\[a-zA-Z]+\s*", "", title)
    return text.replace("{", "").replace("}", "").replace("$", "").replace("\\", "")


def sections(toc: str) -> list[Section]:
    """Every section the TOC lists, in order, from its ``\\contentsline`` lines.

    Parts and subsections are not cut: a part is a page of its own inside the
    section before it, and a subsection is inside its section.
    """
    found = []
    for line in toc.splitlines():
        match = SECTION.match(line.strip())
        if match is None:
            continue
        found.append(Section(match["number"], match["title"], int(match["page"])))
    return found


def ranges(found: list[Section]) -> list[tuple[str, str]]:
    """``(slug, pages)`` per driver: the front matter, then each section to the next's page.

    The last section runs to the end of the book, which ``pdfpages`` writes
    as an open range; the front matter is everything before the first section.
    """
    if not found:
        return []
    cuts = [("00-front-matter", f"1-{found[0].page - 1}")] if found[0].page > 1 else []
    for section, following in zip(found, found[1:] + [None], strict=False):
        stop = "" if following is None else str(following.page - 1)
        start = section.page
        # A section that shares its first page with the next starts a range
        # that would run backwards; it is the page it shares.
        pages = f"{start}-{stop}" if stop == "" or int(stop) >= start else str(start)
        cuts.append((section.slug, pages))
    return cuts


def driver(pages: str, book: str = "../../../textbook.pdf") -> str:
    """The LaTeX that cuts ``pages`` out of the book at their own size.

    ``book`` is the path ``pdfpages`` resolves from the driver's own directory,
    ``docs/tex/generated/sections/``: three levels up to ``docs/textbook.pdf``.
    """
    return (
        "\\documentclass{article}\n"
        "\\usepackage{pdfpages}\n"
        "\\pagestyle{empty}\n"
        "\\begin{document}\n"
        f"\\includepdf[pages={{{pages}}},fitpaper=true]{{{book}}}\n"
        "\\end{document}\n"
    )


def write_drivers(toc: Path = TOC, out: Path = OUT) -> list[Path]:
    """Write one driver per range under ``out`` and return their paths, in book order."""
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for slug, pages in ranges(sections(toc.read_text())):
        path = out / f"{slug}.tex"
        path.write_text(driver(pages))
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--toc", type=Path, default=TOC)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    if not args.toc.exists():
        print(f"{args.toc} does not exist; build the textbook first", file=sys.stderr)
        return 1
    for path in write_drivers(args.toc, args.out):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
