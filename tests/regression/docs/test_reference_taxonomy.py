"""The reference routing table is kept in three places, and this keeps them one.

Root `CLAUDE.md` routes each concern to its sources; the textbook's Reference
Taxonomy appendix mirrors the grouping in `\\citet` form; and
`docs/tex/references.bib` carries the entries either resolves to. Before this
guard the table named three works the bibliography did not carry (Ramalho,
Gorelick & Ozsvald, Antão) and nothing said so (issue #360). An addition lands
in all three or in none, and that is now checked rather than asked for.

The matching rule, chosen so that no file has to carry another's keys:

* every citation key in the appendix is an entry in the bibliography;
* every item in the table -- an author string, an optional year, a
  parenthesized title or phrase -- names surnames that all appear in one
  bibliography entry's `author` field, of that year where a year is given;
* one such entry is cited in the appendix, so the table and the appendix
  name the same works; and
* every entry the appendix cites has its first author's surname in the table.

Surnames are compared after stripping accents and TeX accent commands, so
`M{\\'e}zard` and `Mézard` are one name.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
#: The reference routing table moved out of `CLAUDE.md` into its own
#: document, which is where the appendix is now checked against.
REFERENCES_MD = REPO_ROOT / "REFERENCES.md"
TEXTBOOK = REPO_ROOT / "docs" / "tex" / "textbook.tex"
BIBLIOGRAPHY = REPO_ROOT / "docs" / "tex" / "references.bib"

TABLE_HEADING = "## Documents & Reference Sources"
APPENDIX_HEADING = r"\section{Reference Taxonomy}"

#: One bibliography entry: its key and the text up to the next entry.
_ENTRY = re.compile(r"@\w+\{([^,]+),(.*?)(?=\n@|\Z)", re.DOTALL)
_FIELD = re.compile(r"^\s*(\w+)\s*=\s*\{(.*)\},?\s*$", re.MULTILINE)
_CITE = re.compile(r"\\cite[tp]\{([^}]+)\}")
_ROW = re.compile(r"^\*\s+\*\*[^*]+\*\*\s*(.*)$")
_GROUP_LABEL = re.compile(r"^(?:Reviews?|Papers):\s*")
_MID_ROW_LABEL = re.compile(r"\.\s*(?:Reviews?|Papers):\s*")
_YEAR = re.compile(r"\s+(\d{4})$")


def normalize(text: str) -> str:
    """Lower-case ASCII, with TeX accent commands and diacritics removed."""
    text = re.sub(r"\\[`'\"^~=.uvHtcdb]\s*\{?(\w)\}?", r"\1", text)
    text = text.replace("{", "").replace("}", "").replace("\\", "")
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


def bibliography() -> dict[str, dict[str, str]]:
    """Every entry in ``references.bib`` as ``key -> {field: value}``."""
    entries: dict[str, dict[str, str]] = {}
    for key, body in _ENTRY.findall(BIBLIOGRAPHY.read_text()):
        entries[key.strip()] = {
            name.lower(): value for name, value in _FIELD.findall(body)
        }
    return entries


def appendix_keys() -> set[str]:
    """Every key the Reference Taxonomy appendix cites."""
    text = TEXTBOOK.read_text()
    start = text.index(APPENDIX_HEADING)
    end = text.index(r"\end{document}", start)
    keys: set[str] = set()
    for group in _CITE.findall(text[start:end]):
        keys.update(k.strip() for k in group.split(","))
    return keys


def table_items() -> list[tuple[str, str]]:
    """Every item in the `REFERENCES.md` table, as ``(author string, year or '')``.

    An item is one semicolon-separated cell of a row, with the parenthesized
    title or phrase removed and any leading `Review:`/`Papers:` label dropped.
    """
    text = REFERENCES_MD.read_text()
    start = text.index(TABLE_HEADING)
    # The table was a section of `CLAUDE.md` and is now a document of its own,
    # so there need be no heading after it: the file's end bounds the table.
    following = text.find("\n## ", start + 1)
    end = len(text) if following == -1 else following
    items: list[tuple[str, str]] = []
    for line in text[start:end].splitlines():
        match = _ROW.match(line)
        if match is None:
            continue
        # A `Review:` or `Papers:` label opens a row or follows a full stop
        # mid-row; either way it separates items as a semicolon does.
        # A slash separates two works sharing one title, so it too.
        row = _MID_ROW_LABEL.sub(";", match.group(1)).replace(" / ", ";")
        for cell in row.split(";"):
            item = _GROUP_LABEL.sub("", cell.strip())
            authors = item.split(" (", 1)[0].strip()
            year_match = _YEAR.search(authors)
            year = year_match.group(1) if year_match else ""
            authors = _YEAR.sub("", authors)
            items.append((authors, year))
    return items


def _authors(fields: dict[str, str]) -> str:
    """The `author` field, or `editor` for an edited volume."""
    return fields.get("author") or fields.get("editor", "")


def surnames(authors: str) -> list[str]:
    """The surnames an author string names, `et al.` dropped."""
    parts = re.split(r"\s*(?:&|,)\s*", authors)
    return [normalize(p.replace("et al.", "").strip()) for p in parts if p.strip()]


def matching_entries(
    names: list[str], year: str, entries: dict[str, dict[str, str]]
) -> set[str]:
    """Keys whose `author` field carries every surname, and the year if given."""
    found: set[str] = set()
    for key, fields in entries.items():
        author = normalize(_authors(fields))
        if all(name in author for name in names) and (
            not year or fields.get("year") == year
        ):
            found.add(key)
    return found


@pytest.mark.critical
@pytest.mark.structural
def test_every_key_the_appendix_cites_is_in_the_bibliography() -> None:
    missing = sorted(appendix_keys() - set(bibliography()))

    assert missing == [], f"appendix cites keys the bibliography lacks: {missing}"


@pytest.mark.critical
@pytest.mark.structural
def test_every_table_item_names_a_bibliography_entry_the_appendix_cites() -> None:
    entries = bibliography()
    cited = appendix_keys()
    items = table_items()
    assert len(items) > 20, "the table parser found too few items to be reading it"

    unmatched = [
        f"{authors} {year}".strip()
        for authors, year in items
        if not matching_entries(surnames(authors), year, entries)
    ]
    assert unmatched == [], (
        f"table items with no bibliography entry naming every surname: {unmatched}"
    )

    uncited = [
        f"{authors} {year}".strip()
        for authors, year in items
        if not matching_entries(surnames(authors), year, entries) & cited
    ]
    assert uncited == [], (
        f"table items whose bibliography entry the appendix does not cite: {uncited}"
    )


@pytest.mark.critical
@pytest.mark.structural
def test_every_entry_the_appendix_cites_is_in_the_table() -> None:
    entries = bibliography()
    named = {name for authors, _ in table_items() for name in surnames(authors)}

    absent = sorted(
        key
        for key in appendix_keys()
        if normalize(_authors(entries[key]).split(" and ")[0].split(",")[0])
        not in named
    )
    assert absent == [], (
        f"appendix cites works the CLAUDE.md table does not name: {absent}"
    )


@pytest.mark.structural
def test_the_parsers_read_the_shapes_they_claim_to() -> None:
    # The guard exercised on the shapes it has to read, so a table row it
    # silently skipped would fail here rather than pass the checks above.
    assert surnames("Goodfellow et al.") == ["goodfellow"]
    assert [a for a, _ in table_items()].count("Prince") == 1
    assert _authors({"editor": "Pachter, Lior"}) == "Pachter, Lior"
    assert surnames("Kool, van Hoof & Welling") == ["kool", "van hoof", "welling"]
    assert normalize("M{\\'e}zard") == normalize("Mézard") == "mezard"
    assert normalize('Schollw{\\"o}ck') == "schollwock"
    assert _YEAR.search("Machta 2010") is not None
    assert _YEAR.search("Nielsen & Chuang") is None
    assert _ROW.match("*   **Row:** A (*T*); B 2001 (why)") is not None
