"""The page templates under `docs/templates/` keep what their README states (issue #970).

`docs/CLAUDE.md`: a stated invariant with no test is a defect waiting. The
README names every field of the work-in-flight template and claims the page
is one file with one external resource and a complete dark palette; each claim
is read off the files here.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

import pytest

from tests._paths import REPO_ROOT

TEMPLATES = REPO_ROOT / "docs" / "templates"
WORK_IN_FLIGHT = TEMPLATES / "work_in_flight.html"
README = TEMPLATES / "README.md"

#: Elements with no closing tag.
VOID = frozenset({"link", "meta", "br", "img", "hr", "input", "source", "wbr"})


class _Balance(HTMLParser):
    """Every opened element closed, in order."""

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.errors: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag not in VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"</{tag}> closes {self.stack[-1:] or 'nothing'}")
            return
        self.stack.pop()


def _tokens(block: str) -> set[str]:
    return set(re.findall(r"(--[a-z-]+)\s*:", block))


@pytest.mark.infra
def test_every_field_the_readme_names_is_in_the_template() -> None:
    page = WORK_IN_FLIGHT.read_text()
    named = set(re.findall(r"`\{([A-Z_]+)\}`", README.read_text()))
    assert named, "the README names no field"
    missing = sorted(field for field in named if f"{{{field}}}" not in page)
    assert missing == []


@pytest.mark.infra
def test_the_template_is_one_balanced_file_with_one_external_host() -> None:
    page = WORK_IN_FLIGHT.read_text()
    parser = _Balance()
    parser.feed(page)
    assert parser.errors == []
    assert parser.stack == []
    hosts = set(re.findall(r'(?:href|src)="https?://([^/"]+)', page))
    assert hosts <= {
        "fonts.googleapis.com",
        "fonts.gstatic.com",
        "github.com",
    }, hosts
    assert "<script" not in page


@pytest.mark.infra
def test_every_colour_token_is_redefined_for_both_dark_forms() -> None:
    page = WORK_IN_FLIGHT.read_text()
    light = re.search(r"^:root \{(.*?)\}", page, re.MULTILINE | re.DOTALL)
    system = re.search(
        r':root:not\(\[data-theme="light"\]\) \{(.*?)\}', page, re.DOTALL
    )
    explicit = re.search(r':root\[data-theme="dark"\] \{(.*?)\}', page, re.DOTALL)
    assert light is not None
    assert system is not None
    assert explicit is not None
    tokens = _tokens(light.group(1))
    assert tokens == _tokens(system.group(1)) == _tokens(explicit.group(1))
    assert "color-scheme: dark" in system.group(1)
    assert "color-scheme: dark" in explicit.group(1)
