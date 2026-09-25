r"""Neutralising LaTeX's syntax, once, for everything that generates LaTeX.

Six escapers disagreed about which characters LaTeX reads as syntax: the API
map's table of fifteen, the problem tables' six, the mind map's ``_`` alone
and its separate strip of eight, and two in the QA package. A generator that
escapes five of the fifteen does not fail --- it writes a document that a
later run of ``pdflatex`` refuses, or worse sets a subscript where a name
had an underscore (issue #863, design-audit row R20).

The table is the API map's, which is the widest of the six and the one
`docs/tex/generated/api_map.tex` is built from. Three readings of it, because
three things are genuinely different: text mode; ``\texttt``, where five
characters are set from the typewriter font itself and the symbol-font
commands fail at the shipout of whichever page the line lands on; and a cell
an author wrote as LaTeX, where the markup is the content and only the
tabular's own syntax is neutralised.

``python/sal/qa/`` carries its own copy rather than importing
this: `infra/CLAUDE.md` keeps application references out of ``infra/``, and
the package does not depend on ``infra/``. The copy is held to this table by
`tests/regression/docs/test_latex_escaping.py`, which is the guard that makes
two texts one definition.
"""

from __future__ import annotations

#: Characters LaTeX reads as syntax, and what each becomes in text mode.
ESCAPES = {
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

#: Inside ``\texttt`` the five characters below are set from the typewriter
#: font itself, which carries them at their ASCII slots. ``\textbar``,
#: ``\textless``, ``\textgreater`` and the brace commands ask for the OMS
#: symbol font in the typewriter family, which has no such shape, so LaTeX
#: substitutes ``cmsy`` and pdfTeX's font expansion refuses it at the shipout
#: of whichever page the line lands on (#770's build).
CODE_ESCAPES = {
    **ESCAPES,
    "{": r"\char`\{{}",
    "}": r"\char`\}{}",
    "|": r"\char`\|{}",
    "<": r"\char`\<{}",
    ">": r"\char`\>{}",
}


#: What is escaped in a cell an author wrote *as LaTeX*. A subset of
#: :data:`ESCAPES` and not a shorter table: a hand-written note sets
#: ``$\times$`` and ``5{,}041``, so ``$``, ``\``, ``{``, ``}``, ``^`` and
#: ``~`` are the author's and are left alone, while ``&`` and ``%`` are the
#: cell's own syntax and would end the row or the line. The two dashes are
#: converted rather than escaped, which is what they are in every table.
AUTHORED_ESCAPES = {
    character: replacement
    for character, replacement in ESCAPES.items()
    if character in "&%#_\u2013\u2014"
}


def escape(text: str) -> str:
    """``text`` with every LaTeX-significant character neutralised.

    Returns
    -------
    str
    """
    return "".join(ESCAPES.get(character, character) for character in text)


def escape_code(text: str) -> str:
    """``text`` as it is set inside ``\\texttt``: as :func:`escape`, braces from the font.

    Returns
    -------
    str
    """
    return "".join(CODE_ESCAPES.get(character, character) for character in text)


def escape_authored(text: str) -> str:
    """A cell whose text an author wrote as LaTeX, with its syntax kept.

    Returns
    -------
    str
    """
    return "".join(AUTHORED_ESCAPES.get(character, character) for character in text)
