"""Shared QA figure/caption writing, so every QA script formats consistently.

A figure without the parameters that generated it recorded alongside it is
not QA-usable (see ``sim/CLAUDE.md``'s ground-truth-retention rule); this
module is where that convention is enforced once, rather than per script.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.figure import Figure

# LaTeX special characters a caption may not contain unescaped. Backslash is
# included: the only escape sequence captions are allowed to use is ``\_``,
# so any other backslash is a mistake rather than a choice.
_LATEX_SPECIALS = "\\_%&#"

# Thousands separator for integers in generated LaTeX. An underscore rather
# than a comma or a thin space, matching the numeric literals in this
# project's Python; ``\_`` is the escaped form LaTeX text mode needs.
_THOUSANDS = "\\_"


def latex_integer(value: int) -> str:
    """Format an integer with underscore separators, escaped for LaTeX.

    Parameters
    ----------
    value : int
        The integer to format.

    Returns
    -------
    str
        e.g. ``200000`` becomes ``200\\_000``. Values below 10,000 are
        returned unseparated, since ``2\\_000`` is harder to read than
        ``2000``, not easier.
    """
    if abs(value) < 10_000:
        return str(value)
    return f"{value:,}".replace(",", _THOUSANDS)


def check_latex_safe(text: str) -> None:
    """Raise if ``text`` contains an unescaped LaTeX special character.

    ``qa/CLAUDE.md`` requires captions to be plain text that
    ``docs/tex/main.tex`` can ``\\input`` verbatim. Enforced here, at the
    point of writing, rather than asserted separately in each caption test:
    a contract every caller must satisfy belongs in the one function every
    caller goes through.

    Parameters
    ----------
    text : str
        Caption or table body to check.

    Raises
    ------
    ValueError
        If an unescaped special character remains after removing the one
        permitted escape sequence.
    """
    remaining = text.replace(_THOUSANDS, "")
    offenders = sorted(
        {character for character in remaining if character in _LATEX_SPECIALS}
    )
    if offenders:
        msg = (
            f"caption contains unescaped LaTeX special character(s) "
            f"{offenders}; only {_THOUSANDS!r} may be escaped"
        )
        raise ValueError(msg)


_NUCLEOTIDES = "ACGT"


def pearson_correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Linear correlation of two equal-length samples.

    Used where a figure has to report how closely two scoring surfaces agree.

    **Why not a rank correlation**, which is the more obvious choice for
    "do these order things the same way". A rank statistic is discontinuous
    in its inputs: two values that differ by a rounding error can swap rank,
    and the statistic jumps. That is fatal here, because the surfaces being
    compared come from an optimizer, several of whose optima agree to within
    its own convergence tolerance -- so their order is not a property of the
    science, and on another machine it comes out differently. Measured on the
    6-taxon comparison: perturbing the fitted scores by one part in ``1e9``
    moves Spearman's rho by up to ``0.04``, and leaves this correlation
    unchanged to four decimals.

    That matters beyond reproducibility. ``docs/draft.pdf`` is committed and
    CI rebuilds it, so a caption reporting an unstable number fails the build;
    but the deeper point is that such a number is not a measurement of
    anything.

    Both surfaces here are log-likelihoods in the same units, and this is
    invariant to the affine rescaling that separates them, so it answers the
    question that is actually being asked.

    Parameters
    ----------
    first, second : np.ndarray
        1-D arrays of the same length, at least two entries.

    Returns
    -------
    float
        The correlation, in ``[-1, 1]``.

    Raises
    ------
    ValueError
        If the inputs are not 1-D and of equal length, if either has fewer
        than two entries, or if either is constant --- which has no
        correlation with anything rather than zero correlation.
    """
    if first.ndim != 1 or second.ndim != 1 or first.shape != second.shape:
        msg = (
            f"expected two 1-D arrays of equal length, got shapes "
            f"{first.shape} and {second.shape}"
        )
        raise ValueError(msg)
    if first.size < 2:
        msg = f"need at least 2 entries to correlate, got {first.size}"
        raise ValueError(msg)

    centred = [values - values.mean() for values in (first, second)]
    norms = [float(np.sqrt(values @ values)) for values in centred]
    if min(norms) == 0.0:
        msg = "a constant sample has no correlation with anything"
        raise ValueError(msg)
    return float(centred[0] @ centred[1] / (norms[0] * norms[1]))


def state_label(state: int, k: int) -> str:
    """Render a simulated state as a nucleotide letter when ``k == 4``.

    Parameters
    ----------
    state : int
        Simulated state, in ``[0, k)``.
    k : int
        Number of states in the model.

    Returns
    -------
    str
        The nucleotide letter for ``state`` when ``k == 4``; otherwise the
        state's decimal digit, since the nucleotide labelling only applies
        to the 4-state alphabet.
    """
    if k == len(_NUCLEOTIDES):
        return _NUCLEOTIDES[state]
    return str(state)


@dataclass(frozen=True)
class QATable:
    """A rendered LaTeX table fragment together with its caption.

    Parameters
    ----------
    table_path : Path
        Path the ``tabular`` fragment was written to.
    caption_path : Path
        Path the caption text was written to.
    caption : str
        The caption text itself.
    """

    table_path: Path
    caption_path: Path
    caption: str


def write_qa_table(output_dir: Path, stem: str, body: str, caption: str) -> QATable:
    """Write a LaTeX ``tabular`` fragment and its caption under ``output_dir``.

    A table is typeset by LaTeX rather than drawn by matplotlib and saved as
    an image: an image does not match the surrounding type, does not scale
    with the document, and cannot be selected or searched.

    Parameters
    ----------
    output_dir : Path
        Directory to write into; created if missing.
    stem : str
        Base filename, without extension, shared by both outputs.
    body : str
        The ``tabular`` environment, complete, for ``main.tex`` to
        ``\\input``.
    caption : str
        Caption text, subject to :func:`check_latex_safe`.

    Returns
    -------
    QATable
        Paths written to, and the caption text.
    """
    check_latex_safe(caption)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / f"{stem}.tex"
    caption_path = output_dir / f"{stem}_caption.txt"
    table_path.write_text(body if body.endswith("\n") else body + "\n")
    caption_path.write_text(caption)
    return QATable(table_path=table_path, caption_path=caption_path, caption=caption)


@dataclass(frozen=True)
class QAFigure:
    """A rendered QA figure together with its caption.

    Parameters
    ----------
    figure_path : Path
        Path the figure was written to.
    caption_path : Path
        Path the caption text was written to.
    caption : str
        The caption text itself.
    """

    figure_path: Path
    caption_path: Path
    caption: str


def write_qa_figure(output_dir: Path, stem: str, fig: Figure, caption: str) -> QAFigure:
    """Write ``fig`` and ``caption`` under ``output_dir``, named from ``stem``.

    Parameters
    ----------
    output_dir : Path
        Directory to write into; created if missing.
    stem : str
        Base filename, without extension, shared by both outputs.
    fig : matplotlib.figure.Figure
        Figure to save as a PDF (vector, for inclusion in the LaTeX build).
    caption : str
        Caption text to write alongside the figure. Checked by
        :func:`check_latex_safe`, so a caption that would break the LaTeX
        build fails here instead.

    Returns
    -------
    QAFigure
        Paths written to, and the caption text.
    """
    check_latex_safe(caption)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / f"{stem}.pdf"
    caption_path = output_dir / f"{stem}_caption.txt"
    # matplotlib's PDF backend defaults to embedding text as Type 3 fonts
    # (pdf.fonttype=3), which GitHub's pdf.js-based blob viewer fails to
    # render ("Error rendering embedded code"). Type 42 embeds TrueType
    # outlines instead, which pdf.js renders correctly; scoped to this call
    # so it doesn't change matplotlib's global rc state for callers.
    with mpl.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        fig.savefig(figure_path)
    caption_path.write_text(caption)
    return QAFigure(figure_path=figure_path, caption_path=caption_path, caption=caption)
