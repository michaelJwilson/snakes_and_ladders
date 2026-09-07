"""What the notebook check must catch, and what it must ignore.

`infra/check_notebooks.py` re-executes the committed notebooks and diffs what
they printed. The comparison is separated from the execution so it can be
tested in milliseconds rather than the 92 seconds a real run costs — which
matters, because the rule it implements was wrong twice while being written
and neither error would have been caught by a test nobody runs.

A figure's `text/plain` is `<Figure size 560x340 with 1 Axes>`, a repr of the
artist that moves with the figure size. Comparing it reported every figure
cell as changed. Excluding it too broadly would stop the check noticing a
cell that lost its figure entirely.
"""

from __future__ import annotations

import importlib
import importlib.abc
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "infra"))

from check_notebooks import differences, image_count, text_outputs


def _cell(*outputs: dict[str, Any]) -> dict[str, Any]:
    return {"cell_type": "code", "outputs": list(outputs)}


def _stream(text: str) -> dict[str, Any]:
    return {"output_type": "stream", "name": "stdout", "text": [text]}


def _figure() -> dict[str, Any]:
    """What a real kernel emits for a `plt.show()`: the image *and* a repr."""
    return {
        "output_type": "display_data",
        "data": {
            "image/png": "iVBORw0KGgo=",
            "text/plain": "<Figure size 560x340 with 1 Axes>",
        },
        "metadata": {},
    }


def _result(value: str) -> dict[str, Any]:
    return {
        "output_type": "execute_result",
        "data": {"text/plain": value},
        "metadata": {},
    }


@pytest.mark.critical
@pytest.mark.structural
def test_a_printed_number_is_compared() -> None:
    committed = [_cell(_stream("log-likelihood -11678.1596\n"))]
    executed = [_cell(_stream("log-likelihood -11679.1596\n"))]

    reported = differences("n.ipynb", committed, executed)

    assert len(reported) == 1
    assert "-11678.1596" in reported[0]
    assert "-11679.1596" in reported[0]


@pytest.mark.critical
@pytest.mark.structural
def test_an_unchanged_notebook_reports_nothing() -> None:
    cells = [_cell(_stream("stable\n"), _figure()), _cell(_result("42"))]

    assert differences("n.ipynb", cells, [dict(cell) for cell in cells]) == []


@pytest.mark.critical
@pytest.mark.structural
def test_a_figures_own_repr_is_not_compared() -> None:
    # The false positive that made every figure cell fail: a kernel reports
    # the artist's size, and resizing a figure is not a changed result.
    committed = [_cell(_figure())]
    resized = [
        _cell(
            {
                "output_type": "display_data",
                "data": {
                    "image/png": "iVBORw0KGgo=",
                    "text/plain": "<Figure size 900x400 with 2 Axes>",
                },
                "metadata": {},
            }
        )
    ]

    assert differences("n.ipynb", committed, resized) == []


@pytest.mark.critical
@pytest.mark.structural
def test_a_cell_that_lost_its_figure_is_reported() -> None:
    # The other side of the same rule. Excluding the repr must not make the
    # check blind to a figure that stopped being drawn.
    committed = [_cell(_stream("text\n"), _figure())]
    executed = [_cell(_stream("text\n"))]

    reported = differences("n.ipynb", committed, executed)

    assert len(reported) == 1
    assert "committed 1 figure(s), re-executed produced 0" in reported[0]


@pytest.mark.critical
@pytest.mark.structural
def test_markdown_cells_do_not_shift_the_cell_numbering() -> None:
    # Cells are numbered by code-cell position, so a diff points at the cell
    # a reader counts rather than at an index including prose.
    markdown = {"cell_type": "markdown", "source": ["# heading"]}
    committed = [markdown, _cell(_stream("a\n")), markdown, _cell(_stream("b\n"))]
    executed = [markdown, _cell(_stream("a\n")), markdown, _cell(_stream("B\n"))]

    reported = differences("n.ipynb", committed, executed)

    assert len(reported) == 1
    assert "cell 2" in reported[0]


@pytest.mark.critical
@pytest.mark.structural
def test_text_outputs_reads_streams_and_results_but_not_images() -> None:
    cell = _cell(_stream("printed\n"), _figure(), _result("returned"))

    assert text_outputs(cell) == ["printed\n", "returned"]
    assert image_count(cell) == 1


@pytest.mark.critical
@pytest.mark.edge_case
def test_a_notebook_of_a_different_length_is_refused() -> None:
    # `zip(strict=True)`: comparing a truncated run against a full one by
    # silently stopping at the shorter would hide the truncation.
    with pytest.raises(ValueError, match="argument 2 is shorter"):
        differences("n.ipynb", [_cell(_stream("a\n"))], [])


@pytest.mark.critical
@pytest.mark.structural
def test_the_comparison_imports_without_the_jupyter_stack() -> None:
    # The `python-tests` job syncs `--extra test`, not `--extra notebooks`, so
    # `nbformat` and `nbclient` are absent there. They were imported at module
    # scope, and this whole file failed to collect (#205's first CI run). The
    # comparison is dict arithmetic and needs neither; only `execute` does.
    class Blocked(importlib.abc.MetaPathFinder):
        def find_spec(self, name: str, *_: object, **__: object) -> None:
            if name.split(".")[0] in {"nbformat", "nbclient"}:
                message = f"No module named {name!r}"
                raise ImportError(message)

    hidden = {
        name: module
        for name, module in sys.modules.items()
        if name.split(".")[0] in {"nbformat", "nbclient", "check_notebooks"}
    }
    for name in hidden:
        del sys.modules[name]
    sys.meta_path.insert(0, Blocked())
    try:
        module = importlib.import_module("check_notebooks")
        assert module.differences("n.ipynb", [], []) == []
    finally:
        sys.meta_path.pop(0)
        sys.modules.update(hidden)
