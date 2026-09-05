"""Test for the run_phylo console-script entry point.

Confirms pyproject.toml's [project.scripts] target is importable and runs. The
CLI itself is still a stub.
"""

from __future__ import annotations

import pytest
from phylo.scripts.run_phylo import main


@pytest.mark.structural
def test_main_runs_without_error(capsys: pytest.CaptureFixture[str]) -> None:
    main()
    captured = capsys.readouterr()
    assert "phylo" in captured.out
