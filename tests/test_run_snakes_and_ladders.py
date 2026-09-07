"""Test for the run_snakes_and_ladders console-script entry point.

Confirms pyproject.toml's [project.scripts] target is importable and runs. The
CLI itself is still a stub.
"""

from __future__ import annotations

import pytest
from snakes_and_ladders.scripts.run_snakes_and_ladders import main


@pytest.mark.structural
def test_main_runs_without_error(capsys: pytest.CaptureFixture[str]) -> None:
    main()
    captured = capsys.readouterr()
    assert "snakes_and_ladders" in captured.out
