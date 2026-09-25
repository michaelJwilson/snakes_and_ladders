"""Integration test for the compiled Rust extension.

Requires the package to be built and installed (`pip install .`), unlike the
pure-Python tests under tests/regression and tests/benchmarks.
"""

import pytest
from sal.oxisal import double


@pytest.mark.critical
@pytest.mark.infra
def test_double() -> None:
    assert double(21) == 42
    assert double(0) == 0
    assert double(-3) == -6


@pytest.mark.infra
def test_the_extension_has_one_name() -> None:
    # Issue #1003: the extension is `oxisal`, imported under that name
    # everywhere; the old name survives only in the changelog.
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    old = "oxi_snakes" + "_and_ladders"
    found = subprocess.run(
        ["git", "grep", "-l", old, "--", ":!CHANGELOG.md", ":!changelog.d"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    assert found == []
