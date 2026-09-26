"""NumPy at the public API: a tensor only behind a torch name (issue #1092).

Read from the source tree: a public top-level function in a package that
takes no derivative of its own returns NumPy or ``float``, unless it lives
in a ``torch.py`` twin, carries a ``_torch`` name, or is torch machinery a
differentiating path is built from --- each listed below with the reason.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[2] / "python" / "sal"

#: The packages whose public entries return NumPy.
GUARDED = frozenset({"likelihood", "search", "sim", "sample", "qa", "validation"})

#: Functions that return a tensor by design, and why.
EXEMPT = {
    # The analytic pruning gradient: an autograd function, differentiated.
    "likelihood/pruning_analytic.py:log_likelihood",
    # The torch pruning kernels' shared helpers.
    "likelihood/pruning_common.py:leaf_indicator",
    "likelihood/pruning_common.py:rescale_partial",
    # The covariate the torch HMM path conditions on.
    "likelihood/spatio_sequential/__init__.py:covariate_block",
    # The rows the differentiable `*_log_partition_torch` bounds read.
    "likelihood/surrogate.py:site_rows",
    # The torch samplers' machinery: a start, a buffer, a gradient.
    "sample/chain.py:start_point",
    "sample/chain.py:on_buffer",
    "sample/chain.py:gradient_at",
    # The HMC position a calibrated ladder hands its sampler.
    "sample/initialize.py:calibrate_ladder",
}


@pytest.mark.critical
@pytest.mark.infra
def test_no_public_function_returns_a_tensor_outside_a_torch_name() -> None:
    found = [
        f"{path.relative_to(PACKAGE).as_posix()}:{node.name}"
        for path in sorted(PACKAGE.rglob("*.py"))
        if path.relative_to(PACKAGE).parts[0] in GUARDED and path.name != "torch.py"
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and not node.name.startswith("_")
        and not node.name.endswith("_torch")
        and node.returns is not None
        and "Tensor" in ast.unparse(node.returns)
    ]

    assert sorted(set(found) - EXEMPT) == [], "a public tensor outside a torch name"
    assert sorted(EXEMPT - set(found)) == [], "an exemption nothing needs"
