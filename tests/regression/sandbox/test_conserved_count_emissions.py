"""The conserved count families are frozen, and still score what they scored (issue #631).

They referee the covariate-aware families at a constant covariate. Frozen: all
27 definitions are hashed, so any edit fails; the one declared edit (dropping
the live protocol from the bases) is inside the hash. Live: both families are
scored and their agreement with the live families today is asserted, so a
torch or numpy change lands in its own PR (sandbox/CLAUDE.md).
"""

from __future__ import annotations

import ast
import hashlib

import pytest
import torch
from snakes_and_ladders import emissions as live
from snakes_and_ladders.sandbox import count_emissions as conserved

from tests._paths import REPO_ROOT

CONSERVED = (
    REPO_ROOT / "python" / "snakes_and_ladders" / "sandbox" / "count_emissions.py"
)

#: Everything the two families reach, closed transitively over the live
#: module's own top-level names, in the order that module declares them.
CARRIED = (
    "Values",
    "_DISPERSION_BRACKET_RATIO",
    "_PROBABILITY_MARGIN",
    "_CONCENTRATION_BRACKET_RATIO",
    "_MAX_BISECTIONS",
    "NegativeBinomialEmission",
    "BetaBinomialEmission",
    "_beta_binomial_log_density",
    "_clamped_deviance",
    "_RATE_FLOOR",
    "_SATURATION_STEPS",
    "_rate_rise",
    "_beta_binomial_saturated",
    "_SolvedBetaBinomial",
    "_beta_binomial_rate_score",
    "_beta_binomial_concentration_score",
    "_rate_score_at",
    "_concentration_score_at",
    "_bisect",
    "_solve_beta_binomial",
    "_effective_trials",
    "identifiable_concentration_bound",
    "_validate_counts",
    "_SolvedDispersion",
    "_weighted_dispersion_score",
    "_solve_dispersion",
    "identifiable_dispersion_bound",
)

#: SHA-256 of those 27 definitions' source, joined by two blank lines, as taken
#: from `emissions.py` on the commit before the covariate seam. Changing this
#: number is the decision to un-freeze; changing the module without it is the
#: mistake this catches.
FROZEN = "b338080eae2ddb1211b46a0336478baebd06b9eba0f5852c9529074a4c3b2e18"


def _carried_source() -> str:
    """The carried definitions' source, in ``CARRIED`` order.

    Classes, functions, ``type`` aliases and constants: all four the module declares.
    """
    source = CONSERVED.read_text()
    lines = source.split("\n")
    found: dict[str, str] = {}

    def record(name: str, node: ast.stmt) -> None:
        start = node.lineno - 1
        for decorator in getattr(node, "decorator_list", []):
            start = min(start, decorator.lineno - 1)
        while start > 0 and lines[start - 1].lstrip().startswith("#:"):
            start -= 1
        assert node.end_lineno is not None
        found[name] = "\n".join(lines[start : node.end_lineno])

    for node in ast.parse(source).body:
        name = getattr(node, "name", None)
        if isinstance(name, str):
            record(name, node)
        elif isinstance(node, ast.TypeAlias) and isinstance(node.name, ast.Name):
            record(node.name.id, node)
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    record(target.id, node)

    return "\n\n\n".join(found[name] for name in CARRIED)


@pytest.mark.critical
@pytest.mark.infra
def test_the_conserved_definitions_are_unchanged() -> None:
    # The whole of this module's worth. A referee edited toward its subject
    # stops refereeing, and the edit is exactly the kind nobody reviews.
    assert hashlib.sha256(_carried_source().encode()).hexdigest() == FROZEN


@pytest.mark.critical
@pytest.mark.infra
def test_every_definition_the_families_reach_is_carried() -> None:
    # A copy that imported a helper would track the edit it exists to referee:
    # `_solve_dispersion` takes a Python float here and a tensor in the live
    # module once the exposure lands. Nothing but `Reestimate` is imported.
    imported = {
        alias.name
        for node in ast.parse(CONSERVED.read_text()).body
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("snakes")
        for alias in node.names
    }

    assert imported == {"Reestimate"}


@pytest.mark.critical
@pytest.mark.oracle
def test_the_conserved_families_score_what_the_live_ones_score() -> None:
    # True today and interesting later: the live families gain a covariate, and
    # at a constant one they must still land here, bitwise.
    counts = torch.tensor([0, 1, 2, 5, 11])
    dispersion, mean = torch.tensor([2.0, 5.0]), torch.tensor([1.0, 4.0])
    trials, alpha, beta = (
        torch.tensor([11.0, 11.0]),
        torch.tensor([2.0, 5.0]),
        torch.tensor([5.0, 2.0]),
    )

    assert torch.equal(
        conserved.NegativeBinomialEmission(dispersion, mean).log_density(counts),
        live.NegativeBinomialEmission(dispersion, mean).log_density(counts),
    )
    assert torch.equal(
        conserved.BetaBinomialEmission(trials, alpha, beta).log_density(counts),
        live.BetaBinomialEmission(trials, alpha, beta).log_density(counts),
    )
