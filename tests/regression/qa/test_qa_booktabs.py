"""The one booktabs scaffold every QA table sets its cells in (issue #926).

`qa.figure.booktabs_tabular` and `mathjax_array` hold the frame four modules
wrote out line for line: `qa.likelihood_footprint`, `qa.sim_problem_sizes`,
`qa.starts` (two tables) and `qa.starts_table`. The fold changes how a table
is built and not a byte of it. The referee is the text `main` emitted before
the fold, recorded below from the deterministic builders; the timed starts
tables were compared with the scaffold they replaced on one run, since their
seconds are the host's.
"""

from __future__ import annotations

import pytest
from snakes_and_ladders.qa import likelihood_footprint, sim_problem_sizes
from snakes_and_ladders.qa.figure import booktabs_tabular, mathjax_array
from snakes_and_ladders.qa.starts_table import build_array, build_table

from tests.regression.qa.test_qa_sim_problem_sizes import FIXTURE_PATHS
from tests.regression.qa.test_qa_starts_table import ROWS

#: What each builder emitted on `main` before the fold, for the inputs below.
PRE_FOLD: dict[str, str] = {
    "footprint": "\\begin{tabular}{rrrrr}\n"
    "  \\toprule\n"
    "  Taxa & Sites & Simulate (MB) & Evaluate (MB) & Total (MB) \\\\\n"
    "  \\midrule\n"
    "  20 & 2000 & 0.624 & 2.43 & 3.06 \\\\\n"
    "  20 & 11\\_000 & 3.43 & 13.4 & 16.8 \\\\\n"
    "  100 & 11\\_000 & 17.5 & 69.7 & 87.2 \\\\\n"
    "  \\midrule\n"
    "  1000 & 11\\_000 & 176 & 703 & 879 \\\\\n"
    "  \\bottomrule\n"
    "\\end{tabular}",
    "problem_sizes": "\\begin{tabular}{lrrrr}\n"
    "  \\toprule\n"
    "  Fixture & Taxa & Sites & Seed & Tolerance \\\\\n"
    "  \\midrule\n"
    "  \\texttt{tree\\_jc/stress.yaml} & 4 & 200\\_000 & 20260902 & 0.01 "
    "\\\\\n"
    "  \\texttt{tree\\_jc/ci.yaml} & 4 & 20\\_000 & 20260903 & 0.03 "
    "\\\\\n"
    "  \\texttt{tree\\_jc/release.yaml} & 8 & 200\\_000 & 20260904 & "
    "0.01 \\\\\n"
    "  \\bottomrule\n"
    "\\end{tabular}",
    "starts_array": "\\begin{array}{lrrrr}\n"
    "  \\hline\n"
    "  \\text{initializer} & \\text{init time [s]} & \\text{init. gap "
    "[nats]} & \\text{final time [s]} & \\text{final gap [nats]} \\\\\n"
    "  \\hline\n"
    "  \\texttt{quantile} & 0.1 & 98.0 & 0.0 & -2.0 \\\\\n"
    "  \\texttt{emission++} & 0.1 & 101.2 & 3.2 & 1.2 \\\\\n"
    "  \\texttt{burn-in} & 0.1 \\pm 0.0 & 102.0 \\pm 1.4 & 4.0 \\pm 1.4 & "
    "2.0 \\pm 1.4 \\\\\n"
    "  \\hline\n"
    "\\end{array}",
    "starts_table": "\\begin{tabular}{lrrrr}\n"
    "  \\toprule\n"
    "  initializer & init time [s] & init. gap [nats] & final time [s] & "
    "final gap [nats] \\\\\n"
    "  \\midrule\n"
    "  \\texttt{quantile} & 0.1 & 98.0 & 0.0 & -2.0 \\\\\n"
    "  \\texttt{emission++} & 0.1 & 101.2 & 3.2 & 1.2 \\\\\n"
    "  \\texttt{burn-in} & 0.1 $\\pm$ 0.0 & 102.0 $\\pm$ 1.4 & 4.0 $\\pm$ "
    "1.4 & 2.0 $\\pm$ 1.4 \\\\\n"
    "  \\bottomrule\n"
    "\\end{tabular}",
}


@pytest.mark.critical
@pytest.mark.smoke
@pytest.mark.snapshot
def test_the_folded_tables_are_the_text_main_emitted() -> None:
    built = {
        "footprint": likelihood_footprint.render_footprint(4),
        "problem_sizes": sim_problem_sizes.build_table(
            [sim_problem_sizes._load_named(path) for path in FIXTURE_PATHS]
        )[0],
        "starts_table": build_table(ROWS, instance="x", seconds=5, seeds=2)[0],
        "starts_array": build_array(ROWS),
    }
    assert built == PRE_FOLD


@pytest.mark.analytic
def test_the_scaffold_sets_rules_rows_and_a_separated_last_row() -> None:
    table = booktabs_tabular(
        "lr", ["a", "b"], [["1", "2"], ["3", "4"]], rule_before_last=True
    )
    assert table.splitlines() == [
        r"\begin{tabular}{lr}",
        r"  \toprule",
        r"  a & b \\",
        r"  \midrule",
        r"  1 & 2 \\",
        r"  \midrule",
        r"  3 & 4 \\",
        r"  \bottomrule",
        r"\end{tabular}",
    ]
    assert mathjax_array("l", ["a"], [["1"]]).splitlines() == [
        r"\begin{array}{l}",
        r"  \hline",
        r"  \text{a} \\",
        r"  \hline",
        r"  1 \\",
        r"  \hline",
        r"\end{array}",
    ]
