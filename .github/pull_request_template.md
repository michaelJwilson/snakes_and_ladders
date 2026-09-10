<!--
Title: `[<base branch>] What the change does` -- the branch this PR targets, in
brackets, first (DEV.md, issue #292). `[main]` for a root, the parent branch
for a link in a chain.

This template mirrors CLAUDE.md's "Definition of Done" and "Documentation
Sync" rules. Fill in every section; delete none of them.

The body carries at most 40 content lines (DEV.md, issue #521); blank lines,
these comments and the headings below are not charged, and the `pr-title` job
refuses a body over it. What does not fit belongs where it is refereed --- a
measurement in STATUS.md, a sweep in an experiment file.
-->

### Description

<!-- What changed, and why. Link the issue this closes, if any. -->

### Benchmark

<!--
Required whenever a hot function is new or changed (CLAUDE.md, Performance /
Definition of Done). Report baseline vs. new numbers from pytest-benchmark
or criterion using the table below. If this PR doesn't touch a hot path,
write "N/A — no hot-path change" as text and delete the table instead of
leaving it rendered with blank cells.
-->

| Function | Baseline | This PR | Delta |
| --- | --- | --- | --- |
|  |  |  |  |

<!--
Required whenever this PR touches a regression test that pins output against
a reference within a stated tolerance (CLAUDE.md, "Pin to Independent
Sources" / "Cross-Device Agreement Is a Tolerance") — e.g.
`tests/regression/test_jc_simulate.py::test_simulated_substitution_frequencies_match_analytic_jc`,
which checks `assert_allclose(..., atol=params.tolerance)` against the
closed-form JC probabilities, or
`tests/benchmarks/test_pruning_rust_bench.py::test_numpy_vs_rust_forward_pass`,
which pins the Rust oracle against NumPy with `abs_tol=1e-9`. Report the
realized value even when it passes, not just the tolerance it was checked
against, using the table below. If this PR has no such test, write "N/A" as
text and delete the table instead of leaving it rendered with blank cells.
-->

| Test | Reference | Tolerance (atol/rtol) | Realized value |
| --- | --- | --- | --- |
|  |  |  |  |

### Documentation Sync

<!--
CLAUDE.md requires updating, in this PR, whichever of these the change makes
untrue. Check the ones this PR updates, or check "None applicable".
-->

- [ ] `README.md`
- [ ] `CLAUDE.md` (root or a module's)
- [ ] `DEV.md`
- [ ] `INSTALL.md`
- [ ] `ROADMAP.md`
- [ ] `STATUS.md` / `TICKETS.md`
- [ ] `docs/tex/`
- [ ] `changelog.d/` (add a fragment; see `changelog.d/README.md`, if user-visible)
- [ ] None applicable

### Follow-up / Deferred Work

<!--
Anything intentionally left as a TODO rather than done in this PR. Link the
tracking issue (new or existing) that covers it.
-->

- [ ] None — nothing deferred
- **What's deferred:**
- **Why it's deferred rather than done here:**
- **Tracking issue:**

### Definition of Done

- [ ] **Regression test:** asserts scientific validity (not just shape/execution) and pins the expected output.
- [ ] **Benchmark:** new/changed hot functions include a `pytest-benchmark` (Python) or `criterion` (Rust) bench, with baseline numbers reported above.
- [ ] **Coverage:** the `--cov-fail-under` gate is maintained or raised.
- [ ] **Docs & tooling:** `ruff`, `mypy --strict`, and `cargo` checks pass locally; Documentation Sync above is satisfied.
- [ ] **Dependency hygiene:** new dependencies are OSI-licensed, justified above, and flagged if under 1,000 GitHub stars.
