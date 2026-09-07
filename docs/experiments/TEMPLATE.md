---
id: 000
date: 2026-01-01
commit: 0000000000000000000000000000000000000000
branch: main
pr: 0
tickets: [0]
problem: potts-lattice
fixture: tests/regression/fixtures/potts_params.yaml
size: ci
methods: [greedy, candidate]
budget: 0
seeds: [0]
hardware: Linux-x86_64
status: open
---

# Title: the claim, as a question the results answer

## Feature under test

One sentence, falsifiable: what would be true if the feature does what it
claims, stated so the Results section can contradict it.

## Setup

The fixture and its size tier (`ci`, `stress` or `release`, per `DEV.md`), the
budget every method runs at and the unit it is counted in, the seeds shared
across methods, and the oracle that referees the comparison.

## Results

Per-seed outcomes for every method at the shared budget, the paired-test
p-value on the per-seed differences, and the cost of each method. Generated
from the run store where one exists; typed from the measurement otherwise,
with the script that produced it named.

## Figures

The QA stems (`docs/tex/figures/<stem>`) that render this experiment's plots,
or `none`.

## Finding

Numbers, no adjectives: what the results say about the feature under test.

## Conclusion and actions

Each action as a ticket number (`#NNN`) with one line on what it asks, or
`none`.

## What is not claimed

The readings the results do not support, stated so nobody has to infer them.
