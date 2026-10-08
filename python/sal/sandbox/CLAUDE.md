# sandbox/

The home of developed code not adopted - as failing performance or accuracy guarantees. 
Eg. A hand-rolled implementation that a framework has replaced on a hot path moves in
here, or the framework because it failed to be better.

Root `CLAUDE.md` holds the repository-wide rules, and its **Writing Style**
section binds this file and every docstring, comment and commit message in this
module. It is referenced here, never restated. What follows is local.

## Local rules

- **An implementation moves in when a measured adoption lands, not before.**
  The new approach has to beat the existing on the hot path by the margin
  root `CLAUDE.md` sets, both numbers in the pull request, and the move leaves
  nothing behind but the adapter that fronts the framework. Its regression
  tests move with it and keep passing: they are what the adapter is pinned
  against.
- **A route also moves in when it is superseded in capability, not in speed.**
  The rule above measures; this one does not, because there is nothing to
  measure: the replacement expresses something the conserved route cannot, and
  is not faster. What it must still be is a *referee* — the conserved route
  answers a case the replacement also answers, and the pull request states
  which case and pins the two against each other there. A route conserved
  under this clause with no such case is not conserved, it is abandoned, and
  the rule above applies instead.
- **A route the owner no longer supports moves in by that decision alone.**
  No measurement and no superseding referee: the decision is the reason,
  and the changelog names it. Its tests move with it and keep passing.
- **A declined route moves in only if it was finished.** Clean, scoped to a
  question that was posed, and refereed by tests that still pass — an
  abandoned half-attempt is not conserved, it is reported.
- **What it costs the default build, it does not cost.** A conserved route
  carries its own dependencies behind a build flag, off by default, so the
  wheel, the per-pull-request jobs and a developer's install pay nothing for a
  route that lost. `DEV.md` names the flag and where it is compiled.
- **Something has to build it on a schedule.**
- **Only `tests/` and `sal.qa` import from here.** Nothing supported imports
  it, and `import sal` does not load it.
- **Nothing should be deleted.** An oracle that is slow is still the oracle; an
  oracle nothing tests against is a gap to file as a ticket, not a file to
  remove.

## What is conserved here, and on which rule

Four rules admit a route, and each conserved module names the one it came in
on and the case it referees. The numbers stay where they are produced --- the
pull request that made the move, and `STATUS.md` where they are evidence.

| Module | Rule | The case it referees |
| --- | --- | --- |
| `tropical` | declined | topology search where enumeration gives the optimum (#408) |
| `pruning_burn` | replaced on a hot path | the taped gradient the route that replaced it is pinned against (#449) |
| `count_emissions` | superseded in capability | a constant covariate, where the covariate-aware families must reproduce these bit for bit (#631) |
| `circulant_schedule` | superseded in capability | the circulant case, where a rate per step and the matrix stack it builds are one chain (#658) |
| `rectangular_hmm` | superseded in capability | the equal-length case, where the ragged fit's mask is everywhere true and must fall through to this arithmetic bit for bit (#666) |
| `region_graph` | declined | the Bethe region graph, where the Kikuchi free energy is `message_passing`'s Bethe value and its fixed point; the plaquettes settle at weak coupling on the strip and at strong coupling not at all (#689) |
| `polar_reference` | oracle | the naive successive-cancellation recursion that pins `likelihood.polar.decode_sc` bitwise; the construction and the decoders it referees were promoted to `sim.polar` and `likelihood.polar` when the code got its row (#826) |
| `maxflow_declined` | declined | the minimal minimum cut, arc for arc against the Python Dinic and the package kernel, by three algorithms the kept one is not; the parallel one bitwise across thread counts (#715) |
| `annealed_em` | declined | plain EM, which an empty schedule and one step at one reproduce bitwise; on `emission_mixture/ci` both reach one maximum from every `data` start tried (#903, #916). The HMM twin `annealed_baum_welch`: the tempered E step at one is `ragged.posteriors` bitwise; over ten starts on a six-state Gaussian HMM no schedule beats plain EM by two standard errors (#1171) |
| `covariate_grid` | declined | the continuous exposure, where each gridded score is within its stated bound of the family's `log_density` and a class's evidence within its members' summed bounds of the NumPy oracle; bitwise on a power-of-two grid; linear and log-spaced. No family reaches it: the exposure factors and the trial count is exact; a continuous covariate on a family that does not factor brings it back (#1064) |
| `potts_mip` | declined | the ground state as a MIP over the local polytope, the enumerated minimum on ten small lattices including the frustrated ones; at `spatio_tiling/release` the whole MIP did not leave HiGHS's root node in its time limit (`STATUS.md`), so it brackets the optimum and does not certify it. Tests `release`, the tiling-rung one skipped (#1069) |
| `adaptive_schedules` | declined | the `THERMODYNAMIC` ramp, equal thermodynamic length per step on Kaufman's exact `8 x 8` Ising `sigma_E` to the tolerance its test declares, and the Huang schedule's recurrence bitwise; declined because sal measures no `sigma_E(T)` and `tune_schedule` cannot rank an online schedule (#1333, #1352) |
| `potts_tempering` | unsupported | the Potts `parallel_tempering` with `temperatures="auto"`, `adapt_ladder_potts`, `adapt_ladder_round_trips` and the `tempering` and `tempering-mixed` ground-state arms; enumeration and the product-kernel row referee it in `tests/regression/sandbox/` (#1352) |
