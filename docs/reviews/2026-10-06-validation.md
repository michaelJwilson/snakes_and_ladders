# External validation audit: sal at `ef2fd06c` (origin/main, 2026-10-06)

**TL;DR**
1. **No external referee has run in CI for 12 days.** The last `CI` workflow run is 35992553060 (2026-09-24 11:21Z); the last green `Validation` job is run 35992368708 at `124a0ad5` (2026-09-24 11:29Z). Since then 131 first-parent merges (539 commits) have landed, including the `sal` rename (#1049), HiGHS (#1063) and every sampler/HMM change listed in (c). `infra/release.sh:62` runs the full suite but syncs no `validation-*` extra, and `RELEASE.md` does not mention validation, so the release gate skips the external referees too, unless the environment already holds them.
2. **Inventory:** 11 frameworks are registered (`python/sal/validation/__init__.py:66-159`). On main, 201 tests collect outside `release`: 60 framework tests, 10 infra tests in `tests/regression/test_validation.py`, and 131 goals (72 runtime, 59 memory); 3 more carry `release`. The script took `--noconftest` with `PYTHONPATH=python`, because the checked-out `.so` predates `src/ragged.rs`. Every goal figure was measured from 2026-09-23 to 2026-09-25 (`tests/validation/test_goals.py`). No package-side goal outcome is recorded anywhere.
3. **Licences:** every framework is OSI-approved except **gco-v3.0, which is research-use and not OSI**. It sits under an MIT wrapper (`__init__.py:81`, #974). **PyMaxflow is GPL.** Its METADATA states no version; `FRAMEWORKS` says GPL-3.0. Five dependencies fall under the 1,000-star bar: PyMaxflow, gco, BlackJAX, rustworkx and **optree, a core runtime dependency** (`pyproject.toml:42-47`). **torch's Linux wheel pulls 15 NVIDIA packages under `LicenseRef-NVIDIA-Proprietary`**. They are not OSI, and no document flags them.
4. **A downstream caller's HMC path is refereed nowhere externally.** The downstream caller calls `hmc.sample`, `hmc.anneal`, `hmc.parallel_tempering` and `Adaptation` on `EmissionHmmObjective`, 4 call sites in the caller's module. BlackJAX pins only `Integrator.__call__` on Gaussian and Rosenbrock targets. Its body was rewritten by #1217/#1222 (`38f51af6`, `384b149a`), so that pin is stale. T ≠ 1 (#1220/#1249), warm-up (#1207/#1208) and auto step (#1219/#1251) have no external referee.
5. **A downstream caller's HMM routes are not covered by hmmlearn.** The downstream caller uses `forward_log_likelihood_from_density`, `forward_backward`, `ragged.posteriors`, `ragged.viterbi` and `kronecker_order` (13 call sites), plus `Ragged` (9 call sites). hmmlearn covers only equal-length Gaussian and Poisson through `likelihood.hmm` and `opt.hmm.estimation` (`tests/validation/test_hmmlearn.py`), so none of these is covered.
6. **Cuts with forbidden labels:** gco and PyMaxflow run only finite fields. The downstream caller's `alpha_expansion` callers go through forbidden labels (#1139; the caller's module), and `fuse` has no referee.
7. **TRW-S has an external referee: the HiGHS LP**, 3 per-PR tests and 1 release test (`tests/validation/test_highs.py:81-162`), backed by internal enumeration (`tests/regression/search/test_trws.py:95`, `critical`). `search/trws` has been unchanged since 2026-09-26. HiGHS has never run in CI; its only recorded run is local, at #1076 (2026-09-25).

## (a) Framework inventory

The pinned versions in this table equal the installed ones in the shared development environment, and `uv.lock` holds the same versions. Each extra's lower bound is in `pyproject.toml:86-124`. "Last measured" is the recorded `Goal.measured` string. Every oracle test last ran green in CI at `124a0ad5` (2026-09-24), except HiGHS, which never ran there.

| Framework (lock) | Role | sal functions refereed | Claim, tolerance | How it runs | Last measured | Tests (per-PR / release) |
|---|---|---|---|---|---|---|
| PyMaxflow 1.3.2 | oracle + benchmark (#973) | `search.maxflow` (Python and `rust`), `ground_state.lattice_rung` q = 2 | labels identical, energy `==`, flow `rel=FLOW_RTOL` (`test_pymaxflow.py:47-80`) | subprocess, `validation-pymaxflow` | 2026-09-23, #973 (`test_goals.py:80`); 2 runtime, 2 memory goals | 3 / 0 |
| gco-wrapper 3.0.9 | experiment + benchmark (#974) | `search.alpha_expansion` (finite field), `sim.potts.energy` | gco's labelling is a fixed point (0 moves, energy `==`); within 1% at 71²; ≤ 2× the enumerated optimum (`test_gco.py:48-95`) | subprocess, Linux x86-64 only (`pyproject.toml:93-95`) | 2026-09-23, #974 (`test_goals.py:276`) | 2 / 1 (22.6 s, #1088) |
| hmmlearn 0.3.3 | oracle + benchmark (#975) | `opt.hmm.baum_welch`, `baum_welch_family` (Gaussian, Poisson), `likelihood.hmm.viterbi`, `hmm_log_likelihood` | parameters `atol=1e-11` after 1–10 iterations; family fit `rtol=1e-9`; Viterbi path `==`, log probability `rtol=1e-11`; score `rtol=1e-11`. **Equal lengths only** (`scripts/hmmlearn.py:93`) | subprocess, `min_covar=0` | 2026-09-23, #975/#997 (`test_goals.py:301-513`) | 8 / 0 |
| scikit-learn 1.9.1 | oracle + benchmark (#975) | `opt.mixture.expectation_maximization`, `GaussianMixtureObjective` score | `atol=1e-11`; score `rtol=1e-11` (`test_scikit_learn.py:42,139`) | subprocess, `reg_covar=0` | 2026-09-23, #975/#997 | 3 / 0 |
| BlackJAX 1.6.2 | oracle + experiment + benchmark (#963, #1006, #1008) | `hmc.leapfrog`/`hmc.yoshida` (`Integrator.__call__`), `hmc.sample` T = 1, `langevin.mala`, `metropolis.replay`/`random_walk` | integrator `atol=1e-12` (Rosenbrock momentum 1e-10); RWM draw for draw `atol=1e-10`; MALA acceptance within 0.03; HMC means within 4 SE (`test_blackjax.py:38-175`) | subprocess (JAX x64) | RWM 2026-09-24, #1006 (`:779`); HMC 2026-09-24, #1008 (`:897,992,1051`); Gaussian 2026-09-23, #963 (`:383`); 28 runtime, 28 memory goals, mixture recorded unmet at 0.56–0.58× (`:997`) | 10 / 0 |
| rustworkx 0.18.1 | oracle + benchmark (#976) | `sim.graph` generators, round trip, `sim.topology` isomorphism and NNI, `potts_mcmc.bond_probability`/`find_root`/`union_roots` | exact equality and isomorphism (`test_rustworkx.py:56-219`) | subprocess | 2026-09-23, #976 (`:106`) | 13 / 0 |
| Gymnasium 1.3.0 | oracle (#977) | `learn.environment.Environment`, `learn.rollout` | `check_env`; episode `atol=1e-12` | subprocess | — | 7 / 0 |
| TorchRL 0.13.3 | oracle + benchmark (#977) | `learn.ppo.generalized_advantages`, clipped loss, `learn.reinforce.surrogate_loss` | value and gradient `atol=1e-10` | subprocess | 2026-09-23, #977/#997 (`:143,732`) | 3 / 0 |
| PyTorch Geometric 2.8.0.post1 | oracle + benchmark (#977) | `learn.surrogate` set and graph surrogates | `atol=1e-12`/`1e-11` (`test_torch_geometric.py:99,141`) | subprocess | 2026-09-23, #977/#997 (`:167,750`) | 3 / 0 |
| JAX 0.11.2 | oracle + benchmark (#991); also a **core** dependency since #1000 | `sample.hmc.gradient_at` on Gaussian and mixture targets; `scripts/package.py` routes | gradient `atol=1e-12·scale`, value `rel=1e-12` (`test_jax.py:37-100`) | subprocess, `validation-jax` | 2026-09-23, #991/#997 (`:181,614`) | 5 / 1 (10.8 s) |
| HiGHS (SciPy 1.18.1) | oracle + benchmark (#1063) | `search.trws.trws`, `search.tightening.dual_bound` | bound equals the LP value within 1e-8 relative where it converges; stalled LP pinned; integral primal equals the enumerated minimum (`test_highs.py:39-162`) | subprocess, no extra | 2026-09-25, one run at load 5.3 (`:237`): 343.9 s and 267.6 s | 3 / 1 (~611 s, 2.1 GB, `STATUS.md:4134-4141`) |

The following are adopted per #938 but have no extra: ldpc (#954, open) and dwave-samplers (#955, open). The #938 TL;DR counts both among "7 adopted", and the tree disagrees. Several spikes are still pending: pgmpy (#956), stim/PyMatching (#957), dynamax (#958, the ragged-HMM benchmark) and DendroPy (#959). One external referee runs in-tier: `scipy.stats` referees the count families (`tests/regression/test_emissions_count_densities.py:1-22`).

## (b) Licence table

Sources: METADATA, `licenses/` and `Cargo.toml` `license` fields in the shared development environment and `~/.cargo/registry`. Stars could not be fetched, because `gh api` returned 403 for external repositories, so the star flags are the repository's own records.

| Package (version) | Scope | SPDX (source) | OSI | Flags / use |
|---|---|---|---|---|
| numpy 2.5.2 | core | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 (LE) | yes | — |
| scipy 1.18.1 (+HiGHS) | core | BSD (classifier); HiGHS MIT per `FRAMEWORKS` | yes | — |
| matplotlib 3.11.1 | core | matplotlib licence, PSF-based (classifier PSF) | PSF-2.0 yes; the matplotlib text itself not listed | — |
| numba 0.67.0 / llvmlite 0.49.0 | core | BSD / BSD-2-Clause AND Apache-2.0 WITH LLVM-exception | yes | — |
| pyyaml 6.0.3 | core | MIT | yes | — |
| jax / jaxlib 0.11.2 | core (#1000) + validation | Apache-2.0 | yes | `pyproject.toml:120-123` still says "autodiff in the package stays PyTorch", which contradicts `:37-41` |
| optree 0.20.0 | core (#1129) | Apache-2.0 | yes | **flag: under 1,000 stars** (`pyproject.toml:43`) |
| torch 2.13.0 | core | Apache-2.0 AND BSD-2/3 AND BSL-1.0 AND MIT (LE) | yes | **pulls 15 `nvidia-*`/`cuda-*` wheels on Linux, `LicenseRef-NVIDIA-Proprietary` (not OSI)**, plus triton (MIT) |
| sympy, networkx, jinja2, fsspec, filelock, typing_extensions | core transitive | BSD / BSD-3 / BSD / BSD-3 / MIT / PSF-2.0 | yes | — |
| ml_dtypes, opt_einsum | core transitive | Apache-2.0 / MIT | yes | — |
| PyMaxflow 1.3.2 | validation | "GPL" (METADATA; version unstated) | yes | **copyleft; subprocess only, optional extra; flag: under 1,000 stars** |
| gco-wrapper 3.0.9 | validation | MIT (METADATA, no licence file); wraps gco-v3.0 research-use | wrapper yes, **gco-v3.0 no** | **conflicts with root rule "OSI-approved"; approved on #974 as subprocess-only; flag: under 1,000 stars** |
| hmmlearn 0.3.3 | validation | BSD (classifier) | yes | — |
| OpenGM decdacf4 (2019, unmaintained) | validation, source build (#1279) | MIT (`Licence-OpenGM.txt`) | yes | built header-only by `infra/build_opengm.sh`, no Boost or HDF5; its CMake externals are not fetched: Kolmogorov's TRW-S v1.3 and MRF-LIB 2.1 (Microsoft, research-only), maxflow v3.02 and QPBO v1.3 (`pub.ist.ac.at`, research-only or GPL), gco-v3.0 (research-only), IBFS (`cs.tau.ac.il`); every one of those hosts returned 403 through the proxy, GitHub did not. Stars not fetched (403) |
| scikit-learn 1.9.1 (joblib, threadpoolctl, narwhals) | validation | BSD-3-Clause (BSD-3, BSD-3, MIT) | yes | — |
| blackjax 1.6.2 (optax 0.2.8, absl-py) | validation | Apache-2.0 | yes | **flag: about 1,000 stars** (#938) |
| rustworkx 0.18.1 | validation | Apache-2.0 | yes | **flag: near 1,000 stars** (#322) |
| gymnasium 1.3.0 (cloudpickle, Farama-Notifications) | validation | MIT (BSD-3, MIT) | yes | — |
| torchrl 0.13.3 (tensordict, hoptorch, pyvers, orjson) | validation | MIT (licence file; METADATA empty) (BSD, MIT, MIT, **MPL-2.0 AND (Apache-2.0 OR MIT)**) | yes | orjson MPL-2.0: subprocess, unmodified |
| torch_geometric 2.8.0.post1 (aiohttp, tqdm, psutil, xxhash) | validation | MIT (Apache-2.0 AND MIT, **MPL-2.0 AND MIT**, BSD-3, BSD-2) | yes | tqdm MPL-2.0: subprocess |
| aim 3.29.1 (aimrocks, aimrecords, aim-ui) | `track` extra | Apache (classifier); aimrecords MIT; **aim-ui: not stated** | — | PYSEC-2026-1087/1088 (`DEV.md:70`) |
| ruff, mypy, pre-commit, pip-audit, types-PyYAML | dev | MIT, MIT, MIT, Apache, Apache-2.0 | yes | — |
| towncrier 25.8.0 | dev | MIT | yes | **flag: 919 stars** (`pyproject.toml:57-58`) |
| sphinx 8.2.3; nbclient, nbformat, ipykernel, notebook | docs; notebooks | BSD-2-Clause; BSD-3 ×4 | yes | — |
| pyo3 0.29.2, rayon 1.12.0, rand 0.10.2, rand_chacha 0.10.0, rand_distr 0.6.0 | Rust core | MIT OR Apache-2.0 | yes | — |
| numpy (crate) 0.29.0 | Rust core | BSD-2-Clause | yes | — |
| burn-autodiff/ndarray/tensor 0.18.0 | Rust `sandbox` feature | MIT OR Apache-2.0 | yes | pulls **colored (MPL-2.0)**, statically linked only with `--features sandbox` |
| criterion 0.5.1 | Rust dev | Apache-2.0 OR MIT | yes | — |
| 163 `Cargo.lock` crates | Rust all | 146 MIT and/or Apache-2.0; the rest Unlicense/Zlib/BSD-2/Unicode-3.0 options; colored MPL-2.0; r-efi triple-licensed with LGPL (MIT selectable, UEFI target only); 15 not in registry (embedded-hal, embassy: unused targets) | yes | — |

## (c) Staleness

All of the 19 sampler and HMM issues named in the brief merged on 2026-10-06, after the last CI validation run of 2026-09-24. The issue lists come from `git log --no-merges 124a0ad5..HEAD -- <file>`.

| Framework | Last measured / run | sal changes since on the refereed surface | Rerun? |
|---|---|---|---|
| BlackJAX | goals 2026-09-23/24; oracle CI 2026-09-24 | `hmc/__init__.py`: #1207 #1208 #1218 #1219 #1220 #1249 #1251; `Integrator.__call__`/`_scored` rewritten (38f51af6, 384b149a); `langevin.py` #1220; `metropolis.py`, `src/metropolis.rs` #1207 #1218 #1220 #1249; `GaussianTarget` now `supported_gradient` (70443611) | **yes**: the integrator oracle pins rewritten code, and the T = 1 experiment now runs the compiled route. The goal *figures* are not stale (BlackJAX is still 1.6.2), but the package side is unmeasured since #1220; mixture was unmet at 0.56–0.58× before the `mixture_stream` kernel |
| hmmlearn | 2026-09-23 | `opt/hmm/estimation.py`: #1160 #1163 #1165 #1166 #1170 #1176 #1197 #1235; `likelihood/hmm.py` #1176; ragged and count kernels (#1233 #1248 #1253–#1255 #1262 #1265 #1266) not refereed | **yes**: the refereed functions changed. The ragged and count routes need new tests, not reruns |
| scikit-learn | 2026-09-23 | `opt/mixture.py`: #1160 #1165 #1197 #1220 #1234 #1235; `opt/em.py` #1179 #1235 | **yes**: fit and termination changed; the zero-spread floor (#1234) does not reach the fixture |
| JAX | 2026-09-23 | `hmc.gradient_at` routes via `supported_gradient` (#1220); `opt/hmm/jax.py` #1189 #1206 | **yes**: the gradient route under test changed |
| gco | 2026-09-23 | `alpha_expansion.py`: #1070 #1081 #1085 #1089 #1091 #1100 #1104 #1139; `src/lattice_cut.rs` unchanged | **yes**: cheap (per-PR test 1); the Python wrapper changed. Forbidden labels need a new test |
| PyMaxflow | 2026-09-23 | `search/maxflow` #1070 #1091; `src/maxflow.rs` unchanged | yes, low risk: the wrapper changed only |
| rustworkx | 2026-09-23 | `potts_mcmc/sweeps.py` #1142 #1146 #1154; `sim/graph.py` #1081 #1140 | **yes**: the bond-pass file changed, and `from_csr`/`from_directed_csr` are new and unrefereed |
| HiGHS | 2026-09-25 (local, #1076) | `search/trws` #1081 #1089 #1091 (all 2026-09-26) | **yes**: never in CI; the TRW-S cap and `BoundedLabelling` changed after the recorded run |
| TorchRL / PyG / Gymnasium | 2026-09-23 | `learn/` #1089 #1090 #1091 #1129 (signatures, optree) | yes, low priority: the downstream caller does not use `learn` |

## (d) Gaps, weighted by the downstream caller

"The downstream caller calls" counts AST call sites (an AST scan of the caller's tree), with the areas they occur in. The downstream caller locks sal at `9730280f` (#1267; the caller's lock file).

| sal surface | The downstream caller calls (areas) | External referee? | Internal oracle | Proposed referee (licence) |
|---|---|---|---|---|
| `hmc.sample`/`anneal`/`parallel_tempering` + `Adaptation` on `EmissionHmmObjective` | 4 (sandbox) | integrator only, Gaussian/Rosenbrock, T = 1 | `test_hmc_carried_energy.py`, `test_hmc_compiled_tempering.py` (T = 1, 4, 16 moments), `test_supported_gradient.py` (autograd 1e-10) | BlackJAX (Apache-2.0): (1) integrator on the Gaussian-HMM twin already in `scripts/blackjax.py:83` at 1e-10; (2) HMC replay on `logdensity/T` with shared momenta and uniforms, draw for draw, like `metropolis.replay`; (3) `window_adaptation` mass on a frozen warm-up, since BlackJAX 1.6.2 uses Stan's pseudo-count 5 (`blackjax/adaptation/mass_matrix.py:148,339-353`), which is #1207's rule |
| HMM forward / `forward_backward` / `ragged.posteriors` / `ragged.viterbi` / `kronecker_order`, over `Ragged` | 13 + 9 `Ragged` (tests, extensions) | **none** (hmmlearn equal-length only) | `test_ragged_length_one.py`, `test_ragged_viterbi.py`, `test_hmm_paths.py` (enumeration) | hmmlearn (BSD-3): `lengths` with mixed lengths including 1 (#1233); `predict_proba`/`decode`/`score` against posteriors, Viterbi and evidence; Kronecker transitions expanded to dense K₁K₂; arbitrary densities through a `BaseHMM._compute_log_likelihood` subclass, which covers NB, BB and count pair |
| Count emissions (BB, NB, count pair) in HMM kernels (#1255, #1265) | 23 (tests, patch, sim) | densities only (`scipy.stats`, in-tier) | `test_supported_gradient.py` (autograd 1e-10) | the hmmlearn subclass above, with `scipy.stats` log-pmfs (BSD-3) |
| `alpha_expansion` (forbidden labels, `start`), `fuse` | 7 (extensions, patch, tests) | gco, PyMaxflow on finite fields only | `test_forbidden_labels.py` (enumeration 3×3, 2×4) | gco (MIT wrapper / research-use core) with `penalized`'s finite stand-in: fixed-point test at 71². `fuse`: none proposed (non-submodular in general) |
| `search.trws` | 4 (studies, tests) | HiGHS LP | enumeration, `critical` (`test_trws.py:95`); release graph cut q = 2 (`:227`); MIP release (`test_potts_mip.py`) | HiGHS (MIT): already in place, needs a run. pgmpy (#956, MIT) is not needed for the bound |
| Potts MCMC / anneal (`run_annealed`, `PottsMove` SW/Wolff heat-bath, `cluster_tempering`, `parallel_tempering`, `anneal_potts`) | 15 (studies, sandbox) | rustworkx bond partition only | enumerated kernels (`test_potts_heat_bath_cluster.py`, `test_potts_forbidden_labels.py`, `test_potts_tempering_cluster.py`) | none needed for correctness; dwave-samplers (#955, Apache-2.0, flagged) is a benchmark only |
| `EmissionHmmObjective` value/gradient, `opt.fit` | 9 (tests, sandbox) | JAX (Gaussian targets only, not HMM) | `test_supported_gradient.py` | JAX (Apache-2.0): `jax.grad` of a JAX twin of the count HMM; `opt/hmm/jax.py` is in-package, so the twin belongs in `scripts/jax.py` |
| `opt.emission_mixture` EM, `search.mixture_starts` | 30 (sandbox, tests, patch) | none (scikit-learn is Gaussian only) | `test_emission_mixture_objective.py` (`scipy`) | none external for NB/BB EM; keep internal |
| `sim.potts.energy`, `PottsGraph` (`from_csr`, `from_directed_csr`) | 27 (tests, patch) | energy via gco/PyMaxflow re-scores | `test_graph_csr.py` (tuple constructor) | rustworkx/`scipy.sparse` round trip (Apache-2.0/BSD-3), low priority |
| `icm.iterated_conditional_modes`, `merge_small_labels` | 9 | none | internal | none needed |

## (e) Rerun plan

Every step runs on the 4-core reference host. The extras are installed in the shared development environment. First rebuild the extension, because a checkout's `.so` can predate `src/ragged.rs`.

1. **Per-PR validation tier, all frameworks:** `-m "(validation or infra) and not goal and not release"`, 70 tests. Its last record is 107 s (CI step, 2026-09-24), plus 3 HiGHS tests. Order by the downstream caller weight: `-k blackjax`, `-k hmmlearn`, `-k "gco or pymaxflow"`, `-k highs`, `-k rustworkx`, `-k "jax or scikit_learn"`, then RL/PyG.
2. **Goals:** `-m goal`, 131 tests; the last record is 406 s (CI step, 2026-09-24). Run `-k blackjax` first (56 goals), since #1220 changed the kernels behind the mixture (unmet at 0.56–0.58×) and the HMM goals. Record met or unmet for each goal; no record exists today.
3. **Release tier:** `-m "validation and release"`. gco takes 22.6 s and JAX 10.8 s (#1088). HiGHS full size is about 611 s of solve at 2.1 GB peak (`test_goals.py:239`).
4. **New tests, one ticket each, per-PR unless over the 10 s cap:**
   - hmmlearn ragged + length-1 + Kronecker + custom-density posteriors/Viterbi/evidence;
   - BlackJAX HMM integrator, tempered HMC replay and warm-up mass;
   - gco forbidden-label fixed point;
   - JAX count-HMM gradient.

   No prior timings exist for these.
5. **Process:** restore CI, or add `$(python3 infra/validation_extras.py)` to `release.sh`'s sync and a `RELEASE.md` step, so the release gate runs the referees.

Doc disagreements found:
- `DEV.md:143` says the goal is "a ratio of 1"; `_goals.py:29` sets `GOAL_RATIO = 0.55`.
- `pyproject.toml:120-123` says "autodiff in the package stays PyTorch", against #1000.
- #938 counts ldpc and dwave-samplers as adopted.

## Rerun 2026-10-06 at `2112482f` (#1274 §1)

**TL;DR:** 73 of 73 referee tests pass: 70 per-PR, plus 3 release. Of the 131 goals, 106 are met and 25 unmet: 8 runtime and 17 memory. The BlackJAX mixture goal is now met at 0.49× and 0.52×; it was unmet at 0.56–0.58× before #1220. Five new per-PR `validation` tests are clean: TRW-S against the HiGHS ILP, and forbidden-label expansion against gco and PyMaxflow.

Host: the 4-core reference host, `OMP/OPENBLAS/MKL_NUM_THREADS=1`. The extension was built with `infra/build_extension.sh` (release profile, fat LTO) from `2112482f`. The runs used a clean detached worktree at that commit, which `sal.__file__` confirms. Load average was 1.1–1.6. The per-PR tier shared the host with the scratch drivers below for part of its run; the goals and the release tests did not. Framework versions are the audit's (a).

### Per-PR tier

The command is `-m "(validation or infra) and not goal and not release" tests/validation tests/regression/test_validation.py`, as in CI. It took 208 s in pytest (258 s wall), against 107 s on CI on 2026-09-24. Nothing skipped.

| Framework | Passed / run |
|---|---|
| BlackJAX | 10 / 10 |
| hmmlearn | 8 / 8 |
| scikit-learn | 3 / 3 |
| gco | 2 / 2 |
| PyMaxflow | 3 / 3 |
| HiGHS | 3 / 3 |
| rustworkx | 13 / 13 |
| JAX | 5 / 5 |
| Gymnasium | 7 / 7 |
| TorchRL | 3 / 3 |
| PyTorch Geometric | 3 / 3 |
| infra (`test_validation.py`) | 10 / 10 |

Slowest: the BlackJAX leapfrog integrator at 26.2 s, the AST import guard at 16.4 s, the JAX torch routes at 15.1 s and the HiGHS converged-bound test at 14.2 s. All four exceed the 10 s cap without carrying `release`. The cap is not enforced in this job.

### Release-only tests

`-m "validation and release"`: 3 of 3 pass, 476 s in total. Free memory was 15 GB before the run.

| Test | Seconds, this run | Recorded before |
|---|---|---|
| HiGHS release LP (`spatio_only` + `spatio_tiling` release) | 442.6 | ~611 (#1076) |
| gco factor-two bound, 4x4 q = 3 | 24.4 | 22.6 (#1088) |
| JAX mixture likelihood gradient | 8.1 | 10.8 (#1088) |

This run did not re-measure the HiGHS peak memory, recorded at 2.1 GB.

### Goals

The command is `-m goal tests/validation`: 131 goals in 650 s. Each goal compares the package's figure with the framework's, at `GOAL_RATIO` 0.55. The framework figures are the hardcoded measurements from 2026-09-23..25 and were not re-measured; the constants are unchanged. The package figures were logged by a scratch plugin wrapping `assert_meets`/`assert_fits`, which left both assertions in force.

| Framework | Runtime met / total | Memory met / total |
|---|---|---|
| BlackJAX | 28 / 28 | 17 / 28 |
| hmmlearn | 14 / 14 | 12 / 14 |
| scikit-learn | 4 / 4 | 3 / 3 |
| JAX | 6 / 6 | 6 / 6 |
| rustworkx | 4 / 4 | 2 / 2 |
| HiGHS | 2 / 2 | — |
| gco | 2 / 4 | 0 / 4 |
| PyMaxflow | 0 / 2 | 2 / 2 |
| TorchRL | 2 / 4 | — |
| PyTorch Geometric | 2 / 4 | — |
| **All** | **64 / 72** | **42 / 59** |

The unmet goals, as package figure against framework figure:

| Framework | Goal | Package | Framework | Ratio |
|---|---|---|---|---|
| PyMaxflow | Rust cut, `lattice_rung(142, 2)` / `(284, 2)` | 12.4 / 46.2 ms | 15.5 / 68.7 ms | 0.80 / 0.67 |
| TorchRL | `ppo_loss` + gradient / `surrogate_loss` + gradient, 10^5 decisions | 50.3 / 64.2 ms | 14.7 / 18.0 ms | 3.43 / 3.57 |
| PyTorch Geometric | `GraphSurrogate` forward, 142² / 284² | 3.74 / 22.8 ms | 5.85 / 18.9 ms | 0.64 / 1.21 |
| gco | alpha-beta swap, 71² / 142², q = 10 | 139 / 643 ms | 96.4 / 556 ms | 1.45 / 1.16 |
| gco (memory) | expansion 71² / 142²; swap 71² / 142², q = 10 | 3.51 / 12.2; 2.65 / 6.64 MB | 2.69 / 9.76; 2.57 / 9.57 MB | 1.31 / 1.25; 1.03 / 0.69 |
| hmmlearn (memory) | ten Baum-Welch iterations, 10^5 / 10^6 positions | 0.668 / 0.705 MB | 0.373 / 1.20 MB | 1.79 / 0.59 |
| BlackJAX (memory) | HMC, d = 100 / 1,000 / 10,000 | 0.786 / 7.89 / 80.5 MB | 0.799 / 8.09 / 80.0 MB | 0.98 / 0.98 / 1.01 |
| BlackJAX (memory) | MALA, chain stored, d = 100 / 1,000 / 10,000 | 0.582 / 7.86 / 80.4 MB | 0.815 / 8.07 / 80.5 MB | 0.71 / 0.97 / 1.00 |
| BlackJAX (memory) | random walk, chain stored: gaussian-100 / 1000 / 10000 | 1.09 / 8.11 / 80.6 MB | 0.823 / 8.05 / 80.3 MB | 1.32 / 1.01 / 1.00 |
| BlackJAX (memory) | random walk, chain stored: rosenbrock-10 / 100 | 0.438 / 1.15 MB | 0.090 / 0.815 MB | 4.86 / 1.41 |

Eight of the 17 unmet memory goals (MALA and random walk) store the chain (1,000 × d float64) on both sides, so both figures are about the chain's own size. At 0.55 these goals cannot be met while the package returns the chain. No earlier package-side outcome is recorded (audit TL;DR 2), so a regression cannot be told from a goal that was never met. All 131 rows are in the scratch log `goals.jsonl` and are not committed.

The BlackJAX goals the audit flagged:

| Goal | Package | Framework | Ratio |
|---|---|---|---|
| 100 HMC transitions, mixture of 10^5 | 1.05 s | 2.14 s | 0.49, met |
| 100 warm-up + 100 HMC, mixture of 10^5 | 2.17 s | 4.18 s | 0.52, met |
| 300 HMC transitions, Gaussian HMM of 10^4 | 1.14 s | 3.08 s | 0.37, met |
| 100 warm-up + 300 HMC, Gaussian HMM of 10^4 | 1.52 s | 4.14 s | 0.37, met |

### Targeted check 1: TRW-S bound and labelling against the HiGHS LP and ILP

`highs.local_polytope(..., integral=True)` gives the ILP: the same constraint matrix as the LP, with integer node marginals, solved by `milp` with `mip_rel_gap = 0` in the script's subprocess. The labelling energy is recomputed with `sim.potts.energies`. Tolerances: bound ≤ LP within `LP_AGREEMENT` = 1e-8 relative; every other comparison within `ENERGY_AGREEMENT` = 1e-12 × max(1, |value|).

| Instance | Sites | TRW-S bound | LP | ILP (= enumerated where enumerable) | TRW-S labelling | Gap to ILP |
|---|---|---|---|---|---|---|
| square-0..2, strip-0..2 (`test_trws.py`) | 8–9 | = LP within 2e-15 | integral | = LP within 2e-15 | = ILP within 1.8e-15 | 0 |
| triangular-0 | 9 | −1.16508 | −1.14911 | −0.96500 | 0.50821 | **1.4732** |
| triangular-1 | 9 | −2.05912 | −2.05912 | −2.03840 | −1.99960 | **0.0388** |
| triangular-2 | 9 | −1.42732 | −1.40024 | −0.86925 | −0.86925 | 1e-16 |
| potts_lattice/ci, spatio_only/ci, spatio_tiling/ci | 9–12 | = LP | integral | = enumerated | optimal | ≤ 7e-15 |
| potts_lattice/stress | 144 | −265.333870 | −265.333870 | −265.333870 (1 node, 0.06 s) | optimal | −6e-14 |
| spatio_only/stress | 72 | −84.141992 | −84.141992 | −84.141992 (0.03 s) | optimal | 1e-14 |
| potts_lattice/release | 2,500 | −4997.927073 | −4997.927073 | −4997.927073 (1 node, 1.96 s, 186 MB) | optimal | −9e-13 |
| triangular AF 6², J = −1, seed 1274 | 36 | −4.3755 | −4.2310 | −2.0361 (3 nodes, 1.3 s) | 8.4694 | **10.51** |
| triangular AF 9² | 81 | −9.1564 | −8.9121 | −2.7953 (1,115 nodes, 34.6 s) | 20.9562 | **23.75** |
| triangular AF 12² | 144 | −16.7285 | −16.0675 | −2.9889 (2,212 nodes, 131 s) | 30.5839 | **33.57** |
| triangular AF 18² | 324 | −40.0462 | −38.7043 | not proven: time limit at 540 s, incumbent 51.37 | 49.0958 | — |

(a) bound ≤ LP ≤ ILP holds on every row; the ILP equals the enumerated minimum on all 12 enumerable rows. (b) and (c): TRW-S's labelling is optimal wherever the LP is integral. On the frustrated triangular antiferromagnets the bound stays within 0.15–1.34 of the LP, but the decoded labelling lies 10.5–33.6 above the ILP optimum, and the gap grows with size. That points to the decoding, not the bound. It is a finding for `search.trws`; see the follow-up below.

### Targeted check 2: alpha expansion with forbidden labels against gco and PyMaxflow

Forbidden pairs are `-inf` for the package (`sim.potts.forbid`, #1139). Neither framework takes `-inf`, so each receives a finite stand-in. Each forbidden entry of site *i* becomes its lowest allowed log-weight less 1 + Σ_{e∋i}|J_e|, a penalty of at most 4.2 (3x3), 3.1 (2x4), 5.02 (q = 3 lattices) and 6.70 (q = 10 lattices). Moving *i* to any allowed label then lowers the energy, so no optimum and no expansion fixed point uses a forbidden pair.

gco runs `expansion()`. PyMaxflow runs its own grid expansion `aexpansion_grid`, from the same per-site cheapest start as the package; this is new in `scripts/pymaxflow.py`. Each runs in its subprocess.

| Instances | Forbidden used | Fixed point of the package's move, energy bitwise | Energy against the package | Against the enumerated constrained minimum |
|---|---|---|---|---|
| 13 enumerable (3x3 q = 3; 2x4 q = 2, 3, seeds 0–5) | none, all three | gco 13/13, PyMaxflow 13/13 | equal | all three reach it, 0 difference |
| 16², q = 3 | none | yes / yes | gco +0.078%, PyMaxflow 0 (same labelling) | — |
| 16², q = 10 | none | yes / yes | gco −0.82%, PyMaxflow 0 (same labelling) | — |
| 71², q = 3 | none | yes / yes | gco +0.008%, PyMaxflow +0.029% | — |
| 71², q = 10 | none | yes / yes | gco +0.27%, PyMaxflow +0.13% | — |

All rows fall within the sibling's declared 1% agreement (`test_gco.py`). At 71², q = 10, the times were: package 77 ms, gco 62 ms, PyMaxflow 91 ms.

**`fuse`:** no external equivalent. gco exposes `expansion`, `expansion_on_alpha`, `swap` and `alpha_beta_swap` only. PyMaxflow exposes `aexpansion_grid` and `abswap_grid`, with no QPBO.

### Tests added

Both checks were clean and ran under the 10 s cap (idle host), so they were added as per-PR `validation` tests beside their siblings:

- `test_highs.py::test_the_bound_the_lp_and_the_ilp_are_ordered_and_the_labelling_is_above` (`oracle`, 8.9 s): it runs the nine `test_trws.py` instances plus `potts_lattice/stress`, and pins the gapped set to triangular-0 and triangular-1.
- In both `test_gco.py` and `test_pymaxflow.py`: `test_with_forbidden_labels_both_expansions_reach_the_constrained_minimum` (`oracle`, 1.4 s and 1.3 s), and a fixed-point test at 16² and 71² (`experiment`, 0.8 s).

The adapters gained two opt-in modes, with the default paths unchanged: `highs.local_polytope(integral=True)` and `pymaxflow.alpha_expansion`. The shared instances are in `tests/validation/_forbidden.py`.

### Follow-ups

- TRW-S's decoded labelling on the triangular antiferromagnet is 10.5–33.6 above the ILP optimum at 36–144 sites; the optimum is a near three-colouring.
- Five per-PR referee tests exceed the 10 s cap: four in this run's tier (BlackJAX integrator 26.2 s, the import guard 16.4 s, JAX torch routes 15.1 s, HiGHS converged 14.2 s) and the new HiGHS test in its own run (8.9 s idle; it may exceed 10 s under load).
- The eight chain-stored memory goals cannot be met at 0.55; see Goals.

## Rerun through `sal.external` at `8c509808` (#1274, on #1282)

**TL;DR:** 100 of 100 referee tests pass: 97 per-PR and 3 release, against #1284's 73 of 73. Of the 131 goals, 105 are met, against #1284's 106. One verdict changed: gco's expansion at 71², q = 10, measured 0.71× in the suite and 0.45–0.48× in three isolated reruns. No goal constant moves.

Commit `8c509808`: #1284 (`eb1cec29`) merged onto the #1282 stack (#1293, `eda21bfa`). The host, thread settings and goal-logging plugin are as in the rerun above. The extension was built with `infra/build_extension.sh` (release profile, fat LTO). `sal.__file__` resolves into the worktree. Load was 0.5–1.2, and nothing else ran during the timed runs.

Routing: #1284's HiGHS ILP test calls `potts.lower_bound(..., integral=True)` through the module's session, and its gco tests call `potts.ground_state` through the module's session. `_forbidden.py` imports `stand_in` and `allowed_by` from `sal.external.potts_inputs`. `highs.local_polytope(integral=True)` lost its last caller and is removed. `pymaxflow.alpha_expansion` stays, because `sal.external` has no `aexpansion_grid`.

### Pass counts

| Framework | #1284 per-PR | This run | Of which #1284's tests | Of which #1293's |
|---|---|---|---|---|
| BlackJAX | 10 | 16 | 0 | 6 |
| hmmlearn | 8 | 13 | 0 | 5 |
| gco | 2 | 11 | 2 | 7 |
| PyMaxflow | 3 | 8 | 2 | 3 |
| HiGHS | 3 | 5 | 1 | 1 |
| scikit-learn, rustworkx, JAX, Gymnasium, TorchRL, PyG | 34 | 34 | 0 | 0 |
| infra (`test_validation.py`) | 10 | 10 | 0 | 0 |
| **All** | **70** | **97** | **5** | **22** |

#1284's five tests through `sal.external`: the HiGHS ILP test took 1.06 s, against 8.9 s through one-shot subprocesses. gco's forbidden-label oracle took 0.10 s, against 1.4 s. Both run under the session.

### Goals delta

Of the runtime goals, 63 of 72 are met (64 before); of the memory goals, 42 of 59 (unchanged). Every other framework's met count is unchanged.

| Goal | #1284 | This run, in the suite | Isolated, 3 runs |
|---|---|---|---|
| gco expansion, 71², q = 10 | 58.6 ms, 0.42, met | 98.9 ms, 0.71, unmet | 63.8–68.0 ms, 0.45–0.48, met |
| gco expansion, 142², q = 10 | 277 ms, 0.44, met | 343 ms, 0.55, met | 292–344 ms, 0.47–0.55, met 2 of 3 |

The 142² goal sits at its 0.55 threshold, so its verdict depends on run-to-run variation. Among the unmet goals, these ratios moved by more than 0.1: PyMaxflow 142² from 0.80 to 0.66; PyG 284² from 1.21 to 0.95; TorchRL `surrogate_loss` from 3.57 to 2.96; gco swap 142² from 1.16 to 0.97. All four stay unmet.

### Wall times

| Run | #1284 | This run |
|---|---|---|
| Per-PR tier (pytest / wall) | 208 / 258 s, 70 tests, host shared | 174 / 177 s, 97 tests |
| Goals, `-m goal` | 650 s | 647 s |
| Release-only, `-m "validation and release"` | 476 s | 514 s |
| HiGHS release LP | 442.6 s | 480.1 s (+8.5%) |
| gco factor-two bound | 24.4 s | 23.6 s |
| JAX mixture gradient | 8.1 s | 9.4 s |

The HiGHS release test now makes both of its solves through `potts.lower_bound` in one session worker. #1284 ran the same LPs through the adapter, one subprocess each. This run does not separate HiGHS's own seconds from the 37.5 s difference. Free memory was 12 GB before the release run.
