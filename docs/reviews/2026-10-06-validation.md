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
