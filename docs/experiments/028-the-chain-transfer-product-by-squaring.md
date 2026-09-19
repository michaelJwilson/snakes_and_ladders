---
id: "028"
date: 2026-09-19
commit: fd93e62cc47d456eae8355b5650308b0b99991ab
branch: claude/hmc-754-transfer
pr: 0
tickets: [754]
problem: potts-lattice
fixture: tests/regression/fixtures/potts_chain/ci.yaml
size: stress
methods: [recursion, squaring]
budget: 0
seeds: [0]
hardware: Linux-x86_64
status: confirmed
---

# The chain transfer product by squaring: was the ranking's backward row ours?

## Question

Does reassociating the homogeneous transfer product pay on `hmc.sample`, and is the 41.7% `run_backward` beside it outside the package or this loop's tape?

## Numbers

| two readings | recursion | squaring | ratio |
| --- | --- | --- | --- |
| `hmc.sample`, 1,000 draws, chain of 64, q = 3 | 23.501 / 23.597 s | 6.601 / 6.511 s | 3.57x |
| the objective's forward + backward, one gradient | 3.352 / 3.394 ms | 0.948 / 0.923 ms | 3.60x |
| `log_partition` alone, `pytest-benchmark` mean | 1,209.7 / 1,389.8 us | 245.4 / 246.9 us | 4.93x / 5.63x |
| the same with its gradient | 3,130.9 / 3,146.7 us | 717.7 / 742.5 us | 4.36x / 4.24x |
| `--tier mid` self time: `run_backward`; `logsumexp` | 10.191 s; 8.187 s, 512,000 calls | 2.930 s; 1.693 s, 96,000 | 3.48x; 4.84x |
| the same route at q = 3, length 1,024; at q = 64, length 4 | 19.83 ms; 128.9 us | 415.6 us; 2,531 us | 47.7x; 0.04x |

## Finding

Yes, **3.57x**, saving **17.0 s of 23.5** over the draws and 17.0 ms of 23.6 per draw, against the ticket's predicted 1.5-2x --- and the backward row was **ours**: `run_backward` falls 10.19 s to 2.93 s because it walked one node per site. Squaring pays a cube in `q` to buy a logarithm in the length, so the route is taken on the two together and not on the length, which would take the 0.04x row as well. Agreement is **bitwise** at lengths 1 and 2 and `CROSS_DEVICE_RTOL_FLOAT64` elsewhere, realized 6.815e-16 relative at the stress instance. `python tests/benchmarks/profile_hotpaths.py --tier mid --module opt`, two readings, 1-minute load 4.72 to 6.32 with two other agents on the host. Actions: `no actions`.
