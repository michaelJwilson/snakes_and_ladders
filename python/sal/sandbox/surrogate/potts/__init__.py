"""A multigrid surrogate for the Potts ground state on ``spatio_tiling``: declined, conserved (issues #1067, #1074).

**The question.** Can a network that reads a tiling field and returns a
labelling, polished by ICM, reach the ground state closer or sooner than the
solvers it is trained on --- ICM, alpha-expansion, Swendsen--Wang --- at
``spatio_tiling/release``'s size, 5,041 sites and ten states?

**The measured answer.** A 28,873-parameter network, 14 message-passing
blocks over three coarse grids (:mod:`.model`), trained for 15 minutes over
four gated stages (:mod:`.train`): 11x11 and 21x21 on ICM labels, then 21x21
and 41x41 on alpha-expansion labels (:mod:`.labels`). Gap is energy minus
the TRW-S bound, mean +- SE over held-out mixed draws (:mod:`.fields`,
:mod:`.evaluate`):

=============================== ============== ======== ============== ========
Arm                             41x41, 20      median t 71x71, 10      median t
=============================== ============== ======== ============== ========
(a) surrogate argmax            11.34 +- 3.05  13.8 ms  86.37 +- 13.08 24.0 ms
(b) surrogate + ICM             3.84 +- 1.78   14.5 ms  22.70 +- 3.58  25.9 ms
(c) ICM from field argmax       19.84 +- 5.23  0.3 ms   53.88 +- 14.82 0.8 ms
(d) alpha-expansion             0.41 +- 0.29   5.1 ms   1.29 +- 0.31   20.0 ms
(e) Swendsen--Wang, 1000 sweeps 42.73 +- 5.20  169 ms   147.2 +- 11.3  520 ms
=============================== ============== ======== ============== ========

On the paired difference, (b) is lower than ICM (3.0 SE at 41x41, 2.2 SE at
71x71) at 59 and 31 times its time, the median per-field ratio; lower than
Swendsen--Wang (7.0 and 12.6 SE) at a twelfth and a twentieth of its time;
and **higher than alpha-expansion (2.3 and 6.2 SE), and slower**. #1067's
body gives the ICM ratio as 20 to 30; the recorded times give the two
above. The ICM-labelled stages reproduced ICM; the gain appeared on
expansion's labels. The argmax alone gets worse from 41x41 to 71x71, and the
loss was still falling at the stop.

**Why it is declined.** Alpha-expansion is within 1.29 of the bound at
71x71, so the surrogate's only case on this family is speed, and PyTorch's
forward pass alone, 24.0 ms, exceeds expansion's whole solve. The noisy,
strongly coupled family of :mod:`.fields` was designed to leave expansion
energy to lose (25.68 above the bound at 75x75); two stages trained on it,
300 s each, reached a site accuracy of 0.976 against expansion's labels, on
fields whose tile strengths were twice those labelled (:mod:`.fields`), and
were not evaluated against the arms. It lives here on ``sandbox/CLAUDE.md``'s rule for
a declined route that was finished.

**What it referees.** State-relabelling equivariance, bitwise; the fixture
digest and its declared smoothing and coupling; the gap as
:func:`~sal.sim.potts.energy` minus :func:`~sal.search.trws.trws`'s bound
(``tests/regression/sandbox/test_potts_surrogate.py``).

**What would bring it back.** A field family on which surrogate-plus-ICM, or
expansion warm-started at the surrogate, is lower than cold expansion by two
standard errors, or a forward pass below expansion's time at 5,041 sites.
"""

from sal import _submodules

__getattr__ = _submodules(__name__)
