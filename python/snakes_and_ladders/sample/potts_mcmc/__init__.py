"""Monte Carlo move sets on a Potts lattice: single-site, cluster, gradient-informed.

`ROADMAP.md` §1.4 names both cluster algorithms. Single-site flips slow
critically near the transition --- the autocorrelation time of the energy
diverges as the correlation length does --- so no Potts result at a useful
lattice size is reachable through them. Cluster updates flip whole correlated
regions at once and do not.

**The field is easy to get silently wrong.** The reference instance is a Potts
model *in an external field*, and the Fortuin-Kasteleyn construction both
cluster algorithms rest on is exact only at zero field: recolouring a cluster
changes the field term by ``|C| * (h[new] - h[old])``, which the bond
construction knows nothing about. Left there, the sampler runs, produces
plausible configurations, and converges to the wrong distribution. So a cluster
recolouring carries a Metropolis accept step on that difference, and the
chi-square tests in `tests/regression/search/test_potts_mcmc.py` are run with
and without a field because only the first catches its absence.

**Two of the eight move sets are gradient-informed, and on this energy they are
one kernel.** A locally balanced proposal (Zanella 2020) weights every
single-site change by ``sqrt(pi(s') / pi(s))``; Gibbs-with-gradients
(Grathwohl et al. 2021) weights it by the same function of the *first-order
Taylor estimate* of that ratio at the current one-hot state. On a pairwise
energy the estimate is the ratio --- the relaxed log weight is affine in each
site's row, so a single-site change has no second-order term ---
so the two propose from the same law and differ in what they compute to get
there. That is a property of the Potts energy and not of the implementation,
which is why it is pinned by a test rather than assumed by a shared branch:
:func:`taylor_log_ratios` is the estimate, :func:`autodiff_log_ratios` is the
same quantity from the tape, and the three agree to ``1e-12``.

**Two of the eight run where the Fortuin-Kasteleyn construction cannot.** Its
bond probability ``1 - exp(-J)`` is not a probability below zero, so Wolff and
Swendsen-Wang are refused on an antiferromagnet --- the instance a cluster move
is wanted for. :func:`niedermayer_sweep` activates a bond on its energy
relative to a threshold ``E_0`` instead (Niedermayer 1988), which is a
probability for either sign and is Wolff's own where Wolff runs;
:func:`sample_potts_pair` runs two replicas at one temperature and moves them
by Houdayer's isoenergetic cluster swap (2001), whose acceptance is 1 by an
identity rather than by a construction. Neither is a free lunch and the
package does not report one: on the frustrated triangular lattice both
clusters percolate, which
``docs/experiments/022-cluster-moves-for-frustrated-lattices.md`` measures.

**Two of the eight read the field when they build or propose a cluster**
(issue #1041). :func:`ghost_spin_sweep` bonds each site to a ghost site of
its own label with the field as the coupling, so the field is inside the
Fortuin-Kasteleyn measure and no accept step remains;
:func:`label_directed_sweep` proposes every cluster onto one label, cycled
across passes, with a Metropolis-Hastings step on the cluster's field
difference. :func:`cluster_tempering` runs Swendsen-Wang replicas on a
ladder and Houdayer's swap between its coldest pairs, with the accept step a
swap across two temperatures needs.

These are samplers, not optimizers: they are validated by the distribution they
converge to, and nothing here claims to find a ground state. The exception is
:func:`anneal_potts`, an optimizer built *from* the sampler: the same sweep on
a schedule of falling temperatures (issue #267).

**Temperature is model scaling.** The Potts coupling absorbs ``beta``:
``exp(-E / T)`` with ``E = -h[s] - J [s = s']`` is the Boltzmann weight of the
model with ``(J / T, h / T)`` at temperature 1. So a tempered chain runs the
untempered sweeps on the scaled model and there is no second code path:
:func:`tempered` is checked against the energies, and the chain it produces
against ``exp(-E / T)`` enumerated from the *unscaled* model. Tempering a
likelihood is a different object (`snakes_and_ladders.sample.schedule` says why);
here the objective is an energy and the temperature is physical.

See ``docs/tex/textbook.tex``, ``sec:potts`` (Newman &
Barkema chs. 4 and 6 for both algorithms and for Sokal's windowing; Mezard &
Montanari ch. 2).

**The package is three modules, and each imports only those before it.**
:mod:`~snakes_and_ladders.sample.potts_mcmc.moves` names the move sets and
refuses a cluster move on an antiferromagnet;
:mod:`~snakes_and_ladders.sample.potts_mcmc.sweeps` holds the kernels --- one
sweep of each move set, the cluster constructions they share, and the choice
between the Python oracle and the Rust sweep; and
:mod:`~snakes_and_ladders.sample.potts_mcmc.chains` holds the drivers that run
those kernels over a schedule and record the run. Every public name this module
defined before the split (issue #1010) is importable from here, so
``from snakes_and_ladders.sample.potts_mcmc import X`` is unchanged.

**A private name stays in the module that defines it.** The two sweeps the
drivers call, :func:`~snakes_and_ladders.sample.potts_mcmc.sweeps.sweep_at` and
:func:`~snakes_and_ladders.sample.potts_mcmc.sweeps.balanced_sweep_at`, and the
four names the tests pin --- ``GUARD``, ``bond_probability``,
``single_site_sweep`` and ``site_update`` --- are public in
:mod:`~snakes_and_ladders.sample.potts_mcmc.sweeps`. A test that replaces a
kernel's collaborator patches the submodule whose global the kernel reads, not
this one: a name re-exported here is a copy.
"""

from __future__ import annotations

from snakes_and_ladders.sample.potts_mcmc.chains import (
    AnnealedPotts,
    ClusterTempered,
    PottsChain,
    PottsPair,
    TemperedChains,
    TemperedModel,
    adapt_ladder_potts,
    anneal_potts,
    cluster_tempering,
    parallel_tempering,
    sample_potts,
    sample_potts_pair,
    swap_log_ratio,
    sweep_for,
    tempered,
)
from snakes_and_ladders.sample.potts_mcmc.moves import (
    MoveKind,
    PottsMove,
    refuse_negative_coupling,
)
from snakes_and_ladders.sample.potts_mcmc.sweeps import (
    AdjacencyLists,
    ClusterCounter,
    Clusters,
    Recolour,
    adjacency_lists,
    autodiff_log_ratios,
    bond_roots,
    cluster_members,
    find_root,
    ghost_couplings,
    ghost_spin_sweep,
    houdayer_cluster,
    houdayer_move,
    label_directed_sweep,
    niedermayer_sweep,
    niedermayer_threshold,
    swendsen_wang_sweep,
    taylor_log_ratios,
    union_roots,
    wolff_sweep,
)
from snakes_and_ladders.sim.potts import energies

#: Every public name the module defined before issue #1010 split it, and
#: :func:`snakes_and_ladders.sim.potts.energies`, which `mypy --strict`
#: otherwise refuses to re-export: the energy moved to `sim/` in issue #277 so
#: the simulator could score with it, and every caller that had it from here
#: still does. ``AdjacencyLists``, ``adjacency_lists`` and ``bond_roots`` were
#: public and unlisted before the split, and are listed so a re-export is
#: declared once.
__all__ = [
    "AdjacencyLists",
    "AnnealedPotts",
    "ClusterCounter",
    "ClusterTempered",
    "Clusters",
    "MoveKind",
    "PottsChain",
    "PottsMove",
    "PottsPair",
    "Recolour",
    "TemperedChains",
    "TemperedModel",
    "adapt_ladder_potts",
    "adjacency_lists",
    "anneal_potts",
    "autodiff_log_ratios",
    "bond_roots",
    "cluster_members",
    "cluster_tempering",
    "energies",
    "find_root",
    "ghost_couplings",
    "ghost_spin_sweep",
    "houdayer_cluster",
    "houdayer_move",
    "label_directed_sweep",
    "niedermayer_sweep",
    "niedermayer_threshold",
    "parallel_tempering",
    "refuse_negative_coupling",
    "sample_potts",
    "sample_potts_pair",
    "swap_log_ratio",
    "sweep_for",
    "swendsen_wang_sweep",
    "taylor_log_ratios",
    "tempered",
    "union_roots",
    "wolff_sweep",
]
