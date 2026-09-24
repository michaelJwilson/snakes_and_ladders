"""The oracle ladder, declared once: each rung, the rung below it, and the pin.

Issue #734. `ROADMAP.md` §0.4 states the standard --- validation is a ladder
from simple algorithms on small problems to efficient ones on large,
validated against each other along the way --- so a method is established by
the rung below it and the ladder *is* the oracle coverage. This module is the
one declaration of that ladder: per problem, each rung's callable, the rung
it is pinned against, and the regression test that pins the pair.

**A rung with no test is a ticket, never a blank.** `test=None` carries
`ticket`, so a gap is a row a reader can act on rather than an absence they
have to notice. `tests/regression/test_oracle_ladder.py` holds every named
test to exist and carry `oracle`, and every rung without one to its number.
The rule stands whatever the count: today no row carries a ticket, issue
#734 having pinned the last four --- the mixtures --- in its step 6.

This is a ladder and not a census: where the survey (issue #734, first
comment) lists several tests for one rung, the one recorded here is the one
pinning that rung to the rung below, and the rest are recorded nowhere. Two
conventions follow from that:

* `below` is `None` where nothing in the ladder is under the rung --- the
  exact end of a ladder, or a referee outside it: a closed form, a second
  implementation, a framework. The test's own name states which.
* one rung appears once per rung it is pinned against, so a method pinned at
  the exact end and wanted against a cheaper rung is two rows. HMC is that
  case: an analytic Gaussian under one row, the enumerated assignment
  posterior under the other.

Infrastructure, not science: the callables are strings this module never
imports, read the way `infra/problems_tables.py` reads the fixture registry
(`infra/CLAUDE.md`). The guard resolves them; nothing here does.

The unit a rung spends is `snakes_and_ladders.cost.Cost`, declared here until
issue #860 and imported since: `opt.budget.Budget` holds the same axis equal,
and one axis is one vocabulary.

`Rung` here is a rung of the oracle ladder. `search.ground_state.Rung`, which
predates it, is an instance at a size.
"""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass

from snakes_and_ladders.cost import Cost

#: The five ladders, in the order the survey tables run.
PROBLEMS = ("potts", "tree", "hmm", "codes", "mixture")

#: The issue a rung with no test carries, one bullet per rung. No row carries
#: it today; the guard reads it to hold that a rung without a test names an
#: issue, which is the rule and not the count.
LADDER_TICKET = 734


#: The units the rungs spend. `Cost` is the package's vocabulary and carries
#: the units a budget names as well (issue #860), so a member no rung spends
#: is not a stale one; what this holds is the ladder's own side of it.
LADDER_UNITS: frozenset[Cost] = frozenset(
    {
        Cost.EXACT,
        Cost.PASS,
        Cost.ITERATIONS,
        Cost.SWEEPS,
        Cost.EVALUATIONS,
        Cost.GRADIENTS,
        Cost.TRAINED,
        Cost.SEVERAL,
    }
)


@dataclass(frozen=True)
class Rung:
    """One rung of one problem's ladder, and what pins it to the rung below.

    Parameters
    ----------
    problem : str
        One of :data:`PROBLEMS`.
    name : str
        The rung's short name, unique within the problem up to the rung it is
        pinned against. Where one bullet of issue #734 names several
        callables, they are named here and the first is `callable`.
    callable : str
        The `module.callable` the rung implements, package-relative:
        `likelihood.potts.strip_log_partition`.
    below : str or None
        The `name` of the rung this one is pinned against, or None where
        nothing in the ladder is below it.
    test : str or None
        The pytest node id pinning the pair, `tests/regression/<path>::<function>`,
        or None where no test does.
    cost : Cost
        The unit the rung spends, keyword-only so every declaration states it
        (issue #818). The field names the unit and carries no number; a
        number belongs in `STATUS.md` or a benchmark.
    ticket : int or None
        The issue carrying the missing pin, set exactly where `test` is None.
    """

    problem: str
    name: str
    callable: str
    below: str | None
    test: str | None
    _: KW_ONLY
    cost: Cost
    ticket: int | None = None


T = "tests/regression/"

#: Every rung of every ladder, per problem, in the order the survey's table
#: runs. Each is pinned by a named `oracle` test.
LADDER: tuple[Rung, ...] = (
    # --- Potts / lattice ---------------------------------------------------
    Rung(
        "potts",
        "enumeration",
        "likelihood.potts.enumerate_potts",
        None,
        T + "search/test_search_support.py"
        "::test_the_enumerated_labelling_weight_is_enumerate_potts_s_boltzmann_weight_at_beta_one",
        cost=Cost.EXACT,
    ),
    Rung(
        "potts",
        "transfer matrix",
        "likelihood.potts.strip_log_partition",
        "enumeration",
        T + "likelihood/test_potts_exact.py"
        "::test_the_transfer_matrix_reproduces_exhaustive_enumeration",
        cost=Cost.PASS,
    ),
    Rung(
        "potts",
        "transfer matrix",
        "likelihood.potts.strip_log_partition",
        "sum-product / BP",
        T + "likelihood/test_potts_exact.py"
        "::test_the_transfer_matrix_is_sum_product_on_the_strip_that_is_a_tree",
        cost=Cost.PASS,
    ),
    Rung(
        "potts",
        "log Z by squaring",
        "opt.potts.log_partition_by_squaring",
        "transfer matrix",
        T + "opt/test_opt_potts.py"
        "::test_squaring_is_the_transfer_matrix_on_a_strip_one_site_wide",
        cost=Cost.PASS,
    ),
    Rung(
        "potts",
        "sum-product / BP",
        "likelihood.message_passing.sum_product",
        "enumeration",
        T + "likelihood/test_message_passing.py"
        "::test_sum_product_on_the_potts_tree_is_the_enumeration",
        cost=Cost.ITERATIONS,
    ),
    Rung(
        "potts",
        "flooding schedule",
        "likelihood.belief_propagation.belief_propagation",
        "sum-product / BP",
        T + "likelihood/test_message_passing.py"
        "::test_an_iterative_schedule_on_the_loopy_lattice_is_belief_propagation",
        cost=Cost.ITERATIONS,
    ),
    Rung(
        "potts",
        "residual schedule",
        "likelihood.schedule.ResidualMessageSchedule",
        "flooding schedule",
        T + "likelihood/test_message_passing.py"
        "::test_an_iterative_schedule_on_the_loopy_lattice_is_belief_propagation",
        cost=Cost.ITERATIONS,
    ),
    Rung(
        "potts",
        "Kikuchi / region graph",
        "sandbox.region_graph.kikuchi_free_energy",
        "enumeration",
        T + "sandbox/test_region_graph.py"
        "::test_the_free_energy_is_the_exact_log_partition_on_a_tree",
        cost=Cost.ITERATIONS,
    ),
    Rung(
        "potts",
        "dual / LP bound",
        "search.tightening.dual_bound",
        "enumeration",
        T + "search/test_tightening.py"
        "::test_the_bound_never_exceeds_the_enumerated_ground_state",
        cost=Cost.ITERATIONS,
    ),
    Rung(
        "potts",
        "dual / LP bound",
        "search.tightening.dual_bound",
        "sum-product / BP",
        T + "search/test_tightening.py"
        "::test_the_dual_bound_is_the_zero_temperature_belief_propagation_energy",
        cost=Cost.ITERATIONS,
    ),
    Rung(
        "potts",
        "variational bounds",
        "likelihood.surrogate.mean_field_log_partition",
        "enumeration",
        T + "likelihood/test_surrogate.py"
        "::test_mean_field_and_spanning_tree_bounds_sandwich_log_z",
        cost=Cost.ITERATIONS,
    ),
    Rung(
        "potts",
        "exact cut / max-flow",
        "search.maxflow.ising_ground_state",
        "enumeration",
        T + "search/test_maxflow.py"
        "::test_the_cut_finds_the_enumerated_minimum_with_a_per_node_field",
        cost=Cost.PASS,
    ),
    Rung(
        "potts",
        "simulated bifurcation",
        "search.bifurcation.simulated_bifurcation",
        "enumeration",
        T + "search/test_bifurcation.py"
        "::test_the_relaxation_reaches_the_enumerated_ground_state_of_the_declared_glass",
        cost=Cost.ITERATIONS,
    ),
    Rung(
        "potts",
        "simulated bifurcation",
        "search.bifurcation.simulated_bifurcation",
        "exact cut / max-flow",
        T + "search/test_bifurcation.py"
        "::test_at_two_labels_the_relaxation_is_read_against_the_exact_cut",
        cost=Cost.ITERATIONS,
    ),
    Rung(
        "potts",
        "declined max-flow kernels",
        "sandbox.maxflow_declined.min_cut",
        "exact cut / max-flow",
        T + "sandbox/test_maxflow_declined.py"
        "::test_every_declined_kernel_returns_the_python_cut_on_seeded_networks",
        cost=Cost.PASS,
    ),
    Rung(
        "potts",
        "max-cut (SDP)",
        "search.max_cut.goemans_williamson",
        "enumeration",
        T + "search/test_max_cut.py"
        "::test_the_rounded_cut_reaches_the_enumerated_optimum_on_a_lattice",
        cost=Cost.ITERATIONS,
    ),
    Rung(
        "potts",
        "max-cut (SDP)",
        "search.max_cut.goemans_williamson",
        "alpha-expansion",
        T + "search/test_max_cut.py"
        "::test_the_rounded_cut_is_the_gauged_alpha_expansion_optimum_at_two_labels",
        cost=Cost.ITERATIONS,
    ),
    Rung(
        "potts",
        "alpha-expansion",
        "search.alpha_expansion.alpha_expansion",
        "exact cut / max-flow",
        T + "search/test_alpha_expansion.py"
        "::test_two_labels_reproduce_the_exact_minimum_cut",
        cost=Cost.ITERATIONS,
    ),
    Rung(
        "potts",
        "alpha-expansion backends",
        "search.alpha_expansion.alpha_expansion",
        "alpha-expansion",
        T + "search/test_alpha_expansion.py"
        "::test_the_rust_cut_reproduces_the_python_expansion",
        cost=Cost.ITERATIONS,
    ),
    Rung(
        "potts",
        "ICM",
        "search.alpha_expansion.iterated_conditional_modes",
        "exact cut / max-flow",
        T + "search/test_potts_sizing.py"
        "::test_the_zero_field_optimum_is_a_closed_form_three_ways",
        cost=Cost.SWEEPS,
    ),
    Rung(
        "potts",
        "Gibbs / heat bath",
        "sample.gibbs.gibbs_sweep",
        "enumeration",
        T + "sim/test_potts_simulate.py"
        "::test_gibbs_sampling_matches_brute_force_enumeration_on_a_loopy_lattice",
        cost=Cost.SWEEPS,
    ),
    Rung(
        "potts",
        "Potts MCMC, Wolff, Swendsen--Wang",
        "sample.potts_mcmc.sample_potts",
        "enumeration",
        T + "sample/test_potts_mcmc.py"
        "::test_the_chain_is_drawn_from_the_exact_boltzmann_distribution",
        cost=Cost.SWEEPS,
    ),
    Rung(
        "potts",
        "Swendsen--Wang, Rust",
        "sample.potts_mcmc._cluster_pass_rust",
        "Potts MCMC, Wolff, Swendsen--Wang",
        T + "sample/test_potts_mcmc_cluster_rust.py"
        "::test_the_rust_pass_is_the_oracles_pass_bitwise_on_the_same_draws",
        cost=Cost.SWEEPS,
    ),
    Rung(
        "potts",
        "locally balanced proposal",
        "sample.balanced.log_balanced_weights",
        "enumeration",
        T + "sample/test_potts_mcmc.py"
        "::test_the_locally_balanced_chain_is_drawn_from_the_exact_boltzmann_distribution",
        cost=Cost.SWEEPS,
    ),
    Rung(
        "potts",
        "Gibbs with gradients",
        "sample.potts_mcmc.taylor_log_ratios",
        "enumeration",
        T + "sample/test_potts_mcmc.py"
        "::test_the_gibbs_with_gradients_chain_is_drawn_from_the_exact_boltzmann_distribution",
        cost=Cost.SWEEPS,
    ),
    Rung(
        "potts",
        "Niedermayer cluster move",
        "sample.potts_keyed.NiedermayerMove",
        "enumeration",
        T + "sample/test_potts_mcmc.py"
        "::test_the_niedermayer_chain_is_drawn_from_the_exact_boltzmann_distribution",
        cost=Cost.SWEEPS,
    ),
    Rung(
        "potts",
        "Houdayer pair move",
        "sample.potts_mcmc.sample_potts_pair",
        "enumeration",
        T + "sample/test_potts_mcmc.py"
        "::test_each_replica_of_a_houdayer_pair_is_drawn_from_the_exact_boltzmann_distribution",
        cost=Cost.SWEEPS,
    ),
    Rung(
        "potts",
        "annealed importance sampling",
        "sample.annealed.annealed_importance_sampling",
        "transfer matrix",
        T + "sample/test_search_annealed.py"
        "::test_the_importance_sampled_log_partition_is_the_transfer_matrix",
        cost=Cost.SWEEPS,
    ),
    Rung(
        "potts",
        "population annealing",
        "sample.annealed.population_annealing",
        "transfer matrix",
        T + "sample/test_search_annealed.py"
        "::test_the_population_annealed_log_partition_is_the_transfer_matrix",
        cost=Cost.SWEEPS,
    ),
    Rung(
        "potts",
        "simulated tempering",
        "sample.annealed.simulated_tempering",
        "enumeration",
        T + "sample/test_search_annealed.py"
        "::test_the_simulated_tempering_walker_is_the_enumerated_law_at_every_rung",
        cost=Cost.SWEEPS,
    ),
    Rung(
        "potts",
        "cluster moves as RL arms",
        "sample.potts_keyed.cluster_moves",
        "Potts MCMC, Wolff, Swendsen--Wang",
        T + "learn/test_cluster_arms.py"
        "::test_a_wolff_step_is_potts_mcmcs_own_sweep_bitwise",
        cost=Cost.SWEEPS,
    ),
    Rung(
        "potts",
        "annealing",
        "sample.potts_mcmc.anneal_potts",
        None,
        T + "sample/test_potts_mcmc.py"
        "::test_annealing_reaches_the_closed_form_ground_energy_where_descent_does_not",
        cost=Cost.SWEEPS,
    ),
    Rung(
        "potts",
        "parallel tempering",
        "sample.potts_mcmc.parallel_tempering",
        "annealing",
        T + "sample/test_potts_mcmc.py"
        "::test_tempering_reaches_the_ground_energy_annealing_reaches",
        cost=Cost.SWEEPS,
    ),
    Rung(
        "potts",
        "learned policy",
        "learn.potts_nd.PottsNDEnvironment",
        "exact cut / max-flow",
        T + "learn/test_potts_nd.py::test_two_labels_agree_with_the_exact_cut",
        cost=Cost.TRAINED,
    ),
    Rung(
        "potts",
        "learned lattice surrogate",
        "learn.ranking.lattice_examples",
        "enumeration",
        T + "learn/test_search_lattice_surrogate.py"
        "::test_a_surrogate_learns_the_gap_above_the_mean_field_bound_at_nine_sites",
        cost=Cost.TRAINED,
    ),
    Rung(
        "potts",
        "run_tempering, run_greedy, run_max_product",
        "search.ground_state.run_tempering",
        None,
        T + "search/test_ground_state.py"
        "::test_the_runners_record_the_energy_their_kernels_return",
        cost=Cost.SEVERAL,
    ),
    # --- trees / phylogenetics ---------------------------------------------
    Rung(
        "tree",
        "projection",
        "search.projection.project",
        None,
        T + "search/test_projection_seeding.py"
        "::test_the_projected_draw_is_the_closed_form_mixture_of_the_flattened_families",
        cost=Cost.PASS,
    ),
    Rung(
        "tree",
        "topology enumeration",
        "sim.topology.enumerate_topologies",
        None,
        T + "search/test_search_exhaustive.py"
        "::test_enumeration_produces_every_topology_exactly_once",
        cost=Cost.EXACT,
    ),
    Rung(
        "tree",
        "topology equality",
        "sim.topology.robinson_foulds",
        None,
        "tests/validation/test_rustworkx.py"
        "::test_topology_equality_is_labelled_isomorphism",
        cost=Cost.PASS,
    ),
    Rung(
        "tree",
        "parsimony, Fitch",
        "likelihood.parsimony.fitch_score",
        None,
        T + "likelihood/test_likelihood_parsimony.py"
        "::test_fitch_matches_exhaustive_enumeration_over_internal_labellings",
        cost=Cost.PASS,
    ),
    Rung(
        "tree",
        "parsimony, Sankoff",
        "likelihood.parsimony.sankoff_score",
        "parsimony, Fitch",
        T + "likelihood/test_likelihood_parsimony.py"
        "::test_sankoff_with_the_unit_matrix_is_fitch_on_every_five_taxon_topology",
        cost=Cost.PASS,
    ),
    Rung(
        "tree",
        "parsimony against likelihood",
        "likelihood.parsimony.fitch_score",
        "pruning, NumPy",
        T + "likelihood/test_likelihood_parsimony.py"
        "::test_the_short_branch_likelihood_ranks_the_topologies_as_the_fitch_score_does",
        cost=Cost.PASS,
    ),
    Rung(
        "tree",
        "pruning, NumPy",
        "likelihood.pruning.log_likelihood",
        None,
        T + "likelihood/test_likelihood_pruning.py::test_pruning_matches_brute_force",
        cost=Cost.PASS,
    ),
    Rung(
        "tree",
        "pruning, Rust",
        "likelihood.pruning_rust.log_likelihood",
        "pruning, NumPy",
        T + "likelihood/test_pruning_common.py"
        "::test_the_rust_route_meets_the_oracle_within_the_float64_bound",
        cost=Cost.PASS,
    ),
    Rung(
        "tree",
        "pruning, Torch",
        "likelihood.pruning_torch.log_likelihood",
        "pruning, NumPy",
        T + "likelihood/test_pruning_common.py"
        "::test_the_torch_route_reproduces_the_oracle_bitwise",
        cost=Cost.PASS,
    ),
    Rung(
        "tree",
        "pruning, analytic gradient",
        "likelihood.pruning_analytic.log_likelihood",
        "pruning, Torch",
        T + "likelihood/test_pruning_analytic.py"
        "::test_gradient_matches_the_taped_gradient",
        cost=Cost.PASS,
    ),
    Rung(
        "tree",
        "Hadamard / spectral",
        "likelihood.hadamard.hadamard_conjugation",
        None,
        T + "likelihood/test_likelihood_hadamard.py"
        "::test_the_conjugation_returns_the_true_split_weights_on_the_exact_spectrum",
        cost=Cost.PASS,
    ),
    Rung(
        "tree",
        "distance start (NJ)",
        "search.neighbor_joining.neighbor_joining",
        None,
        T + "search/test_search_neighbor_joining_scipy.py"
        "::test_the_two_return_the_same_tree_on_an_ultrametric_matrix",
        cost=Cost.PASS,
    ),
    Rung(
        "tree",
        "distance start (NJ)",
        "search.neighbor_joining.neighbor_joining",
        "topology enumeration",
        T + "search/test_neighbor_joining.py"
        "::test_the_joined_tree_is_the_least_squares_optimum_over_the_enumerated_topologies",
        cost=Cost.PASS,
    ),
    Rung(
        "tree",
        "tropical / Grassmannian",
        "sandbox.tropical.resolutions",
        None,
        T + "sandbox/test_sandbox_tropical.py"
        "::test_the_combinatorial_resolution_is_the_tropical_plucker_argmin",
        cost=Cost.PASS,
    ),
    Rung(
        "tree",
        "tree search (NNI/SPR)",
        "search.infer.infer",
        "topology enumeration",
        T + "search/test_search_exhaustive.py"
        "::test_hill_climbing_reaches_the_enumerated_maximum",
        cost=Cost.EVALUATIONS,
    ),
    Rung(
        "tree",
        "cheaper searches",
        "search.infer.infer",
        "topology enumeration",
        T + "search/test_search_exhaustive.py"
        "::test_the_cheaper_searches_reach_the_enumerated_maximum",
        cost=Cost.EVALUATIONS,
    ),
    Rung(
        "tree",
        "analytic surrogates",
        "likelihood.surrogate.PlugInLikelihood",
        "pruning, NumPy",
        T + "likelihood/test_surrogate.py"
        "::test_plug_in_bound_is_below_every_fitted_likelihood",
        cost=Cost.PASS,
    ),
    Rung(
        "tree",
        "surrogate-ranked search",
        "learn.ranking.LearnedTreeSurrogate",
        "tree search (NNI/SPR)",
        T + "learn/test_search_surrogate.py"
        "::test_surrogate_ranked_search_reaches_what_the_full_search_reaches",
        cost=Cost.TRAINED,
    ),
    Rung(
        "tree",
        "surrogate-ranked search",
        "learn.ranking.LearnedTreeSurrogate",
        "analytic surrogates",
        T + "learn/test_search_surrogate.py"
        "::test_the_learned_surrogate_ranks_the_topologies_the_plug_in_bound_ranks",
        cost=Cost.TRAINED,
    ),
    Rung(
        "tree",
        "learned GNN surrogate",
        "learn.surrogate.GraphSurrogate",
        None,
        "tests/validation/test_torch_geometric.py"
        "::test_pyg_s_gin_reproduces_the_graph_surrogate_on_tied_weights",
        cost=Cost.TRAINED,
    ),
    Rung(
        "tree",
        "RL policy / PPO",
        "learn.tree.TreeEnvironment",
        "topology enumeration",
        T
        + "learn/test_search_rl.py::test_greedy_search_reaches_the_enumerated_optimum",
        cost=Cost.TRAINED,
    ),
    Rung(
        "tree",
        "support / tempering over topologies",
        "search.support.neighbourhood_support",
        "topology enumeration",
        T + "search/test_search_support.py"
        "::test_the_nni_neighbourhood_of_four_taxa_is_the_whole_space_so_the_two_supports_agree",
        cost=Cost.SWEEPS,
    ),
    Rung(
        "tree",
        "Gibbs / annealing over topologies",
        "sample.gibbs.topology_step",
        "topology enumeration",
        T + "sample/test_gibbs.py"
        "::test_the_topology_move_at_temperature_one_samples_the_enumerated_flat_prior_weight",
        cost=Cost.SWEEPS,
    ),
    # --- HMM / spatio-sequential -------------------------------------------
    Rung(
        "hmm",
        "path enumeration",
        "likelihood.hmm_paths.enumerate_hidden_paths",
        None,
        T + "likelihood/test_hmm_paths.py"
        "::test_a_marginal_is_the_summed_joint_over_paths_through_that_state",
        cost=Cost.EXACT,
    ),
    Rung(
        "hmm",
        "forward / forward-backward",
        "likelihood.forward_backward.forward_backward",
        "path enumeration",
        T + "opt/test_opt_hmm.py::test_forward_matches_brute_force_path_enumeration",
        cost=Cost.PASS,
    ),
    Rung(
        "hmm",
        "path sampling",
        "likelihood.forward_backward.sample_path",
        "path enumeration",
        T + "likelihood/test_hmm_paths.py"
        "::test_the_sampled_paths_are_drawn_from_the_enumerated_path_posterior",
        cost=Cost.PASS,
    ),
    Rung(
        "hmm",
        "forward via sum-product",
        "likelihood.message_passing.sum_product",
        "forward / forward-backward",
        T + "likelihood/test_message_passing.py"
        "::test_the_tree_schedule_on_a_deep_chain_is_the_forward_recursion",
        cost=Cost.ITERATIONS,
    ),
    Rung(
        "hmm",
        "tree schedule, Rust",
        "likelihood.message_passing_rust.tree_messages",
        "forward via sum-product",
        T + "likelihood/test_message_passing_rust.py"
        "::test_the_kernel_writes_the_oracle_s_messages_edge_for_edge",
        cost=Cost.ITERATIONS,
    ),
    Rung(
        "hmm",
        "Viterbi",
        "likelihood.message_passing.max_product",
        "path enumeration",
        T + "likelihood/test_message_passing.py"
        "::test_every_chain_evaluator_is_the_path_enumeration",
        cost=Cost.PASS,
    ),
    Rung(
        "hmm",
        "Viterbi",
        "likelihood.message_passing.max_product",
        "coupled E and M steps",
        T + "likelihood/test_message_passing.py"
        "::test_max_product_decodes_the_mode_of_the_coupled_e_step_s_path_law",
        cost=Cost.PASS,
    ),
    Rung(
        "hmm",
        "relaxed MAP",
        "learn.relaxed.optimize",
        "Viterbi",
        T + "learn/test_learn_relaxed.py"
        "::test_the_relaxed_optimum_of_the_hmm_is_the_viterbi_path",
        cost=Cost.GRADIENTS,
    ),
    Rung(
        "hmm",
        "Baum--Welch",
        "opt.hmm.baum_welch",
        "path enumeration",
        T + "opt/test_opt_hmm.py"
        "::test_baum_welch_reaches_the_enumerated_path_evidence_and_its_fixed_point",
        cost=Cost.ITERATIONS,
    ),
    Rung(
        "hmm",
        "Baum--Welch",
        "opt.hmm.baum_welch",
        None,
        T + "opt/test_opt_hmm.py"
        "::test_baum_welch_ascends_and_settles_on_the_re_estimation_equations",
        cost=Cost.ITERATIONS,
    ),
    Rung(
        "hmm",
        "ragged HMM",
        "likelihood.ragged_rust.posteriors",
        "forward / forward-backward",
        T + "opt/test_ragged_hmm.py"
        "::test_equal_lengths_reproduce_the_conserved_route_bitwise",
        cost=Cost.PASS,
    ),
    Rung(
        "hmm",
        "RL over paths",
        "learn.hmm.optimum",
        "path enumeration",
        T
        + "learn/test_learn_hmm.py::test_hill_climbing_reaches_the_enumerated_optimum",
        cost=Cost.TRAINED,
    ),
    Rung(
        "hmm",
        "coupled model, exact",
        "likelihood.spatio_sequential.enumerate_spatio_sequential",
        None,
        T + "likelihood/test_spatio_sequential.py"
        "::test_the_enumerated_evidence_equals_the_per_class_forward_route",
        cost=Cost.EXACT,
    ),
    Rung(
        "hmm",
        "coupled E and M steps",
        "likelihood.spatio_sequential.class_posteriors",
        "coupled model, exact",
        T + "likelihood/test_spatio_sequential_fit.py"
        "::test_the_class_e_step_is_the_conditional_posterior_by_enumeration",
        cost=Cost.ITERATIONS,
    ),
    Rung(
        "hmm",
        "coupled, Rust",
        "likelihood.spatio_sequential_rust.class_posteriors",
        "coupled E and M steps",
        T + "likelihood/test_spatio_sequential_rust.py"
        "::test_the_rust_e_step_and_field_match_the_numpy_oracle",
        cost=Cost.ITERATIONS,
    ),
    Rung(
        "hmm",
        "coupled label search",
        "search.spatio_sequential.label_step",
        "coupled model, exact",
        T + "search/test_spatio_sequential_fit.py"
        "::test_the_label_step_reaches_the_enumerated_map_from_the_planted_labels",
        cost=Cost.SEVERAL,
    ),
    Rung(
        "hmm",
        "coupled Gibbs",
        "sample.gibbs.sample_factor_graph",
        "coupled model, exact",
        T + "sample/test_gibbs.py"
        "::test_the_coupled_sweep_draws_labellings_from_the_enumerated_joint_law",
        cost=Cost.SWEEPS,
    ),
    # --- codes -------------------------------------------------------------
    Rung(
        "codes",
        "brute-force ML",
        "likelihood.ldpc.enumerate_codewords",
        None,
        T + "likelihood/test_ldpc.py"
        "::test_the_enumeration_oracle_on_a_single_parity_check_by_hand",
        cost=Cost.EXACT,
    ),
    Rung(
        "codes",
        "BCJR",
        "likelihood.convolutional.bcjr",
        "brute-force ML",
        T + "likelihood/test_convolutional.py"
        "::test_bcjr_posteriors_are_the_exact_bitwise_map",
        cost=Cost.PASS,
    ),
    Rung(
        "codes",
        "BCJR, Rust",
        "likelihood.convolutional_rust.bcjr",
        "BCJR",
        T + "likelihood/test_convolutional_rust.py"
        "::test_the_rust_pass_is_the_numpy_oracle_bitwise_on_the_declared_registers",
        cost=Cost.PASS,
    ),
    Rung(
        "codes",
        "Viterbi on the trellis",
        "likelihood.convolutional.viterbi",
        "brute-force ML",
        T + "likelihood/test_convolutional.py"
        "::test_viterbi_returns_the_maximum_likelihood_message_with_a_pinned_margin",
        cost=Cost.PASS,
    ),
    Rung(
        "codes",
        "turbo",
        "likelihood.turbo.decode_turbo",
        "brute-force ML",
        T + "likelihood/test_turbo.py"
        "::test_the_joint_posterior_is_the_enumerated_one_where_the_two_chains_agree",
        cost=Cost.ITERATIONS,
    ),
    Rung(
        "codes",
        "turbo",
        "likelihood.turbo.decode_turbo",
        "BCJR",
        T + "likelihood/test_turbo.py"
        "::test_the_turbo_posterior_is_bcjr_on_the_first_constituent_alone",
        cost=Cost.ITERATIONS,
    ),
    Rung(
        "codes",
        "LDPC sum-product / min-sum",
        "likelihood.ldpc.decode",
        "brute-force ML",
        T + "likelihood/test_ldpc.py::test_sum_product_is_exact_on_a_cycle_free_code",
        cost=Cost.ITERATIONS,
    ),
    Rung(
        "codes",
        "bicycle, loopy",
        "sim.ldpc.bicycle_code",
        "brute-force ML",
        T + "likelihood/test_bicycle.py"
        "::test_the_exact_decodings_bound_belief_propagation",
        cost=Cost.PASS,
    ),
    Rung(
        "codes",
        "CSS / syndrome",
        "likelihood.css.decode_syndrome",
        "brute-force ML",
        T + "likelihood/test_css.py"
        "::test_summing_a_coset_beats_maximizing_over_one_error",
        cost=Cost.ITERATIONS,
    ),
    Rung(
        "codes",
        "CSS / syndrome",
        "likelihood.css.decode_syndrome",
        "LDPC sum-product / min-sum",
        T + "likelihood/test_css.py"
        "::test_the_syndrome_decode_is_belief_propagation_on_the_component_code",
        cost=Cost.ITERATIONS,
    ),
    Rung(
        "codes",
        "polar SC / SCL",
        "likelihood.polar.decode_sc",
        None,
        T + "likelihood/test_polar.py"
        "::test_successive_cancellation_is_the_reference_implementation_bitwise",
        cost=Cost.PASS,
    ),
    Rung(
        "codes",
        "CRC-aided list decoding",
        "likelihood.polar.decode_scl",
        "polar SC / SCL",
        T + "likelihood/test_polar.py"
        "::test_crc_aided_exhaustive_list_is_maximum_likelihood_over_the_outer_code",
        cost=Cost.PASS,
    ),
    Rung(
        "codes",
        "polar SC / SCL",
        "likelihood.polar.decode_sc",
        "LDPC sum-product / min-sum",
        T + "likelihood/test_polar.py"
        "::test_successive_cancellation_against_min_sum_on_the_same_parity_check",
        cost=Cost.PASS,
    ),
    Rung(
        "codes",
        "polar construction",
        "sim.polar.polar_information_set",
        None,
        T + "sim/test_polar.py"
        "::test_the_polar_and_reed_muller_rules_coincide_at_eight_and_part_at_sixteen",
        cost=Cost.PASS,
    ),
    Rung(
        "codes",
        "elementary codes",
        "sim.elementary_codes.single_parity_check",
        None,
        T + "sim/test_elementary_codes.py"
        "::test_the_single_parity_check_posterior_is_the_tanh_rule",
        cost=Cost.PASS,
    ),
    Rung(
        "codes",
        "hamming_correct, golay_code",
        "likelihood.algebraic.hamming_correct",
        "brute-force ML",
        T + "sim/test_elementary_codes.py"
        "::test_syndrome_correction_is_the_nearest_codeword_enumeration_returns",
        cost=Cost.PASS,
    ),
    Rung(
        "codes",
        "Reed--Solomon",
        "likelihood.algebraic.decode",
        None,
        T + "sim/test_reed_solomon.py"
        "::test_reed_solomon_meets_the_singleton_bound_with_equality",
        cost=Cost.PASS,
    ),
    Rung(
        "codes",
        "Reed--Solomon",
        "likelihood.algebraic.decode",
        "brute-force ML",
        T + "sim/test_reed_solomon.py"
        "::test_the_algebraic_decode_is_the_nearest_codeword_enumeration_returns",
        cost=Cost.PASS,
    ),
    # --- mixtures ----------------------------------------------------------
    # The foot and the rung above it name one test: it computes the 65,536
    # whole assignments and the factorized evidence and compares them, and the
    # survey names no other `oracle` test over the enumeration.
    Rung(
        "mixture",
        "assignment enumeration",
        "likelihood.mixture_assignments.enumerate_mixture_assignments",
        None,
        T + "opt/test_opt_mixture.py"
        "::test_the_evidence_and_the_e_step_match_the_enumerated_assignments",
        cost=Cost.EXACT,
    ),
    Rung(
        "mixture",
        "GMM EM",
        "opt.mixture.expectation_maximization",
        "assignment enumeration",
        T + "opt/test_opt_mixture.py"
        "::test_the_evidence_and_the_e_step_match_the_enumerated_assignments",
        cost=Cost.ITERATIONS,
    ),
    Rung(
        "mixture",
        "emission-mixture EM",
        "opt.emission_mixture.expectation_maximization",
        "assignment enumeration",
        T + "opt/test_opt_emission_mixture.py"
        "::test_the_responsibilities_are_the_enumerated_posterior",
        cost=Cost.ITERATIONS,
    ),
    Rung(
        "mixture",
        "k-means++ seeding",
        "opt.mixture.kmeans_plus_plus",
        None,
        T + "opt/test_opt_mixture_seeding.py"
        "::test_the_shipped_rule_is_k_means_plus_plus_on_a_gaussian",
        cost=Cost.PASS,
    ),
    Rung(
        "mixture",
        "optimal clustering cost",
        "opt.mixture.optimal_clustering_cost",
        "assignment enumeration",
        T + "opt/test_opt_mixture.py"
        "::test_the_optimal_clustering_cost_is_the_minimum_over_the_enumerated_assignments",
        cost=Cost.EXACT,
    ),
    Rung(
        "mixture",
        "projection seeding",
        "search.projection.euclidean_seeding",
        "k-means++ seeding",
        T + "search/test_projection_seeding.py"
        "::test_euclidean_seeding_is_one_dimensional_kmeans_plus_plus_on_a_flat_channel",
        cost=Cost.PASS,
    ),
    Rung(
        "mixture",
        "emission_seeding, data_seeding",
        "search.projection.emission_seeding",
        "k-means++ seeding",
        T + "search/test_projection_seeding.py"
        "::test_the_non_euclidean_seedings_draw_the_law_of_the_metric_they_declare",
        cost=Cost.PASS,
    ),
    Rung(
        "mixture",
        "HMC",
        "sample.hmc.sample",
        None,
        T + "sample/test_opt_hmc.py::test_the_chain_recovers_an_analytic_gaussian",
        cost=Cost.GRADIENTS,
    ),
    Rung(
        "mixture",
        "HMC",
        "sample.hmc.sample",
        "assignment enumeration",
        T + "sample/test_opt_hmc.py"
        "::test_the_chain_recovers_the_enumerated_assignment_posterior_of_a_mixture",
        cost=Cost.GRADIENTS,
    ),
    Rung(
        "mixture",
        "MALA",
        "sample.langevin.mala",
        None,
        T + "sample/test_opt_langevin.py"
        "::test_the_langevin_chain_recovers_an_analytic_gaussian",
        cost=Cost.GRADIENTS,
    ),
    Rung(
        "mixture",
        "MALA",
        "sample.langevin.mala",
        "assignment enumeration",
        T + "sample/test_opt_langevin.py"
        "::test_the_langevin_chain_recovers_the_enumerated_assignment_posterior",
        cost=Cost.GRADIENTS,
    ),
    Rung(
        "mixture",
        "slice sampling",
        "sample.slice.slice_sample",
        None,
        T
        + "sample/test_opt_slice.py::test_the_slice_chain_recovers_an_analytic_gaussian",
        cost=Cost.EVALUATIONS,
    ),
    Rung(
        "mixture",
        "slice sampling",
        "sample.slice.slice_sample",
        "assignment enumeration",
        T + "sample/test_opt_slice.py"
        "::test_the_slice_chain_recovers_the_enumerated_assignment_posterior",
        cost=Cost.EVALUATIONS,
    ),
    Rung(
        "mixture",
        "FromChain, FromAnnealing, FromTempering",
        "sample.initialize.FromChain",
        None,
        T + "opt/test_opt_initialize.py"
        "::test_the_sampled_starts_are_their_runs_own_records_and_leave_the_cell_descent_cannot",
        cost=Cost.SWEEPS,
    ),
)


def rungs(problem: str) -> tuple[Rung, ...]:
    """One problem's ladder, in declaration order.

    Raises
    ------
    ValueError
        If ``problem`` is not one of :data:`PROBLEMS`. A typo returning an
        empty ladder would read as a problem with no rungs.
    """
    if problem not in PROBLEMS:
        msg = f"unknown problem {problem!r}; the ladders are {PROBLEMS}"
        raise ValueError(msg)
    return tuple(rung for rung in LADDER if rung.problem == problem)


def pinned() -> tuple[Rung, ...]:
    """The rungs a test pins to the rung below."""
    return tuple(rung for rung in LADDER if rung.test is not None)


def unpinned() -> tuple[Rung, ...]:
    """The rungs no test pins; each carries the ticket that will."""
    return tuple(rung for rung in LADDER if rung.test is None)
