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

This is a ladder and not a census: where the survey (issue #734, first
comment) lists several tests for one rung, the one recorded here is the one
pinning that rung to the rung below, and the rest are recorded nowhere. Two
conventions follow from that:

* `below` is `None` where nothing in the ladder is under the rung --- the
  exact end of a ladder, or a referee outside it: a closed form, a second
  implementation, a framework. The test's own name states which.
* one rung appears once per rung it is pinned against, so a method pinned at
  the exact end and wanted against a cheaper rung is two rows, one of them a
  ticket.

Infrastructure, not science: the callables are strings this module never
imports, read the way `infra/problems_tables.py` reads the fixture registry
(`infra/CLAUDE.md`). The guard resolves them; nothing here does.

`Rung` here is a rung of the oracle ladder. `search.ground_state.Rung`, which
predates it, is an instance at a size.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The five ladders, in the order the survey tables run.
PROBLEMS = ("potts", "tree", "hmm", "codes", "mixture")

#: The issue carrying the rungs no test pins, one bullet per rung.
LADDER_TICKET = 734


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
    ticket : int or None
        The issue carrying the missing pin, set exactly where `test` is None.
    """

    problem: str
    name: str
    callable: str
    below: str | None
    test: str | None
    ticket: int | None = None


T = "tests/regression/"

#: Every rung of every ladder: the pinned ones first, per problem, in the
#: order the survey's table runs, then the rungs issue #734 carries.
LADDER: tuple[Rung, ...] = (
    # --- Potts / lattice ---------------------------------------------------
    Rung(
        "potts",
        "enumeration",
        "likelihood.potts.enumerate_potts",
        None,
        T + "search/test_search_support.py"
        "::test_the_enumerated_labelling_weight_is_enumerate_potts_s_boltzmann_weight_at_beta_one",
    ),
    Rung(
        "potts",
        "transfer matrix",
        "likelihood.potts.strip_log_partition",
        "enumeration",
        T + "likelihood/test_potts_exact.py"
        "::test_the_transfer_matrix_reproduces_exhaustive_enumeration",
    ),
    Rung(
        "potts",
        "transfer matrix",
        "likelihood.potts.strip_log_partition",
        "sum-product / BP",
        T + "likelihood/test_potts_exact.py"
        "::test_the_transfer_matrix_is_sum_product_on_the_strip_that_is_a_tree",
    ),
    Rung(
        "potts",
        "sum-product / BP",
        "likelihood.message_passing.sum_product",
        "enumeration",
        T + "likelihood/test_message_passing.py"
        "::test_sum_product_on_the_potts_tree_is_the_enumeration",
    ),
    Rung(
        "potts",
        "flooding schedule",
        "likelihood.belief_propagation.belief_propagation",
        "sum-product / BP",
        T + "likelihood/test_message_passing.py"
        "::test_flooding_on_the_loopy_lattice_is_belief_propagation",
    ),
    Rung(
        "potts",
        "Kikuchi / region graph",
        "sandbox.region_graph.kikuchi_free_energy",
        "enumeration",
        T + "sandbox/test_region_graph.py"
        "::test_the_free_energy_is_the_exact_log_partition_on_a_tree",
    ),
    Rung(
        "potts",
        "dual / LP bound",
        "search.tightening.dual_bound",
        "enumeration",
        T + "search/test_tightening.py"
        "::test_the_bound_never_exceeds_the_enumerated_ground_state",
    ),
    Rung(
        "potts",
        "dual / LP bound",
        "search.tightening.dual_bound",
        "sum-product / BP",
        T + "search/test_tightening.py"
        "::test_the_dual_bound_is_the_zero_temperature_belief_propagation_energy",
    ),
    Rung(
        "potts",
        "variational bounds",
        "likelihood.surrogate.mean_field_log_partition",
        "enumeration",
        T + "likelihood/test_surrogate.py"
        "::test_mean_field_and_spanning_tree_bounds_sandwich_log_z",
    ),
    Rung(
        "potts",
        "exact cut / max-flow",
        "search.maxflow.ising_ground_state",
        "enumeration",
        T + "search/test_maxflow.py"
        "::test_the_cut_finds_the_enumerated_minimum_with_a_per_node_field",
    ),
    Rung(
        "potts",
        "declined max-flow kernels",
        "sandbox.maxflow_declined.min_cut",
        "exact cut / max-flow",
        T + "sandbox/test_maxflow_declined.py"
        "::test_every_declined_kernel_returns_the_python_cut_on_seeded_networks",
    ),
    Rung(
        "potts",
        "max-cut (SDP)",
        "search.max_cut.goemans_williamson",
        "enumeration",
        T + "search/test_max_cut.py"
        "::test_the_rounded_cut_reaches_the_enumerated_optimum_on_a_lattice",
    ),
    Rung(
        "potts",
        "max-cut (SDP)",
        "search.max_cut.goemans_williamson",
        "alpha-expansion",
        T + "search/test_max_cut.py"
        "::test_the_rounded_cut_is_the_gauged_alpha_expansion_optimum_at_two_labels",
    ),
    Rung(
        "potts",
        "alpha-expansion",
        "search.alpha_expansion.alpha_expansion",
        "exact cut / max-flow",
        T + "search/test_alpha_expansion.py"
        "::test_two_labels_reproduce_the_exact_minimum_cut",
    ),
    Rung(
        "potts",
        "alpha-expansion backends",
        "search.alpha_expansion.alpha_expansion",
        "alpha-expansion",
        T + "search/test_alpha_expansion.py"
        "::test_the_rust_cut_reproduces_the_python_expansion",
    ),
    Rung(
        "potts",
        "ICM",
        "search.alpha_expansion.iterated_conditional_modes",
        "exact cut / max-flow",
        T + "search/test_potts_sizing.py"
        "::test_the_zero_field_optimum_is_a_closed_form_three_ways",
    ),
    Rung(
        "potts",
        "Gibbs / heat bath",
        "search.gibbs.gibbs_sweep",
        "enumeration",
        T + "sim/test_potts_simulate.py"
        "::test_gibbs_sampling_matches_brute_force_enumeration_on_a_loopy_lattice",
    ),
    Rung(
        "potts",
        "Potts MCMC, Wolff, Swendsen--Wang",
        "search.potts_mcmc.sample_potts",
        "enumeration",
        T + "search/test_potts_mcmc.py"
        "::test_the_chain_is_drawn_from_the_exact_boltzmann_distribution",
    ),
    Rung(
        "potts",
        "cluster moves as RL arms",
        "search.potts_keyed.cluster_moves",
        "Potts MCMC, Wolff, Swendsen--Wang",
        T + "learn/test_cluster_arms.py"
        "::test_a_wolff_step_is_potts_mcmcs_own_sweep_bitwise",
    ),
    Rung(
        "potts",
        "annealing",
        "search.potts_mcmc.anneal_potts",
        None,
        T + "search/test_potts_mcmc.py"
        "::test_annealing_reaches_the_closed_form_ground_energy_where_descent_does_not",
    ),
    Rung(
        "potts",
        "parallel tempering",
        "search.potts_mcmc.parallel_tempering",
        "annealing",
        T + "search/test_potts_mcmc.py"
        "::test_tempering_reaches_the_ground_energy_annealing_reaches",
    ),
    Rung(
        "potts",
        "learned policy",
        "learn.potts_nd.PottsNDEnvironment",
        "exact cut / max-flow",
        T + "learn/test_potts_nd.py::test_two_labels_agree_with_the_exact_cut",
    ),
    Rung(
        "potts",
        "learned lattice surrogate",
        "search.surrogate.lattice_examples",
        "enumeration",
        T + "search/test_search_lattice_surrogate.py"
        "::test_a_surrogate_learns_the_gap_above_the_mean_field_bound_at_nine_sites",
    ),
    Rung(
        "potts",
        "run_tempering, run_greedy, run_max_product",
        "search.ground_state.run_tempering",
        None,
        T + "search/test_ground_state.py"
        "::test_the_runners_record_the_energy_their_kernels_return",
    ),
    # --- trees / phylogenetics ---------------------------------------------
    Rung(
        "tree",
        "projection",
        "search.projection.project",
        None,
        T + "search/test_projection_seeding.py"
        "::test_the_projected_draw_is_the_closed_form_mixture_of_the_flattened_families",
    ),
    Rung(
        "tree",
        "topology enumeration",
        "search.topology.enumerate_topologies",
        None,
        T + "search/test_search_exhaustive.py"
        "::test_enumeration_produces_every_topology_exactly_once",
    ),
    Rung(
        "tree",
        "topology equality",
        "search.topology.robinson_foulds",
        None,
        T + "search/test_search_topology_rustworkx.py"
        "::test_two_topologies_our_code_calls_equal_are_isomorphic",
    ),
    Rung(
        "tree",
        "parsimony, Fitch",
        "likelihood.parsimony.fitch_score",
        None,
        T + "likelihood/test_likelihood_parsimony.py"
        "::test_fitch_matches_exhaustive_enumeration_over_internal_labellings",
    ),
    Rung(
        "tree",
        "parsimony, Sankoff",
        "likelihood.parsimony.sankoff_score",
        "parsimony, Fitch",
        T + "likelihood/test_likelihood_parsimony.py"
        "::test_sankoff_with_the_unit_matrix_is_fitch_on_every_five_taxon_topology",
    ),
    Rung(
        "tree",
        "parsimony against likelihood",
        "likelihood.parsimony.fitch_score",
        "pruning, NumPy",
        T + "likelihood/test_likelihood_parsimony.py"
        "::test_the_short_branch_likelihood_ranks_the_topologies_as_the_fitch_score_does",
    ),
    Rung(
        "tree",
        "pruning, NumPy",
        "likelihood.pruning.log_likelihood",
        None,
        T + "likelihood/test_likelihood_pruning.py::test_pruning_matches_brute_force",
    ),
    Rung(
        "tree",
        "pruning, Rust",
        "likelihood.pruning_rust.log_likelihood",
        "pruning, NumPy",
        T + "likelihood/test_pruning_rust.py::test_rust_matches_numpy_oracle",
    ),
    Rung(
        "tree",
        "pruning, Torch",
        "likelihood.pruning_torch.log_likelihood",
        "pruning, NumPy",
        T + "likelihood/test_pruning_torch.py::test_torch_matches_numpy_oracle",
    ),
    Rung(
        "tree",
        "pruning, analytic gradient",
        "likelihood.pruning_analytic.log_likelihood",
        "pruning, Torch",
        T + "likelihood/test_pruning_gradient.py"
        "::test_every_route_agrees_with_the_taped_gradient",
    ),
    Rung(
        "tree",
        "Hadamard / spectral",
        "likelihood.hadamard.hadamard_conjugation",
        None,
        T + "likelihood/test_likelihood_hadamard.py"
        "::test_the_conjugation_returns_the_true_split_weights_on_the_exact_spectrum",
    ),
    Rung(
        "tree",
        "distance start (NJ)",
        "search.neighbor_joining.neighbor_joining",
        None,
        T + "search/test_search_neighbor_joining_scipy.py"
        "::test_the_two_return_the_same_tree_on_an_ultrametric_matrix",
    ),
    Rung(
        "tree",
        "distance start (NJ)",
        "search.neighbor_joining.neighbor_joining",
        "topology enumeration",
        T + "search/test_neighbor_joining.py"
        "::test_the_joined_tree_is_the_least_squares_optimum_over_the_enumerated_topologies",
    ),
    Rung(
        "tree",
        "tropical / Grassmannian",
        "sandbox.tropical.resolutions",
        None,
        T + "sandbox/test_sandbox_tropical.py"
        "::test_the_combinatorial_resolution_is_the_tropical_plucker_argmin",
    ),
    Rung(
        "tree",
        "tree search (NNI/SPR)",
        "search.infer.infer",
        "topology enumeration",
        T + "search/test_search_exhaustive.py"
        "::test_hill_climbing_reaches_the_enumerated_maximum",
    ),
    Rung(
        "tree",
        "cheaper searches",
        "search.infer.infer",
        "topology enumeration",
        T + "search/test_search_exhaustive.py"
        "::test_the_cheaper_searches_reach_the_enumerated_maximum",
    ),
    Rung(
        "tree",
        "analytic surrogates",
        "likelihood.surrogate.PlugInLikelihood",
        "pruning, NumPy",
        T + "likelihood/test_surrogate.py"
        "::test_plug_in_bound_is_below_every_fitted_likelihood",
    ),
    Rung(
        "tree",
        "surrogate-ranked search",
        "search.surrogate.LearnedTreeSurrogate",
        "tree search (NNI/SPR)",
        T + "search/test_search_surrogate.py"
        "::test_surrogate_ranked_search_reaches_what_the_full_search_reaches",
    ),
    Rung(
        "tree",
        "surrogate-ranked search",
        "search.surrogate.LearnedTreeSurrogate",
        "analytic surrogates",
        T + "search/test_search_surrogate.py"
        "::test_the_learned_surrogate_ranks_the_topologies_the_plug_in_bound_ranks",
    ),
    Rung(
        "tree",
        "learned GNN surrogate",
        "learn.surrogate.GraphSurrogate",
        None,
        T + "learn/test_learn_surrogate_pyg.py"
        "::test_pyg_s_gin_reproduces_the_graph_surrogate_on_tied_weights",
    ),
    Rung(
        "tree",
        "RL policy / PPO",
        "search.rl.TreeEnvironment",
        "topology enumeration",
        T
        + "search/test_search_rl.py::test_greedy_search_reaches_the_enumerated_optimum",
    ),
    Rung(
        "tree",
        "support / tempering over topologies",
        "search.support.neighbourhood_support",
        "topology enumeration",
        T + "search/test_search_support.py"
        "::test_the_nni_neighbourhood_of_four_taxa_is_the_whole_space_so_the_two_supports_agree",
    ),
    Rung(
        "tree",
        "Gibbs / annealing over topologies",
        "search.gibbs.topology_step",
        "topology enumeration",
        T + "search/test_gibbs.py"
        "::test_the_topology_move_at_temperature_one_samples_the_enumerated_flat_prior_weight",
    ),
    # --- HMM / spatio-sequential -------------------------------------------
    Rung(
        "hmm",
        "path enumeration",
        "likelihood.hmm_paths.enumerate_hidden_paths",
        None,
        T + "likelihood/test_hmm_paths.py"
        "::test_a_marginal_is_the_summed_joint_over_paths_through_that_state",
    ),
    Rung(
        "hmm",
        "forward / forward-backward",
        "likelihood.forward_backward.forward_backward",
        "path enumeration",
        T + "opt/test_opt_hmm.py::test_forward_matches_brute_force_path_enumeration",
    ),
    Rung(
        "hmm",
        "forward via sum-product",
        "likelihood.message_passing.sum_product",
        "forward / forward-backward",
        T + "likelihood/test_message_passing.py"
        "::test_the_tree_schedule_on_a_deep_chain_is_the_forward_recursion",
    ),
    Rung(
        "hmm",
        "Viterbi",
        "likelihood.message_passing.max_product",
        "path enumeration",
        T
        + "likelihood/test_message_passing.py::test_max_product_on_the_chain_is_viterbi",
    ),
    Rung(
        "hmm",
        "relaxed MAP",
        "learn.relaxed.optimize",
        "Viterbi",
        T + "learn/test_learn_relaxed.py"
        "::test_the_relaxed_optimum_of_the_hmm_is_the_viterbi_path",
    ),
    Rung(
        "hmm",
        "Baum--Welch",
        "opt.hmm.baum_welch",
        "path enumeration",
        T + "opt/test_opt_hmm.py"
        "::test_baum_welch_reaches_the_enumerated_path_evidence_and_its_fixed_point",
    ),
    Rung(
        "hmm",
        "ragged HMM",
        "likelihood.ragged_rust.posteriors",
        "forward / forward-backward",
        T + "opt/test_ragged_hmm.py"
        "::test_equal_lengths_reproduce_the_conserved_route_bitwise",
    ),
    Rung(
        "hmm",
        "RL over paths",
        "learn.hmm.optimum",
        "path enumeration",
        T
        + "learn/test_learn_hmm.py::test_hill_climbing_reaches_the_enumerated_optimum",
    ),
    Rung(
        "hmm",
        "coupled model, exact",
        "likelihood.spatio_sequential.enumerate_spatio_sequential",
        None,
        T + "likelihood/test_spatio_sequential.py"
        "::test_the_enumerated_evidence_equals_the_per_class_forward_route",
    ),
    Rung(
        "hmm",
        "coupled E and M steps",
        "likelihood.spatio_sequential.class_posteriors",
        "coupled model, exact",
        T + "likelihood/test_spatio_sequential_fit.py"
        "::test_the_class_e_step_is_the_conditional_posterior_by_enumeration",
    ),
    Rung(
        "hmm",
        "coupled, Rust",
        "likelihood.spatio_sequential_rust.class_posteriors",
        "coupled E and M steps",
        T + "likelihood/test_spatio_sequential_rust.py"
        "::test_the_rust_e_step_and_field_match_the_numpy_oracle",
    ),
    Rung(
        "hmm",
        "coupled label search",
        "search.spatio_sequential.label_step",
        "coupled model, exact",
        T + "search/test_spatio_sequential_fit.py"
        "::test_the_label_step_reaches_the_enumerated_map_from_the_planted_labels",
    ),
    # --- codes -------------------------------------------------------------
    Rung(
        "codes",
        "brute-force ML",
        "likelihood.ldpc.enumerate_codewords",
        None,
        T + "likelihood/test_ldpc.py"
        "::test_the_enumeration_oracle_on_a_single_parity_check_by_hand",
    ),
    Rung(
        "codes",
        "BCJR",
        "likelihood.convolutional.bcjr",
        "brute-force ML",
        T + "likelihood/test_convolutional.py"
        "::test_bcjr_posteriors_are_the_exact_bitwise_map",
    ),
    Rung(
        "codes",
        "Viterbi on the trellis",
        "likelihood.convolutional.viterbi",
        "brute-force ML",
        T + "likelihood/test_convolutional.py"
        "::test_viterbi_returns_the_maximum_likelihood_message_with_a_pinned_margin",
    ),
    Rung(
        "codes",
        "turbo",
        "likelihood.turbo.decode_turbo",
        "brute-force ML",
        T + "likelihood/test_turbo.py"
        "::test_the_joint_posterior_is_the_enumerated_one_where_the_two_chains_agree",
    ),
    Rung(
        "codes",
        "LDPC sum-product / min-sum",
        "likelihood.ldpc.decode",
        "brute-force ML",
        T + "likelihood/test_ldpc.py::test_sum_product_is_exact_on_a_cycle_free_code",
    ),
    Rung(
        "codes",
        "bicycle, loopy",
        "sim.ldpc.bicycle_code",
        "brute-force ML",
        T + "likelihood/test_bicycle.py"
        "::test_the_exact_decodings_bound_belief_propagation",
    ),
    Rung(
        "codes",
        "CSS / syndrome",
        "likelihood.css.decode_syndrome",
        "brute-force ML",
        T + "likelihood/test_css.py"
        "::test_summing_a_coset_beats_maximizing_over_one_error",
    ),
    Rung(
        "codes",
        "polar SC / SCL",
        "sandbox.polar_decoding.decode_sc",
        None,
        T + "sandbox/test_polar_decoding.py"
        "::test_successive_cancellation_is_the_reference_implementation_bitwise",
    ),
    Rung(
        "codes",
        "polar construction",
        "sandbox.polar.polar_information_set",
        None,
        T + "sandbox/test_polar.py"
        "::test_the_polar_and_reed_muller_rules_coincide_at_eight_and_part_at_sixteen",
    ),
    Rung(
        "codes",
        "elementary codes",
        "sim.elementary_codes.single_parity_check",
        None,
        T + "sim/test_elementary_codes.py"
        "::test_the_single_parity_check_posterior_is_the_tanh_rule",
    ),
    Rung(
        "codes",
        "Reed--Solomon",
        "sim.reed_solomon.decode",
        None,
        T + "sim/test_reed_solomon.py"
        "::test_reed_solomon_meets_the_singleton_bound_with_equality",
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
    ),
    Rung(
        "mixture",
        "GMM EM",
        "opt.mixture.expectation_maximization",
        "assignment enumeration",
        T + "opt/test_opt_mixture.py"
        "::test_the_evidence_and_the_e_step_match_the_enumerated_assignments",
    ),
    Rung(
        "mixture",
        "emission-mixture EM",
        "opt.emission_mixture.expectation_maximization",
        "assignment enumeration",
        T + "opt/test_opt_emission_mixture.py"
        "::test_the_responsibilities_are_the_enumerated_posterior",
    ),
    Rung(
        "mixture",
        "k-means++ seeding",
        "opt.mixture.kmeans_plus_plus",
        None,
        T + "opt/test_opt_mixture_seeding.py"
        "::test_the_shipped_rule_is_k_means_plus_plus_on_a_gaussian",
    ),
    Rung(
        "mixture",
        "projection seeding",
        "search.projection.euclidean_seeding",
        "k-means++ seeding",
        T + "search/test_projection_seeding.py"
        "::test_euclidean_seeding_is_one_dimensional_kmeans_plus_plus_on_a_flat_channel",
    ),
    Rung(
        "mixture",
        "HMC",
        "opt.hmc.sample",
        None,
        T + "opt/test_opt_hmc.py::test_the_chain_recovers_an_analytic_gaussian",
    ),
    # --- the 13 rungs no test pins, one per bullet of issue #734 -----------
    Rung(
        "hmm",
        "path sampling",
        "likelihood.forward_backward.sample_path",
        "path enumeration",
        None,
        LADDER_TICKET,
    ),
    Rung(
        "hmm",
        "coupled Gibbs",
        "search.gibbs.sample_factor_graph",
        "coupled model, exact",
        None,
        LADDER_TICKET,
    ),
    Rung("hmm", "Baum--Welch", "opt.hmm.baum_welch", None, None, LADDER_TICKET),
    Rung(
        "hmm",
        "Viterbi",
        "likelihood.message_passing.max_product",
        "coupled E and M steps",
        None,
        LADDER_TICKET,
    ),
    Rung(
        "codes",
        "hamming_correct, golay_code",
        "sim.elementary_codes.hamming_correct",
        "brute-force ML",
        None,
        LADDER_TICKET,
    ),
    Rung(
        "codes",
        "Reed--Solomon",
        "sim.reed_solomon.decode",
        "brute-force ML",
        None,
        LADDER_TICKET,
    ),
    Rung(
        "codes",
        "polar SC / SCL",
        "sandbox.polar_decoding.decode_sc",
        "LDPC sum-product / min-sum",
        None,
        LADDER_TICKET,
    ),
    Rung(
        "codes",
        "CSS / syndrome",
        "likelihood.css.decode_syndrome",
        "LDPC sum-product / min-sum",
        None,
        LADDER_TICKET,
    ),
    Rung(
        "codes",
        "turbo",
        "likelihood.turbo.decode_turbo",
        "BCJR",
        None,
        LADDER_TICKET,
    ),
    Rung(
        "mixture",
        "HMC",
        "opt.hmc.sample",
        "assignment enumeration",
        None,
        LADDER_TICKET,
    ),
    Rung(
        "mixture",
        "FromChain, FromAnnealing, FromTempering",
        "opt.initialize.FromChain",
        None,
        None,
        LADDER_TICKET,
    ),
    Rung(
        "mixture",
        "optimal clustering cost",
        "opt.mixture.optimal_clustering_cost",
        "assignment enumeration",
        None,
        LADDER_TICKET,
    ),
    Rung(
        "mixture",
        "non-Euclidean projection seedings",
        "search.projection.data_seeding",
        "k-means++ seeding",
        None,
        LADDER_TICKET,
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
