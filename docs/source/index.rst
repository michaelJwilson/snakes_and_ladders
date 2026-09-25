sal
==================

.. automodule:: sal
   :members:

Submodules
----------

`python/sal/__init__.py` re-exports nothing beyond `sal`'s own top-level
utilities; each submodule is autodocumented on its own rather than through the
flat top-level namespace. Entries below cover every submodule that exists,
whether or not it has members yet.

.. automodule:: sal.numerics
   :members:

.. automodule:: sal.enumeration
   :members:

.. automodule:: sal.log
   :members:

.. automodule:: sal.bound
   :members:

.. automodule:: sal.cost
   :members:

.. automodule:: sal.emissions
   :members:

.. automodule:: sal.emissions.base
   :members: Values, FamilyT_co

.. automodule:: sal.emissions.families
   :members: COLLAPSE_EXPONENT

.. automodule:: sal.emissions.mstep
   :members:
   :exclude-members: identifiable_concentration_bound, identifiable_dispersion_bound

.. automodule:: sal.emissions.counts
   :members: lgamma_shifted

.. automodule:: sal.fixtures
   :members:

.. automodule:: sal.inputs
   :members:

.. automodule:: sal.incidence
   :members:

.. automodule:: sal.parallel
   :members:

.. automodule:: sal.ragged
   :members:

.. automodule:: sal.track
   :members:

.. automodule:: sal.likelihood.ragged_rust
   :members:

.. automodule:: sal.sim
   :members:

.. automodule:: sal.sim.tree
   :members:

.. automodule:: sal.sim.jc
   :members:

.. automodule:: sal.sim.gtr
   :members:

.. automodule:: sal.sim.params
   :members:

.. automodule:: sal.sim.fixtures
   :members:

.. automodule:: sal.sim.simulate
   :members:

.. automodule:: sal.sim.simulator
   :members:

.. automodule:: sal.sim.graph
   :members:

.. automodule:: sal.sim.factor_graph
   :members:

.. automodule:: sal.sim.spatio_sequential
   :members:

.. automodule:: sal.sim.count_pairs
   :members:

.. automodule:: sal.sim.count_pairs_rust
   :members:

.. automodule:: sal.sim.potts
   :members:

.. automodule:: sal.sim.potts_chain
   :members:

.. automodule:: sal.sim.hmm
   :members:

.. automodule:: sal.sim.mixture
   :members:

.. automodule:: sal.sim.emission_mixture
   :members:

.. automodule:: sal.sim.canonical
   :members:

.. automodule:: sal.sim.newick
   :members:

.. automodule:: sal.sim.elementary_codes
   :members:

.. automodule:: sal.sim.galois
   :members:

.. automodule:: sal.sim.capacity
   :members:

.. automodule:: sal.sim.reed_solomon
   :members:

.. automodule:: sal.sim.ldpc
   :members:

.. automodule:: sal.sim.polar
   :members:

.. automodule:: sal.sim.css
   :members:

.. automodule:: sal.sim.convolutional
   :members:

.. automodule:: sal.likelihood
   :members:

.. automodule:: sal.likelihood.objective
   :members:

.. automodule:: sal.likelihood.distance
   :members:

.. automodule:: sal.likelihood.hadamard
   :members:

.. automodule:: sal.likelihood.patterns
   :members:

.. automodule:: sal.likelihood.pruning
   :members:

.. automodule:: sal.likelihood.pruning_common
   :members:

.. automodule:: sal.likelihood.pruning_torch
   :members:

.. automodule:: sal.likelihood.pruning_jax
   :members:

.. automodule:: sal.likelihood.pruning_rust
   :members:

.. automodule:: sal.likelihood.pruning_analytic
   :members:

.. automodule:: sal.likelihood.brute_force
   :members:

.. automodule:: sal.likelihood.parsimony
   :members:

.. automodule:: sal.likelihood.device
   :members:

.. automodule:: sal.likelihood.potts
   :members:

.. automodule:: sal.likelihood.belief_propagation
   :members:

.. automodule:: sal.likelihood.schedule
   :members:

.. automodule:: sal.likelihood.message_passing
   :members:

.. automodule:: sal.likelihood.message_passing_reference
   :members:

.. automodule:: sal.likelihood.message_passing_rust
   :members:

.. automodule:: sal.likelihood.spatio_sequential
   :members:

.. automodule:: sal.likelihood.spatio_sequential_rust
   :members:

.. automodule:: sal.likelihood.forward_backward
   :members:

.. automodule:: sal.likelihood.hmm_paths
   :members:

.. automodule:: sal.likelihood.mixture_assignments
   :members:

.. automodule:: sal.likelihood.surrogate
   :members:

.. automodule:: sal.likelihood.blocks
   :members:

.. automodule:: sal.likelihood.features
   :members:

.. automodule:: sal.likelihood.ldpc
   :members:

.. automodule:: sal.likelihood.polar
   :members:

.. automodule:: sal.likelihood.algebraic
   :members:

.. automodule:: sal.likelihood.css
   :members:

.. automodule:: sal.likelihood.convolutional
   :members:

.. automodule:: sal.likelihood.convolutional_rust
   :members:

.. automodule:: sal.likelihood.turbo
   :members:

.. automodule:: sal.opt
   :members:

.. automodule:: sal.opt.objective
   :members:

.. automodule:: sal.opt.constrain
   :members:

.. automodule:: sal.opt.fit
   :members:

.. automodule:: sal.opt.em
   :members:

.. automodule:: sal.opt.initialize
   :members:

.. automodule:: sal.sample.initialize
   :members:

.. automodule:: sal.opt.budget
   :members:

.. automodule:: sal.opt.starts
   :members:

.. automodule:: sal.learn.failure
   :members:

.. automodule:: sal.opt.testfunctions
   :members:

.. automodule:: sal.opt.termination
   :members:

.. automodule:: sal.opt.potts
   :members:

.. automodule:: sal.opt.hmm
   :members:

.. automodule:: sal.opt.hmm.forward

.. automodule:: sal.opt.hmm.objectives

.. automodule:: sal.opt.hmm.estimation

.. automodule:: sal.opt.hmm_jax
   :members:

.. automodule:: sal.opt.mixture
   :members:

.. automodule:: sal.opt.emission_mixture
   :members:

.. automodule:: sal.opt.split_merge
   :members:

.. automodule:: sal.sample
   :members:

.. automodule:: sal.sample.accept
   :members:

.. automodule:: sal.sample.hmc
   :members:
   :exclude-members: Adaptation, Adapted, BLOCK, Chain, DUAL_AVERAGING_GAMMA, DUAL_AVERAGING_KAPPA, DUAL_AVERAGING_T0, Kernel, Transition, gradient_at, run_chain, run_compiled, start_point

.. automodule:: sal.sample.chain
   :members:

.. automodule:: sal.sample.expectation
   :members:

.. automodule:: sal.sample.langevin
   :members:

.. automodule:: sal.sample.metropolis
   :members:

.. automodule:: sal.sample.hmc_jax
   :members:

.. automodule:: sal.sample.declared
   :members:

.. automodule:: sal.sample.mixture_gibbs
   :members:

.. automodule:: sal.sample.relabel
   :members:

.. automodule:: sal.sample.slice
   :members:

.. automodule:: sal.sample.schedule
   :members:

.. automodule:: sal.sample.potts_mcmc
   :members:

.. automodule:: sal.sample.potts_mcmc.moves

.. automodule:: sal.sample.potts_mcmc.sweeps
   :members: GUARD, sweep_at, single_site_sweep, site_update, balanced_sweep_at, bond_probability

.. automodule:: sal.sample.potts_mcmc.chains

.. automodule:: sal.sample.gibbs
   :members:

.. automodule:: sal.sample.balanced
   :members:

.. automodule:: sal.sample.potts_keyed
   :members:

.. automodule:: sal.sample.tempered
   :members:

.. automodule:: sal.sample.annealed
   :members:

.. automodule:: sal.sample.statistics
   :members:

.. automodule:: sal.learn
   :members:

.. automodule:: sal.learn.environment
   :members:

.. automodule:: sal.learn.policy
   :members:

.. automodule:: sal.learn.relaxed
   :members:

.. automodule:: sal.learn.potts
   :members:

.. automodule:: sal.learn.potts_nd
   :members:

.. automodule:: sal.learn.keyed
   :members:

.. automodule:: sal.learn.hmm
   :members:

.. automodule:: sal.learn.rollout
   :members:

.. automodule:: sal.learn.reinforce
   :members:

.. automodule:: sal.learn.exact
   :members:

.. automodule:: sal.learn.arena
   :members:

.. automodule:: sal.learn.canonical
   :members:

.. automodule:: sal.learn.tabular
   :members:

.. automodule:: sal.learn.critic
   :members:

.. automodule:: sal.learn.actor_critic
   :members:

.. automodule:: sal.learn.ppo
   :members:

.. automodule:: sal.learn.planning
   :members:

.. automodule:: sal.learn.surrogate
   :members:

.. automodule:: sal.search
   :members:

.. automodule:: sal.sim.topology
   :members:

.. automodule:: sal.search.infer
   :members:

.. automodule:: sal.search.neighbor_joining
   :members:

.. automodule:: sal.search.initialize
   :members:

.. automodule:: sal.search.support
   :members:

.. automodule:: sal.search.spatio_sequential
   :members:

.. automodule:: sal.search.projection
   :members:

.. automodule:: sal.search.mixture_starts
   :members:

.. automodule:: sal.search.potts_starts
   :members:

.. automodule:: sal.search.ground_state
   :members:

.. automodule:: sal.search.cluster_moves
   :members:

.. automodule:: sal.learn.tree
   :members:

.. automodule:: sal.backend
   :members:

.. automodule:: sal.sample.kernels
   :members:

.. automodule:: sal.search.decoding
   :members:

.. automodule:: sal.search.tightening
   :members:

.. automodule:: sal.search.bifurcation
   :members:

.. automodule:: sal.search.alpha_expansion
   :members:

.. automodule:: sal.search.icm
   :members:

.. automodule:: sal.search.numba

.. automodule:: sal.search.numba.icm
   :members:

.. automodule:: sal.search.trws
   :members:

.. automodule:: sal.search.trws.numba
   :members:

.. automodule:: sal.search.max_cut
   :members:

.. automodule:: sal.search.maxflow
   :members:

.. automodule:: sal.search.maxflow_rust
   :members:

.. automodule:: sal.learn.ranking
   :members:

.. automodule:: sal.sandbox
   :members:

.. automodule:: sal.sandbox.annealed_em
   :members:

.. automodule:: sal.sandbox.tropical
   :members:

.. automodule:: sal.sandbox.pruning_burn
   :members:

.. Its two classes carry the live module's names by design -- it is the same
   code, frozen before the covariate -- so registering them as cross-reference
   targets makes every bare `:class:` reference to either ambiguous. It is a
   referee, not API: documented, and not something to link to.
.. automodule:: sal.sandbox.rectangular_hmm
   :members:

.. automodule:: sal.sandbox.count_emissions
   :members:
   :no-index:

.. automodule:: sal.sandbox.circulant_schedule
   :members:

.. automodule:: sal.sandbox.maxflow_declined
   :members:

.. automodule:: sal.sandbox.region_graph
   :members:

.. automodule:: sal.sandbox.polar_reference
   :members:

.. automodule:: sal.validation
   :members:

.. automodule:: sal.validation.runner
   :members:

.. automodule:: sal.validation.protocol
   :members:

.. automodule:: sal.validation.scripts
   :members:

.. automodule:: sal.validation.scripts.selftest
   :members:

.. automodule:: sal.validation.scripts.package
   :members:

.. automodule:: sal.validation.pymaxflow
   :members:

.. automodule:: sal.validation.scripts.pymaxflow
   :members:

.. automodule:: sal.validation.gco
   :members:

.. automodule:: sal.validation.scripts.gco
   :members:

.. automodule:: sal.validation.hmmlearn
   :members:

.. automodule:: sal.validation.scripts.hmmlearn
   :members:

.. automodule:: sal.validation.scikit_learn
   :members:

.. automodule:: sal.validation.scripts.scikit_learn
   :members:

.. automodule:: sal.validation.blackjax
   :members:

.. automodule:: sal.validation.gaussian
   :members:

.. automodule:: sal.validation.rustworkx
   :members:

.. automodule:: sal.validation.scripts.rustworkx
   :members:

.. automodule:: sal.validation.jax
   :members:

.. automodule:: sal.validation.scripts.jax
   :members:

.. automodule:: sal.validation.scripts.blackjax
   :members:

.. automodule:: sal.validation.gymnasium
   :members:

.. automodule:: sal.validation.scripts.gymnasium
   :members:

.. automodule:: sal.validation.torchrl
   :members:

.. automodule:: sal.validation.scripts.torchrl
   :members:

.. automodule:: sal.validation.torch_geometric
   :members:

.. automodule:: sal.validation.scripts.torch_geometric
   :members:

.. automodule:: sal.validation.highs
   :members:

.. automodule:: sal.validation.scripts.highs
   :members:

.. automodule:: sal.qa
   :members:

.. automodule:: sal.qa.figure
   :members:

.. automodule:: sal.qa.layout
   :members:

.. automodule:: sal.qa.style
   :members:

.. automodule:: sal.qa.runner
   :members:

.. automodule:: sal.qa.manifest
   :members:

.. automodule:: sal.qa.build
   :members:

.. automodule:: sal.qa.sim_tree
   :members:

.. automodule:: sal.qa.sim_example
   :members:

.. automodule:: sal.qa.sim_problem_sizes
   :members:

.. automodule:: sal.qa.likelihood_footprint
   :members:

.. automodule:: sal.qa.backend_agreement
   :members:

.. automodule:: sal.qa.opt_recovery
   :members:

.. automodule:: sal.qa.opt_coverage
   :members:

.. automodule:: sal.qa.opt_branch_recovery
   :members:

.. automodule:: sal.qa.rl_tree_policy
   :members:

.. automodule:: sal.qa.topology_accuracy
   :members:

.. automodule:: sal.qa.opt_model_recovery
   :members:

.. automodule:: sal.qa.search_trajectory
   :members:

.. automodule:: sal.qa.search_topologies
   :members:

.. automodule:: sal.qa.rl_reward_surface
   :members:

.. automodule:: sal.qa.parsimony_zones
   :members:

.. automodule:: sal.qa.tropical_relaxation
   :members:

.. automodule:: sal.qa.frustrated_lattices
   :members:

.. automodule:: sal.qa.mixture_seeding
   :members:

.. automodule:: sal.qa.starts
   :members:

.. automodule:: sal.qa.optimizer_landscapes
   :members:

.. automodule:: sal.qa.tanner_graph
   :members:

.. automodule:: sal.qa.turbo_waterfall
   :members:

.. automodule:: sal.qa.coupled_labelling
   :members:

.. automodule:: sal.qa.forney
   :members:

.. automodule:: sal.qa.starts_table
   :members:

.. automodule:: sal.qa.hmc_warmup
   :members:

.. automodule:: sal.qa.potts_schedule
   :members:

.. automodule:: sal.qa.potts_clusters
   :members:

.. automodule:: sal.scripts
   :members:

.. automodule:: sal.scripts.run_snakes_and_ladders
   :members:
