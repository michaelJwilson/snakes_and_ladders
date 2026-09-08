# Seams

The abstractions the package has, read from the package by
`infra/seams_survey.py` (issue #400): every `Protocol` it declares and every
data contract three or more modules share, with the classes that satisfy each
protocol by its members, the modules that consume it, and the problem classes
of `PROBLEMS.md` it reaches. The rule it is read against is in `DEV.md`
(Core Development Standards): a seam earns its place with 3 or more
consuming modules. Do not edit by hand -- run
`uv run python infra/seams_survey.py --write`.

## Protocols (11)

| Seam | Home | Members | Implementers | Consumers | Problems reached | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| `Objective` | `opt.objective` | constrain, initial, theta_from | `likelihood.objective.BranchLengthObjective`, `likelihood.objective.SubstitutionModelObjective`, `opt.hmc.WithGaussianPrior`, `opt.hmm.BetaBinomialHmmObjective`, `opt.hmm.BinomialHmmObjective`, `opt.hmm.GaussianHmmObjective`, `opt.hmm.HmmObjective`, `opt.hmm.NegativeBinomialHmmObjective`, `opt.hmm.PoissonHmmObjective`, `opt.mixture.GaussianMixtureObjective`, `opt.potts.PottsLatticeObjective`, `opt.potts.PottsObjective`, `opt.testfunctions.Himmelblau`, `opt.testfunctions.Rastrigin`, `opt.testfunctions.Rosenbrock` | 14 | 7 of 12 | earns its place |
| `EmissionFamily` | `emissions` | alignment_key, is_discrete, log_density, n_states, named_parameters, observation_dtype, reestimate, sample, validate | `emissions.BetaBinomialEmission`, `emissions.BinomialEmission`, `emissions.CategoricalEmission`, `emissions.GaussianEmission`, `emissions.NegativeBinomialEmission`, `emissions.PoissonEmission`, `sim.count_pairs.IndependentCountPair` | 5 | 3 of 12 | earns its place |
| `CountEmissionFamily` | `emissions` | mean, variance | `emissions.BetaBinomialEmission`, `emissions.BinomialEmission`, `emissions.NegativeBinomialEmission`, `emissions.PoissonEmission` | 0 | 1 of 12 | under the rule; kept: the count families' shared moments; #399's two-channel emission is its next implementer |
| `Schedule` | `opt.schedule` | n_steps | `opt.schedule.Constant`, `opt.schedule.Cosine`, `opt.schedule.Exponential`, `opt.schedule.Linear` | 5 | 6 of 12 | earns its place |
| `Initializer` | `opt.initialize` | starts | `opt.initialize.FromObjective`, `opt.initialize.Perturbed`, `opt.initialize.RandomRestart`, `opt.mixture.KMeansPlusPlus`, `search.initialize.FromDistances`, `search.initialize.FromHadamard` | 3 | 4 of 12 | earns its place |
| `Environment` | `learn.environment` | actions, features, is_terminal, n_features, reset, step | `learn.hmm.StatePathLandscape`, `learn.potts.PottsLandscape`, `search.rl.TopologyEnvironment` | 12 | 3 of 12 | earns its place |
| `Policy` | `learn.policy` | sample | `emissions.BetaBinomialEmission`, `emissions.BinomialEmission`, `emissions.CategoricalEmission`, `emissions.GaussianEmission`, `emissions.NegativeBinomialEmission`, `emissions.PoissonEmission`, `learn.policy.EpsilonGreedyPolicy`, `learn.policy.LinearPolicy`, `learn.policy.MLPPolicy`, `sim.count_pairs.IndependentCountPair` | 1 | 2 of 12 | under the rule; kept: what a rollout needs and no more, so an untrainable wrapper rolls out (its docstring) |
| `TrainablePolicy` | `learn.policy` | dtype, greedy, log_probabilities, parameters, sample | `learn.policy.LinearPolicy`, `learn.policy.MLPPolicy` | 4 | 0 of 12 | earns its place |
| `RelaxedObjective` | `learn.relaxed` | discrete, n_sites, n_states, relaxed | `learn.relaxed.RelaxedHmmPath`, `learn.relaxed.RelaxedPotts` | 0 | 2 of 12 | under the rule; kept: the relaxation driver's argument, with two implementers in its module |
| `Surrogate` | `bound` | kind | `learn.surrogate.CalibratedBound`, `learn.surrogate.Fitted`, `likelihood.surrogate.MeanFieldLogPartition`, `likelihood.surrogate.ParsimonyUpperBound`, `likelihood.surrogate.PlugInLikelihood`, `likelihood.surrogate.SpanningTreeLogPartition`, `search.support.Support`, `search.support.TemperedSupport`, `search.surrogate.LearnedTreeSurrogate` | 3 | 3 of 12 | earns its place |
| `Channel` | `sim.ldpc` | log_likelihood_ratios | `sim.ldpc.BinaryErasureChannel`, `sim.ldpc.BinaryInputGaussianChannel`, `sim.ldpc.BinarySymmetricChannel` | 1 | 2 of 12 | under the rule; kept: the decoder's channel argument, with three implementers in its module |

## Data contracts (4)

| Seam | Home | Fields | Consumers | Problems reached | Verdict |
| --- | --- | --- | --- | --- | --- |
| `FactorGraph` | `sim.factor_graph` | factors, variables | 7 | 6 of 12 | earns its place |
| `HmmParams` | `sim.hmm` | emissions, initial, n_sequences, n_states, seed, sequence_length, tolerance, transition | 6 | 3 of 12 | earns its place |
| `PottsParams` | `opt.potts` | chain_length, coupling, field, n_chains, n_states, seed | 3 | 2 of 12 | earns its place |
| `SpatioSequentialParams` | `sim.spatio_sequential` | beta, emissions, graph, initial, n_classes, n_positions, n_states, self_transition | 4 | 2 of 12 | earns its place |
