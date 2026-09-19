# learn/

Reinforcement learning applied to discrete search problems. The interface is
model-agnostic by construction, on exactly the terms `opt/CLAUDE.md` sets.

Root `CLAUDE.md` holds the repository-wide rules, and its **Writing Style**
section binds this file and every docstring, comment and commit message in this
module. It is referenced here, never restated. What follows is local.

## What lives here

`environment.py` is the interface — a state, an action set that varies with the
state, a step returning the next state and a reward, and features of each
available action. `policy.py` holds the softmax-over-scored-actions policy the
paper specifies. `rollout.py` generates episodes under a policy and under the
greedy searcher through one loop, since a comparison between them is only
meaningful if the loop is shared; `reinforce.py` is the score-function estimator
and `exact.py` its oracle, by enumerating trajectories; `arena.py` is the table
every learner is read in. The environments here are **reference instances**.

The count is the point: an interface justified by one model is shaped by that
model, so this one carries an energy environment, a decoding problem and a
topology search. The first two unpack a model into index and log-probability
arrays; `tree.py` takes a topology, which is why the rule below exempts it by
name (issue #779).

## Local rules

- **No application imports, and the exceptions are listed by name.** The
  interface, the policy and the estimators import nothing from `sal.sim`,
  `sal.likelihood` or `sal.search`, so no agent is shaped by one problem.
  `tree.py`, `ranking.py` and `potts_nd.py` do, and the guard in
  `tests/regression/learn/` refuses a fourth (issue #779).
- **Closed-form rewards at known parameters are vital for testing.**
- **Truth is a terminal penalty, never a training signal.** An agent that can
  see the answer during training learns to look it up.
- **`gamma = 1`, so the reward telescopes to the total improvement an episode
  achieved; any `gamma < 1` is more greedy.** A hyperparameter later.
- **A sampled return is a diagnostic, never a result.** It is a Monte Carlo
  estimate under a changing policy, so it rises for reasons that include a
  broken estimator. Learning is claimed against `exact.py`'s enumerated
  expected return; the training curve is reported, not asserted on.
- **The estimator is pinned to a brute-force gradient.** With a finite action
  set and a finite horizon the trajectory set is finite, so the expected
  return is a closed form and its gradient follows by autodiff. Both routes
  are checked — autodiff against finite differences, and the sampled
  estimator against the enumerated gradient — because a score-function
  estimator with a sign error is wrong by a factor and still trains.
- **The baseline must not depend on the batch it centres.** Subtracting a
  constant is unbiased because it multiplies a term of zero expectation, and
  that argument needs independence. A within-batch mean is correlated with the
  returns it centres and buys variance at the price of an `O(1/N)` bias; the
  running mean of *earlier* iterations costs nothing and keeps the claim exact.
- **A budget is counted in decisions, never in seconds.** Both the greedy
  searcher and a policy score the whole neighbourhood per decision, so decisions
  are the unit at which they are comparable — as `sal.search.infer` counts fits.
- **A match is a result, and a row states what it cost.** Nothing beats an
  exact baseline, so a learner that reaches one has shown its policy class
  contains the classical method, and one reaching it more cheaply has shown
  more; a row carries the evaluations beside the fraction and names which of
  beat, match and lose it is (issue #705).
- **An episode that may leave a local optimum is scored on its best state,
  not its last.** `rollout(..., stop_at_local_optimum=False)` runs to its
  budget, so its final state is wherever the walk happened to stop, and a
  real search keeps the best thing it saw. Scoring the last state instead
  would make a better searcher look worse the longer it ran.
- **A comparison against a wandering searcher is against *restarts*.** Once
  an episode is no longer bounded by a local optimum, one greedy run stops
  after a few decisions and leaves the budget unspent; restarting it until the
  budget is gone is the comparison, and on the issue #177 fixture it reaches
  the enumerated maximum from every start where the best epsilon does not
  (issue #194). A result stated against single-run greedy overstates itself.
- **A feature is what a move already computes without a fit, and a column an
  arm cannot vary is dropped rather than carried.** The gauge rule is per arm:
  a move set whose actions differ in one coordinate has one informative
  column, and a second would be a weight nothing identifies --- the distance
  from the state under NNI is the standing example, since every NNI neighbour
  sits at the same one. The width is decided from the declared move set and
  ladder, never per state, so a policy's input shape is fixed (#706, #779).
- **A critic is pinned to enumeration, never to its loss.** Where the return
  is exact so is the state value, and a critic's number is its fit to that; a
  baseline may read the state and never the action sampled at that step.
- **A planner is counted in evaluations, and its answer is compared with
  greedy's at the same count.** A search that reaches the optimum by
  evaluating more successors than hill climbing has not won.
- **A learned surrogate predicts the gap above an analytic bound**, scored on
  unseen groups, so a poor fit falls back to the bound (issue #308).
- **`step` is deterministic by contract, so randomness inside a move is keyed
  on the state and the action, never streamed.** `exact.py`'s enumeration
  depends on one successor per pair, so a Monte Carlo move draws from a digest
  of where it is and what it is doing — `sim/count_pairs.py`'s contract,
  applied to a move (issue #706); a stream threaded through the caller would
  make the successor depend on visit order. What keying cannot express — a
  stochastic *reward*, a slippery transition — goes to the issue tracker rather
  than into a deterministic stand-in that tests nothing (issue #597).
- **A learner is pinned on a canonical fixture before a research problem.**
  Every other environment here *is* a research problem, so a tie leaves two
  readings open — the problem is hard, or the learner is broken.
  `canonical.py` carries optima known from outside, refereed by a second
  computation: a sweep over the states against `exact.py` (issue #597).
- **A learner states which values it converges to.** Q-learning's fixed point
  is `q*` and SARSA's is `q_pi` for the epsilon-greedy policy it behaves
  under, so a suite asserting both match value iteration asserts something
  false: one is held at `V*` and the other held away from it (issue #597).

## Framework

**PyTorch**, per root `CLAUDE.md`, and `float64` throughout: the exact oracle
compares autodiff against central finite differences, which `float32` cannot
support.

## Relaxations

A relaxation must reduce to the discrete objective exactly at every corner of the
simplex, checked over every configuration of an enumerable instance. Under a
factorized distribution the expected discrete score equals the relaxed score at
the marginals whenever no term reuses a site — multilinearity, not graph shape,
is the boundary — so the maximum sits at a vertex and a relaxation adds no
optimum the discrete problem lacks. A gradient estimator's bias is measured
against the exact gradient enumeration supplies, never assumed small.
