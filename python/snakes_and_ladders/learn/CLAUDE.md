# learn/

Reinforcement learning applied to discrete search problems. The interface is
model-agnostic by construction, on exactly the terms `opt/CLAUDE.md` sets.

Root `CLAUDE.md` holds the repository-wide rules, and its **Writing Style**
section binds this module too.  It is referenced here, never restated. What follows is local,
and is principle: the numbers behind each rule live with the code that
produces them or in `STATUS.md`.

## What lives here

`environment.py` is the interface — a state, an action set that varies with
the state, a step returning the next state and a reward, and features of each
available action.


`policy.py` currently holds the softmax-over-scored-actions policy the technical
document specifies.

`rollout.py` generates episodes under a policy and under the greedy searcher
through the same loop: a comparison between them is only meaningful if the
loop is shared. `reinforce.py` is the score-function estimator and `exact.py`
its oracle, by enumerating trajectories.

The environments here are **reference instances**, not applications, over the
same models `sal.opt` fits.

## Local rules

- **No application imports.** Nothing here may import from `sal.sim`,
  `sal.likelihood` or `sal.search`, asserted by
  `tests/regression/test_learn_environment.py`.
- **Closed form rewards at known parameters are vital for testing**  Rewards will also
  be solved for in future development.
- **`gamma = 1` currently, but will be a hyperparameter.** With this, the reward telescopes to the total
  improvement an episode achieved. Any `gamma < 1` is more greedy.
- **A sampled return is a diagnostic, never a result.** It is a Monte Carlo
  estimate under a changing policy, so it rises for reasons that include a
  broken estimator. Learning is claimed against the enumerated expected
  return in `exact.py`; the training curve is reported, not asserted on.
- **The estimator is pinned to a brute-force gradient.** With a finite action
  set and a finite horizon the trajectory set is finite, so the expected
  return is a closed form and its gradient follows by autodiff. Both routes
  are checked — autodiff against finite differences, and the sampled
  estimator against the enumerated gradient — because a score-function
  estimator with a sign error is wrong by a factor and still trains.
- **The baseline must not depend on the batch it centres.** Subtracting a
  constant is unbiased because it multiplies a term of zero expectation, and
  that argument needs independence. A within-batch mean is correlated with
  the returns it centres and buys variance at the price of an `O(1/N)` bias;
  the running mean of *earlier* iterations costs nothing and keeps the claim
  exact.
- **A budget is counted in decisions, never in seconds.** Both the greedy
  searcher and a policy score the whole neighbourhood per decision, so
  decisions are the unit at which they are comparable — the same reasoning
  that makes `sal.search.infer` count candidate fits.

- **An episode that may leave a local optimum is scored on its best state,
  not its last.** `rollout(..., stop_at_local_optimum=False)` runs to its
  budget, so its final state is wherever the walk happened to stop, and a
  real search keeps the best thing it saw. Scoring the last state instead
  would make a better searcher look worse the longer it ran.

- **A comparison against a wandering searcher is against *restarts*.** Once
  an episode is no longer bounded by reaching a local optimum, a single
  greedy run is not a budget-matched baseline: greedy stops after a few
  decisions and leaves the rest of the budget unspent. Restarting it until
  the budget is gone is the honest comparison, and on the issue #177 fixture
  it reaches the enumerated maximum from every start where the best epsilon
  measured does not (issue #194; the numbers are in `STATUS.md`). A result
  stated against single-run greedy alone overstates itself.
- **A critic is pinned to enumeration, never to its loss.** Where the return
  is exact so is the state value, and a critic's number is its fit to that;
  a baseline may read the state and never the action sampled at that step.
- **A planner is counted in evaluations, and its answer is compared with
  greedy's at the same count.** A search that reaches the optimum by
  evaluating more successors than hill climbing has not won.
- **A learned surrogate predicts the gap above an analytic bound**, scored on
  unseen groups, so a poor fit falls back to the bound (issue #308).

- **A critic is pinned to enumeration, never to its loss.** Where the return
  is exact so is the state value, and a critic's number is its fit to that;
  a baseline may read the state and never the action sampled at that step.

- **A planner is counted in evaluations, and its answer is compared with
  greedy's at the same count.** A search that reaches the optimum by
  evaluating more successors than hill climbing has not won.

## Framework

**PyTorch**, per root `CLAUDE.md`, and `float64` throughout: the exact
oracle compares autodiff against central finite differences, which `float32`
cannot support.

## The instances

The count is the point: an interface justified by one model is shaped by that
model, so this one carries an energy landscape, a decoding problem, and — in
`sal.search`, which may import both halves — a topology search. None of them
takes an application type. The caller unpacks a model into index and
log-probability arrays, because the no-application-imports rule admits no
exception for convenience.

## Relaxations

A relaxation must reduce to the discrete objective exactly at every corner of
the simplex, checked over every configuration of an enumerable instance. Under
a factorized distribution the expected discrete score equals the relaxed score
at the marginals whenever no term reuses a site — multilinearity, not graph
shape, is the boundary — so the maximum sits at a vertex and a relaxation adds
no optimum the discrete problem lacks. A gradient estimator's bias is measured
against the exact gradient enumeration supplies, never assumed small.
