# learn/

Reinforcement learning over discrete search problems. The interface is
model-agnostic by construction, on exactly the terms `opt/CLAUDE.md` sets:
the same machinery must serve a Potts landscape and a tree search (issue
#131), and the phylogenetic case is one instance of it rather than its
author.

Root `CLAUDE.md` holds the repository-wide rules, and its **Writing Style**
section binds this file too — and every docstring, comment and commit message
in this module. It is referenced here, never restated. What follows is local,
and is principle: the numbers behind each rule live with the code that
produces them or in `STATUS.md`.

## What lives here

`environment.py` is the interface — a state, an action set that varies with
the state, a step returning the next state and a reward, and features of each
available action. `policy.py` holds the softmax-over-scored-actions policy
the technical document specifies.

`rollout.py` generates episodes under a policy and under the greedy searcher
through the same loop: a comparison between them is only meaningful if the
loop is shared. `reinforce.py` is the score-function estimator and `exact.py`
its oracle, by enumerating trajectories.

The environments here are **reference instances**, not applications, over the
same models `snakes_and_ladders.opt` fits. That is deliberate: one model appears here as an
environment searched discretely and there as an objective fitted continuously,
so the claim that both halves of the project share one abstraction is
demonstrated rather than asserted.

## Local rules

- **No application imports.** Nothing here may import from `snakes_and_ladders.sim`,
  `snakes_and_ladders.likelihood` or `snakes_and_ladders.search`, asserted by
  `tests/regression/test_learn_environment.py`. An agent developed against a
  tree is an agent shaped by trees, and the phylogenetic environment
  therefore lives in `snakes_and_ladders.search`, which may import both halves.
  `snakes_and_ladders.opt` is not forbidden: it is infrastructure too, and reusing its
  Potts fixture is the point rather than a shortcut.
- **A reward is a closed form at known parameters, never an inner solve.**
  That is what makes an episode affordable: a fitted reward costs a full solve
  per action and an episode evaluates a whole neighbourhood per step. Where
  the fitted reward is the honest one, the two are *compared* rather than
  swapped silently — the cheap surface can rank candidates differently.
- **`gamma = 1`, and it is not a hyperparameter.** The reward is a difference
  of the objective, so the undiscounted return telescopes to the total
  improvement an episode achieved. Any `gamma < 1` breaks that and prefers
  improvement found early over the same improvement found late.
- **A feature constant across a state's actions is unidentifiable.** A
  softmax over scores cancels anything every action shares — the same gauge
  `snakes_and_ladders.opt` fixes for a simplex. No environment supplies a bias term, and a
  test pins the invariance.
- **The greedy searcher must be inside the policy class.** Where the reward
  decomposes into the features, some weight vector *is* the greedy baseline at
  zero temperature. That is what makes "the agent beat the baseline" a
  statement about learning rather than about two unrelated algorithms, and a
  test checks the two produce identical trajectories.
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
  that makes `snakes_and_ladders.search.infer` count candidate fits.

## Framework

**PyTorch**, per root `CLAUDE.md`, and `float64` throughout: the exact
oracle compares autodiff against central finite differences, which `float32`
cannot support.

## The instances

The count is the point: an interface justified by one model is shaped by that
model, so this one carries an energy landscape, a decoding problem, and — in
`snakes_and_ladders.search`, which may import both halves — a topology search. None of them
takes an application type. The caller unpacks a model into index and
log-probability arrays, because the no-application-imports rule admits no
exception for convenience.

## What is not here yet

PPO and a learned state-value critic; `docs/tex`'s reinforcement-learning
section states the theory they are built against.
