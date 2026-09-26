"""The unit a method spends, declared once for the ladder and for the budget.

Issue #860. `infra/ladder.py` declared this enum for the oracle ladder (issue
#818) while `opt.budget.Budget` took its unit as a bare `str`, spelled eight
ways across the suite --- two of them, ``"site-visits"`` and ``"site visits"``,
the same unit. One axis named twice is two vocabularies, and neither refused a
ninth spelling. This module is the one declaration; the ladder imports it back
and a budget is a ceiling in one of its members.

A member's value is the text a table prints, so it is that unit's one
spelling. Adding a member is how a unit the package spends enters the
vocabulary; spelling one a second way is not.

Infrastructure, not science: nothing here knows what a rung or an instance is.
"""

from __future__ import annotations

from enum import StrEnum


class Cost(StrEnum):
    """The unit a rung spends, as the sampler or optimizer reports it (issue #818).

    A gradient is an objective evaluation and a backward pass through the
    same tape, so a rung spending gradients and one spending evaluations are
    not ranked by either column alone (`sample/CLAUDE.md`): the rung declares
    what it spends, the table does not rank. An exact rung says so, its cost
    being the instance's count rather than a budget's.
    """

    EXACT = "exact"
    """An enumeration: the cost is the instance's count, not a budget."""
    PASS = "one pass"
    """One deterministic pass over the instance: a recursion, a cut, a construction."""
    ITERATIONS = "iterations"
    """Message passes, EM or decoder iterations, cycles of a solver."""
    SWEEPS = "sweeps"
    """Monte Carlo sweeps over the sites, or single-cluster steps counted as one."""
    EVALUATIONS = "objective evaluations"
    """Candidate scorings or density evaluations, no backward pass."""
    GRADIENTS = "gradients"
    """Objective evaluations each with a backward pass through the same tape."""
    TRAINED = "objective evaluations, and gradients to train"
    """A learned method: gradients in training, evaluations when it acts."""
    SEVERAL = "several: each method its own"
    """A rung naming several methods that do not share a unit."""

    # The units a budget holds equal that no rung of the ladder spends
    # (issue #860). Each carries the spelling the suite already used, so a
    # comparison's table names the unit in the words it named it in before.
    DECISIONS = "decisions"
    """Actions a policy takes in an environment: the learner's unit (`learn/CLAUDE.md`)."""
    EPISODES = "episodes"
    """Episodes a learner trains on, each a rollout to termination (issue #1090)."""
    FITS = "fits"
    """Whole fits, where one run of the inner optimizer is the indivisible cost."""
    SITE_VISITS = "site visits"
    """Single-site updates, which match across move sets whose sweeps cost differently."""
    PASSES = "passes"
    """Passes over every observation: what a fit on data spends, counted by the data."""
    CANDIDATES = "candidates"
    """Candidates a search scores, where scoring one is itself a fit."""
    SECONDS = "seconds"
    """Wall clock, on the host that ran the method (issue #891).

    The one unit that is not a count: it belongs to the machine and its load
    as much as to the method, so a number in it is quoted with the host and
    the load it was read under, and never byte-compared against a rerun.
    """
