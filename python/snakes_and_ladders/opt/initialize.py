"""Where an optimization starts, as something the caller can choose.

Issue #251. `Objective.initial()` was already in the protocol and `fit` already
took a `theta0` override, so the *seam* existed. What every implementation put
through it was one fixed constant: a short branch length everywhere, zeros for
the Potts chain and lattice, zeros with a hand-tuned tilt for the HMM, a fixed
`start` for each test function.

**The repository has been bitten by this once, and the scar is in the code.**
`opt/hmm.py`'s `initial()` records that the uniform point is not a poor guess
but a *stationary point*: with every hidden state identical the gradient with
respect to the initial and transition parameters is exactly zero, so an
optimizer started there never moves while the emission rows drift to the pooled
symbol frequency. The fix was a fixed tilt, written into that one file.
`perturbed` here is that idea with the model taken out of it.

**And one start is known not to be enough.** `opt/testfunctions.py` keeps
Himmelblau — four equal minima, the start picks one — and Rastrigin, whose
docstring says a fit from a single start lands in whichever cell it began in.
Meanwhile `search/` has restarted all along: `STATUS.md` records hill climbing
reaching the enumerated optimum from 12 of 12 starts. The discrete side
restarts and the continuous side does not.

**Randomness enters as a generator.** `opt/hmm.py` rejected a jitter because "a
seeded jitter would make the fit depend on a second seed nobody declared" — and
that is an argument against *seeding inside a call*, not against randomness.
An initializer taking a generator is reproducible; one seeding itself is the
defect `sim/CLAUDE.md` forbids and issue #240 removed elsewhere.

**What is not here.** An initializer that reads the data — k-means++ for a
mixture, or a warm start from a moment estimator — cannot live in this module:
`opt/` may import no application module, and `test_opt_objective.py` asserts
it. Those belong beside the objective they initialize, with this module holding
the protocol and the model-free strategies only.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import torch

from snakes_and_ladders.opt.objective import Objective


@runtime_checkable
class Initializer(Protocol):
    """A source of starting points for one objective.

    One start is the degenerate case of many, which is why this returns a
    sequence rather than a tensor: a caller that wants today's behaviour asks
    for one, and the shape of the answer does not change.
    """

    def starts(self, objective: Objective) -> list[torch.Tensor]:
        """Candidate starting points, in unconstrained coordinates.

        Returns
        -------
        list[torch.Tensor]
            At least one point, each of the objective's own dimension.
        """
        ...  # pragma: no cover


class FromObjective:
    """The objective's own `initial()`, unchanged.

    The default, and today's behaviour exactly: every number `STATUS.md` pins
    was produced this way, so this has to stay reachable and stay first.
    """

    def starts(self, objective: Objective) -> list[torch.Tensor]:
        """The single point the objective nominates.

        Returns
        -------
        list[torch.Tensor]
            Exactly one start.
        """
        return [objective.initial()]


class Perturbed:
    """The objective's start, tilted by a fixed amount along each coordinate.

    Deterministic, and that is the point rather than a limitation. It exists
    for a surface whose nominal start is *stationary* — the HMM's uniform
    point, where the gradient is exactly zero and an optimizer never leaves —
    and for that a reproducible nudge off the symmetry is enough. Randomness
    would buy nothing and cost a declared generator.

    The tilt alternates in sign so the perturbation does not translate every
    coordinate the same way, which on a symmetric objective would land on
    another point of the same symmetry.

    Parameters
    ----------
    magnitude : float
        Size of the tilt in unconstrained coordinates.
    """

    def __init__(self, magnitude: float = 0.1) -> None:
        if magnitude <= 0.0:
            msg = f"magnitude must be positive, got {magnitude}"
            raise ValueError(msg)
        self.magnitude = magnitude

    def starts(self, objective: Objective) -> list[torch.Tensor]:
        """One start, off the objective's own by a fixed alternating tilt.

        Returns
        -------
        list[torch.Tensor]
            Exactly one start.
        """
        base = objective.initial()
        signs = torch.tensor(
            [1.0 if index % 2 == 0 else -1.0 for index in range(base.shape[0])],
            dtype=base.dtype,
            device=base.device,
        )
        return [base + self.magnitude * signs]


class RandomRestart:
    """Several starts, drawn around the objective's own.

    For a surface with more than one local optimum, where the answer a single
    fit returns is a property of where it began. `search/CLAUDE.md`'s budget
    rule applies: `n_starts` restarts cost `n_starts` fits, so the count is the
    caller's to justify and this makes it visible rather than hiding it inside
    a fit.

    Parameters
    ----------
    n_starts : int
        Points to draw, at least 1.
    scale : float
        Standard deviation of the Gaussian displacement, in unconstrained
        coordinates.
    rng : np.random.Generator
        Passed in rather than seeded here, so a caller running an ensemble gets
        independent restarts rather than the same set repeatedly, and a
        declared seed still determines the run (`sim/CLAUDE.md`, issue #240).
    include_nominal : bool
        Whether the objective's own start is the first of them. Default
        ``True``: a restart set that cannot reproduce the single-start answer
        can be worse than one fit, and nothing would say so.
    """

    def __init__(
        self,
        n_starts: int,
        scale: float,
        rng: np.random.Generator,
        include_nominal: bool = True,
    ) -> None:
        if n_starts < 1:
            msg = f"n_starts must be at least 1, got {n_starts}"
            raise ValueError(msg)
        if scale <= 0.0:
            msg = f"scale must be positive, got {scale}"
            raise ValueError(msg)
        self.n_starts = n_starts
        self.scale = scale
        self.rng = rng
        self.include_nominal = include_nominal

    def starts(self, objective: Objective) -> list[torch.Tensor]:
        """``n_starts`` points, the first being the nominal one if asked.

        Returns
        -------
        list[torch.Tensor]
            Exactly ``n_starts`` starts.
        """
        base = objective.initial()
        drawn: list[torch.Tensor] = []
        if self.include_nominal:
            drawn.append(base)
        while len(drawn) < self.n_starts:
            displacement = torch.tensor(
                self.rng.normal(scale=self.scale, size=int(base.shape[0])),
                dtype=base.dtype,
                device=base.device,
            )
            drawn.append(base + displacement)
        return drawn
