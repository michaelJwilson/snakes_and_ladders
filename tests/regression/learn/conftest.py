"""The reference Potts instance these tests learn on, declared once.

``FIELD`` was written out in eleven modules and the four-site chain built
from it in nine, each as its own ``_environment()``. They agreed by
coincidence of eleven literals: the rows issue #313 published, the gradients
`learn.exact` referees and the bitwise identities between the two PPO
implementations are all read against *one* instance, and a module that
changed a coupling would have gone on comparing its own numbers to itself
(issue #863, design-audit row R19).

It is the smallest instance with all three of what the learners need: a
three-state alphabet, so a gauge exists; four sites, so enumeration is 81
configurations and an exact optimum is free; and a field that breaks the
symmetry, so a policy has something to learn beyond the coupling.

A module needing another size asks for one --- ``potts_environment(6)`` ---
rather than writing a second constant.
"""

from __future__ import annotations

import numpy as np
from snakes_and_ladders.learn.potts import PottsEnvironment

#: The per-state field of the reference instance.
FIELD = np.array([0.4, -0.1, -0.3])

#: Its coupling: ferromagnetic, and strong enough that the optimum is not the
#: field's own argmax at every site.
COUPLING = 0.75

#: Its length. Enumeration is ``3 ** 4`` configurations, so every test that
#: wants an exact answer can have one.
CHAIN_LENGTH = 4


def potts_environment(chain_length: int = CHAIN_LENGTH) -> PottsEnvironment:
    """The reference Potts chain, at ``chain_length`` sites.

    Parameters
    ----------
    chain_length : int
        Sites. The default is the instance every published row was measured
        on; a longer chain is a different instance and says so at the call.

    Returns
    -------
    PottsEnvironment
    """
    return PottsEnvironment(coupling=COUPLING, field=FIELD, chain_length=chain_length)
