"""The reference Potts instance these tests learn on, declared once.

``FIELD`` was written in eleven modules and the chain in nine, agreeing by
coincidence (issue #863, design-audit row R19); #313's rows, `learn.exact`'s
gradients and the PPO bitwise identities are read against this one instance.
The smallest with a three-state alphabet (a gauge exists), four sites (81
configurations, an exact optimum) and a symmetry-breaking field. Another size
is ``potts_environment(6)``, not a second constant.
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

    The default is the instance every published row was measured on.
    """
    return PottsEnvironment(coupling=COUPLING, field=FIELD, chain_length=chain_length)
