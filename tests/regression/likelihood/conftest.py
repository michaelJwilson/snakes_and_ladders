"""The six-node tree these message-passing tests are exact on, declared once.

`TREE` and the field over it were written out in three modules ---
`test_belief_propagation.py`, `test_message_passing.py` and
`test_message_passing_rust.py` --- and every exactness claim in all three is
read against them (issue #863, design-audit row R19). They agreed by
coincidence of three literals: an edge added to one copy would have left two
routes still exact on the tree they were told about and one measured against
a different graph, which reads as a backend disagreeing.

It is the smallest graph with all of what these tests need: six nodes, so a
node has two children and another has one and the recursion is not a chain;
a mixed-sign coupling, so a message is not monotone; and few enough
configurations --- ``3 ** 6`` --- that enumeration is the referee.

A loopy graph is *not* here. Each module builds its own from
`sim.graph.lattice_graph`, which is the code `PROBLEMS.md` says defines the
lattice problem: a module that stops importing it stops saying which problem
it exercises (`tests/_problems.py`).
"""

from __future__ import annotations

import numpy as np
from snakes_and_ladders.sim.graph import PottsGraph

#: The per-state field over `TREE`; three states, none of them favoured
#: enough to make a marginal degenerate.
FIELD = np.array([0.3, -0.7, 0.15])

#: Six nodes, five edges, one root with two children.
TREE = PottsGraph(
    n_nodes=6,
    edges=((0, 1), (0, 2), (1, 3), (1, 4), (2, 5)),
    coupling=(0.8, -0.4, 1.2, 0.3, 0.9),
)
