"""HiGHS on the explicit local-polytope LP of a Potts model the adapter wrote (issue #1063).

Inputs: ``unary``, the ``(n_nodes, n_states)`` energy of each site's states,
the field's negative; ``first``, ``second`` and ``coupling`` per edge. The
LP minimizes ``sum_i sum_a unary[i, a] mu_i(a) - sum_e J_e sum_a
mu_e(a, a)`` in ``sim.potts.energy``'s sign, over non-negative node
marginals ``mu_i`` and edge marginals ``mu_e`` subject to ``sum_a mu_i(a) =
1`` and each edge's table summing onto both of its endpoints' marginals. The
columns are the node marginals, site-major, then each edge's ``n_states^2``
table, row-major. Outputs: ``value``, the LP's optimal value;
``node_marginals``; ``status`` and ``message``, `linprog`'s; ``iterations``;
``build_seconds``, the constraint matrix's assembly. The measured seconds are
the solve alone; the peak resident memory is the assembly and the solve
together.

With ``time_limit`` among the inputs the node marginals are integer (issue
#1069) and ``milp`` solves the MIP to a relative gap of zero within that many
seconds. The edge tables stay continuous: integral node marginals fix every
table, so the MIP's optimum is the minimum energy. It adds ``dual_bound``,
HiGHS's lower bound, which equals ``value`` when the optimum is proven, and
``nodes``, the branch-and-bound node count; ``iterations`` is then 0.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sal.validation.protocol import dump, peaked, received, timed


def _polytope(
    unary: np.ndarray, first: np.ndarray, second: np.ndarray, coupling: np.ndarray
) -> tuple[np.ndarray, Any, np.ndarray]:
    """The cost vector, the equality matrix and its right-hand side."""
    from scipy import sparse  # the framework's package, imported only here

    n_nodes, n_states = unary.shape
    n_edges = first.shape[0]
    square = n_states * n_states
    node_columns = n_nodes * n_states
    n_columns = node_columns + n_edges * square

    cost = np.zeros(n_columns)
    cost[:node_columns] = unary.reshape(-1)
    diagonal = np.arange(n_states) * (n_states + 1)
    tables = cost[node_columns:].reshape(n_edges, square)
    tables[:, diagonal] = -coupling[:, None]

    # Normalization: one row per site over its n_states columns.
    rows = [np.repeat(np.arange(n_nodes), n_states)]
    columns = [np.arange(node_columns)]
    values = [np.ones(node_columns)]
    # Marginalization, one row per (edge, state) at each end: the table's row
    # `a` (first end) or column `b` (second end) sums to the site's mu(a).
    edge = np.arange(n_edges)[:, None, None]
    state_a = np.arange(n_states)[None, :, None]
    state_b = np.arange(n_states)[None, None, :]
    table = node_columns + edge * square + state_a * n_states + state_b
    shape = (n_edges, n_states, n_states)
    for block, (end, state) in enumerate(((first, state_a), (second, state_b))):
        offset = n_nodes + block * n_edges * n_states
        row = offset + edge * n_states + state
        rows.append(np.broadcast_to(row, shape).reshape(-1))
        columns.append(table.reshape(-1))
        values.append(np.ones(n_edges * square))
        site_rows = offset + np.arange(n_edges * n_states)
        rows.append(site_rows)
        columns.append((end[:, None] * n_states + np.arange(n_states)).reshape(-1))
        values.append(-np.ones(n_edges * n_states))
    n_rows = n_nodes + 2 * n_edges * n_states
    matrix = sparse.csr_array(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(n_rows, n_columns),
    )
    right = np.zeros(n_rows)
    right[:n_nodes] = 1.0
    return cost, matrix, right


def _mip(
    cost: np.ndarray, matrix: Any, right: np.ndarray, n_integer: int, limit: float
) -> Any:
    """``milp`` on the polytope with the first ``n_integer`` columns integer."""
    from scipy.optimize import Bounds, LinearConstraint, milp

    integrality = np.zeros(cost.shape[0], dtype=np.int64)
    integrality[:n_integer] = 1
    return milp(
        cost,
        integrality=integrality,
        bounds=Bounds(0.0, np.inf),
        constraints=LinearConstraint(matrix, right, right),
        options={"time_limit": limit, "mip_rel_gap": 0.0},
    )


def main() -> None:
    """Assemble the LP or the MIP, solve it under the timer, and write the value back."""
    from scipy.optimize import linprog  # HiGHS, imported only in this interpreter

    inputs, returned = received()
    unary = inputs["unary"]
    limit = float(inputs["time_limit"]) if "time_limit" in inputs else None

    def build_and_solve() -> tuple[Any, float, float]:
        (cost, matrix, right), build_seconds = timed(
            lambda: _polytope(
                unary, inputs["first"], inputs["second"], inputs["coupling"]
            )
        )
        if limit is not None:
            result, seconds = timed(
                lambda: _mip(cost, matrix, right, unary.size, limit)
            )
        else:
            result, seconds = timed(
                lambda: linprog(
                    cost, A_eq=matrix, b_eq=right, bounds=(0.0, None), method="highs"
                )
            )
        return result, seconds, build_seconds

    (result, seconds, build_seconds), peak_bytes = peaked(build_and_solve)
    solved = result.x is not None
    node_columns = unary.size
    marginals = (
        result.x[:node_columns].reshape(unary.shape)
        if solved
        else np.full(unary.shape, np.nan)
    )
    dump(
        returned,
        {
            "value": np.asarray(result.fun if solved else np.nan, dtype=np.float64),
            "node_marginals": np.ascontiguousarray(marginals, dtype=np.float64),
            "status": np.asarray(result.status, dtype=np.int64),
            "message": np.asarray(str(result.message)),
            "iterations": np.asarray(getattr(result, "nit", 0), dtype=np.int64),
            "dual_bound": np.asarray(
                getattr(result, "mip_dual_bound", np.nan), dtype=np.float64
            ),
            "nodes": np.asarray(getattr(result, "mip_node_count", 0), dtype=np.int64),
            "build_seconds": np.asarray(build_seconds, dtype=np.float64),
        },
        seconds,
        peak_bytes,
    )


if __name__ == "__main__":
    main()
