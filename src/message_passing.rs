//! The tree schedule's two passes over the factor-graph layout.
//!
//! `sal.likelihood.message_passing._run` --- the vectorized
//! NumPy implementation --- stays as the oracle this is pinned against, and
//! `likelihood.message_passing_reference` stays as its oracle, per root
//! `CLAUDE.md`.
//!
//! **Why this one is a port.** The tree schedule sends one message per level
//! on a chain, so the per-level dispatch is the work: 798 steps at a chain of
//! 200, each a handful of NumPy calls over a single four-wide row, and the
//! plan that groups them rebuilt per call. Issue #754's stress profile put
//! `tree_passes` at 29.8% of the run and `_logsumexp_last` at 10.6%;
//! `docs/experiments/020` carries the ranking and `027` the measurement that
//! admitted the port. The levels are a batching of the order and not a
//! constraint on it --- a message depends only on messages of strictly lower
//! height, and on the way down of strictly lower depth --- so this walks the
//! breadth-first order and its reverse instead of grouping, and the values
//! are the same ones in the same arithmetic.
//!
//! **The arithmetic is the oracle's, operation for operation.** A
//! variable-to-factor message sums its other edges' rows onto zeros in factor
//! order; [`logsumexp`] is `message_passing._logsumexp_last`'s peak, shift,
//! `exp`, sum, `log`, summed left to right, which is NumPy's pairwise
//! reduction below eight terms; a factor adds each incoming row onto its
//! table in axis order and reduces the axes it does not send along in
//! row-major order, which is what `transpose(...).reshape(n, c, -1)` hands
//! `logsumexp`. A factor of degree one returns its table unreduced, as the
//! oracle does, so an indicator's `-inf` stays `-inf` rather than becoming
//! the `nan` a shift by an infinite peak would give.
//!
//! **What crosses.** The layout as eight contiguous arrays and the stacked
//! tables, once per call. Nothing is cached on the layout: the marshalling is
//! 0.266 ms against 18.8 ms of Python sends at the chain of 200, which is the
//! recompute side of root `CLAUDE.md`'s recompute-or-store decision, made
//! rather than defaulted. The rooted walk is derived here rather than passed,
//! for the reason `bcjr::sources` derives its inverse: it is `O(nodes)`
//! against the `O(edges * width^degree)` the call does, and deriving it
//! checks the tree the passes rest on.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// The graph as the flat arrays `likelihood.schedule.Layout` holds, borrowed.
///
/// Every list of lists crosses as offsets plus a flat array, which is root
/// `CLAUDE.md`'s neighbour-list rule on the compiled side of the Python
/// boundary. `edge_factor` is not among them: it is the scatter of
/// `factor_edges`, derived in [`edge_factors`] where deriving it also checks
/// that each edge sits on exactly one factor.
pub struct Layout<'a> {
    /// Each variable's domain size.
    pub cardinality: &'a [i64],
    /// Into `variable_edges`, one past the end per variable.
    pub variable_offsets: &'a [i64],
    /// Each variable's edges, in factor order.
    pub variable_edges: &'a [i64],
    /// Into `factor_edges`, one past the end per factor.
    pub factor_offsets: &'a [i64],
    /// Each factor's edges, in the axis order of its table.
    pub factor_edges: &'a [i64],
    /// The variable each edge ends at.
    pub edge_variable: &'a [i64],
    /// Into `tables`, one past the end per factor.
    pub table_offsets: &'a [i64],
    /// Every factor's log table, row-major and concatenated.
    pub tables: &'a [f64],
    /// The row length of the message arrays: the largest cardinality.
    pub width: usize,
}

/// The factor each edge sits on, and the checks that make the rest safe.
fn edge_factors(layout: &Layout<'_>) -> Result<Vec<usize>, String> {
    let n_edges = layout.edge_variable.len();
    let n_variables = layout.cardinality.len();
    let mut owner = vec![usize::MAX; n_edges];
    for (factor, window) in layout.factor_offsets.windows(2).enumerate() {
        let (start, stop) = (window[0], window[1]);
        if start < 0 || stop < start || stop as usize > n_edges {
            return Err(format!("factor {factor} spans [{start}, {stop})"));
        }
        for &edge in &layout.factor_edges[start as usize..stop as usize] {
            if edge < 0 || edge as usize >= n_edges {
                return Err(format!("edge {edge} is outside [0, {n_edges})"));
            }
            if owner[edge as usize] != usize::MAX {
                return Err(format!("edge {edge} sits on two factors"));
            }
            owner[edge as usize] = factor;
        }
    }
    if let Some(edge) = owner.iter().position(|&f| f == usize::MAX) {
        return Err(format!("edge {edge} sits on no factor"));
    }
    for (edge, &variable) in layout.edge_variable.iter().enumerate() {
        if variable < 0 || variable as usize >= n_variables {
            return Err(format!(
                "edge {edge} ends at variable {variable}, outside [0, {n_variables})"
            ));
        }
    }
    for (variable, &c) in layout.cardinality.iter().enumerate() {
        if c < 1 || c as usize > layout.width {
            return Err(format!(
                "variable {variable} has cardinality {c}, outside [1, {}]",
                layout.width
            ));
        }
    }
    Ok(owner)
}

/// `log(sum(exp(values)))`, shifted by the maximum as the oracle's is.
///
/// Left to right, which is NumPy's pairwise sum below eight terms --- the
/// cardinalities and the reduced axis products this package's fixtures carry.
/// The maximum propagates `nan` the way `numpy.maximum.reduce` does, so a row
/// the oracle answers `nan` for is not answered here with a number.
#[inline]
fn logsumexp(values: &[f64]) -> f64 {
    let mut peak = values[0];
    for &value in &values[1..] {
        if value > peak || value.is_nan() {
            peak = value;
        }
    }
    let mut total = 0.0;
    for &value in values {
        // `exp(0)` is exactly one, so the maximum's term is added as one in
        // its own place: the sum's order and every bit of it are unchanged,
        // and one exponential per reduction is not taken (issue #997). A
        // non-finite peak takes the exponential, whose `nan` is the answer.
        total += if value == peak && peak.is_finite() {
            1.0
        } else {
            (value - peak).exp()
        };
    }
    peak + total.ln()
}

/// `values.max()`, `numpy.maximum.reduce`'s `nan` rule included.
#[inline]
fn maximum(values: &[f64]) -> f64 {
    let mut peak = values[0];
    for &value in &values[1..] {
        if value > peak || value.is_nan() {
            peak = value;
        }
    }
    peak
}

/// The rooted tree: the breadth-first order and each node's parent edge.
///
/// Nodes are the variables and then the factors, as
/// `Layout._edges_of` numbers them, and the root is the first variable ---
/// what `Layout.root` returns and what
/// `likelihood.message_passing_reference` roots at, so the messages compare
/// edge for edge. Breadth-first rather than recursive, so a 20,000-step
/// chain is not a stack overflow.
fn rooted(layout: &Layout<'_>, edge_factor: &[usize]) -> Result<(Vec<usize>, Vec<i64>), String> {
    let n_variables = layout.cardinality.len();
    let n_nodes = n_variables + layout.factor_offsets.len() - 1;
    let mut parent_edge = vec![-1i64; n_nodes];
    let mut seen = vec![false; n_nodes];
    let mut order = Vec::with_capacity(n_nodes);
    order.push(0usize);
    seen[0] = true;
    let mut head = 0;
    while head < order.len() {
        let node = order[head];
        head += 1;
        for index in 0..degree(layout, node) {
            let edge = edge_of(layout, node, index);
            if edge as i64 == parent_edge[node] {
                continue;
            }
            let child = if node < n_variables {
                n_variables + edge_factor[edge]
            } else {
                layout.edge_variable[edge] as usize
            };
            if seen[child] {
                return Err(
                    "the graph has a cycle; the tree schedule is exact only on a tree".to_string(),
                );
            }
            seen[child] = true;
            parent_edge[child] = edge as i64;
            order.push(child);
        }
    }
    if order.len() != n_nodes {
        return Err(
            "the graph is disconnected; the tree schedule is exact only on a tree".to_string(),
        );
    }
    Ok((order, parent_edge))
}

/// How many edges a node carries: its degree, or its table's rank.
#[inline]
fn degree(layout: &Layout<'_>, node: usize) -> usize {
    let n_variables = layout.cardinality.len();
    if node < n_variables {
        (layout.variable_offsets[node + 1] - layout.variable_offsets[node]) as usize
    } else {
        let factor = node - n_variables;
        (layout.factor_offsets[factor + 1] - layout.factor_offsets[factor]) as usize
    }
}

/// A node's `index`-th edge, in factor order for a variable and axis order
/// for a factor.
#[inline]
fn edge_of(layout: &Layout<'_>, node: usize, index: usize) -> usize {
    let n_variables = layout.cardinality.len();
    if node < n_variables {
        layout.variable_edges[layout.variable_offsets[node] as usize + index] as usize
    } else {
        let factor = node - n_variables;
        layout.factor_edges[layout.factor_offsets[factor] as usize + index] as usize
    }
}

/// The scratch rows one send needs, allocated once for the whole call.
///
/// Root `CLAUDE.md`'s allocation rule: a message is a few dozen bytes and a
/// chain sends four per position, so allocating per send would be the call.
struct Scratch {
    /// A variable's accumulated incoming rows, or a factor's reduced axis.
    row: Vec<f64>,
    /// A factor's table plus its incoming messages.
    acc: Vec<f64>,
    /// The values of one state of the axis sent along, contiguous.
    gathered: Vec<f64>,
    /// One factor's axis cardinalities, in axis order.
    dims: Vec<usize>,
}

/// The variable-to-factor message on `target`: `_send_from_variables`.
///
/// The other edges' rows are summed onto zeros in factor order --- the
/// oracle's `total = total + to_variable[sources[:, i], :c]`, term for term
/// --- and the row is normalized as it is sent, which is what keeps a long
/// chain off the floating-point floor.
fn send_from_variable(
    layout: &Layout<'_>,
    variable: usize,
    target: usize,
    to_variable: &[f64],
    to_factor: &mut [f64],
    scratch: &mut Scratch,
) {
    let c = layout.cardinality[variable] as usize;
    let width = layout.width;
    let total = &mut scratch.row[..c];
    total.fill(0.0);
    let start = layout.variable_offsets[variable] as usize;
    let stop = layout.variable_offsets[variable + 1] as usize;
    for &edge in &layout.variable_edges[start..stop] {
        if edge as usize == target {
            continue;
        }
        let row = &to_variable[edge as usize * width..edge as usize * width + c];
        for (slot, &value) in total.iter_mut().zip(row) {
            *slot += value;
        }
    }
    let scale = logsumexp(total);
    let out = &mut to_factor[target * width..target * width + c];
    for (slot, &value) in out.iter_mut().zip(total.iter()) {
        *slot = value - scale;
    }
}

/// The factor-to-variable message along `axis`: `_send_from_factors`, then
/// `_normalize`.
///
/// The table is copied and each incoming row added along its own axis in axis
/// order, which is the oracle's loop over `enumerate(group.shape)`; the axes
/// sent across are then reduced in row-major order, which is the layout
/// `acc.transpose(0, 1 + axis, *others).reshape(n, c, -1)` produces. Degree
/// one takes the oracle's early return: the table is the message.
fn send_from_factor(
    layout: &Layout<'_>,
    factor: usize,
    axis: usize,
    to_factor: &[f64],
    to_variable: &mut [f64],
    scratch: &mut Scratch,
    maximum_product: bool,
) {
    let width = layout.width;
    let start = layout.factor_offsets[factor] as usize;
    let stop = layout.factor_offsets[factor + 1] as usize;
    let edges = &layout.factor_edges[start..stop];
    let rank = edges.len();
    scratch.dims.clear();
    for &edge in edges {
        scratch
            .dims
            .push(layout.cardinality[layout.edge_variable[edge as usize] as usize] as usize);
    }
    let size: usize = scratch.dims.iter().product();
    let table = &layout.tables
        [layout.table_offsets[factor] as usize..layout.table_offsets[factor + 1] as usize];
    let acc = &mut scratch.acc[..size];
    acc.copy_from_slice(table);
    for (index, &incoming) in edges.iter().enumerate() {
        if index == axis {
            continue;
        }
        let inner: usize = scratch.dims[index + 1..].iter().product();
        let c = scratch.dims[index];
        let edge = incoming as usize;
        let message = &to_factor[edge * width..edge * width + c];
        for block in acc.chunks_exact_mut(c * inner) {
            for (state, slice) in block.chunks_exact_mut(inner).enumerate() {
                let term = message[state];
                for slot in slice.iter_mut() {
                    *slot += term;
                }
            }
        }
    }
    let c = scratch.dims[axis];
    let raw = &mut scratch.row[..c];
    if rank == 1 {
        raw.copy_from_slice(acc);
    } else {
        let inner: usize = scratch.dims[axis + 1..].iter().product();
        let outer: usize = scratch.dims[..axis].iter().product();
        let gathered = &mut scratch.gathered[..outer * inner];
        for (state, slot) in raw.iter_mut().enumerate() {
            for block in 0..outer {
                let base = (block * c + state) * inner;
                gathered[block * inner..(block + 1) * inner]
                    .copy_from_slice(&acc[base..base + inner]);
            }
            *slot = if maximum_product {
                maximum(gathered)
            } else {
                logsumexp(gathered)
            };
        }
    }
    let scale = logsumexp(raw);
    let target = edges[axis] as usize;
    let out = &mut to_variable[target * width..target * width + c];
    for (slot, &value) in out.iter_mut().zip(raw.iter()) {
        *slot = value - scale;
    }
}

/// Both passes of the tree schedule, as the two edge-row arrays.
///
/// Leaves to root and then root to leaves, over the breadth-first order and
/// its reverse: the up message of a node is sent after every message into it,
/// which is what the oracle's grouping by height guarantees and what the
/// reverse order guarantees here.
pub fn tree_messages(
    layout: &Layout<'_>,
    maximum_product: bool,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let n_variables = layout.cardinality.len();
    let n_factors = layout.factor_offsets.len() - 1;
    let n_edges = layout.edge_variable.len();
    if layout.variable_offsets.len() != n_variables + 1 {
        return Err("variable_offsets must be one longer than cardinality".to_string());
    }
    if layout.table_offsets.len() != n_factors + 1 {
        return Err("table_offsets must be one longer than the factor count".to_string());
    }
    if layout.variable_edges.len() != n_edges || layout.factor_edges.len() != n_edges {
        return Err("variable_edges and factor_edges must hold one entry per edge".to_string());
    }
    let edge_factor = edge_factors(layout)?;
    let (order, parent_edge) = rooted(layout, &edge_factor)?;

    let mut widest = 1usize;
    for factor in 0..n_factors {
        let start = layout.factor_offsets[factor] as usize;
        let stop = layout.factor_offsets[factor + 1] as usize;
        let size: usize = layout.factor_edges[start..stop]
            .iter()
            .map(|&edge| layout.cardinality[layout.edge_variable[edge as usize] as usize] as usize)
            .product();
        let held = layout.table_offsets[factor + 1] - layout.table_offsets[factor];
        if held != size as i64 {
            return Err(format!(
                "factor {factor} holds {held} table entries for a shape of {size}"
            ));
        }
        widest = widest.max(size);
    }
    let mut scratch = Scratch {
        row: vec![0.0; layout.width],
        acc: vec![0.0; widest],
        gathered: vec![0.0; widest],
        dims: Vec::with_capacity(8),
    };
    let mut to_variable = vec![0.0; n_edges * layout.width];
    let mut to_factor = vec![0.0; n_edges * layout.width];

    for &node in order.iter().rev() {
        if parent_edge[node] < 0 {
            continue;
        }
        let edge = parent_edge[node] as usize;
        if node < n_variables {
            send_from_variable(
                layout,
                node,
                edge,
                &to_variable,
                &mut to_factor,
                &mut scratch,
            );
        } else {
            let factor = node - n_variables;
            let axis = (0..degree(layout, node))
                .find(|&index| edge_of(layout, node, index) == edge)
                .expect("the parent edge is one of the node's edges");
            send_from_factor(
                layout,
                factor,
                axis,
                &to_factor,
                &mut to_variable,
                &mut scratch,
                maximum_product,
            );
        }
    }
    for &node in &order {
        for index in 0..degree(layout, node) {
            let edge = edge_of(layout, node, index);
            if edge as i64 == parent_edge[node] {
                continue;
            }
            if node < n_variables {
                send_from_variable(
                    layout,
                    node,
                    edge,
                    &to_variable,
                    &mut to_factor,
                    &mut scratch,
                );
            } else {
                send_from_factor(
                    layout,
                    node - n_variables,
                    index,
                    &to_factor,
                    &mut to_variable,
                    &mut scratch,
                    maximum_product,
                );
            }
        }
    }
    Ok((to_variable, to_factor))
}

/// `(to_variable, to_factor)` as they cross back: `(n_edges * width,)` each.
type Messages<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);

/// The tree schedule's two passes, for
/// `sal.likelihood.message_passing_rust`.
///
/// Every array is `int64` but `tables`, which is `float64`, and all are
/// borrowed: `as_slice` succeeds only for a C-contiguous array and the
/// wrapper normalizes with `ascontiguousarray`, free when the array already
/// is one.
///
/// Returns the two message arrays flat, `n_edges * width` each, for the
/// caller to view as `(n_edges, width)`. A row's columns past its variable's
/// cardinality are left at zero and never read, as the oracle leaves them.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn tree_message_passing<'py>(
    py: Python<'py>,
    cardinality: PyReadonlyArray1<'_, i64>,
    variable_offsets: PyReadonlyArray1<'_, i64>,
    variable_edges: PyReadonlyArray1<'_, i64>,
    factor_offsets: PyReadonlyArray1<'_, i64>,
    factor_edges: PyReadonlyArray1<'_, i64>,
    edge_variable: PyReadonlyArray1<'_, i64>,
    table_offsets: PyReadonlyArray1<'_, i64>,
    tables: PyReadonlyArray1<'_, f64>,
    width: usize,
    maximum: bool,
) -> PyResult<Messages<'py>> {
    let layout = Layout {
        cardinality: cardinality.as_slice()?,
        variable_offsets: variable_offsets.as_slice()?,
        variable_edges: variable_edges.as_slice()?,
        factor_offsets: factor_offsets.as_slice()?,
        factor_edges: factor_edges.as_slice()?,
        edge_variable: edge_variable.as_slice()?,
        table_offsets: table_offsets.as_slice()?,
        tables: tables.as_slice()?,
        width,
    };
    if layout.cardinality.is_empty() || layout.factor_offsets.len() < 2 {
        return Err(PyValueError::new_err(
            "a factor graph needs at least one variable and one factor",
        ));
    }
    let (to_variable, to_factor) = py
        .detach(|| tree_messages(&layout, maximum))
        .map_err(PyValueError::new_err)?;
    Ok((
        PyArray1::from_vec(py, to_variable),
        PyArray1::from_vec(py, to_factor),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One factor graph as the flat arrays the kernel takes.
    struct Parts {
        cardinality: Vec<i64>,
        variable_offsets: Vec<i64>,
        variable_edges: Vec<i64>,
        factor_offsets: Vec<i64>,
        factor_edges: Vec<i64>,
        edge_variable: Vec<i64>,
        table_offsets: Vec<i64>,
        tables: Vec<f64>,
    }

    impl Parts {
        fn view(&self) -> Layout<'_> {
            Layout {
                cardinality: &self.cardinality,
                variable_offsets: &self.variable_offsets,
                variable_edges: &self.variable_edges,
                factor_offsets: &self.factor_offsets,
                factor_edges: &self.factor_edges,
                edge_variable: &self.edge_variable,
                table_offsets: &self.table_offsets,
                tables: &self.tables,
                width: 2,
            }
        }

        /// The unnormalized log weight of one assignment: `sum_f log psi_f`,
        /// which is `FactorGraph.log_density`.
        fn log_density(&self, states: &[usize]) -> f64 {
            let mut total = 0.0;
            for factor in 0..self.factor_offsets.len() - 1 {
                let start = self.factor_offsets[factor] as usize;
                let stop = self.factor_offsets[factor + 1] as usize;
                let mut index = 0usize;
                for &edge in &self.factor_edges[start..stop] {
                    index = index * 2 + states[self.edge_variable[edge as usize] as usize];
                }
                total += self.tables[self.table_offsets[factor] as usize + index];
            }
            total
        }

        /// A variable's unnormalized log belief: every message into it, summed.
        fn belief(&self, variable: usize, to_variable: &[f64]) -> [f64; 2] {
            let mut total = [0.0f64; 2];
            let start = self.variable_offsets[variable] as usize;
            let stop = self.variable_offsets[variable + 1] as usize;
            for &edge in &self.variable_edges[start..stop] {
                for (state, slot) in total.iter_mut().enumerate() {
                    *slot += to_variable[edge as usize * 2 + state];
                }
            }
            total
        }
    }

    /// A binary chain: a unary table on the first variable and a pairwise
    /// table between each neighbouring pair, the shape `from_hmm` builds.
    fn chain(length: usize) -> Parts {
        let mut parts = Parts {
            cardinality: vec![2; length],
            variable_offsets: Vec::new(),
            variable_edges: Vec::new(),
            factor_offsets: vec![0],
            factor_edges: Vec::new(),
            edge_variable: Vec::new(),
            table_offsets: vec![0],
            tables: Vec::new(),
        };
        let mut incident: Vec<Vec<i64>> = vec![Vec::new(); length];
        let attach = |parts: &mut Parts, incident: &mut Vec<Vec<i64>>, on: &[usize]| {
            for &variable in on {
                let edge = parts.edge_variable.len() as i64;
                parts.edge_variable.push(variable as i64);
                parts.factor_edges.push(edge);
                incident[variable].push(edge);
            }
            parts.factor_offsets.push(parts.factor_edges.len() as i64);
        };
        attach(&mut parts, &mut incident, &[0]);
        parts.tables.extend_from_slice(&[0.3, -0.8]);
        parts.table_offsets.push(parts.tables.len() as i64);
        for position in 1..length {
            attach(&mut parts, &mut incident, &[position - 1, position]);
            for cell in 0..4 {
                parts
                    .tables
                    .push(0.4 * ((cell + position) as f64).sin() - 0.2 * (position as f64));
            }
            parts.table_offsets.push(parts.tables.len() as i64);
        }
        for edges in &incident {
            parts.variable_edges.extend_from_slice(edges);
        }
        parts.variable_offsets.push(0);
        let mut running = 0i64;
        for edges in &incident {
            running += edges.len() as i64;
            parts.variable_offsets.push(running);
        }
        parts
    }

    /// Every assignment of a binary chain, in the order the states index.
    fn assignments(length: usize) -> impl Iterator<Item = Vec<usize>> {
        (0..(1usize << length)).map(move |code| (0..length).map(|v| (code >> v) & 1).collect())
    }

    #[test]
    fn every_marginal_is_the_enumerated_one() {
        for &length in &[1usize, 2, 5, 9] {
            let parts = chain(length);
            let (to_variable, _) = tree_messages(&parts.view(), false).unwrap();
            let scores: Vec<(Vec<usize>, f64)> = assignments(length)
                .map(|states| {
                    let weight = parts.log_density(&states);
                    (states, weight)
                })
                .collect();
            let evidence = logsumexp(&scores.iter().map(|(_, w)| *w).collect::<Vec<_>>());
            for variable in 0..length {
                let belief = parts.belief(variable, &to_variable);
                let normalizer = logsumexp(&belief);
                for (state, &value) in belief.iter().enumerate() {
                    let kept: Vec<f64> = scores
                        .iter()
                        .filter(|(states, _)| states[variable] == state)
                        .map(|(_, w)| *w)
                        .collect();
                    let exact = logsumexp(&kept) - evidence;
                    assert!(
                        (value - normalizer - exact).abs() < 1e-12,
                        "length {length}, variable {variable}, state {state}"
                    );
                }
            }
        }
    }

    #[test]
    fn max_product_recovers_the_enumerated_mode() {
        let length = 7;
        let parts = chain(length);
        let (to_variable, _) = tree_messages(&parts.view(), true).unwrap();
        let best = assignments(length)
            .max_by(|a, b| {
                parts
                    .log_density(a)
                    .partial_cmp(&parts.log_density(b))
                    .expect("no weight is nan")
            })
            .expect("the chain has at least one assignment");
        for (variable, &state) in best.iter().enumerate() {
            let belief = parts.belief(variable, &to_variable);
            assert_eq!(
                usize::from(belief[1] > belief[0]),
                state,
                "variable {variable}"
            );
        }
    }

    #[test]
    fn a_disconnected_graph_is_refused() {
        // Two chains of two, side by side: the walk from the first variable
        // reaches half the nodes.
        let mut left = chain(2);
        let right = chain(2);
        let edges = left.edge_variable.len() as i64;
        let variables = left.cardinality.len() as i64;
        left.cardinality.extend_from_slice(&right.cardinality);
        left.edge_variable
            .extend(right.edge_variable.iter().map(|v| v + variables));
        left.factor_edges
            .extend(right.factor_edges.iter().map(|e| e + edges));
        let offset = *left.factor_offsets.last().expect("one factor at least");
        left.factor_offsets
            .extend(right.factor_offsets[1..].iter().map(|o| o + offset));
        left.variable_edges
            .extend(right.variable_edges.iter().map(|e| e + edges));
        let seen = *left.variable_offsets.last().expect("one variable at least");
        left.variable_offsets
            .extend(right.variable_offsets[1..].iter().map(|o| o + seen));
        let held = *left.table_offsets.last().expect("one table at least");
        left.tables.extend_from_slice(&right.tables);
        left.table_offsets
            .extend(right.table_offsets[1..].iter().map(|o| o + held));

        let error = tree_messages(&left.view(), false).unwrap_err();
        assert!(error.contains("disconnected"), "{error}");
    }

    #[test]
    fn an_edge_on_two_factors_is_refused() {
        let mut parts = chain(3);
        parts.factor_edges[0] = parts.factor_edges[1];
        let error = tree_messages(&parts.view(), false).unwrap_err();
        assert!(error.contains("two factors"), "{error}");
    }

    #[test]
    fn a_table_of_the_wrong_size_is_refused() {
        let mut parts = chain(3);
        parts.tables.push(0.0);
        *parts.table_offsets.last_mut().expect("one table") += 1;
        let error = tree_messages(&parts.view(), false).unwrap_err();
        assert!(error.contains("table entries"), "{error}");
    }

    #[test]
    fn a_degree_one_factor_keeps_an_impossible_state_impossible() {
        // An indicator is a unary table with `-inf` on the states it forbids,
        // and the oracle returns such a table unreduced. A shift by an
        // infinite peak would make it `nan`, which is the case this pins.
        let mut parts = chain(3);
        parts.tables[1] = f64::NEG_INFINITY;
        let (to_variable, _) = tree_messages(&parts.view(), false).unwrap();
        let belief = parts.belief(0, &to_variable);
        assert_eq!(belief[1], f64::NEG_INFINITY);
        assert!(belief[0].is_finite());
    }
}
