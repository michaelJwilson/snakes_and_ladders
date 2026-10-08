//! Iterated conditional modes for the Potts model, ported from
//! `python/sal/search/icm/numba.py::icm_sweeps` (the `numba` oracle, itself
//! pinned bitwise to the Python loop) and exposed as `sal.oxisal.icm_sweeps`.
//!
//! **Bitwise the `numba` kernel.** A site's conditional is read through
//! [`conditional`], the single-site sweep's own reader: `h + sum_j J_ij` in
//! the adjacency's order. The `numba` kernel builds `-h - sum_j J_ij` in the
//! same order; negation is exact and round-to-nearest is symmetric in sign,
//! so each entry is the other's negation bit for bit, and the first maximum
//! here is the first minimum there. The floor reads the caller's uniforms in
//! the oracle's order.
//!
//! **Why Rust beside `numba`.** One compiled path for the annealing loop
//! and its polish (#1368): the Rust loop runs this descent inside its call.

use numpy::PyUntypedArrayMethods;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::potts::conditional;

/// The message both compiled backends and the Python oracle raise where a
/// floor leaves no state to recolour into: `numba.no_survivor`'s.
#[must_use]
pub fn no_survivor(sweep: usize, min_sites: usize) -> String {
    format!(
        "sweep {sweep} left no state holding min_sites={min_sites} sites, so the states below \
         it have nowhere to dissolve into; a floor of at most ceil(n_nodes / n_states) always \
         leaves one"
    )
}

/// The buffers one descent reuses across sweeps.
pub struct IcmScratch {
    /// One site's conditional.
    pub local: Vec<f64>,
    counts: Vec<i64>,
    surviving: Vec<usize>,
    allowed: Vec<usize>,
}

impl IcmScratch {
    /// For `n_states` labels.
    #[must_use]
    pub fn new(n_states: usize) -> Self {
        Self {
            local: vec![0.0; n_states],
            counts: vec![0; n_states],
            surviving: vec![0; n_states],
            allowed: vec![0; n_states],
        }
    }
}

/// Single-site descent in place with the minimum-sites floor; the sweeps run.
///
/// The `numba` kernel's contract: row `orders[sweep % n_rows]` is the
/// visiting order where `n_rows > 0`, index order otherwise; after each
/// sweep, with `min_sites > 0`, every state holding `0 < count < min_sites`
/// sites is dissolved, each of its sites in index order taking
/// `allowed[floor(draws[sweep * n_nodes + node] * m)]` among the ascending
/// surviving states its field allows. `draws` is read only under a floor.
///
/// # Errors
/// Returns `Err` with [`no_survivor`]'s message where a sweep leaves no
/// state at the floor, or where the adjacency names a node outside `state`.
#[allow(clippy::too_many_arguments)]
pub fn icm_sweeps_impl(
    state: &mut [i64],
    field: &[f64],
    offsets: &[usize],
    neighbours: &[i64],
    couplings: &[f64],
    orders: &[i64],
    draws: &[f64],
    n_sweeps: usize,
    stop_when_clean: bool,
    min_sites: usize,
    scratch: &mut IcmScratch,
) -> Result<usize, String> {
    let n_nodes = state.len();
    let n_states = scratch.local.len();
    let n_rows = if n_nodes == 0 {
        0
    } else {
        orders.len() / n_nodes
    };
    let mut sweeps = 0usize;
    for sweep in 0..n_sweeps {
        sweeps += 1;
        let mut changed = false;
        for position in 0..n_nodes {
            let node = if n_rows > 0 {
                orders[(sweep % n_rows) * n_nodes + position] as usize
            } else {
                position
            };
            conditional(
                &mut scratch.local,
                field,
                state,
                offsets,
                neighbours,
                couplings,
                node,
            )?;
            let mut best = 0usize;
            for label in 1..n_states {
                if scratch.local[label] > scratch.local[best] {
                    best = label;
                }
            }
            if best as i64 != state[node] {
                state[node] = best as i64;
                changed = true;
            }
        }
        if min_sites > 0 {
            let IcmScratch {
                counts,
                surviving,
                allowed,
                ..
            } = scratch;
            counts.iter_mut().for_each(|c| *c = 0);
            for &label in state.iter() {
                counts[label as usize] += 1;
            }
            let (mut m, mut below) = (0usize, false);
            for (label, &count) in counts.iter().enumerate() {
                if count >= min_sites as i64 {
                    surviving[m] = label;
                    m += 1;
                } else if count > 0 {
                    below = true;
                }
            }
            if below {
                if m == 0 {
                    return Err(no_survivor(sweeps, min_sites));
                }
                let base = sweep * n_nodes;
                for node in 0..n_nodes {
                    if counts[state[node] as usize] >= min_sites as i64 {
                        continue;
                    }
                    let mut n_allowed = 0usize;
                    for &label in &surviving[..m] {
                        if field[node * n_states + label] > f64::NEG_INFINITY {
                            allowed[n_allowed] = label;
                            n_allowed += 1;
                        }
                    }
                    if n_allowed == 0 {
                        continue;
                    }
                    // `int(draw * n)` truncates toward zero, as `as` does.
                    let pick = (draws[base + node] * n_allowed as f64) as usize;
                    state[node] = allowed[pick.min(n_allowed - 1)] as i64;
                    changed = true;
                }
            }
        }
        if stop_when_clean && !changed {
            break;
        }
    }
    Ok(sweeps)
}

/// PyO3 boundary for [`icm_sweeps_impl`], on `numba.icm_sweeps`' signature;
/// `sal.search.icm` checks the shapes before calling, as it does for
/// `numba`. The GIL is released for the descent.
#[pyfunction]
#[pyo3(signature = (state, field, offsets, neighbours, couplings, orders, draws, n_sweeps, stop_when_clean, min_sites))]
#[allow(clippy::too_many_arguments)]
pub fn icm_sweeps(
    py: Python<'_>,
    mut state: PyReadwriteArray1<'_, i64>,
    field: PyReadonlyArrayDyn<'_, f64>,
    offsets: PyReadonlyArray1<'_, i64>,
    neighbours: PyReadonlyArray1<'_, i64>,
    couplings: PyReadonlyArray1<'_, f64>,
    orders: PyReadonlyArray2<'_, i64>,
    draws: PyReadonlyArray1<'_, f64>,
    n_sweeps: usize,
    stop_when_clean: bool,
    min_sites: usize,
) -> PyResult<usize> {
    let [n_rows, n_states] = *field.shape() else {
        return Err(PyValueError::new_err(format!(
            "field must be 2-D, (n_nodes, n_states), one row per site; got shape {:?}",
            field.shape()
        )));
    };
    if n_rows != state.len() || n_states == 0 {
        return Err(PyValueError::new_err(format!(
            "field has shape ({n_rows}, {n_states}) and state has {} sites",
            state.len()
        )));
    }
    let offsets: Vec<usize> = offsets
        .as_slice()?
        .iter()
        .map(|&value| {
            usize::try_from(value).map_err(|_| PyValueError::new_err("offsets must be >= 0"))
        })
        .collect::<PyResult<_>>()?;
    let state = state.as_slice_mut()?;
    let (field, neighbours, couplings) = (
        field.as_slice()?,
        neighbours.as_slice()?,
        couplings.as_slice()?,
    );
    let (orders, draws) = (orders.as_slice()?, draws.as_slice()?);
    py.detach(|| {
        let mut scratch = IcmScratch::new(n_states);
        icm_sweeps_impl(
            state,
            field,
            &offsets,
            neighbours,
            couplings,
            orders,
            draws,
            n_sweeps,
            stop_when_clean,
            min_sites,
            &mut scratch,
        )
    })
    .map_err(PyValueError::new_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 3-site path 0-1-2, unit couplings, two labels.
    fn path() -> (Vec<usize>, Vec<i64>, Vec<f64>) {
        (vec![0, 1, 3, 4], vec![1, 0, 2, 1], vec![1.0; 4])
    }

    #[test]
    fn a_descent_aligns_a_path_with_its_field() {
        let (offsets, neighbours, couplings) = path();
        // Site 0 prefers label 1 by 3, more than its one bond.
        let field = vec![0.0, 3.0, 0.0, 0.0, 0.0, 0.0];
        let mut state = vec![0, 0, 0];
        let mut scratch = IcmScratch::new(2);
        let sweeps = icm_sweeps_impl(
            &mut state,
            &field,
            &offsets,
            &neighbours,
            &couplings,
            &[],
            &[],
            10,
            true,
            0,
            &mut scratch,
        )
        .unwrap();
        // Sweep 1: site 0 flips (3 > 1); site 1 ties at 1 vs 1 and keeps the
        // first maximum, label 0. Sweep 2 changes nothing.
        assert_eq!(state, vec![1, 0, 0]);
        assert_eq!(sweeps, 2);
    }

    #[test]
    fn a_floor_with_no_survivor_is_refused() {
        let (offsets, neighbours, couplings) = path();
        let field = [0.0; 6];
        let mut scratch = IcmScratch::new(2);
        // Three sites cannot hold four: every state is below the floor.
        let mut state = vec![0, 1, 0];
        let refused = icm_sweeps_impl(
            &mut state,
            &field,
            &offsets,
            &neighbours,
            &couplings,
            &[],
            &[0.5; 30],
            10,
            true,
            4,
            &mut scratch,
        );
        assert!(refused.unwrap_err().contains("left no state"));
    }
}
