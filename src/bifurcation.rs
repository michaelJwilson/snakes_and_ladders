//! Simulated bifurcation's ballistic integration, the whole run in one call (issue #997).
//!
//! `search.bifurcation._integrate_numpy` is the oracle, step for step: each
//! site's force is its field row plus its neighbours' drives summed in the
//! graph's compressed-row order (`PottsGraph.incidence`), and the update,
//! the ramp and the wall at `|x| = 1` are the same arithmetic in the same
//! order. Each node's neighbour sum runs from zero in row order on both
//! sides, so the two agree bit for bit.

use numpy::{PyReadonlyArray1, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// The run's constants.
pub struct Schedule {
    pub steps: usize,
    pub dt: f64,
    pub c0: f64,
    pub a_end: f64,
    pub discrete: bool,
}

/// Integrate `x`, `n_nodes * n_states` row-major, in place.
#[allow(clippy::needless_range_loop)]
pub fn integrate(
    rows: &[f64],
    offsets: &[i64],
    neighbours: &[i64],
    couplings: &[f64],
    x: &mut [f64],
    n_states: usize,
    schedule: &Schedule,
) -> Result<(), String> {
    let n_nodes = offsets.len().saturating_sub(1);
    if rows.len() != n_nodes * n_states || x.len() != rows.len() {
        return Err(format!(
            "{} rows and {} positions for {n_nodes} nodes of {n_states} states",
            rows.len(),
            x.len()
        ));
    }
    if neighbours.len() != couplings.len()
        || offsets.last().copied().unwrap_or(0) as usize != neighbours.len()
    {
        return Err("offsets, neighbours and couplings disagree".to_string());
    }
    if neighbours.iter().any(|&n| n < 0 || n as usize >= n_nodes) {
        return Err("a neighbour lies outside the graph".to_string());
    }
    let mut y = vec![0.0; x.len()];
    let mut drive = vec![0.0; x.len()];
    let mut force = vec![0.0; n_states];
    for step in 0..schedule.steps {
        let ramp = schedule.a_end * step as f64 / schedule.steps as f64;
        for (d, &value) in drive.iter_mut().zip(x.iter()) {
            *d = if schedule.discrete {
                // `np.sign`: zero stays zero.
                if value > 0.0 {
                    1.0
                } else if value < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            } else {
                value
            };
        }
        for node in 0..n_nodes {
            let (start, stop) = (offsets[node] as usize, offsets[node + 1] as usize);
            let row = node * n_states;
            // The neighbours' sum from zero in row order, then the field
            // row, as `_integrate_numpy` forms `rows + summed`.
            force.fill(0.0);
            for entry in start..stop {
                let other = neighbours[entry] as usize * n_states;
                let weight = couplings[entry];
                for state in 0..n_states {
                    force[state] += weight * drive[other + state];
                }
            }
            for state in 0..n_states {
                force[state] += rows[row + state];
            }
            for state in 0..n_states {
                let at = row + state;
                y[at] +=
                    schedule.dt * (-(schedule.a_end - ramp) * x[at] + schedule.c0 * force[state]);
            }
        }
        for at in 0..x.len() {
            x[at] += schedule.dt * schedule.a_end * y[at];
            if x[at].abs() > 1.0 {
                x[at] = if x[at] > 0.0 { 1.0 } else { -1.0 };
                y[at] = 0.0;
            }
        }
    }
    Ok(())
}

/// `_integrate_numpy`'s run in one call, `x` updated in place; see the module docs.
#[pyfunction]
#[pyo3(signature = (rows, offsets, neighbours, couplings, x, n_states, steps, dt, c0, a_end, discrete))]
#[allow(clippy::too_many_arguments)]
pub fn bifurcation_integrate(
    py: Python<'_>,
    rows: PyReadonlyArray1<'_, f64>,
    offsets: PyReadonlyArray1<'_, i64>,
    neighbours: PyReadonlyArray1<'_, i64>,
    couplings: PyReadonlyArray1<'_, f64>,
    mut x: PyReadwriteArray1<'_, f64>,
    n_states: usize,
    steps: usize,
    dt: f64,
    c0: f64,
    a_end: f64,
    discrete: bool,
) -> PyResult<()> {
    let (rows, offsets, neighbours, couplings) = (
        rows.as_slice()?,
        offsets.as_slice()?,
        neighbours.as_slice()?,
        couplings.as_slice()?,
    );
    let x = x.as_slice_mut()?;
    let schedule = Schedule {
        steps,
        dt,
        c0,
        a_end,
        discrete,
    };
    py.detach(|| integrate(rows, offsets, neighbours, couplings, x, n_states, &schedule))
        .map_err(PyValueError::new_err)
}
