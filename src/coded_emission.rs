//! Coded count observations (issue #1340): `encode` and the coded
//! log-emission, exposed to Python as `sal.oxisal.coded_encode` and
//! `sal.oxisal.coded_log_emission`.
//!
//! **Encode** sorts the observed keys, deduplicates them and reads each
//! observation's code by binary search: `inverse[i]` is the code of
//! observation `i`, `-1` where its covariate is zero (unobserved), and
//! `weight[u]` the observations carrying code `u`. A key is a count, or a
//! pair's total in the high 32 bits and successes in the low 32, so the codes
//! run in ascending count order.
//!
//! **The log-emission is `src/dense_emission.rs`'s kernel, not a copy of it.**
//! Its channels take `rows`, the inverse, and tables with one row per code;
//! the covariate term is completed per observation by `coupled.rs`'s
//! [`crate::coupled::ExposureTerm`] and [`crate::coupled::TrialTerm`], in the
//! order the dense route completes it, so a coded score is the dense score.
//!
//! Plain Rust with no PyO3 types in [`encode`], so `cargo test` links it.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::count_mixture::compensated_prefix;
use crate::coupled::{borrowed, exposure_term, trial_term};
use crate::dense_emission::{log_emission_into, ExposureOrder, SuccessChannel, TotalChannel};

/// The codes of `keys`: the sorted distinct observed keys, the inverse, and the weights.
pub struct Encoded {
    /// `U` distinct keys, ascending.
    pub keys: Vec<u64>,
    /// `N` codes, `-1` where the observation is unobserved.
    pub inverse: Vec<i32>,
    /// `U` observation counts, one per code.
    pub weight: Vec<i64>,
}

/// Code `keys`, skipping each observation whose `observed` entry is false.
///
/// # Errors
/// If `observed` is not `N` long, or more than `i32::MAX` codes would be needed.
pub fn encode(keys: &[u64], observed: Option<&[bool]>) -> Result<Encoded, String> {
    if let Some(mask) = observed {
        if mask.len() != keys.len() {
            return Err(format!(
                "the observed mask has {} entries, expected N = {}",
                mask.len(),
                keys.len()
            ));
        }
    }
    let seen = |i: usize| observed.is_none_or(|mask| mask[i]);
    let mut distinct: Vec<u64> = (0..keys.len())
        .filter(|&i| seen(i))
        .map(|i| keys[i])
        .collect();
    distinct.par_sort_unstable();
    distinct.dedup();
    if i32::try_from(distinct.len()).is_err() {
        return Err("more distinct counts than an int32 inverse codes".to_string());
    }
    let inverse: Vec<i32> = (0..keys.len())
        .into_par_iter()
        .map(|i| {
            if seen(i) {
                // Present by construction; the cast is checked above.
                distinct.binary_search(&keys[i]).map_or(-1, |u| u as i32)
            } else {
                -1
            }
        })
        .collect();
    let mut weight = vec![0_i64; distinct.len()];
    for &u in &inverse {
        if let Ok(u) = usize::try_from(u) {
            weight[u] += 1;
        }
    }
    Ok(Encoded {
        keys: distinct,
        inverse,
        weight,
    })
}

/// [`encode`] as a Python binding: `(keys, inverse, weight)`.
///
/// `observed`, where given, is `N` bytes, nonzero where the observation is
/// observed. The GIL is released for the whole pass.
///
/// # Errors
/// `ValueError` naming the first violated precondition.
#[pyfunction]
#[pyo3(signature = (keys, observed=None))]
#[allow(clippy::type_complexity)]
pub fn coded_encode<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'_, u64>,
    observed: Option<PyReadonlyArray1<'_, bool>>,
) -> PyResult<(
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i64>>,
)> {
    let keys = borrowed(&keys, "keys")?;
    let mask = match &observed {
        None => None,
        Some(mask) => Some(borrowed(mask, "observed")?),
    };
    let encoded = py
        .detach(|| encode(keys, mask))
        .map_err(PyValueError::new_err)?;
    Ok((
        encoded.keys.into_pyarray(py),
        encoded.inverse.into_pyarray(py),
        encoded.weight.into_pyarray(py),
    ))
}

/// The coded log-emission, `out[k * N + i]`, through [`log_emission_into`].
///
/// `inverse` is `N` codes; `total_rows` and `success_rows`, where given,
/// are each channel's own `N` rows, `-1` unobserved, so a channel's table
/// has one row per distinct value of that channel rather than per code
/// (a pair's totals, or a count shared across labels); absent, a channel
/// reads `inverse`. `totals` and `total_table` are the distinct totals and
/// their table, one row per row index, and `exposure`, `dispersion`
/// and `mean` its exposure term; `successes`, `success_table` and the trial
/// arguments the second channel, as `class_posteriors` takes them. Every
/// array crosses once, contiguous and borrowed; the GIL is released.
///
/// # Errors
/// `ValueError` naming the first violated precondition.
#[pyfunction]
#[pyo3(signature = (n_states, inverse, out, total_rows=None, success_rows=None, totals=None, total_table=None, exposure=None, dispersion=None, mean=None, successes=None, success_table=None, trials=None, failure_table=None, trial_table=None, log_factorial=None, log_rate=None))]
#[allow(clippy::too_many_arguments)]
pub fn coded_log_emission(
    py: Python<'_>,
    n_states: usize,
    inverse: PyReadonlyArray1<'_, i32>,
    mut out: PyReadwriteArray1<'_, f64>,
    total_rows: Option<PyReadonlyArray1<'_, i32>>,
    success_rows: Option<PyReadonlyArray1<'_, i32>>,
    totals: Option<PyReadonlyArray1<'_, u32>>,
    total_table: Option<PyReadonlyArray1<'_, f64>>,
    exposure: Option<PyReadonlyArray1<'_, f64>>,
    dispersion: Option<PyReadonlyArray1<'_, f64>>,
    mean: Option<PyReadonlyArray1<'_, f64>>,
    successes: Option<PyReadonlyArray1<'_, u32>>,
    success_table: Option<PyReadonlyArray1<'_, f64>>,
    trials: Option<PyReadonlyArray1<'_, u32>>,
    failure_table: Option<PyReadonlyArray1<'_, f64>>,
    trial_table: Option<PyReadonlyArray1<'_, f64>>,
    log_factorial: Option<PyReadonlyArray1<'_, f64>>,
    log_rate: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<()> {
    let rows = borrowed(&inverse, "inverse")?;
    let total_own = match &total_rows {
        Some(own) => Some(borrowed(own, "total_rows")?),
        None => None,
    };
    let success_own = match &success_rows {
        Some(own) => Some(borrowed(own, "success_rows")?),
        None => None,
    };
    if [total_own, success_own]
        .iter()
        .flatten()
        .any(|own| own.len() != rows.len())
    {
        return Err(PyValueError::new_err(format!(
            "a channel's rows are N = {} long, as the inverse",
            rows.len()
        )));
    }
    let total_index = total_own.unwrap_or(rows);
    let success_index = success_own.unwrap_or(rows);
    let total = match (&totals, &total_table) {
        (None, None) => None,
        (Some(counts), Some(table)) => Some(TotalChannel {
            counts: borrowed(counts, "totals")?,
            rows: Some(total_index),
            table: borrowed(table, "total_table")?,
            exposure: exposure_term(&exposure, &dispersion, &mean)?,
            order: ExposureOrder::Family,
        }),
        _ => {
            return Err(PyValueError::new_err(
                "the first channel takes totals and total_table together",
            ))
        }
    };
    let second = match (&successes, &success_table) {
        (None, None) => None,
        (Some(counts), Some(table)) => Some(SuccessChannel {
            counts: borrowed(counts, "successes")?,
            rows: Some(success_index),
            table: borrowed(table, "success_table")?,
            trials: trial_term(
                &trials,
                &failure_table,
                &trial_table,
                &log_factorial,
                &log_rate,
            )?,
        }),
        _ => {
            return Err(PyValueError::new_err(
                "the second channel takes successes and success_table together",
            ))
        }
    };
    let out = out
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("out must be C-contiguous"))?;
    let n = rows.len();
    py.detach(|| log_emission_into(n_states, n, total.as_ref(), second.as_ref(), out))
        .map_err(PyValueError::new_err)
}

/// `out[k] = sum_u values[k, u] * W[k, u]`, `W` the per-state `bincount` of
/// `weights` over `index` (`-1` contributes nothing). `weights` is `(n,)`,
/// shared by every state, `(K, n)`, or absent (each weight 1). The bincount
/// and the dot are sequential in `u` and `i`, so the sum's order is stated:
/// a code with zero total weight is skipped, so `0 * -inf` never enters.
///
/// # Errors
/// A shape that disagrees, or an index outside `values`' columns.
pub fn weighted_sum(
    n_states: usize,
    values: &[f64],
    index: &[i32],
    weights: Option<&[f64]>,
) -> Result<Vec<f64>, String> {
    let n = index.len();
    if n_states == 0 || !values.len().is_multiple_of(n_states) {
        return Err(format!(
            "values hold {} entries, not a multiple of {n_states} states",
            values.len()
        ));
    }
    let m = values.len() / n_states;
    let per_state = match weights {
        None => false,
        Some(w) if w.len() == n => false,
        Some(w) if w.len() == n * n_states => true,
        Some(w) => {
            return Err(format!(
                "weights are (n,) or (K, n): {} entries for n = {n}, K = {n_states}",
                w.len()
            ))
        }
    };
    if let Some(bad) = index.iter().find(|&&u| u >= 0 && u as usize >= m) {
        return Err(format!("index {bad} is outside {m} columns"));
    }
    let count = |row: usize| -> Vec<f64> {
        let mut bins = vec![0.0_f64; m];
        for (i, &u) in index.iter().enumerate() {
            if u >= 0 {
                bins[u as usize] += weights.map_or(1.0, |w| w[row * n + i]);
            }
        }
        bins
    };
    let dot = |k: usize, bins: &[f64]| -> f64 {
        let row = &values[k * m..(k + 1) * m];
        let mut acc = 0.0_f64;
        for (v, &b) in row.iter().zip(bins) {
            if b != 0.0 {
                acc += v * b;
            }
        }
        acc
    };
    if per_state {
        Ok((0..n_states)
            .into_par_iter()
            .map(|k| dot(k, &count(k)))
            .collect())
    } else {
        let bins = count(0);
        Ok((0..n_states)
            .into_par_iter()
            .map(|k| dot(k, &bins))
            .collect())
    }
}

/// `coded_weighted_sum(n_states, values, index, weights=None)`: [`weighted_sum`]
/// over borrowed, contiguous arrays with the GIL released.
///
/// # Errors
/// `ValueError` naming the first violated precondition.
#[pyfunction]
#[pyo3(signature = (n_states, values, index, weights=None))]
pub fn coded_weighted_sum<'py>(
    py: Python<'py>,
    n_states: usize,
    values: PyReadonlyArray1<'py, f64>,
    index: PyReadonlyArray1<'py, i32>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = borrowed(&values, "values")?;
    let index = borrowed(&index, "index")?;
    let weights = match &weights {
        Some(w) => Some(borrowed(w, "weights")?),
        None => None,
    };
    let out = py
        .detach(|| weighted_sum(n_states, values, index, weights))
        .map_err(PyValueError::new_err)?;
    Ok(PyArray1::from_vec(py, out))
}

/// Observations per tile of the partials' parallel pass, per state.
const PARTIAL_TILE: usize = 4096;

/// `D(x, m) = digamma(x + m) - digamma(x) = sum_{j < m} 1 / (x + j)` for each
/// `x` of `shapes`, read at each of `points`: `[k * points.len() + u]`.
///
/// The sum is `count_mixture`'s [`compensated_prefix`], the one `Rising`'s
/// reciprocal table is, walked once per state up to the largest point and
/// read at the points only, so a table holds one entry per distinct count.
fn digamma_rising_at(shapes: &[f64], points: &[u32]) -> Vec<f64> {
    let width = points.len();
    let extent = points.iter().copied().max().unwrap_or(0) as usize;
    let mut out = vec![0.0; shapes.len() * width];
    if width == 0 {
        return out;
    }
    out.par_chunks_mut(width)
        .zip(shapes.par_iter())
        .for_each(|(row, &x)| {
            let prefix = compensated_prefix(extent, |j| 1.0 / (x + j as f64));
            for (value, &m) in row.iter_mut().zip(points) {
                *value = prefix[m as usize];
            }
        });
    out
}

/// `D(x, m)` for `m` in `0..=extent`, state-major: `[k * (extent + 1) + m]`.
fn digamma_rising_dense(shapes: &[f64], extent: usize) -> Vec<f64> {
    let width = extent + 1;
    let mut out = vec![0.0; shapes.len() * width];
    out.par_chunks_mut(width)
        .zip(shapes.par_iter())
        .for_each(|(row, &x)| {
            row.copy_from_slice(&compensated_prefix(extent, |j| 1.0 / (x + j as f64)));
        });
    out
}

/// A state-major `(K, n)` output split into `(state, first observation, tile)`.
fn tiles(out: &mut [f64], n: usize) -> Vec<(usize, usize, &mut [f64])> {
    out.chunks_mut(n)
        .enumerate()
        .flat_map(|(k, row)| {
            row.chunks_mut(PARTIAL_TILE)
                .enumerate()
                .map(move |(t, tile)| (k, t * PARTIAL_TILE, tile))
        })
        .collect()
}

/// The negative-binomial channel of the partials.
pub struct NbPartials<'a> {
    /// Distinct totals, one per row.
    pub totals: &'a [u32],
    /// `N` rows into `totals`, `-1` unobserved.
    pub rows: &'a [i32],
    /// `N` exposures `c`, 1 where absent; `c = 0` is unobserved.
    pub exposure: Option<&'a [f64]>,
    /// `N` shift factors `exp(shift[label])`, where shifted.
    pub factor: Option<&'a [f64]>,
    /// `K` dispersions `r`; `inf` is the Poisson.
    pub dispersion: &'a [f64],
    /// `K` means `mu` at unit exposure.
    pub mean: &'a [f64],
}

/// `d/dr`, `d/dmu` and, given `shift`, `d/dshift` of the NB log pmf, `(K, N)` state-major.
///
/// Per observation, in `sal.emissions.coded._partials_numpy`'s order:
/// `lam = mu (c f)`, `d/dr = (D(r, y) - log1p(lam / r)) + (lam - y) / (r + lam)`,
/// `d/dmu = (r (y - lam)) / (mu (r + lam))`, `d/ds = (r (y - lam)) / (r + lam)`;
/// at `r = inf`, `0`, `y / mu - c f` and `y - lam`. An unobserved observation is 0.
///
/// # Errors
/// A length that disagrees, or a row past the totals.
pub fn nb_partials_into(
    channel: &NbPartials<'_>,
    dispersion: &mut [f64],
    mean: &mut [f64],
    shift: Option<&mut [f64]>,
) -> Result<(), String> {
    let k = channel.dispersion.len();
    let n = channel.rows.len();
    if channel.mean.len() != k {
        return Err(format!("{k} dispersions but {} means", channel.mean.len()));
    }
    if [channel.exposure, channel.factor]
        .iter()
        .flatten()
        .any(|v| v.len() != n)
        || dispersion.len() != k * n
        || mean.len() != k * n
        || shift.as_ref().is_some_and(|s| s.len() != k * n)
    {
        return Err(format!(
            "the first channel's arrays are N = {n} or K * N = {}",
            k * n
        ));
    }
    let width = channel.totals.len();
    if let Some(bad) = channel
        .rows
        .iter()
        .find(|&&u| u >= 0 && u as usize >= width)
    {
        return Err(format!("row {bad} is past the {width} totals"));
    }
    if n == 0 {
        return Ok(());
    }
    let shapes: Vec<f64> = channel
        .dispersion
        .iter()
        .map(|&r| if r.is_finite() { r } else { 1.0 })
        .collect();
    let rising = digamma_rising_at(&shapes, channel.totals);
    let third: Vec<Option<&mut [f64]>> = match shift {
        Some(s) => tiles(s, n).into_iter().map(|(_, _, t)| Some(t)).collect(),
        None => std::iter::repeat_with(|| None)
            .take(k * n.div_ceil(PARTIAL_TILE))
            .collect(),
    };
    tiles(dispersion, n)
        .into_par_iter()
        .zip(tiles(mean, n).into_par_iter())
        .zip(third.into_par_iter())
        .for_each(|(((state, from, dr), (_, _, dmu)), mut ds)| {
            let r = channel.dispersion[state];
            let mu = channel.mean[state];
            let finite = r.is_finite();
            let table = &rising[state * width..(state + 1) * width];
            for j in 0..dr.len() {
                let i = from + j;
                let u = channel.rows[i];
                let c = channel.exposure.map_or(1.0, |e| e[i]);
                let (a, b, s) = if u < 0 || c == 0.0 {
                    (0.0, 0.0, 0.0)
                } else {
                    let row = u as usize;
                    let y = f64::from(channel.totals[row]);
                    let scale = channel.factor.map_or(c, |f| c * f[i]);
                    let lam = mu * scale;
                    if finite {
                        let numerator = r * (y - lam);
                        (
                            (table[row] - (lam / r).ln_1p()) + (lam - y) / (r + lam),
                            numerator / (mu * (r + lam)),
                            numerator / (r + lam),
                        )
                    } else {
                        (0.0, y / mu - scale, y - lam)
                    }
                };
                dr[j] = a;
                dmu[j] = b;
                if let Some(ds) = ds.as_deref_mut() {
                    ds[j] = s;
                }
            }
        });
    Ok(())
}

/// The beta-binomial channel of the partials.
pub struct BbPartials<'a> {
    /// Distinct successes, one per row.
    pub successes: &'a [u32],
    /// `N` rows into `successes`, `-1` unobserved.
    pub rows: &'a [i32],
    /// `N` trial counts per observation, or `K` per state.
    pub trials: &'a [u32],
    /// Whether `trials` is per observation.
    pub per_observation: bool,
    /// `K` shapes `a`.
    pub alpha: &'a [f64],
    /// `K` shapes `b`.
    pub beta: &'a [f64],
    /// `K` rates `p`, where the family is rate/concentration.
    pub rate: Option<&'a [f64]>,
}

/// The beta-binomial's partials, `(K, N)` state-major: `d/da` and `d/db`, or,
/// given `rate`, `d/dp` and `d/dtau`.
///
/// Per observation, in `_partials_numpy`'s order: `da = D(a, z) - D(a + b, n)`,
/// `db = D(b, n - z) - D(a + b, n)`, `d/dp = tau (da - db)`,
/// `d/dtau = p da + (1 - p) db`; at `tau = inf` (the shapes read as 1),
/// `z / p - (n - z) / (1 - p)` and `0`. An observation unobserved, with no
/// trials or past its trials is 0.
///
/// # Errors
/// A length that disagrees, or a row past the successes.
pub fn bb_partials_into(
    channel: &BbPartials<'_>,
    first: &mut [f64],
    second: &mut [f64],
) -> Result<(), String> {
    let k = channel.alpha.len();
    let n = channel.rows.len();
    let trial_len = if channel.per_observation { n } else { k };
    if channel.beta.len() != k
        || channel.rate.is_some_and(|p| p.len() != k)
        || channel.trials.len() != trial_len
        || first.len() != k * n
        || second.len() != k * n
    {
        return Err(format!(
            "the second channel's parameters are K = {k}, its trials {trial_len} and outputs K * N = {}",
            k * n
        ));
    }
    let width = channel.successes.len();
    if let Some(bad) = channel
        .rows
        .iter()
        .find(|&&u| u >= 0 && u as usize >= width)
    {
        return Err(format!("row {bad} is past the {width} successes"));
    }
    if n == 0 {
        return Ok(());
    }
    let limit: Vec<bool> = channel
        .alpha
        .iter()
        .zip(channel.beta)
        .map(|(&a, &b)| (a + b).is_infinite())
        .collect();
    let safe = |values: &[f64]| -> Vec<f64> {
        values
            .iter()
            .zip(&limit)
            .map(|(&v, &l)| if l { 1.0 } else { v })
            .collect()
    };
    let (a, b) = (safe(channel.alpha), safe(channel.beta));
    let ab: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    let extent = channel.trials.iter().copied().max().unwrap_or(0) as usize;
    let span = extent + 1;
    let success = digamma_rising_at(&a, channel.successes);
    let failure = digamma_rising_dense(&b, extent);
    let held = digamma_rising_dense(&ab, extent);
    tiles(first, n)
        .into_par_iter()
        .zip(tiles(second, n).into_par_iter())
        .for_each(|((state, from, one), (_, _, two))| {
            let tau = channel.alpha[state] + channel.beta[state];
            let p = channel.rate.map(|p| p[state]);
            for j in 0..one.len() {
                let i = from + j;
                let u = channel.rows[i];
                let trials = if channel.per_observation {
                    channel.trials[i]
                } else {
                    channel.trials[state]
                };
                let z = if u < 0 {
                    None
                } else {
                    Some(channel.successes[u as usize])
                };
                let (x, y) = match z {
                    Some(z) if z <= trials && trials > 0 => {
                        let whole = held[state * span + trials as usize];
                        let da = success[state * width + u as usize] - whole;
                        let db = failure[state * span + (trials - z) as usize] - whole;
                        match p {
                            None => (da, db),
                            Some(p) if limit[state] => {
                                let (zf, nf) = (f64::from(z), f64::from(trials));
                                (zf / p - (nf - zf) / (1.0 - p), 0.0)
                            }
                            Some(p) => (tau * (da - db), p * da + (1.0 - p) * db),
                        }
                    }
                    _ => (0.0, 0.0),
                };
                one[j] = x;
                two[j] = y;
            }
        });
    Ok(())
}

/// Each parameter's partial of every observation's log-density, written into
/// the `(K, N)` state-major outputs given: the first channel's into
/// `out_dispersion`, `out_mean` and `out_shift`, the second's into
/// `out_first` and `out_second`. Every array crosses once, contiguous and
/// borrowed; the GIL is released for both channels.
///
/// # Errors
/// `ValueError` naming the first violated precondition.
#[pyfunction]
#[pyo3(signature = (total_rows=None, totals=None, exposure=None, factor=None, dispersion=None, mean=None, out_dispersion=None, out_mean=None, out_shift=None, success_rows=None, successes=None, trials=None, per_observation=true, alpha=None, beta=None, rate=None, out_first=None, out_second=None))]
#[allow(clippy::too_many_arguments)]
pub fn coded_log_emission_partials(
    py: Python<'_>,
    total_rows: Option<PyReadonlyArray1<'_, i32>>,
    totals: Option<PyReadonlyArray1<'_, u32>>,
    exposure: Option<PyReadonlyArray1<'_, f64>>,
    factor: Option<PyReadonlyArray1<'_, f64>>,
    dispersion: Option<PyReadonlyArray1<'_, f64>>,
    mean: Option<PyReadonlyArray1<'_, f64>>,
    out_dispersion: Option<PyReadwriteArray1<'_, f64>>,
    out_mean: Option<PyReadwriteArray1<'_, f64>>,
    out_shift: Option<PyReadwriteArray1<'_, f64>>,
    success_rows: Option<PyReadonlyArray1<'_, i32>>,
    successes: Option<PyReadonlyArray1<'_, u32>>,
    trials: Option<PyReadonlyArray1<'_, u32>>,
    per_observation: bool,
    alpha: Option<PyReadonlyArray1<'_, f64>>,
    beta: Option<PyReadonlyArray1<'_, f64>>,
    rate: Option<PyReadonlyArray1<'_, f64>>,
    out_first: Option<PyReadwriteArray1<'_, f64>>,
    out_second: Option<PyReadwriteArray1<'_, f64>>,
) -> PyResult<()> {
    fn optional<'a, T: numpy::Element>(
        array: &'a Option<PyReadonlyArray1<'_, T>>,
        name: &str,
    ) -> PyResult<Option<&'a [T]>> {
        array.as_ref().map(|a| borrowed(a, name)).transpose()
    }
    fn writable<'a>(
        array: &'a mut Option<PyReadwriteArray1<'_, f64>>,
        name: &str,
    ) -> PyResult<Option<&'a mut [f64]>> {
        array
            .as_mut()
            .map(|a| {
                a.as_slice_mut()
                    .map_err(|_| PyValueError::new_err(format!("{name} must be C-contiguous")))
            })
            .transpose()
    }
    let (mut out_dispersion, mut out_mean, mut out_shift) = (out_dispersion, out_mean, out_shift);
    let (mut out_first, mut out_second) = (out_first, out_second);
    let first = match (
        optional(&total_rows, "total_rows")?,
        optional(&totals, "totals")?,
        optional(&dispersion, "dispersion")?,
        optional(&mean, "mean")?,
        writable(&mut out_dispersion, "out_dispersion")?,
        writable(&mut out_mean, "out_mean")?,
    ) {
        (None, None, None, None, None, None) => None,
        (Some(rows), Some(totals), Some(dispersion), Some(mean), Some(dr), Some(dmu)) => Some((
            NbPartials {
                totals,
                rows,
                exposure: optional(&exposure, "exposure")?,
                factor: optional(&factor, "factor")?,
                dispersion,
                mean,
            },
            dr,
            dmu,
            writable(&mut out_shift, "out_shift")?,
        )),
        _ => {
            return Err(PyValueError::new_err(
                "the first channel takes total_rows, totals, dispersion, mean, out_dispersion and out_mean together",
            ))
        }
    };
    let second = match (
        optional(&success_rows, "success_rows")?,
        optional(&successes, "successes")?,
        optional(&trials, "trials")?,
        optional(&alpha, "alpha")?,
        optional(&beta, "beta")?,
        writable(&mut out_first, "out_first")?,
        writable(&mut out_second, "out_second")?,
    ) {
        (None, None, None, None, None, None, None) => None,
        (Some(rows), Some(successes), Some(trials), Some(alpha), Some(beta), Some(one), Some(two)) => {
            Some((
                BbPartials {
                    successes,
                    rows,
                    trials,
                    per_observation,
                    alpha,
                    beta,
                    rate: optional(&rate, "rate")?,
                },
                one,
                two,
            ))
        }
        _ => {
            return Err(PyValueError::new_err(
                "the second channel takes success_rows, successes, trials, alpha, beta, out_first and out_second together",
            ))
        }
    };
    py.detach(|| -> Result<(), String> {
        if let Some((channel, dr, dmu, ds)) = first {
            nb_partials_into(&channel, dr, dmu, ds)?;
        }
        if let Some((channel, one, two)) = second {
            bb_partials_into(&channel, one, two)?;
        }
        Ok(())
    })
    .map_err(PyValueError::new_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_codes_ascending_and_skips_the_unobserved() {
        let keys = [5, 2, 5, 9, 2, 7];
        let observed = [true, true, true, true, true, false];
        let encoded = encode(&keys, Some(&observed)).unwrap();
        assert_eq!(encoded.keys, [2, 5, 9]);
        assert_eq!(encoded.inverse, [1, 0, 1, 2, 0, -1]);
        assert_eq!(encoded.weight, [2, 2, 1]);
    }

    #[test]
    fn a_coded_gather_is_the_dense_read_bitwise() {
        // Two states, counts 0..3, dense against coded on the distinct counts.
        let table = [0.0, 10.0, 1.0, 11.0, 2.0, 12.0];
        let counts = [2_u32, 0, 1, 2];
        let dense = TotalChannel {
            counts: &counts,
            rows: None,
            table: &table,
            exposure: None,
            order: ExposureOrder::Family,
        };
        let mut expected = vec![0.0; 8];
        log_emission_into(2, 4, Some(&dense), None, &mut expected).unwrap();
        let keys: Vec<u64> = counts.iter().map(|&c| u64::from(c)).collect();
        let encoded = encode(&keys, None).unwrap();
        let distinct: Vec<u32> = encoded.keys.iter().map(|&k| k as u32).collect();
        let coded_table: Vec<f64> = distinct
            .iter()
            .flat_map(|&c| table[c as usize * 2..][..2].to_vec())
            .collect();
        let coded = TotalChannel {
            counts: &distinct,
            rows: Some(&encoded.inverse),
            table: &coded_table,
            exposure: None,
            order: ExposureOrder::Family,
        };
        let mut out = vec![0.0; 8];
        log_emission_into(2, 4, Some(&coded), None, &mut out).unwrap();
        assert_eq!(
            out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn an_unobserved_code_scores_zero_and_a_stray_row_is_refused() {
        let table = [3.0, 4.0];
        let counts = [7_u32];
        let rows = [0, -1];
        let channel = TotalChannel {
            counts: &counts,
            rows: Some(&rows),
            table: &table,
            exposure: None,
            order: ExposureOrder::Family,
        };
        let mut out = vec![1.0; 4];
        log_emission_into(2, 2, Some(&channel), None, &mut out).unwrap();
        assert_eq!(out, [3.0, 0.0, 4.0, 0.0]);
        let stray = [1, 0];
        let channel = TotalChannel {
            rows: Some(&stray),
            ..channel
        };
        let error = log_emission_into(2, 2, Some(&channel), None, &mut out).unwrap_err();
        assert!(error.contains("past the 1 codes"), "{error}");
    }

    #[test]
    fn a_weighted_sum_is_the_sequential_bincount_dot() {
        let values = [1.0, 2.0, f64::NEG_INFINITY, 4.0, 5.0, 6.0];
        let index = [0, 1, -1, 1, 0];
        assert_eq!(
            weighted_sum(2, &values, &index, None).unwrap(),
            vec![1.0 * 2.0 + 2.0 * 2.0, 4.0 * 2.0 + 5.0 * 2.0]
        );
        let w = [0.5, 0.25, 9.0, 0.25, 0.5];
        assert_eq!(
            weighted_sum(2, &values, &index, Some(&w)).unwrap(),
            vec![1.0 + 1.0, 4.0 + 2.5]
        );
        assert!(weighted_sum(2, &values, &[3], None).is_err());
    }

    #[test]
    fn the_rising_table_is_count_mixtures_reciprocal_bitwise() {
        let shapes = [0.5, 7.0, 1e8];
        let points = [0_u32, 3, 40];
        let at = digamma_rising_at(&shapes, &points);
        let rising = crate::count_mixture::Rising::new(&shapes, 40);
        for (k, _) in shapes.iter().enumerate() {
            for (u, &m) in points.iter().enumerate() {
                let want = rising.reciprocal[m as usize * shapes.len() + k];
                assert_eq!(at[k * points.len() + u].to_bits(), want.to_bits());
            }
        }
    }

    #[test]
    fn the_poisson_limit_and_the_unobserved_take_their_own_partials() {
        let totals = [0_u32, 4];
        let rows = [1, -1, 0];
        let exposure = [0.5, 1.0, 2.0];
        let channel = NbPartials {
            totals: &totals,
            rows: &rows,
            exposure: Some(&exposure),
            factor: None,
            dispersion: &[f64::INFINITY],
            mean: &[6.0],
        };
        let (mut dr, mut dmu) = (vec![9.0; 3], vec![9.0; 3]);
        nb_partials_into(&channel, &mut dr, &mut dmu, None).unwrap();
        assert_eq!(dr, [0.0, 0.0, 0.0]);
        assert_eq!(dmu, [4.0 / 6.0 - 0.5, 0.0, -2.0]);
        let stray = [2, 0, 0];
        let channel = NbPartials {
            rows: &stray,
            ..channel
        };
        assert!(nb_partials_into(&channel, &mut dr, &mut dmu, None).is_err());
    }
}
