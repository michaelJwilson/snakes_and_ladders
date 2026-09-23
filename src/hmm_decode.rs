//! Viterbi over a batch of equal-length sequences, sequences in parallel (issue #997).
//!
//! hmmlearn's `decode` sets the goal: 0.26 s for 10^6 positions of a
//! three-state Gaussian HMM. Each sequence's path is independent given the
//! parameters, so the sequences are split over `rayon` with each writing its
//! own disjoint slice of the output; the per-sequence log-probabilities are
//! summed in sequence order afterwards, so the total is the serial one.
//!
//! The recursion is the textbook one in log space: `delta_t(j) = max_i
//! delta_{t-1}(i) + ln A_ij + ln b_j(x_t)`, a back-pointer per position and
//! state, and a trace back from the best last state. A tie goes to the lower
//! state, as `numpy.argmax` breaks it, so the NumPy route is the oracle
//! position for position.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::hmm_stream::{count_family, CountFamily};
use crate::special::STIRLING_FROM;

/// What an observation is scored by.
pub enum Emission<'a> {
    /// `m * n_symbols` log-probabilities, row-major; observations are symbols.
    Categorical(&'a [f64]),
    /// Per-state means and standard deviations.
    Gaussian { mean: &'a [f64], scale: &'a [f64] },
    /// A count family, scored directly.
    Count(CountFamily<'a>),
}

impl Emission<'_> {
    #[inline]
    fn log_density(&self, x: f64, out: &mut [f64]) {
        match self {
            Emission::Categorical(log_emission) => {
                let k = log_emission.len() / out.len();
                for (state, o) in out.iter_mut().enumerate() {
                    *o = log_emission[state * k + x as usize];
                }
            }
            Emission::Gaussian { mean, scale } => {
                let half_log_two_pi = 0.5 * (2.0 * std::f64::consts::PI).ln();
                for (state, o) in out.iter_mut().enumerate() {
                    let z = (x - mean[state]) / scale[state];
                    *o = -half_log_two_pi - scale[state].ln() - 0.5 * z * z;
                }
            }
            Emission::Count(family) => family.log_density(x, None, out),
        }
    }
}

/// An observation as the densities read it: real values, or symbols and
/// counts borrowed as the `int64` NumPy holds them rather than copied.
pub trait Observation: Copy + Sync {
    fn value(self) -> f64;
}

impl Observation for f64 {
    #[inline]
    fn value(self) -> f64 {
        self
    }
}

impl Observation for i64 {
    #[inline]
    fn value(self) -> f64 {
        self as f64
    }
}

/// The best path of one sequence into `path`, and its joint log-probability.
fn decode_one<T: Observation>(
    row: &[T],
    m: usize,
    log_initial: &[f64],
    log_transition: &[f64],
    emission: &Emission<'_>,
    path: &mut [i64],
) -> f64 {
    let length = row.len();
    let mut delta = vec![0.0; m];
    let mut next = vec![0.0; m];
    let mut emitted = vec![0.0; m];
    let mut back = vec![0u32; length * m];
    emission.log_density(row[0].value(), &mut emitted);
    for state in 0..m {
        delta[state] = log_initial[state] + emitted[state];
    }
    for t in 1..length {
        emission.log_density(row[t].value(), &mut emitted);
        let pointers = &mut back[t * m..][..m];
        for j in 0..m {
            let (mut best, mut from) = (f64::NEG_INFINITY, 0u32);
            for i in 0..m {
                let score = delta[i] + log_transition[i * m + j];
                if score > best {
                    best = score;
                    from = i as u32;
                }
            }
            next[j] = best + emitted[j];
            pointers[j] = from;
        }
        std::mem::swap(&mut delta, &mut next);
    }
    let (mut best, mut state) = (f64::NEG_INFINITY, 0usize);
    for (j, &value) in delta.iter().enumerate() {
        if value > best {
            best = value;
            state = j;
        }
    }
    for t in (0..length).rev() {
        path[t] = state as i64;
        state = back[t * m + state] as usize;
    }
    best
}

/// Every sequence's best path, `n_sequences * length`, and the total
/// joint log-probability.
pub fn decode<T: Observation>(
    observations: &[T],
    length: usize,
    log_initial: &[f64],
    log_transition: &[f64],
    emission: &Emission<'_>,
) -> Result<(Vec<i64>, f64), String> {
    let m = log_initial.len();
    if m == 0 || log_transition.len() != m * m {
        return Err(format!(
            "{m} states need a {m}x{m} transition, got {} entries",
            log_transition.len()
        ));
    }
    if length == 0 || !observations.len().is_multiple_of(length) {
        return Err(format!(
            "{} observations are not rows of {length}",
            observations.len()
        ));
    }
    let mut states = vec![0i64; observations.len()];
    let scores: Vec<f64> = states
        .par_chunks_mut(length)
        .zip(observations.par_chunks(length))
        .map(|(path, row)| decode_one(row, m, log_initial, log_transition, emission, path))
        .collect();
    Ok((states, scores.iter().sum()))
}

/// Viterbi paths for a batch; see the module docs.
///
/// `family` is `categorical` (`parameters` the `m * n_symbols` log
/// emission), `gaussian` (`mean, scale`), or a count family as
/// `count_em_step` takes it. `observations` is a C-contiguous `int64` or
/// `float64` array, read in place. Returns the paths flattened and the total
/// joint log-probability.
#[pyfunction]
#[pyo3(signature = (observations, log_initial, log_transition, family, parameters))]
pub fn hmm_viterbi<'py>(
    py: Python<'py>,
    observations: &Bound<'py, PyAny>,
    log_initial: PyReadonlyArray1<'py, f64>,
    log_transition: PyReadonlyArray1<'py, f64>,
    family: &str,
    parameters: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<i64>>, f64)> {
    let (log_initial, log_transition, parameters) = (
        log_initial.as_slice()?,
        log_transition.as_slice()?,
        parameters.as_slice()?,
    );
    let m = log_initial.len();
    let emission = match family {
        "categorical" => Emission::Categorical(parameters),
        "gaussian" if parameters.len() == 2 * m => Emission::Gaussian {
            mean: &parameters[..m],
            scale: &parameters[m..],
        },
        "gaussian" => {
            return Err(PyValueError::new_err(format!(
                "a Gaussian takes {m} means and {m} scales, got {} values",
                parameters.len()
            )))
        }
        name => Emission::Count(
            count_family(name, parameters, m, false, STIRLING_FROM)
                .map_err(PyValueError::new_err)?,
        ),
    };
    let result = if let Ok(values) = observations.extract::<PyReadonlyArray2<'py, i64>>() {
        let length = values.shape()[1];
        let values = values.as_slice()?;
        py.detach(|| decode(values, length, log_initial, log_transition, &emission))
    } else {
        let values = observations.extract::<PyReadonlyArray2<'py, f64>>()?;
        let length = values.shape()[1];
        let values = values.as_slice()?;
        py.detach(|| decode(values, length, log_initial, log_transition, &emission))
    };
    let (states, log_probability) = result.map_err(PyValueError::new_err)?;
    Ok((PyArray1::from_vec(py, states), log_probability))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The best path by enumeration of every path of a short chain.
    #[test]
    fn the_path_is_the_enumerated_maximum() {
        let (m, length) = (3usize, 5usize);
        let ln = |v: &[f64]| v.iter().map(|x| x.ln()).collect::<Vec<_>>();
        let initial = ln(&[0.5, 0.3, 0.2]);
        let transition = ln(&[0.7, 0.2, 0.1, 0.1, 0.8, 0.1, 0.2, 0.2, 0.6]);
        let emission = ln(&[0.6, 0.3, 0.1, 0.2, 0.5, 0.3, 0.1, 0.2, 0.7]);
        let row = [0.0, 2.0, 1.0, 1.0, 2.0];
        let (path, score) = decode(
            &row,
            length,
            &initial,
            &transition,
            &Emission::Categorical(&emission),
        )
        .unwrap();
        let (mut best, mut arg) = (f64::NEG_INFINITY, vec![]);
        for code in 0..m.pow(length as u32) {
            let states: Vec<usize> = (0..length).map(|t| (code / m.pow(t as u32)) % m).collect();
            let mut s = initial[states[0]] + emission[states[0] * 3 + row[0] as usize];
            for t in 1..length {
                s += transition[states[t - 1] * m + states[t]]
                    + emission[states[t] * 3 + row[t] as usize];
            }
            if s > best {
                best = s;
                arg = states;
            }
        }
        assert!((score - best).abs() < 1e-12);
        let arg: Vec<i64> = arg.iter().map(|&s| s as i64).collect();
        assert_eq!(path, arg);
    }
}
