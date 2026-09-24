//! One EM step for a one-dimensional Gaussian mixture, streamed (issue #986).
//!
//! `opt.mixture.expectation_maximization` forms the `(n, k)` joint, its
//! normalizer and the responsibilities as whole tensors, then the M step's
//! weighted products over them: 31.1 MB for ten iterations at 10^5 draws of
//! three components, where scikit-learn's fit peaked at 18.1 MB (#987).
//! Nothing of the M step needs a responsibility after its draw is counted,
//! so this keeps three numbers per component:
//!
//! - the mass `W_k = sum_i r_ik`;
//! - the weighted mean, updated in place (West 1979): `mu += (r / W) delta`;
//! - the weighted sum of squares about it, `S += r delta (x - mu)`.
//!
//! West's update is the stable one-pass form of the two-pass weighted mean
//! and variance `GaussianEmission.reestimate` computes, equal to it within
//! rounding; the torch route stays as the oracle that pins it.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Draws per parallel chunk of [`gaussian_gradient`]: the partial sums are
/// reduced in chunk order, so the result does not depend on the pool.
const CHUNK: usize = 4096;

/// New weights, means and variances, and the log-likelihood at the parameters given.
pub struct Step {
    pub weight: Vec<f64>,
    pub mean: Vec<f64>,
    pub variance: Vec<f64>,
    pub log_likelihood: f64,
}

/// One E step and one M step over the draws in `values`.
pub fn gaussian_step(
    values: &[f64],
    log_weight: &[f64],
    mean: &[f64],
    scale: &[f64],
) -> Result<Step, String> {
    let k = log_weight.len();
    if k == 0 || mean.len() != k || scale.len() != k {
        return Err(format!(
            "{k} weights need {k} means and scales, got {} and {}",
            mean.len(),
            scale.len()
        ));
    }
    if values.is_empty() {
        return Err("no draws to fit".into());
    }
    // `log w_k - log s_k - log(2 pi) / 2`, the part of each joint that does
    // not depend on the draw, and `1 / s_k` for the part that does.
    let offset: Vec<f64> = (0..k)
        .map(|c| log_weight[c] - scale[c].ln() - 0.5 * (2.0 * std::f64::consts::PI).ln())
        .collect();
    let precision: Vec<f64> = scale.iter().map(|s| 1.0 / s).collect();
    let mut joint = vec![0.0; k];
    let mut mass = vec![0.0; k];
    let mut located = vec![0.0; k];
    let mut squares = vec![0.0; k];
    let mut log_likelihood = 0.0;
    for &x in values {
        let mut high = f64::NEG_INFINITY;
        for c in 0..k {
            let z = (x - mean[c]) * precision[c];
            joint[c] = offset[c] - 0.5 * z * z;
            high = high.max(joint[c]);
        }
        let mut total = 0.0;
        for value in &mut joint {
            *value = (*value - high).exp();
            total += *value;
        }
        log_likelihood += high + total.ln();
        for c in 0..k {
            let r = joint[c] / total;
            if r == 0.0 {
                continue;
            }
            mass[c] += r;
            let delta = x - located[c];
            located[c] += r / mass[c] * delta;
            squares[c] += r * delta * (x - located[c]);
        }
    }
    let n = values.len() as f64;
    Ok(Step {
        weight: mass.iter().map(|m| m / n).collect(),
        variance: squares.iter().zip(&mass).map(|(s, m)| s / m).collect(),
        mean: located,
        log_likelihood,
    })
}

/// The mixture's negative log-likelihood and its gradient in
/// `theta = (k - 1 free weights, k means, k log scales)`, streamed.
///
/// `GaussianMixtureObjective.gradient` (issue #986): autograd through the
/// objective took 20.4 ms at 10^5 draws where JAX's compiled gradient took
/// 2.84. With responsibilities `r_ik`, `R_k = sum_i r_ik` and the weights
/// `w = softmax([0, free])`, the gradient is
/// `-(R_c - n w_c)` for the free weight of component `c >= 1`,
/// `-sum_i r_ik (x_i - mu_k) / s_k^2` for a mean and
/// `-sum_i r_ik ((x_i - mu_k)^2 / s_k^2 - 1)` for a log scale.
///
/// Without `with_value` the per-draw logarithm the value needs is skipped
/// and the value returned is nan: a leapfrog reads the value at its last
/// force alone (issue #1008).
pub fn gaussian_gradient(
    values: &[f64],
    log_weight: &[f64],
    mean: &[f64],
    scale: &[f64],
    with_value: bool,
) -> Result<(f64, Vec<f64>), String> {
    let k = log_weight.len();
    if k == 0 || mean.len() != k || scale.len() != k {
        return Err(format!(
            "{k} weights need {k} means and scales, got {} and {}",
            mean.len(),
            scale.len()
        ));
    }
    let offset: Vec<f64> = (0..k)
        .map(|c| log_weight[c] - scale[c].ln() - 0.5 * (2.0 * std::f64::consts::PI).ln())
        .collect();
    let precision: Vec<f64> = scale.iter().map(|s| 1.0 / s).collect();
    // Per chunk: the log-likelihood, then R, the first and the second sums.
    let partial: Vec<Vec<f64>> = values
        .par_chunks(CHUNK)
        .map(|chunk| {
            let mut sums = vec![0.0; 1 + 3 * k];
            let mut joint = vec![0.0; k];
            let mut z = vec![0.0; k];
            for &x in chunk {
                let mut high = f64::NEG_INFINITY;
                let mut top = 0;
                for c in 0..k {
                    z[c] = (x - mean[c]) * precision[c];
                    joint[c] = offset[c] - 0.5 * z[c] * z[c];
                    if joint[c] > high {
                        high = joint[c];
                        top = c;
                    }
                }
                // The largest term is exp(0): one exponential fewer per draw.
                let mut total = 0.0;
                for (c, value) in joint.iter_mut().enumerate() {
                    *value = if c == top { 1.0 } else { (*value - high).exp() };
                    total += *value;
                }
                if with_value {
                    sums[0] += high + total.ln();
                }
                for c in 0..k {
                    let r = joint[c] / total;
                    sums[1 + c] += r;
                    sums[1 + k + c] += r * z[c] * precision[c];
                    sums[1 + 2 * k + c] += r * z[c] * z[c];
                }
            }
            sums
        })
        .collect();
    let mut sums = vec![0.0; 1 + 3 * k];
    for chunk in &partial {
        for (total, value) in sums.iter_mut().zip(chunk) {
            *total += value;
        }
    }
    let n = values.len() as f64;
    let mut gradient = Vec::with_capacity(3 * k - 1);
    for c in 1..k {
        gradient.push(-(sums[1 + c] - n * log_weight[c].exp()));
    }
    for c in 0..k {
        gradient.push(-sums[1 + k + c]);
    }
    for c in 0..k {
        gradient.push(-(sums[1 + 2 * k + c] - sums[1 + c]));
    }
    Ok((if with_value { -sums[0] } else { f64::NAN }, gradient))
}

/// The mixture's negative log-likelihood and gradient; see [`gaussian_gradient`].
#[pyfunction]
#[pyo3(signature = (values, log_weight, mean, scale))]
pub fn gaussian_mixture_gradient<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
    log_weight: PyReadonlyArray1<'py, f64>,
    mean: PyReadonlyArray1<'py, f64>,
    scale: PyReadonlyArray1<'py, f64>,
) -> PyResult<(f64, Bound<'py, PyArray1<f64>>)> {
    let (values, log_weight, mean, scale) = (
        values.as_slice()?,
        log_weight.as_slice()?,
        mean.as_slice()?,
        scale.as_slice()?,
    );
    let (value, gradient) = py
        .detach(|| gaussian_gradient(values, log_weight, mean, scale, true))
        .map_err(PyValueError::new_err)?;
    Ok((value, PyArray1::from_vec(py, gradient)))
}

/// One Gaussian mixture EM step; see the module docs.
///
/// Returns `(weights, means, variances, log_likelihood)`, the
/// log-likelihood at the parameters given.
#[pyfunction]
#[pyo3(signature = (values, log_weight, mean, scale))]
#[allow(clippy::type_complexity)]
pub fn gaussian_mixture_em_step<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
    log_weight: PyReadonlyArray1<'py, f64>,
    mean: PyReadonlyArray1<'py, f64>,
    scale: PyReadonlyArray1<'py, f64>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    f64,
)> {
    let (values, log_weight, mean, scale) = (
        values.as_slice()?,
        log_weight.as_slice()?,
        mean.as_slice()?,
        scale.as_slice()?,
    );
    let step = py
        .detach(|| gaussian_step(values, log_weight, mean, scale))
        .map_err(PyValueError::new_err)?;
    Ok((
        PyArray1::from_vec(py, step.weight),
        PyArray1::from_vec(py, step.mean),
        PyArray1::from_vec(py, step.variance),
        step.log_likelihood,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_step_is_the_two_pass_weighted_moments() {
        let values = [-2.1, -1.4, 0.3, 0.9, 1.7, 2.2, 3.5, -0.4];
        let (log_weight, mean, scale) = ([0.3f64.ln(), 0.7f64.ln()], [-1.0, 1.5], [0.8, 1.2]);
        let step = gaussian_step(&values, &log_weight, &mean, &scale).unwrap();
        let mut ll = 0.0;
        let mut r = vec![[0.0; 2]; values.len()];
        for (i, &x) in values.iter().enumerate() {
            let p: Vec<f64> = (0..2)
                .map(|c| {
                    log_weight[c].exp() * (-0.5 * ((x - mean[c]) / scale[c]).powi(2)).exp()
                        / (scale[c] * (2.0 * std::f64::consts::PI).sqrt())
                })
                .collect();
            let total: f64 = p.iter().sum();
            ll += total.ln();
            r[i] = [p[0] / total, p[1] / total];
        }
        assert!((step.log_likelihood - ll).abs() < 1e-12);
        for c in 0..2 {
            let w: f64 = r.iter().map(|row| row[c]).sum();
            let mu = r
                .iter()
                .zip(&values)
                .map(|(row, x)| row[c] * x)
                .sum::<f64>()
                / w;
            let var = r
                .iter()
                .zip(&values)
                .map(|(row, x)| row[c] * (x - mu).powi(2))
                .sum::<f64>()
                / w;
            assert!((step.weight[c] - w / values.len() as f64).abs() < 1e-14);
            assert!((step.mean[c] - mu).abs() < 1e-13);
            assert!((step.variance[c] - var).abs() < 1e-13);
        }
    }
}
