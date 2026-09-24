//! HMC on a zero-mean Gaussian, the whole chain in one call (issue #986).
//!
//! `sample.hmc.sample` makes one autograd call per force evaluation and one
//! Python round trip per leapfrog step, so 1,000 transitions of ten steps at
//! d = 100 took 1.10 s where BlackJAX's compiled chain took 14.1 ms (#963).
//! A torch closure cannot cross the FFI boundary, so what compiles is a
//! declared family rather than an arbitrary objective: `U(x) = x' P x / 2`
//! with `P` a diagonal or a dense symmetric matrix, whose force is `P x`.
//!
//! The transition is `sample.hmc._transition` at unit mass, temperature one
//! and the leapfrog integrator: a standard normal momentum, kicks of half,
//! one, ..., one, half a step around full drifts, and acceptance of a uniform
//! on `[0, 1)` below `exp(H(current) - H(proposal))`. The draws come from
//! ChaCha8 seeded by the caller, so the stream is this route's own and the
//! torch route is not reproduced draw for draw; `gaussian_leapfrog` exposes
//! the trajectory alone, which is what the torch integrator pins exactly.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal, StandardUniform};

/// `P`, diagonal or dense row-major, and the force and potential it defines.
pub enum Precision<'a> {
    Diagonal(&'a [f64]),
    Dense(&'a [f64], usize),
}

impl Precision<'_> {
    pub(crate) fn dimension(&self) -> usize {
        match self {
            Precision::Diagonal(p) => p.len(),
            Precision::Dense(_, d) => *d,
        }
    }

    /// `out = P x`.
    #[inline]
    pub(crate) fn force(&self, x: &[f64], out: &mut [f64]) {
        match self {
            Precision::Diagonal(p) => {
                for ((o, &pi), &xi) in out.iter_mut().zip(p.iter()).zip(x) {
                    *o = pi * xi;
                }
            }
            Precision::Dense(p, d) => {
                for (row, o) in p.chunks_exact(*d).zip(out.iter_mut()) {
                    *o = row.iter().zip(x).map(|(a, b)| a * b).sum();
                }
            }
        }
    }

    /// `x' P x / 2`.
    pub(crate) fn potential(&self, x: &[f64], scratch: &mut [f64]) -> f64 {
        self.force(x, scratch);
        0.5 * x
            .iter()
            .zip(scratch.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>()
    }
}

/// Leapfrog over `n_steps` from `(x, p)` in place, as `hmc.leapfrog` steps it.
pub fn leapfrog(
    precision: &Precision<'_>,
    x: &mut [f64],
    p: &mut [f64],
    step_size: f64,
    n_steps: usize,
    force: &mut [f64],
) {
    precision.force(x, force);
    for (pi, fi) in p.iter_mut().zip(force.iter()) {
        *pi -= 0.5 * step_size * fi;
    }
    for step in 0..n_steps {
        for (xi, pi) in x.iter_mut().zip(p.iter()) {
            *xi += step_size * pi;
        }
        precision.force(x, force);
        let kick = if step + 1 == n_steps { 0.5 } else { 1.0 };
        for (pi, fi) in p.iter_mut().zip(force.iter()) {
            *pi -= kick * step_size * fi;
        }
    }
}

/// What a chain returns: its stored draws, the accepted count and the
/// energy error of every proposal after burn-in.
pub struct Chain {
    pub draws: Vec<f64>,
    pub accepted: usize,
    pub energy_error: Vec<f64>,
}

/// `burn_in + n_samples` transitions from `theta0`.
#[allow(clippy::too_many_arguments)]
pub fn chain(
    precision: &Precision<'_>,
    theta0: &[f64],
    n_samples: usize,
    burn_in: usize,
    step_size: f64,
    n_steps: usize,
    seed: u64,
    store_chain: bool,
) -> Result<Chain, String> {
    let d = precision.dimension();
    if theta0.len() != d {
        return Err(format!("theta0 has {} entries for d = {d}", theta0.len()));
    }
    // A NaN step is refused with the rest.
    if step_size.is_nan() || step_size <= 0.0 || n_steps == 0 {
        return Err(format!(
            "step_size must be positive and n_steps at least 1, got {step_size} and {n_steps}"
        ));
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut position = theta0.to_vec();
    let mut x = vec![0.0; d];
    let mut p = vec![0.0; d];
    let mut force = vec![0.0; d];
    let mut draws = Vec::with_capacity(if store_chain { n_samples * d } else { 0 });
    let mut energy_error = Vec::with_capacity(n_samples);
    let mut accepted = 0;
    let mut current_potential = precision.potential(&position, &mut force);
    for index in 0..burn_in + n_samples {
        for pi in p.iter_mut() {
            *pi = StandardNormal.sample(&mut rng);
        }
        let current = current_potential + 0.5 * p.iter().map(|v| v * v).sum::<f64>();
        x.copy_from_slice(&position);
        leapfrog(precision, &mut x, &mut p, step_size, n_steps, &mut force);
        let proposed_potential = precision.potential(&x, &mut force);
        let proposed = proposed_potential + 0.5 * p.iter().map(|v| v * v).sum::<f64>();
        let uniform: f64 = StandardUniform.sample(&mut rng);
        let take = uniform < (current - proposed).exp();
        if take {
            position.copy_from_slice(&x);
            current_potential = proposed_potential;
        }
        if index >= burn_in {
            accepted += usize::from(take);
            energy_error.push((proposed - current).abs());
            if store_chain {
                draws.extend_from_slice(&position);
            }
        }
    }
    Ok(Chain {
        draws,
        accepted,
        energy_error,
    })
}

pub(crate) fn precision_of<'a>(
    values: &'a [f64],
    dimension: usize,
) -> Result<Precision<'a>, String> {
    if values.len() == dimension {
        Ok(Precision::Diagonal(values))
    } else if values.len() == dimension * dimension {
        Ok(Precision::Dense(values, dimension))
    } else {
        Err(format!(
            "precision has {} entries: neither d = {dimension} nor d * d",
            values.len()
        ))
    }
}

/// A Gaussian HMC chain; see the module docs.
///
/// `precision` is `d` entries for a diagonal or `d * d` row-major for a
/// dense matrix, with `d` the length of `theta0`. Returns the draws
/// flattened `n_samples * d` (empty without `store_chain`), the accepted
/// count and the energy error per recorded proposal.
#[pyfunction]
#[pyo3(signature = (precision, theta0, n_samples, burn_in, step_size, n_steps, seed, store_chain))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn gaussian_hmc<'py>(
    py: Python<'py>,
    precision: PyReadonlyArray1<'py, f64>,
    theta0: PyReadonlyArray1<'py, f64>,
    n_samples: usize,
    burn_in: usize,
    step_size: f64,
    n_steps: usize,
    seed: u64,
    store_chain: bool,
) -> PyResult<(Bound<'py, PyArray1<f64>>, usize, Bound<'py, PyArray1<f64>>)> {
    let (values, theta0) = (precision.as_slice()?, theta0.as_slice()?);
    let result = py
        .detach(|| {
            let precision = precision_of(values, theta0.len())?;
            chain(
                &precision,
                theta0,
                n_samples,
                burn_in,
                step_size,
                n_steps,
                seed,
                store_chain,
            )
        })
        .map_err(PyValueError::new_err)?;
    Ok((
        PyArray1::from_vec(py, result.draws),
        result.accepted,
        PyArray1::from_vec(py, result.energy_error),
    ))
}

/// One leapfrog trajectory from `(theta, momentum)`; returns the end point.
#[pyfunction]
#[allow(clippy::type_complexity)]
#[pyo3(signature = (precision, theta, momentum, step_size, n_steps))]
pub fn gaussian_leapfrog<'py>(
    py: Python<'py>,
    precision: PyReadonlyArray1<'py, f64>,
    theta: PyReadonlyArray1<'py, f64>,
    momentum: PyReadonlyArray1<'py, f64>,
    step_size: f64,
    n_steps: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let mut x = theta.as_slice()?.to_vec();
    let mut p = momentum.as_slice()?.to_vec();
    if p.len() != x.len() {
        return Err(PyValueError::new_err("theta and momentum differ in length"));
    }
    let precision = precision_of(precision.as_slice()?, x.len()).map_err(PyValueError::new_err)?;
    let mut force = vec![0.0; x.len()];
    leapfrog(&precision, &mut x, &mut p, step_size, n_steps, &mut force);
    Ok((PyArray1::from_vec(py, x), PyArray1::from_vec(py, p)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_chain_samples_the_diagonal_gaussian() {
        let d = 50;
        let precision: Vec<f64> = (0..d)
            .map(|i| 1.0 + 3.0 * i as f64 / (d - 1) as f64)
            .collect();
        // 1.2 time units: far from a half or a full turn at every
        // frequency here, which would leave a coordinate nearly frozen.
        let n = 20_000;
        let out = chain(
            &Precision::Diagonal(&precision),
            &vec![0.0; d],
            n,
            100,
            0.15,
            8,
            986,
            true,
        )
        .unwrap();
        assert!(out.accepted as f64 / n as f64 > 0.8);
        for i in [0, d / 2, d - 1] {
            let (mut mean, mut square) = (0.0, 0.0);
            for row in out.draws.chunks_exact(d) {
                mean += row[i];
                square += row[i] * row[i];
            }
            mean /= n as f64;
            let variance = square / n as f64 - mean * mean;
            // Loose: the draws are correlated, and this is a smoke check of
            // the target; the Python pins do the exact work.
            assert!(mean.abs() < 0.05, "mean {mean} at {i}");
            assert!(
                (variance * precision[i] - 1.0).abs() < 0.1,
                "variance {variance} at {i}"
            );
        }
    }

    #[test]
    fn leapfrog_is_reversible() {
        let precision = [1.0, 2.0, 0.5, 0.0, 1.0, 3.0, 0.0, 0.2, 1.5];
        let dense = Precision::Dense(&precision, 3);
        let (x0, p0) = ([0.3, -1.2, 0.8], [1.1, 0.4, -0.7]);
        let (mut x, mut p, mut f) = (x0.to_vec(), p0.to_vec(), vec![0.0; 3]);
        leapfrog(&dense, &mut x, &mut p, 0.1, 25, &mut f);
        for v in &mut p {
            *v = -*v;
        }
        leapfrog(&dense, &mut x, &mut p, 0.1, 25, &mut f);
        for i in 0..3 {
            assert!((x[i] - x0[i]).abs() < 1e-12);
            assert!((-p[i] - p0[i]).abs() < 1e-12);
        }
    }
}
