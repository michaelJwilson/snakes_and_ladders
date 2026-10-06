//! The value+gradient kernels a compiled chain runs (issues #1006, #1220).
//!
//! A torch closure cannot cross the FFI boundary, so a compiled chain runs a
//! kernel the package already has, behind one trait: [`Energy`]. An objective
//! names the kernel and hands over its data through
//! `supported_gradient() -> (kernel, data)`, which
//! `sample.declared.declared_energy` reads; [`build`] is the other end.
//! The Gaussian and Rosenbrock kernels are here; the Gaussian mixture is
//! `mixture_stream`'s, the Gaussian HMM `hmm_stream`'s and the count mixture
//! `count_mixture`'s, each the function its objective's gradient calls, so
//! a chain and a fit evaluate one arithmetic.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::count_mixture::{CountMixture, Slot};
use crate::hmm_stream::GaussianHmm;
use crate::mixture_stream::GaussianMixture;

/// `U(x)` and `grad U(x)` for one kernel on its own data.
pub trait Energy: Send + Sync {
    /// `U(x)`, with `grad U(x)` written into `out`.
    fn value_and_gradient(&self, x: &[f64], out: &mut [f64]) -> f64;

    /// `grad U(x)` alone: a leapfrog reads the value at its last force only
    /// (issue #1008), so a kernel that can skip the value's work does.
    #[inline]
    fn gradient(&self, x: &[f64], out: &mut [f64]) {
        self.value_and_gradient(x, out);
    }

    /// `U(x)` alone; `scratch` is `x`'s length and is overwritten.
    #[inline]
    fn value(&self, x: &[f64], scratch: &mut [f64]) -> f64 {
        self.value_and_gradient(x, scratch)
    }
}

/// `U(x) = x' P x / 2`, `P` diagonal `(d,)` or dense row-major `(d, d)`.
pub struct Gaussian {
    precision: Vec<f64>,
    dimension: usize,
}

impl Gaussian {
    /// # Errors
    /// A precision of neither `d` nor `d * d` entries.
    pub fn new(precision: Vec<f64>, dimension: usize) -> Result<Self, String> {
        if precision.len() != dimension && precision.len() != dimension * dimension {
            return Err(format!(
                "precision has {} entries: neither d = {dimension} nor d * d",
                precision.len()
            ));
        }
        Ok(Self {
            precision,
            dimension,
        })
    }
}

impl Energy for Gaussian {
    #[inline]
    fn value_and_gradient(&self, x: &[f64], out: &mut [f64]) -> f64 {
        if self.precision.len() == self.dimension {
            for ((o, &pi), &xi) in out.iter_mut().zip(&self.precision).zip(x) {
                *o = pi * xi;
            }
        } else {
            for (row, o) in self
                .precision
                .chunks_exact(self.dimension)
                .zip(out.iter_mut())
            {
                *o = row.iter().zip(x).map(|(a, b)| a * b).sum();
            }
        }
        0.5 * x.iter().zip(out.iter()).map(|(a, b)| a * b).sum::<f64>()
    }
}

/// `U(x) = sum_i b (x_{i+1} - x_i^2)^2 + (a - x_i)^2` (`eq:rosenbrock`).
pub struct Rosenbrock {
    pub a: f64,
    pub b: f64,
}

impl Energy for Rosenbrock {
    fn value_and_gradient(&self, x: &[f64], out: &mut [f64]) -> f64 {
        let (a, b) = (self.a, self.b);
        out.iter_mut().for_each(|o| *o = 0.0);
        let mut potential = 0.0;
        for i in 0..x.len() - 1 {
            let residual = x[i + 1] - x[i] * x[i];
            let offset = a - x[i];
            potential += b * residual * residual + offset * offset;
            out[i] += -4.0 * b * residual * x[i] - 2.0 * offset;
            out[i + 1] += 2.0 * b * residual;
        }
        potential
    }

    #[inline]
    fn value(&self, x: &[f64], _scratch: &mut [f64]) -> f64 {
        x.windows(2)
            .map(|pair| {
                let residual = pair[1] - pair[0] * pair[0];
                let offset = self.a - pair[0];
                self.b * residual * residual + offset * offset
            })
            .sum()
    }
}

/// `log_softmax([0, free])`, the simplex a pinned first logit spans.
pub(crate) fn pinned_simplex(free: &[f64]) -> Vec<f64> {
    let mut padded = Vec::with_capacity(free.len() + 1);
    padded.push(0.0);
    padded.extend_from_slice(free);
    let high = padded.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let total = high + padded.iter().map(|v| (v - high).exp()).sum::<f64>().ln();
    padded.iter().map(|v| v - total).collect()
}

/// The kernel names `supported_gradient` may give.
pub const GAUSSIAN: &str = "gaussian";
pub const ROSENBROCK: &str = "rosenbrock";
pub const GAUSSIAN_MIXTURE: &str = "gaussian_mixture";
pub const GAUSSIAN_HMM: &str = "gaussian_hmm";
pub const COUNT_MIXTURE: &str = "count_mixture";

fn item<'py>(data: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    data.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("the kernel's data has no {key:?}")))
}

/// `data[key]` as `float64` values in row-major order, whatever its layout.
fn values(data: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<f64>> {
    let array: PyReadonlyArrayDyn<'_, f64> = item(data, key)?.extract()?;
    Ok(array.as_array().iter().copied().collect())
}

/// `data[key]` as counts, or `None` where it is absent or `None`.
fn counts(data: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<Vec<u32>>> {
    match data.get_item(key)? {
        Some(value) if !value.is_none() => {
            let array: PyReadonlyArray1<'_, u32> = value.extract()?;
            Ok(Some(array.as_array().to_vec()))
        }
        _ => Ok(None),
    }
}

/// The kernel `kernel` on `data`, at `dimension` coordinates.
///
/// # Errors
/// `ValueError` for an unknown kernel, missing data, or data that does not
/// fit `dimension`.
pub fn build(
    kernel: &str,
    data: &Bound<'_, PyDict>,
    dimension: usize,
) -> PyResult<Box<dyn Energy>> {
    let built: Result<Box<dyn Energy>, String> = match kernel {
        GAUSSIAN => Gaussian::new(values(data, "precision")?, dimension)
            .map(|e| Box::new(e) as Box<dyn Energy>),
        ROSENBROCK if dimension >= 2 => Ok(Box::new(Rosenbrock {
            a: item(data, "a")?.extract()?,
            b: item(data, "b")?.extract()?,
        })),
        ROSENBROCK => Err(format!("Rosenbrock takes d >= 2, got d = {dimension}")),
        GAUSSIAN_MIXTURE => GaussianMixture::new(
            values(data, "observations")?,
            item(data, "k")?.extract()?,
            dimension,
        )
        .map(|e| Box::new(e) as Box<dyn Energy>),
        GAUSSIAN_HMM => {
            let observations: PyReadonlyArrayDyn<'_, f64> =
                item(data, "observations")?.extract()?;
            let shape = observations.shape().to_vec();
            if shape.len() != 2 {
                return Err(PyValueError::new_err(
                    "a Gaussian HMM's observations are (n_sequences, length)",
                ));
            }
            GaussianHmm::new(
                observations.as_array().iter().copied().collect(),
                shape[1],
                item(data, "m")?.extract()?,
                dimension,
            )
            .map(|e| Box::new(e) as Box<dyn Energy>)
        }
        COUNT_MIXTURE => {
            let named: Vec<(String, usize)> = item(data, "slots")?.extract()?;
            let slots = named
                .iter()
                .map(|(name, offset)| Slot::named(name).map(|slot| (slot, *offset)))
                .collect::<Result<Vec<_>, _>>()
                .map_err(PyValueError::new_err)?;
            let trials = match data.get_item("trials")? {
                Some(value) if !value.is_none() => {
                    let array: PyReadonlyArray1<'_, f64> = value.extract()?;
                    array.as_array().to_vec()
                }
                _ => Vec::new(),
            };
            CountMixture::new(
                item(data, "k")?.extract()?,
                counts(data, "totals")?,
                counts(data, "successes")?,
                trials,
                slots,
                dimension,
            )
            .map(|e| Box::new(e) as Box<dyn Energy>)
        }
        _ => Err(format!("no supported gradient kernel {kernel:?}")),
    };
    built.map_err(PyValueError::new_err)
}

/// One kernel on its data, built once: what an objective's own gradient
/// calls, so it and a compiled chain evaluate one arithmetic (issue #1220).
///
/// `kernel` and `data` are an objective's `supported_gradient()`.
#[pyclass(module = "sal.oxisal", frozen)]
pub struct SupportedEnergy {
    energy: Box<dyn Energy>,
    dimension: usize,
}

#[pymethods]
impl SupportedEnergy {
    #[new]
    fn new(kernel: &str, data: &Bound<'_, PyDict>, dimension: usize) -> PyResult<Self> {
        Ok(Self {
            energy: build(kernel, data, dimension)?,
            dimension,
        })
    }

    /// `(U(theta), grad U(theta))`, the GIL released for the evaluation.
    fn value_and_gradient<'py>(
        &self,
        py: Python<'py>,
        theta: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<(f64, Bound<'py, PyArray1<f64>>)> {
        let x = theta.as_slice()?;
        if x.len() != self.dimension {
            return Err(PyValueError::new_err(format!(
                "theta has {} coordinates where the kernel has {}",
                x.len(),
                self.dimension
            )));
        }
        let mut out = vec![0.0; x.len()];
        let value = py.detach(|| self.energy.value_and_gradient(x, &mut out));
        Ok((value, PyArray1::from_vec(py, out)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rosenbrock_is_zero_at_its_minimizer_and_positive_off_it() {
        let energy = Rosenbrock { a: 1.0, b: 100.0 };
        let mut scratch = [0.0; 3];
        assert_eq!(energy.value(&[1.0, 1.0, 1.0], &mut scratch), 0.0);
        // b (x1 - x0^2)^2 + (a - x0)^2 at (0, 1): 100 + 1, then (1 - 1)^2 terms.
        assert_eq!(energy.value(&[0.0, 1.0, 1.0], &mut scratch), 101.0);
    }

    #[test]
    fn the_rosenbrock_force_is_the_potential_s_slope() {
        let energy = Rosenbrock { a: 1.0, b: 100.0 };
        let x = [0.3, -0.7, 1.1, 0.4];
        let (mut force, mut scratch) = ([0.0; 4], [0.0; 4]);
        energy.gradient(&x, &mut force);
        for i in 0..4 {
            let (mut up, mut down) = (x, x);
            up[i] += 1e-6;
            down[i] -= 1e-6;
            let slope =
                (energy.value(&up, &mut scratch) - energy.value(&down, &mut scratch)) / 2e-6;
            assert!((slope - force[i]).abs() < 1e-6 * slope.abs().max(1.0));
        }
    }

    #[test]
    fn a_precision_of_the_wrong_size_is_refused() {
        assert!(Gaussian::new(vec![1.0; 3], 2).is_err());
        assert!(Gaussian::new(vec![1.0; 4], 2).is_ok());
    }
}
