//! The declared energy families a compiled chain runs (issue #1006).
//!
//! `sample.declared.declared_energy` hands over a family code and its flat
//! parameters; this is the other end. A torch closure cannot cross the FFI
//! boundary, so a family is declared, never recognized.

/// `P`, diagonal or dense row-major, and the force and potential it defines.
pub enum Precision<'a> {
    Diagonal(&'a [f64]),
    Dense(&'a [f64], usize),
}

impl Precision<'_> {
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

/// The family codes `sample.declared` writes.
pub const GAUSSIAN: u8 = 0;
pub const ROSENBROCK: u8 = 1;
pub const MIXTURE: u8 = 2;

/// `U(x)` for one declared family.
pub enum Energy<'a> {
    /// `x' P x / 2`.
    Gaussian(Precision<'a>),
    /// `sum_i b (x_{i+1} - x_i^2)^2 + (a - x_i)^2` (`eq:rosenbrock`).
    Rosenbrock { a: f64, b: f64 },
    /// A one-channel Gaussian mixture's negative log-likelihood of `values`
    /// in `theta = (k - 1 free weights, k means, k log scales)`, as
    /// `opt.mixture.GaussianMixtureObjective` states it (issue #1008).
    Mixture { values: &'a [f64], k: usize },
}

/// A mixture's `theta` split into what `mixture_stream::gaussian_gradient` reads.
fn mixture_parameters(theta: &[f64], k: usize) -> (Vec<f64>, &[f64], Vec<f64>) {
    let mut padded = Vec::with_capacity(k);
    padded.push(0.0);
    padded.extend_from_slice(&theta[..k - 1]);
    let high = padded.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let total = high + padded.iter().map(|v| (v - high).exp()).sum::<f64>().ln();
    let log_weight = padded.iter().map(|v| v - total).collect();
    let scale = theta[2 * k - 1..3 * k - 1]
        .iter()
        .map(|v| v.exp())
        .collect();
    (log_weight, &theta[k - 1..2 * k - 1], scale)
}

impl Energy<'_> {
    /// `U(x)`; `scratch` is `x`'s length and is overwritten.
    #[inline]
    pub fn potential(&self, x: &[f64], scratch: &mut [f64]) -> f64 {
        match self {
            Energy::Gaussian(precision) => precision.potential(x, scratch),
            Energy::Rosenbrock { a, b } => x
                .windows(2)
                .map(|pair| {
                    let residual = pair[1] - pair[0] * pair[0];
                    let offset = a - pair[0];
                    b * residual * residual + offset * offset
                })
                .sum(),
            Energy::Mixture { values, k } => {
                let (log_weight, mean, scale) = mixture_parameters(x, *k);
                crate::mixture_stream::gaussian_gradient(values, &log_weight, mean, &scale, true)
                    .map_or(f64::NAN, |(value, _)| value)
            }
        }
    }

    /// `out = grad U(x)`, and `U(x)` when `with_value`, nan otherwise: a
    /// trajectory's last force is at the point whose potential the
    /// acceptance reads, so the two are taken in one pass there, and the
    /// forces before it skip the value (issue #1008).
    #[inline]
    pub fn force(&self, x: &[f64], out: &mut [f64], with_value: bool) -> f64 {
        match self {
            Energy::Gaussian(precision) => {
                precision.force(x, out);
                0.5 * x.iter().zip(out.iter()).map(|(a, b)| a * b).sum::<f64>()
            }
            Energy::Rosenbrock { a, b } => {
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
            Energy::Mixture { values, k } => {
                let (log_weight, mean, scale) = mixture_parameters(x, *k);
                match crate::mixture_stream::gaussian_gradient(
                    values,
                    &log_weight,
                    mean,
                    &scale,
                    with_value,
                ) {
                    Ok((value, gradient)) => {
                        out.copy_from_slice(&gradient);
                        value
                    }
                    Err(_) => {
                        out.iter_mut().for_each(|o| *o = f64::NAN);
                        f64::NAN
                    }
                }
            }
        }
    }
}

/// The family `code` with `parameters`, on `dimension` coordinates.
pub fn energy_of(code: u8, parameters: &[f64], dimension: usize) -> Result<Energy<'_>, String> {
    match code {
        GAUSSIAN => Ok(Energy::Gaussian(precision_of(parameters, dimension)?)),
        ROSENBROCK => match parameters {
            [a, b] if dimension >= 2 => Ok(Energy::Rosenbrock { a: *a, b: *b }),
            _ => Err(format!(
                "Rosenbrock takes (a, b) and d >= 2, got {} parameters at d = {dimension}",
                parameters.len()
            )),
        },
        MIXTURE => match parameters.split_first() {
            Some((&k, values)) if k >= 1.0 && dimension as f64 == 3.0 * k - 1.0 => {
                Ok(Energy::Mixture {
                    values,
                    k: k as usize,
                })
            }
            _ => Err(format!(
                "a mixture takes (k, values...) with d = 3k - 1, got d = {dimension}"
            )),
        },
        _ => Err(format!("no declared energy family {code}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rosenbrock_is_zero_at_its_minimizer_and_positive_off_it() {
        let energy = energy_of(ROSENBROCK, &[1.0, 100.0], 3).unwrap();
        let mut scratch = [0.0; 3];
        assert_eq!(energy.potential(&[1.0, 1.0, 1.0], &mut scratch), 0.0);
        // b (x1 - x0^2)^2 + (a - x0)^2 at (0, 1): 100 + 1, then (1 - 1)^2 terms.
        assert_eq!(energy.potential(&[0.0, 1.0, 1.0], &mut scratch), 101.0);
    }

    #[test]
    fn the_rosenbrock_force_is_the_potential_s_slope() {
        let energy = energy_of(ROSENBROCK, &[1.0, 100.0], 4).unwrap();
        let x = [0.3, -0.7, 1.1, 0.4];
        let (mut force, mut scratch) = ([0.0; 4], [0.0; 4]);
        energy.force(&x, &mut force, false);
        for i in 0..4 {
            let (mut up, mut down) = (x, x);
            up[i] += 1e-6;
            down[i] -= 1e-6;
            let slope = (energy.potential(&up, &mut scratch)
                - energy.potential(&down, &mut scratch))
                / 2e-6;
            assert!((slope - force[i]).abs() < 1e-6 * slope.abs().max(1.0));
        }
    }

    #[test]
    fn an_unknown_family_is_refused() {
        assert!(energy_of(7, &[], 2).is_err());
        assert!(energy_of(ROSENBROCK, &[1.0], 2).is_err());
    }
}
