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

/// `U(x)` for one declared family.
pub enum Energy<'a> {
    /// `x' P x / 2`.
    Gaussian(Precision<'a>),
    /// `sum_i b (x_{i+1} - x_i^2)^2 + (a - x_i)^2` (`eq:rosenbrock`).
    Rosenbrock { a: f64, b: f64 },
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
        }
    }

    /// `out = grad U(x)`.
    #[inline]
    pub fn force(&self, x: &[f64], out: &mut [f64]) {
        match self {
            Energy::Gaussian(precision) => precision.force(x, out),
            Energy::Rosenbrock { a, b } => {
                out.iter_mut().for_each(|o| *o = 0.0);
                for i in 0..x.len() - 1 {
                    let residual = x[i + 1] - x[i] * x[i];
                    out[i] += -4.0 * b * residual * x[i] - 2.0 * (a - x[i]);
                    out[i + 1] += 2.0 * b * residual;
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
        energy.force(&x, &mut force);
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
