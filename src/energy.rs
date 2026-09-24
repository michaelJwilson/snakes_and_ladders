//! The declared energy families a compiled chain runs (issue #1006).
//!
//! `sample.declared.declared_energy` hands over a family code and its flat
//! parameters; this is the other end. A torch closure cannot cross the FFI
//! boundary, so a family is declared, never recognized.

use crate::hmc_gaussian::{precision_of, Precision};

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
    fn an_unknown_family_is_refused() {
        assert!(energy_of(7, &[], 2).is_err());
        assert!(energy_of(ROSENBROCK, &[1.0], 2).is_err());
    }
}
