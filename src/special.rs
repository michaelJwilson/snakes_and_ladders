//! `ln Gamma(x)` for positive `x`, with no special-function dependency (issue #997).
//!
//! The Lanczos approximation at `g = 607 / 128` with Godfrey's fifteen
//! coefficients, the form Boost and many numerical libraries take for double
//! precision: relative error near `1e-15` for `x >= 0.5`. Below that the
//! reflection `ln Gamma(x) = ln(pi / sin(pi x)) - ln Gamma(1 - x)` holds.
//! `ln_gamma_approx` takes the Stirling series from `x = 10` instead, one
//! logarithm and one division where the Lanczos sum takes fourteen; a caller
//! asks for it by name.
//! `libm` would supply the same function; taking it as a direct dependency is
//! issue #999, and this is pinned to `torch.lgamma` in the meantime.

const G: f64 = 607.0 / 128.0;

const COEFFICIENTS: [f64; 15] = [
    0.999_999_999_999_997_1,
    57.156_235_665_862_92,
    -59.597_960_355_475_49,
    14.136_097_974_741_746,
    -0.491_913_816_097_620_2,
    3.399_464_998_481_189e-5,
    4.652_362_892_704_858e-5,
    -9.837_447_530_487_956e-5,
    1.580_887_032_249_125e-4,
    -2.102_644_417_241_048_8e-4,
    2.174_396_181_152_126_4e-4,
    -1.643_181_065_367_639e-4,
    8.441_822_398_385_275e-5,
    -2.619_083_840_158_141e-5,
    3.689_918_265_953_162_4e-6,
];

/// Where the Stirling series takes over from the Lanczos sum by default.
pub const STIRLING_FROM: f64 = 10.0;

/// The Stirling series to the `x^-11` term: at `x >= 10` the first omitted
/// term is below `3e-15` absolute against a value above 12, and it costs one
/// logarithm and one division where the Lanczos sum takes fourteen.
#[inline]
fn stirling(x: f64) -> f64 {
    let r = 1.0 / x;
    let r2 = r * r;
    let series = r
        * (1.0 / 12.0
            - r2 * (1.0 / 360.0
                - r2 * (1.0 / 1260.0
                    - r2 * (1.0 / 1680.0 - r2 * (1.0 / 1188.0 - r2 * (691.0 / 360_360.0))))));
    (x - 0.5) * x.ln() - x + 0.5 * (2.0 * std::f64::consts::PI).ln() + series
}

/// `ln_gamma`, with the Stirling series from `x = from`: from the default
/// `STIRLING_FROM` within `4e-15` relative of the Lanczos sum, and cheaper.
/// The series' first omitted term is `3617 / (122400 x^13)`, so a lower
/// `from` trades accuracy for speed: `2.4e-11` absolute at `from = 5`.
#[inline]
pub fn ln_gamma_approx(x: f64, from: f64) -> f64 {
    if x >= from {
        stirling(x)
    } else {
        ln_gamma(x)
    }
}

/// `ln Gamma(x)` for `x > 0`; `+inf` at zero, `nan` below it.
#[inline]
pub fn ln_gamma(x: f64) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::INFINITY;
    }
    if x < 0.5 {
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let z = x - 1.0;
    let mut sum = COEFFICIENTS[0];
    for (k, &c) in COEFFICIENTS.iter().enumerate().skip(1) {
        sum += c / (z + k as f64);
    }
    let t = z + G + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + sum.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ln_gamma_is_the_log_factorial_at_the_integers() {
        let mut factorial: f64 = 0.0;
        for n in 1..170u32 {
            // ln Gamma(n) = ln (n - 1)!
            let got = ln_gamma(f64::from(n));
            assert!(
                (got - factorial).abs() <= 1e-13 * factorial.abs().max(1.0),
                "n = {n}: {got} against {factorial}"
            );
            factorial += f64::from(n).ln();
        }
    }

    #[test]
    fn the_two_forms_agree_where_they_meet() {
        for x in [9.5f64, 10.0, 10.5, 12.25, 40.0] {
            let z = x - 1.0;
            let mut sum = COEFFICIENTS[0];
            for (k, &c) in COEFFICIENTS.iter().enumerate().skip(1) {
                sum += c / (z + k as f64);
            }
            let t = z + G + 0.5;
            let lanczos =
                0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + sum.ln();
            assert!(
                (stirling(x) - lanczos).abs() <= 4e-15 * lanczos.abs(),
                "x = {x}"
            );
        }
    }

    #[test]
    fn ln_gamma_meets_the_half_integers_and_reflection() {
        let sqrt_pi_ln = 0.5 * std::f64::consts::PI.ln();
        assert!((ln_gamma(0.5) - sqrt_pi_ln).abs() < 1e-14);
        // Gamma(3/2) = sqrt(pi) / 2.
        assert!((ln_gamma(1.5) - (sqrt_pi_ln - 2f64.ln())).abs() < 1e-14);
        // Gamma(0.1) Gamma(0.9) = pi / sin(0.1 pi).
        let pi = std::f64::consts::PI;
        let both = ln_gamma(0.1) + ln_gamma(0.9);
        assert!((both - (pi / (0.1 * pi).sin()).ln()).abs() < 1e-13);
    }
}
