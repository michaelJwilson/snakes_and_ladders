// The algorithm and its coefficients are fdlibm's `e_log.c`, which carries:
//
// ====================================================
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//
// Developed at SunSoft, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice
// is preserved.
// ====================================================

//! `ln x` for positive normal `x` in plain arithmetic, so a loop over it vectorizes (issue #1064).
//!
//! `f64::ln` is a call into the platform's libm, opaque to LLVM, and a loop
//! that calls it runs one element per call. This is fdlibm's `__ieee754_log`
//! with its branches taken out: the exponent and mantissa are split by integer
//! operations on the bits, the mantissa is reduced to `[sqrt(1/2), sqrt(2))`,
//! and `ln m = 2 atanh(s)`, `s = (m - 1) / (m + 1)`, is fdlibm's degree-14
//! polynomial in `s`, whose approximation error is below `2^-58.45`. Every
//! step is an add, a multiply, a divide, a shift or a mask on 64-bit lanes,
//! each of which SSE2 carries two to a register.
//!
//! **Domain.** `x` positive, normal and finite. Zero, a negative, a subnormal,
//! an infinity or a NaN returns a finite value with no meaning, where
//! `f64::ln` returns `-inf`, NaN or `inf`; a debug build asserts the domain.
//! The accuracy against `f64::ln` is pinned in this module's tests.

/// `ln 2` split so that `e * LN2_HI` is exact for every exponent `e`.
const LN2_HI: f64 = 6.931_471_803_691_238e-1;
const LN2_LO: f64 = 1.908_214_929_270_587_7e-10;

/// fdlibm's coefficients, bit for bit, of `(ln m - 2s) / (2s) ~ Lg1 s^2 + ... + Lg7 s^14`.
const LG1: f64 = 6.666_666_666_666_735e-1;
const LG2: f64 = 3.999_999_999_940_942e-1;
const LG3: f64 = 2.857_142_874_366_239e-1;
const LG4: f64 = 2.222_219_843_214_978_4e-1;
const LG5: f64 = 1.818_357_216_161_805e-1;
const LG6: f64 = 1.531_383_769_920_937_3e-1;
const LG7: f64 = 1.479_819_860_511_658_6e-1;

/// The bits of `sqrt(1/2)`, where the reduced mantissa's range begins.
const SQRT_HALF_BITS: u64 = 0x3fe6_a09e_667f_3bcd;
/// `2^52` as bits: an integer `n < 2^52` or-ed into it reads as `2^52 + n`.
const TWO_52_BITS: u64 = 0x4330_0000_0000_0000;
/// The exponent bias plus one, which keeps the shifted exponent unsigned.
const OFFSET: u64 = 1024;

/// `ln x` for `x` positive, normal and finite; see the module's domain.
#[inline]
pub fn ln(x: f64) -> f64 {
    debug_assert!(
        x.is_normal() && x > 0.0,
        "ln takes a positive normal x, got {x}"
    );
    let bits = x.to_bits();
    // `x = 2^e m` with `m` in `[sqrt(1/2), sqrt(2))`: subtracting the bits of
    // `sqrt(1/2)` borrows from the exponent field exactly when `x`'s
    // significand is below `sqrt(2)`, so the difference's exponent field is
    // `e` plus the bias. The offset keeps the shift logical, which SSE2 has
    // for 64-bit lanes where the arithmetic one is not.
    let shifted = bits.wrapping_sub(SQRT_HALF_BITS).wrapping_add(OFFSET << 52) >> 52;
    let m = f64::from_bits(bits.wrapping_sub(shifted.wrapping_sub(OFFSET) << 52));
    // `e` as a float without a 64-bit integer conversion, which SSE2 lacks.
    let e = f64::from_bits(shifted | TWO_52_BITS) - (4_503_599_627_370_496.0 + OFFSET as f64);
    let f = m - 1.0;
    let s = f / (2.0 + f);
    let z = s * s;
    let w = z * z;
    let odd = w * (LG2 + w * (LG4 + w * LG6));
    let even = z * (LG1 + w * (LG3 + w * (LG5 + w * LG7)));
    let half_square = 0.5 * f * f;
    // fdlibm's order: `f - hfsq` carries the leading bits and the polynomial
    // corrects them, so the rounding of `s` is scaled down by `s^2`.
    e * LN2_HI - ((half_square - (s * (half_square + (odd + even)) + e * LN2_LO)) - f)
}

#[cfg(test)]
mod tests {
    use super::ln;

    /// The distance from `approx` to `exact` in units of `exact`'s last place.
    fn ulp(approx: f64, exact: f64) -> f64 {
        let spacing = f64::from_bits(exact.abs().to_bits() + 1) - exact.abs();
        (approx - exact).abs() / spacing
    }

    /// A 64-bit xorshift, so the sweep needs no dependency and is the same every run.
    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// The declared bound, in ulp of `f64::ln`. Measured: 1, at 16,727 of the
    /// 2,000,001 points of the geometric sweep; the rest agree bitwise.
    const ULPS: f64 = 1.0;

    #[test]
    fn ln_meets_the_platform_log_over_the_kernel_domain() {
        // The kernel's `t = r + mu c` over dispersions and exposed means from
        // `1e-3` to `1e7`: a dense geometric sweep, plus every binade edge and
        // the reduction's `sqrt(2)` boundary, where a range reduction slips.
        let mut worst = 0.0f64;
        let n = 2_000_000;
        let (low, high) = (1e-3f64.ln(), 1e7f64.ln());
        for i in 0..=n {
            let x = (low + (high - low) * f64::from(i) / f64::from(n)).exp();
            worst = worst.max(ulp(ln(x), x.ln()));
        }
        for e in -10..=24 {
            let binade = 2f64.powi(e);
            for x in [
                binade,
                binade * 2f64.sqrt(),
                binade * std::f64::consts::FRAC_1_SQRT_2,
            ] {
                for step in -4i64..=4 {
                    let y = f64::from_bits((x.to_bits() as i64 + step) as u64);
                    worst = worst.max(ulp(ln(y), y.ln()));
                }
            }
        }
        assert_eq!(ln(1.0), 0.0);
        assert!(
            worst <= ULPS,
            "ln is {worst} ulp from f64::ln on the kernel domain"
        );
    }

    #[test]
    fn ln_meets_the_platform_log_across_every_exponent() {
        // A random mantissa in every normal binade, `2^-1022` to `2^1023`.
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut worst = 0.0f64;
        for _ in 0..4_000_000 {
            let bits = xorshift(&mut state);
            let exponent = 1 + (bits >> 52) % 2046;
            let x = f64::from_bits((exponent << 52) | (bits & 0x000f_ffff_ffff_ffff));
            worst = worst.max(ulp(ln(x), x.ln()));
        }
        assert!(
            worst <= ULPS,
            "ln is {worst} ulp from f64::ln across exponents"
        );
    }
}
