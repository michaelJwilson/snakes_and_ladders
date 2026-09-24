//! Random-walk Metropolis on a declared energy, warm-up included, in one call (issue #1006).
//!
//! `sample.metropolis.random_walk` on the torch route makes one Python round
//! trip and one objective call per proposal. What compiles is a declared
//! family (`energy.rs`). The transition is `metropolis._RandomWalkKernel` at
//! unit temperature: `y = x + h s * z` with `z` standard normal and `s` the
//! metric's per-coordinate scale, accepted when a uniform on `[0, 1)` is
//! below `exp(U(x) - U(y))`. The warm-up is `chain._warm_up` operation for
//! operation, shared with HMC in `chain.rs`. The draws come from ChaCha8 seeded by the caller, so the
//! stream is this route's own and the torch route is matched in
//! distribution.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal, StandardUniform};

use crate::chain::{decide, Kernel, Walk, Warmup};
use crate::energy::Energy;
use crate::walk_class;

/// `metropolis._RandomWalkKernel`: where it is, its energy, and the buffers a step reuses.
pub struct RandomWalk {
    position: Vec<f64>,
    current: f64,
    proposal: Vec<f64>,
    scratch: Vec<f64>,
    scale: Vec<f64>,
}

impl RandomWalk {
    /// At `theta0` on `energy`, at unit scale.
    pub fn new(energy: &Energy<'_>, theta0: &[f64]) -> Self {
        let d = theta0.len();
        let mut scratch = vec![0.0; d];
        let current = energy.potential(theta0, &mut scratch);
        Self {
            position: theta0.to_vec(),
            current,
            proposal: vec![0.0; d],
            scratch,
            scale: vec![1.0; d],
        }
    }
}

impl Kernel for RandomWalk {
    #[inline]
    fn step(
        &mut self,
        energy: &Energy<'_>,
        rng: &mut ChaCha8Rng,
        step_size: f64,
    ) -> (bool, f64, f64) {
        for ((y, &x), &s) in self
            .proposal
            .iter_mut()
            .zip(&self.position)
            .zip(&self.scale)
        {
            let z: f64 = StandardNormal.sample(rng);
            *y = x + step_size * s * z;
        }
        let proposed = energy.potential(&self.proposal, &mut self.scratch);
        let uniform: f64 = StandardUniform.sample(rng);
        let (take, probability) = decide(self.current - proposed, uniform);
        let error = (proposed - self.current).abs();
        if take {
            std::mem::swap(&mut self.position, &mut self.proposal);
            self.current = proposed;
        }
        (take, probability, error)
    }

    fn position(&self) -> &[f64] {
        &self.position
    }

    fn set_scale(&mut self, scale: Vec<f64>) {
        self.scale = scale;
    }
}

/// `Walk::new` for the random walk from `theta0`.
pub fn walk(
    family: u8,
    parameters: Vec<f64>,
    theta0: &[f64],
    step_size: f64,
    seed: u64,
    warmup: Option<&Warmup>,
    powers: &[i32],
) -> Result<Walk<RandomWalk>, String> {
    Walk::new(
        family,
        parameters,
        theta0.len(),
        |energy| RandomWalk::new(energy, theta0),
        step_size,
        seed,
        warmup,
        powers,
    )
}

walk_class!(MetropolisWalk, RandomWalk, walk);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::energy::GAUSSIAN;

    fn warmup(proposals: usize) -> Warmup {
        Warmup {
            proposals,
            target: 0.234,
            jitter: 0.0,
            gamma: 0.2,
            t0: 10.0,
            kappa: 0.75,
        }
    }

    #[test]
    fn the_chain_samples_the_diagonal_gaussian() {
        // Precisions 1 and 4: variances 1 and 0.25.
        let precision = [1.0, 4.0];
        let n = 200_000;
        let mut walk = walk(
            GAUSSIAN,
            precision.to_vec(),
            &[0.0, 0.0],
            1.2,
            1006,
            None,
            &[],
        )
        .unwrap();
        walk.advance(1_000, false, false);
        let result = walk.advance(n, true, false);
        for (coordinate, variance) in [(0, 1.0), (1, 0.25)] {
            let draws: Vec<f64> = result
                .draws
                .iter()
                .skip(coordinate)
                .step_by(2)
                .copied()
                .collect();
            let mean = draws.iter().sum::<f64>() / n as f64;
            let second = draws.iter().map(|x| x * x).sum::<f64>() / n as f64;
            assert!(mean.abs() < 0.03, "mean {mean}");
            assert!(
                (second - variance).abs() < 0.05 * variance.max(0.5),
                "E x^2 {second}"
            );
        }
    }

    #[test]
    fn the_warm_up_reaches_the_target_and_the_metric() {
        let precision = [1.0, 100.0];
        let result = walk(
            GAUSSIAN,
            precision.to_vec(),
            &[0.0, 0.0],
            0.1,
            7,
            Some(&warmup(4_000)),
            &[],
        )
        .unwrap();
        assert!((result.warmup_acceptance - 0.234).abs() < 0.05);
        // The mass diagonal estimates the precision to the window's sampling error.
        assert!((result.mass_diagonal[1] / result.mass_diagonal[0] / 100.0 - 1.0).abs() < 0.5);
    }

    #[test]
    fn blocks_continue_one_chain() {
        // Two blocks of 500 are one block of 1,000: the state and stream carry over.
        let make = || walk(GAUSSIAN, vec![1.0, 4.0], &[0.3, -0.2], 0.8, 11, None, &[]).unwrap();
        let whole = make().advance(1_000, true, false);
        let mut split = make();
        let (mut draws, a) = (
            split.advance(500, true, false),
            split.advance(500, true, false),
        );
        draws.draws.extend(a.draws);
        assert_eq!(draws.draws, whole.draws);
        assert_eq!(draws.accepted + a.accepted, whole.accepted);
    }

    #[test]
    fn the_filter_keeps_kalman_means_sums() {
        let mut walk = walk(
            GAUSSIAN,
            vec![1.0, 4.0],
            &[0.3, -0.2],
            0.8,
            5,
            None,
            &[1, 2],
        )
        .unwrap();
        let draws = walk.advance(300, true, true).draws;
        let f = &walk.filters[1];
        assert_eq!(f.n, 300);
        let ys: Vec<f64> = draws.iter().skip(1).step_by(2).map(|x| x * x).collect();
        assert_eq!(f.first[1], ys[0]);
        assert_eq!(f.last[1], ys[299]);
        let lagged: f64 = ys.windows(2).map(|w| w[0] * w[1]).sum();
        assert!((f.lagged[1] - lagged).abs() < 1e-12 * lagged.abs());
    }

    #[test]
    fn a_short_warm_up_and_a_bad_step_are_refused() {
        assert!(walk(GAUSSIAN, vec![1.0], &[0.0], 0.0, 0, None, &[]).is_err());
        assert!(walk(GAUSSIAN, vec![1.0], &[0.0], 0.1, 0, Some(&warmup(4)), &[]).is_err());
        assert!(walk(
            GAUSSIAN,
            vec![1.0, 2.0, 3.0],
            &[0.0, 0.0],
            0.1,
            0,
            None,
            &[]
        )
        .is_err());
    }
}
