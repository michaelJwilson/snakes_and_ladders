//! Random-walk Metropolis on a declared energy, warm-up included, in one call (issue #1006).
//!
//! `sample.metropolis.random_walk` on the torch route makes one Python round
//! trip and one objective call per proposal. What compiles is a declared
//! family (`energy.rs`). The transition is `metropolis._RandomWalkKernel` at
//! unit temperature: `y = x + h s * z` with `z` standard normal and `s` the
//! metric's per-coordinate scale, accepted when a uniform on `[0, 1)` is
//! below `exp(U(x) - U(y))`. The warm-up is `hmc._warm_up` operation for
//! operation: two windows of dual averaging (`hmc._DualAveraging`, its
//! constants passed in so they have one home), the first at unit scale
//! recording its second half, whose variance is the metric; the second on
//! that metric. The draws come from ChaCha8 seeded by the caller, so the
//! stream is this route's own and the torch route is matched in
//! distribution.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal, StandardUniform};

use crate::energy::{energy_of, Energy};

/// `hmc.Adaptation`'s fields and `hmc`'s dual-averaging constants.
pub struct Warmup {
    pub proposals: usize,
    pub target: f64,
    pub jitter: f64,
    pub gamma: f64,
    pub t0: f64,
    pub kappa: f64,
}

/// `hmc._DualAveraging`, term for term.
struct DualAveraging {
    mu: f64,
    target: f64,
    h_bar: f64,
    log_step: f64,
    log_averaged: f64,
    iteration: f64,
    gamma: f64,
    t0: f64,
    kappa: f64,
}

impl DualAveraging {
    fn new(step_size: f64, warmup: &Warmup) -> Self {
        Self {
            mu: (10.0 * step_size).ln(),
            target: warmup.target,
            h_bar: 0.0,
            log_step: step_size.ln(),
            log_averaged: 0.0,
            iteration: 0.0,
            gamma: warmup.gamma,
            t0: warmup.t0,
            kappa: warmup.kappa,
        }
    }

    fn update(&mut self, probability: f64) -> f64 {
        self.iteration += 1.0;
        let m = self.iteration;
        let weight = 1.0 / (m + self.t0);
        self.h_bar = (1.0 - weight) * self.h_bar + weight * (self.target - probability);
        self.log_step = self.mu - m.sqrt() / self.gamma * self.h_bar;
        let forget = m.powf(-self.kappa);
        self.log_averaged = forget * self.log_step + (1.0 - forget) * self.log_averaged;
        self.log_step.exp()
    }

    fn averaged(&self) -> f64 {
        self.log_averaged.exp()
    }
}

/// The chain's state: where it is, its energy, and the buffers a step reuses.
struct State {
    rng: ChaCha8Rng,
    position: Vec<f64>,
    current: f64,
    proposal: Vec<f64>,
    scratch: Vec<f64>,
    scale: Vec<f64>,
}

impl State {
    /// One proposal; returns whether it was taken, its probability and `|dU|`.
    #[inline]
    fn step(&mut self, energy: &Energy<'_>, step_size: f64) -> (bool, f64, f64) {
        for ((y, &x), &s) in self
            .proposal
            .iter_mut()
            .zip(&self.position)
            .zip(&self.scale)
        {
            let z: f64 = StandardNormal.sample(&mut self.rng);
            *y = x + step_size * s * z;
        }
        let proposed = energy.potential(&self.proposal, &mut self.scratch);
        let uniform: f64 = StandardUniform.sample(&mut self.rng);
        let log_ratio = self.current - proposed;
        // `metropolis._decide`: exp overflows only where acceptance is
        // certain, and a nan energy stays nan, which the comparison refuses.
        let ratio = if log_ratio > 700.0 {
            f64::INFINITY
        } else {
            log_ratio.exp()
        };
        let take = uniform < ratio;
        let probability = if ratio.is_nan() { 0.0 } else { ratio.min(1.0) };
        let error = (proposed - self.current).abs();
        if take {
            std::mem::swap(&mut self.position, &mut self.proposal);
            self.current = proposed;
        }
        (take, probability, error)
    }

    /// `hmc._jittered`: no draw at zero jitter.
    #[inline]
    fn jittered(&mut self, step_size: f64, jitter: f64) -> f64 {
        if jitter == 0.0 {
            return step_size;
        }
        let uniform: f64 = StandardUniform.sample(&mut self.rng);
        step_size * (1.0 + jitter * (2.0 * uniform - 1.0))
    }

    /// The two windows of `hmc._warm_up`; leaves the state on the metric.
    ///
    /// Returns the adapted step, the mass diagonal and the second window's
    /// mean acceptance probability.
    fn warm_up(
        &mut self,
        energy: &Energy<'_>,
        mut step_size: f64,
        warmup: &Warmup,
    ) -> Result<(f64, Vec<f64>, f64), String> {
        let first = warmup.proposals / 2;
        let second = warmup.proposals - first;
        let d = self.position.len();
        let mut averaging = DualAveraging::new(step_size, warmup);
        // Welford over the first window's second half.
        let (mut count, mut mean, mut m2) = (0.0, vec![0.0; d], vec![0.0; d]);
        for index in 0..first {
            let step = self.jittered(step_size, warmup.jitter);
            let (_, probability, _) = self.step(energy, step);
            step_size = averaging.update(probability);
            if index >= first / 2 {
                count += 1.0;
                for ((m, s), &x) in mean.iter_mut().zip(m2.iter_mut()).zip(&self.position) {
                    let delta = x - *m;
                    *m += delta / count;
                    *s += delta * (x - *m);
                }
            }
        }
        let variance: Vec<f64> = m2.iter().map(|s| s / (count - 1.0)).collect();
        let stuck: Vec<usize> = (0..d)
            .filter(|&i| variance[i].partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater))
            .collect();
        if !stuck.is_empty() {
            return Err(format!(
                "warm-up variance is zero on coordinate(s) {stuck:?} over the {count} recorded \
                 proposals: the chain did not move there, so no mass can be estimated; \
                 lengthen the warm-up or start the step size smaller"
            ));
        }
        self.scale = variance.iter().map(|v| v.sqrt()).collect();

        let mut averaging = DualAveraging::new(averaging.averaged(), warmup);
        step_size = averaging.averaged();
        let mut total = 0.0;
        for _ in 0..second {
            let step = self.jittered(step_size, warmup.jitter);
            let (_, probability, _) = self.step(energy, step);
            step_size = averaging.update(probability);
            total += probability;
        }
        let mass = variance.iter().map(|v| 1.0 / v).collect();
        Ok((averaging.averaged(), mass, total / second as f64))
    }
}

/// `sample.expectation.KalmanMean`'s sufficient statistics for `x^power`, per coordinate.
///
/// Updated as `KalmanMean.update` updates them, one draw at a time, so the
/// estimate formed from them in Python is the filter's (issues #988, #1006).
pub struct KalmanStats {
    power: i32,
    pub n: usize,
    pub sum: Vec<f64>,
    pub squares: Vec<f64>,
    pub lagged: Vec<f64>,
    pub first: Vec<f64>,
    pub last: Vec<f64>,
}

impl KalmanStats {
    fn new(power: i32, d: usize) -> Self {
        Self {
            power,
            n: 0,
            sum: vec![0.0; d],
            squares: vec![0.0; d],
            lagged: vec![0.0; d],
            first: vec![0.0; d],
            last: vec![0.0; d],
        }
    }

    #[inline]
    fn update(&mut self, x: &[f64]) {
        let first = self.n == 0;
        for (i, &xi) in x.iter().enumerate() {
            let y = xi.powi(self.power);
            if first {
                self.first[i] = y;
            } else {
                self.lagged[i] += y * self.last[i];
            }
            self.sum[i] += y;
            self.squares[i] += y * y;
            self.last[i] = y;
        }
        self.n += 1;
    }
}

/// A chain that can be advanced in blocks: the warm-up runs once, on construction.
pub struct Walk {
    family: u8,
    parameters: Vec<f64>,
    state: State,
    /// The step the chain runs at, the warm-up's when there was one.
    pub step_size: f64,
    jitter: f64,
    /// `1 / variance` per coordinate; empty without a warm-up.
    pub mass_diagonal: Vec<f64>,
    pub warmup_acceptance: f64,
    /// One filter per declared operator, fed every observed draw.
    pub filters: Vec<KalmanStats>,
}

/// What one block of transitions left: the draws if kept, the count taken, `|dU|` each.
pub struct Block {
    pub draws: Vec<f64>,
    pub accepted: usize,
    pub energy_error: Vec<f64>,
}

impl Walk {
    /// A chain at `theta0` on the family `family`, after the warm-up if one is given.
    pub fn new(
        family: u8,
        parameters: Vec<f64>,
        theta0: &[f64],
        step_size: f64,
        seed: u64,
        warmup: Option<&Warmup>,
        powers: &[i32],
    ) -> Result<Self, String> {
        let d = theta0.len();
        if step_size.is_nan() || step_size <= 0.0 {
            return Err(format!("step_size must be positive, got {step_size}"));
        }
        if let Some(w) = warmup {
            if w.proposals < 8 {
                return Err(format!(
                    "warmup must be at least 8 proposals, got {}",
                    w.proposals
                ));
            }
        }
        let energy = energy_of(family, &parameters, d)?;
        let mut scratch = vec![0.0; d];
        let current = energy.potential(theta0, &mut scratch);
        let mut state = State {
            rng: ChaCha8Rng::seed_from_u64(seed),
            position: theta0.to_vec(),
            current,
            proposal: vec![0.0; d],
            scratch,
            scale: vec![1.0; d],
        };
        let (step_size, mass_diagonal, warmup_acceptance, jitter) = match warmup {
            Some(w) => {
                let (step, mass, acceptance) = state.warm_up(&energy, step_size, w)?;
                (step, mass, acceptance, w.jitter)
            }
            None => (step_size, Vec::new(), 0.0, 0.0),
        };
        Ok(Self {
            family,
            parameters,
            state,
            step_size,
            jitter,
            mass_diagonal,
            warmup_acceptance,
            filters: powers.iter().map(|&p| KalmanStats::new(p, d)).collect(),
        })
    }

    /// `n` more transitions, keeping the draws when `store` is set and
    /// feeding the filters when `observe` is.
    pub fn advance(&mut self, n: usize, store: bool, observe: bool) -> Block {
        let d = self.state.position.len();
        // Checked in `new`, so it cannot fail here.
        let energy = energy_of(self.family, &self.parameters, d).expect("checked in new");
        let mut draws = Vec::with_capacity(if store { n * d } else { 0 });
        let mut energy_error = Vec::with_capacity(n);
        let mut accepted = 0;
        for _ in 0..n {
            let step = self.state.jittered(self.step_size, self.jitter);
            let (take, _, error) = self.state.step(&energy, step);
            accepted += usize::from(take);
            energy_error.push(error);
            if store {
                draws.extend_from_slice(&self.state.position);
            }
            if observe {
                for filter in &mut self.filters {
                    filter.update(&self.state.position);
                }
            }
        }
        Block {
            draws,
            accepted,
            energy_error,
        }
    }
}

/// A random-walk Metropolis chain on a declared family, advanced in blocks; see the module docs.
///
/// `family` and `parameters` are `sample.declared.declared_energy`'s;
/// `warmup` proposals of zero run no warm-up, and any other number runs it
/// here, before the first block. `advance` releases the GIL.
#[pyclass(module = "snakes_and_ladders.oxisal")]
pub struct MetropolisWalk {
    walk: Walk,
}

#[pymethods]
impl MetropolisWalk {
    #[new]
    #[pyo3(signature = (
        family, parameters, theta0, step_size, seed, warmup, target_acceptance, step_jitter,
        constants, powers
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        family: u8,
        parameters: PyReadonlyArray1<'_, f64>,
        theta0: PyReadonlyArray1<'_, f64>,
        step_size: f64,
        seed: u64,
        warmup: usize,
        target_acceptance: f64,
        step_jitter: f64,
        constants: (f64, f64, f64),
        powers: Vec<i32>,
    ) -> PyResult<Self> {
        let (parameters, theta0) = (parameters.as_slice()?.to_vec(), theta0.as_slice()?);
        let (gamma, t0, kappa) = constants;
        let adaptation = Warmup {
            proposals: warmup,
            target: target_acceptance,
            jitter: step_jitter,
            gamma,
            t0,
            kappa,
        };
        let walk = py
            .detach(|| {
                Walk::new(
                    family,
                    parameters,
                    theta0,
                    step_size,
                    seed,
                    (warmup > 0).then_some(&adaptation),
                    &powers,
                )
            })
            .map_err(PyValueError::new_err)?;
        Ok(Self { walk })
    }

    /// `n` transitions: the draws flattened `n * d` (empty unless `store`),
    /// the count accepted, and `|dU|` per proposal. With `observe` every
    /// draw feeds the declared operators' filters.
    #[allow(clippy::type_complexity)]
    fn advance<'py>(
        &mut self,
        py: Python<'py>,
        n: usize,
        store: bool,
        observe: bool,
    ) -> (Bound<'py, PyArray1<f64>>, usize, Bound<'py, PyArray1<f64>>) {
        let walk = &mut self.walk;
        let block = py.detach(|| walk.advance(n, store, observe));
        (
            PyArray1::from_vec(py, block.draws),
            block.accepted,
            PyArray1::from_vec(py, block.energy_error),
        )
    }

    /// Operator `index`'s filter statistics: `n`, then the sums of `y`,
    /// `y^2` and `y_t y_(t-1)`, and the first and last `y`.
    #[allow(clippy::type_complexity)]
    fn statistics<'py>(
        &self,
        py: Python<'py>,
        index: usize,
    ) -> PyResult<(usize, [Bound<'py, PyArray1<f64>>; 5])> {
        let f = self
            .walk
            .filters
            .get(index)
            .ok_or_else(|| PyValueError::new_err(format!("no operator {index}")))?;
        Ok((
            f.n,
            [&f.sum, &f.squares, &f.lagged, &f.first, &f.last]
                .map(|v| PyArray1::from_vec(py, v.clone())),
        ))
    }

    /// The step the chain runs at.
    #[getter]
    fn step_size(&self) -> f64 {
        self.walk.step_size
    }

    /// `1 / variance` per coordinate from the warm-up; empty without one.
    #[getter]
    fn mass_diagonal<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.walk.mass_diagonal.clone())
    }

    /// The warm-up's second-window mean acceptance probability; zero without one.
    #[getter]
    fn warmup_acceptance(&self) -> f64 {
        self.walk.warmup_acceptance
    }
}

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
        let mut walk = Walk::new(
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
        let result = Walk::new(
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
        let make =
            || Walk::new(GAUSSIAN, vec![1.0, 4.0], &[0.3, -0.2], 0.8, 11, None, &[]).unwrap();
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
        let mut walk = Walk::new(
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
        assert!(Walk::new(GAUSSIAN, vec![1.0], &[0.0], 0.0, 0, None, &[]).is_err());
        assert!(Walk::new(GAUSSIAN, vec![1.0], &[0.0], 0.1, 0, Some(&warmup(4)), &[]).is_err());
        assert!(Walk::new(
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
