//! Hamiltonian Monte Carlo on a supported kernel, warm-up included (issues #986, #1008, #1220).
//!
//! `sample.hmc.sample` makes one objective call per force evaluation and one
//! Python round trip per leapfrog step, so 1,000 transitions of ten steps at
//! d = 100 took 1.10 s where BlackJAX's compiled chain took 14.1 ms (#963).
//! A torch closure cannot cross the FFI boundary, so what compiles is one of
//! the package's own value+gradient kernels behind `energy::Energy`.
//!
//! The transition is `sample.hmc._transition` with the leapfrog integrator,
//! on the metric the warm-up sets: in the coordinates `phi = theta / s` a
//! normal momentum of variance `T`, kicks of half, one, ..., one, half a
//! step of `s * grad U` around drifts of `s * p`, and acceptance of a uniform
//! on `[0, 1)` below `exp((H(current) - H(proposal)) / T)` (issue #1220). At
//! unit scale and `T = 1` every multiplication by `s` or `sqrt(T)` and the
//! division by `T` are exact, so the unadapted chain is the one the
//! Gaussian-only chain drew before this module (#986). The warm-up, the blocks and
//! the Kalman statistics are `chain.rs`'s, shared with the random walk. The
//! draws come from ChaCha8 seeded by the caller, so the torch route is
//! matched in distribution; `leapfrog_trajectory` exposes the trajectory
//! alone, which the torch integrator pins exactly.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal, StandardUniform};

use crate::chain::{check_temperature, decide, Kernel, Walk, Warmup};
use crate::energy::{build, Energy};
use crate::walk_class;

/// Leapfrog over `n_steps` from `(x, p)` in place, as `hmc.leapfrog` steps
/// it, on the metric of scale `s`: drifts of `s * p`, kicks of `s * grad U`.
/// `start` is `grad U` at `x`, which the caller holds from the step that
/// reached `x` rather than recomputing it, as BlackJAX's integrator state
/// holds it (issue #1008). Leaves `grad U` at the end point in `force` and
/// returns `U` there, taken with it.
#[allow(clippy::too_many_arguments)]
pub fn leapfrog(
    energy: &dyn Energy,
    x: &mut [f64],
    p: &mut [f64],
    step_size: f64,
    n_steps: usize,
    start: &[f64],
    force: &mut [f64],
    scale: &[f64],
) -> f64 {
    for ((pi, fi), si) in p.iter_mut().zip(start).zip(scale) {
        *pi -= 0.5 * step_size * (si * fi);
    }
    let mut potential = f64::NAN;
    for step in 0..n_steps {
        for ((xi, pi), si) in x.iter_mut().zip(p.iter()).zip(scale) {
            *xi += step_size * si * pi;
        }
        let last = step + 1 == n_steps;
        if last {
            potential = energy.value_and_gradient(x, force);
        } else {
            energy.gradient(x, force);
        }
        let kick = if last { 0.5 } else { 1.0 };
        for ((pi, fi), si) in p.iter_mut().zip(force.iter()).zip(scale) {
            *pi -= kick * step_size * (si * fi);
        }
    }
    potential
}

/// `hmc._HamiltonianKernel`: where it is, its potential, the temperature
/// and the buffers a trajectory reuses.
pub struct Hamiltonian {
    n_steps: usize,
    temperature: f64,
    /// `sqrt(T)`, the momentum's standard deviation.
    root: f64,
    position: Vec<f64>,
    current: f64,
    x: Vec<f64>,
    p: Vec<f64>,
    /// `grad U` at `position`, carried from the trajectory that reached it.
    held: Vec<f64>,
    force: Vec<f64>,
    scale: Vec<f64>,
}

impl Hamiltonian {
    /// At `theta0` on `energy`, at unit scale and temperature `temperature`.
    pub fn new(energy: &dyn Energy, theta0: &[f64], n_steps: usize, temperature: f64) -> Self {
        let d = theta0.len();
        let mut held = vec![0.0; d];
        let current = energy.value_and_gradient(theta0, &mut held);
        Self {
            n_steps,
            temperature,
            root: temperature.sqrt(),
            position: theta0.to_vec(),
            current,
            x: vec![0.0; d],
            p: vec![0.0; d],
            held,
            force: vec![0.0; d],
            scale: vec![1.0; d],
        }
    }
}

impl Kernel for Hamiltonian {
    fn step(
        &mut self,
        energy: &dyn Energy,
        rng: &mut ChaCha8Rng,
        step_size: f64,
    ) -> (bool, f64, f64) {
        for pi in self.p.iter_mut() {
            let z: f64 = StandardNormal.sample(rng);
            *pi = z * self.root;
        }
        let current = self.current + 0.5 * self.p.iter().map(|v| v * v).sum::<f64>();
        self.x.copy_from_slice(&self.position);
        let proposed_potential = leapfrog(
            energy,
            &mut self.x,
            &mut self.p,
            step_size,
            self.n_steps,
            &self.held,
            &mut self.force,
            &self.scale,
        );
        let proposed = proposed_potential + 0.5 * self.p.iter().map(|v| v * v).sum::<f64>();
        let uniform: f64 = StandardUniform.sample(rng);
        let (take, probability) = decide((current - proposed) / self.temperature, uniform);
        if take {
            std::mem::swap(&mut self.position, &mut self.x);
            std::mem::swap(&mut self.held, &mut self.force);
            self.current = proposed_potential;
        }
        (take, probability, (proposed - current).abs())
    }

    fn position(&self) -> &[f64] {
        &self.position
    }

    fn set_scale(&mut self, scale: Vec<f64>) {
        self.scale = scale;
    }
}

/// `Walk::new` for Hamiltonian dynamics of `n_steps` leapfrog steps from
/// `theta0`, targeting `exp(-U / temperature)`.
#[allow(clippy::too_many_arguments)]
pub fn walk(
    energy: Box<dyn Energy>,
    theta0: &[f64],
    step_size: f64,
    seed: u64,
    warmup: Option<&Warmup>,
    powers: &[i32],
    temperature: f64,
    n_steps: usize,
) -> Result<Walk<Hamiltonian>, String> {
    if n_steps == 0 {
        return Err("n_steps must be at least 1, got 0".to_string());
    }
    check_temperature(temperature)?;
    Walk::new(
        energy,
        theta0.len(),
        |energy| Hamiltonian::new(energy, theta0, n_steps, temperature),
        step_size,
        seed,
        warmup,
        powers,
    )
}

walk_class!(HmcWalk, Hamiltonian, walk, n_steps: usize);

/// One leapfrog trajectory from `(theta, momentum)` at unit scale; returns the end point.
///
/// The trajectory runs with the GIL released: it touches no Python object
/// between reading its arguments and building the two arrays (#604, #1059).
#[pyfunction]
#[allow(clippy::type_complexity)]
#[pyo3(signature = (kernel, data, theta, momentum, step_size, n_steps))]
pub fn leapfrog_trajectory<'py>(
    py: Python<'py>,
    kernel: &str,
    data: &Bound<'py, pyo3::types::PyDict>,
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
    let energy = build(kernel, data, x.len())?;
    py.detach(|| {
        let (mut start, mut force) = (vec![0.0; x.len()], vec![0.0; x.len()]);
        let scale = vec![1.0; x.len()];
        energy.gradient(&x, &mut start);
        leapfrog(
            &*energy, &mut x, &mut p, step_size, n_steps, &start, &mut force, &scale,
        );
    });
    Ok((PyArray1::from_vec(py, x), PyArray1::from_vec(py, p)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::energy::{Gaussian, Rosenbrock};

    #[test]
    fn the_chain_samples_the_diagonal_gaussian() {
        let d = 50;
        let precision: Vec<f64> = (0..d)
            .map(|i| 1.0 + 3.0 * i as f64 / (d - 1) as f64)
            .collect();
        // 1.2 time units: far from a half or a full turn at every
        // frequency here, which would leave a coordinate nearly frozen.
        let n = 20_000;
        let mut chain = walk(
            Box::new(Gaussian::new(precision.clone(), d).unwrap()),
            &vec![0.0; d],
            0.15,
            986,
            None,
            &[],
            1.0,
            8,
        )
        .unwrap();
        chain.advance(100, false, false);
        let out = chain.advance(n, true, false);
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
        let dense = Gaussian::new(precision.to_vec(), 3).unwrap();
        let (x0, p0) = ([0.3, -1.2, 0.8], [1.1, 0.4, -0.7]);
        let (mut x, mut p, mut f) = (x0.to_vec(), p0.to_vec(), vec![0.0; 3]);
        let scale = [1.0, 0.5, 2.0];
        let mut start = vec![0.0; 3];
        dense.gradient(&x, &mut start);
        leapfrog(&dense, &mut x, &mut p, 0.1, 25, &start, &mut f, &scale);
        for v in &mut p {
            *v = -*v;
        }
        start.copy_from_slice(&f);
        leapfrog(&dense, &mut x, &mut p, 0.1, 25, &start, &mut f, &scale);
        for i in 0..3 {
            assert!((x[i] - x0[i]).abs() < 1e-12);
            assert!((-p[i] - p0[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn the_warm_up_reaches_the_target_on_rosenbrock() {
        let warmup = Warmup {
            proposals: 2_000,
            target: 0.65,
            jitter: 0.2,
            gamma: 0.2,
            t0: 10.0,
            kappa: 0.75,
        };
        let chain = walk(
            Box::new(Rosenbrock { a: 1.0, b: 1.0 }),
            &[-1.2, 1.0],
            0.05,
            1008,
            Some(&warmup),
            &[],
            1.0,
            10,
        )
        .unwrap();
        assert!(
            (chain.warmup_acceptance - 0.65).abs() < 0.1,
            "{}",
            chain.warmup_acceptance
        );
        assert!(chain
            .mass_diagonal
            .iter()
            .all(|m| m.is_finite() && *m > 0.0));
    }
}
