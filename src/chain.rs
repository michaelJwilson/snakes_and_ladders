//! What every compiled chain shares, whatever its proposal (issues #1006, #1008).
//!
//! `sample.hmc.run_chain` is one loop over any `Kernel`: a warm-up of two
//! dual-averaging windows, a burn-in, the draws, and the operators' Kalman
//! filters. This is that loop compiled, over any [`Kernel`] here: the
//! random walk (`metropolis.rs`) and Hamiltonian dynamics (`hmc.rs`)
//! implement one step each and share the rest, so the warm-up and the filter
//! are written once, as they are in Python.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardUniform};

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

impl Warmup {
    /// The warm-up Python asked for: `None` at zero proposals.
    pub fn from_python(
        proposals: usize,
        target: f64,
        jitter: f64,
        (gamma, t0, kappa): (f64, f64, f64),
    ) -> Option<Self> {
        (proposals > 0).then_some(Self {
            proposals,
            target,
            jitter,
            gamma,
            t0,
            kappa,
        })
    }
}

/// `chain._DualAveraging`, term for term.
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

/// One Metropolis transition at a step size, on a metric: `hmc.Kernel`.
pub trait Kernel {
    /// One proposal; returns whether it was taken, its acceptance
    /// probability `min(1, exp(-dH))` (zero where `dH` is nan) and `|dH|`.
    fn step(
        &mut self,
        energy: &Energy<'_>,
        rng: &mut ChaCha8Rng,
        step_size: f64,
    ) -> (bool, f64, f64);
    /// Where the chain is, in the caller's coordinates.
    fn position(&self) -> &[f64];
    /// The metric's scale per coordinate, `M^(-1/2)`: the warm-up's standard deviation.
    fn set_scale(&mut self, scale: Vec<f64>);
}

/// `min(1, exp(log_ratio))` and the accept against `uniform`, as `metropolis._decide`.
///
/// exp overflows only where acceptance is certain, and a nan ratio stays
/// nan, which the comparison refuses.
#[inline]
pub fn decide(log_ratio: f64, uniform: f64) -> (bool, f64) {
    let ratio = if log_ratio > 700.0 {
        f64::INFINITY
    } else {
        log_ratio.exp()
    };
    let probability = if ratio.is_nan() { 0.0 } else { ratio.min(1.0) };
    (uniform < ratio, probability)
}

/// `chain._jittered`: no draw at zero jitter.
#[inline]
fn jittered(rng: &mut ChaCha8Rng, step_size: f64, jitter: f64) -> f64 {
    if jitter == 0.0 {
        return step_size;
    }
    let uniform: f64 = StandardUniform.sample(rng);
    step_size * (1.0 + jitter * (2.0 * uniform - 1.0))
}

/// The two windows of `chain._warm_up`; leaves the kernel on the metric.
///
/// Returns the adapted step, the mass diagonal and the second window's mean
/// acceptance probability.
fn warm_up<K: Kernel>(
    kernel: &mut K,
    energy: &Energy<'_>,
    rng: &mut ChaCha8Rng,
    mut step_size: f64,
    warmup: &Warmup,
) -> Result<(f64, Vec<f64>, f64), String> {
    let first = warmup.proposals / 2;
    let second = warmup.proposals - first;
    let d = kernel.position().len();
    let mut averaging = DualAveraging::new(step_size, warmup);
    // Welford over the first window's second half.
    let (mut count, mut mean, mut m2) = (0.0, vec![0.0; d], vec![0.0; d]);
    for index in 0..first {
        let step = jittered(rng, step_size, warmup.jitter);
        let (_, probability, _) = kernel.step(energy, rng, step);
        step_size = averaging.update(probability);
        if index >= first / 2 {
            count += 1.0;
            for ((m, s), &x) in mean.iter_mut().zip(m2.iter_mut()).zip(kernel.position()) {
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
    kernel.set_scale(variance.iter().map(|v| v.sqrt()).collect());

    let mut averaging = DualAveraging::new(averaging.averaged(), warmup);
    step_size = averaging.averaged();
    let mut total = 0.0;
    for _ in 0..second {
        let step = jittered(rng, step_size, warmup.jitter);
        let (_, probability, _) = kernel.step(energy, rng, step);
        step_size = averaging.update(probability);
        total += probability;
    }
    let mass = variance.iter().map(|v| 1.0 / v).collect();
    Ok((averaging.averaged(), mass, total / second as f64))
}

/// A chain that can be advanced in blocks: the warm-up runs once, on construction.
pub struct Walk<K: Kernel> {
    family: u8,
    parameters: Vec<f64>,
    rng: ChaCha8Rng,
    kernel: K,
    /// The step the chain runs at, the warm-up's when there was one.
    pub step_size: f64,
    jitter: f64,
    /// `1 / variance` per coordinate; empty without a warm-up.
    pub mass_diagonal: Vec<f64>,
    pub warmup_acceptance: f64,
    /// One filter per declared operator, fed every observed draw.
    pub filters: Vec<KalmanStats>,
}

/// What one block of transitions left: the draws if kept, the count taken, `|dH|` each.
pub struct Block {
    pub draws: Vec<f64>,
    pub accepted: usize,
    pub energy_error: Vec<f64>,
}

impl<K: Kernel> Walk<K> {
    /// A chain on the family `family`, from the kernel `make` builds on its
    /// energy, after the warm-up if one is given.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        family: u8,
        parameters: Vec<f64>,
        dimension: usize,
        make: impl FnOnce(&Energy<'_>) -> K,
        step_size: f64,
        seed: u64,
        warmup: Option<&Warmup>,
        powers: &[i32],
    ) -> Result<Self, String> {
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
        let energy = energy_of(family, &parameters, dimension)?;
        let mut kernel = make(&energy);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let (step_size, mass_diagonal, warmup_acceptance, jitter) = match warmup {
            Some(w) => {
                let (step, mass, acceptance) =
                    warm_up(&mut kernel, &energy, &mut rng, step_size, w)?;
                (step, mass, acceptance, w.jitter)
            }
            None => (step_size, Vec::new(), 0.0, 0.0),
        };
        Ok(Self {
            family,
            parameters,
            rng,
            kernel,
            step_size,
            jitter,
            mass_diagonal,
            warmup_acceptance,
            filters: powers
                .iter()
                .map(|&p| KalmanStats::new(p, dimension))
                .collect(),
        })
    }

    /// `n` more transitions, keeping the draws when `store` is set and
    /// feeding the filters when `observe` is.
    pub fn advance(&mut self, n: usize, store: bool, observe: bool) -> Block {
        let d = self.kernel.position().len();
        // Checked in `new`, so it cannot fail here.
        let energy = energy_of(self.family, &self.parameters, d).expect("checked in new");
        let mut draws = Vec::with_capacity(if store { n * d } else { 0 });
        let mut energy_error = Vec::with_capacity(n);
        let mut accepted = 0;
        for _ in 0..n {
            let step = jittered(&mut self.rng, self.step_size, self.jitter);
            let (take, _, error) = self.kernel.step(&energy, &mut self.rng, step);
            accepted += usize::from(take);
            energy_error.push(error);
            if store {
                draws.extend_from_slice(self.kernel.position());
            }
            if observe {
                for filter in &mut self.filters {
                    filter.update(self.kernel.position());
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

/// The Python class every compiled walk is: a constructor that runs the
/// warm-up, `advance`, the filters' statistics and what the warm-up settled
/// on (issues #1006, #1008). `$make` builds the [`Walk`] from the shared
/// arguments and the kernel's own, `$extra`.
#[macro_export]
macro_rules! walk_class {
    ($name:ident, $kernel:ty, $make:path $(, $extra:ident : $ty:ty)*) => {
        /// A chain on a declared family, advanced in blocks; see the module docs.
        ///
        /// `family` and `parameters` are `sample.declared.declared_energy`'s;
        /// `warmup` proposals of zero run no warm-up, and any other number
        /// runs it on construction. `advance` releases the GIL.
        #[pyclass(module = "snakes_and_ladders.oxisal")]
        pub struct $name {
            walk: $crate::chain::Walk<$kernel>,
        }

        #[pymethods]
        impl $name {
            #[new]
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
                $($extra: $ty),*
            ) -> PyResult<Self> {
                let (parameters, theta0) = (parameters.as_slice()?.to_vec(), theta0.as_slice()?);
                let adaptation = $crate::chain::Warmup::from_python(
                    warmup,
                    target_acceptance,
                    step_jitter,
                    constants,
                );
                let walk = py
                    .detach(|| {
                        $make(
                            family,
                            parameters,
                            theta0,
                            step_size,
                            seed,
                            adaptation.as_ref(),
                            &powers,
                            $($extra),*
                        )
                    })
                    .map_err(PyValueError::new_err)?;
                Ok(Self { walk })
            }

            /// `n` transitions: the draws flattened `n * d` (empty unless
            /// `store`), the count accepted, and `|dH|` per proposal. With
            /// `observe` every draw feeds the declared operators' filters.
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

            /// Operator `index`'s filter statistics: `n`, then the sums of
            /// `y`, `y^2` and `y_t y_(t-1)`, and the first and last `y`.
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
    };
}
