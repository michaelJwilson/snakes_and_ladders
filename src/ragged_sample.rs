//! One posterior draw of every segment's hidden path (issue #1170).
//!
//! The sampling twin of [`crate::ragged_viterbi::ragged_viterbi_into`], over
//! the same inputs and one uniform per position: a forward filter in log
//! space, as [`crate::ragged::ragged_posteriors_into`] takes it, then a
//! backward pass drawing the last state from `alpha[T - 1]` and each earlier
//! state from `alpha[t] + ln P_{t+1}(., path[t + 1])`.
//!
//! **The uniforms are drawn in Python and passed in**, as `src/sampling.rs`
//! takes them, so a seeded `numpy.random.Generator` determines the draw and
//! no second stream exists here. A segment starting at `start` owns
//! `uniforms[start..start + length]`, and its `k`-th draw, at position
//! `start + length - 1 - k`, reads `uniforms[start + k]`. Each draw inverts
//! the CDF by the rule `sal.likelihood.forward_backward.inverse_cdf` states:
//! shift by the maximum, exponentiate, accumulate left to right, and take the
//! first state whose cumulative weight exceeds the uniform times the total.
//! The NumPy oracle `sal.likelihood.ragged.sample_paths_oracle` applies the
//! same rule, so the two select the same state wherever their filters agree
//! to within the distance from a uniform to a CDF step.
//!
//! Segments are independent given the parameters and the uniforms, so they
//! are split over `rayon`, each writing its own disjoint slice: the result is
//! the serial one, bitwise, at any thread count.
//!
//! Plain Rust with no PyO3 types in `ragged_sample_paths_into`, so `cargo
//! test` can link it.

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::ragged::{
    check_inputs, log_add, log_sum, shift, step_kernel, KroneckerStep, SwitchKind,
};

/// The state `uniform` selects from unnormalized log weights.
///
/// `cumulative` is scratch of the same length. A state of zero weight is
/// never selected; where rounding leaves the target at the total, the last
/// state of positive weight is.
#[inline]
fn inverse_cdf(log_weights: &[f64], uniform: f64, cumulative: &mut [f64]) -> usize {
    let high = log_weights.iter().fold(
        f64::NEG_INFINITY,
        |best, &one| if one > best { one } else { best },
    );
    let mut total = 0.0_f64;
    for (slot, &one) in cumulative.iter_mut().zip(log_weights) {
        total += (one - high).exp();
        *slot = total;
    }
    let target = uniform * total;
    if let Some(state) = cumulative.iter().position(|&one| one > target) {
        return state;
    }
    // The last state of positive weight: its cumulative exceeds the one before.
    (0..cumulative.len())
        .rev()
        .find(|&state| (log_weights[state] - high).exp() != 0.0)
        .unwrap_or(0)
}

/// Buffers one thread reuses across the segments it draws.
struct Scratch {
    alpha: Vec<f64>,
    switched: Vec<f64>,
    factors: Vec<f64>,
    scores: Vec<f64>,
    cumulative: Vec<f64>,
}

impl Scratch {
    fn new(n_states: usize, switched: bool) -> Self {
        Self {
            alpha: Vec::new(),
            switched: vec![0.0; if switched { n_states * n_states } else { 0 }],
            factors: vec![0.0; n_states],
            scores: vec![0.0; n_states],
            cumulative: vec![0.0; n_states],
        }
    }
}

/// The step into one position, as the draw reads it: one entry at a time.
enum Step<'a> {
    /// The transition itself, unswitched.
    Fixed(&'a [f64]),
    /// `ln((1 - s) [from == to] + s A[from, to])`, with `A` in probability space.
    StayOrMove(&'a [f64], f64),
    /// Either Kronecker kind, held as its factors.
    Kronecker(KroneckerStep<'a>),
}

impl<'a> Step<'a> {
    #[inline]
    fn at(
        log_transition: &'a [f64],
        probability: &'a [f64],
        switch: &[f64],
        kind: SwitchKind,
        n_states: usize,
        position: usize,
    ) -> Self {
        match kind {
            SwitchKind::StayOrMove if switch.is_empty() => Self::Fixed(log_transition),
            SwitchKind::StayOrMove => Self::StayOrMove(probability, switch[position]),
            _ => Self::Kronecker(KroneckerStep::at(
                log_transition,
                n_states / 2,
                kind == SwitchKind::KroneckerDiagonal,
                switch[position],
            )),
        }
    }

    /// `ln P(from, to)`, by the arithmetic the forward filter's step uses.
    #[inline]
    fn entry(&self, n_states: usize, from: usize, to: usize) -> f64 {
        match self {
            Self::Fixed(log_transition) => log_transition[from * n_states + to],
            Self::StayOrMove(probability, s) => {
                let stay = if from == to { 1.0 - s } else { 0.0 };
                (stay + s * probability[from * n_states + to]).ln()
            }
            Self::Kronecker(factors) => factors.entry(from, to),
        }
    }
}

/// One segment's draw into `path`: its joint log-probability and evidence.
#[allow(clippy::too_many_arguments)]
fn draw_segment(
    density: &[f64],
    n_states: usize,
    start: usize,
    log_initial: &[f64],
    log_transition: &[f64],
    probability: &[f64],
    switch: &[f64],
    kind: SwitchKind,
    uniforms: &[f64],
    scratch: &mut Scratch,
    path: &mut [i64],
) -> (f64, f64) {
    let length = path.len();
    let n = n_states;
    let Scratch {
        alpha,
        switched,
        factors,
        scores,
        cumulative,
    } = scratch;
    alpha.clear();
    alpha.resize(length * n, 0.0);
    for state in 0..n {
        alpha[state] = log_initial[state] + density[state];
    }
    // Each row is held less its maximum, as `ragged_posteriors_into` holds
    // it (issues #1262, #1266): the unshifted log filter grows with the
    // position, and its rounding with it. The draws shift by the maximum
    // anyway, so only the evidence reads the offsets, as their sum.
    let mut offsets = shift(&mut alpha[..n]);
    // Forward: the filter `ragged_posteriors_into` runs, kept whole for the draw.
    for t in 1..length {
        let (done, rest) = alpha.split_at_mut(t * n);
        let previous = &done[(t - 1) * n..];
        let current = &mut rest[..n];
        if kind == SwitchKind::StayOrMove {
            let kernel = step_kernel(log_transition, probability, switch, start + t, n, switched);
            for state in 0..n {
                let mut carried = f64::NEG_INFINITY;
                for from in 0..n {
                    carried = log_add(carried, previous[from] + kernel[from * n + state]);
                }
                current[state] = carried;
            }
        } else {
            KroneckerStep::at(
                log_transition,
                n / 2,
                kind == SwitchKind::KroneckerDiagonal,
                switch[start + t],
            )
            .forward(previous, factors, current);
        }
        for state in 0..n {
            current[state] += density[t * n + state];
        }
        offsets += shift(current);
    }
    let evidence = offsets + log_sum(&alpha[(length - 1) * n..]);
    // Backward: the `k`-th draw, at position `length - 1 - k`, reads `uniforms[k]`.
    let mut next = inverse_cdf(&alpha[(length - 1) * n..], uniforms[0], cumulative);
    path[length - 1] = next as i64;
    for t in (0..length - 1).rev() {
        let step = Step::at(log_transition, probability, switch, kind, n, start + t + 1);
        for from in 0..n {
            scores[from] = alpha[t * n + from] + step.entry(n, from, next);
        }
        next = inverse_cdf(scores, uniforms[length - 1 - t], cumulative);
        path[t] = next as i64;
    }
    // The joint in position order, the order the oracle sums it in.
    let first = path[0] as usize;
    let mut joint = log_initial[first] + density[first];
    for t in 1..length {
        let (from, to) = (path[t - 1] as usize, path[t] as usize);
        let step = Step::at(log_transition, probability, switch, kind, n, start + t);
        joint += step.entry(n, from, to) + density[t * n + to];
    }
    (joint, evidence)
}

/// One posterior draw of every segment's path, its joint and its evidence.
///
/// # Parameters
/// As [`crate::ragged::ragged_posteriors_into`] for `log_density`,
/// `n_states`, `lengths`, `log_initial`, `log_transition`, `switch` and
/// `kind`; then
/// - `uniforms`: `total`, each in `[0, 1)`, read as the module docs state.
/// - `path`: written, `total`, the drawn state at every position.
/// - `log_joint`: written, one per segment, `ln p(path, y)`.
/// - `log_evidence`: written, one per segment, `ln p(y)`.
///
/// # Returns
/// `Ok(())`, or `Err` naming the first violated precondition.
#[allow(clippy::too_many_arguments)]
pub fn ragged_sample_paths_into(
    log_density: &[f64],
    n_states: usize,
    lengths: &[usize],
    log_initial: &[f64],
    log_transition: &[f64],
    switch: &[f64],
    kind: SwitchKind,
    uniforms: &[f64],
    path: &mut [i64],
    log_joint: &mut [f64],
    log_evidence: &mut [f64],
) -> Result<(), String> {
    check_inputs(
        log_density,
        n_states,
        lengths,
        log_initial,
        log_transition,
        switch,
        kind,
    )?;
    let total: usize = lengths.iter().sum();
    if uniforms.len() != total {
        return Err(format!(
            "{} uniforms for {} positions; one per position",
            uniforms.len(),
            total
        ));
    }
    if path.len() != total
        || log_joint.len() != lengths.len()
        || log_evidence.len() != lengths.len()
    {
        return Err(
            "path, log_joint and log_evidence must match the segments they describe".to_string(),
        );
    }
    let switched = kind == SwitchKind::StayOrMove && !switch.is_empty();
    // The transition in probability space, read by every stay-or-move step.
    let probability: Vec<f64> = if switched {
        log_transition.iter().map(|&one| one.exp()).collect()
    } else {
        Vec::new()
    };
    let mut segments = Vec::with_capacity(lengths.len());
    let (mut rest, mut start) = (path, 0_usize);
    for &length in lengths {
        let (head, tail) = rest.split_at_mut(length);
        segments.push((start, head));
        (rest, start) = (tail, start + length);
    }
    segments
        .into_par_iter()
        .zip(log_joint.par_iter_mut().zip(log_evidence.par_iter_mut()))
        .for_each_init(
            || Scratch::new(n_states, switched),
            |scratch, ((start, path), (joint, evidence))| {
                let end = start + path.len();
                (*joint, *evidence) = draw_segment(
                    &log_density[start * n_states..end * n_states],
                    n_states,
                    start,
                    log_initial,
                    log_transition,
                    &probability,
                    switch,
                    kind,
                    &uniforms[start..end],
                    scratch,
                    path,
                );
            },
        );
    Ok(())
}

/// PyO3 wrapper over [`ragged_sample_paths_into`]; converts `Err` to `ValueError`.
///
/// The recursion touches no Python object, so it runs with the GIL released.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(name = "ragged_sample_paths")]
#[pyo3(signature = (log_density, lengths, log_initial, log_transition, uniforms, path, log_joint, log_evidence, switch = None, switch_kind = "stay_or_move"))]
pub fn ragged_sample_paths(
    py: Python<'_>,
    log_density: PyReadonlyArray2<f64>,
    lengths: PyReadonlyArray1<i64>,
    log_initial: PyReadonlyArray1<f64>,
    log_transition: PyReadonlyArray2<f64>,
    uniforms: PyReadonlyArray1<f64>,
    mut path: PyReadwriteArray1<i64>,
    mut log_joint: PyReadwriteArray1<f64>,
    mut log_evidence: PyReadwriteArray1<f64>,
    switch: Option<PyReadonlyArray1<f64>>,
    switch_kind: &str,
) -> PyResult<()> {
    let kind = SwitchKind::parse(switch_kind).map_err(PyValueError::new_err)?;
    let density = log_density.as_slice()?;
    let n_states = log_initial.len()?;
    let widths: Vec<usize> = lengths
        .as_slice()?
        .iter()
        .map(|&one| usize::try_from(one).unwrap_or(0))
        .collect();
    let (initial, transition) = (log_initial.as_slice()?, log_transition.as_slice()?);
    let uniforms = uniforms.as_slice()?;
    let (path, log_joint, log_evidence) = (
        path.as_slice_mut()?,
        log_joint.as_slice_mut()?,
        log_evidence.as_slice_mut()?,
    );
    let switch = match &switch {
        Some(values) => values.as_slice()?,
        None => &[],
    };
    py.detach(|| {
        ragged_sample_paths_into(
            density,
            n_states,
            &widths,
            initial,
            transition,
            switch,
            kind,
            uniforms,
            path,
            log_joint,
            log_evidence,
        )
    })
    .map_err(PyValueError::new_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two segments of a chain that never moves: the first uniform alone decides.
    #[test]
    fn a_frozen_chain_draws_its_first_state_throughout() {
        let ln = f64::ln;
        let transition = [0.0, f64::NEG_INFINITY, f64::NEG_INFINITY, 0.0];
        let initial = [ln(0.25), ln(0.75)];
        let density = [0.0; 8];
        let (mut path, mut joint, mut evidence) = (vec![0_i64; 4], vec![0.0; 2], vec![0.0; 2]);
        // Below 0.25 selects state 0; above it, state 1.
        ragged_sample_paths_into(
            &density,
            2,
            &[2, 2],
            &initial,
            &transition,
            &[],
            SwitchKind::StayOrMove,
            &[0.1, 0.9, 0.5, 0.0],
            &mut path,
            &mut joint,
            &mut evidence,
        )
        .unwrap();
        assert_eq!(path, vec![0, 0, 1, 1]);
        assert!((joint[0] - ln(0.25)).abs() < 1e-15, "{}", joint[0]);
        assert!((joint[1] - ln(0.75)).abs() < 1e-15, "{}", joint[1]);
        assert!(
            evidence.iter().all(|&one| one.abs() < 1e-15),
            "{evidence:?}"
        );
    }

    #[test]
    fn a_zero_weight_state_is_never_drawn() {
        let mut cumulative = [0.0; 3];
        let weights = [f64::NEG_INFINITY, 0.0, f64::NEG_INFINITY];
        for uniform in [0.0, 0.5, 1.0 - f64::EPSILON / 2.0] {
            assert_eq!(inverse_cdf(&weights, uniform, &mut cumulative), 1);
        }
    }

    #[test]
    fn a_short_uniform_array_is_refused() {
        let (mut path, mut joint, mut evidence) = (vec![0_i64; 2], vec![0.0; 1], vec![0.0; 1]);
        let error = ragged_sample_paths_into(
            &[0.0; 2],
            1,
            &[2],
            &[0.0],
            &[0.0],
            &[],
            SwitchKind::StayOrMove,
            &[0.5],
            &mut path,
            &mut joint,
            &mut evidence,
        )
        .unwrap_err();
        assert!(error.contains("one per position"), "{error}");
    }
}
