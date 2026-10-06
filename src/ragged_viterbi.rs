//! The most probable path of every segment of a ragged log-density (issue #1138).
//!
//! The max-product twin of [`crate::ragged::ragged_posteriors_into`], over the
//! same inputs: a log-density laid end to end, one log-initial each segment
//! restarts at, one log-transition, and an optional switch per position taken
//! as [`SwitchKind`] says. The recursion is
//! `delta_t(j) = max_i delta_{t-1}(i) + ln P_t(i, j) + ln b_t(j)`, a
//! back-pointer per position and state, and a trace back from the best last
//! state.
//!
//! A tie goes to the lower state, as `numpy.argmax` breaks it, so the NumPy
//! oracle `sal.likelihood.ragged.viterbi_oracle` is matched position for
//! position. Under the Kronecker kinds the step is taken in its factors, as
//! the forward pass takes it: the maximum over the slow chain per layer, then
//! the `2 x 2` layer mix, `2 K^2 + 4 K` terms a step against `4 K^2`. The
//! factored maximum keeps the lowest-index rule by comparing the per-layer
//! winners on their full index `2 i + a`.
//!
//! Segments are independent given the parameters, so they are split over
//! `rayon`, each writing its own disjoint slice of the path and its own
//! log-joint: the result is the serial one, bitwise, at any thread count.
//!
//! Plain Rust with no PyO3 types in `ragged_viterbi_into`, so `cargo test`
//! can link it.

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::ragged::{check_inputs, step_kernel, KroneckerStep, SwitchKind, LN_HALF};

/// Whether `(score, index)` beats `(best, best_index)`: higher, or equal and lower.
///
/// The rule `numpy.argmax` applies, including over a row of `-inf`.
#[inline]
fn beats(score: f64, index: usize, best: f64, best_index: usize) -> bool {
    score > best || (score == best && index < best_index)
}

/// Buffers one thread reuses across the segments it decodes.
struct Scratch {
    delta: Vec<f64>,
    next: Vec<f64>,
    back: Vec<u32>,
    switched: Vec<f64>,
    /// Per slow state and layer: the best carried score, and its source index.
    carried: Vec<f64>,
    source: Vec<usize>,
}

impl Scratch {
    fn new(n_states: usize, switched: bool) -> Self {
        Self {
            delta: vec![0.0; n_states],
            next: vec![0.0; n_states],
            back: Vec::new(),
            switched: vec![0.0; if switched { n_states * n_states } else { 0 }],
            carried: vec![0.0; n_states],
            source: vec![0; n_states],
        }
    }
}

/// One Kronecker step of the max-product, in its factors.
///
/// Writes `next[(j, b)] = max_(i, a) delta[(i, a)] + ln P[(i, a), (j, b)]`
/// and the arg-max into `pointers`, a tie to the lower index `2 i + a`.
fn kronecker_step(
    step: &KroneckerStep<'_>,
    delta: &[f64],
    carried: &mut [f64],
    source: &mut [usize],
    next: &mut [f64],
    pointers: &mut [u32],
) {
    let k = step.slow;
    let log_slow = step.log_slow;
    if step.diagonal {
        // Both layers of a slow state pooled: a moved chain forgets its layer.
        for i in 0..k {
            let (low, high) = (delta[2 * i], delta[2 * i + 1]);
            (carried[i], source[i]) = if beats(high, 2 * i + 1, low, 2 * i) {
                (high, 2 * i + 1)
            } else {
                (low, 2 * i)
            };
        }
        for j in 0..k {
            let (mut moved, mut moved_from) = (f64::NEG_INFINITY, usize::MAX);
            for i in (0..k).filter(|&i| i != j) {
                let score = carried[i] + log_slow[i * k + j];
                if beats(score, source[i], moved, moved_from) {
                    (moved, moved_from) = (score, source[i]);
                }
            }
            let moved = moved + LN_HALF;
            let kept = log_slow[j * k + j];
            for b in 0..2 {
                let (mut best, mut from) = (moved, moved_from);
                for a in 0..2 {
                    let score = delta[2 * j + a] + kept + step.mix(a, b);
                    if beats(score, 2 * j + a, best, from) {
                        (best, from) = (score, 2 * j + a);
                    }
                }
                next[2 * j + b] = best;
                pointers[2 * j + b] = from as u32;
            }
        }
        return;
    }
    for j in 0..k {
        // The slow chain's maximum per layer, then the layer's `2 x 2` mix.
        for a in 0..2 {
            let (mut best, mut from) = (f64::NEG_INFINITY, usize::MAX);
            for i in 0..k {
                let score = delta[2 * i + a] + log_slow[i * k + j];
                if beats(score, 2 * i + a, best, from) {
                    (best, from) = (score, 2 * i + a);
                }
            }
            (carried[a], source[a]) = (best, from);
        }
        for b in 0..2 {
            let (mut best, mut from) = (f64::NEG_INFINITY, usize::MAX);
            for a in 0..2 {
                let score = carried[a] + step.mix(a, b);
                if beats(score, source[a], best, from) {
                    (best, from) = (score, source[a]);
                }
            }
            next[2 * j + b] = best;
            pointers[2 * j + b] = from as u32;
        }
    }
}

/// One segment's best path into `path`, and its joint log-probability.
#[allow(clippy::too_many_arguments)]
fn decode_segment(
    density: &[f64],
    n_states: usize,
    start: usize,
    log_initial: &[f64],
    log_transition: &[f64],
    probability: &[f64],
    switch: &[f64],
    kind: SwitchKind,
    scratch: &mut Scratch,
    path: &mut [i64],
) -> f64 {
    let length = path.len();
    let n = n_states;
    let Scratch {
        delta,
        next,
        back,
        switched,
        carried,
        source,
    } = scratch;
    back.clear();
    back.resize(length * n, 0);
    for state in 0..n {
        delta[state] = log_initial[state] + density[state];
    }
    for t in 1..length {
        let pointers = &mut back[t * n..][..n];
        let emitted = &density[t * n..][..n];
        if kind == SwitchKind::StayOrMove {
            let kernel = step_kernel(log_transition, probability, switch, start + t, n, switched);
            for j in 0..n {
                let (mut best, mut from) = (f64::NEG_INFINITY, usize::MAX);
                for i in 0..n {
                    let score = delta[i] + kernel[i * n + j];
                    if beats(score, i, best, from) {
                        (best, from) = (score, i);
                    }
                }
                next[j] = best;
                pointers[j] = from as u32;
            }
        } else {
            let step = KroneckerStep::at(
                log_transition,
                n / 2,
                kind == SwitchKind::KroneckerDiagonal,
                switch[start + t],
            );
            kronecker_step(&step, delta, carried, source, next, pointers);
        }
        for j in 0..n {
            next[j] += emitted[j];
        }
        std::mem::swap(delta, next);
    }
    let (mut best, mut state) = (f64::NEG_INFINITY, usize::MAX);
    for (j, &value) in delta.iter().enumerate() {
        if beats(value, j, best, state) {
            (best, state) = (value, j);
        }
    }
    for t in (0..length).rev() {
        path[t] = state as i64;
        state = back[t * n + state] as usize;
    }
    best
}

/// The most probable path of every segment, and each one's joint log-probability.
///
/// # Parameters
/// As [`crate::ragged::ragged_posteriors_into`] for `log_density`,
/// `n_states`, `lengths`, `log_initial`, `log_transition`, `switch` and
/// `kind`; then
/// - `path`: written, `total`, the state at every position.
/// - `log_joint`: written, one per segment, the maximum joint log-probability.
///
/// # Returns
/// `Ok(())`, or `Err` naming the first violated precondition.
#[allow(clippy::too_many_arguments)]
pub fn ragged_viterbi_into(
    log_density: &[f64],
    n_states: usize,
    lengths: &[usize],
    log_initial: &[f64],
    log_transition: &[f64],
    switch: &[f64],
    kind: SwitchKind,
    path: &mut [i64],
    log_joint: &mut [f64],
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
    if path.len() != total || log_joint.len() != lengths.len() {
        return Err("path and log_joint must match the segments they describe".to_string());
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
        .zip(log_joint.par_iter_mut())
        .for_each_init(
            || Scratch::new(n_states, switched),
            |scratch, ((start, path), joint)| {
                let density = &log_density[start * n_states..(start + path.len()) * n_states];
                *joint = decode_segment(
                    density,
                    n_states,
                    start,
                    log_initial,
                    log_transition,
                    &probability,
                    switch,
                    kind,
                    scratch,
                    path,
                );
            },
        );
    Ok(())
}

/// PyO3 wrapper over [`ragged_viterbi_into`]; converts `Err` to `ValueError`.
///
/// The recursion touches no Python object, so it runs with the GIL released.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(name = "ragged_viterbi")]
#[pyo3(signature = (log_density, lengths, log_initial, log_transition, path, log_joint, switch = None, switch_kind = "stay_or_move"))]
pub fn ragged_viterbi(
    py: Python<'_>,
    log_density: PyReadonlyArray2<f64>,
    lengths: PyReadonlyArray1<i64>,
    log_initial: PyReadonlyArray1<f64>,
    log_transition: PyReadonlyArray2<f64>,
    mut path: PyReadwriteArray1<i64>,
    mut log_joint: PyReadwriteArray1<f64>,
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
    let (path, log_joint) = (path.as_slice_mut()?, log_joint.as_slice_mut()?);
    let switch = match &switch {
        Some(values) => values.as_slice()?,
        None => &[],
    };
    py.detach(|| {
        ragged_viterbi_into(
            density, n_states, &widths, initial, transition, switch, kind, path, log_joint,
        )
    })
    .map_err(PyValueError::new_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two segments decoded against the paths enumerated by hand.
    #[test]
    fn each_segment_restarts_at_the_prior() {
        // Two states that never move; the prior favours state 1, and the
        // second segment's density overrules it.
        let ln = f64::ln;
        let transition = [0.0, f64::NEG_INFINITY, f64::NEG_INFINITY, 0.0];
        let initial = [ln(0.25), ln(0.75)];
        let density = [0.0, 0.0, 0.0, 0.0, 0.0, -3.0, 0.0, -3.0];
        let (mut path, mut joint) = (vec![0_i64; 4], vec![0.0; 2]);
        ragged_viterbi_into(
            &density,
            2,
            &[2, 2],
            &initial,
            &transition,
            &[],
            SwitchKind::StayOrMove,
            &mut path,
            &mut joint,
        )
        .unwrap();
        assert_eq!(path, vec![1, 1, 0, 0]);
        assert!((joint[0] - ln(0.75)).abs() < 1e-15, "{}", joint[0]);
        assert!((joint[1] - ln(0.25)).abs() < 1e-15, "{}", joint[1]);
    }

    #[test]
    fn an_empty_segment_is_refused() {
        let (mut path, mut joint) = (vec![0_i64; 2], vec![0.0; 2]);
        let error = ragged_viterbi_into(
            &[0.0; 2],
            1,
            &[2, 0],
            &[0.0],
            &[0.0],
            &[],
            SwitchKind::StayOrMove,
            &mut path,
            &mut joint,
        )
        .unwrap_err();
        assert!(error.contains("at least 1 position"), "{error}");
    }
}
