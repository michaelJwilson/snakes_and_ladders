//! The BCJR (log-MAP) forward--backward pass over a terminated trellis.
//!
//! `sal.likelihood.convolutional.bcjr` --- the vectorized
//! NumPy implementation --- stays as the oracle this is pinned against, per
//! root `CLAUDE.md`. What Rust buys here is what root `CLAUDE.md` reserves it
//! for: the recursion is sequential in the trellis step and the only axis
//! NumPy can vectorize is the state, which the declared `(7, 5)` register
//! makes four wide. The stress profile of issue #754 put `bcjr` at 95.8% of
//! one pass and 95.5% of an eight-iteration turbo decode, all of it the
//! Python loop over `K + m` steps; `docs/experiments/020` carries the
//! ranking.
//!
//! **The arithmetic is the oracle's, operation for operation.** `logaddexp`
//! below is NumPy's `npy_logaddexp` --- the `x == y` branch included, which
//! is what makes two equal infinities finite --- and [`logsumexp`] is
//! `sal.numerics.logsumexp`'s shift by the row maximum with
//! the sum left to right, which is NumPy's pairwise reduction at the state
//! counts a trellis has. The one departure is storage: the forward metrics
//! are two rows rather than a `(T + 1, n_states)` array, since the posterior
//! at step `t` needs `alpha[t]` and nothing earlier. That changes no
//! operation and no order.
//!
//! **What crosses.** The two trellis tables, `(n_states, 2)` row-major and
//! flattened, and the three log-likelihood ratio vectors --- once per call,
//! contiguous, borrowed. The inverse of `next_state[:, u]` is *derived* here
//! rather than passed: it is `O(n_states)` against the `O(T n_states)` the
//! call does, and deriving it checks the permutation the gather rests on
//! instead of trusting a table built elsewhere.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// The log weight of an edge the trellis does not have.
///
/// `sal.sim.convolutional.IMPOSSIBLE_EDGE`, restated here
/// because the constant crosses no boundary: finite rather than `-inf` so
/// that a shift by a row maximum over unreachable states underflows to zero
/// instead of giving `-inf - (-inf)`.
pub const IMPOSSIBLE_EDGE: f64 = -1e30;

/// `NPY_LOGE2`: what NumPy adds when `logaddexp`'s two arguments are equal.
const LOGE2: f64 = std::f64::consts::LN_2;

/// A shift register's state machine, as the two tables the recursion reads.
///
/// Both are `(n_states, 2)` row-major and flattened, so edge `(s, u)` is
/// index `2 * s + u` --- the layout `sal.sim.convolutional.Trellis`
/// already holds, which is why nothing is copied to build this.
pub struct Trellis<'a> {
    /// The state entered from `s` on input `u`.
    pub next_state: &'a [i64],
    /// The parity bit edge `(s, u)` emits; the systematic bit is `u`.
    pub parity: &'a [u8],
}

/// What one pass returns: `TrellisDecoding`'s three fields.
#[derive(Debug)]
pub struct TrellisDecoding {
    /// `log P(u_t = 0 | y) - log P(u_t = 1 | y)`, one per step.
    pub posterior_llr: Vec<f64>,
    /// The posterior less the systematic and a priori ratios.
    pub extrinsic_llr: Vec<f64>,
    /// `log p(y)`, up to the constant every path shares.
    pub log_evidence: f64,
}

/// `log(exp(x) + exp(y))`, NumPy's `npy_logaddexp` branch for branch.
///
/// The `x == y` case is not an optimization: it is what returns `-inf`
/// rather than `nan` when both arguments are the same infinity, and the
/// oracle takes it, so a port that folded it away would differ on the
/// unreachable states a terminated trellis starts with.
#[inline]
fn logaddexp(x: f64, y: f64) -> f64 {
    if x == y {
        return x + LOGE2;
    }
    let tmp = x - y;
    if tmp > 0.0 {
        x + (-tmp).exp().ln_1p()
    } else if tmp <= 0.0 {
        y + tmp.exp().ln_1p()
    } else {
        // Either argument `nan`, which `tmp` already is.
        tmp
    }
}

/// `log(sum(exp(values)))`, shifted by the maximum as the oracle's is.
///
/// Left to right, which is NumPy's pairwise sum below eight terms; a trellis
/// has `2 ** m` states and the declared register has four.
#[inline]
fn logsumexp(values: &[f64]) -> f64 {
    let mut peak = values[0];
    for &value in &values[1..] {
        if value > peak {
            peak = value;
        }
    }
    let mut total = 0.0;
    for &value in values {
        // `exp(0)` is exactly one, so the maximum's term is added as one in
        // its own place: the sum's order and every bit of it are unchanged,
        // and one exponential per reduction is not taken (issue #997). A
        // non-finite peak takes the exponential, whose `nan` is the answer.
        total += if value == peak && peak.is_finite() {
            1.0
        } else {
            (value - peak).exp()
        };
    }
    peak + total.ln()
}

/// The log weight of every edge at one step: `eq:bcjr`'s branch metric.
///
/// An edge with input `u` and parity `p` carries `-(u (L_s + L_a) + p L_p)`,
/// each term in `eq:ldpc-llr`'s convention. `total` is `L_s + L_a` and
/// `channel` is `L_p`. Inlined and given the caller's buffer: this is the
/// body of the time loop and it allocates nothing.
#[inline]
fn edge_metrics(parity: &[u8], total: f64, channel: f64, gamma: &mut [f64]) {
    let one_input = -total;
    for (bits, metric) in parity.chunks_exact(2).zip(gamma.chunks_exact_mut(2)) {
        metric[0] = -if bits[0] != 0 { channel } else { 0.0 };
        metric[1] = one_input - if bits[1] != 0 { channel } else { 0.0 };
    }
}

/// `source[u * n_states + target]`: the state entering `target` on input `u`.
///
/// `next_state[:, u]` is a permutation --- two states differing only in their
/// oldest cell get different feedback values --- so the forward recursion is
/// a gather through this inverse rather than a scatter, which is the
/// reassociation issue #233 measured at 20.9% of a `K = 1024` pass. Built
/// here, and the permutation is checked rather than assumed: a table that is
/// not one leaves a state with no incoming edge, and a gather through it
/// would read whatever was in the slot.
fn sources(next_state: &[i64], n_states: usize) -> Result<Vec<usize>, String> {
    let mut source = vec![usize::MAX; 2 * n_states];
    for state in 0..n_states {
        for input in 0..2 {
            let target = next_state[2 * state + input];
            if target < 0 || target as usize >= n_states {
                return Err(format!(
                    "next_state[{state}, {input}] is {target}, outside [0, {n_states})"
                ));
            }
            let slot = &mut source[input * n_states + target as usize];
            if *slot != usize::MAX {
                return Err(format!(
                    "next_state[:, {input}] is not a permutation: states {} and {state} \
                     both enter {target}",
                    *slot
                ));
            }
            *slot = state;
        }
    }
    Ok(source)
}

/// Forward--backward over the trellis, with the observation on the edge.
///
/// The recursions of `eq:bcjr` in the log domain: the posterior ratio at
/// step `t` is the log-sum-exp over edges with input zero less that over
/// edges with input one. Exact log-MAP, not the max-log approximation.
///
/// `terminated` says the register was driven back to the zero state, so the
/// backward recursion starts concentrated there rather than uniform.
///
/// # Errors
///
/// If the tables disagree in length, are not `(n_states, 2)`, name a state
/// outside the trellis or are not a permutation in either input; or if the
/// three ratio vectors disagree in length.
pub fn bcjr_impl(
    trellis: &Trellis<'_>,
    systematic: &[f64],
    parity_llr: &[f64],
    apriori: &[f64],
    terminated: bool,
) -> Result<TrellisDecoding, String> {
    if trellis.next_state.len() != trellis.parity.len() {
        return Err(format!(
            "next_state has {} entries and parity {}",
            trellis.next_state.len(),
            trellis.parity.len()
        ));
    }
    if trellis.next_state.is_empty() || !trellis.next_state.len().is_multiple_of(2) {
        return Err(format!(
            "next_state has {} entries, not a non-empty (n_states, 2) table",
            trellis.next_state.len()
        ));
    }
    if systematic.len() != parity_llr.len() || systematic.len() != apriori.len() {
        return Err(format!(
            "systematic {}, parity {} and a priori {} must be the same length",
            systematic.len(),
            parity_llr.len(),
            apriori.len()
        ));
    }
    let n_states = trellis.next_state.len() / 2;
    let next_state = trellis.next_state;
    let source = sources(next_state, n_states)?;
    let length = systematic.len();

    // Every buffer the recursions touch, allocated once. Nothing below
    // allocates inside a time loop.
    let mut beta = vec![IMPOSSIBLE_EDGE; (length + 1) * n_states];
    let mut gamma = vec![0.0f64; 2 * n_states];
    let mut edge = vec![0.0f64; 2 * n_states];
    let mut alpha = vec![IMPOSSIBLE_EDGE; n_states];
    let mut ahead = vec![IMPOSSIBLE_EDGE; n_states];
    let mut posterior_llr = vec![0.0f64; length];
    let mut extrinsic_llr = vec![0.0f64; length];

    let last = length * n_states;
    if terminated {
        beta[last] = 0.0;
    } else {
        beta[last..].fill(0.0);
    }
    for t in (0..length).rev() {
        edge_metrics(
            trellis.parity,
            systematic[t] + apriori[t],
            parity_llr[t],
            &mut gamma,
        );
        // A gather: from state `s` the two edges lead to `next_state[s, u]`.
        let (rows, follower) = beta.split_at_mut((t + 1) * n_states);
        let next = &follower[..n_states];
        for (state, value) in rows[t * n_states..].iter_mut().enumerate() {
            *value = logaddexp(
                next[next_state[2 * state] as usize] + gamma[2 * state],
                next[next_state[2 * state + 1] as usize] + gamma[2 * state + 1],
            );
        }
    }

    alpha[0] = 0.0;
    for t in 0..length {
        edge_metrics(
            trellis.parity,
            systematic[t] + apriori[t],
            parity_llr[t],
            &mut gamma,
        );
        let next = &beta[(t + 1) * n_states..(t + 2) * n_states];
        // `edge[u * n_states + s]`, so each input's states are contiguous and
        // the reduction below walks them in stride order.
        for state in 0..n_states {
            let head = alpha[state];
            edge[state] = head + gamma[2 * state] + next[next_state[2 * state] as usize];
            edge[n_states + state] =
                head + gamma[2 * state + 1] + next[next_state[2 * state + 1] as usize];
        }
        let ratio = logsumexp(&edge[..n_states]) - logsumexp(&edge[n_states..]);
        posterior_llr[t] = ratio;
        extrinsic_llr[t] = ratio - systematic[t] - apriori[t];
        // A gather and not a scatter: exactly one edge with input `u` enters
        // each state, and `source` names it.
        for state in 0..n_states {
            let from_zero = source[state];
            let from_one = source[n_states + state];
            ahead[state] = logaddexp(
                alpha[from_zero] + gamma[2 * from_zero],
                alpha[from_one] + gamma[2 * from_one + 1],
            );
        }
        std::mem::swap(&mut alpha, &mut ahead);
    }

    // `alpha + beta` at the final step and not `alpha` alone: on a terminated
    // trellis only paths ending at the zero state are in the model.
    for (state, value) in ahead.iter_mut().enumerate() {
        *value = alpha[state] + beta[last + state];
    }
    let log_evidence = logsumexp(&ahead);
    Ok(TrellisDecoding {
        posterior_llr,
        extrinsic_llr,
        log_evidence,
    })
}

/// `(posterior_llr, extrinsic_llr, log_evidence)` as it crosses back.
type Decoded<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, f64);

/// One log-MAP forward--backward pass, for
/// `sal.likelihood.convolutional.rust`.
///
/// `next_state` is `2 * n_states` `int64` and `parity` `2 * n_states`
/// `uint8`, both `(n_states, 2)` row-major and flattened; the three ratio
/// vectors are `float64` of one common length. All five are borrowed ---
/// `as_slice` succeeds only for a C-contiguous array, and the wrapper
/// normalizes with `ascontiguousarray`, free when the array already is one.
///
/// Returns `(posterior_llr, extrinsic_llr, log_evidence)`. The extrinsic
/// ratio is formed here rather than in Python: it is one subtraction per
/// step and forming it on the far side would cross the boundary for it.
#[pyfunction]
#[pyo3(signature = (next_state, parity, systematic_llr, parity_llr, apriori_llr, terminated))]
pub fn bcjr_forward_backward<'py>(
    py: Python<'py>,
    next_state: PyReadonlyArray1<'_, i64>,
    parity: PyReadonlyArray1<'_, u8>,
    systematic_llr: PyReadonlyArray1<'_, f64>,
    parity_llr: PyReadonlyArray1<'_, f64>,
    apriori_llr: PyReadonlyArray1<'_, f64>,
    terminated: bool,
) -> PyResult<Decoded<'py>> {
    let trellis = Trellis {
        next_state: next_state.as_slice()?,
        parity: parity.as_slice()?,
    };
    let systematic = systematic_llr.as_slice()?;
    let parity_llr = parity_llr.as_slice()?;
    let apriori = apriori_llr.as_slice()?;
    let decoding = py
        .detach(|| bcjr_impl(&trellis, systematic, parity_llr, apriori, terminated))
        .map_err(PyValueError::new_err)?;
    Ok((
        PyArray1::from_vec(py, decoding.posterior_llr),
        PyArray1::from_vec(py, decoding.extrinsic_llr),
        decoding.log_evidence,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The memory-2 `(7, 5)` recursive systematic register, the declared
    /// fixture's: `next_state` and `parity` as
    /// `sim.convolutional.recursive_systematic_trellis` builds them.
    fn rsc_7_5() -> (Vec<i64>, Vec<u8>) {
        let (feedback, feedforward, memory) = ([1u8, 1, 1], [1u8, 0, 1], 2usize);
        let n_states = 1usize << memory;
        let mut next_state = vec![0i64; 2 * n_states];
        let mut parity = vec![0u8; 2 * n_states];
        for state in 0..n_states {
            let cells: Vec<u8> = (0..memory).map(|i| ((state >> i) & 1) as u8).collect();
            let back = (1..=memory).fold(0u8, |acc, i| acc ^ (feedback[i] & cells[i - 1]));
            let forward = (1..=memory).fold(0u8, |acc, i| acc ^ (feedforward[i] & cells[i - 1]));
            for input in 0..2usize {
                let d = (input as u8) ^ back;
                next_state[2 * state + input] =
                    (((state << 1) & (n_states - 1)) | d as usize) as i64;
                parity[2 * state + input] = (feedforward[0] & d) ^ forward;
            }
        }
        (next_state, parity)
    }

    /// The posteriors by enumeration over every input sequence: the oracle
    /// `likelihood.convolutional.exact_bitwise_posterior` is on the Python
    /// side, written here so `cargo test` has one of its own.
    fn enumerated(
        trellis: &Trellis<'_>,
        systematic: &[f64],
        parity_llr: &[f64],
        apriori: &[f64],
        terminated: bool,
    ) -> (Vec<f64>, f64) {
        let n_states = trellis.next_state.len() / 2;
        let length = systematic.len();
        let mut zero = vec![f64::NEG_INFINITY; length];
        let mut one = vec![f64::NEG_INFINITY; length];
        let mut evidence = f64::NEG_INFINITY;
        for word in 0u32..(1u32 << length) {
            let mut state = 0usize;
            let mut score = 0.0;
            let mut bits = vec![0usize; length];
            for (t, bit) in bits.iter_mut().enumerate() {
                let input = ((word >> t) & 1) as usize;
                *bit = input;
                let emitted = trellis.parity[2 * state + input] as f64;
                score -= input as f64 * (systematic[t] + apriori[t]) + emitted * parity_llr[t];
                state = trellis.next_state[2 * state + input] as usize;
            }
            if terminated && state != 0 {
                continue;
            }
            assert!(state < n_states);
            evidence = logaddexp(evidence, score);
            for (t, &input) in bits.iter().enumerate() {
                let side = if input == 0 {
                    &mut zero[t]
                } else {
                    &mut one[t]
                };
                *side = logaddexp(*side, score);
            }
        }
        let posterior = (0..length).map(|t| zero[t] - one[t]).collect();
        (posterior, evidence)
    }

    fn ratios(length: usize, offset: f64) -> Vec<f64> {
        (0..length)
            .map(|t| offset + 0.7 * ((t as f64) * 1.3).sin() - 0.4 * (t as f64) * 0.11)
            .collect()
    }

    #[test]
    fn the_pass_is_the_enumeration_over_every_input_sequence() {
        let (next_state, parity) = rsc_7_5();
        let trellis = Trellis {
            next_state: &next_state,
            parity: &parity,
        };
        for &length in &[1usize, 5, 9] {
            let systematic = ratios(length, 0.9);
            let parity_llr = ratios(length, -0.3);
            let apriori = ratios(length, 0.2);
            let decoding = bcjr_impl(&trellis, &systematic, &parity_llr, &apriori, false).unwrap();
            let (posterior, evidence) =
                enumerated(&trellis, &systematic, &parity_llr, &apriori, false);
            for (t, &exact) in posterior.iter().enumerate() {
                assert!(
                    (decoding.posterior_llr[t] - exact).abs() < 1e-11,
                    "step {t}: {} against {exact}",
                    decoding.posterior_llr[t]
                );
                let extrinsic = exact - systematic[t] - apriori[t];
                assert!((decoding.extrinsic_llr[t] - extrinsic).abs() < 1e-11);
            }
            assert!((decoding.log_evidence - evidence).abs() < 1e-11);
        }
    }

    #[test]
    fn a_terminated_pass_is_the_enumeration_over_the_sequences_that_end_at_zero() {
        let (next_state, parity) = rsc_7_5();
        let trellis = Trellis {
            next_state: &next_state,
            parity: &parity,
        };
        let length = 8;
        let systematic = ratios(length, 0.5);
        let parity_llr = ratios(length, 0.1);
        let apriori = vec![0.0; length];
        let decoding = bcjr_impl(&trellis, &systematic, &parity_llr, &apriori, true).unwrap();
        let (posterior, evidence) = enumerated(&trellis, &systematic, &parity_llr, &apriori, true);
        for (t, &exact) in posterior.iter().enumerate() {
            assert!(
                (decoding.posterior_llr[t] - exact).abs() < 1e-11,
                "step {t}"
            );
        }
        assert!((decoding.log_evidence - evidence).abs() < 1e-11);
    }

    #[test]
    fn terminating_the_trellis_changes_the_posterior() {
        let (next_state, parity) = rsc_7_5();
        let trellis = Trellis {
            next_state: &next_state,
            parity: &parity,
        };
        let length = 6;
        let systematic = ratios(length, 0.4);
        let parity_llr = ratios(length, -0.2);
        let apriori = vec![0.0; length];
        let open = bcjr_impl(&trellis, &systematic, &parity_llr, &apriori, false).unwrap();
        let closed = bcjr_impl(&trellis, &systematic, &parity_llr, &apriori, true).unwrap();
        assert!(open
            .posterior_llr
            .iter()
            .zip(&closed.posterior_llr)
            .any(|(a, b)| (a - b).abs() > 1e-6));
    }

    #[test]
    fn an_a_priori_ratio_is_the_same_evidence_as_a_systematic_one() {
        let (next_state, parity) = rsc_7_5();
        let trellis = Trellis {
            next_state: &next_state,
            parity: &parity,
        };
        let length = 7;
        let systematic = ratios(length, 0.6);
        let parity_llr = ratios(length, 0.3);
        let apriori = ratios(length, -0.5);
        let folded: Vec<f64> = systematic
            .iter()
            .zip(&apriori)
            .map(|(s, a)| s + a)
            .collect();
        let split = bcjr_impl(&trellis, &systematic, &parity_llr, &apriori, true).unwrap();
        let whole = bcjr_impl(&trellis, &folded, &parity_llr, &vec![0.0; length], true).unwrap();
        for t in 0..length {
            assert!(
                (split.posterior_llr[t] - whole.posterior_llr[t]).abs() < 1e-12,
                "step {t}"
            );
        }
    }

    #[test]
    fn a_table_that_is_not_a_permutation_is_refused() {
        let next_state = vec![0i64, 1, 0, 1];
        let parity = vec![0u8, 1, 1, 0];
        let trellis = Trellis {
            next_state: &next_state,
            parity: &parity,
        };
        let error = bcjr_impl(&trellis, &[0.4], &[0.2], &[0.0], true).unwrap_err();
        assert!(error.contains("not a permutation"), "{error}");
    }

    #[test]
    fn a_state_outside_the_trellis_is_refused() {
        let next_state = vec![0i64, 5, 1, 0];
        let parity = vec![0u8, 1, 1, 0];
        let trellis = Trellis {
            next_state: &next_state,
            parity: &parity,
        };
        let error = bcjr_impl(&trellis, &[0.4], &[0.2], &[0.0], true).unwrap_err();
        assert!(error.contains("outside [0, 2)"), "{error}");
    }

    #[test]
    fn ratio_vectors_of_disagreeing_length_are_refused() {
        let (next_state, parity) = rsc_7_5();
        let trellis = Trellis {
            next_state: &next_state,
            parity: &parity,
        };
        let error = bcjr_impl(&trellis, &[0.4, 0.1], &[0.2], &[0.0, 0.0], true).unwrap_err();
        assert!(error.contains("must be the same length"), "{error}");
    }

    #[test]
    fn logaddexp_keeps_two_equal_infinities_finite_the_way_numpy_does() {
        assert_eq!(
            logaddexp(f64::NEG_INFINITY, f64::NEG_INFINITY),
            f64::NEG_INFINITY
        );
        assert!((logaddexp(0.0, 0.0) - LOGE2).abs() < 1e-16);
    }
}
