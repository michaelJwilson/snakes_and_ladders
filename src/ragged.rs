//! Forward-backward over segments of unequal length, laid end to end.
//!
//! Issue #666. The Python path pads every segment to the longest and masks,
//! which is what lets it batch one step per position of the longest. This one
//! does not pad at all: it walks the segments in place, so it does the work the
//! problem has rather than the work the longest segment implies. Where the
//! lengths are uneven that is the whole difference between the two, and the
//! benchmark reports it rather than this comment claiming it.
//!
//! The recursions restart at each boundary --- `alpha` at the initial
//! distribution, `beta` at one --- and the pair spanning a boundary is not
//! counted as a transition, because the model never took it.
//!
//! The segments are independent given the parameters, so they run in
//! parallel over `rayon` (issue #1191), cut into blocks by their lengths
//! alone; the transition counts are summed within a block and then across
//! blocks in order, so the result is the same, bitwise, at every thread count.
//!
//! Plain Rust with no PyO3 types in `ragged_posteriors_into`, so `cargo test`
//! can link it, per `src/pruning.rs`'s module docs.

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1, PyReadwriteArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::maxflow::on_pool;

/// `ln(exp(a) + exp(b))`, stable and with the `-inf` case exact.
#[inline]
pub(crate) fn log_add(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let (high, low) = if a > b { (a, b) } else { (b, a) };
    high + (low - high).exp().ln_1p()
}

/// `ln(sum(exp(values)))` over a slice, by the same reduction as `log_add`.
#[inline]
pub(crate) fn log_sum(values: &[f64]) -> f64 {
    values
        .iter()
        .fold(f64::NEG_INFINITY, |total, &one| log_add(total, one))
}

/// The log transition into `position`: `log_transition` itself without a
/// switch, else `ln((1 - s) [from == to] + s A[from, to])` written to `out`.
#[inline]
pub(crate) fn step_kernel<'a>(
    log_transition: &'a [f64],
    probability: &[f64],
    switch: &[f64],
    position: usize,
    n_states: usize,
    out: &'a mut [f64],
) -> &'a [f64] {
    if switch.is_empty() {
        return log_transition;
    }
    let s = switch[position];
    for from in 0..n_states {
        for to in 0..n_states {
            let stay = if from == to { 1.0 - s } else { 0.0 };
            out[from * n_states + to] = (stay + s * probability[from * n_states + to]).ln();
        }
    }
    out
}

/// How a switch probability enters the step into a position (issues #1082, #1133).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwitchKind {
    /// `(1 - s) I + s A`: stay, or move by `A` (#1082).
    StayOrMove,
    /// `A ⊗ S`, `S = [[1 - s, s], [s, 1 - s]]`: a slow chain `A` over `K`
    /// states and a fast binary layer switching with `s`, state `(i, a)` at
    /// index `2 i + a`, as `np.kron(A, S)` lays it out.
    Kronecker,
    /// `A ⊗ S` on `A`'s diagonal blocks alone: the layer switches with `s`
    /// where the slow chain stays, and lands on either layer with probability
    /// one half where it moves.
    KroneckerDiagonal,
}

impl SwitchKind {
    /// The kind a Python caller names: `stay_or_move`, `kronecker` or
    /// `kronecker_diagonal`.
    ///
    /// # Errors
    /// Any other name.
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "stay_or_move" => Ok(Self::StayOrMove),
            "kronecker" => Ok(Self::Kronecker),
            "kronecker_diagonal" => Ok(Self::KroneckerDiagonal),
            other => Err(format!(
                "switch_kind is {other:?}; one of stay_or_move, kronecker, kronecker_diagonal"
            )),
        }
    }
}

/// `ln(1 / 2)`: the layer a moved slow chain lands on, under the diagonal switch.
pub(crate) const LN_HALF: f64 = -std::f64::consts::LN_2;

/// One step's Kronecker transition, held as its factors and never as a `2K x 2K` matrix.
///
/// The forward and backward products factor through the slow chain: a
/// `K x K` product per layer, then a `2 x 2` mix per slow state --- `2 K^2 +
/// 4 K` terms a step against the `4 K^2` of the explicit matrix (#1133).
pub(crate) struct KroneckerStep<'a> {
    /// `ln A`, row-major `K x K`.
    pub(crate) log_slow: &'a [f64],
    /// `K`.
    pub(crate) slow: usize,
    /// Whether the layer switches only where the slow chain stays.
    pub(crate) diagonal: bool,
    /// `ln(1 - s)`.
    pub(crate) stay: f64,
    /// `ln s`.
    pub(crate) flip: f64,
}

impl<'a> KroneckerStep<'a> {
    pub(crate) fn at(log_slow: &'a [f64], slow: usize, diagonal: bool, s: f64) -> Self {
        Self {
            log_slow,
            slow,
            diagonal,
            stay: (1.0 - s).ln(),
            flip: s.ln(),
        }
    }

    /// `ln S[a, b]`.
    #[inline]
    pub(crate) fn mix(&self, a: usize, b: usize) -> f64 {
        if a == b {
            self.stay
        } else {
            self.flip
        }
    }

    /// `ln P[(i, a), (j, b)]`, one entry of the matrix this never builds.
    #[inline]
    pub(crate) fn entry(&self, from: usize, to: usize) -> f64 {
        let (i, a, j, b) = (from / 2, from % 2, to / 2, to % 2);
        let layer = if !self.diagonal || i == j {
            self.mix(a, b)
        } else {
            LN_HALF
        };
        self.log_slow[i * self.slow + j] + layer
    }

    /// `out[(j, b)] = ln sum_(i, a) exp(previous[(i, a)] + ln P[(i, a), (j, b)])`.
    ///
    /// `scratch` holds `2 K` entries.
    pub(crate) fn forward(&self, previous: &[f64], scratch: &mut [f64], out: &mut [f64]) {
        let k = self.slow;
        if self.diagonal {
            // Both layers of a slow state pooled: a moved chain forgets its layer.
            for i in 0..k {
                scratch[i] = log_add(previous[2 * i], previous[2 * i + 1]);
            }
            for j in 0..k {
                let mut moved = f64::NEG_INFINITY;
                for i in (0..k).filter(|&i| i != j) {
                    moved = log_add(moved, scratch[i] + self.log_slow[i * k + j]);
                }
                let kept = self.log_slow[j * k + j];
                for b in 0..2 {
                    let stayed = log_add(
                        previous[2 * j] + kept + self.mix(0, b),
                        previous[2 * j + 1] + kept + self.mix(1, b),
                    );
                    out[2 * j + b] = log_add(stayed, moved + LN_HALF);
                }
            }
            return;
        }
        // The slow chain's product per layer, then the layer's `2 x 2` mix.
        for j in 0..k {
            for a in 0..2 {
                let mut carried = f64::NEG_INFINITY;
                for i in 0..k {
                    carried = log_add(carried, previous[2 * i + a] + self.log_slow[i * k + j]);
                }
                scratch[2 * j + a] = carried;
            }
            for b in 0..2 {
                out[2 * j + b] = log_add(
                    scratch[2 * j] + self.mix(0, b),
                    scratch[2 * j + 1] + self.mix(1, b),
                );
            }
        }
    }

    /// `out[(i, a)] = ln sum_(j, b) exp(ln P[(i, a), (j, b)] + ahead[(j, b)])`.
    ///
    /// `scratch` holds `2 K` entries.
    fn backward(&self, ahead: &[f64], scratch: &mut [f64], out: &mut [f64]) {
        let k = self.slow;
        // The layer's mix first, per slow state it lands in.
        for j in 0..k {
            for a in 0..2 {
                scratch[2 * j + a] = log_add(
                    self.mix(a, 0) + ahead[2 * j],
                    self.mix(a, 1) + ahead[2 * j + 1],
                );
            }
        }
        if self.diagonal {
            for i in 0..k {
                let mut moved = f64::NEG_INFINITY;
                for j in (0..k).filter(|&j| j != i) {
                    moved = log_add(
                        moved,
                        self.log_slow[i * k + j] + log_add(ahead[2 * j], ahead[2 * j + 1]),
                    );
                }
                for a in 0..2 {
                    out[2 * i + a] = log_add(
                        self.log_slow[i * k + i] + scratch[2 * i + a],
                        moved + LN_HALF,
                    );
                }
            }
            return;
        }
        for i in 0..k {
            for a in 0..2 {
                let mut carried = f64::NEG_INFINITY;
                for j in 0..k {
                    carried = log_add(carried, self.log_slow[i * k + j] + scratch[2 * j + a]);
                }
                out[2 * i + a] = carried;
            }
        }
    }
}

/// The preconditions [`ragged_posteriors_into`] and
/// [`crate::ragged_viterbi::ragged_viterbi_into`] share, on the inputs alone.
///
/// # Errors
/// The first violated precondition, named.
pub(crate) fn check_inputs(
    log_density: &[f64],
    n_states: usize,
    lengths: &[usize],
    log_initial: &[f64],
    log_transition: &[f64],
    switch: &[f64],
    kind: SwitchKind,
) -> Result<(), String> {
    if n_states == 0 {
        return Err("n_states must be positive".to_string());
    }
    let kronecker = kind != SwitchKind::StayOrMove;
    let slow = n_states / 2;
    if kronecker && (!n_states.is_multiple_of(2) || switch.is_empty()) {
        return Err(format!(
            "a {kind:?} switch takes 2 K states and a switch per position, got {n_states} \
             states and {} switch entries",
            switch.len()
        ));
    }
    let width = if kronecker { slow } else { n_states };
    if log_initial.len() != n_states {
        return Err(format!(
            "log_initial has {} entries for {} states",
            log_initial.len(),
            n_states
        ));
    }
    if log_transition.len() != width * width {
        return Err(format!(
            "log_transition has {} entries for {} states",
            log_transition.len(),
            width
        ));
    }
    // A segment of one position reads no transition: its forward value is
    // the prior plus the emission and its counts are empty (issue #1233).
    if let Some(index) = lengths.iter().position(|&one| one < 1) {
        return Err(format!(
            "segment {} has length {}; a segment carries at least 1 position",
            index, lengths[index]
        ));
    }
    let total: usize = lengths.iter().sum();
    if log_density.len() != total * n_states {
        return Err(format!(
            "log_density has {} entries for {} positions over {} states",
            log_density.len(),
            total,
            n_states
        ));
    }
    if !switch.is_empty() && switch.len() != total {
        return Err(format!(
            "switch has {} entries for {} positions; one per position or none",
            switch.len(),
            total
        ));
    }
    if let Some(index) = switch.iter().position(|&one| !(0.0..=1.0).contains(&one)) {
        return Err(format!(
            "switch[{}] is {}; a switch probability is in [0, 1]",
            index, switch[index]
        ));
    }
    Ok(())
}

/// Blocks of consecutive segments the work is cut into (issue #1191).
///
/// A constant, so the cut, and with it the order the transition counts are
/// summed in, is a function of the lengths alone and never of the thread count.
pub const BLOCKS: usize = 64;

/// The segment ranges of the blocks: consecutive, in segment order, each
/// closed once it holds `ceil(total / BLOCKS)` positions, so at most
/// [`BLOCKS`] of them and each about as long as the next.
fn cut_blocks(lengths: &[usize]) -> Vec<std::ops::Range<usize>> {
    let total: usize = lengths.iter().sum();
    let target = total.div_ceil(BLOCKS).max(1);
    let mut ranges = Vec::with_capacity(BLOCKS.min(lengths.len()));
    let (mut first, mut held) = (0_usize, 0_usize);
    for (segment, &length) in lengths.iter().enumerate() {
        held += length;
        if held >= target {
            ranges.push(first..segment + 1);
            (first, held) = (segment + 1, 0);
        }
    }
    if first < lengths.len() {
        ranges.push(first..lengths.len());
    }
    ranges
}

/// One thread's buffers, reused across the blocks it runs.
struct Workspace {
    alpha: Vec<f64>,
    previous: Vec<f64>,
    forward: Vec<f64>,
    beta: Vec<f64>,
    ahead: Vec<f64>,
    switched: Vec<f64>,
    scratch: Vec<f64>,
}

impl Workspace {
    fn new(n_states: usize, switched: bool, kronecker: bool) -> Self {
        Self {
            alpha: vec![0.0; n_states],
            previous: vec![0.0; n_states],
            forward: Vec::new(),
            beta: vec![0.0; n_states],
            ahead: vec![0.0; n_states],
            switched: vec![0.0; if switched { n_states * n_states } else { 0 }],
            scratch: vec![0.0; if kronecker { n_states } else { 0 }],
        }
    }
}

/// The inputs every block reads and none writes.
struct Shared<'a> {
    log_density: &'a [f64],
    n_states: usize,
    log_initial: &'a [f64],
    log_transition: &'a [f64],
    probability: &'a [f64],
    switch: &'a [f64],
    kind: SwitchKind,
}

/// One block's segments, in order: the serial recursion, with the counts
/// summed into the block's own `counts` from `-inf`.
///
/// `first` is the block's first position; `gamma` and `evidence` are the
/// block's own slices.
fn posteriors_block(
    shared: &Shared<'_>,
    lengths: &[usize],
    first: usize,
    gamma: &mut [f64],
    counts: &mut [f64],
    evidence: &mut [f64],
    work: &mut Workspace,
) {
    let Shared {
        log_density,
        n_states,
        log_initial,
        log_transition,
        probability,
        switch,
        kind,
    } = *shared;
    let kronecker = kind != SwitchKind::StayOrMove;
    let slow = n_states / 2;
    let step_at = |position: usize| {
        KroneckerStep::at(
            log_transition,
            slow,
            kind == SwitchKind::KroneckerDiagonal,
            switch[position],
        )
    };
    let Workspace {
        alpha,
        previous,
        forward,
        beta,
        ahead,
        switched,
        scratch,
    } = work;

    counts.fill(f64::NEG_INFINITY);
    let mut start = first;
    for (segment, &length) in lengths.iter().enumerate() {
        let base = start * n_states;
        let local = (start - first) * n_states;
        forward.clear();
        forward.resize(length * n_states, 0.0);

        // Forward: the chain restarts here, at the prior and not at a kernel.
        for state in 0..n_states {
            alpha[state] = log_initial[state] + log_density[base + state];
            forward[state] = alpha[state];
        }
        for step in 1..length {
            previous.copy_from_slice(alpha);
            if kronecker {
                step_at(start + step).forward(previous, scratch, alpha);
                for state in 0..n_states {
                    alpha[state] += log_density[base + step * n_states + state];
                    forward[step * n_states + state] = alpha[state];
                }
                continue;
            }
            let kernel = step_kernel(
                log_transition,
                probability,
                switch,
                start + step,
                n_states,
                switched,
            );
            for state in 0..n_states {
                let mut carried = f64::NEG_INFINITY;
                for from in 0..n_states {
                    carried = log_add(carried, previous[from] + kernel[from * n_states + state]);
                }
                alpha[state] = carried + log_density[base + step * n_states + state];
                forward[step * n_states + state] = alpha[state];
            }
        }
        let total_evidence = log_sum(alpha);
        evidence[segment] = total_evidence;

        // Backward, and the pair counts as it goes. `beta` is one at the last
        // position: the chain ends, it does not continue into the next segment.
        beta.fill(0.0);
        for state in 0..n_states {
            gamma[local + (length - 1) * n_states + state] =
                forward[(length - 1) * n_states + state] - total_evidence;
        }
        for step in (0..length - 1).rev() {
            let next = base + (step + 1) * n_states;
            for state in 0..n_states {
                ahead[state] = log_density[next + state] + beta[state];
            }
            if kronecker {
                let factors = step_at(start + step + 1);
                factors.backward(ahead, scratch, beta);
                for from in 0..n_states {
                    let row = from * n_states;
                    for to in 0..n_states {
                        let pair =
                            forward[step * n_states + from] + factors.entry(from, to) + ahead[to]
                                - total_evidence;
                        counts[row + to] = log_add(counts[row + to], pair);
                    }
                    gamma[local + step * n_states + from] =
                        forward[step * n_states + from] + beta[from] - total_evidence;
                }
                continue;
            }
            let kernel = step_kernel(
                log_transition,
                probability,
                switch,
                start + step + 1,
                n_states,
                switched,
            );
            for from in 0..n_states {
                let row = from * n_states;
                let mut carried = f64::NEG_INFINITY;
                for to in 0..n_states {
                    let pair = forward[step * n_states + from] + kernel[row + to] + ahead[to]
                        - total_evidence;
                    counts[row + to] = log_add(counts[row + to], pair);
                    carried = log_add(carried, kernel[row + to] + ahead[to]);
                }
                beta[from] = carried;
                gamma[local + step * n_states + from] =
                    forward[step * n_states + from] + carried - total_evidence;
            }
        }
        start += length;
    }
}

/// Posterior marginals, transition counts and evidence, segment by segment.
///
/// # Parameters
/// - `log_density`: row-major `total * n_states`, the segments end to end.
/// - `lengths`: one per segment, each at least 1, summing to `total`.
/// - `log_initial`: `n_states`, the distribution each segment restarts at.
/// - `log_transition`: row-major `n_states * n_states`.
/// - `gamma`: written, `total * n_states`, each row a log posterior.
/// - `counts`: written, `n_states * n_states`, log expected transitions summed
///   over every segment --- the boundary pairs are not among them.
/// - `evidence`: written, one log evidence per segment.
/// - `switch`: empty, or one `s` in `[0, 1]` per position (issue #1082). The
///   step into position `t` then takes `(1 - s[t]) I + s[t] A` with `A` the
///   transition, the stay-or-switch form, built per step from the one `A`
///   rather than stored `T` times; a segment's first entry is never read.
/// - `kind`: how `switch` enters. Under [`SwitchKind::Kronecker`] and
///   [`SwitchKind::KroneckerDiagonal`], `n_states` is `2 K`,
///   `log_transition` is the slow chain's `K x K`, and `switch` is required;
///   the step is taken in its factors, and `counts` are over the `2 K`
///   states, the matrix's own.
///
/// # Threads
/// Segments are independent given the parameters, so the blocks
/// [`cut_blocks`] makes run over `rayon`'s current pool, each writing its own
/// slices of `gamma` and `evidence` and its own partial counts (issue #1191).
/// Within a block the counts are summed position by position, segment by
/// segment, as the serial kernel sums them; the block partials are then
/// summed in block order. The cut depends on `lengths` alone, so every output
/// is the same, bitwise, at every thread count. Against the kernel before
/// #1191, which summed every pair into one accumulator, `gamma` and
/// `evidence` are unchanged bitwise and `counts` move by the rounding of the
/// regrouped sum; with one segment, one block, they too are unchanged.
///
/// # Returns
/// `Ok(())`, or `Err` naming the first violated precondition.
#[allow(clippy::too_many_arguments)]
pub fn ragged_posteriors_into(
    log_density: &[f64],
    n_states: usize,
    lengths: &[usize],
    log_initial: &[f64],
    log_transition: &[f64],
    gamma: &mut [f64],
    counts: &mut [f64],
    evidence: &mut [f64],
    switch: &[f64],
    kind: SwitchKind,
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
    if gamma.len() != log_density.len() || evidence.len() != lengths.len() {
        return Err("gamma and evidence must match the segments they describe".to_string());
    }
    if counts.len() != n_states * n_states {
        return Err(format!(
            "counts has {} entries for {} states",
            counts.len(),
            n_states
        ));
    }
    let kronecker = kind != SwitchKind::StayOrMove;
    let switched = !switch.is_empty() && !kronecker;
    // The transition in probability space, read by every switched step; the
    // unswitched path reads `log_transition` itself and is unchanged.
    let probability: Vec<f64> = if switched {
        log_transition.iter().map(|&one| one.exp()).collect()
    } else {
        Vec::new()
    };
    let shared = Shared {
        log_density,
        n_states,
        log_initial,
        log_transition,
        probability: &probability,
        switch,
        kind,
    };

    // Each block's disjoint slices of `gamma` and `evidence`, and its partial counts.
    let ranges = cut_blocks(lengths);
    let square = n_states * n_states;
    let mut partials = vec![f64::NEG_INFINITY; ranges.len() * square];
    let mut blocks = Vec::with_capacity(ranges.len());
    let (mut rest_gamma, mut rest_evidence, mut first) = (gamma, evidence, 0_usize);
    for (range, partial) in ranges.iter().zip(partials.chunks_mut(square)) {
        let held: usize = lengths[range.clone()].iter().sum();
        let (head, tail) = rest_gamma.split_at_mut(held * n_states);
        let (ev_head, ev_tail) = rest_evidence.split_at_mut(range.len());
        blocks.push((&lengths[range.clone()], first, head, partial, ev_head));
        (rest_gamma, rest_evidence, first) = (tail, ev_tail, first + held);
    }
    blocks.into_par_iter().for_each_init(
        || Workspace::new(n_states, switched, kronecker),
        |work, (widths, first, gamma, partial, evidence)| {
            posteriors_block(&shared, widths, first, gamma, partial, evidence, work);
        },
    );
    // The partials in block order: the first is copied, as `log_add(-inf, x)` is `x`.
    counts.fill(f64::NEG_INFINITY);
    for partial in partials.chunks(square) {
        for (total, &one) in counts.iter_mut().zip(partial) {
            *total = log_add(*total, one);
        }
    }
    Ok(())
}

/// PyO3 wrapper over [`ragged_posteriors_into`]; converts `Err` to `ValueError`.
///
/// The recursion touches no Python object, so it runs with the GIL released
/// and a thread pool runs segments' batches at once (#604, #1059). Its blocks
/// run on `rayon`'s global pool, or on a pool of `threads` when given, as
/// `ising_ground_states` takes it; the result is the same at every count (#1191).
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(name = "ragged_posteriors")]
#[pyo3(signature = (log_density, lengths, log_initial, log_transition, gamma, counts, evidence, switch = None, switch_kind = "stay_or_move", threads = None))]
pub fn ragged_posteriors(
    py: Python<'_>,
    log_density: PyReadonlyArray2<f64>,
    lengths: PyReadonlyArray1<i64>,
    log_initial: PyReadonlyArray1<f64>,
    log_transition: PyReadonlyArray2<f64>,
    mut gamma: PyReadwriteArray2<f64>,
    mut counts: PyReadwriteArray2<f64>,
    mut evidence: PyReadwriteArray1<f64>,
    switch: Option<PyReadonlyArray1<f64>>,
    switch_kind: &str,
    threads: Option<usize>,
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
    let (gamma, counts, evidence) = (
        gamma.as_slice_mut()?,
        counts.as_slice_mut()?,
        evidence.as_slice_mut()?,
    );
    let switch = match &switch {
        Some(values) => values.as_slice()?,
        None => &[],
    };
    py.detach(|| {
        on_pool(threads, || {
            ragged_posteriors_into(
                density, n_states, &widths, initial, transition, gamma, counts, evidence, switch,
                kind,
            )
        })
    })
    .map_err(PyValueError::new_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two segments of two positions, against the value worked by hand.
    #[test]
    fn a_two_segment_batch_sums_its_evidences() {
        let density = vec![0.0; 8];
        let lengths = [2_usize, 2];
        let initial = vec![(0.5_f64).ln(), (0.5_f64).ln()];
        let transition = vec![(0.5_f64).ln(); 4];
        let mut gamma = vec![0.0; 8];
        let mut counts = vec![0.0; 4];
        let mut evidence = vec![0.0; 2];
        ragged_posteriors_into(
            &density,
            2,
            &lengths,
            &initial,
            &transition,
            &mut gamma,
            &mut counts,
            &mut evidence,
            &[],
            SwitchKind::StayOrMove,
        )
        .unwrap();
        // Every density is one and every choice even, so each segment's
        // evidence is one and each posterior a half.
        for one in &evidence {
            assert!((one - 0.0).abs() < 1e-12, "evidence {one}");
        }
        for one in &gamma {
            assert!((one - (0.5_f64).ln()).abs() < 1e-12, "gamma {one}");
        }
    }

    /// The blocks cover the segments in order, at most [`BLOCKS`] of them.
    #[test]
    fn the_blocks_cover_the_segments_in_order() {
        let lengths: Vec<usize> = (0..700).map(|one| 2 + (one * 37) % 400).collect();
        let ranges = cut_blocks(&lengths);
        assert!(ranges.len() <= BLOCKS, "{}", ranges.len());
        assert_eq!(ranges[0].start, 0);
        assert_eq!(ranges[ranges.len() - 1].end, lengths.len());
        for pair in ranges.windows(2) {
            assert_eq!(pair[0].end, pair[1].start);
        }
        assert_eq!(cut_blocks(&[5]), vec![0..1]);
    }

    /// Every output is the same, bitwise, on pools of one to eight threads.
    #[test]
    fn every_thread_count_returns_the_same_bits() {
        let lengths: Vec<usize> = (0..300).map(|one| 2 + (one * 53) % 90).collect();
        let total: usize = lengths.iter().sum();
        let n = 4;
        let density: Vec<f64> = (0..total * n)
            .map(|one| -(((one * 7919) % 101) as f64) / 37.0)
            .collect();
        let initial = vec![(0.25_f64).ln(); n];
        let transition: Vec<f64> = (0..n * n)
            .map(|one| ((1 + one % 5) as f64 / 15.0).ln())
            .collect();
        let run = |threads: usize| {
            let (mut gamma, mut counts) = (vec![0.0; total * n], vec![0.0; n * n]);
            let mut evidence = vec![0.0; lengths.len()];
            on_pool(Some(threads), || {
                ragged_posteriors_into(
                    &density,
                    n,
                    &lengths,
                    &initial,
                    &transition,
                    &mut gamma,
                    &mut counts,
                    &mut evidence,
                    &[],
                    SwitchKind::StayOrMove,
                )
            })
            .unwrap();
            (gamma, counts, evidence)
        };
        let one = run(1);
        for threads in [2, 4, 8] {
            let other = run(threads);
            assert!(one
                .0
                .iter()
                .zip(&other.0)
                .all(|(a, b)| a.to_bits() == b.to_bits()));
            assert!(one
                .1
                .iter()
                .zip(&other.1)
                .all(|(a, b)| a.to_bits() == b.to_bits()));
            assert!(one
                .2
                .iter()
                .zip(&other.2)
                .all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }

    #[test]
    fn an_empty_segment_is_refused() {
        let mut gamma = vec![0.0; 2];
        let mut counts = vec![0.0; 1];
        let mut evidence = vec![0.0; 2];
        let error = ragged_posteriors_into(
            &[0.0; 2],
            1,
            &[2, 0],
            &[0.0],
            &[0.0],
            &mut gamma,
            &mut counts,
            &mut evidence,
            &[],
            SwitchKind::StayOrMove,
        )
        .unwrap_err();
        assert!(error.contains("at least 1 position"), "{error}");
    }

    #[test]
    fn a_one_position_segment_is_the_prior_times_the_emission() {
        // Issue #1233: no transition is read, so the counts stay `-inf` and
        // the posterior is `initial + density` less its log-sum-exp.
        let (initial, density) = ([0.2_f64.ln(), 0.8_f64.ln()], [0.5, -1.5]);
        let mut gamma = vec![0.0; 2];
        let mut counts = vec![0.0; 4];
        let mut evidence = vec![0.0; 1];
        ragged_posteriors_into(
            &density,
            2,
            &[1],
            &initial,
            &[0.0; 4],
            &mut gamma,
            &mut counts,
            &mut evidence,
            &[],
            SwitchKind::StayOrMove,
        )
        .unwrap();
        let joint = [initial[0] + density[0], initial[1] + density[1]];
        let total = log_sum(&joint);
        assert_eq!(evidence[0].to_bits(), total.to_bits());
        for state in 0..2 {
            assert_eq!(gamma[state].to_bits(), (joint[state] - total).to_bits());
        }
        assert!(counts.iter().all(|&count| count == f64::NEG_INFINITY));
    }
}
