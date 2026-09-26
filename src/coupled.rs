//! The E step of the coupled spatio-sequential model, ported from
//! `python/sal/likelihood/spatio_sequential/__init__.py` (the NumPy
//! oracle) to Rust, exposed to Python via PyO3 as
//! `sal.oxisal.class_posteriors` and
//! `...external_field`.
//!
//! **What the port is for.** At the declared 5,041-vertex instance
//! (`tests/regression/fixtures/spatio_sequential_counts/stress.yaml`) the
//! field costs `S x V x M x K` emission log-densities per sweep --- 1.0e9 at
//! bin factor 10 --- and each of those is three `lgamma` calls in the NumPy
//! path. `cProfile`'s self time there is dominated by `torch.lgamma`; the
//! numbers are in `STATUS.md`.
//!
//! **Table reads replace the special functions.** The counts are integers in
//! a range of a few thousand, so the emission log-density under state `k` of
//! class `m` is a *function of an integer* and can be tabulated once per call:
//! `total_table[y, m, k]` and `success_table[z, m, k]`, built by the NumPy
//! families themselves, so the kernel does two loads and an add where the
//! oracle does three `lgamma` calls. Tabulating in Python rather than here
//! also means the tables are the oracle's own arithmetic, and this crate
//! needs no special-function dependency.
//!
//! **The tables are indexed count-major.** `[y][m][k]` and not `[m][k][y]`:
//! the field's inner loop wants every class and state *at one count*, which
//! count-major makes `M x K` contiguous doubles --- 800 bytes at the declared
//! size, a handful of cache lines --- where state-major would scatter them
//! across `M x K` lines one stride apart.
//!
//! **The observations are walked position-major.** They cross the boundary in
//! the `(S, n_nodes)` layout `sal.sim` already holds them in,
//! and the loops run position outer, vertex inner, so both count arrays are
//! read in stride order. The live accumulator is then one `(M, K)` block per
//! position, which stays in L1 at every declared size; a vertex-major walk
//! would read the counts with a stride of `n_nodes` instead.
//!
//! **Forward--backward is scaled, and the oracle's is in the log domain.**
//! The class density is a sum over a class's members, so at 500 members it
//! runs to thousands in magnitude and `exp` of it underflows: each position's
//! row maximum is subtracted before exponentiating and added back into the
//! evidence. That is a different arithmetic from the oracle's `logsumexp`,
//! which is why the two are pinned at a relative tolerance rather than
//! bitwise (`likelihood/CLAUDE.md`).
//!
//! **A negative-binomial exposure is factored, not tabulated** (issue #1064).
//! Under a per-observation exposure `c` the total's density is a function of
//! the count and a real number, and a table by count and distinct exposure has
//! as many rows as there are observations. It splits instead:
//! `log p(y | m, k, c) = A[y, m, k] + r ln(r / t) + y ln(mu c / t)` with
//! `t = r + mu c`, `A` the `lgamma` terms the caller tabulates by count through
//! the family, and the two exposure terms computed here in the pass that
//! already sums the member scores. The kernel gains `ln` and no special
//! function, and nothing of size `S x V x M x K` is built. The terms are the
//! family's own, in its order: written as `y ln c - (y + r) ln t` with
//! `r ln r + y ln mu` moved into `A`, one `ln` per score fewer, the per-score
//! difference from the family was 263 ulp at the ci instance, the cancellation
//! of terms near `y ln mu`; in the family's order it is 2.3 ulp.
//!
//! **A beta-binomial trial count may be factored too** (issue #1064). Under a
//! per-observation trial count `n` the second channel's density splits into
//! terms each indexed by one integer: `lgamma(z + a)` by the successes `z`,
//! `lgamma(n - z + b)` by the failures, `lgamma(n + a + b)` by the trials, and
//! `lgamma(j + 1)` at each of the three. The caller tabulates each through the
//! family and [`TrialTerm`] adds them per score in the family's order of
//! operations, so each score is the family's `log_density` bit for bit and no
//! table is indexed by a `(count, trial count)` pair. It is one of three layouts
//! the caller chooses; the default tabulates the pairs, which reads one table
//! row per score where this adds six terms, and the field was 14% faster for it
//! at stress.
//!
//! The implementations are plain Rust with no PyO3 types so `cargo test` and
//! `benches/` can link them, per `src/pruning.rs`'s module docs.

use numpy::{PyReadonlyArray1, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// The shape of one coupled E step.
///
/// Carried as one value rather than four arguments because every function
/// here takes all four and a caller that transposes two of them silently
/// computes something else.
#[derive(Clone, Copy, Debug)]
pub struct CoupledShape {
    /// `S`, sequential positions.
    pub n_positions: usize,
    /// `V`, lattice vertices.
    pub n_nodes: usize,
    /// `M`, class labels.
    pub n_classes: usize,
    /// `K`, hidden states per class.
    pub n_states: usize,
}

impl CoupledShape {
    /// `M * K`, the width of one position's block of the emission tables.
    #[inline]
    fn block(&self) -> usize {
        self.n_classes * self.n_states
    }
}

/// A table row index as a `usize`.
///
/// `usize::from` is not defined for `u32` --- `usize` may be 32 bits --- and
/// the rows are validated against the table's extent before any of this runs,
/// so the widening is stated once here rather than as a cast at four sites.
#[inline]
fn row_index(row: u32) -> usize {
    row as usize
}

/// The emission log-density of every class and state, tabulated by count.
///
/// `total[(y * M + m) * K + k]` is `log p(row y | class m, state k)` in the
/// first channel and `success[(z * M + m) * K + k]` the same in the second. A
/// row past a table's extent is a caller error and is refused, never clamped:
/// the table is built from the rows it will be indexed by.
///
/// A *row* is whatever the caller tabulated. Without a covariate it is the
/// count itself, which is what this kernel indexed by before issue #658 and
/// still does, bit for bit. With one, the density is a function of the count
/// *and* the covariate, so the caller tabulates the distinct pairs and hands
/// the row each observation falls in. The kernel is the same two loads and an
/// add either way: it never knew what the index meant, only that the table was
/// built from it.
pub struct EmissionTables<'a> {
    /// The first channel's table, `n_totals * M * K`.
    pub total: &'a [f64],
    /// The second channel's table, `n_successes * M * K`.
    pub success: &'a [f64],
    /// The first channel's exposure term, where it carries one. The first
    /// channel's row is then the count itself and `total` is the table `A`
    /// of [`ExposureTerm`].
    pub exposure: Option<ExposureTerm<'a>>,
    /// The second channel's trial count, where it carries one. The second
    /// channel's row is then the successes `z` and `success` is the table
    /// `lgamma(z + a)` of [`TrialTerm`].
    pub trials: Option<TrialTerm<'a>>,
}

/// The negative binomial's exposure, factored out of the first channel's table.
///
/// `log p(y | m, k, c) = B[y, m, k] + y ln c - (y + r) ln t`, with
/// `t = r_mk + mu_mk c` and `B[y, m, k] = lgamma(y + r) - lgamma(r) -
/// lgamma(y + 1) + r ln r + y ln mu` the first channel's table. A zero exposure marks the count
/// unobserved and it scores zero under every class and state, as the family
/// scores it.
#[derive(Clone, Copy)]
pub struct ExposureTerm<'a> {
    /// `S * V` exposures, position-major, each non-negative.
    pub exposure: &'a [f64],
    /// `M * K` dispersions `r`, row-major.
    pub dispersion: &'a [f64],
    /// `M * K` means `mu` at unit exposure, row-major.
    pub mean: &'a [f64],
}

impl ExposureTerm<'_> {
    /// The first channel's score at one observation, for every state of class `m`.
    ///
    /// One logarithm per class and state, `ln t`, and `ln c` once per call:
    /// the form chosen for speed over the family's order of operations, which
    /// costs two logarithms and two divisions per score and agrees to fewer
    /// ulp (issue #1064).
    #[inline]
    pub(crate) fn score_into(
        &self,
        table: &[f64],
        count: u32,
        c: f64,
        from: usize,
        out: &mut [f64],
    ) {
        if c == 0.0 {
            out.fill(0.0);
            return;
        }
        let y = f64::from(count);
        let y_log_c = y * c.ln();
        // Pre-sliced rows, walked in three passes: `t`, `ln t`, the score. It
        // is the same arithmetic in the same order, so bitwise to indexing
        // `from + k`; the stress kernel's E step measured 0.88 s against
        // 1.07 s indexed (issue #1064).
        let n = out.len();
        let (dispersion, mean, table) = (
            &self.dispersion[from..][..n],
            &self.mean[from..][..n],
            &table[..n],
        );
        for ((cell, &r), &mu) in out.iter_mut().zip(dispersion).zip(mean) {
            *cell = r + c * mu;
        }
        for cell in out.iter_mut() {
            *cell = cell.ln();
        }
        for ((cell, &r), &b) in out.iter_mut().zip(dispersion).zip(table) {
            *cell = b + y_log_c - (y + r) * *cell;
        }
    }

    /// The same score in the family's order of operations, `table` the row of `A`.
    ///
    /// `A[y] + r ln(r / t) + y ln(mu c / t)`, `t = r + mu c`, `A[y] = lgamma(y +
    /// r) - lgamma(r) - lgamma(y + 1)` (`sal.emissions.nb.count_log_factor`):
    /// the operations of `NegativeBinomialEmission.log_density` on the same
    /// numbers, so each score differs from it by the rounding of the two `ln`
    /// alone. Two logarithms and two divisions per score where
    /// [`Self::score_into`] takes one logarithm (issue #1132).
    #[inline]
    pub(crate) fn score_family_into(
        &self,
        table: &[f64],
        count: u32,
        c: f64,
        from: usize,
        out: &mut [f64],
    ) {
        if c == 0.0 {
            out.fill(0.0);
            return;
        }
        let y = f64::from(count);
        let n = out.len();
        let (dispersion, mean, table) = (
            &self.dispersion[from..][..n],
            &self.mean[from..][..n],
            &table[..n],
        );
        for (((cell, &r), &mu), &a) in out.iter_mut().zip(dispersion).zip(mean).zip(table) {
            let rate = c * mu;
            let t = r + rate;
            *cell = (a + r * (r / t).ln()) + y * (rate / t).ln();
        }
    }

    /// Check the lengths against the shape, and every exposure against its support.
    pub(crate) fn validate(&self, shape: &CoupledShape) -> Result<(), String> {
        let observations = shape.n_positions * shape.n_nodes;
        if self.exposure.len() != observations {
            return Err(format!(
                "exposure has {} entries, expected S * V = {observations}",
                self.exposure.len()
            ));
        }
        for (name, values) in [("dispersion", self.dispersion), ("mean", self.mean)] {
            if values.len() != shape.block() {
                return Err(format!(
                    "{name} has {} entries, expected M * K = {}",
                    values.len(),
                    shape.block()
                ));
            }
        }
        if self.exposure.iter().any(|&c| !(c >= 0.0 && c.is_finite())) {
            return Err(
                "every exposure must be finite and non-negative; zero marks the count unobserved"
                    .to_string(),
            );
        }
        Ok(())
    }
}

/// The beta-binomial's trial count, factored out of the second channel's table.
///
/// `log p(z | m, k, n) = lgamma(n + 1) - lgamma(z + 1) - lgamma(n - z + 1) +
/// U[z, m, k] + V[n - z, m, k] - W[n, m, k] + lgamma(a + b) - lgamma(a) -
/// lgamma(b)`, with `U[z] = lgamma(z + a)` the second channel's table, `V[j] =
/// lgamma(j + b)` and `W[n] = lgamma(n + a + b)`: the family's nine terms, each
/// a function of one integer or of none. A zero trial count marks the successes
/// unobserved and they score zero; successes past their trial count score
/// `-inf`; both as the family scores them.
#[derive(Clone, Copy)]
pub struct TrialTerm<'a> {
    /// `S * V` trial counts `n`, position-major.
    pub trials: &'a [u32],
    /// `V[j, m, k] = lgamma(j + b_mk)`, `extent * M * K`.
    pub failure: &'a [f64],
    /// `W[n, m, k] = lgamma(n + a_mk + b_mk)`, `extent * M * K`.
    pub trial: &'a [f64],
    /// `lgamma(j + 1)`, `extent` entries.
    pub log_factorial: &'a [f64],
    /// `lgamma(a + b)`, `lgamma(a)` and `lgamma(b)`, each `M * K` row-major, in
    /// that order.
    pub log_beta: &'a [f64],
}

impl TrialTerm<'_> {
    /// The second channel's score at one observation, for `out.len()` columns from `from`.
    ///
    /// `table` is the row of `U` at the successes, already offset by `from`.
    /// The sum runs in the family's order --- the three state-free terms
    /// first, then `U`, `V`, `W` and the three `lgamma` of the Beta
    /// function --- so each score is its `log_density` to the bit.
    #[inline]
    pub(crate) fn score_into(
        &self,
        table: &[f64],
        successes: u32,
        trials: u32,
        block: usize,
        from: usize,
        out: &mut [f64],
    ) {
        if trials == 0 {
            out.fill(0.0);
            return;
        }
        if successes > trials {
            out.fill(f64::NEG_INFINITY);
            return;
        }
        let (n, z) = (row_index(trials), row_index(successes));
        let free = (self.log_factorial[n] - self.log_factorial[z]) - self.log_factorial[n - z];
        let len = out.len();
        let (table, failure, trial) = (
            &table[..len],
            &self.failure[(n - z) * block + from..][..len],
            &self.trial[n * block + from..][..len],
        );
        let (total, alpha, beta) = (
            &self.log_beta[from..][..len],
            &self.log_beta[block + from..][..len],
            &self.log_beta[2 * block + from..][..len],
        );
        for i in 0..len {
            out[i] =
                (((((free + table[i]) + failure[i]) - trial[i]) + total[i]) - alpha[i]) - beta[i];
        }
    }

    /// Check the lengths against the shape, and every trial count against the tables' extent.
    pub(crate) fn validate(&self, shape: &CoupledShape) -> Result<(), String> {
        let observations = shape.n_positions * shape.n_nodes;
        if self.trials.len() != observations {
            return Err(format!(
                "trials has {} entries, expected S * V = {observations}",
                self.trials.len()
            ));
        }
        let block = shape.block();
        if self.log_beta.len() != 3 * block {
            return Err(format!(
                "log_beta has {} entries, expected 3 * M * K = {}",
                self.log_beta.len(),
                3 * block
            ));
        }
        let largest = self.trials.iter().copied().max().map_or(0, row_index);
        for (name, table, width) in [
            ("failure", self.failure, block),
            ("trial", self.trial, block),
            ("log_factorial", self.log_factorial, 1),
        ] {
            if !table.len().is_multiple_of(width) {
                return Err(format!(
                    "the {name} table has {} entries, not a multiple of M * K = {width}",
                    table.len()
                ));
            }
            let extent = table.len() / width;
            if largest >= extent {
                return Err(format!(
                    "a trial count of {largest} is past the {name} table's extent {extent}"
                ));
            }
        }
        Ok(())
    }
}

impl EmissionTables<'_> {
    /// The first channel's scores at one observation, `out.len()` columns from `from`.
    ///
    /// The table's row itself where there is no exposure, and `out` filled by
    /// [`ExposureTerm::score_into`] where there is one.
    #[inline]
    fn total_scores<'b>(
        &'b self,
        block: usize,
        observation: usize,
        count: u32,
        from: usize,
        out: &'b mut [f64],
    ) -> &'b [f64] {
        let row = &self.total[row_index(count) * block + from..][..out.len()];
        match &self.exposure {
            Some(term) => {
                term.score_into(row, count, term.exposure[observation], from, out);
                out
            }
            None => row,
        }
    }

    /// The second channel's scores at one observation, as [`Self::total_scores`].
    #[inline]
    fn success_scores<'b>(
        &'b self,
        block: usize,
        observation: usize,
        count: u32,
        from: usize,
        out: &'b mut [f64],
    ) -> &'b [f64] {
        let row = &self.success[row_index(count) * block + from..][..out.len()];
        match &self.trials {
            Some(term) => {
                term.score_into(row, count, term.trials[observation], block, from, out);
                out
            }
            None => row,
        }
    }

    /// Whether either channel is completed per observation rather than read from its table.
    #[inline]
    fn is_factored(&self) -> bool {
        self.exposure.is_some() || self.trials.is_some()
    }

    /// Check that both tables are whole numbers of blocks and cover the counts.
    fn validate(
        &self,
        shape: &CoupledShape,
        totals: &[u32],
        successes: &[u32],
    ) -> Result<(), String> {
        let block = shape.block();
        if block == 0 {
            return Err("a coupled instance has at least one class and one state".to_string());
        }
        for (name, table, counts) in [
            ("total", self.total, totals),
            ("success", self.success, successes),
        ] {
            if !table.len().is_multiple_of(block) {
                return Err(format!(
                    "the {name} table has {} entries, not a multiple of M * K = {block}",
                    table.len()
                ));
            }
            let extent = table.len() / block;
            match counts.iter().max() {
                Some(&largest) if row_index(largest) >= extent => {
                    return Err(format!(
                        "a {name} count of {largest} is past the table's extent {extent}"
                    ));
                }
                _ => {}
            }
        }
        if let Some(term) = &self.exposure {
            term.validate(shape)?;
        }
        match &self.trials {
            Some(term) => term.validate(shape),
            None => Ok(()),
        }
    }
}

/// Accumulate every class's summed member scores, `(M, S, K)` row-major.
///
/// The `class_log_density` of the oracle: given the labels the classes
/// decouple, and a vertex contributes to its own class alone.
fn class_log_density(
    shape: &CoupledShape,
    tables: &EmissionTables<'_>,
    totals: &[u32],
    successes: &[u32],
    labels: &[i64],
    density: &mut [f64],
) {
    let (n_positions, n_nodes, n_states) = (shape.n_positions, shape.n_nodes, shape.n_states);
    let block = shape.block();
    density.fill(0.0);
    if tables.is_factored() {
        // One observation's scores per channel, reused across observations;
        // a channel read from its table borrows the row instead.
        let mut total_scores = vec![0.0f64; n_states];
        let mut success_scores = vec![0.0f64; n_states];
        for s in 0..n_positions {
            let row = s * n_nodes;
            for (v, &label) in labels.iter().enumerate() {
                let m = label as usize;
                let from = m * n_states;
                let observation = row + v;
                let total = tables.total_scores(
                    block,
                    observation,
                    totals[observation],
                    from,
                    &mut total_scores,
                );
                let success = tables.success_scores(
                    block,
                    observation,
                    successes[observation],
                    from,
                    &mut success_scores,
                );
                let into = &mut density[(m * n_positions + s) * n_states..][..n_states];
                for k in 0..n_states {
                    into[k] += total[k] + success[k];
                }
            }
        }
        return;
    }
    for s in 0..n_positions {
        let row = s * n_nodes;
        for v in 0..n_nodes {
            let m = labels[v] as usize;
            let total = &tables.total[row_index(totals[row + v]) * block + m * n_states..];
            let success = &tables.success[row_index(successes[row + v]) * block + m * n_states..];
            let into = &mut density[(m * n_positions + s) * n_states..][..n_states];
            for k in 0..n_states {
                into[k] += total[k] + success[k];
            }
        }
    }
}

/// Scaled forward--backward on one chain, writing the posterior and the pairwise.
///
/// Returns the chain's log evidence. `density` is `(T, K)` row-major.
fn forward_backward(
    density: &[f64],
    log_initial: &[f64],
    log_transition: &[f64],
    n_states: usize,
    posterior: &mut [f64],
    pairwise: &mut [f64],
) -> f64 {
    let length = density.len() / n_states;
    // The transition and initial probabilities, exponentiated once rather
    // than once per position: the chain is `S` long and the matrix is not.
    let transition: Vec<f64> = log_transition.iter().map(|value| value.exp()).collect();
    // Each position's emission weights with its row maximum divided out. The
    // density is a sum over a class's members, so at hundreds of members it
    // runs to thousands in magnitude and `exp` of it underflows; the maximum
    // is added back into the evidence, where it belongs.
    let mut weight = vec![0.0f64; density.len()];
    let mut evidence = 0.0f64;
    for t in 0..length {
        let row = &density[t * n_states..][..n_states];
        let largest = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        evidence += largest;
        for (k, value) in weight[t * n_states..][..n_states].iter_mut().enumerate() {
            *value = (row[k] - largest).exp();
        }
    }

    let mut alpha = vec![0.0f64; density.len()];
    let mut scale = vec![0.0f64; length];
    for (k, value) in alpha[..n_states].iter_mut().enumerate() {
        *value = log_initial[k].exp() * weight[k];
    }
    let first: f64 = alpha[..n_states].iter().sum();
    scale[0] = first;
    evidence += first.ln();
    for value in alpha[..n_states].iter_mut() {
        *value /= first;
    }
    for t in 1..length {
        let (previous, current) = alpha.split_at_mut(t * n_states);
        let previous = &previous[(t - 1) * n_states..];
        for (j, cell) in current[..n_states].iter_mut().enumerate() {
            let mut total = 0.0;
            for (i, before) in previous.iter().enumerate() {
                total += before * transition[i * n_states + j];
            }
            *cell = total * weight[t * n_states + j];
        }
        let sum: f64 = current[..n_states].iter().sum();
        scale[t] = sum;
        evidence += sum.ln();
        for value in current[..n_states].iter_mut() {
            *value /= sum;
        }
    }

    let mut beta = vec![1.0f64; density.len()];
    for t in (0..length.saturating_sub(1)).rev() {
        for i in 0..n_states {
            let mut total = 0.0;
            for j in 0..n_states {
                total += transition[i * n_states + j]
                    * weight[(t + 1) * n_states + j]
                    * beta[(t + 1) * n_states + j];
            }
            beta[t * n_states + i] = total / scale[t + 1];
        }
    }

    for (index, cell) in posterior.iter_mut().enumerate() {
        *cell = alpha[index] * beta[index];
    }
    for t in 1..length {
        for i in 0..n_states {
            for j in 0..n_states {
                pairwise[((t - 1) * n_states + i) * n_states + j] = alpha[(t - 1) * n_states + i]
                    * transition[i * n_states + j]
                    * weight[t * n_states + j]
                    * beta[t * n_states + j]
                    / scale[t];
            }
        }
    }
    evidence
}

/// The coupled E step: every class's state posterior, pairwise and evidence.
///
/// # Parameters
/// - `shape`: `S`, `V`, `M`, `K`.
/// - `tables`: the row-indexed emission log-densities.
/// - `totals`, `successes`: `S * V` table rows, position-major --- the counts
///   themselves where the caller tabulated by count.
/// - `labels`: one class per vertex, entries in `[0, M)`.
/// - `log_initial`: `M * K`.
/// - `log_transition`: `M * K * K`, rows the source state.
/// - `posterior`: written, `M * S * K`.
/// - `pairwise`: written, `M * (S - 1) * K * K`.
/// - `log_evidence`: written, `M`.
///
/// # Returns
/// `Ok(())`, or `Err` naming the first violated precondition.
#[allow(clippy::too_many_arguments)]
pub fn class_posteriors_into(
    shape: CoupledShape,
    tables: &EmissionTables<'_>,
    totals: &[u32],
    successes: &[u32],
    labels: &[i64],
    log_initial: &[f64],
    log_transition: &[f64],
    posterior: &mut [f64],
    pairwise: &mut [f64],
    log_evidence: &mut [f64],
) -> Result<(), String> {
    check_inputs(&shape, tables, totals, successes, labels)?;
    let (n_positions, n_classes, n_states) = (shape.n_positions, shape.n_classes, shape.n_states);
    let expected = n_classes * n_positions * n_states;
    if posterior.len() != expected {
        return Err(format!(
            "posterior has {} entries, expected M * S * K = {expected}",
            posterior.len()
        ));
    }
    let pairs = n_classes * n_positions.saturating_sub(1) * n_states * n_states;
    if pairwise.len() != pairs {
        return Err(format!(
            "pairwise has {} entries, expected M * (S - 1) * K * K = {pairs}",
            pairwise.len()
        ));
    }
    if log_evidence.len() != n_classes {
        return Err(format!(
            "log_evidence has {} entries, expected M = {n_classes}",
            log_evidence.len()
        ));
    }

    let mut density = vec![0.0f64; expected];
    class_log_density(&shape, tables, totals, successes, labels, &mut density);
    let per_class = n_positions * n_states;
    let per_class_pairs = n_positions.saturating_sub(1) * n_states * n_states;
    // **Serial, and measured to be right.** `m` is a clean axis --- each class
    // writes its own slice of `posterior`, `pairwise` and `log_evidence`, and
    // nothing is summed across them --- so a `rayon` port was written and
    // benchmarked. It ran **26.703 ms against this loop's 23.064 ms, 0.86x**:
    // ten items against four cores, and `forward_backward` allocates per
    // class, so the pool costs more than it saves. Declined by measurement
    // (issue #627), and recorded in `STATUS.md` so it is not proposed again.
    for m in 0..n_classes {
        log_evidence[m] = forward_backward(
            &density[m * per_class..][..per_class],
            &log_initial[m * n_states..][..n_states],
            &log_transition[m * n_states * n_states..][..n_states * n_states],
            n_states,
            &mut posterior[m * per_class..][..per_class],
            &mut pairwise[m * per_class_pairs..][..per_class_pairs],
        );
    }
    Ok(())
}

/// The external field `H[v, m]`: minus the posterior-expected emission score.
///
/// # Parameters
/// - `weights`: the state posterior transposed to `(S, M, K)` row-major, so
///   one position's `M x K` weights are contiguous beside the tables' rows.
/// - `field`: written, `V * M`.
///
/// # Returns
/// `Ok(())`, or `Err` naming the first violated precondition.
pub fn external_field_into(
    shape: CoupledShape,
    tables: &EmissionTables<'_>,
    totals: &[u32],
    successes: &[u32],
    weights: &[f64],
    field: &mut [f64],
) -> Result<(), String> {
    let block = shape.block();
    if block > 0 {
        tables.validate(&shape, totals, successes)?;
    }
    let (n_positions, n_nodes, n_states) = (shape.n_positions, shape.n_nodes, shape.n_states);
    if totals.len() != n_positions * n_nodes || successes.len() != totals.len() {
        return Err(format!(
            "the counts have {} and {} entries, expected S * V = {}",
            totals.len(),
            successes.len(),
            n_positions * n_nodes
        ));
    }
    if weights.len() != n_positions * block {
        return Err(format!(
            "weights has {} entries, expected S * M * K = {}",
            weights.len(),
            n_positions * block
        ));
    }
    if field.len() != n_nodes * shape.n_classes {
        return Err(format!(
            "field has {} entries, expected V * M = {}",
            field.len(),
            n_nodes * shape.n_classes
        ));
    }

    // **The axis is `v`, not `s`.** Read by cost alone the outer loop looks
    // like the one to split -- 5,041 positions against 200 vertices -- but it
    // is the loop that *accumulates*: every `s` adds into `field[v]`, so
    // splitting it reassociates a sum the oracles pin. Inverting the loops
    // gives each `v` its own `field[v * M..][..M]` to write and leaves the sum
    // over `s` sequential inside it, in the same order and so to the same
    // bits. 200 items against 4 cores is ample; the reassociated version
    // would have been faster and wrong (issue #627).
    if tables.is_factored() {
        // The same walk and the same order of summation as below, with a
        // factored channel's `M x K` scores formed per observation rather than
        // read from a table row.
        field
            .par_chunks_mut(shape.n_classes)
            .enumerate()
            .for_each(|(v, into)| {
                let mut total_scores = vec![0.0f64; block];
                let mut success_scores = vec![0.0f64; block];
                into.fill(0.0);
                for s in 0..n_positions {
                    let observation = s * n_nodes + v;
                    let weight = &weights[s * block..][..block];
                    let total = tables.total_scores(
                        block,
                        observation,
                        totals[observation],
                        0,
                        &mut total_scores,
                    );
                    let success = tables.success_scores(
                        block,
                        observation,
                        successes[observation],
                        0,
                        &mut success_scores,
                    );
                    for (m, cell) in into.iter_mut().enumerate() {
                        let mut accumulated = 0.0;
                        for k in 0..n_states {
                            let index = m * n_states + k;
                            accumulated += (total[index] + success[index]) * weight[index];
                        }
                        *cell -= accumulated;
                    }
                }
            });
        return Ok(());
    }
    field
        .par_chunks_mut(shape.n_classes)
        .enumerate()
        .for_each(|(v, into)| {
            into.fill(0.0);
            for s in 0..n_positions {
                let row = s * n_nodes;
                let weight = &weights[s * block..][..block];
                let total = &tables.total[row_index(totals[row + v]) * block..][..block];
                let success = &tables.success[row_index(successes[row + v]) * block..][..block];
                for (m, cell) in into.iter_mut().enumerate() {
                    let mut accumulated = 0.0;
                    for k in 0..n_states {
                        let index = m * n_states + k;
                        accumulated += (total[index] + success[index]) * weight[index];
                    }
                    *cell -= accumulated;
                }
            }
        });
    Ok(())
}

/// The preconditions both kernels share.
fn check_inputs(
    shape: &CoupledShape,
    tables: &EmissionTables<'_>,
    totals: &[u32],
    successes: &[u32],
    labels: &[i64],
) -> Result<(), String> {
    if shape.n_positions == 0 || shape.n_nodes == 0 {
        return Err("a coupled instance has at least one position and one vertex".to_string());
    }
    if totals.len() != shape.n_positions * shape.n_nodes || successes.len() != totals.len() {
        return Err(format!(
            "the counts have {} and {} entries, expected S * V = {}",
            totals.len(),
            successes.len(),
            shape.n_positions * shape.n_nodes
        ));
    }
    if labels.len() != shape.n_nodes {
        return Err(format!(
            "labels has {} entries, expected V = {}",
            labels.len(),
            shape.n_nodes
        ));
    }
    if labels
        .iter()
        .any(|&label| label < 0 || label as usize >= shape.n_classes)
    {
        return Err(format!("every label must lie in [0, {})", shape.n_classes));
    }
    tables.validate(shape, totals, successes)
}

/// Read a borrowed 1-D array as a contiguous slice, or say why it is not one.
pub(crate) fn borrowed<'a, T: numpy::Element>(
    array: &'a PyReadonlyArray1<'_, T>,
    name: &str,
) -> PyResult<&'a [T]> {
    array.as_slice().map_err(|_| {
        PyValueError::new_err(format!("{name} must be C-contiguous to cross the boundary"))
    })
}

/// The exposure term from its three optional arrays: all three, or none.
pub(crate) fn exposure_term<'a>(
    exposure: &'a Option<PyReadonlyArray1<'_, f64>>,
    dispersion: &'a Option<PyReadonlyArray1<'_, f64>>,
    mean: &'a Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<Option<ExposureTerm<'a>>> {
    match (exposure, dispersion, mean) {
        (None, None, None) => Ok(None),
        (Some(exposure), Some(dispersion), Some(mean)) => Ok(Some(ExposureTerm {
            exposure: borrowed(exposure, "exposure")?,
            dispersion: borrowed(dispersion, "dispersion")?,
            mean: borrowed(mean, "mean")?,
        })),
        _ => Err(PyValueError::new_err(
            "an exposure term takes exposure, dispersion and mean together",
        )),
    }
}

/// The trial term from its five optional arrays: all five, or none.
pub(crate) fn trial_term<'a>(
    trials: &'a Option<PyReadonlyArray1<'_, u32>>,
    failure_table: &'a Option<PyReadonlyArray1<'_, f64>>,
    trial_table: &'a Option<PyReadonlyArray1<'_, f64>>,
    log_factorial: &'a Option<PyReadonlyArray1<'_, f64>>,
    log_beta: &'a Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<Option<TrialTerm<'a>>> {
    match (trials, failure_table, trial_table, log_factorial, log_beta) {
        (None, None, None, None, None) => Ok(None),
        (Some(trials), Some(failure), Some(trial), Some(log_factorial), Some(log_beta)) => {
            Ok(Some(TrialTerm {
                trials: borrowed(trials, "trials")?,
                failure: borrowed(failure, "failure_table")?,
                trial: borrowed(trial, "trial_table")?,
                log_factorial: borrowed(log_factorial, "log_factorial")?,
                log_beta: borrowed(log_beta, "log_beta")?,
            }))
        }
        _ => Err(PyValueError::new_err(
            "a trial term takes trials, failure_table, trial_table, log_factorial and log_beta together",
        )),
    }
}

/// `class_posteriors_into` as a Python binding.
///
/// Every array crosses the boundary once, contiguous and borrowed rather than
/// copied; the three results are written into the caller's own arrays for the
/// reason `src/sampling.rs` gives, that a second pass over `M x S x K` is the
/// last copy left at this size.
///
/// # Returns
/// `None`; the results are written into `posterior`, `pairwise` and
/// `log_evidence`.
///
/// `exposure`, `dispersion` and `mean`, given together, are the first
/// channel's [`ExposureTerm`]; `total_table` is then its table `B`.
/// `trials`, `failure_table`, `trial_table`, `log_factorial` and `log_beta`,
/// given together, are the second channel's [`TrialTerm`]; `success_table` is
/// then its table `U`.
///
/// # Errors
/// `ValueError` naming the first violated precondition, including a count
/// past the extent of the table that is indexed by it.
#[pyfunction]
#[pyo3(signature = (totals, successes, labels, total_table, success_table, log_initial, log_transition, n_positions, n_nodes, n_classes, n_states, posterior, pairwise, log_evidence, exposure=None, dispersion=None, mean=None, trials=None, failure_table=None, trial_table=None, log_factorial=None, log_beta=None))]
#[allow(clippy::too_many_arguments)]
pub fn class_posteriors(
    py: Python<'_>,
    totals: PyReadonlyArray1<'_, u32>,
    successes: PyReadonlyArray1<'_, u32>,
    labels: PyReadonlyArray1<'_, i64>,
    total_table: PyReadonlyArray1<'_, f64>,
    success_table: PyReadonlyArray1<'_, f64>,
    log_initial: PyReadonlyArray1<'_, f64>,
    log_transition: PyReadonlyArray1<'_, f64>,
    n_positions: usize,
    n_nodes: usize,
    n_classes: usize,
    n_states: usize,
    mut posterior: PyReadwriteArray1<'_, f64>,
    mut pairwise: PyReadwriteArray1<'_, f64>,
    mut log_evidence: PyReadwriteArray1<'_, f64>,
    exposure: Option<PyReadonlyArray1<'_, f64>>,
    dispersion: Option<PyReadonlyArray1<'_, f64>>,
    mean: Option<PyReadonlyArray1<'_, f64>>,
    trials: Option<PyReadonlyArray1<'_, u32>>,
    failure_table: Option<PyReadonlyArray1<'_, f64>>,
    trial_table: Option<PyReadonlyArray1<'_, f64>>,
    log_factorial: Option<PyReadonlyArray1<'_, f64>>,
    log_beta: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<()> {
    let tables = EmissionTables {
        total: borrowed(&total_table, "total_table")?,
        success: borrowed(&success_table, "success_table")?,
        exposure: exposure_term(&exposure, &dispersion, &mean)?,
        trials: trial_term(
            &trials,
            &failure_table,
            &trial_table,
            &log_factorial,
            &log_beta,
        )?,
    };
    let shape = CoupledShape {
        n_positions,
        n_nodes,
        n_classes,
        n_states,
    };
    let totals = borrowed(&totals, "totals")?;
    let successes = borrowed(&successes, "successes")?;
    let labels = borrowed(&labels, "labels")?;
    let log_initial = borrowed(&log_initial, "log_initial")?;
    let log_transition = borrowed(&log_transition, "log_transition")?;
    let posterior = posterior
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("posterior must be C-contiguous"))?;
    let pairwise = pairwise
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("pairwise must be C-contiguous"))?;
    let log_evidence = log_evidence
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("log_evidence must be C-contiguous"))?;
    py.detach(|| {
        class_posteriors_into(
            shape,
            &tables,
            totals,
            successes,
            labels,
            log_initial,
            log_transition,
            posterior,
            pairwise,
            log_evidence,
        )
    })
    .map_err(PyValueError::new_err)
}

/// `external_field_into` as a Python binding.
///
/// The exposure and the trial term are as [`class_posteriors`] takes them.
///
/// # Returns
/// `None`; the result is written into `field`.
///
/// # Errors
/// `ValueError` naming the first violated precondition.
#[pyfunction]
#[pyo3(signature = (totals, successes, total_table, success_table, weights, n_positions, n_nodes, n_classes, n_states, field, exposure=None, dispersion=None, mean=None, trials=None, failure_table=None, trial_table=None, log_factorial=None, log_beta=None))]
#[allow(clippy::too_many_arguments)]
pub fn external_field(
    py: Python<'_>,
    totals: PyReadonlyArray1<'_, u32>,
    successes: PyReadonlyArray1<'_, u32>,
    total_table: PyReadonlyArray1<'_, f64>,
    success_table: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    n_positions: usize,
    n_nodes: usize,
    n_classes: usize,
    n_states: usize,
    mut field: PyReadwriteArray1<'_, f64>,
    exposure: Option<PyReadonlyArray1<'_, f64>>,
    dispersion: Option<PyReadonlyArray1<'_, f64>>,
    mean: Option<PyReadonlyArray1<'_, f64>>,
    trials: Option<PyReadonlyArray1<'_, u32>>,
    failure_table: Option<PyReadonlyArray1<'_, f64>>,
    trial_table: Option<PyReadonlyArray1<'_, f64>>,
    log_factorial: Option<PyReadonlyArray1<'_, f64>>,
    log_beta: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<()> {
    let tables = EmissionTables {
        total: borrowed(&total_table, "total_table")?,
        success: borrowed(&success_table, "success_table")?,
        exposure: exposure_term(&exposure, &dispersion, &mean)?,
        trials: trial_term(
            &trials,
            &failure_table,
            &trial_table,
            &log_factorial,
            &log_beta,
        )?,
    };
    let shape = CoupledShape {
        n_positions,
        n_nodes,
        n_classes,
        n_states,
    };
    let totals = borrowed(&totals, "totals")?;
    let successes = borrowed(&successes, "successes")?;
    let weights = borrowed(&weights, "weights")?;
    let field = field
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("field must be C-contiguous"))?;
    py.detach(|| external_field_into(shape, &tables, totals, successes, weights, field))
        .map_err(PyValueError::new_err)
}

/// Distinct values in first-appearance order: each value's code, and each code's value.
///
/// One pass and no sort: a table built over the codes needs every distinct
/// value once and in no particular order, and the rows it is indexed by are
/// the same numbers whatever order the codes take (issue #1064). Integers
/// whose range is no longer than the input are looked up in an array over
/// `[min, max]`; anything else is hashed on its bits. `-0.0` is coded as
/// `0.0`, so equal values share a code.
///
/// # Errors
/// `Err` if a value is NaN, which equals nothing and so has no code, or if
/// there are more values than a `u32` code carries.
pub fn factorize_into(values: &[f64]) -> Result<(Vec<u32>, Vec<f64>), String> {
    if values.len() > u32::MAX as usize {
        return Err(format!(
            "{} values, past the {} a u32 code carries",
            values.len(),
            u32::MAX
        ));
    }
    if values.iter().any(|value| value.is_nan()) {
        return Err("a NaN equals nothing and has no code".to_string());
    }
    let mut codes = Vec::with_capacity(values.len());
    let mut levels = Vec::new();
    let (low, high) = values
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), &value| {
            (low.min(value), high.max(value))
        });
    let integral = values.iter().all(|value| *value == value.trunc());
    if integral && high - low < values.len() as f64 {
        let mut seen = vec![u32::MAX; (high - low) as usize + 1];
        for &value in values {
            let slot = &mut seen[(value - low) as usize];
            if *slot == u32::MAX {
                *slot = levels.len() as u32;
                levels.push(value + 0.0);
            }
            codes.push(*slot);
        }
        return Ok((codes, levels));
    }
    let mut seen = std::collections::HashMap::<u64, u32>::new();
    for &value in values {
        // `-0.0 + 0.0` is `0.0` and every other value is unchanged.
        let value = value + 0.0;
        let code = *seen.entry(value.to_bits()).or_insert_with(|| {
            levels.push(value);
            (levels.len() - 1) as u32
        });
        codes.push(code);
    }
    Ok((codes, levels))
}

/// [`factorize_into`] as a Python binding: `(codes, levels)`.
///
/// # Errors
/// `ValueError` as [`factorize_into`] refuses.
#[pyfunction]
#[pyo3(signature = (values))]
#[allow(clippy::type_complexity)]
pub fn factorize<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<(
    Bound<'py, numpy::PyArray1<u32>>,
    Bound<'py, numpy::PyArray1<f64>>,
)> {
    let values = borrowed(&values, "values")?;
    let (codes, levels) = py
        .detach(|| factorize_into(values))
        .map_err(PyValueError::new_err)?;
    Ok((
        numpy::PyArray1::from_vec(py, codes),
        numpy::PyArray1::from_vec(py, levels),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A two-position, two-vertex, one-class, two-state instance whose
    /// forward--backward can be written out by hand.
    fn tiny() -> (CoupledShape, Vec<f64>, Vec<f64>) {
        let shape = CoupledShape {
            n_positions: 2,
            n_nodes: 2,
            n_classes: 1,
            n_states: 2,
        };
        // Counts 0 and 1; one class, two states.
        let total = vec![-1.0, -2.0, -3.0, -0.5];
        let success = vec![0.0, 0.0, 0.0, 0.0];
        (shape, total, success)
    }

    #[test]
    fn posteriors_sum_to_one_at_every_position() {
        let (shape, total, success) = tiny();
        let tables = EmissionTables {
            total: &total,
            success: &success,
            exposure: None,
            trials: None,
        };
        let totals = [0u32, 1, 1, 0];
        let successes = [0u32, 0, 0, 0];
        let labels = [0i64, 0];
        let log_initial = [(0.5f64).ln(), (0.5f64).ln()];
        let log_transition = [(0.7f64).ln(), (0.3f64).ln(), (0.4f64).ln(), (0.6f64).ln()];
        let mut posterior = vec![0.0; 4];
        let mut pairwise = vec![0.0; 4];
        let mut evidence = vec![0.0; 1];

        class_posteriors_into(
            shape,
            &tables,
            &totals,
            &successes,
            &labels,
            &log_initial,
            &log_transition,
            &mut posterior,
            &mut pairwise,
            &mut evidence,
        )
        .unwrap();

        assert!((posterior[0] + posterior[1] - 1.0).abs() < 1e-12);
        assert!((posterior[2] + posterior[3] - 1.0).abs() < 1e-12);
        assert!((pairwise.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(evidence[0] < 0.0);
    }

    #[test]
    fn a_count_past_the_table_is_refused() {
        let (shape, total, success) = tiny();
        let tables = EmissionTables {
            total: &total,
            success: &success,
            exposure: None,
            trials: None,
        };
        let totals = [0u32, 2, 1, 0];
        let successes = [0u32, 0, 0, 0];
        let labels = [0i64, 0];
        let mut posterior = vec![0.0; 4];
        let mut pairwise = vec![0.0; 4];
        let mut evidence = vec![0.0; 1];

        let refused = class_posteriors_into(
            shape,
            &tables,
            &totals,
            &successes,
            &labels,
            &[0.0, 0.0],
            &[0.0, 0.0, 0.0, 0.0],
            &mut posterior,
            &mut pairwise,
            &mut evidence,
        );

        assert!(refused.unwrap_err().contains("past the table's extent"));
    }

    /// `A[y] + r ln(r / t) + y ln(mu c / t)` written out for one class and
    /// two states, against the kernel's accumulation of it.
    #[test]
    fn an_exposure_term_is_added_to_the_count_table_and_zero_exposure_scores_zero() {
        let (shape, total, success) = tiny();
        let dispersion = [2.0, 5.0];
        let mean = [3.0, 7.0];
        let exposure = [0.5, 2.0, 0.0, 1.5];
        let tables = EmissionTables {
            total: &total,
            success: &success,
            exposure: Some(ExposureTerm {
                exposure: &exposure,
                dispersion: &dispersion,
                mean: &mean,
            }),
            trials: None,
        };
        let totals = [0u32, 1, 1, 0];
        let successes = [0u32, 0, 0, 0];
        let weights = [1.0, 0.0, 1.0, 0.0];
        let mut field = vec![0.0; 2];

        external_field_into(shape, &tables, &totals, &successes, &weights, &mut field).unwrap();

        let score = |b: f64, y: f64, c: f64| {
            let r = dispersion[0];
            b + y * c.ln() - (y + r) * (r + mean[0] * c).ln()
        };
        // Vertex 0: count 0 at exposure 0.5, then count 1 at exposure 0,
        // which is unobserved and scores zero.
        assert!((field[0] + score(total[0], 0.0, 0.5)).abs() < 1e-12);
        // Vertex 1: count 1 at exposure 2.0, then count 0 at exposure 1.5.
        let expected = score(total[2], 1.0, 2.0) + score(total[0], 0.0, 1.5);
        assert!((field[1] + expected).abs() < 1e-12);
    }

    #[test]
    fn a_negative_exposure_is_refused() {
        let (shape, total, success) = tiny();
        let exposure = [0.5, -1.0, 1.0, 1.0];
        let tables = EmissionTables {
            total: &total,
            success: &success,
            exposure: Some(ExposureTerm {
                exposure: &exposure,
                dispersion: &[1.0, 1.0],
                mean: &[1.0, 1.0],
            }),
            trials: None,
        };
        let mut field = vec![0.0; 2];

        let refused = external_field_into(
            shape,
            &tables,
            &[0u32, 1, 1, 0],
            &[0u32, 0, 0, 0],
            &[1.0, 0.0, 1.0, 0.0],
            &mut field,
        );

        assert!(refused.unwrap_err().contains("non-negative"));
    }

    #[test]
    fn the_field_is_minus_the_expected_score() {
        let (shape, total, success) = tiny();
        let tables = EmissionTables {
            total: &total,
            success: &success,
            exposure: None,
            trials: None,
        };
        let totals = [0u32, 1, 1, 0];
        let successes = [0u32, 0, 0, 0];
        // One class, two states, two positions: put all the weight on state 0.
        let weights = [1.0, 0.0, 1.0, 0.0];
        let mut field = vec![0.0; 2];

        external_field_into(shape, &tables, &totals, &successes, &weights, &mut field).unwrap();

        // Vertex 0 sees counts 0 then 1: -(t[0][0] + t[1][0]) = -(-1 + -3).
        assert!((field[0] - 4.0).abs() < 1e-12);
        // Vertex 1 sees counts 1 then 0.
        assert!((field[1] - 4.0).abs() < 1e-12);
    }

    /// A one-class, two-state trial term over trial counts up to 3, its
    /// tables filled with distinct values so a wrong index shows.
    #[allow(clippy::type_complexity)]
    fn trial_tables() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let success: Vec<f64> = (0..8).map(|i| 0.5 + f64::from(i)).collect();
        let failure: Vec<f64> = (0..8).map(|i| 0.25 * f64::from(i)).collect();
        let trial: Vec<f64> = (0..8).map(|i| 3.0 + 0.125 * f64::from(i)).collect();
        let log_factorial = vec![0.0, 0.0, 2f64.ln(), 6f64.ln()];
        let log_beta = vec![0.7, 0.9, 0.2, 0.3, 0.4, 0.1];
        (success, failure, trial, log_factorial, log_beta)
    }

    /// The trial term's nine terms written out, against the field's
    /// accumulation of them; zero trials score zero and successes past their
    /// trials score `-inf`.
    #[test]
    fn a_trial_term_is_the_nine_terms_in_order() {
        let shape = CoupledShape {
            n_positions: 2,
            n_nodes: 2,
            n_classes: 1,
            n_states: 2,
        };
        let total = vec![0.0; 2];
        let (success, failure, trial, log_factorial, log_beta) = trial_tables();
        let trials = [3u32, 0, 2, 1];
        let tables = EmissionTables {
            total: &total,
            success: &success,
            exposure: None,
            trials: Some(TrialTerm {
                trials: &trials,
                failure: &failure,
                trial: &trial,
                log_factorial: &log_factorial,
                log_beta: &log_beta,
            }),
        };
        let successes = [1u32, 3, 2, 2];
        let weights = [1.0, 0.0, 1.0, 0.0];
        let mut field = vec![0.0; 2];

        external_field_into(shape, &tables, &[0u32; 4], &successes, &weights, &mut field).unwrap();

        let score = |z: usize, n: usize| {
            let free = (log_factorial[n] - log_factorial[z]) - log_factorial[n - z];
            (((((free + success[2 * z]) + failure[2 * (n - z)]) - trial[2 * n]) + log_beta[0])
                - log_beta[2])
                - log_beta[4]
        };
        // Vertex 0: one success in three trials, then two in two.
        assert_eq!(field[0], -(score(1, 3) + score(2, 2)));
        // Vertex 1: three successes in no trials, then two in one. Scored
        // directly, since the field weights state 1's `-inf` by zero.
        let term = tables.trials.unwrap();
        let mut scores = [f64::NAN; 2];
        term.score_into(&success[6..], 3, 0, 2, 0, &mut scores);
        assert_eq!(scores, [0.0, 0.0]);
        term.score_into(&success[4..], 2, 1, 2, 0, &mut scores);
        assert_eq!(scores, [f64::NEG_INFINITY; 2]);
    }

    #[test]
    fn a_trial_count_past_the_tables_is_refused() {
        let (shape, total, _) = tiny();
        let (success, failure, trial, log_factorial, log_beta) = trial_tables();
        let trials = [4u32, 0, 0, 0];
        let tables = EmissionTables {
            total: &total,
            success: &success,
            exposure: None,
            trials: Some(TrialTerm {
                trials: &trials,
                failure: &failure,
                trial: &trial,
                log_factorial: &log_factorial,
                log_beta: &log_beta[..4],
            }),
        };
        let mut field = vec![0.0; 2];
        let refused = external_field_into(
            shape,
            &tables,
            &[0u32; 4],
            &[0u32; 4],
            &[1.0, 0.0, 1.0, 0.0],
            &mut field,
        );
        assert!(refused.unwrap_err().contains("3 * M * K"));

        let tables = EmissionTables {
            trials: Some(TrialTerm {
                log_beta: &log_beta,
                ..tables.trials.unwrap()
            }),
            ..tables
        };
        let refused = external_field_into(
            shape,
            &tables,
            &[0u32; 4],
            &[0u32; 4],
            &[1.0, 0.0, 1.0, 0.0],
            &mut field,
        );
        assert!(refused
            .unwrap_err()
            .contains("a trial count of 4 is past the failure table"));
    }

    #[test]
    fn factorize_codes_in_first_appearance_order() {
        // Integers, looked up over their range.
        let (codes, levels) = factorize_into(&[7.0, 3.0, 7.0, 5.0, 3.0]).unwrap();
        assert_eq!(codes, vec![0, 1, 0, 2, 1]);
        assert_eq!(levels, vec![7.0, 3.0, 5.0]);
        // Floats, and integers too sparse to look up, hashed on their bits;
        // `-0.0` shares `0.0`'s code.
        let (codes, levels) = factorize_into(&[0.5, -0.0, 1e9, 0.0, 0.5]).unwrap();
        assert_eq!(codes, vec![0, 1, 2, 1, 0]);
        assert_eq!(levels, vec![0.5, 0.0, 1e9]);
        assert!(levels[1].is_sign_positive());
        let (codes, _) = factorize_into(&[1e9, -0.0, 0.0]).unwrap();
        assert_eq!(codes, vec![0, 1, 1]);
        assert!(factorize_into(&[1.0, f64::NAN])
            .unwrap_err()
            .contains("NaN"));
    }
}
