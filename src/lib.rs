//! The `oxisal` extension: the kernels root `CLAUDE.md`'s
//! backend rule admits, each beside the NumPy oracle that pins it.
//!
//! **Every kernel releases the GIL.** PyO3 holds it for the whole body of a
//! `#[pyfunction]` unless the body gives it up, and these need none: each
//! extracts `as_slice()` from its arrays and then touches no Python object,
//! so the work runs inside `Python::detach` and `Ungil` makes a
//! Python object in that closure a compile error. Before this, four Python
//! threads calling `single_site_sweeps` took 4.03x the wall of one --
//! serialization exactly (issue #604). It changes no arithmetic and no
//! single-call timing; what it buys is `sal.parallel`'s
//! `backend="threads"`, which `DEV.md` documents and which had no eligible
//! site in the package. [`double`] is the exception: a placeholder integer
//! multiply, where the release would cost more than the body.

use pyo3::prelude::*;

pub mod bcjr;
pub mod bifurcation;
pub mod bk;
pub mod chain;
pub mod count_mstep;
pub mod count_pairs;
pub mod coupled;
pub mod dense_emission;
pub mod energy;
pub mod hmc;
pub mod hmm_decode;
pub mod hmm_stream;
pub mod lattice_cut;
pub mod maxflow;
#[cfg(feature = "sandbox")]
pub mod maxflow_declined;
pub mod message_passing;
pub mod metropolis;
pub mod mixture_stream;
pub mod potts;
pub mod pruning;
#[cfg(feature = "sandbox")]
pub mod pruning_burn;
pub mod ragged;
pub mod sampling;
pub mod special;

pub use bcjr::bcjr_forward_backward;
pub use bifurcation::bifurcation_integrate;
pub use count_mstep::{
    beta_binomial_parameters, negative_binomial_dispersions, negative_binomial_dispersions_exposed,
};
pub use count_pairs::simulate_count_pairs;
pub use coupled::{class_posteriors, external_field, factorize};
pub use dense_emission::dense_log_emission;
pub use hmc::leapfrog_trajectory;
pub use hmm_decode::{hmm_score, hmm_viterbi};
pub use hmm_stream::{
    categorical_em_step, count_cells, count_em_step, gaussian_em_step, gaussian_hmm_statistics,
};
pub use lattice_cut::LatticeCut;
pub use maxflow::{ising_ground_state, ising_ground_states, max_flow};
#[cfg(feature = "sandbox")]
pub use maxflow_declined::{ising_ground_state_declined, max_flow_declined};
pub use message_passing::tree_message_passing;
pub use mixture_stream::{gaussian_mixture_em_step, gaussian_mixture_gradient};
pub use potts::{bond_roots, single_site_sweeps, swendsen_wang_sweep};
pub use pruning::pruning_log_likelihood;
#[cfg(feature = "sandbox")]
pub use pruning_burn::pruning_gradient;
pub use ragged::{ragged_posteriors_into, SwitchKind};
pub use sampling::sample_rows;

/// Doubles an integer.
///
/// Placeholder binding: it demonstrates the Rust-to-Python pattern real
/// numerical kernels follow (`pruning::pruning_log_likelihood` is now the
/// substantive one) and implements no phylogenetics itself. Left in place
/// because `sal.__init__` re-exports it and `tests/test_oxisal_bindings.py`
/// asserts it exists.
#[pyfunction]
pub fn double(x: i64) -> i64 {
    x * 2
}

/// The compiled extension module. `python/sal/__init__.py` re-exports it as
/// `sal.oxisal` (see `module-name` in `pyproject.toml`).
#[pymodule]
fn oxisal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(double, m)?)?;
    m.add_function(wrap_pyfunction!(pruning_log_likelihood, m)?)?;
    #[cfg(feature = "sandbox")]
    m.add_function(wrap_pyfunction!(pruning_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(sample_rows, m)?)?;
    m.add_function(wrap_pyfunction!(ragged::ragged_posteriors, m)?)?;
    m.add_function(wrap_pyfunction!(max_flow, m)?)?;
    m.add_class::<LatticeCut>()?;
    m.add_function(wrap_pyfunction!(ising_ground_state, m)?)?;
    m.add_function(wrap_pyfunction!(ising_ground_states, m)?)?;
    #[cfg(feature = "sandbox")]
    m.add_function(wrap_pyfunction!(max_flow_declined, m)?)?;
    #[cfg(feature = "sandbox")]
    m.add_function(wrap_pyfunction!(ising_ground_state_declined, m)?)?;
    m.add_function(wrap_pyfunction!(single_site_sweeps, m)?)?;
    m.add_function(wrap_pyfunction!(swendsen_wang_sweep, m)?)?;
    m.add_function(wrap_pyfunction!(class_posteriors, m)?)?;
    m.add_function(wrap_pyfunction!(external_field, m)?)?;
    m.add_function(wrap_pyfunction!(dense_log_emission, m)?)?;
    m.add_function(wrap_pyfunction!(factorize, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_count_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(negative_binomial_dispersions, m)?)?;
    m.add_function(wrap_pyfunction!(negative_binomial_dispersions_exposed, m)?)?;
    m.add_function(wrap_pyfunction!(beta_binomial_parameters, m)?)?;
    m.add_function(wrap_pyfunction!(bcjr_forward_backward, m)?)?;
    m.add_function(wrap_pyfunction!(tree_message_passing, m)?)?;
    m.add_function(wrap_pyfunction!(categorical_em_step, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_em_step, m)?)?;
    m.add_function(wrap_pyfunction!(count_cells, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_hmm_statistics, m)?)?;
    m.add_function(wrap_pyfunction!(hmm_viterbi, m)?)?;
    m.add_function(wrap_pyfunction!(bifurcation_integrate, m)?)?;
    m.add_function(wrap_pyfunction!(hmm_score, m)?)?;
    m.add_function(wrap_pyfunction!(count_em_step, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_mixture_em_step, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_mixture_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(leapfrog_trajectory, m)?)?;
    m.add_class::<hmc::HmcWalk>()?;
    m.add_function(wrap_pyfunction!(bond_roots, m)?)?;
    m.add_class::<metropolis::MetropolisWalk>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_double() {
        assert_eq!(double(21), 42);
        assert_eq!(double(0), 0);
        assert_eq!(double(-3), -6);
    }
}
