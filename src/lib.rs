use pyo3::prelude::*;

pub mod maxflow;
pub mod pruning;
pub mod sampling;

pub use maxflow::{ising_ground_state, max_flow};
pub use pruning::pruning_log_likelihood;
pub use sampling::sample_rows;

/// Doubles an integer.
///
/// Placeholder binding: it demonstrates the Rust-to-Python pattern real
/// numerical kernels follow (`pruning::pruning_log_likelihood` is now the
/// substantive one) and implements no phylogenetics itself. Left in place
/// because `snakes_and_ladders.__init__` re-exports it and `tests/test_oxiphylo_bindings.py`
/// asserts it exists.
#[pyfunction]
pub fn double(x: i64) -> i64 {
    x * 2
}

/// The compiled extension module. `python/snakes_and_ladders/__init__.py` re-exports it as
/// `snakes_and_ladders.oxi_snakes_and_ladders` (see `module-name` in `pyproject.toml`).
#[pymodule]
fn oxi_snakes_and_ladders(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(double, m)?)?;
    m.add_function(wrap_pyfunction!(pruning_log_likelihood, m)?)?;
    m.add_function(wrap_pyfunction!(sample_rows, m)?)?;
    m.add_function(wrap_pyfunction!(max_flow, m)?)?;
    m.add_function(wrap_pyfunction!(ising_ground_state, m)?)?;
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
