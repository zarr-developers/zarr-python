use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

mod node;
mod store;

pyo3::create_exception!(
    _zarrs_bindings,
    NodeExistsError,
    PyValueError,
    "A node already exists at the given path."
);
pyo3::create_exception!(
    _zarrs_bindings,
    NodeNotFoundError,
    PyValueError,
    "No node was found at the given path."
);

pub(crate) fn runtime_err(err: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}

pub(crate) fn value_err(err: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(err.to_string())
}

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn _zarrs_bindings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("NodeExistsError", m.py().get_type::<NodeExistsError>())?;
    m.add("NodeNotFoundError", m.py().get_type::<NodeNotFoundError>())?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(node::create_array, m)?)?;
    m.add_function(wrap_pyfunction!(node::create_group, m)?)?;
    m.add_function(wrap_pyfunction!(node::delete_node, m)?)?;
    m.add_function(wrap_pyfunction!(node::list_children, m)?)?;
    m.add_function(wrap_pyfunction!(node::read_metadata, m)?)?;
    Ok(())
}
