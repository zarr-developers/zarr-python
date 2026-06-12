use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use zarrs::array::{Array, ArrayBytes, ArraySubset};
use zarrs::metadata::ArrayMetadata;
use zarrs::storage::ReadableWritableListableStorage;

use crate::store::resolve_store;
use crate::{runtime_err, value_err};

type DynArray = Array<dyn zarrs::storage::ReadableWritableListableStorageTraits>;

/// Construct an Array view from an explicit metadata document, without
/// consulting the store for metadata.
fn array_view(
    storage: ReadableWritableListableStorage,
    path: &str,
    metadata_json: &str,
) -> PyResult<DynArray> {
    let metadata = ArrayMetadata::try_from(metadata_json).map_err(value_err)?;
    Array::new_with_metadata(storage, path, metadata).map_err(value_err)
}

#[pyfunction]
pub(crate) fn retrieve_chunk(
    py: Python<'_>,
    store: &Bound<'_, PyAny>,
    path: String,
    metadata_json: String,
    chunk_coords: Vec<u64>,
) -> PyResult<Py<PyBytes>> {
    let storage = resolve_store(store)?;
    let data = py.detach(move || -> PyResult<Vec<u8>> {
        let array = array_view(storage, &path, &metadata_json)?;
        let bytes: ArrayBytes<'static> =
            array.retrieve_chunk(&chunk_coords).map_err(runtime_err)?;
        let fixed = bytes.into_fixed().map_err(|_| {
            PyNotImplementedError::new_err("variable-length data types are not supported")
        })?;
        Ok(fixed.into_owned())
    })?;
    Ok(PyBytes::new(py, &data).unbind())
}

#[pyfunction]
pub(crate) fn retrieve_encoded_chunk(
    py: Python<'_>,
    store: &Bound<'_, PyAny>,
    path: String,
    metadata_json: String,
    chunk_coords: Vec<u64>,
) -> PyResult<Option<Py<PyBytes>>> {
    let storage = resolve_store(store)?;
    let data = py.detach(move || -> PyResult<Option<Vec<u8>>> {
        let array = array_view(storage, &path, &metadata_json)?;
        array
            .retrieve_encoded_chunk(&chunk_coords)
            .map_err(runtime_err)
    })?;
    Ok(data.map(|d| PyBytes::new(py, &d).unbind()))
}

#[pyfunction]
pub(crate) fn store_chunk(
    py: Python<'_>,
    store: &Bound<'_, PyAny>,
    path: String,
    metadata_json: String,
    chunk_coords: Vec<u64>,
    data: Vec<u8>,
) -> PyResult<()> {
    let storage = resolve_store(store)?;
    py.detach(move || {
        let array = array_view(storage, &path, &metadata_json)?;
        array
            .store_chunk(&chunk_coords, ArrayBytes::new_flen(data))
            .map_err(runtime_err)
    })
}

#[pyfunction]
pub(crate) fn erase_chunk(
    py: Python<'_>,
    store: &Bound<'_, PyAny>,
    path: String,
    metadata_json: String,
    chunk_coords: Vec<u64>,
) -> PyResult<()> {
    let storage = resolve_store(store)?;
    py.detach(move || {
        let array = array_view(storage, &path, &metadata_json)?;
        array.erase_chunk(&chunk_coords).map_err(runtime_err)
    })
}

#[pyfunction]
pub(crate) fn retrieve_array_subset(
    py: Python<'_>,
    store: &Bound<'_, PyAny>,
    path: String,
    metadata_json: String,
    start: Vec<u64>,
    shape: Vec<u64>,
) -> PyResult<Py<PyBytes>> {
    let storage = resolve_store(store)?;
    let data = py.detach(move || -> PyResult<Vec<u8>> {
        let array = array_view(storage, &path, &metadata_json)?;
        let subset = ArraySubset::new_with_start_shape(start, shape).map_err(value_err)?;
        let bytes: ArrayBytes<'static> =
            array.retrieve_array_subset(&subset).map_err(runtime_err)?;
        let fixed = bytes.into_fixed().map_err(|_| {
            PyNotImplementedError::new_err("variable-length data types are not supported")
        })?;
        Ok(fixed.into_owned())
    })?;
    Ok(PyBytes::new(py, &data).unbind())
}
