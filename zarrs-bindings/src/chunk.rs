use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex, OnceLock};

use lru::LruCache;
use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use zarrs::array::{Array, ArrayBytes, ArraySubset};
use zarrs::metadata::ArrayMetadata;
use zarrs::storage::ReadableWritableListableStorage;

use crate::store::resolve_store_with_key;
use crate::{runtime_err, value_err};

type DynArray = Array<dyn zarrs::storage::ReadableWritableListableStorageTraits>;

/// Cache of constructed Arrays keyed by (filesystem root, node path, metadata
/// JSON). Only native filesystem stores are cached (see `resolve_store_with_key`).
/// Bounded by an LRU; entries hold only a filesystem path + codec chain, no data.
type CacheKey = (String, String, String);
static ARRAY_CACHE: OnceLock<Mutex<LruCache<CacheKey, Arc<DynArray>>>> = OnceLock::new();

fn array_cache() -> &'static Mutex<LruCache<CacheKey, Arc<DynArray>>> {
    ARRAY_CACHE.get_or_init(|| Mutex::new(LruCache::new(NonZeroUsize::new(128).unwrap())))
}

fn build_array(
    storage: ReadableWritableListableStorage,
    path: &str,
    metadata_json: &str,
) -> PyResult<DynArray> {
    let metadata = ArrayMetadata::try_from(metadata_json).map_err(value_err)?;
    Array::new_with_metadata(storage, path, metadata).map_err(value_err)
}

/// Construct (or fetch from cache) an Array view from an explicit metadata
/// document, without consulting the store for metadata. When `cache_key` is
/// `Some(root)` the result is memoized on (root, path, metadata_json).
fn array_view(
    storage: ReadableWritableListableStorage,
    cache_key: Option<String>,
    path: &str,
    metadata_json: &str,
) -> PyResult<Arc<DynArray>> {
    if let Some(root) = cache_key {
        let key = (root, path.to_string(), metadata_json.to_string());
        if let Some(array) = array_cache().lock().unwrap().get(&key).cloned() {
            return Ok(array);
        }
        let array = Arc::new(build_array(storage, path, metadata_json)?);
        array_cache().lock().unwrap().put(key, Arc::clone(&array));
        Ok(array)
    } else {
        Ok(Arc::new(build_array(storage, path, metadata_json)?))
    }
}

#[pyfunction]
pub(crate) fn array_cache_len() -> usize {
    array_cache().lock().unwrap().len()
}

#[pyfunction]
pub(crate) fn clear_array_cache() {
    array_cache().lock().unwrap().clear();
}

#[pyfunction]
pub(crate) fn retrieve_chunk(
    py: Python<'_>,
    store: &Bound<'_, PyAny>,
    path: String,
    metadata_json: String,
    chunk_coords: Vec<u64>,
) -> PyResult<Py<PyBytes>> {
    let (storage, cache_key) = resolve_store_with_key(store)?;
    let data = py.detach(move || -> PyResult<Vec<u8>> {
        let array = array_view(storage, cache_key, &path, &metadata_json)?;
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
    let (storage, cache_key) = resolve_store_with_key(store)?;
    let data = py.detach(move || -> PyResult<Option<Vec<u8>>> {
        let array = array_view(storage, cache_key, &path, &metadata_json)?;
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
    let (storage, cache_key) = resolve_store_with_key(store)?;
    py.detach(move || {
        let array = array_view(storage, cache_key, &path, &metadata_json)?;
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
    let (storage, cache_key) = resolve_store_with_key(store)?;
    py.detach(move || {
        let array = array_view(storage, cache_key, &path, &metadata_json)?;
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
    let (storage, cache_key) = resolve_store_with_key(store)?;
    let data = py.detach(move || -> PyResult<Vec<u8>> {
        let array = array_view(storage, cache_key, &path, &metadata_json)?;
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
