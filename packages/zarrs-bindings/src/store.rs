use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::byte_range::{ByteRange, ByteRangeIterator};
use zarrs::storage::{
    Bytes, ListableStorageTraits, MaybeBytes, MaybeBytesIterator, OffsetBytesIterator,
    ReadableStorageTraits, ReadableWritableListableStorage, StorageError, StoreKey, StoreKeys,
    StoreKeysPrefixes, StorePrefix, WritableStorageTraits,
};

/// A zarrs store backed by a Python `zarr.zarrs._bridge.StoreShim`.
///
/// Every method attaches to the Python interpreter and calls the shim, which
/// blocks on the zarr event loop. Blocking waits in Python release the GIL, so
/// the loop thread can make progress while a Rust worker waits here.
pub(crate) struct PyStore(Py<PyAny>);

fn py_err(err: PyErr) -> StorageError {
    StorageError::Other(err.to_string())
}

fn invalid(err: impl std::fmt::Display) -> StorageError {
    StorageError::Other(err.to_string())
}

impl PyStore {
    fn get_with_range(
        &self,
        key: &StoreKey,
        range: Option<&ByteRange>,
    ) -> Result<MaybeBytes, StorageError> {
        Python::attach(|py| {
            let shim = self.0.bind(py);
            let result = match range {
                None => shim.call_method1("get", (key.as_str(),)),
                Some(ByteRange::FromStart(offset, length)) => {
                    shim.call_method1("get_range", (key.as_str(), *offset, *length))
                }
                Some(ByteRange::Suffix(suffix)) => {
                    shim.call_method1("get_suffix", (key.as_str(), *suffix))
                }
            }
            .map_err(py_err)?;
            if result.is_none() {
                Ok(None)
            } else {
                let bytes: Vec<u8> = result.extract().map_err(py_err)?;
                Ok(Some(Bytes::from(bytes)))
            }
        })
    }
}

impl ReadableStorageTraits for PyStore {
    fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        self.get_with_range(key, None)
    }

    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        let mut out = Vec::new();
        for byte_range in byte_ranges {
            match self.get_with_range(key, Some(&byte_range))? {
                Some(bytes) => out.push(Ok(bytes)),
                None => return Ok(None),
            }
        }
        Ok(Some(Box::new(out.into_iter())))
    }

    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        Python::attach(|py| {
            self.0
                .bind(py)
                .call_method1("getsize", (key.as_str(),))
                .map_err(py_err)?
                .extract()
                .map_err(py_err)
        })
    }

    fn supports_get_partial(&self) -> bool {
        true
    }
}

impl WritableStorageTraits for PyStore {
    fn set(&self, key: &StoreKey, value: Bytes) -> Result<(), StorageError> {
        Python::attach(|py| {
            let data = PyBytes::new(py, &value);
            self.0
                .bind(py)
                .call_method1("set", (key.as_str(), data))
                .map_err(py_err)?;
            Ok(())
        })
    }

    fn set_partial_many(
        &self,
        key: &StoreKey,
        offset_values: OffsetBytesIterator,
    ) -> Result<(), StorageError> {
        // read-modify-write fallback provided by zarrs
        zarrs::storage::store_set_partial_many(self, key, offset_values)
    }

    fn supports_set_partial(&self) -> bool {
        false
    }

    fn erase(&self, key: &StoreKey) -> Result<(), StorageError> {
        Python::attach(|py| {
            self.0
                .bind(py)
                .call_method1("delete", (key.as_str(),))
                .map_err(py_err)?;
            Ok(())
        })
    }

    fn erase_prefix(&self, prefix: &StorePrefix) -> Result<(), StorageError> {
        Python::attach(|py| {
            self.0
                .bind(py)
                .call_method1("delete_prefix", (prefix.as_str(),))
                .map_err(py_err)?;
            Ok(())
        })
    }
}

impl ListableStorageTraits for PyStore {
    fn list(&self) -> Result<StoreKeys, StorageError> {
        Python::attach(|py| {
            let keys: Vec<String> = self
                .0
                .bind(py)
                .call_method0("list")
                .map_err(py_err)?
                .extract()
                .map_err(py_err)?;
            keys.into_iter()
                .map(|k| StoreKey::new(k).map_err(invalid))
                .collect()
        })
    }

    fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
        Python::attach(|py| {
            let keys: Vec<String> = self
                .0
                .bind(py)
                .call_method1("list_prefix", (prefix.as_str(),))
                .map_err(py_err)?
                .extract()
                .map_err(py_err)?;
            keys.into_iter()
                .map(|k| StoreKey::new(k).map_err(invalid))
                .collect()
        })
    }

    fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
        Python::attach(|py| {
            let (keys, prefixes): (Vec<String>, Vec<String>) = self
                .0
                .bind(py)
                .call_method1("list_dir", (prefix.as_str(),))
                .map_err(py_err)?
                .extract()
                .map_err(py_err)?;
            let keys = keys
                .into_iter()
                .map(|k| StoreKey::new(k).map_err(invalid))
                .collect::<Result<Vec<_>, StorageError>>()?;
            let prefixes = prefixes
                .into_iter()
                .map(|p| StorePrefix::new(p).map_err(invalid))
                .collect::<Result<Vec<_>, StorageError>>()?;
            Ok(StoreKeysPrefixes::new(keys, prefixes))
        })
    }

    fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError> {
        Python::attach(|py| {
            self.0
                .bind(py)
                .call_method1("getsize_prefix", (prefix.as_str(),))
                .map_err(py_err)?
                .extract()
                .map_err(py_err)
        })
    }
}

/// Like `resolve_store`, but also returns a cache key for the constructed
/// storage: `Some(root)` for native filesystem stores (which are safe to key an
/// Array cache on), `None` for the generic Python-callback path (uncached).
pub(crate) fn resolve_store_with_key(
    obj: &Bound<'_, PyAny>,
) -> PyResult<(ReadableWritableListableStorage, Option<String>)> {
    if let Ok(config) = obj.cast::<PyDict>() {
        if let Some(root) = config.get_item("filesystem")? {
            let root: String = root.extract()?;
            let store =
                FilesystemStore::new(&root).map_err(|e| PyValueError::new_err(e.to_string()))?;
            return Ok((Arc::new(store), Some(root)));
        }
        return Err(PyValueError::new_err("unrecognized store configuration"));
    }
    Ok((Arc::new(PyStore(obj.clone().unbind())), None))
}

/// Convert the Python-side store representation (`zarr.zarrs._bridge.resolve_store`
/// output) into a zarrs storage handle.
pub(crate) fn resolve_store(obj: &Bound<'_, PyAny>) -> PyResult<ReadableWritableListableStorage> {
    Ok(resolve_store_with_key(obj)?.0)
}
