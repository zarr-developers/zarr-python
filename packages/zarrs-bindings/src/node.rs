use pyo3::prelude::*;
use zarrs::array::Array;
use zarrs::group::Group;
use zarrs::metadata::{ArrayMetadata, GroupMetadata};
use zarrs::node::{Node, NodePath, node_exists};
use zarrs::storage::{ReadableWritableListableStorage, StorePrefix};

use crate::store::resolve_store;
use crate::{NodeExistsError, NodeNotFoundError, runtime_err, value_err};

/// `path` arguments throughout this module are zarrs node paths, e.g. "/" or
/// "/foo/bar" (already normalized by the Python layer's `_node_path`).
pub(crate) fn parse_node_path(path: &str) -> PyResult<NodePath> {
    NodePath::new(path).map_err(value_err)
}

/// When a node exists at `node_path`: erase it (and everything under it) if
/// `overwrite`, otherwise raise `NodeExistsError`.
pub(crate) fn prepare_target(
    storage: &ReadableWritableListableStorage,
    node_path: &NodePath,
    overwrite: bool,
) -> PyResult<()> {
    if node_exists(storage, node_path).map_err(runtime_err)? {
        if !overwrite {
            return Err(NodeExistsError::new_err(format!(
                "a node already exists at path {}",
                node_path.as_str()
            )));
        }
        let prefix: StorePrefix = node_path.try_into().map_err(value_err)?;
        storage.erase_prefix(&prefix).map_err(runtime_err)?;
    }
    Ok(())
}

#[pyfunction]
pub(crate) fn create_group(
    py: Python<'_>,
    store: &Bound<'_, PyAny>,
    path: String,
    metadata_json: String,
    overwrite: bool,
) -> PyResult<()> {
    let storage = resolve_store(store)?;
    let metadata = GroupMetadata::try_from(metadata_json.as_str()).map_err(value_err)?;
    py.detach(move || {
        let node_path = parse_node_path(&path)?;
        prepare_target(&storage, &node_path, overwrite)?;
        let group = Group::new_with_metadata(storage, &path, metadata).map_err(value_err)?;
        group.store_metadata().map_err(runtime_err)
    })
}

#[pyfunction]
pub(crate) fn create_array(
    py: Python<'_>,
    store: &Bound<'_, PyAny>,
    path: String,
    metadata_json: String,
    overwrite: bool,
) -> PyResult<()> {
    let storage = resolve_store(store)?;
    let metadata = ArrayMetadata::try_from(metadata_json.as_str()).map_err(value_err)?;
    py.detach(move || {
        let node_path = parse_node_path(&path)?;
        prepare_target(&storage, &node_path, overwrite)?;
        let array = Array::new_with_metadata(storage, &path, metadata).map_err(value_err)?;
        array.store_metadata().map_err(runtime_err)
    })
}

#[pyfunction]
pub(crate) fn read_metadata(
    py: Python<'_>,
    store: &Bound<'_, PyAny>,
    path: String,
) -> PyResult<String> {
    let storage = resolve_store(store)?;
    py.detach(move || {
        let node =
            Node::open(&storage, &path).map_err(|e| NodeNotFoundError::new_err(e.to_string()))?;
        serde_json::to_string(node.metadata()).map_err(runtime_err)
    })
}

#[pyfunction]
pub(crate) fn delete_node(py: Python<'_>, store: &Bound<'_, PyAny>, path: String) -> PyResult<()> {
    let storage = resolve_store(store)?;
    py.detach(move || {
        let node_path = parse_node_path(&path)?;
        if !node_exists(&storage, &node_path).map_err(runtime_err)? {
            return Err(NodeNotFoundError::new_err(format!(
                "no node found at path {}",
                node_path.as_str()
            )));
        }
        let prefix: StorePrefix = (&node_path).try_into().map_err(value_err)?;
        storage.erase_prefix(&prefix).map_err(runtime_err)
    })
}

#[pyfunction]
pub(crate) fn list_children(
    py: Python<'_>,
    store: &Bound<'_, PyAny>,
    path: String,
) -> PyResult<Vec<(String, String)>> {
    let storage = resolve_store(store)?;
    py.detach(move || {
        let group =
            Group::open(storage, &path).map_err(|e| NodeNotFoundError::new_err(e.to_string()))?;
        let children = group.children(false).map_err(runtime_err)?;
        children
            .into_iter()
            .map(|node| {
                let metadata = serde_json::to_string(node.metadata()).map_err(runtime_err)?;
                Ok((node.path().as_str().to_string(), metadata))
            })
            .collect()
    })
}
