use pyo3::prelude::*;
use zarrs::group::Group;
use zarrs::metadata::GroupMetadata;
use zarrs::node::{NodePath, node_exists};
use zarrs::storage::{ReadableWritableListableStorage, StorePrefix};

use crate::store::resolve_store;
use crate::{NodeExistsError, runtime_err, value_err};

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
