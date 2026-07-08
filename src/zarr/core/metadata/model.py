"""Conversions between zarr-python's runtime metadata classes and the
``zarr_metadata`` model layer.

The model classes (``zarr_metadata.model``) are canonical, lossless,
JSON-shaped representations of metadata documents: extension points (data
types, codecs, chunk grids, chunk key encodings) are held as uninterpreted
name + configuration pairs, and fill values are held verbatim in their JSON
form. The runtime classes (``zarr.core.metadata``, ``zarr.core.group``)
interpret those extension points into live objects (``ZDType`` instances,
codec instances, cast fill-value scalars).

These converters bridge the two worlds during the migration of zarr-python's
node classes from the runtime metadata classes to the model classes (see
``AsyncArray._future_metadata`` / ``AsyncGroup._future_metadata``):

- runtime -> model is total: every runtime metadata object serializes to a
  valid document, so ``*_to_model`` never fails on well-formed input.
- model -> runtime is partial: a model may hold extension points that this
  zarr-python installation cannot interpret, in which case ``*_from_model``
  raises just like opening the corresponding document from a store would.

Known representational gaps, resolved in favor of current runtime semantics:

- A v2 model distinguishes "no ``.zattrs`` file" (``UNSET``) from an empty
  ``.zattrs`` (``{}``); the runtime classes collapse both to ``{}``.
- A v2 ``GroupMetadata`` may carry ``consolidated_metadata``, which in the v2
  document world lives in a separate ``.zmetadata`` document and has no field
  in ``GroupMetadataModelV2``; it is dropped by ``group_metadata_to_model``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, overload

from zarr_metadata.model import (
    ArrayMetadataModelV2,
    ArrayMetadataModelV3,
    ConsolidatedMetadataModelV3,
    GroupMetadataModelV2,
    GroupMetadataModelV3,
)

from zarr.core.metadata.v2 import ArrayV2Metadata
from zarr.core.metadata.v3 import ArrayV3Metadata

if TYPE_CHECKING:
    from zarr.core.group import GroupMetadata

__all__ = [
    "array_metadata_from_model",
    "array_metadata_to_model",
    "group_metadata_from_model",
    "group_metadata_to_model",
]


def _arrays_to_lists(data: Any) -> Any:
    """Recursively convert sequences in a JSON-shaped value to lists.

    The model layer canonicalizes JSON arrays as tuples (immutability); the
    runtime classes hold whatever the JSON decoder produced, i.e. lists.
    Converting a model's document form back to lists before handing it to the
    runtime ``from_dict`` makes model -> runtime indistinguishable from
    reading the equivalent document out of a store.
    """
    if isinstance(data, Mapping):
        return {k: _arrays_to_lists(v) for k, v in data.items()}
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        return [_arrays_to_lists(v) for v in data]
    return data


@overload
def array_metadata_to_model(metadata: ArrayV2Metadata) -> ArrayMetadataModelV2: ...
@overload
def array_metadata_to_model(metadata: ArrayV3Metadata) -> ArrayMetadataModelV3: ...


def array_metadata_to_model(
    metadata: ArrayV2Metadata | ArrayV3Metadata,
) -> ArrayMetadataModelV2 | ArrayMetadataModelV3:
    """Convert a runtime array metadata object to its document model.

    The conversion round-trips through the JSON document form, which is the
    shared language of the two layers: ``to_dict`` serializes the interpreted
    extension points (dtype, codecs, fill value) back to their document
    spellings, and ``from_json`` validates and captures them losslessly.
    """
    if isinstance(metadata, ArrayV2Metadata):
        return ArrayMetadataModelV2.from_json(metadata.to_dict())
    return ArrayMetadataModelV3.from_json(metadata.to_dict())


@overload
def array_metadata_from_model(model: ArrayMetadataModelV2) -> ArrayV2Metadata: ...
@overload
def array_metadata_from_model(model: ArrayMetadataModelV3) -> ArrayV3Metadata: ...


def array_metadata_from_model(
    model: ArrayMetadataModelV2 | ArrayMetadataModelV3,
) -> ArrayV2Metadata | ArrayV3Metadata:
    """Convert an array metadata document model to a runtime metadata object.

    Raises the same errors as reading the equivalent document from a store:
    unknown data types, codecs, or must-understand extension fields that this
    installation cannot interpret are rejected by the runtime ``from_dict``.
    """
    if isinstance(model, ArrayMetadataModelV2):
        return ArrayV2Metadata.from_dict(_arrays_to_lists(model.to_json()))
    return ArrayV3Metadata.from_dict(_arrays_to_lists(model.to_json()))


def group_metadata_to_model(
    metadata: GroupMetadata,
) -> GroupMetadataModelV2 | GroupMetadataModelV3:
    """Convert a runtime ``GroupMetadata`` to its document model.

    This is where the single runtime group class splits into per-format
    models: a ``GroupMetadata`` with ``zarr_format == 2`` becomes a
    ``GroupMetadataModelV2``, otherwise a ``GroupMetadataModelV3``.

    For v2, ``consolidated_metadata`` is dropped: in the v2 document world it
    lives in a separate ``.zmetadata`` document, not in ``.zgroup``, so the
    group document model has no field for it.
    """
    if metadata.zarr_format == 2:
        # GroupMetadata.to_dict emits v3-only keys (node_type, and possibly
        # consolidated_metadata) even for v2, so build the v2 document
        # explicitly rather than passing to_dict output through.
        return GroupMetadataModelV2.from_json({"zarr_format": 2, "attributes": metadata.attributes})
    return GroupMetadataModelV3.from_json(metadata.to_dict())


def group_metadata_from_model(
    model: GroupMetadataModelV2 | GroupMetadataModelV3,
) -> GroupMetadata:
    """Convert a group metadata document model to a runtime ``GroupMetadata``.

    A v3 model carrying must-understand extension fields is rejected, matching
    how the runtime class treats unknown keys in a v3 group document.
    """
    # Deferred import: zarr.core.group imports from zarr.core.metadata, so a
    # module-level import here would be circular.
    from zarr.core.group import ConsolidatedMetadata, GroupMetadata

    if isinstance(model, GroupMetadataModelV2):
        # isinstance rather than `is UNSET`: PEP 661 sentinel narrowing is not
        # yet supported by mypy (https://github.com/python/mypy/pull/21647);
        # mypy also drops the UNSET arm of the field's union, hence the ignore.
        attributes = (
            _arrays_to_lists(model.attributes)
            if isinstance(model.attributes, Mapping)  # type: ignore[redundant-expr]
            else {}
        )
        return GroupMetadata(attributes=attributes, zarr_format=2)
    if model.extra_fields:
        raise ValueError(
            "Cannot convert a v3 group metadata model with extra fields "
            f"({sorted(model.extra_fields)}) to zarr.core.group.GroupMetadata, "
            "which does not support extension fields."
        )
    consolidated = None
    if isinstance(model.consolidated_metadata, ConsolidatedMetadataModelV3):
        consolidated = ConsolidatedMetadata.from_dict(
            _arrays_to_lists(model.consolidated_metadata.to_json())
        )
    return GroupMetadata(
        attributes=_arrays_to_lists(model.attributes),
        zarr_format=3,
        consolidated_metadata=consolidated,
    )
