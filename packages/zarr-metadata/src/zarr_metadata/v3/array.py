"""Zarr v3 array metadata types."""

from collections.abc import Mapping
from typing import Annotated, Final, Literal, NotRequired

from annotated_types import Ge
from typing_extensions import TypedDict

from zarr_metadata._common import JSONValue
from zarr_metadata.v3._common import (
    ZarrV3ChunkGridField,
    ZarrV3ChunkKeyEncodingField,
    ZarrV3CodecField,
    ZarrV3DataTypeField,
    ZarrV3ExtensionField,
    ZarrV3StorageTransformerField,
)


class ZarrV3ArrayMetadataJSON(TypedDict, extra_items=ZarrV3ExtensionField):
    """
    Zarr v3 array metadata document (the `zarr.json` content for an array).

    Extra keys may contain arbitrary JSON values. Each extension point is
    annotated with the field alias of its kind. For example, `data_type` is
    a `ZarrV3DataTypeField`, and each entry of `codecs` is a
    `ZarrV3CodecField`. The alias describes the JSON of the metadata field,
    and it tells a scope which kind to read the field as.

    See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#array-metadata
    """

    zarr_format: Literal[3]
    """The Zarr format version, `3`."""
    node_type: Literal["array"]
    """The kind of node the document describes, `"array"`."""
    data_type: ZarrV3DataTypeField
    """The data type of the array's elements, e.g. `"float64"`."""
    shape: tuple[Annotated[int, Ge(0)], ...]
    """The length of each dimension of the array."""
    chunk_grid: ZarrV3ChunkGridField
    """How the array is divided into chunks, e.g. a `"regular"` grid of one chunk shape."""
    chunk_key_encoding: ZarrV3ChunkKeyEncodingField
    """How a chunk's grid coordinates become its key in the store."""
    fill_value: JSONValue
    """The value of an element in a chunk that was never written.

    Which JSON values it may take depends on `data_type`.
    """
    codecs: tuple[ZarrV3CodecField, ...]
    """The codecs that encode each chunk, in encoding order.

    Zero or more array -> array codecs, then exactly one array -> bytes codec,
    then zero or more bytes -> bytes codecs, so the list is never empty
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L974-L983).
    """
    attributes: NotRequired[Mapping[str, JSONValue]]
    """User attributes: arbitrary JSON values by string key."""
    storage_transformers: NotRequired[tuple[ZarrV3StorageTransformerField, ...]]
    """The storage transformers between the array and its store.

    Absent means none, as does an empty list.
    """
    dimension_names: NotRequired[tuple[str | None, ...]]
    """A name for each dimension, `null` for an unnamed one.

    As long as `shape`; absent means every dimension is unnamed.
    """


ZarrV3ArrayMetadataStoreKey = Literal["zarr.json"]
"""Literal type of the store key holding a v3 array's metadata document."""

ZARR_V3_ARRAY_METADATA_STORE_KEY: Final[ZarrV3ArrayMetadataStoreKey] = "zarr.json"
"""The store key a v3 array's metadata document is persisted under.

v3 uses one key for both node types; the document's `node_type` field
distinguishes an array from a group.
"""


__all__ = [
    "ZARR_V3_ARRAY_METADATA_STORE_KEY",
    "ZarrV3ArrayMetadataJSON",
    "ZarrV3ArrayMetadataStoreKey",
]
