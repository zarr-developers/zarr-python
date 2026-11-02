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
    node_type: Literal["array"]
    data_type: ZarrV3DataTypeField
    shape: tuple[Annotated[int, Ge(0)], ...]
    chunk_grid: ZarrV3ChunkGridField
    chunk_key_encoding: ZarrV3ChunkKeyEncodingField
    fill_value: JSONValue
    codecs: tuple[ZarrV3CodecField, ...]
    attributes: NotRequired[Mapping[str, JSONValue]]
    storage_transformers: NotRequired[tuple[ZarrV3StorageTransformerField, ...]]
    dimension_names: NotRequired[tuple[str | None, ...]]


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
