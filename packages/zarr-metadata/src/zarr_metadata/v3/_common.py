"""The v3 metadata field: its JSON type, and the type aliases for members that hold a field.

This module is private. It sits below both code paths that read a field:
the model, which checks the fields of a document, and the definitions,
which read a field's configuration. Public consumers import
`ZarrV3MetadataFieldJSON` from `zarr_metadata.v3`, and the aliases from
`zarr_metadata.v3.definition`. One field is read with
`zarr_metadata.definition.resolve`.
"""

from typing import TypeAlias

from typing_extensions import TypeAliasType

from zarr_metadata._common import JSONValue, ZarrV3NamedConfigJSON

ZarrV3ExtensionField: TypeAlias = JSONValue
"""The JSON value of an unknown top-level v3 metadata field.

An object carrying the literal member `must_understand: false` may be ignored.
Every other JSON shape implicitly requires understanding; recognition itself
belongs to the reader rather than this structural type.
"""

ZarrV3MetadataFieldJSON: TypeAlias = str | ZarrV3NamedConfigJSON
"""The JSON shape of any v3 metadata extension-point entry: either a bare
short-hand name string or a `{name, configuration, must_understand}` envelope.

Used for `data_type`, `chunk_grid`, `chunk_key_encoding`, individual
codec entries, and `storage_transformers` in v3 array metadata, and for
the inner `codecs` / `index_codecs` lists of the `sharding_indexed`
codec.
"""


# A member that holds a metadata field is annotated with the alias for its
# kind. A scope uses that alias to decide which kind to read the field as.
# Each alias is defined as the field's JSON type, so a type checker and
# `check` both see it as `ZarrV3MetadataFieldJSON`.

ZarrV3DataTypeField = TypeAliasType("ZarrV3DataTypeField", ZarrV3MetadataFieldJSON)
"""A member that holds a data type: a document's `data_type`, or a struct field's data type. It is read in the same scope as the object that contains it."""
ZarrV3ChunkGridField = TypeAliasType("ZarrV3ChunkGridField", ZarrV3MetadataFieldJSON)
"""A member that holds a chunk grid: a document's `chunk_grid`."""
ZarrV3ChunkKeyEncodingField = TypeAliasType("ZarrV3ChunkKeyEncodingField", ZarrV3MetadataFieldJSON)
"""A member that holds a chunk key encoding: a document's `chunk_key_encoding`."""
ZarrV3CodecField = TypeAliasType("ZarrV3CodecField", ZarrV3MetadataFieldJSON)
"""A member that holds a codec. A document's `codecs` is `tuple[ZarrV3CodecField, ...]`, and so is a shard's."""
ZarrV3StaticCodecField = TypeAliasType("ZarrV3StaticCodecField", ZarrV3MetadataFieldJSON)
"""A member that holds a codec with a static encoded size. A shard's `index_codecs` uses
this, because a reader locates the index from a size it knows before reading it.
"""
ZarrV3StorageTransformerField = TypeAliasType(
    "ZarrV3StorageTransformerField", ZarrV3MetadataFieldJSON
)
"""A member that holds a storage transformer. A document's `storage_transformers` is `tuple[ZarrV3StorageTransformerField, ...]`."""


__all__ = [
    "ZarrV3ChunkGridField",
    "ZarrV3ChunkKeyEncodingField",
    "ZarrV3CodecField",
    "ZarrV3DataTypeField",
    "ZarrV3ExtensionField",
    "ZarrV3MetadataFieldJSON",
    "ZarrV3StaticCodecField",
    "ZarrV3StorageTransformerField",
]
