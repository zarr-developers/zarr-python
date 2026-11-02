"""The v3 declaration: its JSON type, and the type aliases for members that hold a declaration.

This module is private. It sits below both code paths that read a declaration:
the model, which checks the members of a document, and the definitions,
which read a declaration's configuration. Public consumers import
`ZarrV3DeclarationJSON` from `zarr_metadata.v3`, and the aliases from
`zarr_metadata.v3.definition`. One declaration is read with
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

ZarrV3DeclarationJSON: TypeAlias = str | ZarrV3NamedConfigJSON
"""The JSON shape of any v3 metadata extension-point entry: either a bare
short-hand name string or a `{name, configuration, must_understand}` envelope.

Used for `data_type`, `chunk_grid`, `chunk_key_encoding`, individual
codec entries, and `storage_transformers` in v3 array metadata, and for
the inner `codecs` / `index_codecs` lists of the `sharding_indexed`
codec.
"""


# A member that holds a declaration is annotated with the alias for its
# kind. A scope uses that alias to decide which kind to read the declaration as.
# Each alias is defined as the declaration's JSON type, so a type checker and
# `check` both see it as `ZarrV3DeclarationJSON`.

ZarrV3DataTypeDeclaration = TypeAliasType("ZarrV3DataTypeDeclaration", ZarrV3DeclarationJSON)
"""A member that holds a data type: a document's `data_type`, or a struct field's data type. It is read in the same scope as the object that contains it."""
ZarrV3ChunkGridDeclaration = TypeAliasType("ZarrV3ChunkGridDeclaration", ZarrV3DeclarationJSON)
"""A member that holds a chunk grid: a document's `chunk_grid`."""
ZarrV3ChunkKeyEncodingDeclaration = TypeAliasType(
    "ZarrV3ChunkKeyEncodingDeclaration", ZarrV3DeclarationJSON
)
"""A member that holds a chunk key encoding: a document's `chunk_key_encoding`."""
ZarrV3CodecDeclaration = TypeAliasType("ZarrV3CodecDeclaration", ZarrV3DeclarationJSON)
"""A member that holds a codec. A document's `codecs` is `tuple[ZarrV3CodecDeclaration, ...]`, and so is a shard's."""
ZarrV3StaticCodecDeclaration = TypeAliasType("ZarrV3StaticCodecDeclaration", ZarrV3DeclarationJSON)
"""A member that holds a codec with a static encoded size. A shard's `index_codecs` uses
this, because a reader locates the index from a size it knows before reading it.
"""
ZarrV3StorageTransformerDeclaration = TypeAliasType(
    "ZarrV3StorageTransformerDeclaration", ZarrV3DeclarationJSON
)
"""A member that holds a storage transformer. A document's `storage_transformers` is `tuple[ZarrV3StorageTransformerDeclaration, ...]`."""


__all__ = [
    "ZarrV3ChunkGridDeclaration",
    "ZarrV3ChunkKeyEncodingDeclaration",
    "ZarrV3CodecDeclaration",
    "ZarrV3DataTypeDeclaration",
    "ZarrV3DeclarationJSON",
    "ZarrV3ExtensionField",
    "ZarrV3StaticCodecDeclaration",
    "ZarrV3StorageTransformerDeclaration",
]
