"""The public entry point for v2 definitions: the scope that zarr-python 2.x reads in, and reading one field in it."""

from __future__ import annotations

import pickle

import pytest

from zarr_metadata.definition import (
    AcceptedField,
    UnclaimedField,
    resolve,
)
from zarr_metadata.v2.codec import CODECS_V2
from zarr_metadata.v2.data_type import DATA_TYPES_V2
from zarr_metadata.v2.definition import (
    CORE_V2_CONTEXT,
    ZarrV2CodecDefinition,
    ZarrV2Context,
    ZarrV2DataTypeDefinition,
)
from zarr_metadata.v3.codec.gzip import GZIP_CODEC
from zarr_metadata.v3.definition import (
    CORE_AND_EXTENSIONS_CONTEXT,
    CodecDefinition,
)


def test_core_v2_files_every_v2_definition_apart_from_v3() -> None:
    """`CORE_V2_CONTEXT` files the 12 data types and 21 codecs by their v2 kinds, and is a scope of format 2: a v2 `gzip` and a v3 `gzip` are two definitions of two different kinds. No scope can file both, because `union` rejects scopes of two formats."""
    assert set(CORE_V2_CONTEXT.data_types.values()) == set(DATA_TYPES_V2)
    assert set(CORE_V2_CONTEXT.codecs.values()) == set(CODECS_V2)
    assert CORE_V2_CONTEXT.zarr_format == 2
    assert CORE_V2_CONTEXT.claimant(ZarrV2CodecDefinition, "gzip") is not GZIP_CODEC
    assert CORE_V2_CONTEXT.claimant(ZarrV2CodecDefinition, "gzip") is not None
    assert CORE_AND_EXTENSIONS_CONTEXT.claimant(CodecDefinition, "gzip") is GZIP_CODEC
    assert CORE_V2_CONTEXT.claimant(ZarrV2DataTypeDefinition, "<f4") is not None
    with pytest.raises(TypeError, match="one Zarr format"):
        CORE_V2_CONTEXT.union(CORE_AND_EXTENSIONS_CONTEXT)  # pyright: ignore[reportArgumentType]
    assert pickle.loads(pickle.dumps(CORE_V2_CONTEXT)) == CORE_V2_CONTEXT


def test_resolve_reads_one_v2_field_in_a_v2_scope() -> None:
    """`resolve` reads a v2 dtype or codec with its kind in a v2 scope, prefixing every problem with `loc`, as it reads a v3 field."""
    dtype, problems = resolve("<f4", ZarrV2DataTypeDefinition, CORE_V2_CONTEXT)
    assert isinstance(dtype, AcceptedField)
    assert problems == ()
    codec, problems = resolve(
        {"id": "zlib", "level": 1}, ZarrV2CodecDefinition, CORE_V2_CONTEXT, ("compressor",)
    )
    assert isinstance(codec, AcceptedField)
    assert problems == ()
    _, problems = resolve(
        {"id": "zlib", "level": 10}, ZarrV2CodecDefinition, CORE_V2_CONTEXT, ("compressor",)
    )
    assert [p.loc for p in problems] == [("compressor", "level")]
    assert isinstance(
        resolve({"id": "zlib"}, ZarrV2CodecDefinition, ZarrV2Context())[0], UnclaimedField
    )
