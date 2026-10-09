"""The public door to v2 definitions: the scope zarr-python 2.x reads in, and one field read in it."""

from __future__ import annotations

import pickle

import pytest

from zarr_metadata.v2.definition import (
    CORE_V2,
    V2_CODECS,
    V2_DATA_TYPES,
    AcceptedField,
    Context,
    UnclaimedField,
    ZarrV2CodecDefinition,
    ZarrV2DataTypeDefinition,
    resolve_codec_v2,
    resolve_dtype_v2,
)
from zarr_metadata.v3.codec.gzip import GZIP_CODEC
from zarr_metadata.v3.definition import (
    CORE_AND_EXTENSIONS,
    CodecDefinition,
)


def test_core_v2_files_every_v2_definition_apart_from_v3() -> None:
    """`CORE_V2` files the 12 data types and 21 codecs by their v2 kinds, and is a scope of format 2: a v2 `gzip` and a v3 `gzip` are two definitions of two kinds, and no scope files both, since `Context.joined` refuses two formats."""
    assert set(CORE_V2.definitions()) == {*V2_DATA_TYPES, *V2_CODECS}
    assert CORE_V2.format == 2
    assert CORE_V2.claimant(ZarrV2CodecDefinition, "gzip") is not GZIP_CODEC
    assert CORE_V2.claimant(ZarrV2CodecDefinition, "gzip") is not None
    assert CORE_V2.claimant(CodecDefinition, "gzip") is None
    assert CORE_AND_EXTENSIONS.claimant(CodecDefinition, "gzip") is GZIP_CODEC
    assert CORE_V2.claimant(ZarrV2DataTypeDefinition, "<f4") is not None
    with pytest.raises(TypeError, match="one Zarr format"):
        Context.joined(CORE_V2, CORE_AND_EXTENSIONS)
    assert pickle.loads(pickle.dumps(CORE_V2)) == CORE_V2


def test_the_v2_readers_read_one_field_in_core_v2_by_default() -> None:
    """`resolve_dtype_v2` and `resolve_codec_v2` read one field in `CORE_V2` when no scope is given, and in the scope given otherwise, prefixing every problem with `loc`."""
    dtype, problems = resolve_dtype_v2("<f4")
    assert isinstance(dtype, AcceptedField)
    assert problems == ()
    codec, problems = resolve_codec_v2({"id": "zlib", "level": 1}, loc=("compressor",))
    assert isinstance(codec, AcceptedField)
    assert problems == ()
    _, problems = resolve_codec_v2({"id": "zlib", "level": 10}, loc=("compressor",))
    assert [p.loc for p in problems] == [("compressor", "level")]
    assert isinstance(resolve_codec_v2({"id": "zlib"}, Context.of())[0], UnclaimedField)
