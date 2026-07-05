"""Test that the curated front-door names are accessible from the top-level zarr_metadata package."""

import re
from typing import get_args

import zarr_metadata as zm


def _group_rank(s: str) -> int:
    """RUF022 groups `__all__` as: SCREAMING_SNAKE (0), then TitleCase (1), then dunders (2).

    The exact intra-group ordering is ruff's own natural sort and is enforced by
    ruff itself (pre-commit + CI); this test only asserts the grouping, not the
    fragile tie-breaking, so it can't drift out of sync with ruff's implementation.
    """
    if s.startswith("__") and s.endswith("__"):
        return 2
    stripped = re.sub(r"[\d_]", "", s)
    return 0 if stripped.isupper() else 1


EXPECTED = [
    # Category A — metadata-document types
    "ArrayMetadataV2",
    "ArrayMetadataV2Partial",
    "ZArrayMetadata",
    "GroupMetadataV2",
    "GroupMetadataV2Partial",
    "ZGroupMetadata",
    "ConsolidatedMetadataV2",
    "ZAttrsMetadata",
    "CodecMetadataV2",
    "ArrayMetadataV3",
    "ArrayMetadataV3Partial",
    "ExtensionFieldV3",
    "GroupMetadataV3",
    "GroupMetadataV3Partial",
    "ConsolidatedMetadataV3",
    "NamedConfigV3",
    "MetadataV3",
    "JSONValue",
    # Category A' — metadata models (in-memory dataclasses over the documents)
    "ArrayMetadataModelV2",
    "ArrayMetadataModelV2Partial",
    "ArrayMetadataModelV3",
    "ArrayMetadataModelV3Partial",
    "GroupMetadataModelV2",
    "GroupMetadataModelV2Partial",
    "GroupMetadataModelV3",
    "GroupMetadataModelV3Partial",
    "ConsolidatedMetadataModelV2",
    "ConsolidatedMetadataModelV3",
    "ZarrMetadataV3",
    "ValidationProblem",
    "MetadataValidationError",
    # v2 data-type encoding union
    "DataTypeMetadataV2",
    # Category B — codec canonical unions
    "BloscCodecMetadata",
    "BytesCodecMetadata",
    "CastValueCodecMetadata",
    "Crc32cCodecMetadata",
    "GzipCodecMetadata",
    "ScaleOffsetCodecMetadata",
    "ShardingIndexedCodecMetadata",
    "TransposeCodecMetadata",
    "ZstdCodecMetadata",
    # Category C — grid/key canonical unions
    "RegularChunkGridMetadata",
    "RectilinearChunkGridMetadata",
    "DefaultChunkKeyEncodingMetadata",
    "V2ChunkKeyEncodingMetadata",
    # Category D — dtype trios
    # bool
    "BoolDataTypeName",
    "BOOL_DATA_TYPE_NAME",
    "BoolFillValue",
    # int8/16/32/64
    "Int8DataTypeName",
    "INT8_DATA_TYPE_NAME",
    "Int8FillValue",
    "Int16DataTypeName",
    "INT16_DATA_TYPE_NAME",
    "Int16FillValue",
    "Int32DataTypeName",
    "INT32_DATA_TYPE_NAME",
    "Int32FillValue",
    "Int64DataTypeName",
    "INT64_DATA_TYPE_NAME",
    "Int64FillValue",
    # uint8/16/32/64 (actual casing is Uint, not UInt)
    "Uint8DataTypeName",
    "UINT8_DATA_TYPE_NAME",
    "Uint8FillValue",
    "Uint16DataTypeName",
    "UINT16_DATA_TYPE_NAME",
    "Uint16FillValue",
    "Uint32DataTypeName",
    "UINT32_DATA_TYPE_NAME",
    "Uint32FillValue",
    "Uint64DataTypeName",
    "UINT64_DATA_TYPE_NAME",
    "Uint64FillValue",
    # float16/32/64
    "Float16DataTypeName",
    "FLOAT16_DATA_TYPE_NAME",
    "Float16FillValue",
    "Float32DataTypeName",
    "FLOAT32_DATA_TYPE_NAME",
    "Float32FillValue",
    "Float64DataTypeName",
    "FLOAT64_DATA_TYPE_NAME",
    "Float64FillValue",
    # complex64/128
    "Complex64DataTypeName",
    "COMPLEX64_DATA_TYPE_NAME",
    "Complex64FillValue",
    "Complex128DataTypeName",
    "COMPLEX128_DATA_TYPE_NAME",
    "Complex128FillValue",
    # bytes
    "BytesDataTypeName",
    "BYTES_DATA_TYPE_NAME",
    "BytesFillValue",
    # string
    "StringDataTypeName",
    "STRING_DATA_TYPE_NAME",
    "StringFillValue",
    # numpy_datetime64
    "NumpyDatetime64DataTypeName",
    "NUMPY_DATETIME64_DATA_TYPE_NAME",
    "NumpyDatetime64FillValue",
    # numpy_timedelta64
    "NumpyTimedelta64DataTypeName",
    "NUMPY_TIMEDELTA64_DATA_TYPE_NAME",
    "NumpyTimedelta64FillValue",
    # struct
    "StructDataTypeName",
    "STRUCT_DATA_TYPE_NAME",
    "StructFillValue",
    # raw (no _DATA_TYPE_NAME constant)
    "RawBytesDataTypeName",
    "RawBytesFillValue",
    # Category E — constant+Literal pairs
    "ARRAY_ORDER_V2",
    "ArrayOrderV2",
    "ARRAY_DIMENSION_SEPARATOR_V2",
    "ArrayDimensionSeparatorV2",
    "ENDIANNESS",
    "Endianness",
    "BYTES_CODEC_NAME",
    "BytesCodecName",
    "BLOSC_CODEC_NAME",
    "BloscCodecName",
    "BLOSC_CNAME",
    "BloscCName",
    "BLOSC_SHUFFLE",
    "BloscShuffle",
    "CAST_ROUNDING_MODE",
    "CastRoundingMode",
    "CAST_OUT_OF_RANGE_MODE",
    "CastOutOfRangeMode",
    "CAST_VALUE_CODEC_NAME",
    "CastValueCodecName",
    "CRC32C_CODEC_NAME",
    "Crc32cCodecName",
    "GZIP_CODEC_NAME",
    "GzipCodecName",
    "SCALE_OFFSET_CODEC_NAME",
    "ScaleOffsetCodecName",
    "SHARDING_INDEX_LOCATION",
    "ShardingIndexLocation",
    "SHARDING_INDEXED_CODEC_NAME",
    "ShardingIndexedCodecName",
    "TRANSPOSE_CODEC_NAME",
    "TransposeCodecName",
    "ZSTD_CODEC_NAME",
    "ZstdCodecName",
    "REGULAR_CHUNK_GRID_NAME",
    "RegularChunkGridName",
    "RECTILINEAR_CHUNK_GRID_NAME",
    "RectilinearChunkGridName",
    "DEFAULT_CHUNK_KEY_ENCODING_NAME",
    "DefaultChunkKeyEncodingName",
    "DEFAULT_CHUNK_KEY_ENCODING_SEPARATOR",
    "DefaultChunkKeyEncodingSeparator",
    "V2_CHUNK_KEY_ENCODING_NAME",
    "V2ChunkKeyEncodingName",
    "V2_CHUNK_KEY_ENCODING_SEPARATOR",
    "V2ChunkKeyEncodingSeparator",
    "NUMPY_TIME_UNIT",
    "NumpyTimeUnit",
]


def test_front_door_names_public() -> None:
    missing = [n for n in EXPECTED if n not in zm.__all__ or not hasattr(zm, n)]
    assert not missing, f"missing from top-level API: {missing}"


def test_front_door_is_exactly_expected() -> None:
    """`__all__` must contain exactly the curated names (plus `__version__`).

    Guards against a name being promoted to the front door without a
    corresponding, deliberate entry in `EXPECTED` — i.e. an accidental
    addition to the public API surface.
    """
    assert set(zm.__all__) - {"__version__"} == set(EXPECTED)


def test_all_is_grouped_and_unique() -> None:
    ranks = [_group_rank(n) for n in zm.__all__]
    assert ranks == sorted(ranks), "`__all__` groups out of order (SCREAMING, TitleCase, dunder)"
    assert len(zm.__all__) == len(set(zm.__all__))


def test_promoted_pairs_drift() -> None:
    pairs = [
        (zm.ENDIANNESS, zm.Endianness),
        (zm.BLOSC_CNAME, zm.BloscCName),
        (zm.BLOSC_SHUFFLE, zm.BloscShuffle),
        (zm.SHARDING_INDEX_LOCATION, zm.ShardingIndexLocation),
        (zm.NUMPY_TIME_UNIT, zm.NumpyTimeUnit),
        (zm.CAST_ROUNDING_MODE, zm.CastRoundingMode),
        (zm.CAST_OUT_OF_RANGE_MODE, zm.CastOutOfRangeMode),
        (zm.ARRAY_ORDER_V2, zm.ArrayOrderV2),
    ]
    for const, lit in pairs:
        assert set(const) == set(get_args(lit))
