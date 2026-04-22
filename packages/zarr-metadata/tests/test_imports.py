"""
Smoke test: every public name is importable.
"""

from __future__ import annotations


def test_top_level_imports() -> None:
    from zarr_metadata import (
        JSON,
        ArrayMetadata,
        ArrayMetadataV2,
        ArrayMetadataV3,
        GroupMetadata,
        GroupMetadataV2,
        GroupMetadataV3,
        NamedConfig,
        NamedRequiredConfig,
    )

    _ = (
        JSON,
        ArrayMetadata,
        ArrayMetadataV2,
        ArrayMetadataV3,
        GroupMetadata,
        GroupMetadataV2,
        GroupMetadataV3,
        NamedConfig,
        NamedRequiredConfig,
    )


def test_v2_imports() -> None:
    from zarr_metadata.v2 import (
        ArrayMetadataV2,
        ConsolidatedMetadataV2,
        DataTypeV2,
        DataTypeV2Structured,
        GroupMetadataV2,
    )

    _ = (
        ArrayMetadataV2,
        ConsolidatedMetadataV2,
        DataTypeV2,
        DataTypeV2Structured,
        GroupMetadataV2,
    )


def test_v3_imports() -> None:
    from zarr_metadata.v3 import (
        AllowedExtraField,
        ArrayMetadataV3,
        ConsolidatedMetadataV3,
        GroupMetadataV3,
        MetadataField,
    )

    _ = (
        AllowedExtraField,
        ArrayMetadataV3,
        ConsolidatedMetadataV3,
        GroupMetadataV3,
        MetadataField,
    )


def test_v3_chunk_grid_imports() -> None:
    from zarr_metadata.v3.chunk_grid.rectilinear import (
        RECTILINEAR_CHUNK_GRID_NAME,
        RectilinearChunkGrid,
        RectilinearChunkGridConfiguration,
        RectilinearChunkGridName,
        RectilinearDimSpec,
    )
    from zarr_metadata.v3.chunk_grid.regular import (
        REGULAR_CHUNK_GRID_NAME,
        RegularChunkGrid,
        RegularChunkGridConfiguration,
        RegularChunkGridName,
    )

    _ = (
        RECTILINEAR_CHUNK_GRID_NAME,
        REGULAR_CHUNK_GRID_NAME,
        RectilinearChunkGrid,
        RectilinearChunkGridConfiguration,
        RectilinearChunkGridName,
        RectilinearDimSpec,
        RegularChunkGrid,
        RegularChunkGridConfiguration,
        RegularChunkGridName,
    )


def test_v3_chunk_key_encoding_imports() -> None:
    from zarr_metadata.v3.chunk_key_encoding import ChunkKeySeparator
    from zarr_metadata.v3.chunk_key_encoding.default import (
        DEFAULT_CHUNK_KEY_ENCODING_NAME,
        DefaultChunkKeyEncoding,
        DefaultChunkKeyEncodingConfiguration,
        DefaultChunkKeyEncodingName,
    )
    from zarr_metadata.v3.chunk_key_encoding.v2 import (
        V2_CHUNK_KEY_ENCODING_NAME,
        V2ChunkKeyEncoding,
        V2ChunkKeyEncodingConfiguration,
        V2ChunkKeyEncodingName,
    )

    _ = (
        ChunkKeySeparator,
        DEFAULT_CHUNK_KEY_ENCODING_NAME,
        DefaultChunkKeyEncoding,
        DefaultChunkKeyEncodingConfiguration,
        DefaultChunkKeyEncodingName,
        V2_CHUNK_KEY_ENCODING_NAME,
        V2ChunkKeyEncoding,
        V2ChunkKeyEncodingConfiguration,
        V2ChunkKeyEncodingName,
    )


def test_codec_imports() -> None:
    from zarr_metadata.v3.codec import Codec
    from zarr_metadata.v3.codec.blosc import (
        BloscCName,
        BloscCodec,
        BloscCodecConfiguration,
        BloscCodecName,
        BloscShuffle,
    )
    from zarr_metadata.v3.codec.bytes import (
        BytesCodec,
        BytesCodecConfiguration,
        BytesCodecName,
    )
    from zarr_metadata.v3.codec.crc32c import Crc32cCodec, Crc32cCodecName
    from zarr_metadata.v3.codec.gzip import GzipCodec, GzipCodecConfiguration, GzipCodecName
    from zarr_metadata.v3.codec.sharding import (
        ShardingCodec,
        ShardingCodecConfiguration,
        ShardingCodecName,
    )
    from zarr_metadata.v3.codec.transpose import (
        TransposeCodec,
        TransposeCodecConfiguration,
        TransposeCodecName,
    )
    from zarr_metadata.v3.codec.zstd import ZstdCodec, ZstdCodecConfiguration, ZstdCodecName

    _ = (
        Codec,
        BloscCodec,
        BloscCodecConfiguration,
        BloscCodecName,
        BytesCodec,
        BytesCodecConfiguration,
        BytesCodecName,
        BloscCName,
        Crc32cCodec,
        Crc32cCodecName,
        GzipCodec,
        GzipCodecConfiguration,
        GzipCodecName,
        ShardingCodec,
        ShardingCodecConfiguration,
        ShardingCodecName,
        BloscShuffle,
        TransposeCodec,
        TransposeCodecConfiguration,
        TransposeCodecName,
        ZstdCodec,
        ZstdCodecConfiguration,
        ZstdCodecName,
    )


def test_dtype_imports() -> None:
    from zarr_metadata.v3.data_type import DType
    from zarr_metadata.v3.data_type.bool import BOOL_DTYPE_NAME, BoolDTypeName, BoolFillValue
    from zarr_metadata.v3.data_type.bytes import BYTES_DTYPE_NAME, BytesDTypeName, BytesFillValue
    from zarr_metadata.v3.data_type.complex64 import (
        COMPLEX64_DTYPE_NAME,
        Complex64Component,
        Complex64DTypeName,
        Complex64FillValue,
    )
    from zarr_metadata.v3.data_type.complex128 import (
        COMPLEX128_DTYPE_NAME,
        Complex128Component,
        Complex128DTypeName,
        Complex128FillValue,
    )
    from zarr_metadata.v3.data_type.float16 import (
        FLOAT16_DTYPE_NAME,
        Float16DTypeName,
        Float16FillValue,
        Float16SpecialFillValue,
    )
    from zarr_metadata.v3.data_type.float32 import (
        FLOAT32_DTYPE_NAME,
        Float32DTypeName,
        Float32FillValue,
        Float32SpecialFillValue,
    )
    from zarr_metadata.v3.data_type.float64 import (
        FLOAT64_DTYPE_NAME,
        Float64DTypeName,
        Float64FillValue,
        Float64SpecialFillValue,
    )
    from zarr_metadata.v3.data_type.int8 import INT8_DTYPE_NAME, Int8DTypeName, Int8FillValue
    from zarr_metadata.v3.data_type.int16 import INT16_DTYPE_NAME, Int16DTypeName, Int16FillValue
    from zarr_metadata.v3.data_type.int32 import INT32_DTYPE_NAME, Int32DTypeName, Int32FillValue
    from zarr_metadata.v3.data_type.int64 import INT64_DTYPE_NAME, Int64DTypeName, Int64FillValue
    from zarr_metadata.v3.data_type.numpy_datetime64 import (
        NUMPY_DATETIME64_DTYPE_NAME,
        NumpyDatetime64,
        NumpyDatetime64Configuration,
        NumpyDatetime64DTypeName,
        NumpyDatetime64FillValue,
    )
    from zarr_metadata.v3.data_type.numpy_timedelta64 import (
        NUMPY_TIMEDELTA64_DTYPE_NAME,
        NumpyTimedelta64,
        NumpyTimedelta64Configuration,
        NumpyTimedelta64DTypeName,
        NumpyTimedelta64FillValue,
    )
    from zarr_metadata.v3.data_type.string import (
        STRING_DTYPE_NAME,
        StringDTypeName,
        StringFillValue,
    )
    from zarr_metadata.v3.data_type.struct import (
        STRUCT_DTYPE_NAME,
        Struct,
        StructConfiguration,
        StructDTypeName,
        StructField,
        StructFillValue,
    )
    from zarr_metadata.v3.data_type.uint8 import UINT8_DTYPE_NAME, Uint8DTypeName, Uint8FillValue
    from zarr_metadata.v3.data_type.uint16 import (
        UINT16_DTYPE_NAME,
        Uint16DTypeName,
        Uint16FillValue,
    )
    from zarr_metadata.v3.data_type.uint32 import (
        UINT32_DTYPE_NAME,
        Uint32DTypeName,
        Uint32FillValue,
    )
    from zarr_metadata.v3.data_type.uint64 import (
        UINT64_DTYPE_NAME,
        Uint64DTypeName,
        Uint64FillValue,
    )

    _ = (
        DType,
        BOOL_DTYPE_NAME,
        BYTES_DTYPE_NAME,
        BoolDTypeName,
        BoolFillValue,
        BytesDTypeName,
        BytesFillValue,
        COMPLEX64_DTYPE_NAME,
        COMPLEX128_DTYPE_NAME,
        Complex64Component,
        Complex64DTypeName,
        Complex64FillValue,
        Complex128Component,
        Complex128DTypeName,
        Complex128FillValue,
        FLOAT16_DTYPE_NAME,
        FLOAT32_DTYPE_NAME,
        FLOAT64_DTYPE_NAME,
        Float16DTypeName,
        Float16FillValue,
        Float16SpecialFillValue,
        Float32DTypeName,
        Float32FillValue,
        Float32SpecialFillValue,
        Float64DTypeName,
        Float64FillValue,
        Float64SpecialFillValue,
        INT8_DTYPE_NAME,
        INT16_DTYPE_NAME,
        INT32_DTYPE_NAME,
        INT64_DTYPE_NAME,
        Int8DTypeName,
        Int8FillValue,
        Int16DTypeName,
        Int16FillValue,
        Int32DTypeName,
        Int32FillValue,
        Int64DTypeName,
        Int64FillValue,
        NUMPY_DATETIME64_DTYPE_NAME,
        NUMPY_TIMEDELTA64_DTYPE_NAME,
        NumpyDatetime64,
        NumpyDatetime64Configuration,
        NumpyDatetime64DTypeName,
        NumpyDatetime64FillValue,
        NumpyTimedelta64,
        NumpyTimedelta64Configuration,
        NumpyTimedelta64DTypeName,
        NumpyTimedelta64FillValue,
        STRING_DTYPE_NAME,
        STRUCT_DTYPE_NAME,
        StringDTypeName,
        StringFillValue,
        Struct,
        StructConfiguration,
        StructDTypeName,
        StructField,
        StructFillValue,
        UINT8_DTYPE_NAME,
        UINT16_DTYPE_NAME,
        UINT32_DTYPE_NAME,
        UINT64_DTYPE_NAME,
        Uint8DTypeName,
        Uint8FillValue,
        Uint16DTypeName,
        Uint16FillValue,
        Uint32DTypeName,
        Uint32FillValue,
        Uint64DTypeName,
        Uint64FillValue,
    )
