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
        RectilinearChunkGrid,
        RectilinearChunkGridConfig,
        RectilinearDimSpec,
        RegularChunkGrid,
        RegularChunkGridConfig,
    )

    _ = (
        AllowedExtraField,
        ArrayMetadataV3,
        ConsolidatedMetadataV3,
        GroupMetadataV3,
        RectilinearChunkGrid,
        RectilinearChunkGridConfig,
        RectilinearDimSpec,
        RegularChunkGrid,
        RegularChunkGridConfig,
    )


def test_codec_imports() -> None:
    from zarr_metadata.codec import Codec
    from zarr_metadata.codec.blosc import (
        BloscCName,
        BloscCodec,
        BloscCodecConfiguration,
        BloscCodecName,
        BloscShuffle,
    )
    from zarr_metadata.codec.bytes import (
        BytesCodec,
        BytesCodecConfiguration,
        BytesCodecName,
    )
    from zarr_metadata.codec.crc32c import Crc32cCodec, Crc32cCodecName
    from zarr_metadata.codec.gzip import GzipCodec, GzipCodecConfiguration, GzipCodecName
    from zarr_metadata.codec.sharding import (
        ShardingCodec,
        ShardingCodecConfiguration,
        ShardingCodecName,
    )
    from zarr_metadata.codec.transpose import (
        TransposeCodec,
        TransposeCodecConfiguration,
        TransposeCodecName,
    )
    from zarr_metadata.codec.zstd import ZstdCodec, ZstdCodecConfiguration, ZstdCodecName

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
    from zarr_metadata.dtype import DType
    from zarr_metadata.dtype.bytes import (
        BYTES_DTYPE_NAME,
        NULL_TERMINATED_BYTES_DTYPE_NAME,
        BytesDTypeName,
        FixedLengthBytesConfig,
        NullTerminatedBytes,
        NullTerminatedBytesDTypeName,
    )
    from zarr_metadata.dtype.primitive import (
        BOOL_DTYPE_NAME,
        COMPLEX64_DTYPE_NAME,
        COMPLEX128_DTYPE_NAME,
        FLOAT16_DTYPE_NAME,
        FLOAT32_DTYPE_NAME,
        FLOAT64_DTYPE_NAME,
        INT8_DTYPE_NAME,
        INT16_DTYPE_NAME,
        INT32_DTYPE_NAME,
        INT64_DTYPE_NAME,
        UINT8_DTYPE_NAME,
        UINT16_DTYPE_NAME,
        UINT32_DTYPE_NAME,
        UINT64_DTYPE_NAME,
        PrimitiveDTypeName,
    )
    from zarr_metadata.dtype.string import (
        FIXED_LENGTH_UTF32_DTYPE_NAME,
        STRING_DTYPE_NAME,
        FixedLengthUtf32,
        FixedLengthUtf32DTypeName,
        LengthBytesConfig,
        StringDTypeName,
    )
    from zarr_metadata.dtype.struct import (
        STRUCT_DTYPE_NAME,
        Struct,
        StructConfig,
        StructDTypeName,
        StructField,
    )
    from zarr_metadata.dtype.time import (
        NUMPY_DATETIME64_DTYPE_NAME,
        NUMPY_TIMEDELTA64_DTYPE_NAME,
        DateTimeUnit,
        NumpyDatetime64,
        NumpyDatetime64DTypeName,
        NumpyTimedelta64,
        NumpyTimedelta64DTypeName,
        TimeConfig,
    )

    _ = (
        DType,
        BOOL_DTYPE_NAME,
        BYTES_DTYPE_NAME,
        BytesDTypeName,
        COMPLEX64_DTYPE_NAME,
        COMPLEX128_DTYPE_NAME,
        DateTimeUnit,
        FIXED_LENGTH_UTF32_DTYPE_NAME,
        FLOAT16_DTYPE_NAME,
        FLOAT32_DTYPE_NAME,
        FLOAT64_DTYPE_NAME,
        FixedLengthBytesConfig,
        FixedLengthUtf32,
        FixedLengthUtf32DTypeName,
        INT8_DTYPE_NAME,
        INT16_DTYPE_NAME,
        INT32_DTYPE_NAME,
        INT64_DTYPE_NAME,
        LengthBytesConfig,
        NULL_TERMINATED_BYTES_DTYPE_NAME,
        NUMPY_DATETIME64_DTYPE_NAME,
        NUMPY_TIMEDELTA64_DTYPE_NAME,
        NullTerminatedBytes,
        NullTerminatedBytesDTypeName,
        NumpyDatetime64,
        NumpyDatetime64DTypeName,
        NumpyTimedelta64,
        NumpyTimedelta64DTypeName,
        PrimitiveDTypeName,
        STRING_DTYPE_NAME,
        STRUCT_DTYPE_NAME,
        StringDTypeName,
        Struct,
        StructConfig,
        StructDTypeName,
        StructField,
        TimeConfig,
        UINT8_DTYPE_NAME,
        UINT16_DTYPE_NAME,
        UINT32_DTYPE_NAME,
        UINT64_DTYPE_NAME,
    )
