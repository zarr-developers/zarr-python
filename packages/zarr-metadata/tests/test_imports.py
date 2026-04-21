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
    from zarr_metadata.dtype.bool import BOOL_DTYPE_NAME, BoolDTypeName, BoolFillValue
    from zarr_metadata.dtype.bytes import BYTES_DTYPE_NAME, BytesDTypeName, BytesFillValue
    from zarr_metadata.dtype.complex64 import (
        COMPLEX64_DTYPE_NAME,
        Complex64Component,
        Complex64DTypeName,
        Complex64FillValue,
    )
    from zarr_metadata.dtype.complex128 import (
        COMPLEX128_DTYPE_NAME,
        Complex128Component,
        Complex128DTypeName,
        Complex128FillValue,
    )
    from zarr_metadata.dtype.float16 import (
        FLOAT16_DTYPE_NAME,
        Float16DTypeName,
        Float16FillValue,
        Float16SpecialFillValue,
    )
    from zarr_metadata.dtype.float32 import (
        FLOAT32_DTYPE_NAME,
        Float32DTypeName,
        Float32FillValue,
        Float32SpecialFillValue,
    )
    from zarr_metadata.dtype.float64 import (
        FLOAT64_DTYPE_NAME,
        Float64DTypeName,
        Float64FillValue,
        Float64SpecialFillValue,
    )
    from zarr_metadata.dtype.int8 import INT8_DTYPE_NAME, Int8DTypeName, Int8FillValue
    from zarr_metadata.dtype.int16 import INT16_DTYPE_NAME, Int16DTypeName, Int16FillValue
    from zarr_metadata.dtype.int32 import INT32_DTYPE_NAME, Int32DTypeName, Int32FillValue
    from zarr_metadata.dtype.int64 import INT64_DTYPE_NAME, Int64DTypeName, Int64FillValue
    from zarr_metadata.dtype.numpy_datetime64 import (
        NUMPY_DATETIME64_DTYPE_NAME,
        NumpyDatetime64,
        NumpyDatetime64Configuration,
        NumpyDatetime64DTypeName,
        NumpyDatetime64FillValue,
    )
    from zarr_metadata.dtype.numpy_timedelta64 import (
        NUMPY_TIMEDELTA64_DTYPE_NAME,
        NumpyTimedelta64,
        NumpyTimedelta64Configuration,
        NumpyTimedelta64DTypeName,
        NumpyTimedelta64FillValue,
    )
    from zarr_metadata.dtype.string import STRING_DTYPE_NAME, StringDTypeName, StringFillValue
    from zarr_metadata.dtype.struct import (
        STRUCT_DTYPE_NAME,
        Struct,
        StructConfiguration,
        StructDTypeName,
        StructField,
        StructFillValue,
    )
    from zarr_metadata.dtype.uint8 import UINT8_DTYPE_NAME, Uint8DTypeName, Uint8FillValue
    from zarr_metadata.dtype.uint16 import UINT16_DTYPE_NAME, Uint16DTypeName, Uint16FillValue
    from zarr_metadata.dtype.uint32 import UINT32_DTYPE_NAME, Uint32DTypeName, Uint32FillValue
    from zarr_metadata.dtype.uint64 import UINT64_DTYPE_NAME, Uint64DTypeName, Uint64FillValue

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
