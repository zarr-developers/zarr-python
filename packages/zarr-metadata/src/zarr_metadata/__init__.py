from importlib.metadata import version

from zarr_metadata._common import JSONValue, NamedConfigV3
from zarr_metadata.v2.array import (
    ARRAY_DIMENSION_SEPARATOR_V2,
    ARRAY_ORDER_V2,
    ArrayDimensionSeparatorV2,
    ArrayMetadataV2,
    ArrayMetadataV2Partial,
    ArrayOrderV2,
    DataTypeMetadataV2,
    ZArrayMetadata,
)
from zarr_metadata.v2.attributes import ZAttrsMetadata
from zarr_metadata.v2.codec import CodecMetadataV2
from zarr_metadata.v2.consolidated import ConsolidatedMetadataV2
from zarr_metadata.v2.group import GroupMetadataV2, GroupMetadataV2Partial, ZGroupMetadata
from zarr_metadata.v3._common import MetadataV3
from zarr_metadata.v3.array import ArrayMetadataV3, ArrayMetadataV3Partial, ExtensionFieldV3
from zarr_metadata.v3.chunk_grid.rectilinear import (
    RECTILINEAR_CHUNK_GRID_NAME,
    RectilinearChunkGridMetadata,
    RectilinearChunkGridName,
)
from zarr_metadata.v3.chunk_grid.regular import (
    REGULAR_CHUNK_GRID_NAME,
    RegularChunkGridMetadata,
    RegularChunkGridName,
)
from zarr_metadata.v3.chunk_key_encoding.default import (
    DEFAULT_CHUNK_KEY_ENCODING_NAME,
    DEFAULT_CHUNK_KEY_ENCODING_SEPARATOR,
    DefaultChunkKeyEncodingMetadata,
    DefaultChunkKeyEncodingName,
    DefaultChunkKeyEncodingSeparator,
)
from zarr_metadata.v3.chunk_key_encoding.v2 import (
    V2_CHUNK_KEY_ENCODING_NAME,
    V2_CHUNK_KEY_ENCODING_SEPARATOR,
    V2ChunkKeyEncodingMetadata,
    V2ChunkKeyEncodingName,
    V2ChunkKeyEncodingSeparator,
)
from zarr_metadata.v3.codec.blosc import (
    BLOSC_CNAME,
    BLOSC_CODEC_NAME,
    BLOSC_SHUFFLE,
    BloscCName,
    BloscCodecMetadata,
    BloscCodecName,
    BloscShuffle,
)
from zarr_metadata.v3.codec.bytes import (
    BYTES_CODEC_NAME,
    ENDIANNESS,
    BytesCodecMetadata,
    BytesCodecName,
    Endianness,
)
from zarr_metadata.v3.codec.cast_value import (
    CAST_OUT_OF_RANGE_MODE,
    CAST_ROUNDING_MODE,
    CAST_VALUE_CODEC_NAME,
    CastOutOfRangeMode,
    CastRoundingMode,
    CastValueCodecMetadata,
    CastValueCodecName,
)
from zarr_metadata.v3.codec.crc32c import CRC32C_CODEC_NAME, Crc32cCodecMetadata, Crc32cCodecName
from zarr_metadata.v3.codec.gzip import GZIP_CODEC_NAME, GzipCodecMetadata, GzipCodecName
from zarr_metadata.v3.codec.scale_offset import (
    SCALE_OFFSET_CODEC_NAME,
    ScaleOffsetCodecMetadata,
    ScaleOffsetCodecName,
)
from zarr_metadata.v3.codec.sharding_indexed import (
    SHARDING_INDEX_LOCATION,
    SHARDING_INDEXED_CODEC_NAME,
    ShardingIndexedCodecMetadata,
    ShardingIndexedCodecName,
    ShardingIndexLocation,
)
from zarr_metadata.v3.codec.transpose import (
    TRANSPOSE_CODEC_NAME,
    TransposeCodecMetadata,
    TransposeCodecName,
)
from zarr_metadata.v3.codec.zstd import ZSTD_CODEC_NAME, ZstdCodecMetadata, ZstdCodecName
from zarr_metadata.v3.consolidated import ConsolidatedMetadataV3
from zarr_metadata.v3.data_type.bool import (
    BOOL_DATA_TYPE_NAME,
    BoolDataTypeName,
    BoolFillValue,
)
from zarr_metadata.v3.data_type.bytes import (
    BYTES_DATA_TYPE_NAME,
    BytesDataTypeName,
    BytesFillValue,
)
from zarr_metadata.v3.data_type.complex64 import (
    COMPLEX64_DATA_TYPE_NAME,
    Complex64DataTypeName,
    Complex64FillValue,
)
from zarr_metadata.v3.data_type.complex128 import (
    COMPLEX128_DATA_TYPE_NAME,
    Complex128DataTypeName,
    Complex128FillValue,
)
from zarr_metadata.v3.data_type.float16 import (
    FLOAT16_DATA_TYPE_NAME,
    Float16DataTypeName,
    Float16FillValue,
)
from zarr_metadata.v3.data_type.float32 import (
    FLOAT32_DATA_TYPE_NAME,
    Float32DataTypeName,
    Float32FillValue,
)
from zarr_metadata.v3.data_type.float64 import (
    FLOAT64_DATA_TYPE_NAME,
    Float64DataTypeName,
    Float64FillValue,
)
from zarr_metadata.v3.data_type.int8 import (
    INT8_DATA_TYPE_NAME,
    Int8DataTypeName,
    Int8FillValue,
)
from zarr_metadata.v3.data_type.int16 import (
    INT16_DATA_TYPE_NAME,
    Int16DataTypeName,
    Int16FillValue,
)
from zarr_metadata.v3.data_type.int32 import (
    INT32_DATA_TYPE_NAME,
    Int32DataTypeName,
    Int32FillValue,
)
from zarr_metadata.v3.data_type.int64 import (
    INT64_DATA_TYPE_NAME,
    Int64DataTypeName,
    Int64FillValue,
)
from zarr_metadata.v3.data_type.numpy_datetime64 import (
    NUMPY_DATETIME64_DATA_TYPE_NAME,
    NumpyDatetime64DataTypeName,
    NumpyDatetime64FillValue,
)
from zarr_metadata.v3.data_type.numpy_timedelta64 import (
    NUMPY_TIME_UNIT,
    NUMPY_TIMEDELTA64_DATA_TYPE_NAME,
    NumpyTimedelta64DataTypeName,
    NumpyTimedelta64FillValue,
    NumpyTimeUnit,
)
from zarr_metadata.v3.data_type.raw import RawBytesDataTypeName, RawBytesFillValue
from zarr_metadata.v3.data_type.string import (
    STRING_DATA_TYPE_NAME,
    StringDataTypeName,
    StringFillValue,
)
from zarr_metadata.v3.data_type.struct import (
    STRUCT_DATA_TYPE_NAME,
    StructDataTypeName,
    StructFillValue,
)
from zarr_metadata.v3.data_type.uint8 import (
    UINT8_DATA_TYPE_NAME,
    Uint8DataTypeName,
    Uint8FillValue,
)
from zarr_metadata.v3.data_type.uint16 import (
    UINT16_DATA_TYPE_NAME,
    Uint16DataTypeName,
    Uint16FillValue,
)
from zarr_metadata.v3.data_type.uint32 import (
    UINT32_DATA_TYPE_NAME,
    Uint32DataTypeName,
    Uint32FillValue,
)
from zarr_metadata.v3.data_type.uint64 import (
    UINT64_DATA_TYPE_NAME,
    Uint64DataTypeName,
    Uint64FillValue,
)
from zarr_metadata.v3.group import GroupMetadataV3, GroupMetadataV3Partial

__version__ = version("zarr-metadata")


__all__ = [
    "ARRAY_DIMENSION_SEPARATOR_V2",
    "ARRAY_ORDER_V2",
    "BLOSC_CNAME",
    "BLOSC_CODEC_NAME",
    "BLOSC_SHUFFLE",
    "BOOL_DATA_TYPE_NAME",
    "BYTES_CODEC_NAME",
    "BYTES_DATA_TYPE_NAME",
    "CAST_OUT_OF_RANGE_MODE",
    "CAST_ROUNDING_MODE",
    "CAST_VALUE_CODEC_NAME",
    "COMPLEX64_DATA_TYPE_NAME",
    "COMPLEX128_DATA_TYPE_NAME",
    "CRC32C_CODEC_NAME",
    "DEFAULT_CHUNK_KEY_ENCODING_NAME",
    "DEFAULT_CHUNK_KEY_ENCODING_SEPARATOR",
    "ENDIANNESS",
    "FLOAT16_DATA_TYPE_NAME",
    "FLOAT32_DATA_TYPE_NAME",
    "FLOAT64_DATA_TYPE_NAME",
    "GZIP_CODEC_NAME",
    "INT8_DATA_TYPE_NAME",
    "INT16_DATA_TYPE_NAME",
    "INT32_DATA_TYPE_NAME",
    "INT64_DATA_TYPE_NAME",
    "NUMPY_DATETIME64_DATA_TYPE_NAME",
    "NUMPY_TIMEDELTA64_DATA_TYPE_NAME",
    "NUMPY_TIME_UNIT",
    "RECTILINEAR_CHUNK_GRID_NAME",
    "REGULAR_CHUNK_GRID_NAME",
    "SCALE_OFFSET_CODEC_NAME",
    "SHARDING_INDEXED_CODEC_NAME",
    "SHARDING_INDEX_LOCATION",
    "STRING_DATA_TYPE_NAME",
    "STRUCT_DATA_TYPE_NAME",
    "TRANSPOSE_CODEC_NAME",
    "UINT8_DATA_TYPE_NAME",
    "UINT16_DATA_TYPE_NAME",
    "UINT32_DATA_TYPE_NAME",
    "UINT64_DATA_TYPE_NAME",
    "V2_CHUNK_KEY_ENCODING_NAME",
    "V2_CHUNK_KEY_ENCODING_SEPARATOR",
    "ZSTD_CODEC_NAME",
    "ArrayDimensionSeparatorV2",
    "ArrayMetadataV2",
    "ArrayMetadataV2Partial",
    "ArrayMetadataV3",
    "ArrayMetadataV3Partial",
    "ArrayOrderV2",
    "BloscCName",
    "BloscCodecMetadata",
    "BloscCodecName",
    "BloscShuffle",
    "BoolDataTypeName",
    "BoolFillValue",
    "BytesCodecMetadata",
    "BytesCodecName",
    "BytesDataTypeName",
    "BytesFillValue",
    "CastOutOfRangeMode",
    "CastRoundingMode",
    "CastValueCodecMetadata",
    "CastValueCodecName",
    "CodecMetadataV2",
    "Complex64DataTypeName",
    "Complex64FillValue",
    "Complex128DataTypeName",
    "Complex128FillValue",
    "ConsolidatedMetadataV2",
    "ConsolidatedMetadataV3",
    "Crc32cCodecMetadata",
    "Crc32cCodecName",
    "DataTypeMetadataV2",
    "DefaultChunkKeyEncodingMetadata",
    "DefaultChunkKeyEncodingName",
    "DefaultChunkKeyEncodingSeparator",
    "Endianness",
    "ExtensionFieldV3",
    "Float16DataTypeName",
    "Float16FillValue",
    "Float32DataTypeName",
    "Float32FillValue",
    "Float64DataTypeName",
    "Float64FillValue",
    "GroupMetadataV2",
    "GroupMetadataV2Partial",
    "GroupMetadataV3",
    "GroupMetadataV3Partial",
    "GzipCodecMetadata",
    "GzipCodecName",
    "Int8DataTypeName",
    "Int8FillValue",
    "Int16DataTypeName",
    "Int16FillValue",
    "Int32DataTypeName",
    "Int32FillValue",
    "Int64DataTypeName",
    "Int64FillValue",
    "JSONValue",
    "MetadataV3",
    "NamedConfigV3",
    "NumpyDatetime64DataTypeName",
    "NumpyDatetime64FillValue",
    "NumpyTimeUnit",
    "NumpyTimedelta64DataTypeName",
    "NumpyTimedelta64FillValue",
    "RawBytesDataTypeName",
    "RawBytesFillValue",
    "RectilinearChunkGridMetadata",
    "RectilinearChunkGridName",
    "RegularChunkGridMetadata",
    "RegularChunkGridName",
    "ScaleOffsetCodecMetadata",
    "ScaleOffsetCodecName",
    "ShardingIndexLocation",
    "ShardingIndexedCodecMetadata",
    "ShardingIndexedCodecName",
    "StringDataTypeName",
    "StringFillValue",
    "StructDataTypeName",
    "StructFillValue",
    "TransposeCodecMetadata",
    "TransposeCodecName",
    "Uint8DataTypeName",
    "Uint8FillValue",
    "Uint16DataTypeName",
    "Uint16FillValue",
    "Uint32DataTypeName",
    "Uint32FillValue",
    "Uint64DataTypeName",
    "Uint64FillValue",
    "V2ChunkKeyEncodingMetadata",
    "V2ChunkKeyEncodingName",
    "V2ChunkKeyEncodingSeparator",
    "ZArrayMetadata",
    "ZAttrsMetadata",
    "ZGroupMetadata",
    "ZstdCodecMetadata",
    "ZstdCodecName",
    "__version__",
]
