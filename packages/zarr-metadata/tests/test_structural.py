"""
Sample-dict construction tests for zarr-metadata TypedDicts.

These don't validate at runtime (TypedDicts have no runtime shape check),
but they let pyright in CI catch shape mismatches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zarr_metadata.codec.blosc import BloscCodec, BloscCodecConfiguration
    from zarr_metadata.codec.bytes import BytesCodec, BytesCodecConfiguration
    from zarr_metadata.codec.crc32c import Crc32cCodec
    from zarr_metadata.codec.gzip import GzipCodec, GzipCodecConfiguration
    from zarr_metadata.codec.sharding import ShardingCodec, ShardingCodecConfiguration
    from zarr_metadata.codec.transpose import TransposeCodec, TransposeCodecConfiguration
    from zarr_metadata.codec.zstd import ZstdCodec, ZstdCodecConfiguration
    from zarr_metadata.dtype.bytes import FixedLengthBytesConfig
    from zarr_metadata.dtype.string import LengthBytesConfig
    from zarr_metadata.dtype.time import TimeConfig
    from zarr_metadata.v2.array import ArrayMetadataV2
    from zarr_metadata.v2.codec import NumcodecsConfig
    from zarr_metadata.v2.group import GroupMetadataV2
    from zarr_metadata.v3.array import ArrayMetadataV3, RegularChunkGrid
    from zarr_metadata.v3.consolidated import ConsolidatedMetadataV3
    from zarr_metadata.v3.group import GroupMetadataV3


def test_array_metadata_v3_minimal() -> None:
    meta: ArrayMetadataV3 = {
        "zarr_format": 3,
        "node_type": "array",
        "data_type": "float32",
        "shape": (100, 100),
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [10, 10]}},
        "chunk_key_encoding": {"name": "default"},
        "fill_value": 0,
        "codecs": ({"name": "bytes", "configuration": {"endian": "little"}},),
    }
    assert meta["zarr_format"] == 3


def test_group_metadata_v3_minimal() -> None:
    meta: GroupMetadataV3 = {
        "zarr_format": 3,
        "node_type": "group",
    }
    assert meta["zarr_format"] == 3


def test_consolidated_metadata_v3_minimal() -> None:
    cm: ConsolidatedMetadataV3 = {
        "kind": "inline",
        "must_understand": False,
        "metadata": {},
    }
    assert cm["kind"] == "inline"


def test_array_metadata_v2_simple_dtype() -> None:
    meta: ArrayMetadataV2 = {
        "zarr_format": 2,
        "shape": (100, 100),
        "chunks": (10, 10),
        "dtype": "<f4",
        "compressor": None,
        "fill_value": 0,
        "order": "C",
        "filters": None,
    }
    assert meta["dtype"] == "<f4"


def test_array_metadata_v2_structured_dtype() -> None:
    meta: ArrayMetadataV2 = {
        "zarr_format": 2,
        "shape": (100,),
        "chunks": (10,),
        "dtype": (
            {"fieldname": "a", "datatype": "<i4"},
            {"fieldname": "b", "datatype": "<f8", "shape": (3,)},
        ),
        "compressor": None,
        "fill_value": 0,
        "order": "C",
        "filters": None,
    }
    assert isinstance(meta["dtype"], tuple)


def test_group_metadata_v2_minimal() -> None:
    meta: GroupMetadataV2 = {"zarr_format": 2}
    assert meta["zarr_format"] == 2


def test_regular_chunk_grid_metadata() -> None:
    grid: RegularChunkGrid = {
        "name": "regular",
        "configuration": {"chunk_shape": (10, 10)},
    }
    assert grid["name"] == "regular"


def test_blosc_config_v1() -> None:
    cfg: BloscCodecConfiguration = {
        "cname": "zstd",
        "clevel": 5,
        "shuffle": "shuffle",
        "blocksize": 0,
        "typesize": 4,
    }
    assert cfg["cname"] == "zstd"


def test_length_bytes_config() -> None:
    cfg: LengthBytesConfig = {"length_bytes": 16}
    assert cfg["length_bytes"] == 16


def test_fixed_length_bytes_config() -> None:
    cfg: FixedLengthBytesConfig = {"length_bytes": 16}
    assert cfg["length_bytes"] == 16


def test_time_config() -> None:
    cfg: TimeConfig = {"unit": "ns", "scale_factor": 1}
    assert cfg["unit"] == "ns"


def test_numcodecs_config_minimal() -> None:
    cfg: NumcodecsConfig = {"id": "zstd"}
    assert cfg["id"] == "zstd"


def test_array_metadata_v2_with_compressor_and_filters() -> None:
    compressor: NumcodecsConfig = {"id": "zstd"}
    filter0: NumcodecsConfig = {"id": "delta"}
    meta: ArrayMetadataV2 = {
        "zarr_format": 2,
        "shape": (100,),
        "chunks": (10,),
        "dtype": "<f4",
        "compressor": compressor,
        "fill_value": 0,
        "order": "C",
        "filters": (filter0,),
    }
    compressor_val = meta["compressor"]
    assert compressor_val is not None
    assert compressor_val["id"] == "zstd"


def test_bytes_codec_config() -> None:
    cfg: BytesCodecConfiguration = {"endian": "little"}
    assert cfg["endian"] == "little"


def test_bytes_codec_config_no_endian() -> None:
    cfg: BytesCodecConfiguration = {}
    assert cfg == {}


def test_gzip_codec_config() -> None:
    cfg: GzipCodecConfiguration = {"level": 5}
    assert cfg["level"] == 5


def test_zstd_codec_config() -> None:
    cfg: ZstdCodecConfiguration = {"level": 3, "checksum": False}
    assert cfg["level"] == 3


def test_transpose_codec_config() -> None:
    cfg: TransposeCodecConfiguration = {"order": (1, 0, 2)}
    assert cfg["order"] == (1, 0, 2)


def test_sharding_codec_config() -> None:
    cfg: ShardingCodecConfiguration = {
        "chunk_shape": (16, 16),
        "codecs": ({"name": "bytes", "configuration": {"endian": "little"}},),
        "index_codecs": (
            {"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "crc32c"},
        ),
        "index_location": "end",
    }
    assert cfg["chunk_shape"] == (16, 16)


def test_bytes_codec_metadata() -> None:
    codec: BytesCodec = {"name": "bytes", "configuration": {"endian": "little"}}
    assert codec["name"] == "bytes"


def test_gzip_codec_metadata() -> None:
    codec: GzipCodec = {"name": "gzip", "configuration": {"level": 5}}
    assert codec["name"] == "gzip"


def test_zstd_codec_metadata() -> None:
    codec: ZstdCodec = {
        "name": "zstd",
        "configuration": {"level": 3, "checksum": False},
    }
    assert codec["name"] == "zstd"


def test_transpose_codec_metadata() -> None:
    codec: TransposeCodec = {"name": "transpose", "configuration": {"order": (1, 0)}}
    assert codec["name"] == "transpose"


def test_sharding_codec_metadata() -> None:
    codec: ShardingCodec = {
        "name": "sharding_indexed",
        "configuration": {
            "chunk_shape": (16, 16),
            "codecs": ({"name": "bytes", "configuration": {"endian": "little"}},),
            "index_codecs": ({"name": "crc32c"},),
        },
    }
    assert codec["name"] == "sharding_indexed"


def test_crc32c_codec_metadata() -> None:
    codec: Crc32cCodec = {"name": "crc32c"}
    assert codec["name"] == "crc32c"


def test_blosc_codec_metadata() -> None:
    codec: BloscCodec = {
        "name": "blosc",
        "configuration": {
            "cname": "zstd",
            "clevel": 5,
            "shuffle": "shuffle",
            "blocksize": 0,
            "typesize": 4,
        },
    }
    assert codec["name"] == "blosc"


def test_codec_name_constants() -> None:
    """Final constants carry the same string values as the Literal types."""
    from zarr_metadata.codec.blosc import BLOSC_CODEC_NAME
    from zarr_metadata.codec.bytes import BYTES_CODEC_NAME
    from zarr_metadata.codec.crc32c import CRC32C_CODEC_NAME
    from zarr_metadata.codec.gzip import GZIP_CODEC_NAME
    from zarr_metadata.codec.sharding import SHARDING_CODEC_NAME
    from zarr_metadata.codec.transpose import TRANSPOSE_CODEC_NAME
    from zarr_metadata.codec.zstd import ZSTD_CODEC_NAME

    assert BLOSC_CODEC_NAME == "blosc"
    assert BYTES_CODEC_NAME == "bytes"
    assert CRC32C_CODEC_NAME == "crc32c"
    assert GZIP_CODEC_NAME == "gzip"
    assert SHARDING_CODEC_NAME == "sharding_indexed"
    assert TRANSPOSE_CODEC_NAME == "transpose"
    assert ZSTD_CODEC_NAME == "zstd"


def test_blosc_enum_value_constants() -> None:
    """Blosc shuffle and cname constants can be used as codec config values."""
    from zarr_metadata.codec.blosc import (
        BLOSC_CNAME_ZSTD,
        BLOSC_SHUFFLE_BITSHUFFLE,
    )

    cfg: BloscCodecConfiguration = {
        "cname": BLOSC_CNAME_ZSTD,
        "clevel": 5,
        "shuffle": BLOSC_SHUFFLE_BITSHUFFLE,
        "blocksize": 0,
        "typesize": 4,
    }
    assert cfg["cname"] == "zstd"
    assert cfg["shuffle"] == "bitshuffle"


def test_bytes_endian_constants() -> None:
    from zarr_metadata.codec.bytes import BYTES_ENDIAN_BIG, BYTES_ENDIAN_LITTLE

    cfg_little: BytesCodecConfiguration = {"endian": BYTES_ENDIAN_LITTLE}
    cfg_big: BytesCodecConfiguration = {"endian": BYTES_ENDIAN_BIG}
    assert cfg_little["endian"] == "little"
    assert cfg_big["endian"] == "big"


def test_sharding_index_location_constants() -> None:
    from zarr_metadata.codec.sharding import (
        SHARDING_INDEX_LOCATION_END,
        SHARDING_INDEX_LOCATION_START,
    )

    cfg_end: ShardingCodecConfiguration = {
        "chunk_shape": (16, 16),
        "codecs": ({"name": "bytes", "configuration": {"endian": "little"}},),
        "index_codecs": ({"name": "crc32c"},),
        "index_location": SHARDING_INDEX_LOCATION_END,
    }
    cfg_start: ShardingCodecConfiguration = {
        "chunk_shape": (16, 16),
        "codecs": ({"name": "bytes", "configuration": {"endian": "little"}},),
        "index_codecs": ({"name": "crc32c"},),
        "index_location": SHARDING_INDEX_LOCATION_START,
    }
    assert cfg_end["index_location"] == "end"
    assert cfg_start["index_location"] == "start"
