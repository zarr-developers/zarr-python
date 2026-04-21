"""
Sample-dict construction tests for zarr-metadata TypedDicts.

These don't validate at runtime (TypedDicts have no runtime shape check),
but they let pyright in CI catch shape mismatches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zarr_metadata.codec.blosc import BloscCodecConfigurationV1
    from zarr_metadata.dtype.bytes import FixedLengthBytesConfig
    from zarr_metadata.dtype.string import LengthBytesConfig
    from zarr_metadata.dtype.time import TimeConfig
    from zarr_metadata.v2.array import ArrayMetadataV2
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
    }
    assert meta["dtype"] == "<f4"


def test_array_metadata_v2_structured_dtype() -> None:
    meta: ArrayMetadataV2 = {
        "zarr_format": 2,
        "shape": (100,),
        "chunks": (10,),
        "dtype": [
            {"fieldname": "a", "datatype": "<i4"},
            {"fieldname": "b", "datatype": "<f8", "shape": [3]},
        ],
    }
    assert isinstance(meta["dtype"], list)


def test_group_metadata_v2_minimal() -> None:
    meta: GroupMetadataV2 = {"zarr_format": 2}
    assert meta["zarr_format"] == 2


def test_regular_chunk_grid_envelope() -> None:
    grid: RegularChunkGrid = {
        "name": "regular",
        "configuration": {"chunk_shape": [10, 10]},
    }
    assert grid["name"] == "regular"


def test_blosc_config_v1() -> None:
    cfg: BloscCodecConfigurationV1 = {
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
