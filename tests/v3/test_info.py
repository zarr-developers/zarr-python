import textwrap
from typing import Literal

import pytest

from zarr.core.common import ZarrFormat
from zarr._info import ArrayInfo, GroupInfo


ZARR_FORMATS = [2, 3]


@pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
def test_group_info_repr(zarr_format: ZarrFormat) -> None:
    info = GroupInfo(name="a", store_type="MemoryStore", read_only=False, zarr_format=zarr_format)
    result = repr(info)
    expected = textwrap.dedent(f"""\
        Name        : a
        Type        : Group
        Zarr format : {zarr_format}
        Read-only   : False
        Store type  : MemoryStore""")
    assert result == expected


@pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
def test_group_info_complete(zarr_format: ZarrFormat) -> None:
    info = GroupInfo(
        name="a",
        store_type="MemoryStore",
        zarr_format=zarr_format,
        read_only=False,
        count_arrays=10,
        count_groups=4,
        count_members=14,
    )
    result = repr(info)
    expected = textwrap.dedent(f"""\
        Name        : a
        Type        : Group
        Zarr format : {zarr_format}
        Read-only   : False
        Store type  : MemoryStore
        No. members : 14
        No. arrays  : 10
        No. groups  : 4""")
    assert result == expected


@pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
def test_array_info(zarr_format: ZarrFormat):
    info = ArrayInfo(
        zarr_format=zarr_format,
        data_type="int32",
        shape=(100, 100),
        chunk_shape=(10, 100),
        order="C",
        read_only=True,
        store_type="MemoryStore",
        codecs=["BytesCodec(endian=<Endian.little: 'little'>"],
    )
    result = repr(info)
    assert result == textwrap.dedent(f"""\
        Type               : Array
        Zarr format        : {zarr_format}
        Data type          : int32
        Shape              : (100, 100)
        Chunk shape        : (10, 100)
        Order              : C
        Read-only          : True
        Store type         : MemoryStore
        Codecs             : ["BytesCodec(endian=<Endian.little: 'little'>"]""")


@pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
@pytest.mark.parametrize("bytes_things", [(1_000_000, "976.6K", 500_000, "5", "2.0", 5)])
def test_array_info_complete(
    zarr_format: ZarrFormat, bytes_things: tuple[int, str, int, str, str, int]
):
    (
        count_bytes,
        count_bytes_formatted,
        count_bytes_stored,
        count_bytes_stored_formatted,
        storage_ratio_formatted,
        count_chunks_initialized,
    ) = bytes_things
    info = ArrayInfo(
        zarr_format=zarr_format,
        data_type="int32",
        shape=(100, 100),
        chunk_shape=(10, 100),
        order="C",
        read_only=True,
        store_type="MemoryStore",
        codecs=["BytesCodec(endian=<Endian.little: 'little'>"],
        count_bytes=count_bytes,
        count_bytes_stored=count_bytes_stored,
        count_chunks_initialized=count_chunks_initialized,
    )
    result = repr(info)
    assert result == textwrap.dedent(f"""\
        Type               : Array
        Zarr format        : {zarr_format}
        Data type          : int32
        Shape              : (100, 100)
        Chunk shape        : (10, 100)
        Order              : C
        Read-only          : True
        Store type         : MemoryStore
        Codecs             : ["BytesCodec(endian=<Endian.little: 'little'>"]
        No. bytes          : 1000000 (976.6K)
        No. bytes stored   : 500000
        Storage ratio      : 2.0
        Chunks Initialized : 5""")
