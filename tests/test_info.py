import textwrap

import pytest

from zarr.codecs.bytes import BytesCodec
from zarr.core._info import ArrayInfo, GroupInfo, human_readable_size
from zarr.core.common import ZarrFormat
from zarr.core.dtype.npy.int import Int32

ZARR_FORMATS = [2, 3]


@pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
def test_group_info_repr(zarr_format: ZarrFormat) -> None:
    info = GroupInfo(
        _name="a", _store_type="MemoryStore", _read_only=False, _zarr_format=zarr_format
    )
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
        _name="a",
        _store_type="MemoryStore",
        _zarr_format=zarr_format,
        _read_only=False,
        _count_arrays=10,
        _count_groups=4,
        _count_members=14,
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
def test_array_info(zarr_format: ZarrFormat) -> None:
    info = ArrayInfo(
        _zarr_format=zarr_format,
        _data_type=Int32(),
        _fill_value=0,
        _shape=(100, 100),
        _chunk_shape=(10, 100),
        _order="C",
        _read_only=True,
        _store_type="MemoryStore",
        _serializer=BytesCodec(),
    )
    result = repr(info)
    assert result == textwrap.dedent(f"""\
        Type               : Array
        Zarr format        : {zarr_format}
        Data type          : Int32(endianness='little')
        Fill value         : 0
        Shape              : (100, 100)
        Chunk shape        : (10, 100)
        Order              : C
        Read-only          : True
        Store type         : MemoryStore
        Filters            : ()
        Serializer         : BytesCodec(endian=<Endian.little: 'little'>)
        Compressors        : ()""")


@pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
@pytest.mark.parametrize("bytes_things", [(1_000_000, "976.6K", 500_000, "488.3K", "2.0", 5)])
def test_array_info_complete(
    zarr_format: ZarrFormat, bytes_things: tuple[int, str, int, str, str, int]
) -> None:
    (
        count_bytes,
        count_bytes_formatted,
        count_bytes_stored,
        count_bytes_stored_formatted,
        storage_ratio_formatted,
        count_chunks_initialized,
    ) = bytes_things
    info = ArrayInfo(
        _zarr_format=zarr_format,
        _data_type=Int32(),
        _fill_value=0,
        _shape=(100, 100),
        _chunk_shape=(10, 100),
        _order="C",
        _read_only=True,
        _store_type="MemoryStore",
        _serializer=BytesCodec(),
        _count_bytes=count_bytes,
        _count_bytes_stored=count_bytes_stored,
        _count_chunks_initialized=count_chunks_initialized,
    )
    result = repr(info)
    assert result == textwrap.dedent(f"""\
        Type               : Array
        Zarr format        : {zarr_format}
        Data type          : Int32(endianness='little')
        Fill value         : 0
        Shape              : (100, 100)
        Chunk shape        : (10, 100)
        Order              : C
        Read-only          : True
        Store type         : MemoryStore
        Filters            : ()
        Serializer         : BytesCodec(endian=<Endian.little: 'little'>)
        Compressors        : ()
        No. bytes          : {count_bytes} ({count_bytes_formatted})
        No. bytes stored   : {count_bytes_stored} ({count_bytes_stored_formatted})
        Storage ratio      : {storage_ratio_formatted}
        Chunks Initialized : 5""")


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (1, "1"),
        (2**10, "1.0K"),
        (2**20, "1.0M"),
        (2**30, "1.0G"),
        (2**40, "1.0T"),
        (2**50, "1.0P"),
    ],
)
def test_human_readable_size(size: int, expected: str) -> None:
    result = human_readable_size(size)
    assert result == expected
