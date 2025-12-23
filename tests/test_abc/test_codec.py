from __future__ import annotations

from typing import TYPE_CHECKING

from zarr.abc.codec import ReadBatchInfo, WriteBatchInfo, _check_codecjson_v2
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import Buffer, BufferPrototype, default_buffer_prototype
from zarr.dtype import UInt8

if TYPE_CHECKING:
    from zarr.abc.store import ByteRequest


def test_check_codecjson_v2_valid() -> None:
    """
    Test that the _check_codecjson_v2 function works
    """
    assert _check_codecjson_v2({"id": "gzip"})
    assert not _check_codecjson_v2({"id": 10})
    assert not _check_codecjson_v2([10, 11])


def test_read_batch_info_iterable() -> None:
    class ByteGetter:
        async def get(
            self, prototype: BufferPrototype, byte_range: ByteRequest | None = None
        ) -> Buffer | None:
            pass

    byte_getter = ByteGetter()
    info = ReadBatchInfo(
        byte_operator=byte_getter,
        array_spec=ArraySpec(
            shape=(16, 16),
            dtype=UInt8(),
            fill_value=0,
            config=ArrayConfig(order="C", write_empty_chunks=True),
            prototype=default_buffer_prototype(),
        ),
        chunk_selection=(0, 0),
        out_selection=(0, 0),
        is_complete_chunk=True,
    )

    assert tuple(info) == (
        info.byte_operator,
        info.array_spec,
        info.chunk_selection,
        info.out_selection,
        info.is_complete_chunk,
    )


def test_write_batch_info_iterable() -> None:
    class ByteSetter:
        async def get(
            self, prototype: BufferPrototype, byte_range: ByteRequest | None = None
        ) -> Buffer | None:
            pass

        async def set(self, value: Buffer) -> None:
            pass

        async def delete(self) -> None:
            pass

        async def set_if_not_exists(self, default: Buffer) -> None:
            pass

    byte_setter = ByteSetter()

    info = WriteBatchInfo(
        byte_operator=byte_setter,
        array_spec=ArraySpec(
            shape=(16, 16),
            dtype=UInt8(),
            fill_value=0,
            config=ArrayConfig(order="C", write_empty_chunks=True),
            prototype=default_buffer_prototype(),
        ),
        chunk_selection=(0, 0),
        out_selection=(0, 0),
        is_complete_chunk=True,
    )
    assert tuple(info) == (
        info.byte_operator,
        info.array_spec,
        info.chunk_selection,
        info.out_selection,
        info.is_complete_chunk,
    )
