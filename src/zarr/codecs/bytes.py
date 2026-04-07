from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING

from zarr.abc.codec import ArrayBytesCodec, PreparedWrite, SupportsChunkCodec
from zarr.core.buffer import Buffer, NDBuffer
from zarr.core.common import JSON, parse_enum, parse_named_configuration
from zarr.core.dtype.common import HasEndianness

if TYPE_CHECKING:
    from typing import Any, Self

    from zarr.core.array_spec import ArraySpec
    from zarr.core.indexing import SelectorTuple


class Endian(Enum):
    """
    Enum for endian type used by bytes codec.
    """

    big = "big"
    little = "little"


default_system_endian = Endian(sys.byteorder)


@dataclass(frozen=True)
class BytesCodec(ArrayBytesCodec):
    """bytes codec"""

    is_fixed_size = True

    endian: Endian | None

    def __init__(self, *, endian: Endian | str | None = default_system_endian) -> None:
        endian_parsed = None if endian is None else parse_enum(endian, Endian)

        object.__setattr__(self, "endian", endian_parsed)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(
            data, "bytes", require_configuration=False
        )
        configuration_parsed = configuration_parsed or {}
        return cls(**configuration_parsed)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, JSON]:
        if self.endian is None:
            return {"name": "bytes"}
        else:
            return {"name": "bytes", "configuration": {"endian": self.endian.value}}

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        if not isinstance(array_spec.dtype, HasEndianness):
            if self.endian is not None:
                return replace(self, endian=None)
        elif self.endian is None:
            raise ValueError(
                "The `endian` configuration needs to be specified for multi-byte data types."
            )
        return self

    def _decode_sync(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        # TODO: remove endianness enum in favor of literal union
        endian_str = self.endian.value if self.endian is not None else None
        if isinstance(chunk_spec.dtype, HasEndianness):
            dtype = replace(chunk_spec.dtype, endianness=endian_str).to_native_dtype()  # type: ignore[call-arg]
        else:
            dtype = chunk_spec.dtype.to_native_dtype()
        as_array_like = chunk_bytes.as_array_like()
        chunk_array = chunk_spec.prototype.nd_buffer.from_ndarray_like(
            as_array_like.view(dtype=dtype)  # type: ignore[attr-defined]
        )

        # ensure correct chunk shape
        if chunk_array.shape != chunk_spec.shape:
            chunk_array = chunk_array.reshape(
                chunk_spec.shape,
            )
        return chunk_array

    async def _decode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        return self._decode_sync(chunk_bytes, chunk_spec)

    def _encode_sync(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> Buffer | None:
        assert isinstance(chunk_array, NDBuffer)
        if (
            chunk_array.dtype.itemsize > 1
            and self.endian is not None
            and self.endian != chunk_array.byteorder
        ):
            # type-ignore is a numpy bug
            # see https://github.com/numpy/numpy/issues/26473
            new_dtype = chunk_array.dtype.newbyteorder(self.endian.name)  # type: ignore[arg-type]
            chunk_array = chunk_array.astype(new_dtype)

        nd_array = chunk_array.as_ndarray_like()
        # Flatten the nd-array (only copy if needed) and reinterpret as bytes
        nd_array = nd_array.ravel().view(dtype="B")
        return chunk_spec.prototype.buffer.from_array_like(nd_array)

    async def _encode_single(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> Buffer | None:
        return self._encode_sync(chunk_array, chunk_spec)

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        return input_byte_length

    # -- SupportsChunkPacking --

    @property
    def inner_codec_chain(self) -> SupportsChunkCodec | None:
        """Returns `None` — the pipeline should use its own codec chain."""
        return None

    def unpack_chunks(
        self,
        raw: Buffer | None,
        chunk_spec: ArraySpec,
    ) -> dict[tuple[int, ...], Buffer | None]:
        """Single chunk keyed at `(0,)`."""
        return {(0,): raw}

    def pack_chunks(
        self,
        chunk_dict: dict[tuple[int, ...], Buffer | None],
        chunk_spec: ArraySpec,
    ) -> Buffer | None:
        """Return the single chunk's bytes."""
        return chunk_dict.get((0,))

    def prepare_read_sync(
        self,
        byte_getter: Any,
        chunk_selection: SelectorTuple,
        codec_chain: SupportsChunkCodec,
    ) -> NDBuffer | None:
        """Fetch, decode, and return the selected region synchronously."""
        raw = byte_getter.get_sync(prototype=codec_chain.array_spec.prototype)
        if raw is None:
            return None
        chunk_array = codec_chain.decode_chunk(raw)
        return chunk_array[chunk_selection]

    def prepare_write_sync(
        self,
        byte_setter: Any,
        codec_chain: SupportsChunkCodec,
        chunk_selection: SelectorTuple,
        out_selection: SelectorTuple,
        replace: bool,
    ) -> PreparedWrite:
        """Fetch existing data if needed, unpack, return `PreparedWrite`."""
        from zarr.core.indexing import ChunkProjection

        existing: Buffer | None = None
        if not replace:
            existing = byte_setter.get_sync(prototype=codec_chain.array_spec.prototype)
        chunk_dict = self.unpack_chunks(existing, codec_chain.array_spec)
        indexer = [ChunkProjection((0,), chunk_selection, out_selection, replace)]  # type: ignore[arg-type]
        return PreparedWrite(chunk_dict=chunk_dict, indexer=indexer)

    def finalize_write_sync(
        self,
        prepared: PreparedWrite,
        chunk_spec: ArraySpec,
        byte_setter: Any,
    ) -> None:
        """Pack and write to store, or delete if empty."""
        blob = self.pack_chunks(prepared.chunk_dict, chunk_spec)
        if blob is None:
            byte_setter.delete_sync()
        else:
            byte_setter.set_sync(blob)

    async def prepare_read(
        self,
        byte_getter: Any,
        chunk_selection: SelectorTuple,
        codec_chain: SupportsChunkCodec,
    ) -> NDBuffer | None:
        """Async variant of `prepare_read_sync`."""
        raw = await byte_getter.get(prototype=codec_chain.array_spec.prototype)
        if raw is None:
            return None
        chunk_array = codec_chain.decode_chunk(raw)
        return chunk_array[chunk_selection]

    async def prepare_write(
        self,
        byte_setter: Any,
        codec_chain: SupportsChunkCodec,
        chunk_selection: SelectorTuple,
        out_selection: SelectorTuple,
        replace: bool,
    ) -> PreparedWrite:
        """Async variant of `prepare_write_sync`."""
        from zarr.core.indexing import ChunkProjection

        existing: Buffer | None = None
        if not replace:
            existing = await byte_setter.get(prototype=codec_chain.array_spec.prototype)
        chunk_dict = self.unpack_chunks(existing, codec_chain.array_spec)
        indexer = [ChunkProjection((0,), chunk_selection, out_selection, replace)]  # type: ignore[arg-type]
        return PreparedWrite(chunk_dict=chunk_dict, indexer=indexer)

    async def finalize_write(
        self,
        prepared: PreparedWrite,
        chunk_spec: ArraySpec,
        byte_setter: Any,
    ) -> None:
        """Async variant of `finalize_write_sync`."""
        blob = self.pack_chunks(prepared.chunk_dict, chunk_spec)
        if blob is None:
            await byte_setter.delete()
        else:
            await byte_setter.set(blob)
