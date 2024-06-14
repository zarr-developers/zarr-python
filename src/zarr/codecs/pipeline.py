from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import islice
from typing import TYPE_CHECKING, TypeVar
from warnings import warn

from zarr.abc.codec import (
    ArrayArrayCodec,
    ArrayBytesCodec,
    ArrayBytesCodecPartialDecodeMixin,
    ArrayBytesCodecPartialEncodeMixin,
    BytesBytesCodec,
    Codec,
    CodecPipeline,
)
from zarr.abc.store import ByteGetter, ByteSetter
from zarr.buffer import Buffer, BufferPrototype, NDBuffer
from zarr.codecs.registry import get_codec_class
from zarr.common import JSON, concurrent_map, parse_named_configuration
from zarr.config import config
from zarr.indexing import SelectorTuple, is_scalar, is_total_slice
from zarr.metadata import ArrayMetadata

if TYPE_CHECKING:
    from typing_extensions import Self

    from zarr.array_spec import ArraySpec

T = TypeVar("T")
U = TypeVar("U")


def _unzip2(iterable: Iterable[tuple[T, U]]) -> tuple[list[T], list[U]]:
    out0: list[T] = []
    out1: list[U] = []
    for item0, item1 in iterable:
        out0.append(item0)
        out1.append(item1)
    return (out0, out1)


def batched(iterable: Iterable[T], n: int) -> Iterable[tuple[T, ...]]:
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def resolve_batched(codec: Codec, chunk_specs: Iterable[ArraySpec]) -> Iterable[ArraySpec]:
    return [codec.resolve_metadata(chunk_spec) for chunk_spec in chunk_specs]


@dataclass(frozen=True)
class BatchedCodecPipeline(CodecPipeline):
    """Default codec pipeline.

    This batched codec pipeline divides the chunk batches into batches of a configurable
    batch size ("mini-batch"). Fetching, decoding, encoding and storing are performed in
    lock step for each mini-batch. Multiple mini-batches are processing concurrently.
    """

    array_array_codecs: tuple[ArrayArrayCodec, ...]
    array_bytes_codec: ArrayBytesCodec
    bytes_bytes_codecs: tuple[BytesBytesCodec, ...]
    batch_size: int

    @classmethod
    def from_dict(cls, data: Iterable[JSON | Codec], *, batch_size: int | None = None) -> Self:
        out: list[Codec] = []
        if not isinstance(data, Iterable):
            raise TypeError(f"Expected iterable, got {type(data)}")

        for c in data:
            if isinstance(
                c, ArrayArrayCodec | ArrayBytesCodec | BytesBytesCodec
            ):  # Can't use Codec here because of mypy limitation
                out.append(c)
            else:
                name_parsed, _ = parse_named_configuration(c, require_configuration=False)
                out.append(get_codec_class(name_parsed).from_dict(c))  # type: ignore[arg-type]
        return cls.from_list(out, batch_size=batch_size)

    def to_dict(self) -> JSON:
        return [c.to_dict() for c in self]

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        return type(self).from_list([c.evolve_from_array_spec(array_spec) for c in self])

    @staticmethod
    def codecs_from_list(
        codecs: list[Codec],
    ) -> tuple[tuple[ArrayArrayCodec, ...], ArrayBytesCodec, tuple[BytesBytesCodec, ...]]:
        from zarr.codecs.sharding import ShardingCodec

        if not any(isinstance(codec, ArrayBytesCodec) for codec in codecs):
            raise ValueError("Exactly one array-to-bytes codec is required.")

        prev_codec: Codec | None = None
        for codec in codecs:
            if prev_codec is not None:
                if isinstance(codec, ArrayBytesCodec) and isinstance(prev_codec, ArrayBytesCodec):
                    raise ValueError(
                        f"ArrayBytesCodec '{type(codec)}' cannot follow after ArrayBytesCodec '{type(prev_codec)}' because exactly 1 ArrayBytesCodec is allowed."
                    )
                if isinstance(codec, ArrayBytesCodec) and isinstance(prev_codec, BytesBytesCodec):
                    raise ValueError(
                        f"ArrayBytesCodec '{type(codec)}' cannot follow after BytesBytesCodec '{type(prev_codec)}'."
                    )
                if isinstance(codec, ArrayArrayCodec) and isinstance(prev_codec, ArrayBytesCodec):
                    raise ValueError(
                        f"ArrayArrayCodec '{type(codec)}' cannot follow after ArrayBytesCodec '{type(prev_codec)}'."
                    )
                if isinstance(codec, ArrayArrayCodec) and isinstance(prev_codec, BytesBytesCodec):
                    raise ValueError(
                        f"ArrayArrayCodec '{type(codec)}' cannot follow after BytesBytesCodec '{type(prev_codec)}'."
                    )
            prev_codec = codec

        if any(isinstance(codec, ShardingCodec) for codec in codecs) and len(codecs) > 1:
            warn(
                "Combining a `sharding_indexed` codec disables partial reads and "
                "writes, which may lead to inefficient performance.",
                stacklevel=3,
            )

        return (
            tuple(codec for codec in codecs if isinstance(codec, ArrayArrayCodec)),
            next(codec for codec in codecs if isinstance(codec, ArrayBytesCodec)),
            tuple(codec for codec in codecs if isinstance(codec, BytesBytesCodec)),
        )

    @classmethod
    def from_list(cls, codecs: list[Codec], *, batch_size: int | None = None) -> Self:
        array_array_codecs, array_bytes_codec, bytes_bytes_codecs = cls.codecs_from_list(codecs)

        return cls(
            array_array_codecs=array_array_codecs,
            array_bytes_codec=array_bytes_codec,
            bytes_bytes_codecs=bytes_bytes_codecs,
            batch_size=batch_size or config.get("codec_pipeline.batch_size"),
        )

    @property
    def supports_partial_decode(self) -> bool:
        """Determines whether the codec pipeline supports partial decoding.

        Currently, only codec pipelines with a single ArrayBytesCodec that supports
        partial decoding can support partial decoding. This limitation is due to the fact
        that ArrayArrayCodecs can change the slice selection leading to non-contiguous
        slices and BytesBytesCodecs can change the chunk bytes in a way that slice
        selections cannot be attributed to byte ranges anymore which renders partial
        decoding infeasible.

        This limitation may softened in the future."""
        return (len(self.array_array_codecs) + len(self.bytes_bytes_codecs)) == 0 and isinstance(
            self.array_bytes_codec, ArrayBytesCodecPartialDecodeMixin
        )

    @property
    def supports_partial_encode(self) -> bool:
        """Determines whether the codec pipeline supports partial encoding.

        Currently, only codec pipelines with a single ArrayBytesCodec that supports
        partial encoding can support partial encoding. This limitation is due to the fact
        that ArrayArrayCodecs can change the slice selection leading to non-contiguous
        slices and BytesBytesCodecs can change the chunk bytes in a way that slice
        selections cannot be attributed to byte ranges anymore which renders partial
        encoding infeasible.

        This limitation may softened in the future."""
        return (len(self.array_array_codecs) + len(self.bytes_bytes_codecs)) == 0 and isinstance(
            self.array_bytes_codec, ArrayBytesCodecPartialEncodeMixin
        )

    def __iter__(self) -> Iterator[Codec]:
        yield from self.array_array_codecs
        yield self.array_bytes_codec
        yield from self.bytes_bytes_codecs

    def validate(self, array_metadata: ArrayMetadata) -> None:
        for codec in self:
            codec.validate(array_metadata)

    def compute_encoded_size(self, byte_length: int, array_spec: ArraySpec) -> int:
        for codec in self:
            byte_length = codec.compute_encoded_size(byte_length, array_spec)
            array_spec = codec.resolve_metadata(array_spec)
        return byte_length

    def _codecs_with_resolved_metadata_batched(
        self, chunk_specs: Iterable[ArraySpec]
    ) -> tuple[
        list[tuple[ArrayArrayCodec, list[ArraySpec]]],
        tuple[ArrayBytesCodec, list[ArraySpec]],
        list[tuple[BytesBytesCodec, list[ArraySpec]]],
    ]:
        aa_codecs_with_spec: list[tuple[ArrayArrayCodec, list[ArraySpec]]] = []
        chunk_specs = list(chunk_specs)
        for aa_codec in self.array_array_codecs:
            aa_codecs_with_spec.append((aa_codec, chunk_specs))
            chunk_specs = [aa_codec.resolve_metadata(chunk_spec) for chunk_spec in chunk_specs]

        ab_codec_with_spec = (self.array_bytes_codec, chunk_specs)
        chunk_specs = [
            self.array_bytes_codec.resolve_metadata(chunk_spec) for chunk_spec in chunk_specs
        ]

        bb_codecs_with_spec: list[tuple[BytesBytesCodec, list[ArraySpec]]] = []
        for bb_codec in self.bytes_bytes_codecs:
            bb_codecs_with_spec.append((bb_codec, chunk_specs))
            chunk_specs = [bb_codec.resolve_metadata(chunk_spec) for chunk_spec in chunk_specs]

        return (aa_codecs_with_spec, ab_codec_with_spec, bb_codecs_with_spec)

    async def decode_batch(
        self,
        chunk_bytes_and_specs: Iterable[tuple[Buffer | None, ArraySpec]],
    ) -> Iterable[NDBuffer | None]:
        chunk_bytes_batch: Iterable[Buffer | None]
        chunk_bytes_batch, chunk_specs = _unzip2(chunk_bytes_and_specs)

        (
            aa_codecs_with_spec,
            ab_codec_with_spec,
            bb_codecs_with_spec,
        ) = self._codecs_with_resolved_metadata_batched(chunk_specs)

        for bb_codec, chunk_spec_batch in bb_codecs_with_spec[::-1]:
            chunk_bytes_batch = await bb_codec.decode(
                zip(chunk_bytes_batch, chunk_spec_batch, strict=False)
            )

        ab_codec, chunk_spec_batch = ab_codec_with_spec
        chunk_array_batch = await ab_codec.decode(
            zip(chunk_bytes_batch, chunk_spec_batch, strict=False)
        )

        for aa_codec, chunk_spec_batch in aa_codecs_with_spec[::-1]:
            chunk_array_batch = await aa_codec.decode(
                zip(chunk_array_batch, chunk_spec_batch, strict=False)
            )

        return chunk_array_batch

    async def decode_partial_batch(
        self,
        batch_info: Iterable[tuple[ByteGetter, SelectorTuple, ArraySpec]],
    ) -> Iterable[NDBuffer | None]:
        assert self.supports_partial_decode
        assert isinstance(self.array_bytes_codec, ArrayBytesCodecPartialDecodeMixin)
        return await self.array_bytes_codec.decode_partial(batch_info)

    async def encode_batch(
        self,
        chunk_arrays_and_specs: Iterable[tuple[NDBuffer | None, ArraySpec]],
    ) -> Iterable[Buffer | None]:
        chunk_array_batch: Iterable[NDBuffer | None]
        chunk_specs: Iterable[ArraySpec]
        chunk_array_batch, chunk_specs = _unzip2(chunk_arrays_and_specs)

        for aa_codec in self.array_array_codecs:
            chunk_array_batch = await aa_codec.encode(
                zip(chunk_array_batch, chunk_specs, strict=False)
            )
            chunk_specs = resolve_batched(aa_codec, chunk_specs)

        chunk_bytes_batch = await self.array_bytes_codec.encode(
            zip(chunk_array_batch, chunk_specs, strict=False)
        )
        chunk_specs = resolve_batched(self.array_bytes_codec, chunk_specs)

        for bb_codec in self.bytes_bytes_codecs:
            chunk_bytes_batch = await bb_codec.encode(
                zip(chunk_bytes_batch, chunk_specs, strict=False)
            )
            chunk_specs = resolve_batched(bb_codec, chunk_specs)

        return chunk_bytes_batch

    async def encode_partial_batch(
        self,
        batch_info: Iterable[tuple[ByteSetter, NDBuffer, SelectorTuple, ArraySpec]],
    ) -> None:
        assert self.supports_partial_encode
        assert isinstance(self.array_bytes_codec, ArrayBytesCodecPartialEncodeMixin)
        await self.array_bytes_codec.encode_partial(batch_info)

    async def read_batch(
        self,
        batch_info: Iterable[tuple[ByteGetter, ArraySpec, SelectorTuple, SelectorTuple]],
        out: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> None:
        if self.supports_partial_decode:
            chunk_array_batch = await self.decode_partial_batch(
                [
                    (byte_getter, chunk_selection, chunk_spec)
                    for byte_getter, chunk_spec, chunk_selection, _ in batch_info
                ]
            )
            for chunk_array, (_, chunk_spec, _, out_selection) in zip(
                chunk_array_batch, batch_info, strict=False
            ):
                if chunk_array is not None:
                    out[out_selection] = chunk_array
                else:
                    out[out_selection] = chunk_spec.fill_value
        else:
            chunk_bytes_batch = await concurrent_map(
                [
                    (byte_getter, array_spec.prototype)
                    for byte_getter, array_spec, _, _ in batch_info
                ],
                lambda byte_getter, prototype: byte_getter.get(prototype),
                config.get("async.concurrency"),
            )
            chunk_array_batch = await self.decode_batch(
                [
                    (chunk_bytes, chunk_spec)
                    for chunk_bytes, (_, chunk_spec, _, _) in zip(
                        chunk_bytes_batch, batch_info, strict=False
                    )
                ],
            )
            for chunk_array, (_, chunk_spec, chunk_selection, out_selection) in zip(
                chunk_array_batch, batch_info, strict=False
            ):
                if chunk_array is not None:
                    tmp = chunk_array[chunk_selection]
                    if drop_axes != ():
                        tmp = tmp.squeeze(axis=drop_axes)
                    out[out_selection] = tmp
                else:
                    out[out_selection] = chunk_spec.fill_value

    def _merge_chunk_array(
        self,
        existing_chunk_array: NDBuffer | None,
        value: NDBuffer,
        out_selection: SelectorTuple,
        chunk_spec: ArraySpec,
        chunk_selection: SelectorTuple,
        drop_axes: tuple[int, ...],
    ) -> NDBuffer:
        if is_total_slice(chunk_selection, chunk_spec.shape) and value.shape == chunk_spec.shape:
            return value
        if existing_chunk_array is None:
            chunk_array = chunk_spec.prototype.nd_buffer.create(
                shape=chunk_spec.shape,
                dtype=chunk_spec.dtype,
                order=chunk_spec.order,
                fill_value=chunk_spec.fill_value,
            )
        else:
            chunk_array = existing_chunk_array.copy()  # make a writable copy
        if chunk_selection == ():
            chunk_value = value
        elif is_scalar(value.as_ndarray_like(), chunk_spec.dtype):
            chunk_value = value
        else:
            chunk_value = value[out_selection]
            # handle missing singleton dimensions
            if drop_axes != ():
                item = tuple(
                    None  # equivalent to np.newaxis
                    if idx in drop_axes
                    else slice(None)
                    for idx in range(chunk_spec.ndim)
                )
                chunk_value = chunk_value[item]
        chunk_array[chunk_selection] = chunk_value
        return chunk_array

    async def write_batch(
        self,
        batch_info: Iterable[tuple[ByteSetter, ArraySpec, SelectorTuple, SelectorTuple]],
        value: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> None:
        if self.supports_partial_encode:
            await self.encode_partial_batch(
                [
                    (byte_setter, value[out_selection], chunk_selection, chunk_spec)
                    for byte_setter, chunk_spec, chunk_selection, out_selection in batch_info
                ],
            )

        else:
            # Read existing bytes if not total slice
            async def _read_key(
                byte_setter: ByteSetter | None, prototype: BufferPrototype
            ) -> Buffer | None:
                if byte_setter is None:
                    return None
                return await byte_setter.get(prototype=prototype)

            chunk_bytes_batch: Iterable[Buffer | None]
            chunk_bytes_batch = await concurrent_map(
                [
                    (
                        None if is_total_slice(chunk_selection, chunk_spec.shape) else byte_setter,
                        chunk_spec.prototype,
                    )
                    for byte_setter, chunk_spec, chunk_selection, _ in batch_info
                ],
                _read_key,
                config.get("async.concurrency"),
            )
            chunk_array_batch = await self.decode_batch(
                [
                    (chunk_bytes, chunk_spec)
                    for chunk_bytes, (_, chunk_spec, _, _) in zip(
                        chunk_bytes_batch, batch_info, strict=False
                    )
                ],
            )

            chunk_array_batch = [
                self._merge_chunk_array(
                    chunk_array, value, out_selection, chunk_spec, chunk_selection, drop_axes
                )
                for chunk_array, (_, chunk_spec, chunk_selection, out_selection) in zip(
                    chunk_array_batch, batch_info, strict=False
                )
            ]

            chunk_array_batch = [
                None
                if chunk_array is None or chunk_array.all_equal(chunk_spec.fill_value)
                else chunk_array
                for chunk_array, (_, chunk_spec, _, _) in zip(
                    chunk_array_batch, batch_info, strict=False
                )
            ]

            chunk_bytes_batch = await self.encode_batch(
                [
                    (chunk_array, chunk_spec)
                    for chunk_array, (_, chunk_spec, _, _) in zip(
                        chunk_array_batch, batch_info, strict=False
                    )
                ],
            )

            async def _write_key(byte_setter: ByteSetter, chunk_bytes: Buffer | None) -> None:
                if chunk_bytes is None:
                    await byte_setter.delete()
                else:
                    await byte_setter.set(chunk_bytes)

            await concurrent_map(
                [
                    (byte_setter, chunk_bytes)
                    for chunk_bytes, (byte_setter, _, _, _) in zip(
                        chunk_bytes_batch, batch_info, strict=False
                    )
                ],
                _write_key,
                config.get("async.concurrency"),
            )

    async def decode(
        self,
        chunk_bytes_and_specs: Iterable[tuple[Buffer | None, ArraySpec]],
    ) -> Iterable[NDBuffer | None]:
        output: list[NDBuffer | None] = []
        for batch_info in batched(chunk_bytes_and_specs, self.batch_size):
            output.extend(await self.decode_batch(batch_info))
        return output

    async def encode(
        self,
        chunk_arrays_and_specs: Iterable[tuple[NDBuffer | None, ArraySpec]],
    ) -> Iterable[Buffer | None]:
        output: list[Buffer | None] = []
        for single_batch_info in batched(chunk_arrays_and_specs, self.batch_size):
            output.extend(await self.encode_batch(single_batch_info))
        return output

    async def read(
        self,
        batch_info: Iterable[tuple[ByteGetter, ArraySpec, SelectorTuple, SelectorTuple]],
        out: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> None:
        await concurrent_map(
            [
                (single_batch_info, out, drop_axes)
                for single_batch_info in batched(batch_info, self.batch_size)
            ],
            self.read_batch,
            config.get("async.concurrency"),
        )

    async def write(
        self,
        batch_info: Iterable[tuple[ByteSetter, ArraySpec, SelectorTuple, SelectorTuple]],
        value: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> None:
        await concurrent_map(
            [
                (single_batch_info, value, drop_axes)
                for single_batch_info in batched(batch_info, self.batch_size)
            ],
            self.write_batch,
            config.get("async.concurrency"),
        )
