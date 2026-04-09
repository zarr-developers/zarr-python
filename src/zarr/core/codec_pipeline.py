from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from itertools import islice, pairwise
from typing import TYPE_CHECKING, Any
from warnings import warn

import numpy as np

from zarr.abc.codec import (
    ArrayArrayCodec,
    ArrayBytesCodec,
    ArrayBytesCodecPartialDecodeMixin,
    ArrayBytesCodecPartialEncodeMixin,
    BytesBytesCodec,
    Codec,
    CodecPipeline,
    GetResult,
    SupportsSyncCodec,
)
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import numpy_buffer_prototype
from zarr.core.common import concurrent_map
from zarr.core.config import config
from zarr.core.indexing import SelectorTuple, is_scalar
from zarr.errors import ZarrUserWarning
from zarr.registry import register_pipeline

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from typing import Self

    from zarr.abc.store import ByteGetter, ByteSetter
    from zarr.core.buffer import Buffer, BufferPrototype, NDBuffer
    from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType
    from zarr.core.metadata.v3 import ChunkGridMetadata


def _unzip2[T, U](iterable: Iterable[tuple[T, U]]) -> tuple[list[T], list[U]]:
    out0: list[T] = []
    out1: list[U] = []
    for item0, item1 in iterable:
        out0.append(item0)
        out1.append(item1)
    return (out0, out1)


def batched[T](iterable: Iterable[T], n: int) -> Iterable[tuple[T, ...]]:
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def resolve_batched(codec: Codec, chunk_specs: Iterable[ArraySpec]) -> Iterable[ArraySpec]:
    return [codec.resolve_metadata(chunk_spec) for chunk_spec in chunk_specs]


def fill_value_or_default(chunk_spec: ArraySpec) -> Any:
    fill_value = chunk_spec.fill_value
    if fill_value is None:
        # Zarr V2 allowed `fill_value` to be null in the metadata.
        # Zarr V3 requires it to be set. This has already been
        # validated when decoding the metadata, but we support reading
        # Zarr V2 data and need to support the case where fill_value
        # is None.
        return chunk_spec.dtype.default_scalar()
    else:
        return fill_value


@dataclass(slots=True, kw_only=True)
class ChunkTransform:
    """A synchronous codec chain bound to an ArraySpec.

    Provides `encode` and `decode` for pure-compute codec operations
    (no IO, no threading, no batching).

    All codecs must implement `SupportsSyncCodec`. Construction will
    raise `TypeError` if any codec does not.
    """

    codecs: tuple[Codec, ...]
    array_spec: ArraySpec

    # (sync codec, input_spec) pairs in pipeline order.
    _aa_codecs: tuple[tuple[SupportsSyncCodec[NDBuffer, NDBuffer], ArraySpec], ...] = field(
        init=False, repr=False, compare=False
    )
    _ab_codec: SupportsSyncCodec[NDBuffer, Buffer] = field(init=False, repr=False, compare=False)
    _ab_spec: ArraySpec = field(init=False, repr=False, compare=False)
    _bb_codecs: tuple[SupportsSyncCodec[Buffer, Buffer], ...] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        non_sync = [c for c in self.codecs if not isinstance(c, SupportsSyncCodec)]
        if non_sync:
            names = ", ".join(type(c).__name__ for c in non_sync)
            raise TypeError(
                f"All codecs must implement SupportsSyncCodec. The following do not: {names}"
            )

        aa, ab, bb = codecs_from_list(list(self.codecs))

        aa_codecs: list[tuple[SupportsSyncCodec[NDBuffer, NDBuffer], ArraySpec]] = []
        spec = self.array_spec
        for aa_codec in aa:
            assert isinstance(aa_codec, SupportsSyncCodec)
            aa_codecs.append((aa_codec, spec))
            spec = aa_codec.resolve_metadata(spec)

        self._aa_codecs = tuple(aa_codecs)
        assert isinstance(ab, SupportsSyncCodec)
        self._ab_codec = ab
        self._ab_spec = spec
        bb_sync: list[SupportsSyncCodec[Buffer, Buffer]] = []
        for bb_codec in bb:
            assert isinstance(bb_codec, SupportsSyncCodec)
            bb_sync.append(bb_codec)
        self._bb_codecs = tuple(bb_sync)

    def _spec_for_shape(self, shape: tuple[int, ...]) -> ArraySpec:
        """Build an ArraySpec with the given shape, inheriting dtype/fill/config/prototype."""
        if shape == self._ab_spec.shape:
            return self._ab_spec
        return replace(self._ab_spec, shape=shape)

    def decode_chunk(
        self,
        chunk_bytes: Buffer,
        chunk_shape: tuple[int, ...] | None = None,
    ) -> NDBuffer:
        """Decode a single chunk through the full codec chain, synchronously.

        Pure compute -- no IO.

        Parameters
        ----------
        chunk_bytes : Buffer
            The encoded chunk bytes.
        chunk_shape : tuple[int, ...] or None
            The shape of this chunk. If None, uses the shape from the
            ArraySpec provided at construction. Required for rectilinear
            grids where chunks have different shapes.
        """
        spec = self._ab_spec if chunk_shape is None else self._spec_for_shape(chunk_shape)

        data: Buffer = chunk_bytes
        for bb_codec in reversed(self._bb_codecs):
            data = bb_codec._decode_sync(data, spec)

        chunk_array: NDBuffer = self._ab_codec._decode_sync(data, spec)

        for aa_codec, aa_spec in reversed(self._aa_codecs):
            aa_spec_resolved = aa_spec if chunk_shape is None else self._spec_for_shape(chunk_shape)
            chunk_array = aa_codec._decode_sync(chunk_array, aa_spec_resolved)

        return chunk_array

    def encode_chunk(
        self,
        chunk_array: NDBuffer,
        chunk_shape: tuple[int, ...] | None = None,
    ) -> Buffer | None:
        """Encode a single chunk through the full codec chain, synchronously.

        Pure compute -- no IO.

        Parameters
        ----------
        chunk_array : NDBuffer
            The chunk data to encode.
        chunk_shape : tuple[int, ...] or None
            The shape of this chunk. If None, uses the shape from the
            ArraySpec provided at construction.
        """
        spec = self._ab_spec if chunk_shape is None else self._spec_for_shape(chunk_shape)

        aa_data: NDBuffer = chunk_array
        for aa_codec, aa_spec in self._aa_codecs:
            aa_spec_resolved = aa_spec if chunk_shape is None else self._spec_for_shape(chunk_shape)
            aa_result = aa_codec._encode_sync(aa_data, aa_spec_resolved)
            if aa_result is None:
                return None
            aa_data = aa_result

        ab_result = self._ab_codec._encode_sync(aa_data, spec)
        if ab_result is None:
            return None

        bb_data: Buffer = ab_result
        for bb_codec in self._bb_codecs:
            bb_result = bb_codec._encode_sync(bb_data, spec)
            if bb_result is None:
                return None
            bb_data = bb_result

        return bb_data

    def compute_encoded_size(self, byte_length: int, array_spec: ArraySpec) -> int:
        for codec in self.codecs:
            byte_length = codec.compute_encoded_size(byte_length, array_spec)
            array_spec = codec.resolve_metadata(array_spec)
        return byte_length


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

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        return type(self).from_codecs(c.evolve_from_array_spec(array_spec=array_spec) for c in self)

    @classmethod
    def from_codecs(cls, codecs: Iterable[Codec], *, batch_size: int | None = None) -> Self:
        array_array_codecs, array_bytes_codec, bytes_bytes_codecs = codecs_from_list(codecs)

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

    def validate(
        self,
        *,
        shape: tuple[int, ...],
        dtype: ZDType[TBaseDType, TBaseScalar],
        chunk_grid: ChunkGridMetadata,
    ) -> None:
        for codec in self:
            codec.validate(shape=shape, dtype=dtype, chunk_grid=chunk_grid)

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
        batch_info: Iterable[tuple[ByteGetter, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        out: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> tuple[GetResult, ...]:
        results: list[GetResult] = []
        if self.supports_partial_decode:
            batch_info_list = list(batch_info)
            chunk_array_batch = await self.decode_partial_batch(
                [
                    (byte_getter, chunk_selection, chunk_spec)
                    for byte_getter, chunk_spec, chunk_selection, *_ in batch_info_list
                ]
            )
            for chunk_array, (_, chunk_spec, _, out_selection, _) in zip(
                chunk_array_batch, batch_info_list, strict=False
            ):
                if chunk_array is not None:
                    if drop_axes:
                        chunk_array = chunk_array.squeeze(axis=drop_axes)
                    out[out_selection] = chunk_array
                    results.append(GetResult(status="present"))
                else:
                    out[out_selection] = fill_value_or_default(chunk_spec)
                    results.append(GetResult(status="missing"))
        else:
            batch_info_list = list(batch_info)
            chunk_bytes_batch = await concurrent_map(
                [
                    (byte_getter, array_spec.prototype)
                    for byte_getter, array_spec, *_ in batch_info_list
                ],
                lambda byte_getter, prototype: byte_getter.get(prototype),
                config.get("async.concurrency"),
            )
            chunk_array_batch = await self.decode_batch(
                [
                    (chunk_bytes, chunk_spec)
                    for chunk_bytes, (_, chunk_spec, *_) in zip(
                        chunk_bytes_batch, batch_info_list, strict=False
                    )
                ],
            )
            for chunk_array, (_, chunk_spec, chunk_selection, out_selection, _) in zip(
                chunk_array_batch, batch_info_list, strict=False
            ):
                if chunk_array is not None:
                    tmp = chunk_array[chunk_selection]
                    if drop_axes:
                        tmp = tmp.squeeze(axis=drop_axes)
                    out[out_selection] = tmp
                    results.append(GetResult(status="present"))
                else:
                    out[out_selection] = fill_value_or_default(chunk_spec)
                    results.append(GetResult(status="missing"))
        return tuple(results)

    def _merge_chunk_array(
        self,
        existing_chunk_array: NDBuffer | None,
        value: NDBuffer,
        out_selection: SelectorTuple,
        chunk_spec: ArraySpec,
        chunk_selection: SelectorTuple,
        is_complete_chunk: bool,
        drop_axes: tuple[int, ...],
    ) -> NDBuffer:
        if (
            is_complete_chunk
            and value.shape == chunk_spec.shape
            # Guard that this is not a partial chunk at the end with is_complete_chunk=True
            and value[out_selection].shape == chunk_spec.shape
        ):
            return value
        if existing_chunk_array is None:
            chunk_array = chunk_spec.prototype.nd_buffer.create(
                shape=chunk_spec.shape,
                dtype=chunk_spec.dtype.to_native_dtype(),
                order=chunk_spec.order,
                fill_value=fill_value_or_default(chunk_spec),
            )
        else:
            chunk_array = existing_chunk_array.copy()  # make a writable copy
        if chunk_selection == () or is_scalar(
            value.as_ndarray_like(), chunk_spec.dtype.to_native_dtype()
        ):
            chunk_value = value
        else:
            chunk_value = value[out_selection]
            # handle missing singleton dimensions
            if drop_axes:
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
        batch_info: Iterable[tuple[ByteSetter, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        value: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> None:
        if self.supports_partial_encode:
            # Pass scalar values as is
            if len(value.shape) == 0:
                await self.encode_partial_batch(
                    [
                        (byte_setter, value, chunk_selection, chunk_spec)
                        for byte_setter, chunk_spec, chunk_selection, out_selection, _ in batch_info
                    ],
                )
            else:
                await self.encode_partial_batch(
                    [
                        (byte_setter, value[out_selection], chunk_selection, chunk_spec)
                        for byte_setter, chunk_spec, chunk_selection, out_selection, _ in batch_info
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
                        None if is_complete_chunk else byte_setter,
                        chunk_spec.prototype,
                    )
                    for byte_setter, chunk_spec, chunk_selection, _, is_complete_chunk in batch_info
                ],
                _read_key,
                config.get("async.concurrency"),
            )
            chunk_array_decoded = await self.decode_batch(
                [
                    (chunk_bytes, chunk_spec)
                    for chunk_bytes, (_, chunk_spec, *_) in zip(
                        chunk_bytes_batch, batch_info, strict=False
                    )
                ],
            )

            chunk_array_merged = [
                self._merge_chunk_array(
                    chunk_array,
                    value,
                    out_selection,
                    chunk_spec,
                    chunk_selection,
                    is_complete_chunk,
                    drop_axes,
                )
                for chunk_array, (
                    _,
                    chunk_spec,
                    chunk_selection,
                    out_selection,
                    is_complete_chunk,
                ) in zip(chunk_array_decoded, batch_info, strict=False)
            ]
            chunk_array_batch: list[NDBuffer | None] = []
            for chunk_array, (_, chunk_spec, *_) in zip(
                chunk_array_merged, batch_info, strict=False
            ):
                if chunk_array is None:
                    chunk_array_batch.append(None)  # type: ignore[unreachable]
                else:
                    if not chunk_spec.config.write_empty_chunks and chunk_array.all_equal(
                        fill_value_or_default(chunk_spec)
                    ):
                        chunk_array_batch.append(None)
                    else:
                        chunk_array_batch.append(chunk_array)

            chunk_bytes_batch = await self.encode_batch(
                [
                    (chunk_array, chunk_spec)
                    for chunk_array, (_, chunk_spec, *_) in zip(
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
                    for chunk_bytes, (byte_setter, *_) in zip(
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
        batch_info: Iterable[tuple[ByteGetter, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        out: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> tuple[GetResult, ...]:
        batch_results = await concurrent_map(
            [
                (single_batch_info, out, drop_axes)
                for single_batch_info in batched(batch_info, self.batch_size)
            ],
            self.read_batch,
            config.get("async.concurrency"),
        )
        results: list[GetResult] = []
        for batch in batch_results:
            results.extend(batch)
        return tuple(results)

    async def write(
        self,
        batch_info: Iterable[tuple[ByteSetter, ArraySpec, SelectorTuple, SelectorTuple, bool]],
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


def codecs_from_list(
    codecs: Iterable[Codec],
) -> tuple[tuple[ArrayArrayCodec, ...], ArrayBytesCodec, tuple[BytesBytesCodec, ...]]:
    from zarr.codecs.sharding import ShardingCodec

    codecs = tuple(codecs)  # materialize to avoid generator consumption issues

    array_array: tuple[ArrayArrayCodec, ...] = ()
    array_bytes_maybe: ArrayBytesCodec | None = None
    bytes_bytes: tuple[BytesBytesCodec, ...] = ()

    if any(isinstance(codec, ShardingCodec) for codec in codecs) and len(codecs) > 1:
        warn(
            "Combining a `sharding_indexed` codec disables partial reads and "
            "writes, which may lead to inefficient performance.",
            category=ZarrUserWarning,
            stacklevel=3,
        )

    for prev_codec, cur_codec in pairwise((None, *codecs)):
        if isinstance(cur_codec, ArrayArrayCodec):
            if isinstance(prev_codec, ArrayBytesCodec | BytesBytesCodec):
                msg = (
                    f"Invalid codec order. ArrayArrayCodec {cur_codec}"
                    "must be preceded by another ArrayArrayCodec. "
                    f"Got {type(prev_codec)} instead."
                )
                raise TypeError(msg)
            array_array += (cur_codec,)

        elif isinstance(cur_codec, ArrayBytesCodec):
            if isinstance(prev_codec, BytesBytesCodec):
                msg = (
                    f"Invalid codec order. ArrayBytes codec {cur_codec}"
                    f" must be preceded by an ArrayArrayCodec. Got {type(prev_codec)} instead."
                )
                raise TypeError(msg)

            if array_bytes_maybe is not None:
                msg = (
                    f"Got two instances of ArrayBytesCodec: {array_bytes_maybe} and {cur_codec}. "
                    "Only one array-to-bytes codec is allowed."
                )
                raise ValueError(msg)

            array_bytes_maybe = cur_codec

        elif isinstance(cur_codec, BytesBytesCodec):
            if isinstance(prev_codec, ArrayArrayCodec):
                msg = (
                    f"Invalid codec order. BytesBytesCodec {cur_codec}"
                    "must be preceded by either another BytesBytesCodec, or an ArrayBytesCodec. "
                    f"Got {type(prev_codec)} instead."
                )
            bytes_bytes += (cur_codec,)
        else:
            raise TypeError

    if array_bytes_maybe is None:
        raise ValueError("Required ArrayBytesCodec was not found.")
    else:
        return array_array, array_bytes_maybe, bytes_bytes


register_pipeline(BatchedCodecPipeline)


@dataclass(frozen=True)
class ShardLayout:
    """Configuration extracted from a ShardingCodec that tells the pipeline
    how to interpret a stored blob as a collection of inner chunks.

    This is a data structure, not an actor — the pipeline reads its fields
    and handles all IO and compute itself.
    """

    inner_chunk_shape: tuple[int, ...]
    chunks_per_shard: tuple[int, ...]
    index_transform: ChunkTransform  # for encoding/decoding the shard index
    inner_transform: ChunkTransform  # for encoding/decoding inner chunks
    index_location: Any  # ShardingCodecIndexLocation
    index_size: int  # byte size of the encoded shard index

    def decode_index(self, index_bytes: Buffer) -> Any:
        """Decode a shard index from bytes. Pure compute."""
        from zarr.codecs.sharding import _ShardIndex

        index_array = self.index_transform.decode_chunk(index_bytes)
        return _ShardIndex(index_array.as_numpy_array())

    def encode_index(self, index: Any) -> Buffer:
        """Encode a shard index to bytes. Pure compute."""
        from zarr.registry import get_ndbuffer_class

        index_nd = get_ndbuffer_class().from_numpy_array(index.offsets_and_lengths)
        result = self.index_transform.encode_chunk(index_nd)
        assert result is not None
        return result

    async def fetch_index(self, byte_getter: Any) -> Any:
        """Fetch and decode the shard index via byte-range read. IO + compute."""
        from zarr.abc.store import RangeByteRequest, SuffixByteRequest
        from zarr.codecs.sharding import ShardingCodecIndexLocation

        if self.index_location == ShardingCodecIndexLocation.start:
            index_bytes = await byte_getter.get(
                prototype=numpy_buffer_prototype(),
                byte_range=RangeByteRequest(0, self.index_size),
            )
        else:
            index_bytes = await byte_getter.get(
                prototype=numpy_buffer_prototype(),
                byte_range=SuffixByteRequest(self.index_size),
            )
        if index_bytes is None:
            return None
        return self.decode_index(index_bytes)

    def fetch_index_sync(self, byte_getter: Any) -> Any:
        """Sync variant of fetch_index."""
        from zarr.abc.store import RangeByteRequest, SuffixByteRequest
        from zarr.codecs.sharding import ShardingCodecIndexLocation

        if self.index_location == ShardingCodecIndexLocation.start:
            index_bytes = byte_getter.get_sync(
                prototype=numpy_buffer_prototype(),
                byte_range=RangeByteRequest(0, self.index_size),
            )
        else:
            index_bytes = byte_getter.get_sync(
                prototype=numpy_buffer_prototype(),
                byte_range=SuffixByteRequest(self.index_size),
            )
        if index_bytes is None:
            return None
        return self.decode_index(index_bytes)

    async def fetch_chunks(
        self, byte_getter: Any, index: Any, needed_coords: set[tuple[int, ...]]
    ) -> dict[tuple[int, ...], Buffer | None]:
        """Fetch only the needed inner chunks via byte-range reads, concurrently."""
        from zarr.abc.store import RangeByteRequest
        from zarr.core.buffer import default_buffer_prototype

        coords_list = list(needed_coords)
        slices = [index.get_chunk_slice(c) for c in coords_list]

        async def _fetch_one(
            coords: tuple[int, ...], chunk_slice: tuple[int, int] | None
        ) -> tuple[tuple[int, ...], Buffer | None]:
            if chunk_slice is not None:
                chunk_bytes = await byte_getter.get(
                    prototype=default_buffer_prototype(),
                    byte_range=RangeByteRequest(chunk_slice[0], chunk_slice[1]),
                )
                return (coords, chunk_bytes)
            return (coords, None)

        fetched = await concurrent_map(
            list(zip(coords_list, slices, strict=True)),
            _fetch_one,
            config.get("async.concurrency"),
        )
        return dict(fetched)

    def fetch_chunks_sync(
        self, byte_getter: Any, index: Any, needed_coords: set[tuple[int, ...]]
    ) -> dict[tuple[int, ...], Buffer | None]:
        """Sync variant of fetch_chunks."""
        from zarr.abc.store import RangeByteRequest
        from zarr.core.buffer import default_buffer_prototype

        result: dict[tuple[int, ...], Buffer | None] = {}
        for coords in needed_coords:
            chunk_slice = index.get_chunk_slice(coords)
            if chunk_slice is not None:
                chunk_bytes = byte_getter.get_sync(
                    prototype=default_buffer_prototype(),
                    byte_range=RangeByteRequest(chunk_slice[0], chunk_slice[1]),
                )
                result[coords] = chunk_bytes
            else:
                result[coords] = None
        return result

    def unpack_blob(self, blob: Buffer) -> dict[tuple[int, ...], Buffer | None]:
        """Unpack a shard blob into per-inner-chunk buffers. Pure compute."""
        from zarr.codecs.sharding import ShardingCodecIndexLocation

        if self.index_location == ShardingCodecIndexLocation.start:
            index_bytes = blob[: self.index_size]
        else:
            index_bytes = blob[-self.index_size :]

        index = self.decode_index(index_bytes)
        result: dict[tuple[int, ...], Buffer | None] = {}
        for chunk_coords in np.ndindex(self.chunks_per_shard):
            chunk_slice = index.get_chunk_slice(chunk_coords)
            if chunk_slice is not None:
                result[chunk_coords] = blob[chunk_slice[0] : chunk_slice[1]]
            else:
                result[chunk_coords] = None
        return result

    def pack_blob(
        self, chunk_dict: dict[tuple[int, ...], Buffer | None], prototype: BufferPrototype
    ) -> Buffer | None:
        """Pack per-inner-chunk buffers into a shard blob. Pure compute."""
        from zarr.codecs.sharding import MAX_UINT_64, ShardingCodecIndexLocation, _ShardIndex
        from zarr.core.indexing import morton_order_iter

        index = _ShardIndex.create_empty(self.chunks_per_shard)
        buffers: list[Buffer] = []
        template = prototype.buffer.create_zero_length()
        chunk_start = 0

        for chunk_coords in morton_order_iter(self.chunks_per_shard):
            value = chunk_dict.get(chunk_coords)
            if value is None or len(value) == 0:
                continue
            chunk_length = len(value)
            buffers.append(value)
            index.set_chunk_slice(chunk_coords, slice(chunk_start, chunk_start + chunk_length))
            chunk_start += chunk_length

        if not buffers:
            return None

        index_bytes = self.encode_index(index)
        if self.index_location == ShardingCodecIndexLocation.start:
            empty_mask = index.offsets_and_lengths[..., 0] == MAX_UINT_64
            index.offsets_and_lengths[~empty_mask, 0] += len(index_bytes)
            index_bytes = self.encode_index(index)
            buffers.insert(0, index_bytes)
        else:
            buffers.append(index_bytes)

        return template.combine(buffers)

    @classmethod
    def from_sharding_codec(cls, codec: Any, shard_spec: ArraySpec) -> ShardLayout:
        """Extract layout configuration from a ShardingCodec."""
        chunk_shape = codec.chunk_shape
        shard_shape = shard_spec.shape
        chunks_per_shard = tuple(s // c for s, c in zip(shard_shape, chunk_shape, strict=True))

        # Build inner chunk spec
        inner_spec = ArraySpec(
            shape=chunk_shape,
            dtype=shard_spec.dtype,
            fill_value=shard_spec.fill_value,
            config=shard_spec.config,
            prototype=shard_spec.prototype,
        )
        inner_evolved = tuple(c.evolve_from_array_spec(array_spec=inner_spec) for c in codec.codecs)
        inner_transform = ChunkTransform(codecs=inner_evolved, array_spec=inner_spec)

        # Build index spec and transform
        from zarr.codecs.sharding import MAX_UINT_64
        from zarr.core.array_spec import ArrayConfig
        from zarr.core.buffer import default_buffer_prototype
        from zarr.core.dtype.npy.int import UInt64

        index_spec = ArraySpec(
            shape=chunks_per_shard + (2,),
            dtype=UInt64(endianness="little"),
            fill_value=MAX_UINT_64,
            config=ArrayConfig(order="C", write_empty_chunks=False),
            prototype=default_buffer_prototype(),
        )
        index_evolved = tuple(
            c.evolve_from_array_spec(array_spec=index_spec) for c in codec.index_codecs
        )
        index_transform = ChunkTransform(codecs=index_evolved, array_spec=index_spec)

        # Compute index size
        index_size = index_transform.compute_encoded_size(
            16 * int(np.prod(chunks_per_shard)), index_spec
        )

        return cls(
            inner_chunk_shape=chunk_shape,
            chunks_per_shard=chunks_per_shard,
            index_transform=index_transform,
            inner_transform=inner_transform,
            index_location=codec.index_location,
            index_size=index_size,
        )


@dataclass(frozen=True)
class PhasedCodecPipeline(CodecPipeline):
    """Codec pipeline that cleanly separates IO from compute.

    The zarr v3 spec describes each codec as a function that may perform
    IO — the sharding codec, for example, is specified as reading and
    writing inner chunks from storage. This framing suggests that IO is
    distributed throughout the codec chain, making it difficult to
    parallelize or optimize.

    In practice, **codecs are pure compute**. Every codec transforms
    bytes to bytes, bytes to arrays, or arrays to arrays — none of them
    need to touch storage. The only IO happens at the pipeline level:
    reading a blob from a store key, and writing a blob back. Even the
    sharding codec is just a transform: it takes the full shard blob
    (already fetched) and splits it into inner-chunk buffers using an
    index, then decodes each inner chunk through its inner codec chain.
    No additional IO occurs inside the codec.

    This insight enables a strict three-phase architecture:

    1. **IO phase** — fetch raw bytes from the store (one key per chunk
       or shard). This is the only phase that touches storage.
    2. **Compute phase** — decode, merge, and re-encode chunks through
       the full codec chain, including sharding. This is pure CPU work
       with no IO, and can safely run in a thread pool.
    3. **IO phase** — write results back to the store.

    Because the compute phase is IO-free, it can be parallelized with
    threads (sync path) or ``asyncio.to_thread`` (async path) without
    holding IO resources or risking deadlocks.

    Nested sharding (a shard whose inner chunks are themselves shards)
    works the same way: the outer shard blob is fetched once in phase 1,
    then the compute phase unpacks it into inner shard blobs, each of
    which is decoded by the inner sharding codec — still pure compute,
    still no IO. The entire decode tree runs from the single blob
    fetched in phase 1.
    """

    codecs: tuple[Codec, ...]
    array_array_codecs: tuple[ArrayArrayCodec, ...]
    array_bytes_codec: ArrayBytesCodec
    bytes_bytes_codecs: tuple[BytesBytesCodec, ...]
    chunk_transform: ChunkTransform | None
    shard_layout: ShardLayout | None
    batch_size: int

    @classmethod
    def from_codecs(cls, codecs: Iterable[Codec], *, batch_size: int | None = None) -> Self:
        """Create a pipeline from codecs.

        The pipeline is not usable for read/write until ``evolve_from_array_spec``
        is called with the chunk's ArraySpec. This matches the CodecPipeline ABC
        contract.
        """
        codec_list = tuple(codecs)
        aa, ab, bb = codecs_from_list(codec_list)

        if batch_size is None:
            batch_size = config.get("codec_pipeline.batch_size")

        # chunk_transform and shard_layout require an ArraySpec.
        # They'll be built in evolve_from_array_spec.
        return cls(
            codecs=codec_list,
            array_array_codecs=aa,
            array_bytes_codec=ab,
            bytes_bytes_codecs=bb,
            chunk_transform=None,
            shard_layout=None,
            batch_size=batch_size,
        )

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        from zarr.codecs.sharding import ShardingCodec

        evolved_codecs = tuple(c.evolve_from_array_spec(array_spec=array_spec) for c in self.codecs)
        aa, ab, bb = codecs_from_list(evolved_codecs)

        chunk_transform = ChunkTransform(codecs=evolved_codecs, array_spec=array_spec)

        shard_layout: ShardLayout | None = None
        if isinstance(ab, ShardingCodec):
            shard_layout = ShardLayout.from_sharding_codec(ab, array_spec)

        return type(self)(
            codecs=evolved_codecs,
            array_array_codecs=aa,
            array_bytes_codec=ab,
            bytes_bytes_codecs=bb,
            chunk_transform=chunk_transform,
            shard_layout=shard_layout,
            batch_size=self.batch_size,
        )

    def __iter__(self) -> Iterator[Codec]:
        return iter(self.codecs)

    @property
    def supports_partial_decode(self) -> bool:
        return isinstance(self.array_bytes_codec, ArrayBytesCodecPartialDecodeMixin)

    @property
    def supports_partial_encode(self) -> bool:
        return isinstance(self.array_bytes_codec, ArrayBytesCodecPartialEncodeMixin)

    def validate(
        self,
        *,
        shape: tuple[int, ...],
        dtype: ZDType[TBaseDType, TBaseScalar],
        chunk_grid: ChunkGridMetadata,
    ) -> None:
        for codec in self.codecs:
            codec.validate(shape=shape, dtype=dtype, chunk_grid=chunk_grid)

    def compute_encoded_size(self, byte_length: int, array_spec: ArraySpec) -> int:
        if self.chunk_transform is None:
            raise RuntimeError(
                "Cannot compute encoded size before evolve_from_array_spec is called."
            )
        return self.chunk_transform.compute_encoded_size(byte_length, array_spec)

    async def decode(
        self,
        chunk_bytes_and_specs: Iterable[tuple[Buffer | None, ArraySpec]],
    ) -> Iterable[NDBuffer | None]:
        """Decode a batch of chunks through the full codec chain.

        Required by the ``CodecPipeline`` ABC. Not used internally by
        this pipeline — reads go through ``_transform_read`` or
        ``_read_shard_selective`` instead.
        """
        chunk_bytes_batch: Iterable[Buffer | None]
        chunk_bytes_batch, chunk_specs = _unzip2(chunk_bytes_and_specs)

        for bb_codec in self.bytes_bytes_codecs[::-1]:
            chunk_bytes_batch = await bb_codec.decode(
                zip(chunk_bytes_batch, chunk_specs, strict=False)
            )
        chunk_array_batch = await self.array_bytes_codec.decode(
            zip(chunk_bytes_batch, chunk_specs, strict=False)
        )
        for aa_codec in self.array_array_codecs[::-1]:
            chunk_array_batch = await aa_codec.decode(
                zip(chunk_array_batch, chunk_specs, strict=False)
            )
        return chunk_array_batch

    async def encode(
        self,
        chunk_arrays_and_specs: Iterable[tuple[NDBuffer | None, ArraySpec]],
    ) -> Iterable[Buffer | None]:
        """Encode a batch of chunks through the full codec chain.

        Required by the ``CodecPipeline`` ABC. Not used internally by
        this pipeline — writes go through ``_transform_write`` instead.
        """
        chunk_array_batch: Iterable[NDBuffer | None]
        chunk_array_batch, chunk_specs = _unzip2(chunk_arrays_and_specs)

        for aa_codec in self.array_array_codecs:
            chunk_array_batch = await aa_codec.encode(
                zip(chunk_array_batch, chunk_specs, strict=False)
            )
        chunk_bytes_batch = await self.array_bytes_codec.encode(
            zip(chunk_array_batch, chunk_specs, strict=False)
        )
        for bb_codec in self.bytes_bytes_codecs:
            chunk_bytes_batch = await bb_codec.encode(
                zip(chunk_bytes_batch, chunk_specs, strict=False)
            )
        return chunk_bytes_batch

    # -- Phase 2: pure compute (no IO) --

    def _transform_read(
        self,
        raw: Buffer | None,
        chunk_spec: ArraySpec,
    ) -> NDBuffer | None:
        """Decode raw bytes into an array. Pure sync compute, no IO.

        For non-sharded arrays, decodes through the full codec chain.
        For sharded arrays, unpacks the shard blob using the layout,
        decodes each inner chunk through the inner transform, and
        assembles the shard-shaped output.
        """
        if raw is None:
            return None

        if self.shard_layout is not None:
            return self._decode_shard(raw, chunk_spec, self.shard_layout)

        assert self.chunk_transform is not None
        return self.chunk_transform.decode_chunk(raw, chunk_shape=chunk_spec.shape)

    def _decode_shard(self, blob: Buffer, shard_spec: ArraySpec, layout: ShardLayout) -> NDBuffer:
        """Decode a full shard blob into a shard-shaped array. Pure compute.

        Used by the write path (via ``_transform_read``) to decode existing
        shard data before merging. For reads, ``_read_shard_selective`` is
        preferred since it fetches only the needed inner chunks.
        """
        from zarr.core.chunk_grids import ChunkGrid as _ChunkGrid
        from zarr.core.indexing import BasicIndexer

        chunk_dict = layout.unpack_blob(blob)

        out = shard_spec.prototype.nd_buffer.empty(
            shape=shard_spec.shape,
            dtype=shard_spec.dtype.to_native_dtype(),
            order=shard_spec.order,
        )

        indexer = BasicIndexer(
            tuple(slice(0, s) for s in shard_spec.shape),
            shape=shard_spec.shape,
            chunk_grid=_ChunkGrid.from_sizes(shard_spec.shape, layout.inner_chunk_shape),
        )

        for chunk_coords, chunk_selection, out_selection, _ in indexer:
            chunk_bytes = chunk_dict.get(chunk_coords)
            if chunk_bytes is not None:
                chunk_array = layout.inner_transform.decode_chunk(chunk_bytes)
                out[out_selection] = chunk_array[chunk_selection]
            else:
                out[out_selection] = shard_spec.fill_value

        return out

    def _transform_write(
        self,
        existing: Buffer | None,
        chunk_spec: ArraySpec,
        chunk_selection: SelectorTuple,
        out_selection: SelectorTuple,
        value: NDBuffer,
        drop_axes: tuple[int, ...],
    ) -> Buffer | None:
        """Decode existing, merge new data, re-encode. Pure sync compute, no IO."""
        if self.shard_layout is not None:
            return self._transform_write_shard(
                existing,
                chunk_spec,
                chunk_selection,
                out_selection,
                value,
                drop_axes,
                self.shard_layout,
            )

        assert self.chunk_transform is not None

        chunk_shape = chunk_spec.shape

        if existing is not None:
            chunk_array: NDBuffer | None = self.chunk_transform.decode_chunk(
                existing, chunk_shape=chunk_shape
            )
        else:
            chunk_array = None

        if chunk_array is None:
            chunk_array = chunk_spec.prototype.nd_buffer.create(
                shape=chunk_shape,
                dtype=chunk_spec.dtype.to_native_dtype(),
                fill_value=fill_value_or_default(chunk_spec),
            )

        if chunk_selection == () or is_scalar(
            value.as_ndarray_like(), chunk_spec.dtype.to_native_dtype()
        ):
            chunk_value = value
        else:
            chunk_value = value[out_selection]
            if drop_axes:
                item = tuple(
                    None if idx in drop_axes else slice(None) for idx in range(chunk_spec.ndim)
                )
                chunk_value = chunk_value[item]
        chunk_array[chunk_selection] = chunk_value

        return self.chunk_transform.encode_chunk(chunk_array, chunk_shape=chunk_shape)

    def _transform_write_shard(
        self,
        existing: Buffer | None,
        shard_spec: ArraySpec,
        chunk_selection: SelectorTuple,
        out_selection: SelectorTuple,
        value: NDBuffer,
        drop_axes: tuple[int, ...],
        layout: ShardLayout,
    ) -> Buffer | None:
        """Write into a shard, only decoding/encoding the affected inner chunks.

        Operates at the chunk mapping level: the existing shard blob is
        unpacked into a mapping of inner-chunk coordinates to raw bytes.
        Only inner chunks touched by the selection are decoded, merged,
        and re-encoded. Untouched chunks pass through as raw bytes.
        """
        from zarr.core.buffer import default_buffer_prototype
        from zarr.core.chunk_grids import ChunkGrid as _ChunkGrid
        from zarr.core.indexing import get_indexer

        # Unpack existing shard into chunk mapping (no decode — just index parse + byte slicing)
        if existing is not None:
            chunk_dict = layout.unpack_blob(existing)
        else:
            chunk_dict = dict.fromkeys(np.ndindex(layout.chunks_per_shard))

        # Determine which inner chunks are affected by the write selection
        indexer = get_indexer(
            chunk_selection,
            shape=shard_spec.shape,
            chunk_grid=_ChunkGrid.from_sizes(shard_spec.shape, layout.inner_chunk_shape),
        )

        inner_spec = ArraySpec(
            shape=layout.inner_chunk_shape,
            dtype=shard_spec.dtype,
            fill_value=shard_spec.fill_value,
            config=shard_spec.config,
            prototype=shard_spec.prototype,
        )

        # Only decode, merge, re-encode the affected inner chunks
        for inner_coords, inner_sel, value_sel, _ in indexer:
            existing_bytes = chunk_dict.get(inner_coords)

            # Decode just this inner chunk
            if existing_bytes is not None:
                inner_array = layout.inner_transform.decode_chunk(existing_bytes)
            else:
                inner_array = inner_spec.prototype.nd_buffer.create(
                    shape=inner_spec.shape,
                    dtype=inner_spec.dtype.to_native_dtype(),
                    fill_value=fill_value_or_default(inner_spec),
                )

            # Merge new data into this inner chunk
            if inner_sel == () or is_scalar(
                value.as_ndarray_like(), inner_spec.dtype.to_native_dtype()
            ):
                inner_value = value
            else:
                inner_value = value[value_sel]
                if drop_axes:
                    item = tuple(
                        None if idx in drop_axes else slice(None) for idx in range(inner_spec.ndim)
                    )
                    inner_value = inner_value[item]
            inner_array[inner_sel] = inner_value

            # Re-encode just this inner chunk
            chunk_dict[inner_coords] = layout.inner_transform.encode_chunk(inner_array)

        # Pack the mapping back into a blob (untouched chunks pass through as raw bytes)
        return layout.pack_blob(chunk_dict, default_buffer_prototype())

    # -- Phase 3: scatter (read) / store (write) --

    @staticmethod
    def _scatter(
        batch: list[tuple[Any, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        decoded: list[NDBuffer | None],
        out: NDBuffer,
        drop_axes: tuple[int, ...],
    ) -> tuple[GetResult, ...]:
        """Write decoded chunk arrays into the output buffer."""
        results: list[GetResult] = []
        for (_, chunk_spec, chunk_selection, out_selection, _), chunk_array in zip(
            batch, decoded, strict=True
        ):
            if chunk_array is not None:
                selected = chunk_array[chunk_selection]
                if drop_axes:
                    selected = selected.squeeze(axis=drop_axes)
                out[out_selection] = selected
                results.append(GetResult(status="present"))
            else:
                out[out_selection] = fill_value_or_default(chunk_spec)
                results.append(GetResult(status="missing"))
        return tuple(results)

    # -- Async API --

    async def _read_shard_selective(
        self,
        byte_getter: Any,
        shard_spec: ArraySpec,
        chunk_selection: SelectorTuple,
        layout: ShardLayout,
    ) -> NDBuffer | None:
        """Read from a shard fetching only the needed inner chunks.

        1. Fetch shard index (byte-range read)
        2. Determine which inner chunks are needed
        3. Fetch only those inner chunks (byte-range reads)
        4. Decode and assemble (pure compute)
        """
        from zarr.core.chunk_grids import ChunkGrid as _ChunkGrid
        from zarr.core.indexing import get_indexer

        # Phase 1: fetch index
        index = await layout.fetch_index(byte_getter)
        if index is None:
            return None

        # Determine needed inner chunks
        indexer = list(
            get_indexer(
                chunk_selection,
                shape=shard_spec.shape,
                chunk_grid=_ChunkGrid.from_sizes(shard_spec.shape, layout.inner_chunk_shape),
            )
        )
        needed_coords = {coords for coords, *_ in indexer}

        # Phase 2: fetch only needed inner chunks
        chunk_dict = await layout.fetch_chunks(byte_getter, index, needed_coords)

        # Phase 3: decode and assemble
        out = shard_spec.prototype.nd_buffer.empty(
            shape=shard_spec.shape,
            dtype=shard_spec.dtype.to_native_dtype(),
            order=shard_spec.order,
        )

        for inner_coords, inner_sel, out_sel, _ in indexer:
            chunk_bytes = chunk_dict.get(inner_coords)
            if chunk_bytes is not None:
                inner_array = layout.inner_transform.decode_chunk(chunk_bytes)
                out[out_sel] = inner_array[inner_sel]
            else:
                out[out_sel] = shard_spec.fill_value

        return out

    async def read(
        self,
        batch_info: Iterable[tuple[ByteGetter, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        out: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> tuple[GetResult, ...]:
        batch = list(batch_info)
        if not batch:
            return ()

        if self.shard_layout is not None:
            # Sharded: use selective byte-range reads per shard
            decoded: list[NDBuffer | None] = list(
                await concurrent_map(
                    [(bg, cs, chunk_sel, self.shard_layout) for bg, cs, chunk_sel, _, _ in batch],
                    self._read_shard_selective,
                    config.get("async.concurrency"),
                )
            )
        elif len(batch) == 1:
            # Non-sharded single chunk: fetch and decode inline
            bg, cs, _, _, _ = batch[0]
            raw = await bg.get(prototype=cs.prototype)
            decoded = [self._transform_read(raw, cs)]
        else:
            # Non-sharded multiple chunks: fetch all, decode in parallel threads
            import asyncio

            raw_buffers: list[Buffer | None] = await concurrent_map(
                [(bg, cs.prototype) for bg, cs, *_ in batch],
                lambda bg, proto: bg.get(prototype=proto),
                config.get("async.concurrency"),
            )
            decoded = list(
                await asyncio.gather(
                    *[
                        asyncio.to_thread(self._transform_read, raw, cs)
                        for raw, (_, cs, *_) in zip(raw_buffers, batch, strict=True)
                    ]
                )
            )

        # Scatter
        return self._scatter(batch, decoded, out, drop_axes)

    async def write(
        self,
        batch_info: Iterable[tuple[ByteSetter, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        value: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> None:
        batch = list(batch_info)
        if not batch:
            return

        # Phase 1: IO — fetch existing bytes concurrently (skip for complete writes)
        async def _fetch_existing(
            byte_setter: ByteSetter, chunk_spec: ArraySpec, is_complete: bool
        ) -> Buffer | None:
            if is_complete:
                return None
            return await byte_setter.get(prototype=chunk_spec.prototype)

        existing_buffers: list[Buffer | None] = await concurrent_map(
            [(bs, cs, ic) for bs, cs, _, _, ic in batch],
            _fetch_existing,
            config.get("async.concurrency"),
        )

        # Phase 2: compute — decode, merge, re-encode
        if len(batch) == 1:
            _, cs, csel, osel, _ = batch[0]
            blobs: list[Buffer | None] = [
                self._transform_write(existing_buffers[0], cs, csel, osel, value, drop_axes)
            ]
        else:
            import asyncio

            blobs = list(
                await asyncio.gather(
                    *[
                        asyncio.to_thread(
                            self._transform_write, existing, cs, csel, osel, value, drop_axes
                        )
                        for existing, (_, cs, csel, osel, _) in zip(
                            existing_buffers, batch, strict=True
                        )
                    ]
                )
            )

        # Phase 3: IO — write results concurrently
        async def _store_one(byte_setter: ByteSetter, blob: Buffer | None) -> None:
            if blob is None:
                await byte_setter.delete()
            else:
                await byte_setter.set(blob)

        await concurrent_map(
            [(bs, blob) for (bs, *_), blob in zip(batch, blobs, strict=True)],
            _store_one,
            config.get("async.concurrency"),
        )

    # -- Sync API --

    def _read_shard_selective_sync(
        self,
        byte_getter: Any,
        shard_spec: ArraySpec,
        chunk_selection: SelectorTuple,
        layout: ShardLayout,
    ) -> NDBuffer | None:
        """Sync variant of _read_shard_selective."""
        from zarr.core.chunk_grids import ChunkGrid as _ChunkGrid
        from zarr.core.indexing import get_indexer

        index = layout.fetch_index_sync(byte_getter)
        if index is None:
            return None

        indexer = list(
            get_indexer(
                chunk_selection,
                shape=shard_spec.shape,
                chunk_grid=_ChunkGrid.from_sizes(shard_spec.shape, layout.inner_chunk_shape),
            )
        )
        needed_coords = {coords for coords, *_ in indexer}

        chunk_dict = layout.fetch_chunks_sync(byte_getter, index, needed_coords)

        out = shard_spec.prototype.nd_buffer.empty(
            shape=shard_spec.shape,
            dtype=shard_spec.dtype.to_native_dtype(),
            order=shard_spec.order,
        )

        for inner_coords, inner_sel, out_sel, _ in indexer:
            chunk_bytes = chunk_dict.get(inner_coords)
            if chunk_bytes is not None:
                inner_array = layout.inner_transform.decode_chunk(chunk_bytes)
                out[out_sel] = inner_array[inner_sel]
            else:
                out[out_sel] = shard_spec.fill_value

        return out

    def read_sync(
        self,
        batch_info: Iterable[tuple[ByteGetter, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        out: NDBuffer,
        drop_axes: tuple[int, ...] = (),
        n_workers: int = 0,
    ) -> None:
        """Synchronous read."""
        batch = list(batch_info)
        if not batch:
            return

        if self.shard_layout is not None:
            # Sharded: selective byte-range reads per shard
            decoded: list[NDBuffer | None] = [
                self._read_shard_selective_sync(bg, cs, chunk_sel, self.shard_layout)
                for bg, cs, chunk_sel, _, _ in batch
            ]
        else:
            # Non-sharded: fetch full blobs, decode (optionally threaded)
            raw_buffers: list[Buffer | None] = [
                bg.get_sync(prototype=cs.prototype)  # type: ignore[attr-defined]
                for bg, cs, *_ in batch
            ]
            specs = [cs for _, cs, *_ in batch]
            if n_workers > 0 and len(batch) > 1:
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    decoded = list(pool.map(self._transform_read, raw_buffers, specs))
            else:
                decoded = [
                    self._transform_read(raw, cs)
                    for raw, cs in zip(raw_buffers, specs, strict=True)
                ]

        # Scatter
        self._scatter(batch, decoded, out, drop_axes)

    def write_sync(
        self,
        batch_info: Iterable[tuple[ByteSetter, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        value: NDBuffer,
        drop_axes: tuple[int, ...] = (),
        n_workers: int = 0,
    ) -> None:
        """Synchronous write. Same three phases as async, different IO wrapper."""
        batch = list(batch_info)
        if not batch:
            return

        # Phase 1: IO — fetch existing bytes serially
        existing_buffers: list[Buffer | None] = [
            None if ic else bs.get_sync(prototype=cs.prototype)  # type: ignore[attr-defined]
            for bs, cs, _, _, ic in batch
        ]

        # Phase 2: compute — decode, merge, re-encode (optionally threaded)
        def _compute(idx: int) -> Buffer | None:
            _, cs, csel, osel, _ = batch[idx]
            return self._transform_write(existing_buffers[idx], cs, csel, osel, value, drop_axes)

        indices = list(range(len(batch)))
        if n_workers > 0 and len(batch) > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                blobs: list[Buffer | None] = list(pool.map(_compute, indices))
        else:
            blobs = [_compute(i) for i in indices]

        # Phase 3: IO — write results serially
        for (bs, *_), blob in zip(batch, blobs, strict=True):
            if blob is None:
                bs.delete_sync()  # type: ignore[attr-defined]
            else:
                bs.set_sync(blob)  # type: ignore[attr-defined]


register_pipeline(PhasedCodecPipeline)
