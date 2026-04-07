from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import islice, pairwise
from typing import TYPE_CHECKING, Any
from warnings import warn

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
from zarr.core.common import concurrent_map
from zarr.core.config import config
from zarr.core.indexing import SelectorTuple, is_scalar
from zarr.errors import ZarrUserWarning
from zarr.registry import register_pipeline

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from typing import Self

    from zarr.abc.store import ByteGetter, ByteSetter
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer, BufferPrototype, NDBuffer
    from zarr.core.chunk_grids import ChunkGrid
    from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType


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

    def decode_chunk(
        self,
        chunk_bytes: Buffer,
    ) -> NDBuffer:
        """Decode a single chunk through the full codec chain, synchronously.

        Pure compute -- no IO.
        """
        data: Buffer = chunk_bytes
        for bb_codec in reversed(self._bb_codecs):
            data = bb_codec._decode_sync(data, self._ab_spec)

        chunk_array: NDBuffer = self._ab_codec._decode_sync(data, self._ab_spec)

        for aa_codec, spec in reversed(self._aa_codecs):
            chunk_array = aa_codec._decode_sync(chunk_array, spec)

        return chunk_array

    def encode_chunk(
        self,
        chunk_array: NDBuffer,
    ) -> Buffer | None:
        """Encode a single chunk through the full codec chain, synchronously.

        Pure compute -- no IO.
        """
        aa_data: NDBuffer = chunk_array
        for aa_codec, spec in self._aa_codecs:
            aa_result = aa_codec._encode_sync(aa_data, spec)
            if aa_result is None:
                return None
            aa_data = aa_result

        ab_result = self._ab_codec._encode_sync(aa_data, self._ab_spec)
        if ab_result is None:
            return None

        bb_data: Buffer = ab_result
        for bb_codec in self._bb_codecs:
            bb_result = bb_codec._encode_sync(bb_data, self._ab_spec)
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
        chunk_grid: ChunkGrid,
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

    array_array: tuple[ArrayArrayCodec, ...] = ()
    array_bytes_maybe: ArrayBytesCodec | None = None
    bytes_bytes: tuple[BytesBytesCodec, ...] = ()

    if any(isinstance(codec, ShardingCodec) for codec in codecs) and len(tuple(codecs)) > 1:
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
class PhasedCodecPipeline(CodecPipeline):
    """Codec pipeline using the three-phase prepare/compute/finalize pattern.

    Separates IO (prepare, finalize) from compute (encode, decode) so that
    the compute phase can run without holding IO resources. This is the
    foundation for thread-pool-based parallelism.

    Works with any ``ArrayBytesCodec``. The sync path (``read_sync`` /
    ``write_sync``) requires ``SupportsChunkPacking`` and ``SupportsSyncCodec``.
    """

    codecs: tuple[Codec, ...]
    chunk_transform: ChunkTransform | None
    batch_size: int

    @classmethod
    def from_codecs(cls, codecs: Iterable[Codec], *, batch_size: int | None = None) -> Self:
        codec_list = tuple(codecs)
        codecs_from_list(codec_list)  # validate codec ordering

        if batch_size is None:
            batch_size = config.get("codec_pipeline.batch_size")

        return cls(
            codecs=codec_list,
            chunk_transform=None,
            batch_size=batch_size,
        )

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        evolved_codecs = tuple(c.evolve_from_array_spec(array_spec=array_spec) for c in self.codecs)
        # Only create ChunkTransform if all codecs support sync
        all_sync = all(isinstance(c, SupportsSyncCodec) for c in evolved_codecs)
        chunk_transform = ChunkTransform(codecs=evolved_codecs, array_spec=array_spec) if all_sync else None
        return type(self)(
            codecs=evolved_codecs,
            chunk_transform=chunk_transform,
            batch_size=self.batch_size,
        )

    def __iter__(self) -> Iterator[Codec]:
        return iter(self.codecs)

    @property
    def supports_partial_decode(self) -> bool:
        ab = self._ab_codec
        return isinstance(ab, ArrayBytesCodecPartialDecodeMixin)

    @property
    def supports_partial_encode(self) -> bool:
        ab = self._ab_codec
        return isinstance(ab, ArrayBytesCodecPartialEncodeMixin)

    def validate(
        self, *, shape: tuple[int, ...], dtype: ZDType[TBaseDType, TBaseScalar], chunk_grid: ChunkGrid
    ) -> None:
        for codec in self.codecs:
            codec.validate(shape=shape, dtype=dtype, chunk_grid=chunk_grid)

    def compute_encoded_size(self, byte_length: int, array_spec: ArraySpec) -> int:
        if self.chunk_transform is not None:
            return self.chunk_transform.compute_encoded_size(byte_length, array_spec)
        return byte_length

    async def decode(
        self,
        chunk_bytes_and_specs: Iterable[tuple[Buffer | None, ArraySpec]],
    ) -> Iterable[NDBuffer | None]:
        """Decode a batch of chunks through the full codec chain."""
        aa, ab, bb = codecs_from_list(self.codecs)
        chunk_bytes_batch: Iterable[Buffer | None]
        chunk_bytes_batch, chunk_specs = _unzip2(chunk_bytes_and_specs)

        for bb_codec in bb[::-1]:
            chunk_bytes_batch = await bb_codec.decode(
                zip(chunk_bytes_batch, chunk_specs, strict=False)
            )
        chunk_array_batch = await ab.decode(
            zip(chunk_bytes_batch, chunk_specs, strict=False)
        )
        for aa_codec in aa[::-1]:
            chunk_array_batch = await aa_codec.decode(
                zip(chunk_array_batch, chunk_specs, strict=False)
            )
        return chunk_array_batch

    async def encode(
        self,
        chunk_arrays_and_specs: Iterable[tuple[NDBuffer | None, ArraySpec]],
    ) -> Iterable[Buffer | None]:
        """Encode a batch of chunks through the full codec chain."""
        aa, ab, bb = codecs_from_list(self.codecs)
        chunk_array_batch: Iterable[NDBuffer | None]
        chunk_array_batch, chunk_specs = _unzip2(chunk_arrays_and_specs)

        for aa_codec in aa:
            chunk_array_batch = await aa_codec.encode(
                zip(chunk_array_batch, chunk_specs, strict=False)
            )
        chunk_bytes_batch = await ab.encode(
            zip(chunk_array_batch, chunk_specs, strict=False)
        )
        for bb_codec in bb:
            chunk_bytes_batch = await bb_codec.encode(
                zip(chunk_bytes_batch, chunk_specs, strict=False)
            )
        return chunk_bytes_batch

    @property
    def _ab_codec(self) -> ArrayBytesCodec:
        _, ab, _ = codecs_from_list(self.codecs)
        return ab

    # -- Phase 2: pure compute (no IO) --

    def _transform_read(
        self,
        raw: Buffer | None,
        _chunk_spec: ArraySpec,
    ) -> NDBuffer | None:
        """Decode raw bytes into an array. Pure sync compute, no IO.

        Requires ``chunk_transform`` (all codecs must support sync).
        Raises ``RuntimeError`` if called without a chunk transform.
        """
        if raw is None:
            return None
        if self.chunk_transform is None:
            raise RuntimeError(
                "Cannot call _transform_read without a ChunkTransform. "
                "All codecs must implement SupportsSyncCodec for sync compute."
            )
        return self.chunk_transform.decode_chunk(raw)

    def _transform_write(
        self,
        existing: Buffer | None,
        chunk_spec: ArraySpec,
        chunk_selection: SelectorTuple,
        out_selection: SelectorTuple,
        value: NDBuffer,
        drop_axes: tuple[int, ...],
    ) -> Buffer | None:
        """Decode existing, merge new data, re-encode. Pure sync compute, no IO.

        Requires ``chunk_transform`` (all codecs must support sync).
        Raises ``RuntimeError`` if called without a chunk transform.
        """
        if self.chunk_transform is None:
            raise RuntimeError(
                "Cannot call _transform_write without a ChunkTransform. "
                "All codecs must implement SupportsSyncCodec for sync compute."
            )

        if existing is not None:
            chunk_array: NDBuffer | None = self.chunk_transform.decode_chunk(existing)
        else:
            chunk_array = None

        if chunk_array is None:
            chunk_array = chunk_spec.prototype.nd_buffer.create(
                shape=chunk_spec.shape,
                dtype=chunk_spec.dtype.to_native_dtype(),
                fill_value=fill_value_or_default(chunk_spec),
            )

        # Merge new data
        if drop_axes:
            chunk_value = value[out_selection]
            chunk_array[chunk_selection] = chunk_value.squeeze(axis=drop_axes)
        else:
            chunk_array[chunk_selection] = value[out_selection]

        return self.chunk_transform.encode_chunk(chunk_array)

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

    async def read(
        self,
        batch_info: Iterable[tuple[ByteGetter, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        out: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> tuple[GetResult, ...]:
        batch = list(batch_info)
        if not batch:
            return ()

        # Phase 1: IO — fetch all raw bytes concurrently
        raw_buffers: list[Buffer | None] = await concurrent_map(
            [(bg, cs.prototype) for bg, cs, *_ in batch],
            lambda bg, proto: bg.get(prototype=proto),
            config.get("async.concurrency"),
        )

        # Phase 2: compute — decode all chunks
        if self.chunk_transform is not None:
            # All codecs support sync — offload to threads for parallelism
            import asyncio

            decoded: list[NDBuffer | None] = list(await asyncio.gather(*[
                asyncio.to_thread(self._transform_read, raw, cs)
                for raw, (_, cs, *_) in zip(raw_buffers, batch, strict=True)
            ]))
        else:
            # Some codecs are async-only — decode inline (no threading, no deadlock)
            decoded = list(await self.decode(
                zip(raw_buffers, [cs for _, cs, *_ in batch], strict=False)
            ))

        # Phase 3: scatter
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
        if self.chunk_transform is not None:
            # All codecs support sync — offload to threads for parallelism
            import asyncio

            blobs: list[Buffer | None] = list(await asyncio.gather(*[
                asyncio.to_thread(
                    self._transform_write, existing, cs, csel, osel, value, drop_axes
                )
                for existing, (_, cs, csel, osel, _) in zip(
                    existing_buffers, batch, strict=True
                )
            ]))
        else:
            # Some codecs are async-only — encode inline (no threading, no deadlock)
            blobs = []
            for existing, (_, cs, csel, osel, _) in zip(
                existing_buffers, batch, strict=True
            ):
                if existing is not None:
                    chunk_array_batch = await self.decode([(existing, cs)])
                    chunk_array = next(iter(chunk_array_batch))
                else:
                    chunk_array = None

                if chunk_array is None:
                    chunk_array = cs.prototype.nd_buffer.create(
                        shape=cs.shape,
                        dtype=cs.dtype.to_native_dtype(),
                        fill_value=fill_value_or_default(cs),
                    )

                if drop_axes:
                    chunk_value = value[osel]
                    chunk_array[csel] = chunk_value.squeeze(axis=drop_axes)
                else:
                    chunk_array[csel] = value[osel]

                encoded_batch = await self.encode([(chunk_array, cs)])
                blobs.append(next(iter(encoded_batch)))

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

    def read_sync(
        self,
        batch_info: Iterable[tuple[ByteGetter, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        out: NDBuffer,
        drop_axes: tuple[int, ...] = (),
        n_workers: int = 0,
    ) -> None:
        """Synchronous read. Same three phases as async, different IO wrapper."""
        batch = list(batch_info)
        if not batch:
            return

        # Phase 1: IO — fetch all raw bytes serially
        raw_buffers: list[Buffer | None] = [
            bg.get_sync(prototype=cs.prototype) for bg, cs, *_ in batch
        ]

        # Phase 2: compute — decode (optionally threaded)
        specs = [cs for _, cs, *_ in batch]
        if n_workers > 0 and len(batch) > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                decoded = list(pool.map(self._transform_read, raw_buffers, specs))
        else:
            decoded = [
                self._transform_read(raw, cs)
                for raw, cs in zip(raw_buffers, specs, strict=True)
            ]

        # Phase 3: scatter
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
            None if ic else bs.get_sync(prototype=cs.prototype)
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
                bs.delete_sync()
            else:
                bs.set_sync(blob)
