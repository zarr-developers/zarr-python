from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import islice, pairwise
from typing import TYPE_CHECKING, Any, TypeVar, cast
from warnings import warn

from zarr.abc.codec import (
    ArrayArrayCodec,
    ArrayBytesCodec,
    ArrayBytesCodecPartialDecodeMixin,
    ArrayBytesCodecPartialEncodeMixin,
    BytesBytesCodec,
    Codec,
    CodecPipeline,
    SupportsSyncCodec,
)
from zarr.core.common import concurrent_map, product
from zarr.core.config import config
from zarr.core.indexing import SelectorTuple, is_scalar
from zarr.errors import ZarrUserWarning
from zarr.registry import register_pipeline

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from typing import Self

    from zarr.abc.store import ByteGetter, ByteSetter
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer, BufferPrototype, NDBuffer
    from zarr.core.chunk_grids import ChunkGrid
    from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType

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


# ---------------------------------------------------------------------------
# Thread-pool infrastructure for synchronous codec paths
# ---------------------------------------------------------------------------

_MIN_CHUNK_NBYTES_FOR_POOL = 100_000  # 100 KB


def _get_codec_worker_config() -> tuple[bool, int, int]:
    """Read the ``threading.codec_workers`` config.

    Returns
    -------
    tuple[bool, int, int]
        ``(enabled, min_workers, max_workers)``
    """
    codec_workers = config.get("threading.codec_workers")
    enabled: bool = codec_workers.get("enabled", True)
    min_workers: int = codec_workers.get("min", 0)
    max_workers: int = max(codec_workers.get("max") or os.cpu_count() or 4, min_workers)
    return enabled, min_workers, max_workers


def _choose_workers(n_chunks: int, chunk_nbytes: int, codecs: Iterable[Codec]) -> int:
    """Decide how many thread-pool workers to use (0 = don't use pool)."""
    if getattr(_thread_local, "in_pool_worker", False):
        return 0

    enabled, min_workers, max_workers = _get_codec_worker_config()
    if not enabled:
        return 0

    if n_chunks < 2:
        return min_workers

    if not any(isinstance(c, BytesBytesCodec) for c in codecs) and min_workers == 0:
        return 0
    if chunk_nbytes < _MIN_CHUNK_NBYTES_FOR_POOL and min_workers == 0:
        return 0

    return max(min_workers, min(n_chunks, max_workers))


def _get_pool() -> ThreadPoolExecutor:
    """Get the module-level thread pool, creating it lazily."""
    global _pool
    if _pool is None:
        _, _, max_workers = _get_codec_worker_config()
        _pool = ThreadPoolExecutor(max_workers=max_workers)
    return _pool


_pool: ThreadPoolExecutor | None = None
_thread_local = threading.local()


def _mark_pool_worker(fn: Callable[..., T]) -> Callable[..., T]:
    """Wrap *fn* so that ``_thread_local.in_pool_worker`` is ``True`` while it runs."""

    def wrapper(*args: Any, **kwargs: Any) -> T:
        _thread_local.in_pool_worker = True
        try:
            return fn(*args, **kwargs)
        finally:
            _thread_local.in_pool_worker = False

    return wrapper


_DELETED = object()


@dataclass(slots=True, kw_only=True)
class ChunkTransform:
    """A stored chunk, modeled as a layered array.

    Each layer corresponds to one ArrayArrayCodec and the ArraySpec
    at its input boundary.  ``layers[0]`` is the outermost (user-visible)
    transform; after the last layer comes the ArrayBytesCodec.

    The chunk's ``shape`` and ``dtype`` reflect the representation
    **after** all ArrayArrayCodec layers have been applied — i.e. the
    spec that feeds the ArrayBytesCodec.
    """

    codecs: tuple[Codec, ...]
    array_spec: ArraySpec

    # Each element is (ArrayArrayCodec, input_spec_for_that_codec).
    layers: tuple[tuple[ArrayArrayCodec, ArraySpec], ...] = field(
        init=False, repr=False, compare=False
    )
    _ab_codec: ArrayBytesCodec = field(init=False, repr=False, compare=False)
    _ab_spec: ArraySpec = field(init=False, repr=False, compare=False)
    _bb_codecs: tuple[BytesBytesCodec, ...] = field(init=False, repr=False, compare=False)
    _all_sync: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        aa, ab, bb = codecs_from_list(list(self.codecs))

        layers: tuple[tuple[ArrayArrayCodec, ArraySpec], ...] = ()
        spec = self.array_spec
        for aa_codec in aa:
            layers = (*layers, (aa_codec, spec))
            spec = aa_codec.resolve_metadata(spec)

        self.layers = layers
        self._ab_codec = ab
        self._ab_spec = spec
        self._bb_codecs = bb
        self._all_sync = all(isinstance(c, SupportsSyncCodec) for c in self.codecs)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape after all ArrayArrayCodec layers (input to the ArrayBytesCodec)."""
        return self._ab_spec.shape

    @property
    def dtype(self) -> ZDType[TBaseDType, TBaseScalar]:
        """Dtype after all ArrayArrayCodec layers (input to the ArrayBytesCodec)."""
        return self._ab_spec.dtype

    @property
    def all_sync(self) -> bool:
        return self._all_sync

    def decode_chunk(
        self,
        chunk_bytes: Buffer,
    ) -> NDBuffer:
        """Decode a single chunk through the full codec chain, synchronously.

        Pure compute -- no IO. Only callable when all codecs support sync.
        """
        bb_out: Any = chunk_bytes
        for bb_codec in reversed(self._bb_codecs):
            bb_out = cast("SupportsSyncCodec", bb_codec)._decode_sync(bb_out, self._ab_spec)

        ab_out: Any = cast("SupportsSyncCodec", self._ab_codec)._decode_sync(bb_out, self._ab_spec)

        for aa_codec, spec in reversed(self.layers):
            ab_out = cast("SupportsSyncCodec", aa_codec)._decode_sync(ab_out, spec)

        return ab_out  # type: ignore[no-any-return]

    def encode_chunk(
        self,
        chunk_array: NDBuffer,
    ) -> Buffer | None:
        """Encode a single chunk through the full codec chain, synchronously.

        Pure compute -- no IO. Only callable when all codecs support sync.
        """
        aa_out: Any = chunk_array

        for aa_codec, spec in self.layers:
            if aa_out is None:
                return None
            aa_out = cast("SupportsSyncCodec", aa_codec)._encode_sync(aa_out, spec)

        if aa_out is None:
            return None
        bb_out: Any = cast("SupportsSyncCodec", self._ab_codec)._encode_sync(aa_out, self._ab_spec)

        for bb_codec in self._bb_codecs:
            if bb_out is None:
                return None
            bb_out = cast("SupportsSyncCodec", bb_codec)._encode_sync(bb_out, self._ab_spec)

        return bb_out  # type: ignore[no-any-return]

    async def decode_chunk_async(
        self,
        chunk_bytes: Buffer,
    ) -> NDBuffer:
        """Decode a single chunk through the full codec chain, asynchronously.

        Needed when the codec chain contains async-only codecs (e.g. nested sharding).
        """
        bb_out: Any = chunk_bytes
        for bb_codec in reversed(self._bb_codecs):
            bb_out = await bb_codec._decode_single(bb_out, self._ab_spec)

        ab_out: Any = await self._ab_codec._decode_single(bb_out, self._ab_spec)

        for aa_codec, spec in reversed(self.layers):
            ab_out = await aa_codec._decode_single(ab_out, spec)

        return ab_out  # type: ignore[no-any-return]

    async def encode_chunk_async(
        self,
        chunk_array: NDBuffer,
    ) -> Buffer | None:
        """Encode a single chunk through the full codec chain, asynchronously.

        Needed when the codec chain contains async-only codecs (e.g. nested sharding).
        """
        aa_out: Any = chunk_array

        for aa_codec, spec in self.layers:
            if aa_out is None:
                return None
            aa_out = await aa_codec._encode_single(aa_out, spec)

        if aa_out is None:
            return None
        bb_out: Any = await self._ab_codec._encode_single(aa_out, self._ab_spec)

        for bb_codec in self._bb_codecs:
            if bb_out is None:
                return None
            bb_out = await bb_codec._encode_single(bb_out, self._ab_spec)

        return bb_out  # type: ignore[no-any-return]

    def compute_encoded_size(self, byte_length: int, array_spec: ArraySpec) -> int:
        for codec in self.codecs:
            byte_length = codec.compute_encoded_size(byte_length, array_spec)
            array_spec = codec.resolve_metadata(array_spec)
        return byte_length


@dataclass(slots=True)
class ReadChunkRequest:
    """A request to read and decode a single chunk."""

    byte_getter: ByteGetter
    transform: ChunkTransform
    chunk_selection: SelectorTuple
    out_selection: SelectorTuple


@dataclass(slots=True)
class WriteChunkRequest:
    """A request to encode and write a single chunk."""

    byte_setter: ByteSetter
    transform: ChunkTransform
    chunk_selection: SelectorTuple
    out_selection: SelectorTuple
    is_complete_chunk: bool


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

    def get_chunk_transform(self, array_spec: ArraySpec) -> ChunkTransform:
        return ChunkTransform(codecs=tuple(self), array_spec=array_spec)

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
        batch_info: Iterable[ReadChunkRequest],
        out: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> None:
        batch_info = list(batch_info)
        if self.supports_partial_decode:
            chunk_array_batch = await self.decode_partial_batch(
                [
                    (req.byte_getter, req.chunk_selection, req.transform.array_spec)
                    for req in batch_info
                ]
            )
            for chunk_array, req in zip(chunk_array_batch, batch_info, strict=False):
                if chunk_array is not None:
                    out[req.out_selection] = chunk_array
                else:
                    out[req.out_selection] = fill_value_or_default(req.transform.array_spec)
        else:
            chunk_bytes_batch = await concurrent_map(
                [(req.byte_getter, req.transform.array_spec.prototype) for req in batch_info],
                lambda byte_getter, prototype: byte_getter.get(prototype),
                config.get("async.concurrency"),
            )

            async def _decode_one(
                chunk_bytes: Buffer | None, req: ReadChunkRequest
            ) -> NDBuffer | None:
                if chunk_bytes is None:
                    return None
                return await req.transform.decode_chunk_async(chunk_bytes)

            chunk_array_batch = await concurrent_map(
                list(zip(chunk_bytes_batch, batch_info, strict=False)),
                _decode_one,
                config.get("async.concurrency"),
            )
            for chunk_array, req in zip(chunk_array_batch, batch_info, strict=False):
                if chunk_array is not None:
                    tmp = chunk_array[req.chunk_selection]
                    if drop_axes != ():
                        tmp = tmp.squeeze(axis=drop_axes)
                    out[req.out_selection] = tmp
                else:
                    out[req.out_selection] = fill_value_or_default(req.transform.array_spec)

    def _merge_chunk_array(
        self,
        existing_chunk_array: NDBuffer | None,
        value: NDBuffer,
        out_selection: SelectorTuple,
        chunk_spec: ArraySpec,
        chunk_selection: SelectorTuple,
        drop_axes: tuple[int, ...],
    ) -> NDBuffer:
        if (
            existing_chunk_array is None
            and value.shape == chunk_spec.shape
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
        batch_info: Iterable[WriteChunkRequest],
        value: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> None:
        batch_info = list(batch_info)
        if self.supports_partial_encode:
            # Pass scalar values as is
            if len(value.shape) == 0:
                await self.encode_partial_batch(
                    [
                        (
                            req.byte_setter,
                            value,
                            req.chunk_selection,
                            req.transform.array_spec,
                        )
                        for req in batch_info
                    ],
                )
            else:
                await self.encode_partial_batch(
                    [
                        (
                            req.byte_setter,
                            value[req.out_selection],
                            req.chunk_selection,
                            req.transform.array_spec,
                        )
                        for req in batch_info
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
                        None if req.is_complete_chunk else req.byte_setter,
                        req.transform.array_spec.prototype,
                    )
                    for req in batch_info
                ],
                _read_key,
                config.get("async.concurrency"),
            )

            async def _decode_one(
                chunk_bytes: Buffer | None, req: WriteChunkRequest
            ) -> NDBuffer | None:
                if chunk_bytes is None:
                    return None
                return await req.transform.decode_chunk_async(chunk_bytes)

            chunk_array_decoded = await concurrent_map(
                list(zip(chunk_bytes_batch, batch_info, strict=False)),
                _decode_one,
                config.get("async.concurrency"),
            )

            chunk_array_merged = [
                self._merge_chunk_array(
                    chunk_array,
                    value,
                    req.out_selection,
                    req.transform.array_spec,
                    req.chunk_selection,
                    drop_axes,
                )
                for chunk_array, req in zip(chunk_array_decoded, batch_info, strict=False)
            ]
            chunk_array_batch: list[NDBuffer | None] = []
            for chunk_array, req in zip(chunk_array_merged, batch_info, strict=False):
                chunk_spec = req.transform.array_spec
                if chunk_array is None:
                    chunk_array_batch.append(None)  # type: ignore[unreachable]
                else:
                    if not chunk_spec.config.write_empty_chunks and chunk_array.all_equal(
                        fill_value_or_default(chunk_spec)
                    ):
                        chunk_array_batch.append(None)
                    else:
                        chunk_array_batch.append(chunk_array)

            async def _encode_one(
                chunk_array: NDBuffer | None, req: WriteChunkRequest
            ) -> Buffer | None:
                if chunk_array is None:
                    return None
                return await req.transform.encode_chunk_async(chunk_array)

            chunk_bytes_batch = await concurrent_map(
                list(zip(chunk_array_batch, batch_info, strict=False)),
                _encode_one,
                config.get("async.concurrency"),
            )

            async def _write_key(byte_setter: ByteSetter, chunk_bytes: Buffer | None) -> None:
                if chunk_bytes is None:
                    await byte_setter.delete()
                else:
                    await byte_setter.set(chunk_bytes)

            await concurrent_map(
                [
                    (req.byte_setter, chunk_bytes)
                    for chunk_bytes, req in zip(chunk_bytes_batch, batch_info, strict=False)
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
        batch_info: Iterable[ReadChunkRequest],
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
        batch_info: Iterable[WriteChunkRequest],
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

    # ------------------------------------------------------------------
    # Synchronous read / write
    # ------------------------------------------------------------------

    @property
    def supports_sync_io(self) -> bool:
        return all(isinstance(c, SupportsSyncCodec) for c in self)

    def _scatter(
        self,
        chunk_arrays: list[NDBuffer | None],
        batch_info_list: list[ReadChunkRequest],
        out: NDBuffer,
        drop_axes: tuple[int, ...],
    ) -> None:
        """Assign decoded chunk arrays into the output buffer."""
        for chunk_array, req in zip(chunk_arrays, batch_info_list, strict=False):
            if chunk_array is not None:
                tmp = chunk_array[req.chunk_selection]
                if drop_axes != ():
                    tmp = tmp.squeeze(axis=drop_axes)
                out[req.out_selection] = tmp
            else:
                out[req.out_selection] = fill_value_or_default(req.transform.array_spec)

    def read_sync(
        self,
        batch_info: Iterable[ReadChunkRequest],
        out: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> None:
        batch_info_list = list(batch_info)
        if not batch_info_list:
            return

        first_spec = batch_info_list[0].transform.array_spec
        chunk_nbytes = product(first_spec.shape) * first_spec.dtype.to_native_dtype().itemsize
        n_workers = _choose_workers(len(batch_info_list), chunk_nbytes, self)

        if n_workers > 0:
            # Threaded: fetch all bytes, decode in parallel, scatter.
            chunk_bytes_list: list[Buffer | None] = [
                req.byte_getter.get_sync(prototype=req.transform.array_spec.prototype)  # type: ignore[attr-defined]
                for req in batch_info_list
            ]
            pool = _get_pool()
            chunk_arrays: list[NDBuffer | None] = list(
                pool.map(
                    _mark_pool_worker(
                        lambda cb, req: req.transform.decode_chunk(cb) if cb is not None else None
                    ),
                    chunk_bytes_list,
                    batch_info_list,
                )
            )
            self._scatter(chunk_arrays, batch_info_list, out, drop_axes)
        else:
            # Non-threaded: prepare_read_sync handles IO + decode per chunk.
            ab_codec = self.array_bytes_codec
            for req in batch_info_list:
                result = ab_codec.prepare_read_sync(
                    req.byte_getter,
                    req.chunk_selection,
                    req.transform,
                )
                if result is not None:
                    if drop_axes != ():
                        result = result.squeeze(axis=drop_axes)
                    out[req.out_selection] = result
                else:
                    out[req.out_selection] = fill_value_or_default(req.transform.array_spec)

    @staticmethod
    def _write_chunk_compute(
        existing_bytes: Buffer | None,
        req: WriteChunkRequest,
        value: NDBuffer,
        drop_axes: tuple[int, ...],
    ) -> Buffer | None | object:
        """Per-chunk compute for the threaded write path.

        Returns encoded bytes, ``None`` for a no-op, or ``_DELETED``
        to signal that the key should be removed.
        """
        chunk_spec = req.transform.array_spec
        existing_chunk_array: NDBuffer | None = None
        if existing_bytes is not None:
            existing_chunk_array = req.transform.decode_chunk(existing_bytes)

        # Merge
        if (
            existing_chunk_array is None
            and value.shape == chunk_spec.shape
            and value[req.out_selection].shape == chunk_spec.shape
        ):
            merged = value
        else:
            if existing_chunk_array is None:
                chunk_array = chunk_spec.prototype.nd_buffer.create(
                    shape=chunk_spec.shape,
                    dtype=chunk_spec.dtype.to_native_dtype(),
                    order=chunk_spec.order,
                    fill_value=fill_value_or_default(chunk_spec),
                )
            else:
                chunk_array = existing_chunk_array.copy()

            if req.chunk_selection == () or is_scalar(
                value.as_ndarray_like(), chunk_spec.dtype.to_native_dtype()
            ):
                chunk_value = value
            else:
                chunk_value = value[req.out_selection]
                if drop_axes != ():
                    item = tuple(
                        None if idx in drop_axes else slice(None) for idx in range(chunk_spec.ndim)
                    )
                    chunk_value = chunk_value[item]
            chunk_array[req.chunk_selection] = chunk_value
            merged = chunk_array

        # Check write_empty_chunks
        if not chunk_spec.config.write_empty_chunks and merged.all_equal(
            fill_value_or_default(chunk_spec)
        ):
            return _DELETED

        return req.transform.encode_chunk(merged)

    def write_sync(
        self,
        batch_info: Iterable[WriteChunkRequest],
        value: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> None:
        batch_info_list = list(batch_info)
        if not batch_info_list:
            return

        first_spec = batch_info_list[0].transform.array_spec
        chunk_nbytes = product(first_spec.shape) * first_spec.dtype.to_native_dtype().itemsize
        n_workers = _choose_workers(len(batch_info_list), chunk_nbytes, self)

        if n_workers > 0:
            # Threaded: fetch existing, compute in parallel, write results.
            existing_bytes_list: list[Buffer | None] = [
                req.byte_setter.get_sync(prototype=req.transform.array_spec.prototype)  # type: ignore[attr-defined]
                if not req.is_complete_chunk
                else None
                for req in batch_info_list
            ]
            pool = _get_pool()
            n = len(batch_info_list)
            encoded_list: list[Buffer | None | object] = list(
                pool.map(
                    _mark_pool_worker(self._write_chunk_compute),
                    existing_bytes_list,
                    batch_info_list,
                    [value] * n,
                    [drop_axes] * n,
                )
            )
            for encoded, req in zip(encoded_list, batch_info_list, strict=False):
                if encoded is _DELETED:
                    req.byte_setter.delete_sync()  # type: ignore[attr-defined]
                elif encoded is not None:
                    req.byte_setter.set_sync(encoded)  # type: ignore[attr-defined]
        else:
            # Non-threaded: prepare/compute/finalize per chunk.
            ab_codec = self.array_bytes_codec
            for req in batch_info_list:
                prepared = ab_codec.prepare_write_sync(
                    req.byte_setter,
                    req.transform,
                    req.chunk_selection,
                    req.out_selection,
                    req.is_complete_chunk,
                )
                for coords, chunk_sel, out_sel, _is_complete in prepared.indexer:
                    existing_inner = prepared.chunk_dict.get(coords)
                    if existing_inner is not None:
                        existing_array = req.transform.decode_chunk(existing_inner)
                    else:
                        existing_array = None
                    merged = self._merge_chunk_array(
                        existing_array,
                        value,
                        out_sel,
                        req.transform.array_spec,
                        chunk_sel,
                        drop_axes,
                    )
                    inner_spec = req.transform.array_spec
                    if not inner_spec.config.write_empty_chunks and merged.all_equal(
                        fill_value_or_default(inner_spec)
                    ):
                        prepared.chunk_dict[coords] = None
                    else:
                        prepared.chunk_dict[coords] = req.transform.encode_chunk(merged)

                ab_codec.finalize_write_sync(prepared, req.transform, req.byte_setter)


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
