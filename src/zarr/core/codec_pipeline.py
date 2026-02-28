from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import pairwise
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
# Thread pool for parallel codec compute
# ---------------------------------------------------------------------------

# Minimum chunk size (in bytes) to consider using the thread pool.
# Below this, per-chunk codec work is too small to offset dispatch overhead.
_MIN_CHUNK_NBYTES_FOR_POOL = 100_000  # 100 KB


def _get_codec_worker_config() -> tuple[bool, int, int]:
    """Read the ``threading.codec_workers`` config.

    Returns (enabled, min_workers, max_workers).
    """
    codec_workers = config.get("threading.codec_workers")
    enabled: bool = codec_workers.get("enabled", True)
    min_workers: int = codec_workers.get("min", 0)
    max_workers: int = max(codec_workers.get("max") or os.cpu_count() or 4, min_workers)
    return enabled, min_workers, max_workers


def _choose_workers(n_chunks: int, chunk_nbytes: int, codecs: Iterable[Codec]) -> int:
    """Decide how many thread pool workers to use (0 = don't use pool).

    Respects ``threading.codec_workers`` config:
    - ``enabled``: if False, always returns 0.
    - ``min``: floor for the number of workers.
    - ``max``: ceiling for the number of workers (default: ``os.cpu_count()``).

    Returns 0 if already running on a pool worker thread (prevents deadlock).
    """
    # Prevent nested pool usage: if we're already on a pool worker, don't
    # submit more work to the same pool (classic nested-pool deadlock).
    if getattr(_thread_local, "in_pool_worker", False):
        return 0

    enabled, min_workers, max_workers = _get_codec_worker_config()
    if not enabled:
        return 0

    if n_chunks < 2:
        return min_workers

    # Only use the pool when at least one codec does real work
    # (BytesBytesCodec = compression/checksum, which releases the GIL in C)
    # and the chunks are large enough to offset dispatch overhead.
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

# Thread-local flag to prevent nested thread pool deadlock.
# When a pool worker is running codec compute, inner pipelines (e.g. sharding)
# must not submit work to the same pool.
_thread_local = threading.local()


def _mark_pool_worker(fn: Callable[..., T]) -> Callable[..., T]:
    """Wrap *fn* so that ``_thread_local.in_pool_worker`` is ``True`` while it runs.

    Used around functions dispatched to the thread pool so that nested
    ``_choose_workers`` calls (e.g. from sharding) return 0 instead of
    deadlocking by submitting more work to the same pool.
    """

    def wrapper(*args: Any, **kwargs: Any) -> T:
        _thread_local.in_pool_worker = True
        try:
            return fn(*args, **kwargs)
        finally:
            _thread_local.in_pool_worker = False

    return wrapper


# Sentinel to distinguish "delete this key" from None.
_DELETED = object()


@dataclass(frozen=True)
class CodecChain:
    """Lightweight codec chain: array-array -> array-bytes -> bytes-bytes.

    Pure compute only — no IO methods, no threading, no batching.
    The pipeline accesses IO methods (prepare_read, prepare_write)
    via ``codec_chain.array_bytes_codec`` directly.
    """

    array_array_codecs: tuple[ArrayArrayCodec, ...]
    array_bytes_codec: ArrayBytesCodec
    bytes_bytes_codecs: tuple[BytesBytesCodec, ...]

    _all_sync: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_all_sync",
            all(isinstance(c, SupportsSyncCodec) for c in self),
        )

    def __iter__(self) -> Iterator[Codec]:
        yield from self.array_array_codecs
        yield self.array_bytes_codec
        yield from self.bytes_bytes_codecs

    @classmethod
    def from_codecs(cls, codecs: Iterable[Codec]) -> CodecChain:
        aa, ab, bb = codecs_from_list(list(codecs))
        return cls(array_array_codecs=aa, array_bytes_codec=ab, bytes_bytes_codecs=bb)

    def resolve_metadata_chain(
        self, chunk_spec: ArraySpec
    ) -> tuple[
        list[tuple[ArrayArrayCodec, ArraySpec]],
        tuple[ArrayBytesCodec, ArraySpec],
        list[tuple[BytesBytesCodec, ArraySpec]],
    ]:
        """Resolve metadata through the codec chain for a single chunk_spec."""
        aa_codecs_with_spec: list[tuple[ArrayArrayCodec, ArraySpec]] = []
        spec = chunk_spec
        for aa_codec in self.array_array_codecs:
            aa_codecs_with_spec.append((aa_codec, spec))
            spec = aa_codec.resolve_metadata(spec)

        ab_codec_with_spec = (self.array_bytes_codec, spec)
        spec = self.array_bytes_codec.resolve_metadata(spec)

        bb_codecs_with_spec: list[tuple[BytesBytesCodec, ArraySpec]] = []
        for bb_codec in self.bytes_bytes_codecs:
            bb_codecs_with_spec.append((bb_codec, spec))
            spec = bb_codec.resolve_metadata(spec)

        return (aa_codecs_with_spec, ab_codec_with_spec, bb_codecs_with_spec)

    def decode_chunk(
        self,
        chunk_bytes: Buffer | None,
        chunk_spec: ArraySpec,
        aa_chain: list[tuple[ArrayArrayCodec, ArraySpec]] | None = None,
        ab_pair: tuple[ArrayBytesCodec, ArraySpec] | None = None,
        bb_chain: list[tuple[BytesBytesCodec, ArraySpec]] | None = None,
    ) -> NDBuffer | None:
        """Decode a single chunk through the full codec chain, synchronously.

        Pure compute — no IO. Only callable when all codecs support sync.

        The optional ``aa_chain``, ``ab_pair``, ``bb_chain`` parameters allow
        pre-resolved metadata to be reused across many chunks with the same spec.
        If not provided, ``resolve_metadata_chain`` is called internally.
        """
        if chunk_bytes is None:
            return None

        if aa_chain is None or ab_pair is None or bb_chain is None:
            aa_chain, ab_pair, bb_chain = self.resolve_metadata_chain(chunk_spec)

        bb_out: Any = chunk_bytes
        for bb_codec, spec in reversed(bb_chain):
            bb_out = cast("SupportsSyncCodec", bb_codec)._decode_sync(bb_out, spec)

        ab_codec, ab_spec = ab_pair
        ab_out: Any = cast("SupportsSyncCodec", ab_codec)._decode_sync(bb_out, ab_spec)

        for aa_codec, spec in reversed(aa_chain):
            ab_out = cast("SupportsSyncCodec", aa_codec)._decode_sync(ab_out, spec)

        return ab_out  # type: ignore[no-any-return]

    def encode_chunk(
        self,
        chunk_array: NDBuffer | None,
        chunk_spec: ArraySpec,
    ) -> Buffer | None:
        """Encode a single chunk through the full codec chain, synchronously.

        Pure compute — no IO. Only callable when all codecs support sync.
        """
        if chunk_array is None:
            return None

        spec = chunk_spec
        aa_out: Any = chunk_array

        for aa_codec in self.array_array_codecs:
            if aa_out is None:
                return None
            aa_out = cast("SupportsSyncCodec", aa_codec)._encode_sync(aa_out, spec)
            spec = aa_codec.resolve_metadata(spec)

        if aa_out is None:
            return None
        bb_out: Any = cast("SupportsSyncCodec", self.array_bytes_codec)._encode_sync(aa_out, spec)
        spec = self.array_bytes_codec.resolve_metadata(spec)

        for bb_codec in self.bytes_bytes_codecs:
            if bb_out is None:
                return None
            bb_out = cast("SupportsSyncCodec", bb_codec)._encode_sync(bb_out, spec)
            spec = bb_codec.resolve_metadata(spec)

        return bb_out  # type: ignore[no-any-return]

    def compute_encoded_size(self, byte_length: int, array_spec: ArraySpec) -> int:
        for codec in self:
            byte_length = codec.compute_encoded_size(byte_length, array_spec)
            array_spec = codec.resolve_metadata(array_spec)
        return byte_length

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        for codec in self:
            chunk_spec = codec.resolve_metadata(chunk_spec)
        return chunk_spec


# ---------------------------------------------------------------------------
# Module-level helpers used by both BatchedCodecPipeline and ArrayBytesCodec
# ---------------------------------------------------------------------------


def _merge_chunk_array(
    existing_chunk_array: NDBuffer | None,
    value: NDBuffer,
    out_selection: SelectorTuple,
    chunk_spec: ArraySpec,
    chunk_selection: SelectorTuple,
    is_complete_chunk: bool,
    drop_axes: tuple[int, ...],
) -> NDBuffer:
    """Merge new data into an existing (or freshly-created) chunk array."""
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


def _write_chunk_compute_default(
    existing_bytes: Buffer | None,
    chunk_spec: ArraySpec,
    chunk_selection: SelectorTuple,
    out_selection: SelectorTuple,
    is_complete_chunk: bool,
    value: NDBuffer,
    drop_axes: tuple[int, ...],
    codec_chain: CodecChain,
    aa_chain: list[tuple[ArrayArrayCodec, ArraySpec]] | None = None,
    ab_pair: tuple[ArrayBytesCodec, ArraySpec] | None = None,
    bb_chain: list[tuple[BytesBytesCodec, ArraySpec]] | None = None,
) -> Buffer | None | object:
    """Per-chunk compute for write: decode existing -> merge -> encode.

    Returns the encoded chunk bytes, or ``_DELETED`` if the chunk should be
    removed from the store.
    """
    existing_array: NDBuffer | None = None
    if existing_bytes is not None:
        if aa_chain is None or ab_pair is None or bb_chain is None:
            aa_chain, ab_pair, bb_chain = codec_chain.resolve_metadata_chain(chunk_spec)
        existing_array = codec_chain.decode_chunk(
            existing_bytes, chunk_spec, aa_chain, ab_pair, bb_chain
        )

    chunk_array: NDBuffer | None = _merge_chunk_array(
        existing_array,
        value,
        out_selection,
        chunk_spec,
        chunk_selection,
        is_complete_chunk,
        drop_axes,
    )

    if (
        chunk_array is not None
        and not chunk_spec.config.write_empty_chunks
        and chunk_array.all_equal(fill_value_or_default(chunk_spec))
    ):
        chunk_array = None

    if chunk_array is None:
        return _DELETED
    chunk_bytes = codec_chain.encode_chunk(chunk_array, chunk_spec)
    if chunk_bytes is None:
        return _DELETED
    return chunk_bytes


@dataclass(frozen=True)
class BatchedCodecPipeline(CodecPipeline):
    """Codec pipeline that automatically selects the optimal execution strategy.

    When all codecs support synchronous operations and the store supports
    sync IO, this pipeline runs the entire read/write path on the calling
    thread with zero async overhead, using a thread pool for parallel codec
    compute on multi-chunk operations.

    When the store requires async IO (e.g. cloud stores), this pipeline uses
    the async path with concurrent IO overlap via ``concurrent_map``.

    This automatic dispatch eliminates the need for users to choose between
    pipeline implementations — the right strategy is selected based on codec
    and store capabilities.
    """

    codec_chain: CodecChain
    batch_size: int | None = None

    def __init__(
        self,
        *,
        codec_chain: CodecChain | None = None,
        array_array_codecs: tuple[ArrayArrayCodec, ...] | None = None,
        array_bytes_codec: ArrayBytesCodec | None = None,
        bytes_bytes_codecs: tuple[BytesBytesCodec, ...] | None = None,
        batch_size: int | None = None,
    ) -> None:
        if batch_size is not None:
            warn(
                "The 'batch_size' parameter is deprecated and has no effect. "
                "Batch size is now determined automatically.",
                FutureWarning,
                stacklevel=2,
            )
        object.__setattr__(self, "batch_size", batch_size)

        if codec_chain is not None:
            object.__setattr__(self, "codec_chain", codec_chain)
        elif array_bytes_codec is not None:
            object.__setattr__(
                self,
                "codec_chain",
                CodecChain(
                    array_array_codecs=array_array_codecs or (),
                    array_bytes_codec=array_bytes_codec,
                    bytes_bytes_codecs=bytes_bytes_codecs or (),
                ),
            )
        else:
            raise ValueError("Either codec_chain or array_bytes_codec must be provided.")

    @property
    def array_array_codecs(self) -> tuple[ArrayArrayCodec, ...]:
        return self.codec_chain.array_array_codecs

    @property
    def array_bytes_codec(self) -> ArrayBytesCodec:
        return self.codec_chain.array_bytes_codec

    @property
    def bytes_bytes_codecs(self) -> tuple[BytesBytesCodec, ...]:
        return self.codec_chain.bytes_bytes_codecs

    @property
    def _all_sync(self) -> bool:
        return self.codec_chain._all_sync

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        return type(self).from_codecs(c.evolve_from_array_spec(array_spec=array_spec) for c in self)

    @classmethod
    def from_codecs(cls, codecs: Iterable[Codec]) -> Self:
        return cls(codec_chain=CodecChain.from_codecs(codecs))

    @property
    def supports_partial_decode(self) -> bool:
        return (len(self.array_array_codecs) + len(self.bytes_bytes_codecs)) == 0 and isinstance(
            self.array_bytes_codec, ArrayBytesCodecPartialDecodeMixin
        )

    @property
    def supports_partial_encode(self) -> bool:
        return (len(self.array_array_codecs) + len(self.bytes_bytes_codecs)) == 0 and isinstance(
            self.array_bytes_codec, ArrayBytesCodecPartialEncodeMixin
        )

    def __iter__(self) -> Iterator[Codec]:
        yield from self.codec_chain

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
        return self.codec_chain.compute_encoded_size(byte_length, array_spec)

    # -------------------------------------------------------------------
    # Batched async decode/encode (layer-by-layer across all chunks)
    # -------------------------------------------------------------------

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

    # -------------------------------------------------------------------
    # Top-level decode / encode
    # -------------------------------------------------------------------

    async def decode(
        self,
        chunk_bytes_and_specs: Iterable[tuple[Buffer | None, ArraySpec]],
    ) -> Iterable[NDBuffer | None]:
        items = list(chunk_bytes_and_specs)
        if not items:
            return []

        if self._all_sync:
            # All codecs support sync -- run the full chain inline (no threading).
            _, first_spec = items[0]
            aa_chain, ab_pair, bb_chain = self.codec_chain.resolve_metadata_chain(first_spec)
            return [
                self.codec_chain.decode_chunk(chunk_bytes, chunk_spec, aa_chain, ab_pair, bb_chain)
                for chunk_bytes, chunk_spec in items
            ]

        # Async fallback: layer-by-layer across all chunks.
        return list(await self.decode_batch(items))

    async def encode(
        self,
        chunk_arrays_and_specs: Iterable[tuple[NDBuffer | None, ArraySpec]],
    ) -> Iterable[Buffer | None]:
        items = list(chunk_arrays_and_specs)
        if not items:
            return []

        if self._all_sync:
            # All codecs support sync -- run the full chain inline (no threading).
            return [
                self.codec_chain.encode_chunk(chunk_array, chunk_spec)
                for chunk_array, chunk_spec in items
            ]

        # Async fallback: layer-by-layer across all chunks.
        return list(await self.encode_batch(items))

    # -------------------------------------------------------------------
    # Async read / write (IO overlap via concurrent_map)
    # -------------------------------------------------------------------

    async def read_batch(
        self,
        batch_info: Iterable[tuple[ByteGetter, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        out: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> None:
        batch_info = list(batch_info)

        if self._all_sync:
            _, first_spec, *_ = batch_info[0]
            aa_chain, ab_pair, bb_chain = self.codec_chain.resolve_metadata_chain(first_spec)
            ab_codec = self.array_bytes_codec
            codec_chain = self.codec_chain

            async def _read_chunk(
                byte_getter: ByteGetter,
                chunk_spec: ArraySpec,
                chunk_selection: SelectorTuple,
                out_selection: SelectorTuple,
            ) -> None:
                result = await ab_codec.prepare_read(
                    byte_getter,
                    chunk_spec,
                    chunk_selection,
                    codec_chain,
                    aa_chain,
                    ab_pair,
                    bb_chain,
                )
                if result is not None:
                    if drop_axes != ():
                        result = result.squeeze(axis=drop_axes)
                    out[out_selection] = result
                else:
                    out[out_selection] = fill_value_or_default(chunk_spec)

            await concurrent_map(
                [
                    (byte_getter, chunk_spec, chunk_selection, out_selection)
                    for byte_getter, chunk_spec, chunk_selection, out_selection, _ in batch_info
                ],
                _read_chunk,
                config.get("async.concurrency"),
            )
        else:
            # Async fallback: fetch all → decode all (async codec API) → scatter.
            # Used for codecs that don't implement _decode_sync (e.g. numcodecs).

            async def _fetch(byte_getter: ByteGetter, prototype: BufferPrototype) -> Buffer | None:
                return await byte_getter.get(prototype=prototype)

            chunk_bytes_batch = await concurrent_map(
                [(byte_getter, chunk_spec.prototype) for byte_getter, chunk_spec, *_ in batch_info],
                _fetch,
                config.get("async.concurrency"),
            )
            chunk_array_batch = await self.decode_batch(
                zip(
                    chunk_bytes_batch,
                    [chunk_spec for _, chunk_spec, *_ in batch_info],
                    strict=False,
                )
            )
            self._scatter(chunk_array_batch, batch_info, out, drop_axes)

    @staticmethod
    def _scatter(
        chunk_array_batch: Iterable[NDBuffer | None],
        batch_info: list[tuple[ByteGetter, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        out: NDBuffer,
        drop_axes: tuple[int, ...],
    ) -> None:
        for chunk_array, (_, chunk_spec, chunk_selection, out_selection, _) in zip(
            chunk_array_batch, batch_info, strict=False
        ):
            if chunk_array is not None:
                tmp = chunk_array[chunk_selection]
                if drop_axes != ():
                    tmp = tmp.squeeze(axis=drop_axes)
                out[out_selection] = tmp
            else:
                out[out_selection] = fill_value_or_default(chunk_spec)

    async def read(
        self,
        batch_info: Iterable[tuple[ByteGetter, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        out: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> None:
        await self.read_batch(batch_info, out, drop_axes)

    @staticmethod
    def _merge_chunk_array(
        existing_chunk_array: NDBuffer | None,
        value: NDBuffer,
        out_selection: SelectorTuple,
        chunk_spec: ArraySpec,
        chunk_selection: SelectorTuple,
        is_complete_chunk: bool,
        drop_axes: tuple[int, ...],
    ) -> NDBuffer:
        return _merge_chunk_array(
            existing_chunk_array,
            value,
            out_selection,
            chunk_spec,
            chunk_selection,
            is_complete_chunk,
            drop_axes,
        )

    async def write_batch(
        self,
        batch_info: Iterable[tuple[ByteSetter, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        value: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> None:
        batch_info = list(batch_info)

        if self._all_sync:
            ab_codec = self.array_bytes_codec
            codec_chain = self.codec_chain

            async def _write_chunk(
                byte_setter: ByteSetter,
                chunk_spec: ArraySpec,
                chunk_selection: SelectorTuple,
                out_selection: SelectorTuple,
                _is_complete_chunk: bool,
            ) -> None:
                prepared = await ab_codec.prepare_write(
                    byte_setter,
                    chunk_spec,
                    chunk_selection,
                    out_selection,
                    codec_chain,
                )

                if prepared.is_complete_shard:
                    if prepared.value_selection is not None and not is_scalar(
                        value.as_ndarray_like(),
                        prepared.inner_chunk_spec.dtype.to_native_dtype(),
                    ):
                        prepared.shard_data = value[prepared.value_selection]
                    else:
                        prepared.shard_data = value
                    await ab_codec.finalize_write(prepared, chunk_spec, byte_setter)
                    return

                inner_chain = prepared.inner_codec_chain
                inner_spec = prepared.inner_chunk_spec
                inner_aa, inner_ab, inner_bb = inner_chain.resolve_metadata_chain(inner_spec)

                if prepared.value_selection is not None and not is_scalar(
                    value.as_ndarray_like(), inner_spec.dtype.to_native_dtype()
                ):
                    write_value = value[prepared.value_selection]
                else:
                    write_value = value

                for coords, chunk_sel, out_sel, _is_complete in prepared.indexer:
                    existing_bytes_inner = prepared.chunk_dict.get(coords)
                    if existing_bytes_inner is not None:
                        existing_array = inner_chain.decode_chunk(
                            existing_bytes_inner,
                            inner_spec,
                            inner_aa,
                            inner_ab,
                            inner_bb,
                        )
                    else:
                        existing_array = None
                    merged = _merge_chunk_array(
                        existing_array,
                        write_value,
                        out_sel,
                        inner_spec,
                        chunk_sel,
                        _is_complete,
                        drop_axes,
                    )
                    if not inner_spec.config.write_empty_chunks and merged.all_equal(
                        fill_value_or_default(inner_spec)
                    ):
                        prepared.chunk_dict[coords] = None
                    else:
                        prepared.chunk_dict[coords] = inner_chain.encode_chunk(merged, inner_spec)

                await ab_codec.finalize_write(prepared, chunk_spec, byte_setter)

            await concurrent_map(
                [
                    (byte_setter, chunk_spec, chunk_selection, out_selection, is_complete_chunk)
                    for byte_setter, chunk_spec, chunk_selection, out_selection, is_complete_chunk in batch_info
                ],
                _write_chunk,
                config.get("async.concurrency"),
            )
        else:
            # Async fallback: phased approach for codecs without sync support.
            # Phase 1: Fetch existing chunks for partial writes.

            async def _fetch_existing(
                byte_setter: ByteSetter, chunk_spec: ArraySpec, is_complete_chunk: bool
            ) -> Buffer | None:
                if is_complete_chunk:
                    return None
                return await byte_setter.get(prototype=chunk_spec.prototype)

            existing_bytes_list: list[Buffer | None] = await concurrent_map(
                [
                    (byte_setter, chunk_spec, is_complete_chunk)
                    for byte_setter, chunk_spec, _, _, is_complete_chunk in batch_info
                ],
                _fetch_existing,
                config.get("async.concurrency"),
            )

            # Phase 2: Decode → merge → encode (async codec API).
            decode_items: list[tuple[Buffer | None, ArraySpec]] = [
                (existing_bytes if not is_complete_chunk else None, chunk_spec)
                for existing_bytes, (_, chunk_spec, _, _, is_complete_chunk) in zip(
                    existing_bytes_list, batch_info, strict=False
                )
            ]
            encoded_list = await self._write_batch_compute(
                decode_items, batch_info, value, drop_axes
            )

            # Phase 3: Write encoded chunks to store.
            async def _write_out(byte_setter: ByteSetter, chunk_bytes: Buffer | None) -> None:
                if chunk_bytes is None:
                    await byte_setter.delete()
                else:
                    await byte_setter.set(chunk_bytes)

            await concurrent_map(
                [
                    (byte_setter, chunk_bytes)
                    for (byte_setter, *_), chunk_bytes in zip(
                        batch_info, encoded_list, strict=False
                    )
                ],
                _write_out,
                config.get("async.concurrency"),
            )

    async def _write_batch_compute(
        self,
        decode_items: list[tuple[Buffer | None, ArraySpec]],
        batch_info: list[tuple[ByteSetter, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        value: NDBuffer,
        drop_axes: tuple[int, ...],
    ) -> list[Buffer | None]:
        chunk_array_decoded: Iterable[NDBuffer | None] = await self.decode(decode_items)

        chunk_array_batch = self._merge_and_filter(
            chunk_array_decoded, batch_info, value, drop_axes
        )

        encoded_batch: Iterable[Buffer | None] = await self.encode(
            [
                (chunk_array, chunk_spec)
                for chunk_array, (_, chunk_spec, *_) in zip(
                    chunk_array_batch, batch_info, strict=False
                )
            ]
        )
        return list(encoded_batch)

    def _merge_and_filter(
        self,
        chunk_array_decoded: Iterable[NDBuffer | None],
        batch_info: list[tuple[Any, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        value: NDBuffer,
        drop_axes: tuple[int, ...],
    ) -> list[NDBuffer | None]:
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
        for chunk_array, (_, chunk_spec, *_) in zip(chunk_array_merged, batch_info, strict=False):
            if chunk_array is None:
                chunk_array_batch.append(None)  # type: ignore[unreachable]
            else:
                if not chunk_spec.config.write_empty_chunks and chunk_array.all_equal(
                    fill_value_or_default(chunk_spec)
                ):
                    chunk_array_batch.append(None)
                else:
                    chunk_array_batch.append(chunk_array)
        return chunk_array_batch

    async def write(
        self,
        batch_info: Iterable[tuple[ByteSetter, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        value: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> None:
        await self.write_batch(batch_info, value, drop_axes)

    # -------------------------------------------------------------------
    # Fully synchronous read / write (no event loop)
    # -------------------------------------------------------------------

    @property
    def supports_sync_io(self) -> bool:
        return self._all_sync

    def read_sync(
        self,
        batch_info: Iterable[tuple[Any, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        out: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> None:
        batch_info_list = list(batch_info)
        if not batch_info_list:
            return

        _, first_spec, *_ = batch_info_list[0]
        aa_chain, ab_pair, bb_chain = self.codec_chain.resolve_metadata_chain(first_spec)

        chunk_nbytes = product(first_spec.shape) * getattr(first_spec.dtype, "item_size", 1)
        n_workers = _choose_workers(len(batch_info_list), chunk_nbytes, self)
        if n_workers > 0:
            # Threaded: fetch all, decode in parallel, scatter.
            chunk_bytes_list: list[Buffer | None] = [
                byte_getter.get_sync(prototype=chunk_spec.prototype)
                for byte_getter, chunk_spec, *_ in batch_info_list
            ]
            pool = _get_pool()
            chunk_arrays: list[NDBuffer | None] = list(
                pool.map(
                    _mark_pool_worker(self.codec_chain.decode_chunk),
                    chunk_bytes_list,
                    [chunk_spec for _, chunk_spec, *_ in batch_info_list],
                    [aa_chain] * len(batch_info_list),
                    [ab_pair] * len(batch_info_list),
                    [bb_chain] * len(batch_info_list),
                )
            )
            self._scatter(chunk_arrays, batch_info_list, out, drop_axes)
        else:
            # Non-threaded: prepare_read_sync handles IO + decode.
            # ShardingCodec overrides for optimized partial IO (byte-range reads).
            ab_codec = self.array_bytes_codec
            for byte_getter, chunk_spec, chunk_selection, out_selection, _ in batch_info_list:
                result = ab_codec.prepare_read_sync(
                    byte_getter,
                    chunk_spec,
                    chunk_selection,
                    self.codec_chain,
                    aa_chain,
                    ab_pair,
                    bb_chain,
                )
                if result is not None:
                    if drop_axes != ():
                        result = result.squeeze(axis=drop_axes)
                    out[out_selection] = result
                else:
                    out[out_selection] = fill_value_or_default(chunk_spec)

    def _write_chunk_compute(
        self,
        existing_bytes: Buffer | None,
        chunk_spec: ArraySpec,
        chunk_selection: SelectorTuple,
        out_selection: SelectorTuple,
        is_complete_chunk: bool,
        value: NDBuffer,
        drop_axes: tuple[int, ...],
        aa_chain: list[tuple[ArrayArrayCodec, ArraySpec]] | None = None,
        ab_pair: tuple[ArrayBytesCodec, ArraySpec] | None = None,
        bb_chain: list[tuple[BytesBytesCodec, ArraySpec]] | None = None,
    ) -> Buffer | None | object:
        """Per-chunk compute for write: decode existing -> merge -> encode."""
        return _write_chunk_compute_default(
            existing_bytes,
            chunk_spec,
            chunk_selection,
            out_selection,
            is_complete_chunk,
            value,
            drop_axes,
            self.codec_chain,
            aa_chain,
            ab_pair,
            bb_chain,
        )

    def write_sync(
        self,
        batch_info: Iterable[tuple[Any, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        value: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> None:
        batch_info_list = list(batch_info)
        if not batch_info_list:
            return

        _, first_spec, *_ = batch_info_list[0]
        aa_chain, ab_pair, bb_chain = self.codec_chain.resolve_metadata_chain(first_spec)
        chunk_nbytes = product(first_spec.shape) * getattr(first_spec.dtype, "item_size", 1)
        n_workers = _choose_workers(len(batch_info_list), chunk_nbytes, self)
        if n_workers > 0:
            # Threaded: fetch all, compute in parallel, write all.
            existing_bytes_list: list[Buffer | None] = [
                byte_setter.get_sync(prototype=chunk_spec.prototype)
                if not is_complete_chunk
                else None
                for byte_setter, chunk_spec, _, _, is_complete_chunk in batch_info_list
            ]
            pool = _get_pool()
            n = len(batch_info_list)
            encoded_list: list[Buffer | None | object] = list(
                pool.map(
                    _mark_pool_worker(self._write_chunk_compute),
                    existing_bytes_list,
                    [chunk_spec for _, chunk_spec, *_ in batch_info_list],
                    [chunk_selection for _, _, chunk_selection, _, _ in batch_info_list],
                    [out_selection for _, _, _, out_selection, _ in batch_info_list],
                    [is_complete for _, _, _, _, is_complete in batch_info_list],
                    [value] * n,
                    [drop_axes] * n,
                    [aa_chain] * n,
                    [ab_pair] * n,
                    [bb_chain] * n,
                )
            )
            for encoded, (byte_setter, *_) in zip(encoded_list, batch_info_list, strict=False):
                if encoded is _DELETED:
                    byte_setter.delete_sync()
                else:
                    byte_setter.set_sync(encoded)
        else:
            # Non-threaded: prepare_write_sync handles IO + deserialize.
            # Pipeline does decode/merge/encode loop, then serialize + write.
            ab_codec = self.array_bytes_codec
            for (
                byte_setter,
                chunk_spec,
                chunk_selection,
                out_selection,
                _,
            ) in batch_info_list:
                prepared = ab_codec.prepare_write_sync(
                    byte_setter,
                    chunk_spec,
                    chunk_selection,
                    out_selection,
                    self.codec_chain,
                )

                if prepared.is_complete_shard:
                    # Complete shard: pass the shard value to finalize_write
                    # which encodes and writes in one shot, bypassing the
                    # per-inner-chunk loop.
                    if prepared.value_selection is not None and not is_scalar(
                        value.as_ndarray_like(),
                        prepared.inner_chunk_spec.dtype.to_native_dtype(),
                    ):
                        prepared.shard_data = value[prepared.value_selection]
                    else:
                        prepared.shard_data = value
                    ab_codec.finalize_write_sync(prepared, chunk_spec, byte_setter)
                    continue

                inner_chain = prepared.inner_codec_chain
                inner_spec = prepared.inner_chunk_spec
                inner_aa, inner_ab, inner_bb = inner_chain.resolve_metadata_chain(inner_spec)

                if prepared.value_selection is not None and not is_scalar(
                    value.as_ndarray_like(), inner_spec.dtype.to_native_dtype()
                ):
                    write_value = value[prepared.value_selection]
                else:
                    write_value = value

                for coords, chunk_sel, out_sel, _is_complete in prepared.indexer:
                    existing_bytes_inner = prepared.chunk_dict.get(coords)
                    if existing_bytes_inner is not None:
                        existing_array = inner_chain.decode_chunk(
                            existing_bytes_inner,
                            inner_spec,
                            inner_aa,
                            inner_ab,
                            inner_bb,
                        )
                    else:
                        existing_array = None
                    merged = _merge_chunk_array(
                        existing_array,
                        write_value,
                        out_sel,
                        inner_spec,
                        chunk_sel,
                        _is_complete,
                        drop_axes,
                    )
                    if not inner_spec.config.write_empty_chunks and merged.all_equal(
                        fill_value_or_default(inner_spec)
                    ):
                        prepared.chunk_dict[coords] = None
                    else:
                        prepared.chunk_dict[coords] = inner_chain.encode_chunk(merged, inner_spec)

                ab_codec.finalize_write_sync(prepared, chunk_spec, byte_setter)


def codecs_from_list(
    codecs: Iterable[Codec],
) -> tuple[tuple[ArrayArrayCodec, ...], ArrayBytesCodec, tuple[BytesBytesCodec, ...]]:
    from zarr.codecs.sharding import ShardingCodec

    codecs = list(codecs)
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
