from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeGuard, TypeVar, runtime_checkable

from typing_extensions import ReadOnly, TypedDict

from zarr.abc.metadata import Metadata
from zarr.core.buffer import Buffer, NDBuffer
from zarr.core.common import NamedConfig, concurrent_map
from zarr.core.config import config

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable
    from typing import Self

    from zarr.abc.store import ByteGetter, ByteSetter, Store
    from zarr.core.array_spec import ArraySpec
    from zarr.core.chunk_grids import ChunkGrid
    from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType
    from zarr.core.indexing import ChunkProjection, SelectorTuple
    from zarr.core.metadata import ArrayMetadata

__all__ = [
    "ArrayArrayCodec",
    "ArrayBytesCodec",
    "ArrayBytesCodecPartialDecodeMixin",
    "ArrayBytesCodecPartialEncodeMixin",
    "BaseCodec",
    "BytesBytesCodec",
    "CodecInput",
    "CodecOutput",
    "CodecPipeline",
    "PreparedWrite",
    "SupportsChunkCodec",
    "SupportsSyncCodec",
]

CodecInput = TypeVar("CodecInput", bound=NDBuffer | Buffer)
CodecOutput = TypeVar("CodecOutput", bound=NDBuffer | Buffer)

TName = TypeVar("TName", bound=str, covariant=True)


class CodecJSON_V2(TypedDict, Generic[TName]):
    """The JSON representation of a codec for Zarr V2"""

    id: ReadOnly[TName]


def _check_codecjson_v2(data: object) -> TypeGuard[CodecJSON_V2[str]]:
    return isinstance(data, Mapping) and "id" in data and isinstance(data["id"], str)


CodecJSON_V3 = str | NamedConfig[str, Mapping[str, object]]
"""The JSON representation of a codec for Zarr V3."""

# The widest type we will *accept* for a codec JSON
# This covers v2 and v3
CodecJSON = str | Mapping[str, object]
"""The widest type of JSON-like input that could specify a codec."""


@runtime_checkable
class SupportsSyncCodec(Protocol):
    """Protocol for codecs that support synchronous encode/decode.

    Codecs implementing this protocol provide ``_decode_sync`` and ``_encode_sync``
    methods that perform encoding/decoding without requiring an async event loop.
    """

    def _decode_sync(
        self, chunk_data: NDBuffer | Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer | Buffer: ...

    def _encode_sync(
        self, chunk_data: NDBuffer | Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer | Buffer | None: ...


class SupportsChunkCodec(Protocol):
    """Protocol for objects that can decode/encode whole chunks synchronously.

    [`ChunkTransform`][zarr.core.codec_pipeline.ChunkTransform] satisfies this protocol.
    """

    array_spec: ArraySpec

    def decode_chunk(self, chunk_bytes: Buffer) -> NDBuffer: ...

    def encode_chunk(self, chunk_array: NDBuffer) -> Buffer | None: ...


class BaseCodec(Metadata, Generic[CodecInput, CodecOutput]):
    """Generic base class for codecs.

    Codecs can be registered via zarr.codecs.registry.

    Warnings
    --------
    This class is not intended to be directly, please use
    ArrayArrayCodec, ArrayBytesCodec or BytesBytesCodec for subclassing.
    """

    is_fixed_size: bool

    @abstractmethod
    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        """Given an input byte length, this method returns the output byte length.
        Raises a NotImplementedError for codecs with variable-sized outputs (e.g. compressors).

        Parameters
        ----------
        input_byte_length : int
        chunk_spec : ArraySpec

        Returns
        -------
        int
        """
        ...

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        """Computed the spec of the chunk after it has been encoded by the codec.
        This is important for codecs that change the shape, data type or fill value of a chunk.
        The spec will then be used for subsequent codecs in the pipeline.

        Parameters
        ----------
        chunk_spec : ArraySpec

        Returns
        -------
        ArraySpec
        """
        return chunk_spec

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        """Fills in codec configuration parameters that can be automatically
        inferred from the array metadata.

        Parameters
        ----------
        array_spec : ArraySpec

        Returns
        -------
        Self
        """
        return self

    def validate(
        self,
        *,
        shape: tuple[int, ...],
        dtype: ZDType[TBaseDType, TBaseScalar],
        chunk_grid: ChunkGrid,
    ) -> None:
        """Validates that the codec configuration is compatible with the array metadata.
        Raises errors when the codec configuration is not compatible.

        Parameters
        ----------
        shape : tuple[int, ...]
            The array shape
        dtype : np.dtype[Any]
            The array data type
        chunk_grid : ChunkGrid
            The array chunk grid
        """

    async def _decode_single(self, chunk_data: CodecOutput, chunk_spec: ArraySpec) -> CodecInput:
        raise NotImplementedError  # pragma: no cover

    async def decode(
        self,
        chunks_and_specs: Iterable[tuple[CodecOutput | None, ArraySpec]],
    ) -> Iterable[CodecInput | None]:
        """Decodes a batch of chunks.
        Chunks can be None in which case they are ignored by the codec.

        Parameters
        ----------
        chunks_and_specs : Iterable[tuple[CodecOutput | None, ArraySpec]]
            Ordered set of encoded chunks with their accompanying chunk spec.

        Returns
        -------
        Iterable[CodecInput | None]
        """
        return await _batching_helper(self._decode_single, chunks_and_specs)

    async def _encode_single(
        self, chunk_data: CodecInput, chunk_spec: ArraySpec
    ) -> CodecOutput | None:
        raise NotImplementedError  # pragma: no cover

    async def encode(
        self,
        chunks_and_specs: Iterable[tuple[CodecInput | None, ArraySpec]],
    ) -> Iterable[CodecOutput | None]:
        """Encodes a batch of chunks.
        Chunks can be None in which case they are ignored by the codec.

        Parameters
        ----------
        chunks_and_specs : Iterable[tuple[CodecInput | None, ArraySpec]]
            Ordered set of to-be-encoded chunks with their accompanying chunk spec.

        Returns
        -------
        Iterable[CodecOutput | None]
        """
        return await _batching_helper(self._encode_single, chunks_and_specs)


class ArrayArrayCodec(BaseCodec[NDBuffer, NDBuffer]):
    """Base class for array-to-array codecs."""


@dataclass
class PreparedWrite:
    """Result of the prepare phase of a write operation.

    Carries deserialized chunk data and selection metadata between
    [`prepare_write`][zarr.abc.codec.ArrayBytesCodec.prepare_write] (or
    [`prepare_write_sync`][zarr.abc.codec.ArrayBytesCodec.prepare_write_sync])
    and [`finalize_write`][zarr.abc.codec.ArrayBytesCodec.finalize_write] (or
    [`finalize_write_sync`][zarr.abc.codec.ArrayBytesCodec.finalize_write_sync]).

    Attributes
    ----------
    chunk_dict : dict[tuple[int, ...], Buffer | None]
        Per-inner-chunk buffers keyed by chunk coordinates.
    indexer : list[ChunkProjection]
        Mapping from inner-chunk coordinates to value/output selections.
    """

    chunk_dict: dict[tuple[int, ...], Buffer | None]
    indexer: list[ChunkProjection]


class ArrayBytesCodec(BaseCodec[NDBuffer, Buffer]):
    """Base class for array-to-bytes codecs."""

    @property
    def inner_codec_chain(self) -> SupportsChunkCodec | None:
        """The codec chain for decoding inner chunks after deserialization.

        Returns ``None`` by default, meaning the pipeline should use its own
        codec chain. ``ShardingCodec`` overrides this to return its inner
        codec chain.

        Returns
        -------
        SupportsChunkCodec or None
            A [`SupportsChunkCodec`][zarr.abc.codec.SupportsChunkCodec] instance,
            or ``None``.
        """
        return None

    def deserialize(
        self, raw: Buffer | None, chunk_spec: ArraySpec
    ) -> dict[tuple[int, ...], Buffer | None]:
        """Unpack stored bytes into per-inner-chunk buffers.

        The default implementation returns a single-entry dict keyed at
        ``(0,)``. ``ShardingCodec`` overrides this to decode the shard index
        and split the blob into per-chunk buffers.

        Parameters
        ----------
        raw : Buffer or None
            The raw bytes read from the store, or ``None`` if the key was
            absent.
        chunk_spec : ArraySpec
            The [`ArraySpec`][zarr.core.array_spec.ArraySpec] for the chunk.

        Returns
        -------
        dict[tuple[int, ...], Buffer | None]
            Mapping from inner-chunk coordinates to their encoded bytes.
        """
        return {(0,): raw}

    def serialize(
        self,
        chunk_dict: dict[tuple[int, ...], Buffer | None],
        chunk_spec: ArraySpec,
    ) -> Buffer | None:
        """Pack per-inner-chunk buffers into a storage blob.

        The default implementation returns the single entry at ``(0,)``.
        ``ShardingCodec`` overrides this to concatenate chunks and build a
        shard index.

        Parameters
        ----------
        chunk_dict : dict[tuple[int, ...], Buffer | None]
            Mapping from inner-chunk coordinates to their encoded bytes.
        chunk_spec : ArraySpec
            The [`ArraySpec`][zarr.core.array_spec.ArraySpec] for the chunk.

        Returns
        -------
        Buffer or None
            The serialized blob, or ``None`` when all chunks are empty
            (the caller should delete the key).
        """
        return chunk_dict.get((0,))

    # ------------------------------------------------------------------
    # prepare / finalize — sync
    # ------------------------------------------------------------------

    def prepare_read_sync(
        self,
        byte_getter: Any,
        chunk_selection: SelectorTuple,
        codec_chain: SupportsChunkCodec,
    ) -> NDBuffer | None:
        """Read a chunk from the store synchronously, decode it, and
        return the selected region.

        Parameters
        ----------
        byte_getter : Any
            An object supporting ``get_sync`` (e.g.
            [`StorePath`][zarr.storage._common.StorePath]).
        chunk_selection : SelectorTuple
            Selection within the decoded chunk array.
        codec_chain : SupportsChunkCodec
            The [`SupportsChunkCodec`][zarr.abc.codec.SupportsChunkCodec] used to
            decode the chunk. Must carry an ``array_spec`` attribute.

        Returns
        -------
        NDBuffer or None
            The decoded chunk data at *chunk_selection*, or ``None`` if the
            chunk does not exist in the store.
        """
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
        """Prepare a synchronous write by optionally reading existing data.

        When *replace* is ``False``, the existing chunk bytes are fetched
        from the store so they can be merged with the new data. When
        *replace* is ``True``, the fetch is skipped.

        Parameters
        ----------
        byte_setter : Any
            An object supporting ``get_sync`` and ``set_sync`` (e.g.
            [`StorePath`][zarr.storage._common.StorePath]).
        codec_chain : SupportsChunkCodec
            The [`SupportsChunkCodec`][zarr.abc.codec.SupportsChunkCodec]
            carrying the ``array_spec`` for the chunk.
        chunk_selection : SelectorTuple
            Selection within the chunk being written.
        out_selection : SelectorTuple
            Corresponding selection within the source value array.
        replace : bool
            If ``True``, the write replaces all data in the chunk and no
            read-modify-write is needed. If ``False``, existing data is
            fetched first.

        Returns
        -------
        PreparedWrite
            A [`PreparedWrite`][zarr.abc.codec.PreparedWrite] carrying the
            deserialized chunk data and selection metadata.
        """
        chunk_spec = codec_chain.array_spec
        existing: Buffer | None = None
        if not replace:
            existing = byte_setter.get_sync(prototype=chunk_spec.prototype)
        chunk_dict = self.deserialize(existing, chunk_spec)
        return PreparedWrite(
            chunk_dict=chunk_dict,
            indexer=[
                (  # type: ignore[list-item]
                    (0,),
                    chunk_selection,
                    out_selection,
                    replace,
                )
            ],
        )

    def finalize_write_sync(
        self,
        prepared: PreparedWrite,
        codec_chain: SupportsChunkCodec,
        byte_setter: Any,
    ) -> None:
        """Serialize the prepared chunk data and write it to the store.

        If serialization produces ``None`` (all chunks empty), the key is
        deleted instead.

        Parameters
        ----------
        prepared : PreparedWrite
            The [`PreparedWrite`][zarr.abc.codec.PreparedWrite] returned by
            [`prepare_write_sync`][zarr.abc.codec.ArrayBytesCodec.prepare_write_sync].
        codec_chain : SupportsChunkCodec
            The [`SupportsChunkCodec`][zarr.abc.codec.SupportsChunkCodec]
            carrying the ``array_spec`` for the chunk.
        byte_setter : Any
            An object supporting ``set_sync`` and ``delete_sync`` (e.g.
            [`StorePath`][zarr.storage._common.StorePath]).
        """
        blob = self.serialize(prepared.chunk_dict, codec_chain.array_spec)
        if blob is None:
            byte_setter.delete_sync()
        else:
            byte_setter.set_sync(blob)

    # ------------------------------------------------------------------
    # prepare / finalize — async
    # ------------------------------------------------------------------

    async def prepare_read(
        self,
        byte_getter: Any,
        chunk_selection: SelectorTuple,
        codec_chain: SupportsChunkCodec,
    ) -> NDBuffer | None:
        """Async variant of
        [`prepare_read_sync`][zarr.abc.codec.ArrayBytesCodec.prepare_read_sync].

        Parameters
        ----------
        byte_getter : Any
            An object supporting ``get`` (e.g.
            [`StorePath`][zarr.storage._common.StorePath]).
        chunk_selection : SelectorTuple
            Selection within the decoded chunk array.
        codec_chain : SupportsChunkCodec
            The [`SupportsChunkCodec`][zarr.abc.codec.SupportsChunkCodec] used to
            decode the chunk. Must carry an ``array_spec`` attribute.

        Returns
        -------
        NDBuffer or None
            The decoded chunk data at *chunk_selection*, or ``None`` if the
            chunk does not exist in the store.
        """
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
        """Async variant of
        [`prepare_write_sync`][zarr.abc.codec.ArrayBytesCodec.prepare_write_sync].

        Parameters
        ----------
        byte_setter : Any
            An object supporting ``get`` and ``set`` (e.g.
            [`StorePath`][zarr.storage._common.StorePath]).
        codec_chain : SupportsChunkCodec
            The [`SupportsChunkCodec`][zarr.abc.codec.SupportsChunkCodec]
            carrying the ``array_spec`` for the chunk.
        chunk_selection : SelectorTuple
            Selection within the chunk being written.
        out_selection : SelectorTuple
            Corresponding selection within the source value array.
        replace : bool
            If ``True``, the write replaces all data in the chunk and no
            read-modify-write is needed. If ``False``, existing data is
            fetched first.

        Returns
        -------
        PreparedWrite
            A [`PreparedWrite`][zarr.abc.codec.PreparedWrite] carrying the
            deserialized chunk data and selection metadata.
        """
        chunk_spec = codec_chain.array_spec
        existing: Buffer | None = None
        if not replace:
            existing = await byte_setter.get(prototype=chunk_spec.prototype)
        chunk_dict = self.deserialize(existing, chunk_spec)
        return PreparedWrite(
            chunk_dict=chunk_dict,
            indexer=[
                (  # type: ignore[list-item]
                    (0,),
                    chunk_selection,
                    out_selection,
                    replace,
                )
            ],
        )

    async def finalize_write(
        self,
        prepared: PreparedWrite,
        codec_chain: SupportsChunkCodec,
        byte_setter: Any,
    ) -> None:
        """Async variant of
        [`finalize_write_sync`][zarr.abc.codec.ArrayBytesCodec.finalize_write_sync].

        Parameters
        ----------
        prepared : PreparedWrite
            The [`PreparedWrite`][zarr.abc.codec.PreparedWrite] returned by
            [`prepare_write`][zarr.abc.codec.ArrayBytesCodec.prepare_write].
        codec_chain : SupportsChunkCodec
            The [`SupportsChunkCodec`][zarr.abc.codec.SupportsChunkCodec]
            carrying the ``array_spec`` for the chunk.
        byte_setter : Any
            An object supporting ``set`` and ``delete`` (e.g.
            [`StorePath`][zarr.storage._common.StorePath]).
        """
        blob = self.serialize(prepared.chunk_dict, codec_chain.array_spec)
        if blob is None:
            await byte_setter.delete()
        else:
            await byte_setter.set(blob)


class BytesBytesCodec(BaseCodec[Buffer, Buffer]):
    """Base class for bytes-to-bytes codecs."""


Codec = ArrayArrayCodec | ArrayBytesCodec | BytesBytesCodec


class ArrayBytesCodecPartialDecodeMixin:
    """Mixin for array-to-bytes codecs that implement partial decoding."""

    async def _decode_partial_single(
        self, byte_getter: ByteGetter, selection: SelectorTuple, chunk_spec: ArraySpec
    ) -> NDBuffer | None:
        raise NotImplementedError

    async def decode_partial(
        self,
        batch_info: Iterable[tuple[ByteGetter, SelectorTuple, ArraySpec]],
    ) -> Iterable[NDBuffer | None]:
        """Partially decodes a batch of chunks.
        This method determines parts of a chunk from the slice selection,
        fetches these parts from the store (via ByteGetter) and decodes them.

        Parameters
        ----------
        batch_info : Iterable[tuple[ByteGetter, SelectorTuple, ArraySpec]]
            Ordered set of information about slices of encoded chunks.
            The slice selection determines which parts of the chunk will be fetched.
            The ByteGetter is used to fetch the necessary bytes.
            The chunk spec contains information about the construction of an array from the bytes.

        Returns
        -------
        Iterable[NDBuffer | None]
        """
        return await concurrent_map(
            list(batch_info),
            self._decode_partial_single,
            config.get("async.concurrency"),
        )


class ArrayBytesCodecPartialEncodeMixin:
    """Mixin for array-to-bytes codecs that implement partial encoding."""

    async def _encode_partial_single(
        self,
        byte_setter: ByteSetter,
        chunk_array: NDBuffer,
        selection: SelectorTuple,
        chunk_spec: ArraySpec,
    ) -> None:
        raise NotImplementedError  # pragma: no cover

    async def encode_partial(
        self,
        batch_info: Iterable[tuple[ByteSetter, NDBuffer, SelectorTuple, ArraySpec]],
    ) -> None:
        """Partially encodes a batch of chunks.
        This method determines parts of a chunk from the slice selection, encodes them and
        writes these parts to the store (via ByteSetter).
        If merging with existing chunk data in the store is necessary, this method will
        read from the store first and perform the merge.

        Parameters
        ----------
        batch_info : Iterable[tuple[ByteSetter, NDBuffer, SelectorTuple, ArraySpec]]
            Ordered set of information about slices of to-be-encoded chunks.
            The slice selection determines which parts of the chunk will be encoded.
            The ByteSetter is used to write the necessary bytes and fetch bytes for existing chunk data.
            The chunk spec contains information about the chunk.
        """
        await concurrent_map(
            list(batch_info),
            self._encode_partial_single,
            config.get("async.concurrency"),
        )


class CodecPipeline:
    """Base class for implementing CodecPipeline.
    A CodecPipeline implements the read and write paths for chunk data.
    On the read path, it is responsible for fetching chunks from a store (via ByteGetter),
    decoding them and assembling an output array. On the write path, it encodes the chunks
    and writes them to a store (via ByteSetter)."""

    @abstractmethod
    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        """Fills in codec configuration parameters that can be automatically
        inferred from the array metadata.

        Parameters
        ----------
        array_spec : ArraySpec

        Returns
        -------
        Self
        """
        ...

    @classmethod
    @abstractmethod
    def from_codecs(cls, codecs: Iterable[Codec]) -> Self:
        """Creates a codec pipeline from an iterable of codecs.

        Parameters
        ----------
        codecs : Iterable[Codec]

        Returns
        -------
        Self
        """
        ...

    @classmethod
    def from_array_metadata_and_store(cls, array_metadata: ArrayMetadata, store: Store) -> Self:
        """Creates a codec pipeline from array metadata and a store path.

        Raises NotImplementedError by default, indicating the CodecPipeline must be created with from_codecs instead.

        Parameters
        ----------
        array_metadata : ArrayMetadata
        store : Store

        Returns
        -------
        Self
        """
        raise NotImplementedError(
            f"'{type(cls).__name__}' does not implement CodecPipeline.from_array_metadata_and_store."
        )

    @property
    @abstractmethod
    def supports_partial_decode(self) -> bool: ...

    @property
    @abstractmethod
    def supports_partial_encode(self) -> bool: ...

    @abstractmethod
    def validate(
        self,
        *,
        shape: tuple[int, ...],
        dtype: ZDType[TBaseDType, TBaseScalar],
        chunk_grid: ChunkGrid,
    ) -> None:
        """Validates that all codec configurations are compatible with the array metadata.
        Raises errors when a codec configuration is not compatible.

        Parameters
        ----------
        shape : tuple[int, ...]
            The array shape
        dtype : np.dtype[Any]
            The array data type
        chunk_grid : ChunkGrid
            The array chunk grid
        """
        ...

    @abstractmethod
    def compute_encoded_size(self, byte_length: int, array_spec: ArraySpec) -> int:
        """Given an input byte length, this method returns the output byte length.
        Raises a NotImplementedError for codecs with variable-sized outputs (e.g. compressors).

        Parameters
        ----------
        byte_length : int
        array_spec : ArraySpec

        Returns
        -------
        int
        """
        ...

    @abstractmethod
    async def decode(
        self,
        chunk_bytes_and_specs: Iterable[tuple[Buffer | None, ArraySpec]],
    ) -> Iterable[NDBuffer | None]:
        """Decodes a batch of chunks.
        Chunks can be None in which case they are ignored by the codec.

        Parameters
        ----------
        chunk_bytes_and_specs : Iterable[tuple[Buffer | None, ArraySpec]]
            Ordered set of encoded chunks with their accompanying chunk spec.

        Returns
        -------
        Iterable[NDBuffer | None]
        """
        ...

    @abstractmethod
    async def encode(
        self,
        chunk_arrays_and_specs: Iterable[tuple[NDBuffer | None, ArraySpec]],
    ) -> Iterable[Buffer | None]:
        """Encodes a batch of chunks.
        Chunks can be None in which case they are ignored by the codec.

        Parameters
        ----------
        chunk_arrays_and_specs : Iterable[tuple[NDBuffer | None, ArraySpec]]
            Ordered set of to-be-encoded chunks with their accompanying chunk spec.

        Returns
        -------
        Iterable[Buffer | None]
        """
        ...

    @abstractmethod
    async def read(
        self,
        batch_info: Iterable[tuple[ByteGetter, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        out: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> None:
        """Reads chunk data from the store, decodes it and writes it into an output array.
        Partial decoding may be utilized if the codecs and stores support it.

        Parameters
        ----------
        batch_info : Iterable[tuple[ByteGetter, ArraySpec, SelectorTuple, SelectorTuple]]
            Ordered set of information about the chunks.
            The first slice selection determines which parts of the chunk will be fetched.
            The second slice selection determines where in the output array the chunk data will be written.
            The ByteGetter is used to fetch the necessary bytes.
            The chunk spec contains information about the construction of an array from the bytes.

            If the Store returns ``None`` for a chunk, then the chunk was not
            written and the implementation must set the values of that chunk (or
            ``out``) to the fill value for the array.

        out : NDBuffer
        """
        ...

    @abstractmethod
    async def write(
        self,
        batch_info: Iterable[tuple[ByteSetter, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        value: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> None:
        """Encodes chunk data and writes it to the store.
        Merges with existing chunk data by reading first, if necessary.
        Partial encoding may be utilized if the codecs and stores support it.

        Parameters
        ----------
        batch_info : Iterable[tuple[ByteSetter, ArraySpec, SelectorTuple, SelectorTuple]]
            Ordered set of information about the chunks.
            The first slice selection determines which parts of the chunk will be encoded.
            The second slice selection determines where in the value array the chunk data is located.
            The ByteSetter is used to fetch and write the necessary bytes.
            The chunk spec contains information about the chunk.
        value : NDBuffer
        """
        ...


async def _batching_helper(
    func: Callable[[CodecInput, ArraySpec], Awaitable[CodecOutput | None]],
    batch_info: Iterable[tuple[CodecInput | None, ArraySpec]],
) -> list[CodecOutput | None]:
    return await concurrent_map(
        list(batch_info),
        _noop_for_none(func),
        config.get("async.concurrency"),
    )


def _noop_for_none(
    func: Callable[[CodecInput, ArraySpec], Awaitable[CodecOutput | None]],
) -> Callable[[CodecInput | None, ArraySpec], Awaitable[CodecOutput | None]]:
    async def wrap(chunk: CodecInput | None, chunk_spec: ArraySpec) -> CodecOutput | None:
        if chunk is None:
            return None
        return await func(chunk, chunk_spec)

    return wrap
