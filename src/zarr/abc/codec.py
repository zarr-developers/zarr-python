from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Generic, Iterable, TypeVar

from zarr.abc.metadata import Metadata
from zarr.abc.store import ByteGetter, ByteSetter
from zarr.buffer import Buffer, NDBuffer


if TYPE_CHECKING:
    from typing_extensions import Self
    from zarr.common import ArraySpec, SliceSelection
    from zarr.metadata import ArrayMetadata


CodecInput = TypeVar("CodecInput", bound=NDBuffer | Buffer)
CodecOutput = TypeVar("CodecOutput", bound=NDBuffer | Buffer)


class _Codec(Generic[CodecInput, CodecOutput], Metadata):
    """Generic base class for codecs.
    Please use ArrayArrayCodec, ArrayBytesCodec or BytesBytesCodec for subclassing.

    Codecs can be registered via zarr.codecs.registry.
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

    def evolve(self, array_spec: ArraySpec) -> Self:
        """Fills in codec configuration parameters that can be automatically
        inferred from the array metadata.

        Parameters
        ----------
        chunk_spec : ArraySpec

        Returns
        -------
        Self
        """
        return self

    def validate(self, array_metadata: ArrayMetadata) -> None:
        """Validates that the codec configuration is compatible with the array metadata.
        Raises errors when the codec configuration is not compatible.

        Parameters
        ----------
        array_metadata : ArrayMetadata
        """
        ...

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...


class ArrayArrayCodec(_Codec[NDBuffer, NDBuffer]):
    """Base class for array-to-array codecs."""

    ...


class ArrayBytesCodec(_Codec[NDBuffer, Buffer]):
    """Base class for array-to-bytes codecs."""

    ...


class BytesBytesCodec(_Codec[Buffer, Buffer]):
    """Base class for bytes-to-bytes codecs."""

    ...


Codec = ArrayArrayCodec | ArrayBytesCodec | BytesBytesCodec


class ArrayBytesCodecPartialDecodeMixin:
    """Mixin for array-to-bytes codecs that implement partial decoding."""

    @abstractmethod
    async def decode_partial(
        self,
        batch_info: Iterable[tuple[ByteGetter, SliceSelection, ArraySpec]],
    ) -> Iterable[NDBuffer | None]:
        """Partially decodes a batch of chunks.
        This method determines parts of a chunk from the slice selection,
        fetches these parts from the store (via ByteGetter) and decodes them.

        Parameters
        ----------
        batch_info : Iterable[tuple[ByteGetter, SliceSelection, ArraySpec]]
            Ordered set of information about slices of encoded chunks.
            The slice selection determines which parts of the chunk will be fetched.
            The ByteGetter is used to fetch the necessary bytes.
            The chunk spec contains information about the construction of an array from the bytes.

        Returns
        -------
        Iterable[NDBuffer | None]
        """
        ...


class ArrayBytesCodecPartialEncodeMixin:
    """Mixin for array-to-bytes codecs that implement partial encoding."""

    @abstractmethod
    async def encode_partial(
        self,
        batch_info: Iterable[tuple[ByteSetter, NDBuffer, SliceSelection, ArraySpec]],
    ) -> None:
        """Partially encodes a batch of chunks.
        This method determines parts of a chunk from the slice selection, encodes them and
        writes these parts to the store (via ByteSetter).
        If merging with existing chunk data in the store is necessary, this method will
        read from the store first and perform the merge.

        Parameters
        ----------
        batch_info : Iterable[tuple[ByteSetter, NDBuffer, SliceSelection, ArraySpec]]
            Ordered set of information about slices of to-be-encoded chunks.
            The slice selection determines which parts of the chunk will be encoded.
            The ByteSetter is used to write the necessary bytes and fetch bytes for existing chunk data.
            The chunk spec contains information about the chunk.
        """
        ...


class CodecPipeline(Metadata):
    """Base class for implementing CodecPipeline.
    A CodecPipeline implements the read and write paths for chunk data.
    On the read path, it is responsible for fetching chunks from a store (via ByteGetter),
    decoding them and assembling an output array. On the write path, it encodes the chunks
    and writes them to a store (via ByteSetter)."""

    @abstractmethod
    def evolve(self, array_spec: ArraySpec) -> Self:
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
    def from_list(cls, codecs: list[Codec]) -> Self:
        """Creates a codec pipeline from a list of codecs.

        Parameters
        ----------
        codecs : list[Codec]

        Returns
        -------
        Self
        """
        ...

    @property
    @abstractmethod
    def supports_partial_decode(self) -> bool: ...

    @property
    @abstractmethod
    def supports_partial_encode(self) -> bool: ...

    @abstractmethod
    def validate(self, array_metadata: ArrayMetadata) -> None:
        """Validates that all codec configurations are compatible with the array metadata.
        Raises errors when a codec configuration is not compatible.

        Parameters
        ----------
        array_metadata : ArrayMetadata
        """
        ...

    @abstractmethod
    def compute_encoded_size(self, byte_length: int, array_spec: ArraySpec) -> int:
        """Given an input byte length, this method returns the output byte length.
        Raises a NotImplementedError for codecs with variable-sized outputs (e.g. compressors).

        Parameters
        ----------
        input_byte_length : int
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
        chunks_and_specs : Iterable[tuple[Buffer | None, ArraySpec]]
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
        chunks_and_specs : Iterable[tuple[NDBuffer | None, ArraySpec]]
            Ordered set of to-be-encoded chunks with their accompanying chunk spec.

        Returns
        -------
        Iterable[Buffer | None]
        """
        ...

    @abstractmethod
    async def read(
        self,
        batch_info: Iterable[tuple[ByteGetter, ArraySpec, SliceSelection, SliceSelection]],
        out: NDBuffer,
    ) -> None:
        """Reads chunk data from the store, decodes it and writes it into an output array.
        Partial decoding may be utilized if the codecs and stores support it.

        Parameters
        ----------
        batch_info : Iterable[tuple[ByteGetter, ArraySpec, SliceSelection, SliceSelection]]
            Ordered set of information about the chunks.
            The first slice selection determines which parts of the chunk will be fetched.
            The second slice selection determines where in the output array the chunk data will be written.
            The ByteGetter is used to fetch the necessary bytes.
            The chunk spec contains information about the construction of an array from the bytes.
        out : NDBuffer
        """
        ...

    @abstractmethod
    async def write(
        self,
        batch_info: Iterable[tuple[ByteSetter, ArraySpec, SliceSelection, SliceSelection]],
        value: NDBuffer,
    ) -> None:
        """Encodes chunk data and writes it to the store.
        Merges with existing chunk data by reading first, if necessary.
        Partial encoding may be utilized if the codecs and stores support it.

        Parameters
        ----------
        batch_info : Iterable[tuple[ByteSetter, ArraySpec, SliceSelection, SliceSelection]]
            Ordered set of information about the chunks.
            The first slice selection determines which parts of the chunk will be encoded.
            The second slice selection determines where in the value array the chunk data is located.
            The ByteSetter is used to fetch and write the necessary bytes.
            The chunk spec contains information about the chunk.
        value : NDBuffer
        """
        ...
