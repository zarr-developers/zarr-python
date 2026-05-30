"""Codec interface definitions (v1).

This module defines the abstract interfaces for zarr codecs.
External codec implementations should subclass ``ArrayArrayCodec``,
``ArrayBytesCodec``, or ``BytesBytesCodec`` from this module.

The ``Buffer`` and ``NDBuffer`` types here are protocols — they define
the structural interface that zarr's concrete buffer types implement.
Codec authors should type against these protocols, not zarr's concrete
buffer classes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol, Self, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from zarr_interfaces.data_type.v1 import JSON, TBaseDType, TBaseScalar, ZDType


# ---------------------------------------------------------------------------
# Buffer protocols
# ---------------------------------------------------------------------------


class Buffer(Protocol):
    """Protocol for a flat contiguous memory block (bytes-like)."""

    def __len__(self) -> int: ...
    def __getitem__(self, key: slice) -> Buffer: ...


class NDBuffer(Protocol):
    """Protocol for an N-dimensional array buffer."""

    @property
    def dtype(self) -> np.dtype[np.generic]: ...

    @property
    def shape(self) -> tuple[int, ...]: ...

    def as_ndarray_like(self) -> npt.NDArray[np.generic]: ...

    @classmethod
    def from_ndarray_like(cls, data: npt.NDArray[np.generic]) -> NDBuffer: ...

    def transpose(self, axes: tuple[int, ...]) -> NDBuffer: ...

    def __getitem__(self, key: object) -> NDBuffer: ...

    def __setitem__(self, key: object, value: object) -> None: ...


# ---------------------------------------------------------------------------
# ArraySpec protocol
# ---------------------------------------------------------------------------


class ArraySpec(Protocol):
    """Protocol for the specification of a chunk's metadata."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> ZDType[TBaseDType, TBaseScalar]: ...

    @property
    def fill_value(self) -> object: ...

    @property
    def ndim(self) -> int: ...


# ---------------------------------------------------------------------------
# Codec input/output type aliases
# ---------------------------------------------------------------------------

type CodecInput = NDBuffer | Buffer
type CodecOutput = NDBuffer | Buffer


# ---------------------------------------------------------------------------
# Sync codec protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class SupportsSyncCodec[CI: CodecInput, CO: CodecOutput](Protocol):
    """Protocol for codecs that support synchronous encode/decode.

    The type parameters mirror ``BaseCodec``: ``CI`` is the decoded type
    and ``CO`` is the encoded type.
    """

    def _decode_sync(self, chunk_data: CO, chunk_spec: ArraySpec) -> CI: ...

    def _encode_sync(self, chunk_data: CI, chunk_spec: ArraySpec) -> CO | None: ...


# ---------------------------------------------------------------------------
# Codec ABCs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaseCodec[CI: CodecInput, CO: CodecOutput](ABC):
    """Generic base class for codecs.

    Subclass ``ArrayArrayCodec``, ``ArrayBytesCodec``, or
    ``BytesBytesCodec`` instead of this class directly.
    """

    is_fixed_size: ClassVar[bool]

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        """Create an instance from a JSON dictionary."""
        return cls(**data)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, JSON]:
        """Serialize this codec to a JSON dictionary."""
        raise NotImplementedError

    @abstractmethod
    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        """Return the encoded byte length for a given input byte length."""
        ...

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        """Return the chunk spec after encoding by this codec.

        Override this for codecs that change shape, dtype, or fill value.
        """
        return chunk_spec

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        """Fill in codec parameters that can be inferred from array metadata."""
        return self

    def validate(
        self,
        *,
        shape: tuple[int, ...],
        dtype: ZDType[TBaseDType, TBaseScalar],
        chunk_grid: object,
    ) -> None:
        """Validate that this codec is compatible with the array metadata.

        The default implementation does nothing. Override to add checks.
        """

    async def _decode_single(self, chunk_data: CO, chunk_spec: ArraySpec) -> CI:
        """Decode a single chunk. Override this or ``_decode_sync``."""
        raise NotImplementedError

    async def decode(
        self,
        chunks_and_specs: Iterable[tuple[CO | None, ArraySpec]],
    ) -> Iterable[CI | None]:
        """Decode a batch of chunks."""
        results: list[CI | None] = []
        for chunk_data, chunk_spec in chunks_and_specs:
            if chunk_data is not None:
                results.append(await self._decode_single(chunk_data, chunk_spec))
            else:
                results.append(None)
        return results

    async def _encode_single(self, chunk_data: CI, chunk_spec: ArraySpec) -> CO | None:
        """Encode a single chunk. Override this or ``_encode_sync``."""
        raise NotImplementedError

    async def encode(
        self,
        chunks_and_specs: Iterable[tuple[CI | None, ArraySpec]],
    ) -> Iterable[CO | None]:
        """Encode a batch of chunks."""
        results: list[CO | None] = []
        for chunk_data, chunk_spec in chunks_and_specs:
            if chunk_data is not None:
                results.append(await self._encode_single(chunk_data, chunk_spec))
            else:
                results.append(None)
        return results


class ArrayArrayCodec(BaseCodec[NDBuffer, NDBuffer]):
    """Base class for array-to-array codecs (e.g. transpose, scale_offset)."""


class ArrayBytesCodec(BaseCodec[NDBuffer, Buffer]):
    """Base class for array-to-bytes codecs (e.g. bytes, sharding)."""


class BytesBytesCodec(BaseCodec[Buffer, Buffer]):
    """Base class for bytes-to-bytes codecs (e.g. gzip, zstd)."""
