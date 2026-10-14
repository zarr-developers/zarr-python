from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from zarr.abc.codec import BytesBytesCodec
from zarr.core.common import JSON, parse_named_configuration
from zarr.registry import register_codec

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Self

    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer

try:
    import cupy as cp
except ImportError:  # pragma: no cover
    cp = None

try:
    from nvidia import nvcomp
except ImportError:  # pragma: no cover
    nvcomp = None


def _parse_zstd_level(data: JSON) -> int:
    if isinstance(data, int):
        if data >= 23:
            raise ValueError(f"Value must be less than or equal to 22. Got {data} instead.")
        return data
    raise TypeError(f"Got value with type {type(data)}, but expected an int.")


def _parse_checksum(data: JSON) -> bool:
    if isinstance(data, bool):
        return data
    raise TypeError(f"Expected bool. Got {type(data)}.")


@dataclass(frozen=True)
class NvcompZstdCodec(BytesBytesCodec):
    is_fixed_size = False

    level: int = 0
    checksum: bool = False

    def __init__(self, *, level: int = 0, checksum: bool = False) -> None:
        # TODO: Set CUDA device appropriately here and also set CUDA stream

        level_parsed = _parse_zstd_level(level)
        checksum_parsed = _parse_checksum(checksum)

        object.__setattr__(self, "level", level_parsed)
        object.__setattr__(self, "checksum", checksum_parsed)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(data, "zstd")
        return cls(**configuration_parsed)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, JSON]:
        return {
            "name": "zstd",
            "configuration": {"level": self.level, "checksum": self.checksum},
        }

    def _zstd_codec(self, stream: cp.cuda.Stream) -> nvcomp.Codec:
        # Built fresh on every call (not cached) and bound to the *caller's*
        # current device/stream, passed in explicitly. `NvcompZstdCodec`
        # instances are cached and reused (e.g. by the sharding codec's
        # pipeline cache), so caching this per-instance would silently pin
        # every future call to whatever device/stream happened to be current
        # the first time this codec instance ran.
        device = cp.cuda.Device()  # the current device
        checksum_policy = (
            nvcomp.ChecksumPolicy.COMPUTE_AND_VERIFY
            if self.checksum
            else nvcomp.ChecksumPolicy.NO_COMPUTE_NO_VERIFY
        )
        return nvcomp.Codec(
            algorithm="Zstd",
            bitstream_kind=nvcomp.BitstreamKind.RAW,
            device_id=device.id,
            cuda_stream=stream.ptr,
            checksum_policy=checksum_policy,
        )

    def _convert_to_nvcomp_arrays(
        self,
        chunks_and_specs: Iterable[tuple[Buffer | None, ArraySpec]],
    ) -> tuple[list[nvcomp.Array], list[int]]:
        none_indices = [i for i, (b, _) in enumerate(chunks_and_specs) if b is None]
        filtered_inputs = [b.as_array_like() for b, _ in chunks_and_specs if b is not None]
        # TODO: add CUDA stream here
        return nvcomp.as_arrays(filtered_inputs), none_indices

    def _convert_from_nvcomp_arrays(
        self,
        arrays: Iterable[nvcomp.Array],
        chunks_and_specs: Iterable[tuple[Buffer | None, ArraySpec]],
    ) -> Iterable[Buffer | None]:
        result: list[Buffer | None] = []

        for a, (_, spec) in zip(arrays, chunks_and_specs, strict=True):
            if a is None:
                result.append(None)
            else:
                a2 = cp.array(a, dtype=a.dtype, copy=False)
                if a2.dtype != np.dtype("B"):
                    a2 = a2.view(dtype=np.dtype("B"))
                result.append(spec.prototype.buffer.from_array_like(a2))

        return result

    async def decode(
        self,
        chunks_and_specs: Iterable[tuple[Buffer | None, ArraySpec]],
    ) -> Iterable[Buffer | None]:
        """Decodes a batch of chunks.
        Chunks can be None in which case they are ignored by the codec.

        Parameters
        ----------
        chunks_and_specs : Iterable[tuple[Buffer | None, ArraySpec]]
            Ordered set of encoded chunks with their accompanying chunk spec.

        Returns
        -------
        Iterable[Buffer | None]
        """
        chunks_and_specs = list(chunks_and_specs)

        # Convert to nvcomp arrays
        filtered_inputs, none_indices = self._convert_to_nvcomp_arrays(chunks_and_specs)

        stream = cp.cuda.get_current_stream()
        codec = self._zstd_codec(stream)
        outputs = codec.decode(filtered_inputs) if len(filtered_inputs) > 0 else []

        # Record the event on the stream the decode was enqueued on, then
        # wait for it off the event loop thread. Previously this created an
        # event but never called `.record()`, so an unrecorded event is
        # already "complete" and `.synchronize()` returned immediately
        # regardless of whether the decode kernels had actually finished --
        # a race, not just a wasted thread-pool hop.
        event = cp.cuda.Event()
        event.record(stream)
        await asyncio.to_thread(event.synchronize)

        for index in none_indices:
            outputs.insert(index, None)

        return self._convert_from_nvcomp_arrays(outputs, chunks_and_specs)

    async def encode(
        self,
        chunks_and_specs: Iterable[tuple[Buffer | None, ArraySpec]],
    ) -> Iterable[Buffer | None]:
        """Encodes a batch of chunks.
        Chunks can be None in which case they are ignored by the codec.

        Parameters
        ----------
        chunks_and_specs : Iterable[tuple[Buffer | None, ArraySpec]]
            Ordered set of to-be-encoded chunks with their accompanying chunk spec.

        Returns
        -------
        Iterable[Buffer | None]
        """
        # TODO: Make this actually async
        chunks_and_specs = list(chunks_and_specs)

        # Convert to nvcomp arrays
        filtered_inputs, none_indices = self._convert_to_nvcomp_arrays(chunks_and_specs)

        stream = cp.cuda.get_current_stream()
        codec = self._zstd_codec(stream)
        outputs = codec.encode(filtered_inputs) if len(filtered_inputs) > 0 else []

        # See the matching comment in `decode` -- this now actually
        # synchronizes on the stream the encode was enqueued on.
        event = cp.cuda.Event()
        event.record(stream)
        await asyncio.to_thread(event.synchronize)

        for index in none_indices:
            outputs.insert(index, None)

        return self._convert_from_nvcomp_arrays(outputs, chunks_and_specs)

    def compute_encoded_size(self, _input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        raise NotImplementedError


register_codec("zstd", NvcompZstdCodec)
