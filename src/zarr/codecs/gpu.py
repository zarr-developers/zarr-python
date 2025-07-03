from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from zarr.abc.codec import BytesBytesCodec
from zarr.core.common import JSON, parse_named_configuration
from zarr.registry import register_codec

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable
    from typing import Any, Self

    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    from nvidia import nvcomp
except ImportError:
    nvcomp = None


class AsyncCUDAEvent(Awaitable[None]):
    """An awaitable wrapper around a CuPy CUDA event for asynchronous waiting."""

    def __init__(
        self, event: cp.cuda.Event, initial_delay: float = 0.001, max_delay: float = 0.1
    ) -> None:
        """
        Initialize the async CUDA event.

        Args:
            event (cp.cuda.Event): The CuPy CUDA event to wait on.
            initial_delay (float): Initial polling delay in seconds (default: 0.001s).
            max_delay (float): Maximum polling delay in seconds (default: 0.1s).
        """
        self.event = event
        self.initial_delay = initial_delay
        self.max_delay = max_delay

    def __await__(self) -> Generator[Any, None, None]:
        """Makes the event awaitable by yielding control until the event is complete."""
        return self._wait().__await__()

    async def _wait(self) -> None:
        """Polls the CUDA event asynchronously with exponential backoff until it completes."""
        delay = self.initial_delay
        while not self.event.done:  # `done` returns True if the event is complete
            await asyncio.sleep(delay)  # Yield control to other async tasks
            delay = min(delay * 2, self.max_delay)  # Exponential backoff


def parse_zstd_level(data: JSON) -> int:
    if isinstance(data, int):
        if data >= 23:
            raise ValueError(f"Value must be less than or equal to 22. Got {data} instead.")
        return data
    raise TypeError(f"Got value with type {type(data)}, but expected an int.")


def parse_checksum(data: JSON) -> bool:
    if isinstance(data, bool):
        return data
    raise TypeError(f"Expected bool. Got {type(data)}.")


@dataclass(frozen=True)
class NvcompZstdCodec(BytesBytesCodec):
    is_fixed_size = True

    level: int = 0
    checksum: bool = False

    def __init__(self, *, level: int = 0, checksum: bool = False) -> None:
        # TODO: Set CUDA device appropriately here and also set CUDA stream

        level_parsed = parse_zstd_level(level)
        checksum_parsed = parse_checksum(checksum)

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

    @cached_property
    def _zstd_codec(self) -> nvcomp.Codec:
        # config_dict = {algorithm = "Zstd", "level": self.level, "checksum": self.checksum}
        # return Zstd.from_config(config_dict)
        device = cp.cuda.Device()  # Select the current default device
        stream = cp.cuda.get_current_stream()  # Use the current default stream
        return nvcomp.Codec(
            algorithm="Zstd",
            bitstream_kind=nvcomp.BitstreamKind.RAW,
            device_id=device.id,
            cuda_stream=stream.ptr,
        )

    async def _convert_to_nvcomp_arrays(
        self,
        chunks_and_specs: Iterable[tuple[Buffer | None, ArraySpec]],
    ) -> tuple[list[nvcomp.Array], list[int]]:
        none_indices = [i for i, (b, _) in enumerate(chunks_and_specs) if b is None]
        filtered_inputs = [b.as_array_like() for b, _ in chunks_and_specs if b is not None]
        # TODO: add CUDA stream here
        return nvcomp.as_arrays(filtered_inputs), none_indices

    async def _convert_from_nvcomp_arrays(
        self,
        arrays: Iterable[nvcomp.Array],
        chunks_and_specs: Iterable[tuple[Buffer | None, ArraySpec]],
        awaitable: AsyncCUDAEvent,
    ) -> Iterable[Buffer | None]:
        await awaitable  # Wait for array computation to complete before accessing
        return [
            spec.prototype.buffer.from_array_like(cp.array(a, dtype=np.dtype("b"), copy=False))
            if a
            else None
            for a, (_, spec) in zip(arrays, chunks_and_specs, strict=True)
        ]

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
        filtered_inputs, none_indices = await self._convert_to_nvcomp_arrays(chunks_and_specs)

        outputs = self._zstd_codec.decode(filtered_inputs) if len(filtered_inputs) > 0 else []

        # Record event for synchronization
        event = cp.cuda.Event()
        awaitable = AsyncCUDAEvent(event)  # Convert CUDA event to awaitable object

        for index in none_indices:
            outputs.insert(index, None)

        return await self._convert_from_nvcomp_arrays(outputs, chunks_and_specs, awaitable)

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
        filtered_inputs, none_indices = await self._convert_to_nvcomp_arrays(chunks_and_specs)

        outputs = self._zstd_codec.encode(filtered_inputs) if len(filtered_inputs) > 0 else []

        # Record event for synchronization
        event = cp.cuda.Event()
        awaitable = AsyncCUDAEvent(event)  # Convert CUDA event to awaitable object

        for index in none_indices:
            outputs.insert(index, None)

        return await self._convert_from_nvcomp_arrays(outputs, chunks_and_specs, awaitable)

    def compute_encoded_size(self, _input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        raise NotImplementedError


register_codec("zstd", NvcompZstdCodec)
