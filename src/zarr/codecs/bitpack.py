from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from zarr.abc.codec import BytesBytesCodec
from zarr.core.common import JSON, parse_named_configuration
from zarr.registry import register_codec

if TYPE_CHECKING:
    from typing import Self

    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer


@dataclass(frozen=True)
class BitPackingCodec(BytesBytesCodec):
    """
    Codec for bit-packing integer data that doesn't use the full range of its data type.

    This codec is particularly useful for ADC (Analog-to-Digital Converter) data that
    typically returns values using fewer bits (e.g., 10 or 12 bits) than standard integer
    types (16, 32, or 64 bits).
    """

    # Number of bits to use for each value in the packed format.
    bits_per_value: int

    # Original data type (for unpacking)
    original_dtype: np.dtype[Any]

    def __init__(
        self,
        *,
        bits_per_value: int,
        original_dtype: str | np.dtype[Any],
    ) -> None:
        if bits_per_value <= 0:  # ignore
            raise ValueError(f"bits_per_value must be a positive integer, got {bits_per_value}")

        if isinstance(original_dtype, str):
            original_dtype = np.dtype(original_dtype)

        object.__setattr__(self, "bits_per_value", bits_per_value)
        object.__setattr__(self, "original_dtype", original_dtype)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(data, "bitpacking")
        return cls(**configuration_parsed)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, JSON]:
        return {
            "name": "bitpacking",
            "configuration": {
                "bits_per_value": self.bits_per_value,
                "original_dtype": str(self.original_dtype),
            },
        }

    async def _encode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer | None:
        """Pack the data using only the necessary bits per value."""
        return await asyncio.to_thread(
            self._bit_pack,
            chunk_bytes,
            chunk_spec,
        )

    async def _decode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer:
        """Unpack the bit-packed data back to original format."""
        return await asyncio.to_thread(
            self._bit_unpack,
            chunk_bytes,
            chunk_spec,
        )

    def _bit_pack(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> Buffer:
        """
        Implement the bit-packing algorithm here.
        Convert the input array to a bit-packed format.
        """
        dtype = chunk_spec.dtype

        arr = np.frombuffer(chunk_bytes.as_numpy_array(), dtype=dtype).reshape(chunk_spec.shape)

        print(arr)
        original_bytes = arr.nbytes
        original_bits = original_bytes * 8

        print("===== BIT PACKING STATISTICS =====")
        print(f"Original array shape: {arr.shape}")
        print(f"Original data type: {arr.dtype} ({arr.dtype.itemsize} bytes per value)")
        print(f"Original data size: {original_bytes} bytes ({original_bits} bits)")

        # Create a bit mask for the values
        mask = np.uint16((1 << self.bits_per_value) - 1)

        total_values = arr.size
        output_size = (total_values * self.bits_per_value + 7) // 8

        # Print bit packing settings
        print(f"Bit-packing using {self.bits_per_value} bits per value")
        print(f"Total values: {total_values}")
        print(f"Theoretical packed size: {total_values * self.bits_per_value / 8:.2f} bytes")
        print(f"Actual packed size: {output_size} bytes")
        print(f"Storage savings: {(1 - output_size / original_bytes) * 100:.2f}%")

        # Calculate output size
        total_values = arr.size
        output_size = (total_values * self.bits_per_value + 7) // 8

        packed = np.zeros(output_size, dtype=np.uint8)

        # Pack the values
        for i in range(total_values):
            value = arr.flat[i] & mask
            bit_pos = (i * self.bits_per_value) % 8
            byte_pos = (i * self.bits_per_value) // 8

            # Handle values that cross byte boundaries
            if bit_pos + self.bits_per_value <= 8:
                packed[byte_pos] |= value << bit_pos
            else:
                # Value spans two bytes
                bits_in_first = 8 - bit_pos

                packed[byte_pos] |= (value & ((1 << bits_in_first) - 1)) << bit_pos
                packed[byte_pos + 1] |= value >> bits_in_first

        print("==============================")

        return chunk_spec.prototype.buffer.from_bytes(packed.tobytes())

    def _bit_unpack(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> Buffer:
        """
        Implement the bit-unpacking algorithm here.
        Convert the bit-packed format back to the original array.
        """

        # Print packed data information
        packed_bytes = chunk_bytes.as_numpy_array()
        print("===== BIT UNPACKING STATISTICS =====")
        print(f"Packed data size: {len(packed_bytes)} bytes")

        packed = np.frombuffer(chunk_bytes.as_numpy_array(), dtype=np.uint8)

        # Calculate original array size
        total_bits = packed.size * 8
        total_values = total_bits // self.bits_per_value
        expected_output_bytes = total_values * np.dtype(self.original_dtype).itemsize

        print(f"Unpacking using {self.bits_per_value} bits per value")
        print(f"Total packed bits: {total_bits}")
        print(f"Calculated number of values: {total_values}")
        print(f"Expected output size: {expected_output_bytes} bytes")

        unpacked = np.zeros(total_values, dtype=self.original_dtype)

        mask = (1 << self.bits_per_value) - 1

        for i in range(total_values):
            bit_pos = (i * self.bits_per_value) % 8
            byte_pos = (i * self.bits_per_value) // 8

            if bit_pos + self.bits_per_value <= 8:
                value = (packed[byte_pos] >> bit_pos) & mask
            else:
                bits_in_first = 8 - bit_pos
                bits_in_second = self.bits_per_value - bits_in_first

                value_first = packed[byte_pos] >> bit_pos
                value_second = packed[byte_pos + 1] & ((1 << bits_in_second) - 1)

                value = value_first | (value_second << bits_in_first)

            unpacked[i] = value

        # Reshape to match original array shape
        unpacked = unpacked.reshape(chunk_spec.shape)

        print(f"First few unpacked values: {unpacked.flat[: min(10, unpacked.size)]}")
        print(f"Actual unpacked size: {unpacked.nbytes} bytes")
        print(f"Size expansion: {(unpacked.nbytes / len(packed_bytes)):.2f}x")
        print("================================")

        return chunk_spec.prototype.buffer.from_bytes(unpacked.tobytes())


register_codec("bitpack", BitPackingCodec)
