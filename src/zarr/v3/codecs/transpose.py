from __future__ import annotations
from dataclasses import dataclass, field

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Literal,
    Optional,
    Tuple,
    Type,
)

import numpy as np

from zarr.v3.abc.codec import ArrayArrayCodec
from zarr.v3.codecs.registry import register_codec
from zarr.v3.common import NamedConfig, RuntimeConfiguration

if TYPE_CHECKING:
    from zarr.v3.metadata import CoreArrayMetadata


@dataclass(frozen=True)
class TransposeCodecConfigurationMetadata:
    order: Tuple[int, ...]


@dataclass(frozen=True)
class TransposeCodecMetadata:
    configuration: TransposeCodecConfigurationMetadata
    name: Literal["transpose"] = field(default="transpose", init=False)


@dataclass(frozen=True)
class TransposeCodec(ArrayArrayCodec):
    array_metadata: CoreArrayMetadata
    configuration: TransposeCodecConfigurationMetadata
    order: Tuple[int, ...]
    is_fixed_size = True

    @classmethod
    def from_metadata(
        cls, codec_metadata: NamedConfig, array_metadata: CoreArrayMetadata
    ) -> TransposeCodec:
        assert isinstance(codec_metadata, TransposeCodecMetadata)

        configuration = codec_metadata.configuration
        # Compatibility with older version of ZEP1
        if configuration.order == "F":  # type: ignore
            order = tuple(array_metadata.ndim - x - 1 for x in range(array_metadata.ndim))

        elif configuration.order == "C":  # type: ignore
            order = tuple(range(array_metadata.ndim))

        else:
            assert len(configuration.order) == array_metadata.ndim, (
                "The `order` tuple needs have as many entries as "
                + f"there are dimensions in the array. Got: {configuration.order}"
            )
            assert len(configuration.order) == len(set(configuration.order)), (
                "There must not be duplicates in the `order` tuple. "
                + f"Got: {configuration.order}"
            )
            assert all(0 <= x < array_metadata.ndim for x in configuration.order), (
                "All entries in the `order` tuple must be between 0 and "
                + f"the number of dimensions in the array. Got: {configuration.order}"
            )
            order = tuple(configuration.order)

        return cls(
            array_metadata=array_metadata,
            configuration=TransposeCodecConfigurationMetadata(order=order),
            order=order,
        )

    @classmethod
    def get_metadata_class(cls) -> Type[TransposeCodecMetadata]:
        return TransposeCodecMetadata

    def resolve_metadata(self) -> CoreArrayMetadata:
        from zarr.v3.metadata import CoreArrayMetadata

        return CoreArrayMetadata(
            shape=tuple(
                self.array_metadata.shape[self.order[i]] for i in range(self.array_metadata.ndim)
            ),
            chunk_shape=tuple(
                self.array_metadata.chunk_shape[self.order[i]]
                for i in range(self.array_metadata.ndim)
            ),
            dtype=self.array_metadata.dtype,
            fill_value=self.array_metadata.fill_value,
        )

    async def decode(
        self, chunk_array: np.ndarray, runtime_configuration: RuntimeConfiguration
    ) -> np.ndarray:
        inverse_order = [0 for _ in range(self.array_metadata.ndim)]
        for x, i in enumerate(self.order):
            inverse_order[x] = i
        chunk_array = chunk_array.transpose(inverse_order)
        return chunk_array

    async def encode(
        self, chunk_array: np.ndarray, runtime_configuration: RuntimeConfiguration
    ) -> Optional[np.ndarray]:
        chunk_array = chunk_array.transpose(self.order)
        return chunk_array

    def compute_encoded_size(self, input_byte_length: int) -> int:
        return input_byte_length


register_codec("transpose", TransposeCodec)
