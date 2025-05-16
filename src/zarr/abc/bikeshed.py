from collections.abc import Mapping
from typing import Generic, Self, TypeVar

import numpy as np
from typing_extensions import Buffer, Protocol, runtime_checkable

BufferOrNDArray = Buffer | np.ndarray[tuple[int, ...], np.dtype[np.generic]]
BaseConfig = Mapping[str, object]
TNCodecConfig = TypeVar("TNCodecConfig", bound=BaseConfig)


@runtime_checkable
class Numcodec(Protocol, Generic[TNCodecConfig]):
    """
    This protocol models the numcodecs.abc.Codec interface.
    """

    codec_id: str

    def encode(self, buf: BufferOrNDArray) -> BufferOrNDArray: ...

    def decode(
        self, buf: BufferOrNDArray, out: BufferOrNDArray | None = None
    ) -> BufferOrNDArray: ...

    def get_config(self) -> TNCodecConfig: ...

    @classmethod
    def from_config(cls, config: TNCodecConfig) -> Self: ...
