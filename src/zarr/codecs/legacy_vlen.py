from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from numcodecs.vlen import VLenUTF8

from zarr.abc.codec import ArrayBytesCodec
from zarr.core.buffer import Buffer, NDBuffer
from zarr.core.common import JSON, parse_named_configuration
from zarr.registry import register_codec

if TYPE_CHECKING:
    from typing import Self

    from zarr.core.array_spec import ArraySpec


# can use a global because there are no parameters
vlen_utf8_codec = VLenUTF8()


@dataclass(frozen=True)
class VLenUTF8Codec(ArrayBytesCodec):
    def __init__(self) -> None:
        pass

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(
            data, "vlen-utf8", require_configuration=False
        )
        configuration_parsed = configuration_parsed or {}
        return cls(**configuration_parsed)

    def to_dict(self) -> dict[str, JSON]:
        return {"name": "vlen-utf8"}

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        return self

    async def _decode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        assert isinstance(chunk_bytes, Buffer)

        raw_bytes = chunk_bytes.as_array_like()
        decoded = vlen_utf8_codec.decode(raw_bytes)
        decoded.shape = chunk_spec.shape
        return chunk_spec.prototype.nd_buffer.from_numpy_array(decoded)

    async def _encode_single(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> Buffer | None:
        assert isinstance(chunk_array, NDBuffer)
        return chunk_spec.prototype.buffer.from_bytes(
            vlen_utf8_codec.encode(chunk_array.as_numpy_array())
        )

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        # what is input_byte_length for an object dtype?
        raise NotImplementedError("compute_encoded_size is not implemented for VLen codecs")


register_codec("vlen-utf8", VLenUTF8Codec)
