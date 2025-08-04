from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypedDict, TypeGuard, overload

import numpy as np
from numcodecs.vlen import VLenBytes, VLenUTF8

from zarr.abc.codec import ArrayBytesCodec, CodecJSON, CodecJSON_V2
from zarr.core.buffer import Buffer, NDBuffer
from zarr.core.common import JSON, NamedConfig, ZarrFormat, parse_named_configuration
from zarr.registry import register_codec

if TYPE_CHECKING:
    from typing import Self

    from zarr.core.array_spec import ArraySpec


# can use a global because there are no parameters
_vlen_utf8_codec = VLenUTF8()
_vlen_bytes_codec = VLenBytes()


class VlenUF8Config(TypedDict): ...


class VLenUTF8JSON_V2(CodecJSON_V2[Literal["vlen-utf8"]]): ...


class VLenUTF8JSON_V3(NamedConfig[Literal["vlen-utf8"], VlenUF8Config]): ...


class VLenBytesConfig(TypedDict): ...


class VLenBytesJSON_V2(CodecJSON_V2[Literal["vlen-bytes"]]): ...


VLenBytesJSON_V3 = NamedConfig[Literal["vlen-bytes"], VLenBytesConfig] | Literal["vlen-bytes"]


@dataclass(frozen=True)
class VLenUTF8Codec(ArrayBytesCodec):
    """Variable-length UTF8 codec"""

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls.from_json(data, zarr_format=3)
        _, configuration_parsed = parse_named_configuration(
            data, "vlen-utf8", require_configuration=False
        )
        configuration_parsed = configuration_parsed or {}
        return cls(**configuration_parsed)

    def to_dict(self) -> dict[str, JSON]:
        return {"name": "vlen-utf8", "configuration": {}}

    @overload
    def to_json(self, zarr_format: Literal[2]) -> VLenUTF8JSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> VLenUTF8JSON_V3: ...
    def to_json(self, zarr_format: ZarrFormat) -> VLenUTF8JSON_V2 | VLenUTF8JSON_V3:
        if zarr_format == 2:
            return {"id": "vlen-utf8"}
        else:
            return {"name": "vlen-utf8"}

    @classmethod
    def _check_json_v2(cls, data: CodecJSON) -> TypeGuard[VLenUTF8JSON_V2]:
        return data == {"id": "vlen-utf8"}

    @classmethod
    def _check_json_v3(cls, data: CodecJSON) -> TypeGuard[VLenUTF8JSON_V3]:
        return data in (
            {"name": "vlen-utf8"},
            {"name": "vlen-utf8", "configuration": {}},
            "vlen-utf8",
        )

    @classmethod
    def _from_json_v2(cls, data: CodecJSON) -> Self:
        if cls._check_json_v2(data):
            return cls()
        raise ValueError(f"Invalid VLenUTF8 JSON data for Zarr format 2: {data!r}")

    @classmethod
    def _from_json_v3(cls, data: CodecJSON) -> Self:
        if cls._check_json_v3(data):
            return cls()
        raise ValueError(f"Invalid VLenUTF8 JSON data for Zarr format 3: {data!r}")

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        return self

    # TODO: expand the tests for this function
    async def _decode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        assert isinstance(chunk_bytes, Buffer)

        raw_bytes = chunk_bytes.as_array_like()
        decoded = _vlen_utf8_codec.decode(raw_bytes)
        assert decoded.dtype == np.object_
        decoded.shape = chunk_spec.shape
        as_string_dtype = decoded.astype(chunk_spec.dtype.to_native_dtype(), copy=False)
        return chunk_spec.prototype.nd_buffer.from_numpy_array(as_string_dtype)

    async def _encode_single(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> Buffer | None:
        assert isinstance(chunk_array, NDBuffer)
        return chunk_spec.prototype.buffer.from_bytes(
            _vlen_utf8_codec.encode(chunk_array.as_numpy_array())
        )

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        # what is input_byte_length for an object dtype?
        raise NotImplementedError("compute_encoded_size is not implemented for VLen codecs")


@dataclass(frozen=True)
class VLenBytesCodec(ArrayBytesCodec):
    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls.from_json(data, zarr_format=3)

    def to_dict(self) -> dict[str, JSON]:
        return {"name": "vlen-bytes", "configuration": {}}

    @overload
    def to_json(self, zarr_format: Literal[2]) -> VLenBytesJSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> VLenBytesJSON_V3: ...
    def to_json(self, zarr_format: ZarrFormat) -> VLenBytesJSON_V2 | VLenBytesJSON_V3:
        if zarr_format == 2:
            return {"id": "vlen-bytes"}
        else:
            return {"name": "vlen-bytes"}

    @classmethod
    def _check_json_v2(cls, data: CodecJSON) -> TypeGuard[VLenBytesJSON_V2]:
        return data == {"id": "vlen-bytes"}

    @classmethod
    def _check_json_v3(cls, data: CodecJSON) -> TypeGuard[VLenBytesJSON_V3]:
        return data in (
            {"name": "vlen-bytes"},
            {"name": "vlen-bytes", "configuration": {}},
            "vlen-bytes",
        )

    @classmethod
    def _from_json_v2(cls, data: CodecJSON) -> Self:
        if cls._check_json_v2(data):
            return cls()
        raise ValueError(f"Invalid VLenBytes JSON data for Zarr format 2: {data!r}")

    @classmethod
    def _from_json_v3(cls, data: CodecJSON) -> Self:
        if cls._check_json_v3(data):
            return cls()
        raise ValueError(f"Invalid VLenBytes JSON data for Zarr format 3: {data!r}")

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        return self

    async def _decode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        assert isinstance(chunk_bytes, Buffer)

        raw_bytes = chunk_bytes.as_array_like()
        decoded = _vlen_bytes_codec.decode(raw_bytes)
        assert decoded.dtype == np.object_
        decoded.shape = chunk_spec.shape
        return chunk_spec.prototype.nd_buffer.from_numpy_array(decoded)

    async def _encode_single(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> Buffer | None:
        assert isinstance(chunk_array, NDBuffer)
        return chunk_spec.prototype.buffer.from_bytes(
            _vlen_bytes_codec.encode(chunk_array.as_numpy_array())
        )

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        # what is input_byte_length for an object dtype?
        raise NotImplementedError("compute_encoded_size is not implemented for VLen codecs")


register_codec("vlen-utf8", VLenUTF8Codec)
register_codec("vlen-bytes", VLenBytesCodec)
