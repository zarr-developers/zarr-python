from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypedDict, TypeGuard, cast, overload

import numpy as np
import typing_extensions
from crc32c import crc32c

from zarr.abc.codec import BytesBytesCodec, CodecJSON, CodecJSON_V2
from zarr.core.common import JSON, NamedConfig, ZarrFormat, parse_named_configuration
from zarr.errors import CodecValidationError

if TYPE_CHECKING:
    from typing import Self

    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer


class Crc32Config(TypedDict): ...


class Crc32cJSON_V2(CodecJSON_V2[Literal["crc32c"]]): ...


class Crc32cJSON_V3(NamedConfig[Literal["crc32c"], Crc32Config]): ...


def check_json_v2(data: CodecJSON) -> TypeGuard[Crc32cJSON_V2]:
    return isinstance(data, Mapping) and set(data.keys()) == {"id"} and data["id"] == "crc32c"


def check_json_v3(data: CodecJSON) -> TypeGuard[Crc32cJSON_V3]:
    return (
        isinstance(data, Mapping)
        and set(data.keys()) in ({"name", "configuration"}, {"name"})
        and data["name"] == "crc32c"
        and data.get("configuration") in ({}, None)
    )


@dataclass(frozen=True)
class Crc32cCodec(BytesBytesCodec):
    is_fixed_size = True

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls.from_json(data, zarr_format=3)
        parse_named_configuration(data, "crc32c", require_configuration=False)
        return cls()

    @classmethod
    def _from_json_v2(cls, data: CodecJSON) -> Self:
        if check_json_v2(data):
            return cls()
        msg = (
            "Invalid Zarr V2 JSON representation of the crc32c codec. "
            f"Got {data!r}, expected a Mapping with keys ('id')"
        )
        raise CodecValidationError(msg)

    @classmethod
    def _from_json_v3(cls, data: CodecJSON) -> Self:
        if check_json_v3(data):
            return cls()
        msg = (
            "Invalid Zarr V3 JSON representation of the crc32c codec. "
            f"Got {data!r}, expected a Mapping with keys ('name')"
        )
        raise CodecValidationError(msg)

    def to_dict(self) -> dict[str, JSON]:
        return self.to_json(zarr_format=3)
        return {"name": "crc32c"}

    @overload
    def to_json(self, zarr_format: Literal[2]) -> Crc32cJSON_V2: ...

    @overload
    def to_json(self, zarr_format: Literal[3]) -> Crc32cJSON_V3: ...

    def to_json(self, zarr_format: ZarrFormat) -> CodecJSON:
        if zarr_format == 2:
            return {"id": "crc32c"}
        elif zarr_format == 3:
            return {"name": "crc32c"}
        raise ValueError(
            f"Unsupported Zarr format {zarr_format}. Expected 2 or 3."
        )  # pragma: no cover

    async def _decode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer:
        data = chunk_bytes.as_numpy_array()
        crc32_bytes = data[-4:]
        inner_bytes = data[:-4]

        # Need to do a manual cast until https://github.com/numpy/numpy/issues/26783 is resolved
        computed_checksum = np.uint32(
            crc32c(cast("typing_extensions.Buffer", inner_bytes))
        ).tobytes()
        stored_checksum = bytes(crc32_bytes)
        if computed_checksum != stored_checksum:
            raise ValueError(
                f"Stored and computed checksum do not match. Stored: {stored_checksum!r}. Computed: {computed_checksum!r}."
            )
        return chunk_spec.prototype.buffer.from_array_like(inner_bytes)

    async def _encode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer | None:
        data = chunk_bytes.as_numpy_array()
        # Calculate the checksum and "cast" it to a numpy array
        checksum = np.array([crc32c(cast("typing_extensions.Buffer", data))], dtype=np.uint32)
        # Append the checksum (as bytes) to the data
        return chunk_spec.prototype.buffer.from_array_like(np.append(data, checksum.view("B")))

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        return input_byte_length + 4
