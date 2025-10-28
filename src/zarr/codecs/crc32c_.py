from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict, TypeGuard, cast, overload

import google_crc32c
import numpy as np
import typing_extensions
from typing_extensions import ReadOnly

from zarr.abc.codec import BytesBytesCodec
from zarr.core.common import (
    JSON,
    CodecJSON,
    NamedConfig,
    ZarrFormat,
    check_named_config,
    parse_named_configuration,
)
from zarr.errors import CodecValidationError

if TYPE_CHECKING:
    from typing import Self

    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer


class Crc32cConfig_V2(TypedDict):
    location: NotRequired[ReadOnly[Literal["start", "end"]]]


class Crc32cConfig_V3(TypedDict): ...


class Crc32cJSON_V2(Crc32cConfig_V2):
    id: ReadOnly[Literal["crc32c"]]


Crc32cJSON_V3 = NamedConfig[Literal["crc32c"], Crc32cConfig_V3] | Literal["crc32c"]


def check_json_v2(data: object) -> TypeGuard[Crc32cJSON_V2]:
    return (
        isinstance(data, Mapping)
        and "id" in data
        and data["id"] == "crc32c"
        and data.get("location", "end") in ("start", "end")
    )


def check_json_v3(data: object) -> TypeGuard[Crc32cJSON_V3]:
    if data == "crc32c":
        return True
    return (
        check_named_config(data)
        and set(data.keys()) in ({"name", "configuration"}, {"name"})
        and data["name"] == "crc32c"
        and data.get("configuration") in ({}, None)
    )


@dataclass(frozen=True)
class Crc32cCodec(BytesBytesCodec):
    """
    References
    ----------
    This specification document for this codec can be found at
    https://zarr-specs.readthedocs.io/en/latest/v3/codecs/crc32c/index.html
    """

    is_fixed_size = True

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        parse_named_configuration(data, "crc32c", require_configuration=False)
        return cls()

    @classmethod
    def _from_json_v2(cls, data: CodecJSON) -> Self:
        if check_json_v2(data):
            if data.get("location", "end") != "end":
                raise ValueError('The crc32c codec only supports the "end" location')
            return cls()
        msg = (
            "Invalid Zarr V2 JSON representation of the crc32c codec. "
            f"Got {data!r}, expected a Mapping with keys ('id', 'location')"
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
            google_crc32c.value(cast("typing_extensions.Buffer", inner_bytes))
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
        checksum = np.array(
            [google_crc32c.value(cast("typing_extensions.Buffer", data))], dtype=np.uint32
        )
        # Append the checksum (as bytes) to the data
        return chunk_spec.prototype.buffer.from_array_like(np.append(data, checksum.view("B")))

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        return input_byte_length + 4
