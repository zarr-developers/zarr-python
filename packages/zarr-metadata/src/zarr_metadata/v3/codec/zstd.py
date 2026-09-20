"""
Zstandard codec types.

See https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/zstd/README.md
(the zarr-extensions registry entry; zarr-specs PR #256, which first
proposed the codec, was never merged).
"""

from dataclasses import dataclass
from typing import ClassVar, Final, Literal, NotRequired, cast

from typing_extensions import TypedDict

from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.v3._entity import (
    CodecKind,
    MemberTypes,
    MetadataEntity,
    is_bool,
    is_int,
    problem,
)

ZSTD_CODEC_NAME: Final = "zstd"
"""The `name` field value of the `zstd` codec."""

ZstdCodecName = Literal["zstd"]
"""Literal type of the `name` field of the `zstd` codec."""

ZSTD_MIN_LEVEL: Final = -131072
"""The lowest `level` zstd accepts: ZSTD_minCLevel(), -(1 << 17)."""

ZSTD_MAX_LEVEL: Final = 22
"""The highest `level` zstd accepts: ZSTD_maxCLevel()."""


class ZstdCodecConfiguration(TypedDict, closed=True):
    """
    Configuration for the Zarr v3 `zstd` codec.

    `level` is required; `checksum` is optional ("Should be omitted if
    false").
      https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/zstd/README.md#L9-L19
    """

    level: int
    checksum: NotRequired[bool]


class ZstdCodecObject(TypedDict, closed=True):
    """`zstd` codec metadata in object form."""

    name: ZstdCodecName
    configuration: ZstdCodecConfiguration
    must_understand: NotRequired[bool]


ZstdCodecMetadata = ZstdCodecObject
"""Permitted JSON shape for `zstd` codec metadata.

`level` is required, so only the object form is valid; the short-hand-name
form is not permitted by the spec for this codec.
  https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/zstd/README.md#L9-L19
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1562-L1564 (short-hand names only "if no configuration metadata is required")
"""

__all__ = [
    "ZSTD_CODEC_NAME",
    "ZSTD_MAX_LEVEL",
    "ZSTD_MIN_LEVEL",
    "ZstdCodec",
    "ZstdCodecConfiguration",
    "ZstdCodecMetadata",
    "ZstdCodecName",
    "ZstdCodecObject",
]


@dataclass(frozen=True)
class ZstdCodec(MetadataEntity):
    """The `zstd` codec, coerced from its metadata."""

    level: int = 0
    checksum: bool | None = None

    identifier: ClassVar[str] = ZSTD_CODEC_NAME
    kind: ClassVar[CodecKind] = "bytes_bytes"

    configuration_required: ClassVar[bool] = True
    member_types: ClassVar[MemberTypes] = {
        "level": (True, is_int),
        "checksum": (False, is_bool),
    }

    def problems(self) -> tuple[ValidationProblem, ...]:
        """zstd compression levels run -131072 to 22."""
        if not ZSTD_MIN_LEVEL <= self.level <= ZSTD_MAX_LEVEL:
            return problem(
                ("level",),
                f"expected an integer in [{ZSTD_MIN_LEVEL}, {ZSTD_MAX_LEVEL}], got {self.level}",
                "invalid_value",
            )
        return ()

    def to_json(self) -> ZstdCodecObject:
        return cast("ZstdCodecObject", super().to_json())
