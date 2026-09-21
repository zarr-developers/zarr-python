"""
Zstandard codec types.

See https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/zstd/README.md
(the zarr-extensions registry entry; zarr-specs PR #256, which first
proposed the codec, was never merged).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Final, Literal, NotRequired

from typing_extensions import TypedDict

from zarr_metadata.model._sentinel import UNSET
from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.v3._entity import (
    BytesBytesCodec,
    Configuration,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

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
class ZstdOptions(Configuration):
    """What `zstd` is configured with."""

    level: int
    checksum: bool | UNSET = UNSET

    def problems(self) -> "Iterator[ValidationProblem]":
        if not ZSTD_MIN_LEVEL <= self.level <= ZSTD_MAX_LEVEL:
            yield ValidationProblem(
                ("level",),
                f"expected an integer in [{ZSTD_MIN_LEVEL}, {ZSTD_MAX_LEVEL}], got {self.level}",
                "invalid_value",
            )


@dataclass(frozen=True)
class ZstdCodec(BytesBytesCodec):
    """The `zstd` codec, coerced from its metadata."""

    configuration: ZstdOptions

    identifier: ClassVar[str] = ZSTD_CODEC_NAME
    variable_size: ClassVar[bool] = True

    @property
    def level(self) -> int:
        return self.configuration.level

    @property
    def checksum(self) -> bool | UNSET:
        return self.configuration.checksum
