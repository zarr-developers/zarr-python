"""
Gzip codec types.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/gzip/index.html
"""

from dataclasses import dataclass
from typing import ClassVar, Final, Literal, NotRequired, cast

from typing_extensions import TypedDict

from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.v3._entity import (
    CodecKind,
    MemberTypes,
    MetadataEntity,
    is_int,
    problem,
)

GZIP_CODEC_NAME: Final = "gzip"
"""The `name` field value of the `gzip` codec."""

GzipCodecName = Literal["gzip"]
"""Literal type of the `name` field of the `gzip` codec."""


class GzipCodecConfiguration(TypedDict, closed=True):
    """
    Configuration for the Zarr v3 `gzip` codec.

    `level` is an integer in the range 0-9; 0 disables compression and 9
    is slowest with the best compression ratio. The codec's compressed
    output depends on `level`, so metadata that omits it cannot
    reproducibly identify the chunk bytes produced by a writer — `level`
    is required for the metadata to fulfill its reproducibility role,
    even though the spec text does not mark it required with RFC 2119
    keywords.
      https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/codecs/gzip/index.rst#L57-L66
    """

    level: int


class GzipCodecObject(TypedDict, closed=True):
    """`gzip` codec metadata in object form."""

    name: GzipCodecName
    configuration: GzipCodecConfiguration
    must_understand: NotRequired[bool]


GzipCodecMetadata = GzipCodecObject
"""Permitted JSON shape for `gzip` codec metadata.

`configuration.level` is required (it determines the codec's output bytes
and is therefore part of the metadata's reproducibility contract), so
only the object form is valid; the short-hand-name form is not permitted.
"""

__all__ = [
    "GZIP_CODEC_NAME",
    "GzipCodec",
    "GzipCodecConfiguration",
    "GzipCodecMetadata",
    "GzipCodecName",
    "GzipCodecObject",
]


@dataclass(frozen=True)
class GzipCodec(MetadataEntity):
    """The `gzip` codec, coerced from its metadata."""

    level: int = 5

    identifier: ClassVar[str] = GZIP_CODEC_NAME
    kind: ClassVar[CodecKind] = "bytes_bytes"

    configuration_required: ClassVar[bool] = True
    member_types: ClassVar[MemberTypes] = {"level": (True, is_int)}

    def problems(self) -> tuple[ValidationProblem, ...]:
        """gzip compression levels run 0 to 9."""
        if not 0 <= self.level <= 9:
            return problem(
                ("level",), f"expected an integer in [0, 9], got {self.level}", "invalid_value"
            )
        return ()

    def to_json(self) -> GzipCodecObject:
        return cast("GzipCodecObject", super().to_json())
