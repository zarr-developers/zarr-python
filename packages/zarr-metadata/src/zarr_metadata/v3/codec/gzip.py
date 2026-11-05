"""
Gzip codec types.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/gzip/index.html
"""

from typing import Annotated, Final, Literal, NotRequired

from annotated_types import Interval
from typing_extensions import TypedDict

from zarr_metadata.v3._definition import CodecDefinition

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

    level: Annotated[int, Interval(ge=0, le=9)]
    """The compression level, an integer from 0 to 9.

    0 turns compression off; 1 is the fastest and compresses least, 9 the
    slowest and compresses most.
    """


class GzipCodecObject(TypedDict, closed=True):
    """`gzip` codec metadata in object form."""

    name: GzipCodecName
    """The codec's name, `"gzip"`."""
    configuration: GzipCodecConfiguration
    """The codec's parameters."""
    must_understand: NotRequired[bool]
    """Whether a reader that does not recognize this codec must refuse the document.

    Absent means `true`. The spec permits `false` for a codec
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1571-L1581),
    though `resolve` reports a `false` as a problem at every extension point.
    """


GzipCodecMetadata = GzipCodecObject
"""Permitted JSON shape for `gzip` codec metadata.

`configuration.level` is required (it determines the codec's output bytes
and is therefore part of the metadata's reproducibility contract), so
only the object form is valid; the short-hand-name form is not permitted.
"""


GZIP_CODEC: Final = CodecDefinition(
    name=GZIP_CODEC_NAME,
    configuration=GzipCodecConfiguration,
    kind="bytes_bytes",
    size="dynamic",
)
"""The `gzip` codec."""

__all__ = [
    "GZIP_CODEC",
    "GZIP_CODEC_NAME",
    "GzipCodecConfiguration",
    "GzipCodecMetadata",
    "GzipCodecName",
    "GzipCodecObject",
]
