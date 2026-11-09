"""
Zstandard codec types.

See https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/zstd/README.md
(the zarr-extensions registry entry; zarr-specs PR #256, which first
proposed the codec, was never merged).
"""

from typing import Annotated, Final, Literal, NotRequired

from annotated_types import Interval
from typing_extensions import TypedDict

from zarr_metadata.v3._definition import CodecDefinition

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

    level: Annotated[int, Interval(ge=ZSTD_MIN_LEVEL, le=ZSTD_MAX_LEVEL)]
    """The compression level, an integer from `ZSTD_MIN_LEVEL` (-131072) to `ZSTD_MAX_LEVEL` (22).

    0 means zstd's default level; otherwise a higher level compresses more,
    more slowly. It has no effect on decoding.
    """
    checksum: NotRequired[bool]
    """Whether a checksum is stored when writing and verified when reading.

    Absent means `false`; the registry says to omit it rather than write
    `false`.
    """


class ZstdCodecObject(TypedDict, closed=True):
    """`zstd` codec metadata in object form."""

    name: ZstdCodecName
    """The codec's name, `"zstd"`."""
    configuration: ZstdCodecConfiguration
    """The codec's parameters."""
    must_understand: NotRequired[bool]
    """Whether a reader that does not recognize this codec must refuse the document.

    Absent means `true`. The spec permits `false` for a codec
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1571-L1581),
    though `resolve` reports a `false` as a problem at every extension point.
    """


ZstdCodecMetadata = ZstdCodecObject
"""Permitted JSON shape for `zstd` codec metadata.

`level` is required, so only the object form is valid; the short-hand-name
form is not permitted by the spec for this codec.
  https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/zstd/README.md#L9-L19
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1562-L1564 (short-hand names only "if no configuration metadata is required")
"""


def _canonical(configuration: ZstdCodecConfiguration) -> ZstdCodecConfiguration:
    """Remove a `checksum` of `false`, because an absent `checksum` means the same thing.

    The spec says of `checksum` that it "should be omitted if false"
    (https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/zstd/README.md?plain=1#L17-L19),
    so the two spellings describe the same codec.
    """
    if configuration.get("checksum") is not False:
        return configuration
    simpler = configuration.copy()
    del simpler["checksum"]
    return simpler


ZSTD_CODEC: Final = CodecDefinition(
    name=ZSTD_CODEC_NAME,
    configuration=ZstdCodecConfiguration,
    canonical=_canonical,
    kind="bytes_bytes",
    size="dynamic",
)
"""The `zstd` codec."""


__all__ = [
    "ZSTD_CODEC",
    "ZSTD_CODEC_NAME",
    "ZSTD_MAX_LEVEL",
    "ZSTD_MIN_LEVEL",
    "ZstdCodecConfiguration",
    "ZstdCodecMetadata",
    "ZstdCodecName",
    "ZstdCodecObject",
]
