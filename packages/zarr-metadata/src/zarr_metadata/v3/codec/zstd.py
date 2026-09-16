"""
Zstandard codec types.

See https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/zstd/README.md
(the zarr-extensions registry entry; zarr-specs PR #256, which first
proposed the codec, was never merged).
"""

from typing import Final, Literal, NotRequired

from typing_extensions import TypedDict

ZSTD_CODEC_NAME: Final = "zstd"
"""The `name` field value of the `zstd` codec."""

ZstdCodecName = Literal["zstd"]
"""Literal type of the `name` field of the `zstd` codec."""


class ZstdCodecConfiguration(TypedDict):
    """
    Configuration for the Zarr v3 `zstd` codec.

    `level` is required; `checksum` is optional ("Should be omitted if
    false").
      https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/zstd/README.md#L9-L19
    """

    level: int
    checksum: NotRequired[bool]


class ZstdCodecObject(TypedDict):
    """`zstd` codec metadata in object form."""

    name: ZstdCodecName
    configuration: ZstdCodecConfiguration


ZstdCodecMetadata = ZstdCodecObject
"""Permitted JSON shape for `zstd` codec metadata.

`level` is required, so only the object form is valid; the short-hand-name
form is not permitted by the spec for this codec.
  https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/zstd/README.md#L9-L19
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1562-L1564 (short-hand names only "if no configuration metadata is required")
"""

__all__ = [
    "ZSTD_CODEC_NAME",
    "ZstdCodecConfiguration",
    "ZstdCodecMetadata",
    "ZstdCodecName",
    "ZstdCodecObject",
]
