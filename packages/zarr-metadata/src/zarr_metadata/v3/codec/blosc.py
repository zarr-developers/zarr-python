"""
Blosc codec types.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/blosc/index.html
"""

from typing import Final, Literal, NotRequired

from typing_extensions import TypedDict

BLOSC_CODEC_NAME: Final = "blosc"
"""The `name` field value of the `blosc` codec."""

BloscCodecName = Literal["blosc"]
"""Literal type of the `name` field of the `blosc` codec."""

BloscShuffle = Literal["noshuffle", "shuffle", "bitshuffle"]
"""Literal type of blosc shuffle mode names."""

BLOSC_SHUFFLE: Final = ("noshuffle", "shuffle", "bitshuffle")
"""Tuple of permitted values for the `shuffle` field of the `blosc` codec."""

BloscCName = Literal["lz4", "lz4hc", "blosclz", "snappy", "zlib", "zstd"]
"""Literal type of blosc compressor identifiers."""

BLOSC_CNAME: Final = ("lz4", "lz4hc", "blosclz", "snappy", "zlib", "zstd")
"""Tuple of permitted values for the `cname` field of the `blosc` codec."""


class BloscCodecConfiguration(TypedDict):
    """Configuration for the Zarr v3 `blosc` codec."""

    cname: BloscCName
    clevel: int
    shuffle: BloscShuffle
    blocksize: int
    typesize: NotRequired[int]


class BloscCodecObject(TypedDict):
    """`blosc` codec metadata in object form."""

    name: BloscCodecName
    configuration: BloscCodecConfiguration


BloscCodecMetadata = BloscCodecObject
"""Permitted JSON shape for `blosc` codec metadata.

The configuration has multiple required keys (`cname`, `clevel`, `shuffle`,
`blocksize`), so only the object form is valid; the short-hand-name form
is not permitted by the spec for this codec.
"""

__all__ = [
    "BLOSC_CNAME",
    "BLOSC_CODEC_NAME",
    "BLOSC_SHUFFLE",
    "BloscCName",
    "BloscCodecConfiguration",
    "BloscCodecMetadata",
    "BloscCodecName",
    "BloscCodecObject",
    "BloscShuffle",
]
