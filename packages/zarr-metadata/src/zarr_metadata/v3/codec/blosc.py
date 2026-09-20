"""
Blosc codec types.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/blosc/index.html
"""

from collections.abc import Mapping
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

BLOSC_NO_SHUFFLE: Final = "noshuffle"
"""The `shuffle` value under which `typesize` carries no information.

The spec requires `typesize` "unless `shuffle` is `"noshuffle"`, in which
case the value is ignored", so this is the one value that changes whether
another member is required.
"""

BloscCName = Literal["lz4", "lz4hc", "blosclz", "snappy", "zlib", "zstd"]
"""Literal type of blosc compressor identifiers."""

BLOSC_CNAME: Final = ("lz4", "lz4hc", "blosclz", "snappy", "zlib", "zstd")
"""Tuple of permitted values for the `cname` field of the `blosc` codec."""


class BloscCodecConfiguration(TypedDict, closed=True):
    """Configuration for the Zarr v3 `blosc` codec."""

    cname: BloscCName
    clevel: int
    shuffle: BloscShuffle
    blocksize: int
    typesize: NotRequired[int]


class BloscCodecObject(TypedDict, closed=True):
    """`blosc` codec metadata in object form."""

    name: BloscCodecName
    configuration: BloscCodecConfiguration
    must_understand: NotRequired[bool]


BloscCodecMetadata = BloscCodecObject
"""Permitted JSON shape for `blosc` codec metadata.

The configuration has multiple required keys (`cname`, `clevel`, `shuffle`,
`blocksize`), so only the object form is valid; the short-hand-name form
is not permitted by the spec for this codec.
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/codecs/blosc/index.rst#L57-L98 (configuration parameters)
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1562-L1564 (short-hand names only "if no configuration metadata is required")
"""


def canonical_configuration(configuration: Mapping[str, object]) -> Mapping[str, object]:
    """A blosc configuration in its simplest equivalent form.

    Under `shuffle: "noshuffle"` the spec says of `typesize` that "the
    value is ignored", so whatever it holds carries no meaning and two
    documents differing only there describe the same codec. Dropping it
    makes that equality visible.

    Assumes a configuration the shape validator has already accepted.
    """
    if configuration.get("shuffle") != BLOSC_NO_SHUFFLE or "typesize" not in configuration:
        return configuration
    return {key: value for key, value in configuration.items() if key != "typesize"}


__all__ = [
    "BLOSC_CNAME",
    "BLOSC_CODEC_NAME",
    "BLOSC_NO_SHUFFLE",
    "BLOSC_SHUFFLE",
    "BloscCName",
    "BloscCodecConfiguration",
    "BloscCodecMetadata",
    "BloscCodecName",
    "BloscCodecObject",
    "BloscShuffle",
    "canonical_configuration",
]
