"""
Blosc codec types.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/blosc/index.html
"""

from collections.abc import Iterator
from typing import Annotated, Final, Literal, NotRequired

from annotated_types import Ge, Interval
from typing_extensions import TypedDict

from zarr_metadata._json import ValidationProblem
from zarr_metadata.definition._definition import Nested
from zarr_metadata.v3._definition import CodecDefinition

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


class BloscCodecConfiguration(TypedDict, closed=True):
    """Configuration for the Zarr v3 `blosc` codec."""

    cname: BloscCName
    """The compressor blosc applies to each block, one of `BLOSC_CNAME`."""
    clevel: Annotated[int, Interval(ge=0, le=9)]
    """The compression level, an integer from 0 to 9.

    0 turns compression off; 1 is the fastest and compresses least, 9 the
    slowest and compresses most.
    """
    shuffle: BloscShuffle
    """The shuffling applied before compression.

    `"noshuffle"` for none, `"shuffle"` for byte-wise, `"bitshuffle"` for
    bit-wise.
    """
    blocksize: Annotated[int, Ge(0)]
    """The size in bytes of the blocks the buffer is divided into before compression.

    At least 0; 0 lets blosc choose the size.
    """
    typesize: NotRequired[int]
    """The stride in bytes over which shuffling is performed.

    Required, and at least 1, unless `shuffle` is `"noshuffle"`, under
    which the spec ignores it and `canonicalize` drops it.
    """


class BloscCodecObject(TypedDict, closed=True):
    """`blosc` codec metadata in object form."""

    name: BloscCodecName
    """The codec's name, `"blosc"`."""
    configuration: BloscCodecConfiguration
    """The codec's parameters."""
    must_understand: NotRequired[bool]
    """Whether a reader that does not recognize this codec must refuse the document.

    Absent means `true`. The spec permits `false` for a codec
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1571-L1581),
    though `resolve` reports a `false` as a problem at every extension point.
    """


BloscCodecMetadata = BloscCodecObject
"""Permitted JSON shape for `blosc` codec metadata.

The configuration has multiple required keys (`cname`, `clevel`, `shuffle`,
`blocksize`), so only the object form is valid; the short-hand-name form
is not permitted by the spec for this codec.
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/codecs/blosc/index.rst#L57-L98 (configuration parameters)
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1562-L1564 (short-hand names only "if no configuration metadata is required")
"""

BLOSC_NO_SHUFFLE: Final = "noshuffle"
"""The `shuffle` value under which `typesize` carries no information.

The spec requires `typesize` "unless `shuffle` is `"noshuffle"`, in which
case the value is ignored", so this is the one value that changes whether
another member is required.
"""


def _rules(configuration: BloscCodecConfiguration, nested: Nested) -> Iterator[ValidationProblem]:
    """Check `typesize` against `shuffle`.

    For `noshuffle`, the spec says of `typesize` that "the value is
    ignored", and the canonical form drops it. For either other shuffle,
    `typesize` is required and must be positive. The type cannot express
    whether `typesize` is required, so it does not express the bound
    either. These rules check both.
    """
    shuffle = configuration["shuffle"]
    if shuffle != BLOSC_NO_SHUFFLE:
        typesize = configuration.get("typesize")
        if typesize is None:
            yield ValidationProblem(
                ("typesize",), f"typesize is required when shuffle is {shuffle!r}", "missing_key"
            )
        elif typesize < 1:
            yield ValidationProblem(
                ("typesize",),
                f"expected a positive integer, got {typesize}",
                "invalid_value",
                ctx={"ge": 1},
            )


def _canonical(configuration: BloscCodecConfiguration) -> BloscCodecConfiguration:
    """Remove `typesize` when `shuffle` is `noshuffle`, because it has no meaning there.

    The spec says of that case that "the value is ignored", so two
    configurations that differ only in `typesize` describe the same codec.
    """
    if configuration["shuffle"] != BLOSC_NO_SHUFFLE or "typesize" not in configuration:
        return configuration
    simpler = configuration.copy()
    del simpler["typesize"]
    return simpler


BLOSC_CODEC: Final = CodecDefinition(
    name=BLOSC_CODEC_NAME,
    configuration=BloscCodecConfiguration,
    kind="bytes_bytes",
    size="dynamic",
    rules=_rules,
    canonical=_canonical,
)
"""The `blosc` codec."""


__all__ = [
    "BLOSC_CNAME",
    "BLOSC_CODEC",
    "BLOSC_CODEC_NAME",
    "BLOSC_NO_SHUFFLE",
    "BLOSC_SHUFFLE",
    "BloscCName",
    "BloscCodecConfiguration",
    "BloscCodecMetadata",
    "BloscCodecName",
    "BloscCodecObject",
    "BloscShuffle",
]
