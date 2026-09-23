"""
Blosc codec types.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/blosc/index.html
"""

from collections.abc import Iterator
from typing import Final, Literal, NotRequired, cast

from typing_extensions import TypedDict

from zarr_metadata._json import ValidationProblem
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

BLOSC_NO_SHUFFLE: Final = "noshuffle"
"""The `shuffle` value under which `typesize` carries no information.

The spec requires `typesize` "unless `shuffle` is `"noshuffle"`, in which
case the value is ignored", so this is the one value that changes whether
another member is required.
"""


def _rules(configuration: BloscCodecConfiguration) -> Iterator[ValidationProblem]:
    """Bounds on `clevel` and `blocksize`; `typesize` against `shuffle`.

    Under `noshuffle` the spec says of `typesize` that "the value is
    ignored", and the canonical form drops it; under either shuffle it is
    required, and positive.
    """
    clevel, blocksize = configuration["clevel"], configuration["blocksize"]
    if not 0 <= clevel <= 9:
        yield ValidationProblem(
            ("clevel",), f"expected an integer in [0, 9], got {clevel}", "invalid_value"
        )
    if blocksize < 0:
        yield ValidationProblem(
            ("blocksize",), f"expected an integer >= 0, got {blocksize}", "invalid_value"
        )
    shuffle = configuration["shuffle"]
    if shuffle != BLOSC_NO_SHUFFLE:
        typesize = configuration.get("typesize")
        if typesize is None:
            yield ValidationProblem(
                ("typesize",), f"typesize is required when shuffle is {shuffle!r}", "missing_key"
            )
        elif typesize < 1:
            yield ValidationProblem(
                ("typesize",), f"expected a positive integer, got {typesize}", "invalid_value"
            )


def _canonical(configuration: BloscCodecConfiguration) -> BloscCodecConfiguration:
    """Without a `typesize` that `noshuffle` renders meaningless.

    The spec says of that case that "the value is ignored", so two
    configurations differing only there describe the same codec.
    """
    if configuration["shuffle"] != BLOSC_NO_SHUFFLE or "typesize" not in configuration:
        return configuration
    return cast(
        "BloscCodecConfiguration",
        {key: value for key, value in configuration.items() if key != "typesize"},
    )


BLOSC_CODEC: Final = CodecDefinition(
    name=BLOSC_CODEC_NAME,
    configuration=BloscCodecConfiguration,
    kind="bytes_bytes",
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
