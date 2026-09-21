"""
Blosc codec types.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/blosc/index.html
"""

from dataclasses import dataclass, replace
from typing import ClassVar, Final, Literal, NotRequired, Self, cast

from typing_extensions import TypedDict, Unpack

from zarr_metadata.model._sentinel import UNSET
from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.v3._entity import (
    CodecEntity,
    CodecKind,
    problem,
)

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


__all__ = [
    "BLOSC_CNAME",
    "BLOSC_CODEC_NAME",
    "BLOSC_NO_SHUFFLE",
    "BLOSC_SHUFFLE",
    "BloscCName",
    "BloscCodec",
    "BloscCodecConfiguration",
    "BloscCodecMetadata",
    "BloscCodecName",
    "BloscCodecObject",
    "BloscShuffle",
]


@dataclass(frozen=True)
class BloscCodec(CodecEntity):
    """The `blosc` codec, coerced from its metadata.

    Everything blosc knows about itself: the shape its metadata takes, the
    values the spec allows in it, and the simplest spelling of an
    equivalent document.
    """

    cname: BloscCName
    clevel: int
    shuffle: BloscShuffle
    blocksize: int
    typesize: int | UNSET = UNSET

    identifier: ClassVar[str] = BLOSC_CODEC_NAME
    variable_size: ClassVar[bool] = True
    kind: ClassVar[CodecKind] = "bytes_bytes"

    # Every member is required but `typesize`, which only means something
    # when shuffling; `problems` is where that conditional lives.

    @staticmethod
    def value_problems(
        **members: Unpack[BloscCodecConfiguration],
    ) -> tuple[ValidationProblem, ...]:
        """The value constraints the spec places on a blosc configuration."""
        found: list[ValidationProblem] = []
        clevel = members["clevel"]
        if not 0 <= clevel <= 9:
            found.extend(
                problem(
                    ("clevel",), f"expected an integer in [0, 9], got {clevel}", "invalid_value"
                )
            )
        blocksize = members["blocksize"]
        if blocksize < 0:
            found.extend(
                problem(
                    ("blocksize",),
                    f"expected a non-negative integer, got {blocksize}",
                    "invalid_value",
                )
            )
        shuffle = members["shuffle"]
        typesize = members.get("typesize")
        # Only where it means something: under `noshuffle` the spec says
        # "the value is ignored", and `canonical` drops it.
        if typesize is not None and shuffle != BLOSC_NO_SHUFFLE and typesize < 1:
            found.extend(
                problem(
                    ("typesize",), f"expected a positive integer, got {typesize}", "invalid_value"
                )
            )
        if shuffle != BLOSC_NO_SHUFFLE and typesize is None:
            found.extend(
                problem(
                    ("typesize",),
                    f"typesize is required when shuffle is {shuffle!r}",
                    "missing_key",
                )
            )
        return tuple(found)

    def canonical(self) -> Self:
        """Without a `typesize` that `noshuffle` renders meaningless.

        The spec says of that case that "the value is ignored", so two
        documents differing only there describe the same codec.
        """
        canonical = super().canonical()
        if canonical.shuffle != BLOSC_NO_SHUFFLE or canonical.typesize is UNSET:
            return canonical
        return replace(canonical, typesize=UNSET)

    def to_json(self) -> BloscCodecObject:
        return cast("BloscCodecObject", super().to_json())
