"""
Blosc codec types.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/blosc/index.html
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, Final, Literal, NotRequired, Self, cast

from typing_extensions import TypedDict, Unpack

from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.v3._entity import (
    CodecEntity,
    CodecKind,
    MemberTypes,
    is_int,
    one_of,
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
    "BloscCodec",
    "BloscCodecConfiguration",
    "BloscCodecMetadata",
    "BloscCodecName",
    "BloscCodecObject",
    "BloscShuffle",
    "canonical_configuration",
]


@dataclass(frozen=True)
class BloscCodec(CodecEntity):
    """The `blosc` codec, coerced from its metadata.

    Everything blosc knows about itself: the shape its metadata takes, the
    values the spec allows in it, and the simplest spelling of an
    equivalent document.
    """

    cname: BloscCName = "zstd"
    clevel: int = 5
    shuffle: BloscShuffle = "noshuffle"
    blocksize: int = 0
    typesize: int | None = None

    identifier: ClassVar[str] = BLOSC_CODEC_NAME
    kind: ClassVar[CodecKind] = "bytes_bytes"

    # Every member is required but `typesize`, which only means something
    # when shuffling; `problems` is where that conditional lives.
    configuration_required: ClassVar[bool] = True

    member_types: ClassVar[MemberTypes] = {
        "cname": (True, one_of(BLOSC_CNAME)),
        "clevel": (True, is_int),
        "shuffle": (True, one_of(BLOSC_SHUFFLE)),
        "blocksize": (True, is_int),
        "typesize": (False, is_int),
    }

    def problems(self) -> tuple[ValidationProblem, ...]:
        """The value constraints the spec places on a blosc configuration."""
        found: list[ValidationProblem] = []
        if not 0 <= self.clevel <= 9:
            found.extend(
                problem(
                    ("clevel",),
                    f"expected an integer in [0, 9], got {self.clevel}",
                    "invalid_value",
                )
            )
        if self.blocksize < 0:
            found.extend(
                problem(
                    ("blocksize",),
                    f"expected a non-negative integer, got {self.blocksize}",
                    "invalid_value",
                )
            )
        if self.typesize is not None and self.typesize < 1:
            found.extend(
                problem(
                    ("typesize",),
                    f"expected a positive integer, got {self.typesize}",
                    "invalid_value",
                )
            )
        if self.shuffle != BLOSC_NO_SHUFFLE and self.typesize is None:
            found.extend(
                problem(
                    ("typesize",),
                    f"typesize is required when shuffle is {self.shuffle!r}",
                    "missing_key",
                )
            )
        return tuple(found)

    @classmethod
    def from_configuration(cls, **configuration: Unpack[BloscCodecConfiguration]) -> Self:
        """This codec from its configuration members.

        The configuration TypedDict unpacked *is* this constructor's
        signature, so a caller with a well-typed configuration builds a
        well-typed codec, and a type checker says so at the call site.
        """
        return cls(**configuration)

    def configuration(self) -> dict[str, object]:
        """The simplest spelling of this codec's configuration.

        `typesize` is dropped under `noshuffle`, where the spec says of it
        that "the value is ignored" -- so two documents differing only
        there describe the same codec.
        """
        members = super().configuration()
        if self.shuffle == BLOSC_NO_SHUFFLE:
            members.pop("typesize", None)
        return members

    def to_json(self) -> BloscCodecObject:
        return cast("BloscCodecObject", super().to_json())
