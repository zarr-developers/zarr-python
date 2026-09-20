"""Composition rules for the core `blosc` codec.

`typesize` is the one configuration member of any entity this package
models whose requiredness the spec makes *conditional*: "Positive integer
specifying the stride in bytes over which shuffling is performed. Required
unless `shuffle` is `"noshuffle"`, in which case the value is ignored."

A TypedDict cannot express that, so `BloscCodecConfiguration` declares it
`NotRequired` and the condition is enforced here. Without this rule the
member the spec singles out as required would be the one member of the
blosc configuration that could always be omitted, and the v3 changelog is
explicit that this is the parameter that must now be written down: "When
shuffling is enabled, the `typesize` must now be specified explicitly in
the metadata, rather than determined implicitly from the input data."

The remaining rules judge the values the spec constrains and the shape
validator cannot: `clevel` "an integer from 0 to 9", `typesize` a
"positive integer", and `blocksize` a size in bytes, where "a value of 0
indicates that an automatic size will be used" and a negative one names
nothing.

https://zarr-specs.readthedocs.io/en/latest/v3/codecs/blosc/index.html
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.rules._registry import entity_rule
from zarr_metadata.v3._extension_points import CODECS
from zarr_metadata.v3.codec.blosc import BLOSC_CODEC_NAME

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr_metadata.rules._spec import ArrayParts

_ARRAY_V3 = "zarr_v3_array"
_NO_SHUFFLE = "noshuffle"


@entity_rule(
    _ARRAY_V3,
    CODECS,
    BLOSC_CODEC_NAME,
    reads=frozenset({"shuffle"}),
    reads_optional=frozenset({"typesize"}),
)
def typesize_is_present_when_shuffling(
    configuration: Mapping[str, object],
    document: Mapping[str, object],
    incoming: ArrayParts | None,
) -> tuple[ValidationProblem, ...]:
    """`typesize` is required unless `shuffle` is `"noshuffle"`."""
    shuffle = configuration["shuffle"]
    if shuffle == _NO_SHUFFLE or "typesize" in configuration:
        return ()
    return (
        ValidationProblem(
            ("typesize",),
            f"typesize is required when shuffle is {shuffle!r}",
            "missing_key",
        ),
    )


@entity_rule(_ARRAY_V3, CODECS, BLOSC_CODEC_NAME, reads_optional=frozenset({"typesize"}))
def typesize_is_positive(
    configuration: Mapping[str, object],
    document: Mapping[str, object],
    incoming: ArrayParts | None,
) -> tuple[ValidationProblem, ...]:
    if "typesize" not in configuration:
        return ()
    typesize = cast("int", configuration["typesize"])
    if typesize >= 1:
        return ()
    return (
        ValidationProblem(
            ("typesize",), f"expected a positive integer, got {typesize}", "invalid_value"
        ),
    )


@entity_rule(_ARRAY_V3, CODECS, BLOSC_CODEC_NAME, reads=frozenset({"clevel"}))
def clevel_is_in_range(
    configuration: Mapping[str, object],
    document: Mapping[str, object],
    incoming: ArrayParts | None,
) -> tuple[ValidationProblem, ...]:
    clevel = cast("int", configuration["clevel"])
    if 0 <= clevel <= 9:
        return ()
    return (
        ValidationProblem(
            ("clevel",), f"expected an integer in [0, 9], got {clevel}", "invalid_value"
        ),
    )


@entity_rule(_ARRAY_V3, CODECS, BLOSC_CODEC_NAME, reads=frozenset({"blocksize"}))
def blocksize_is_non_negative(
    configuration: Mapping[str, object],
    document: Mapping[str, object],
    incoming: ArrayParts | None,
) -> tuple[ValidationProblem, ...]:
    blocksize = cast("int", configuration["blocksize"])
    if blocksize >= 0:
        return ()
    return (
        ValidationProblem(
            ("blocksize",), f"expected a non-negative integer, got {blocksize}", "invalid_value"
        ),
    )
