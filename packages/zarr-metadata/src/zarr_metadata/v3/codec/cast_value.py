"""
Cast-value codec types.

See https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/cast_value/README.md
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar, Final, Literal, NotRequired, Self, cast

from zarr_metadata.model._sentinel import UNSET
from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.v3._entity import (
    DATA_TYPE,
    CodecEntity,
    CodecKind,
    Coerced,
    DataTypeEntity,
    Loc,
    MemberTypes,
    Opaque,
    is_json_value,
    one_of,
    problem,
    within,
)
from zarr_metadata.v3._parts import ArrayParts

if TYPE_CHECKING:
    from zarr_metadata.v3._registry import Context

from typing_extensions import TypedDict

from zarr_metadata._common import JSONValue
from zarr_metadata.v3._common import ZarrV3MetadataFieldJSON

CAST_VALUE_CODEC_NAME: Final = "cast_value"
"""The `name` field value of the `cast_value` codec."""

CastValueCodecName = Literal["cast_value"]
"""Literal type of the `name` field of the `cast_value` codec."""

CastRoundingMode = Literal[
    "nearest-even",
    "towards-zero",
    "towards-positive",
    "towards-negative",
    "nearest-away",
]
"""Literal type of permitted values for the `rounding` configuration field.

Defaults to `"nearest-even"` if absent.
"""

CAST_ROUNDING_MODE: Final = (
    "nearest-even",
    "towards-zero",
    "towards-positive",
    "towards-negative",
    "nearest-away",
)
"""Tuple of permitted values for the `rounding` field of the `cast_value` codec."""

CastOutOfRangeMode = Literal["clamp", "wrap"]
"""Literal type of permitted values for the `out_of_range` configuration field.

If absent, out-of-range values are an encoding/decoding error.
"""

CAST_OUT_OF_RANGE_MODE: Final = ("clamp", "wrap")
"""Tuple of permitted values for the `out_of_range` field of the `cast_value` codec."""

ScalarMapEntry = tuple[JSONValue, JSONValue]
"""A single `[input, output]` mapping in a `scalar_map` direction.

Each scalar is JSON-encoded per its data type's fill-value rules (so
e.g. `"NaN"` and `"+Infinity"` are permitted).
"""


class ScalarMap(TypedDict, closed=True):
    """Optional encode/decode scalar overrides for the cast_value codec."""

    encode: NotRequired[tuple[ScalarMapEntry, ...]]
    decode: NotRequired[tuple[ScalarMapEntry, ...]]


class CastValueCodecConfiguration(TypedDict, closed=True):
    """
    Configuration for the Zarr v3 `cast_value` codec.

    `data_type` is the target data type that input values are cast to. It
    is the same shape as the top-level array `data_type` field: either a
    bare-string primitive name or a `{name, configuration}` envelope.
    """

    data_type: ZarrV3MetadataFieldJSON
    rounding: NotRequired[CastRoundingMode]
    out_of_range: NotRequired[CastOutOfRangeMode]
    scalar_map: NotRequired[ScalarMap]


class CastValueCodecObject(TypedDict, closed=True):
    """`cast_value` codec metadata in object form."""

    name: CastValueCodecName
    configuration: CastValueCodecConfiguration
    must_understand: NotRequired[bool]


CastValueCodecMetadata = CastValueCodecObject
"""Permitted JSON shape for `cast_value` codec metadata.

`configuration.data_type` is required, so only the object form is valid;
the short-hand-name form is not permitted by the spec for this codec.
  https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/cast_value/README.md#L33-L36 and #L46-L48 (required fields)
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1562-L1564 (short-hand names only "if no configuration metadata is required")
"""


__all__ = [
    "CAST_OUT_OF_RANGE_MODE",
    "CAST_ROUNDING_MODE",
    "CAST_VALUE_CODEC_NAME",
    "SCALAR_MAP_KEYS",
    "CastOutOfRangeMode",
    "CastRoundingMode",
    "CastValueCodec",
    "CastValueCodecConfiguration",
    "CastValueCodecMetadata",
    "CastValueCodecName",
    "CastValueCodecObject",
    "ScalarMap",
    "ScalarMapEntry",
]


SCALAR_MAP_KEYS: Final = ("encode", "decode")
"""The two directions a `scalar_map` can override, both optional."""


def _is_scalar_map(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
    """An object of `[old, new]` pairs per direction."""
    if not isinstance(value, Mapping):
        return problem(loc, f"expected an object, got {value!r}")
    mapping = cast("Mapping[str, object]", value)
    found: list[ValidationProblem] = []
    for key in mapping:
        if key not in SCALAR_MAP_KEYS:
            found.extend(problem(loc, f"unexpected key {key!r}", "unknown_key"))
    for key in SCALAR_MAP_KEYS:
        if key in mapping:
            found.extend(_is_scalar_pairs(mapping[key], (*loc, key)))
    return tuple(found)


def _is_scalar_pairs(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
    if not isinstance(value, tuple):
        return problem(loc, f"expected an array of [old, new] pairs, got {value!r}")
    entries = cast("tuple[object, ...]", value)
    found: list[ValidationProblem] = []
    for index, entry in enumerate(entries):
        pair = cast("tuple[object, ...]", entry) if isinstance(entry, tuple) else ()
        if len(pair) != 2:
            found.extend(problem((*loc, index), f"expected an [old, new] pair, got {entry!r}"))
            continue
        for position, scalar in enumerate(pair):
            found.extend(is_json_value(scalar, (*loc, index, position)))
    return tuple(found)


def _is_data_type_field(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
    """A metadata field -- which data type it names is settled on recursion."""
    if not isinstance(value, (str, Mapping)):
        return problem(loc, f"expected a data type, got {value!r}")
    return ()


_UNREAD: Final = Opaque(None, "invalid")
"""Placeholder for `data_type`, which is required and so never defaulted."""


@dataclass(frozen=True)
class CastValueCodec(CodecEntity):
    """The `cast_value` codec, coerced from its metadata.

    Holds the data type it casts to, so like `sharding_indexed` it is
    read in a scope rather than on its own.
    """

    data_type: DataTypeEntity | Opaque = _UNREAD
    rounding: CastRoundingMode | UNSET = UNSET
    out_of_range: CastOutOfRangeMode | UNSET = UNSET
    scalar_map: ScalarMap | UNSET = UNSET

    identifier: ClassVar[str] = CAST_VALUE_CODEC_NAME
    kind: ClassVar[CodecKind] = "array_array"

    configuration_required: ClassVar[bool] = True
    member_types: ClassVar[MemberTypes] = {
        "data_type": (True, _is_data_type_field),
        "rounding": (False, one_of(CAST_ROUNDING_MODE)),
        "out_of_range": (False, one_of(CAST_OUT_OF_RANGE_MODE)),
        "scalar_map": (False, _is_scalar_map),
    }

    @classmethod
    def coerce(cls, value: object, context: "Context") -> Coerced[Self]:
        codec, problems = super().coerce(value, context)
        if codec is None:
            return None, problems
        data_type, found = context.coerce(
            DATA_TYPE, codec.data_type, ("configuration", "data_type")
        )
        return replace(codec, data_type=data_type), (*problems, *found)

    def problems(self) -> tuple[ValidationProblem, ...]:
        """Whatever the data type being cast to says about itself."""
        if not isinstance(self.data_type, DataTypeEntity):
            return ()
        return within(("data_type",), self.data_type.problems())

    def canonical(self) -> Self:
        """The target data type in its own canonical form."""
        if not isinstance(self.data_type, DataTypeEntity):
            return self
        return replace(self, data_type=self.data_type.canonical())

    def configuration(self) -> dict[str, object]:
        """The target data type in its canonical spelling."""
        members = super().configuration()
        data_type = self.data_type
        if isinstance(data_type, DataTypeEntity):
            members["data_type"] = data_type.to_json()
        else:
            members["data_type"] = data_type.json
        return members

    def transition(self, incoming: ArrayParts) -> ArrayParts | None:
        """The same parts, holding the type this codec casts to."""
        data_type = self.data_type
        return incoming.with_data_type(data_type if isinstance(data_type, DataTypeEntity) else None)

    def to_json(self) -> CastValueCodecObject:
        return cast("CastValueCodecObject", super().to_json())
