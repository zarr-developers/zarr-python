"""
Zarr v3 `r<N>` raw-bytes data type (parameterised by bit count).

The `data_type` value is a string of the form `r<N>` where `N` is a
positive multiple of 8 (e.g. `r8`, `r16`, `r24`).

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
(https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/data-types/index.rst#L46-L47; fill value: https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/data-types/index.rst#L97-L99)
"""

import re
from dataclasses import dataclass
from typing import ClassVar, Final, NewType, Self, cast

from typing_extensions import TypedDict, Unpack

from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.v3._common import ZarrV3MetadataFieldJSON
from zarr_metadata.v3._entity import (
    Coerced,
    DataTypeEntity,
    Loc,
    StorageClass,
    named_configuration,
    problem,
)
from zarr_metadata.v3.data_type._families import byte_values

RawBytesDataTypeName = NewType("RawBytesDataTypeName", str)
"""A spec-conformant `r<N>` raw-bytes name (e.g. `"r8"`, `"r16"`).

"raw bits, variable size given by *, limited to be a multiple of 8":
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/data-types/index.rst#L46-L47
"""

RAW_BYTES_FAMILY: Final = "r<N>"
"""Canonical key for the parameterized raw-bytes data type family.

Spelled as the spec writes the family; the angle brackets keep it
unforgeable by a real name.
"""

RAW_BYTES_NAME_PATTERN: Final = re.compile(r"^r([0-9]+)$")
"""The *shape* of a raw-bytes data type name, not its validity.

ASCII digits only: `\\d` would also match every other Unicode decimal, so
`r\uff11\uff16` would be read as sixteen bits and a genuine third-party
name spelled that way would be folded into this family.

Matches every `r<N>` spelling including malformed ones (`r0`, `r12`), so
that a misspelled member of this family is recognized as belonging to it
and reported as a misspelling, rather than passing as an unknown
third-party extension. `raw_bytes_dtype_name` applies the validity rule
on top. Sole owner of this grammar: other modules match through it."""


def raw_bytes_dtype_name(value: str) -> RawBytesDataTypeName:
    """Validate `value` as a `r<N>` raw-bytes name and brand it.

    Raises ValueError if `value` is not `r` followed by a positive
    multiple of 8.
    """
    match = RAW_BYTES_NAME_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"Expected 'r' followed by a positive integer, got {value!r}")
    bits = int(match.group(1))
    if bits == 0 or bits % 8 != 0:
        raise ValueError(f"Expected 'r<N>' where N is a positive multiple of 8, got {value!r}")
    return RawBytesDataTypeName(value)


RawBytesFillValue = tuple[int, ...]
"""Permitted JSON shape of the `fill_value` field for `r<N>`.

A JSON array of N/8 integers in `[0, 255]` (one per byte).
"""


__all__ = [
    "RAW_BYTES_FAMILY",
    "RAW_BYTES_NAME_PATTERN",
    "RawBytesDataType",
    "RawBytesDataTypeName",
    "RawBytesFillValue",
    "raw_bytes_dtype_name",
]


def _name_problems(name: str) -> tuple[ValidationProblem, ...]:
    """Why `name` is not a well-formed `r<N>`, if it is not.

    "raw bits, variable size given by *, limited to be a multiple of 8"
    -- and zero bits is not a data type.
    """
    try:
        raw_bytes_dtype_name(name)
    except ValueError as error:
        return problem((), str(error), "invalid_value")
    return ()


class RawBytesMembers(TypedDict):
    """A raw-bytes type's members: the spelling, which carries the width."""

    data_type_name: str


@dataclass(frozen=True)
class RawBytesDataType(DataTypeEntity):
    """An `r<N>` raw-bytes data type, coerced from its metadata.

    One class for the whole family, because `r8` and `r4096` differ only
    in a number. That is why this is the one entity whose `identifier` is
    not a name any document carries: `r<N>` is a shape, not a spelling,
    and no real name can collide with it.

    The spelling is kept rather than the bit count, so a document comes
    back out as it went in. `r008` is a valid and distinct way of writing
    `r8`, and canonicalizing it away is not this package's call.
    """

    data_type_name: str

    scalar_storage: ClassVar[StorageClass] = "single_byte"
    identifier: ClassVar[str] = RAW_BYTES_FAMILY

    @classmethod
    def accepts(cls, name: str) -> bool:
        """Every `r<N>` spelling, valid or not.

        A malformed member of the family is recognized as belonging to it
        and reported as malformed, rather than passing unjudged as some
        third party's extension.
        """
        return RAW_BYTES_NAME_PATTERN.fullmatch(name) is not None

    @classmethod
    def coerce(cls, value: object, context: object) -> Coerced[Self]:
        name, configuration, _ = named_configuration(value)
        if name is None or not cls.accepts(name):
            return None, problem((), "expected an 'r<N>' raw-bytes data type")
        found: tuple[ValidationProblem, ...] = ()
        if configuration is not None and len(configuration) != 0:
            # Survivable, as an unknown key is everywhere else: the name
            # still says everything this type is, so it is still read and
            # its fill values are still judged. Returning nothing here let
            # a stray key hide every other problem in the document.
            found = problem(("configuration",), "'r<N>' takes no configuration", "unknown_key")
        found = (*found, *cls.value_problems(data_type_name=name))
        if any(entry.kind != "unknown_key" for entry in found):
            return None, found
        return cls.unchecked(data_type_name=name), found

    @staticmethod
    def value_problems(**members: Unpack[RawBytesMembers]) -> tuple[ValidationProblem, ...]:
        """This family's validity is in its name, not in a configuration.

        Which is the one place a member is not a configuration key, and
        why `value_problems` judges the entity's fields rather than its
        configuration: there is no configuration here to judge.
        """
        return _name_problems(members["data_type_name"])

    def to_json(self) -> ZarrV3MetadataFieldJSON:
        return cast("ZarrV3MetadataFieldJSON", self.data_type_name)

    def fill_value_problems(self, value: object, loc: Loc = ()) -> tuple[ValidationProblem, ...]:
        """One byte value per byte of the scalar.

        A malformed name says nothing about how wide the scalar is, so
        there is no length to check against; `problems` reports the name.
        """
        try:
            raw_bytes_dtype_name(self.data_type_name)
        except ValueError:
            return ()
        return byte_values(value, int(self.data_type_name[1:]) // 8, loc)
