"""The kinds of field in a Zarr v2 array document: a dtype, and a codec in `compressor` or `filters`.

A v2 dtype is written as a NumPy typestr, which is a byte order, a type
code and a size, such as `<f4`
(https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L127-L150).
A structured type is written as an array of field records
(https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L152-L174).
The definition of a type is filed under its family, such as `float`, and
the name carries the rest as configuration. For example, `<f4` reads as
`float` with `{"byteorder": "<", "itemsize": 4}`, in the same way that a
v3 `r16` reads as `r*` with `{"bits": 16}`. A records array reads as
`struct`. The records are its configuration, and the type of each record
is a nested field.

A v2 codec is a numcodecs configuration: an object whose `id` member
names the codec and whose other members are its parameters
(https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L76-L79).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Final, cast

from typing_extensions import TypeAliasType

from zarr_metadata._json import ValidationProblem, is_object, shown
from zarr_metadata.definition._definition import (
    C,
    Definition,
    WithFillValue,
)
from zarr_metadata.v2._field_json import ZarrV2DataTypeMetadata

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr_metadata._common import JSONValue
    from zarr_metadata._json import Problems
    from zarr_metadata._typed_json import Loc

ZarrV2DataTypeField = TypeAliasType("ZarrV2DataTypeField", ZarrV2DataTypeMetadata)
"""A member that holds a v2 dtype: a document's `dtype`, or the type of a struct record. It is read in the same scope as its container."""

STRUCT_NAME: Final = "struct"
"""The name that a records array reads as. No typestr has this spelling."""

FAMILIES: Final[Mapping[str, str]] = {
    "b": "bool",
    "i": "int",
    "u": "uint",
    "f": "float",
    "c": "complex",
    "m": "timedelta64",
    "M": "datetime64",
    "O": "object",
    "S": "bytes",
    "U": "str",
    "V": "void",
}
"""Each type code that the v2 spec lists, mapped to the family its definition is filed under."""

# ASCII digits only: `\d` matches every Unicode digit, but NumPy does not read them.
TYPESTR_PATTERN: Final = re.compile(r"([<>|])([A-Za-z])([0-9]*)(?:\[([0-9]*)([A-Za-z\u03bc]+)\])?")
"""A typestr: a byte order, a type code, an optional size in bytes, and an optional bracketed time unit with its multiplier."""


def parse_typestr(name: str) -> tuple[str, dict[str, JSONValue]] | None:
    """Split `name` into its type code and the values it carries. Return None if `name` is not a typestr.

    The carried values are the byte order, the item size (only `|O`
    omits it), and, when a unit is in brackets, the unit and its
    multiplier. The multiplier is 1 when none is written.
    """
    match = TYPESTR_PATTERN.fullmatch(name)
    if match is None:
        return None
    byteorder, code, size, factor, unit = match.groups()
    if size == "" and code != "O":
        # "An integer specifying the number of bytes": only `|O` omits it.
        return None
    carried: dict[str, JSONValue] = {"byteorder": byteorder}
    if size != "":
        carried["itemsize"] = int(size)
    if unit is not None:
        carried["unit"] = unit
        carried["scale_factor"] = 1 if factor == "" else int(factor)
    return code, carried


CODE_POINT_BYTES: Final = 4
"""The bytes that one code point of a `U` type takes. The size in a `U` typestr counts code points, as NumPy writes it, so `<U3` is 12 bytes. The v2 spec says the size counts bytes for every type code, but zarr-python writes the typestr NumPy gives."""


def item_bytes(name: str) -> int | None:
    """The number of bytes that one item of the typestr `name` takes: its size, or four bytes per code point for `U`. None when `name` is not a typestr with a size, such as `|O`."""
    parsed = parse_typestr(name)
    if parsed is None:
        return None
    code, carried = parsed
    size = carried.get("itemsize")
    if not isinstance(size, int):
        return None
    return size * CODE_POINT_BYTES if code == "U" else size


def typestr_problem(name: str, at: Loc) -> ValidationProblem | None:
    """Return the problem at `at` if `name` is not a typestr. Return None if it is a typestr."""
    if TYPESTR_PATTERN.fullmatch(name) is not None:
        if parse_typestr(name) is not None:
            return None
        # "An integer specifying the number of bytes": a listed code without a size.
        return ValidationProblem(
            at,
            f"expected a size in bytes after the type code, '<f4', got {shown(name)}",
            "invalid_value",
        )
    return ValidationProblem(
        at,
        "expected a NumPy typestr -- a byte order '<', '>' or '|', a type code and a size in "
        f"bytes, '<f4' -- got {shown(name)}",
        "invalid_value",
    )


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class ZarrV2DataTypeDefinition(WithFillValue[C], kind=True, format=2):
    """A v2 data type: one family of NumPy types, and the fill value an array of that family takes.

    The definition is filed under the family, such as `float`, and is
    read for every typestr of the family. The byte order, size and unit
    of the typestr are its configuration. The rules state which of those
    the family accepts. `struct` is read for an array of field records,
    with the configuration `{"fields": records}`. The document writes the
    records as the dtype itself, so a problem in a record is located
    without `fields`, for example at `("dtype", 0, 1)` for the type of the
    first record. A definition's functions still find the type of that
    record in `nested` at `("fields", 0, 1)`, its place in the
    configuration.
    """

    label: ClassVar[str] = "v2 data type"
    field_aliases: ClassVar[tuple[TypeAliasType, ...]] = (ZarrV2DataTypeField,)

    def _refusal(self) -> str | None:
        if parse_typestr(self.name) is not None:
            return (
                f"{self.name!r} is how a document writes one type of a family, which reads as "
                "the family; define the family"
            )
        # Use the explicit form, not a bare `super()`, because a dataclass with slots is rebuilt.
        return super(ZarrV2DataTypeDefinition, self)._refusal()

    @classmethod
    def _name_problem(cls, name: str, at: Loc) -> ValidationProblem | None:
        # `struct` is the name a records array reads as, never a name a
        # document writes: `_named_configuration` gives it only to an array.
        if name == STRUCT_NAME:
            return None
        return typestr_problem(name, at)

    @classmethod
    def _filed_under(cls, name: str) -> tuple[str | None, dict[str, JSONValue] | None]:
        if name == STRUCT_NAME:
            return name, None
        if name in FAMILIES.values():
            return None, None
        parsed = parse_typestr(name)
        if parsed is None:
            return name, None
        code, carried = parsed
        family = FAMILIES.get(code)
        if family is None:
            return name, None
        return family, carried

    def _carrying_name(self, configuration: Mapping[str, JSONValue]) -> str | None:
        if self.name == STRUCT_NAME:
            return None
        code = next(code for code, family in FAMILIES.items() if family == self.name)
        name = f"{configuration['byteorder']}{code}{configuration.get('itemsize', '')}"
        if "unit" in configuration:
            factor = configuration.get("scale_factor", 1)
            name += f"[{'' if factor == 1 else factor}{configuration['unit']}]"
        return name

    @classmethod
    def _named_configuration(
        cls, value: object
    ) -> tuple[str | None, Mapping[str, object] | None, Problems]:
        if isinstance(value, str):
            # The string `struct` is not a typestr; only records read as a struct.
            return (None, None, ()) if value == STRUCT_NAME else (value, None, ())
        if isinstance(value, (list, tuple)):
            return STRUCT_NAME, {"fields": cast("JSONValue", value)}, ()
        return None, None, ()

    @classmethod
    def _envelope_problems(cls, value: object) -> Problems:
        if isinstance(value, str):
            bad = typestr_problem(value, ())
            return () if bad is None else (bad,)
        if isinstance(value, (list, tuple)):
            return ()
        return (
            ValidationProblem(
                (),
                "expected a v2 dtype -- a NumPy typestr, or an array of field records -- got "
                f"{shown(value)}",
                "invalid_type",
            ),
        )

    @classmethod
    def _envelope_json(cls, name: str, configuration: Mapping[str, JSONValue]) -> JSONValue:
        if name == STRUCT_NAME and "fields" in configuration:
            return configuration["fields"]
        return name

    @classmethod
    def _in_configuration(cls, loc: Loc, place: Loc) -> Loc:
        # A v2 dtype has a configuration only as a struct's
        # `{"fields": records}`, and the document writes the records as the
        # dtype itself.
        return (*loc, *place[1:])

    @classmethod
    def _name_loc(cls, loc: Loc) -> Loc:
        return loc


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class ZarrV2CodecDefinition(Definition[C], kind=True, format=2):
    """A v2 codec: a numcodecs id, and the TypedDict that describes its parameters.

    A document writes `{"id": name, **parameters}`. The configuration of
    the definition is the parameters, read at the location of the field
    itself.
    """

    label: ClassVar[str] = "v2 codec"

    @classmethod
    def _name_problem(cls, name: str, at: Loc) -> ValidationProblem | None:
        if len(name) != 0:
            return None
        return ValidationProblem(at, "expected a codec id, got ''", "invalid_value")

    @classmethod
    def _named_configuration(
        cls, value: object
    ) -> tuple[str | None, Mapping[str, object] | None, Problems]:
        if not is_object(value):
            return None, None, ()
        name = value.get("id")
        if not isinstance(name, str):
            return None, None, ()
        return (
            name,
            {key: item for key, item in value.items() if isinstance(key, str) and key != "id"},
            (),
        )

    @classmethod
    def _envelope_problems(cls, value: object) -> Problems:
        if not is_object(value):
            return (
                ValidationProblem(
                    (), "expected a codec configuration with a string 'id'", "invalid_type"
                ),
            )
        entry = value
        if "id" not in entry:
            return (ValidationProblem(("id",), "missing required key", "missing_key"),)
        if not isinstance(entry["id"], str):
            return (
                ValidationProblem(
                    ("id",),
                    f"expected a string codec id, got {shown(entry['id'])}",
                    "invalid_type",
                ),
            )
        bad = cls._name_problem(entry["id"], ("id",))
        return () if bad is None else (bad,)

    @classmethod
    def _envelope_json(cls, name: str, configuration: Mapping[str, JSONValue]) -> JSONValue:
        return {"id": name, **configuration}

    @classmethod
    def _in_configuration(cls, loc: Loc, place: Loc) -> Loc:
        return (*loc, *place)

    @classmethod
    def _name_loc(cls, loc: Loc) -> Loc:
        return (*loc, "id")


__all__ = [
    "CODE_POINT_BYTES",
    "FAMILIES",
    "STRUCT_NAME",
    "TYPESTR_PATTERN",
    "ZarrV2CodecDefinition",
    "ZarrV2DataTypeDefinition",
    "ZarrV2DataTypeField",
    "item_bytes",
    "parse_typestr",
    "typestr_problem",
]
