"""The kinds of field a Zarr v2 array document holds: a dtype, and a codec in `compressor` or `filters`.

A v2 dtype is written as NumPy writes a typestr -- a byte order, a type
code and a size, `<f4`
(https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L127-L150)
-- or as an array of field records for a structured type
(https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L152-L174).
The definition of a type is filed under its family, `float`, and the
name carries the rest: `<f4` reads as `float` with
`{"byteorder": "<", "itemsize": 4}`, as a v3 `r16` reads as `r*` with
`{"bits": 16}`. A records array reads as `struct`, its records the
configuration, each record's type a nested field.

A v2 codec is a numcodecs configuration: an object whose `id` names
the codec and whose other members are its parameters
(https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L76-L79).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Final, cast

from typing_extensions import TypeAliasType

from zarr_metadata._json import ValidationProblem, is_object, shown
from zarr_metadata.v2.array import ZarrV2DataTypeMetadata
from zarr_metadata.v3._definition import C, Definition, WithFillValue

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr_metadata._common import JSONValue
    from zarr_metadata._typed_json import Loc
    from zarr_metadata.v3._definition import Problems

ZarrV2DataTypeField = TypeAliasType("ZarrV2DataTypeField", ZarrV2DataTypeMetadata)
"""A member holding a v2 dtype: a document's `dtype`, or a struct record's type; read in the scope what holds it is read in."""

STRUCT_NAME: Final = "struct"
"""The name a records array reads as, which no typestr spells."""

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
"""Each type code the v2 spec lists, and the family its definition is filed under."""

# ASCII digits only: `\d` matches every Unicode digit, which NumPy does not read.
TYPESTR_PATTERN: Final = re.compile(r"([<>|])([A-Za-z])([0-9]*)(?:\[([0-9]*)([A-Za-z\u03bc]+)\])?")
"""A typestr: a byte order, a type code, a size in bytes if any, and a bracketed time unit with its multiplier if any."""


def parse_typestr(name: str) -> tuple[str, dict[str, JSONValue]] | None:
    """`name` split into its type code and what it carries; None when it is no typestr.

    What is carried is the byte order; the item size, which only `|O`
    leaves out; and, when a unit is bracketed, the unit
    and its multiplier, 1 when none is written.
    """
    match = TYPESTR_PATTERN.fullmatch(name)
    if match is None:
        return None
    byteorder, code, size, factor, unit = match.groups()
    if size == "" and code != "O":
        # "An integer specifying the number of bytes": only `|O` writes none.
        return None
    carried: dict[str, JSONValue] = {"byteorder": byteorder}
    if size != "":
        carried["itemsize"] = int(size)
    if unit is not None:
        carried["unit"] = unit
        carried["scale_factor"] = 1 if factor == "" else int(factor)
    return code, carried


def typestr_problem(name: str, at: Loc) -> ValidationProblem | None:
    """The problem `name`, at `at`, is when it is no typestr; None when it is one, or is `struct`."""
    if name == STRUCT_NAME:
        return None
    if TYPESTR_PATTERN.fullmatch(name) is not None:
        if parse_typestr(name) is not None:
            return None
        # "An integer specifying the number of bytes": a listed code without one.
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
class ZarrV2DataTypeDefinition(WithFillValue[C]):
    """A v2 data type: one family of NumPy types, and the fill value an array of it takes.

    Filed under the family -- `float` -- and read for every typestr of
    the family, whose byte order, size and unit are its configuration;
    the rules say which of those the family takes. `struct` is read for
    an array of field records, its configuration `{"fields": records}`,
    so a problem in a record is located under `fields`: at
    `("dtype", "fields", 0, 1)` for the type of the first record.
    """

    is_kind: ClassVar[bool] = True
    label: ClassVar[str] = "v2 data type"
    field_aliases: ClassVar[tuple[TypeAliasType, ...]] = (ZarrV2DataTypeField,)

    def _refusal(self) -> str | None:
        if parse_typestr(self.name) is not None:
            return (
                f"{self.name!r} is how a document writes one type of a family, which reads as "
                "the family; define the family"
            )
        # Named, not a bare `super()`: a dataclass with slots is rebuilt.
        return super(ZarrV2DataTypeDefinition, self)._refusal()

    @classmethod
    def name_problem(cls, name: str, at: Loc) -> ValidationProblem | None:
        return typestr_problem(name, at)

    @classmethod
    def spelled(cls, name: str) -> tuple[str | None, dict[str, JSONValue] | None]:
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

    def carrying_name(self, configuration: Mapping[str, JSONValue]) -> str | None:
        if self.name == STRUCT_NAME:
            return None
        code = next(code for code, family in FAMILIES.items() if family == self.name)
        name = f"{configuration['byteorder']}{code}{configuration.get('itemsize', '')}"
        if "unit" in configuration:
            factor = configuration.get("scale_factor", 1)
            name += f"[{'' if factor == 1 else factor}{configuration['unit']}]"
        return name

    @classmethod
    def named_configuration(
        cls, value: object
    ) -> tuple[str | None, Mapping[str, object] | None, Problems]:
        if isinstance(value, str):
            return value, None, ()
        if isinstance(value, (list, tuple)):
            return STRUCT_NAME, {"fields": cast("JSONValue", value)}, ()
        return None, None, ()

    @classmethod
    def envelope_problems(cls, value: object) -> Problems:
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
    def envelope_json(cls, name: str, configuration: Mapping[str, JSONValue]) -> JSONValue:
        if name == STRUCT_NAME and "fields" in configuration:
            return configuration["fields"]
        return name

    @classmethod
    def configuration_loc(cls, loc: Loc) -> Loc:
        return loc

    @classmethod
    def name_loc(cls, loc: Loc) -> Loc:
        return loc


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class ZarrV2CodecDefinition(Definition[C]):
    """A v2 codec: a numcodecs id, and the TypedDict its parameters are.

    A document writes `{"id": name, **parameters}`; the definition's
    configuration is the parameters, read at the field itself.
    """

    is_kind: ClassVar[bool] = True
    label: ClassVar[str] = "v2 codec"

    @classmethod
    def name_problem(cls, name: str, at: Loc) -> ValidationProblem | None:
        if len(name) != 0:
            return None
        return ValidationProblem(at, "expected a codec id, got ''", "invalid_value")

    @classmethod
    def named_configuration(
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
    def envelope_problems(cls, value: object) -> Problems:
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
        bad = cls.name_problem(entry["id"], ("id",))
        return () if bad is None else (bad,)

    @classmethod
    def envelope_json(cls, name: str, configuration: Mapping[str, JSONValue]) -> JSONValue:
        return {"id": name, **configuration}

    @classmethod
    def configuration_loc(cls, loc: Loc) -> Loc:
        return loc

    @classmethod
    def name_loc(cls, loc: Loc) -> Loc:
        return (*loc, "id")


__all__ = [
    "FAMILIES",
    "STRUCT_NAME",
    "TYPESTR_PATTERN",
    "ZarrV2CodecDefinition",
    "ZarrV2DataTypeDefinition",
    "ZarrV2DataTypeField",
    "parse_typestr",
    "typestr_problem",
]
