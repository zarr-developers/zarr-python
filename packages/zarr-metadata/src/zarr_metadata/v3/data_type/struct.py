"""
Zarr `struct` data type (heterogeneous record, zarr-extensions).

See https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/struct/README.md
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar, Final, Literal, NotRequired, Self, cast

from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.v3._entity import (
    DATA_TYPE,
    DataTypeEntity,
    Loc,
    MemberTypes,
    Opaque,
    StorageClass,
    ValueRoutine,
    problem,
)

if TYPE_CHECKING:
    from zarr_metadata.v3._registry import Context

from typing_extensions import ReadOnly, TypedDict, Unpack

from zarr_metadata._common import JSONValue
from zarr_metadata.v3._common import ZarrV3MetadataFieldJSON

STRUCT_DATA_TYPE_NAME: Final = "struct"
"""The `name` field value of the `struct` data type."""

StructDataTypeName = Literal["struct"]
"""Literal type of the `name` field of the `struct` data type."""


STRUCT_FIELD_KEYS: Final = ("name", "data_type")
"""The members a struct field entry carries, both required."""


class StructField(TypedDict, closed=True):
    """
    A single field entry inside a structured dtype.

    Attributes
    ----------
    name
        The field name (must be unique within a struct and non-empty).
    data_type
        The field's data type. Recursive: may be a bare-string primitive
        or a named-config envelope including another `struct`.
    """

    name: ReadOnly[str]
    data_type: ReadOnly[ZarrV3MetadataFieldJSON]


class StructConfiguration(TypedDict, closed=True):
    """Configuration for the `struct` data type."""

    fields: ReadOnly[tuple[StructField, ...]]


class Struct(TypedDict, closed=True):
    """`struct` data type metadata."""

    name: StructDataTypeName
    configuration: StructConfiguration
    must_understand: NotRequired[bool]


StructFillValue = Mapping[str, JSONValue]
"""Permitted JSON shape of the `fill_value` field for `struct`.

A JSON object mapping each field name to that field's fill value. Field
fill values are themselves shaped per the field's `data_type`, recursively.
"""

__all__ = [
    "STRUCT_DATA_TYPE_NAME",
    "STRUCT_FIELD_KEYS",
    "Struct",
    "StructConfiguration",
    "StructDataType",
    "StructDataTypeName",
    "StructField",
    "StructFieldComponent",
    "StructFillValue",
]


def _is_fields(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
    """An array of `{name, data_type}` objects.

    Whether a `data_type` names anything is settled on recursion; this
    only asks whether the field entry has the two members at all.
    """
    if not isinstance(value, tuple):
        return problem(loc, f"expected an array of struct fields, got {value!r}")
    entries = cast("tuple[object, ...]", value)
    found: list[ValidationProblem] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            found.extend(problem((*loc, index), f"expected a struct field, got {entry!r}"))
            continue
        field = cast("Mapping[str, object]", entry)
        for key in STRUCT_FIELD_KEYS:
            if key not in field:
                found.extend(problem((*loc, index), f"missing required key {key!r}", "missing_key"))
        for key in field:
            if key not in STRUCT_FIELD_KEYS:
                found.extend(problem((*loc, index), f"unexpected key {key!r}", "unknown_key"))
        if "name" in field and not isinstance(field["name"], str):
            found.extend(
                problem((*loc, index, "name"), f"expected a string, got {field['name']!r}")
            )
    return tuple(found)


@dataclass(frozen=True)
class StructFieldComponent:
    """One field of a struct: a name, and the type of its values.

    `data_type` is the coerced entity when the field's type is in scope,
    and the metadata untouched when it is not.
    """

    name: str
    data_type: DataTypeEntity | Opaque

    def to_json(self) -> StructField:
        data_type = self.data_type
        return cast(
            "StructField",
            {
                "name": self.name,
                "data_type": (
                    data_type.to_json() if isinstance(data_type, DataTypeEntity) else data_type.json
                ),
            },
        )


class StructMembers(TypedDict):
    """A struct's members as the entity holds them.

    Not `StructConfiguration`, which describes the JSON: by the time
    values are judged, `prepare` has read each field's data type, so
    these are components holding entities rather than field objects.
    """

    fields: tuple[StructFieldComponent, ...]


def _value_problems(**members: Unpack[StructMembers]) -> tuple[ValidationProblem, ...]:
    """What a struct can judge about its own fields.

    Names have to exist, be non-empty and be distinct, because a fill
    value addresses fields by name. Field types have to be fixed-size,
    because a record's layout is otherwise not determined. Nothing about
    a field type's own values: it is an entity, so it exists only if
    those are allowed.
    """
    fields = members["fields"]
    found: list[ValidationProblem] = []
    if len(fields) == 0:
        found.extend(problem(("fields",), "expected at least one struct field", "invalid_value"))
    seen: dict[str, int] = {}
    for index, field in enumerate(fields):
        at: Loc = ("fields", index)
        if field.name == "":
            found.extend(problem((*at, "name"), "expected a non-empty field name", "invalid_value"))
        first = seen.setdefault(field.name, index)
        if first != index:
            found.extend(
                problem(
                    (*at, "name"),
                    f"duplicate field name {field.name!r}, already used by field {first}",
                    "invalid_value",
                )
            )
        if (
            isinstance(field.data_type, DataTypeEntity)
            and field.data_type.storage_class() == "variable_length"
        ):
            found.extend(
                problem(
                    (*at, "data_type"),
                    "struct fields must use fixed-size data types",
                    "invalid_value",
                )
            )
    return tuple(found)


@dataclass(frozen=True)
class StructDataType(DataTypeEntity):
    """The `struct` data type, coerced from its metadata.

    A record of named fields, each with a data type of its own -- so this
    is a data type that contains data types, and needs the scope it is
    read in to make sense of them.
    """

    fields: tuple[StructFieldComponent, ...] = ()

    identifier: ClassVar[str] = STRUCT_DATA_TYPE_NAME
    scalar_storage: ClassVar[StorageClass] = "single_byte"

    configuration_required: ClassVar[bool] = True
    member_types: ClassVar[MemberTypes] = {"fields": (True, _is_fields)}

    value_problems: ClassVar[ValueRoutine] = staticmethod(_value_problems)

    @classmethod
    def prepare(
        cls, members: dict[str, object], context: "Context"
    ) -> tuple[dict[str, object], tuple[ValidationProblem, ...]]:
        """Each field's data type, read in this scope."""
        fields: list[StructFieldComponent] = []
        found: list[ValidationProblem] = []
        for index, entry in enumerate(cast("tuple[object, ...]", members["fields"])):
            field = cast("Mapping[str, object]", entry)
            data_type, from_field = context.coerce(
                DATA_TYPE, field["data_type"], ("configuration", "fields", index, "data_type")
            )
            found.extend(from_field)
            fields.append(
                StructFieldComponent(name=cast("str", field["name"]), data_type=data_type)
            )
        return {**members, "fields": tuple(fields)}, tuple(found)

    def storage_class(self) -> StorageClass | None:
        """The widest class among the fields.

        A struct of `uint8` and `int32` is `multi_byte`, and one holding
        a `string` anywhere inside it is `variable_length`. None when any
        field's type is out of scope: the answer would be a guess.
        """
        widest: StorageClass = "single_byte"
        for field in self.fields:
            if not isinstance(field.data_type, DataTypeEntity):
                return None
            found = field.data_type.storage_class()
            if found is None:
                return None
            if found == "variable_length":
                return "variable_length"
            if found == "multi_byte":
                widest = "multi_byte"
        return widest

    def fill_value_problems(self, value: object, loc: Loc = ()) -> tuple[ValidationProblem, ...]:
        """A fill value per field, addressed by name.

        Every field needs one and nothing else may appear, because a
        record's value is not determined otherwise. A field whose type is
        out of scope still needs an entry -- that much is structural --
        but what the entry holds is left unjudged.
        """
        if not isinstance(value, Mapping):
            return problem(
                loc,
                f"expected an object of per-field fill values, got {value!r}",
                "invalid_value",
            )
        fills = cast("Mapping[str, object]", value)
        found: list[ValidationProblem] = []
        for field in self.fields:
            at: Loc = (*loc, field.name)
            if field.name not in fills:
                found.extend(
                    problem(
                        at, f"missing fill value for struct field {field.name!r}", "missing_key"
                    )
                )
                continue
            if not isinstance(field.data_type, DataTypeEntity):
                continue
            found.extend(field.data_type.fill_value_problems(fills[field.name], at))
        declared = {field.name for field in self.fields}
        found.extend(
            ValidationProblem((*loc, key), f"unknown struct fill field {key!r}", "unknown_key")
            for key in sorted(fills.keys() - declared)
        )
        return tuple(found)

    def canonical(self) -> Self:
        """Each field's data type in its own canonical form."""
        return replace(
            self,
            fields=tuple(
                replace(field, data_type=field.data_type.canonical())
                if isinstance(field.data_type, DataTypeEntity)
                else field
                for field in self.fields
            ),
        )

    def configuration(self) -> dict[str, object]:
        """Each field in its canonical spelling, type included."""
        return {"fields": tuple(field.to_json() for field in self.fields)}

    def to_json(self) -> Struct:
        return cast("Struct", super().to_json())
