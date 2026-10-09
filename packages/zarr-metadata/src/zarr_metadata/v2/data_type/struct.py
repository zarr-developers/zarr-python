"""The v2 `struct` type: an array of field records, each record's type a field of its own."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Annotated, Any, Final, cast

from annotated_types import Ge
from typing_extensions import ReadOnly, TypedDict

from zarr_metadata._json import ValidationProblem
from zarr_metadata.v2._definition import STRUCT_NAME, ZarrV2DataTypeDefinition, ZarrV2DataTypeField
from zarr_metadata.v2.data_type.fixed_width import ZarrV2Base64FillValue, sized_base64_rules
from zarr_metadata.v3._definition import AcceptedField

if TYPE_CHECKING:
    from collections.abc import Iterator

    from zarr_metadata.v3._definition import Nested

# The longer record first: a record that fits neither is reported by the
# first branch, and a shape that is wrong is deeper than a length that is.
ZarrV2StructRecord = (
    tuple[str, ZarrV2DataTypeField, tuple[Annotated[int, Ge(0)], ...]]
    | tuple[str, ZarrV2DataTypeField]
)
"""A field record: `[name, dtype]` or `[name, dtype, shape]`, the dtype a nested field (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L152-L174)."""


class ZarrV2StructConfiguration(TypedDict, closed=True):
    """The records of a structured dtype, which the document writes as the dtype itself."""

    fields: ReadOnly[tuple[ZarrV2StructRecord, ...]]


def _rules(configuration: ZarrV2StructConfiguration, nested: Nested) -> Iterator[ValidationProblem]:
    """At least one record, the names distinct; `""` is NumPy's name for the padding of an aligned struct, which zarr-python 2.x writes, and may repeat."""
    fields = configuration["fields"]
    if len(fields) == 0:
        yield ValidationProblem(("fields",), "expected at least one field record", "invalid_value")
    seen: dict[str, int] = {}
    for index, record in enumerate(fields):
        name = record[0]
        if name == "":
            continue
        first = seen.setdefault(name, index)
        if first != index:
            yield ValidationProblem(
                ("fields", index, 0),
                f"duplicate field name {name!r}, already used by record {first}",
                "invalid_value",
            )


def _fill_value_rules(
    configuration: ZarrV2StructConfiguration, nested: Nested, value: ZarrV2Base64FillValue
) -> Iterator[ValidationProblem]:
    """Base64 of exactly one record, when every field's size is known."""
    yield from sized_base64_rules(value, record_size(configuration, nested), exact=True)


def record_size(configuration: ZarrV2StructConfiguration, nested: Nested) -> int | None:
    """The bytes one record of `configuration` takes: each field's item size times the product of its subarray shape, a nested struct's its own record; None when a field's size is not known, as `|O`'s is not, or its dtype was not read."""
    total = 0
    for index, record in enumerate(configuration["fields"]):
        field = nested.get(("fields", index, 1))
        if not isinstance(field, AcceptedField):
            return None
        size = _item_size(field)
        if size is None:
            return None
        shape = record[2] if len(record) == 3 else ()
        total += size * math.prod(shape)
    return total


def _item_size(field: AcceptedField[Any]) -> int | None:
    """The bytes one item of the dtype `field` read takes; None when not known."""
    if field.name == STRUCT_NAME or "fields" in field.configuration:
        return record_size(cast("ZarrV2StructConfiguration", field.configuration), field.nested)
    size = field.configuration.get("itemsize")
    return size if isinstance(size, int) and not isinstance(size, bool) else None


STRUCT_V2: Final = ZarrV2DataTypeDefinition(
    name=STRUCT_NAME,
    configuration=ZarrV2StructConfiguration,
    rules=_rules,
    fill_value=ZarrV2Base64FillValue,
    fill_value_rules=_fill_value_rules,
)
"""A structured type: an array of field records; the fill value base64 of one record (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L191-L193)."""

__all__ = ["STRUCT_V2", "ZarrV2StructConfiguration", "ZarrV2StructRecord"]
