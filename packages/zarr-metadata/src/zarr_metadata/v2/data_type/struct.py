"""The v2 `struct` type: an array of field records. The type of each record is a field of its own."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Annotated, Any, Final, cast

from annotated_types import Ge
from typing_extensions import ReadOnly, TypedDict

from zarr_metadata._json import ValidationProblem
from zarr_metadata.definition._definition import AcceptedField
from zarr_metadata.v2._definition import (
    STRUCT_NAME,
    ZarrV2DataTypeDefinition,
    ZarrV2DataTypeField,
    item_bytes,
)
from zarr_metadata.v2.data_type._rules import sized_base64_rules
from zarr_metadata.v2.data_type.fixed_width import ZarrV2Base64FillValue

if TYPE_CHECKING:
    from collections.abc import Iterator

    from zarr_metadata.definition._definition import Nested

# List the longer record first. A record that matches neither form is
# reported against the first branch, and a wrong shape is a deeper
# problem than a wrong length.
ZarrV2StructRecord = (
    tuple[str, ZarrV2DataTypeField, tuple[Annotated[int, Ge(0)], ...]]
    | tuple[str, ZarrV2DataTypeField]
)
"""A field record: `[name, dtype]` or `[name, dtype, shape]`. The dtype is a nested field (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L152-L174)."""


class ZarrV2StructConfiguration(TypedDict, closed=True):
    """The records of a structured dtype. The document writes the records as the dtype itself."""

    fields: ReadOnly[tuple[ZarrV2StructRecord, ...]]


def _rules(configuration: ZarrV2StructConfiguration, nested: Nested) -> Iterator[ValidationProblem]:
    """Check that the names are distinct. The name `""` may repeat: NumPy uses it for the padding of an aligned struct, and zarr-python 2.x writes it. A struct with no records is allowed: NumPy's `np.dtype([])` takes no bytes, and zarr-python 2.x writes it as `[]` and reads it back."""
    fields = configuration["fields"]
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
    """Check that the fill value is base64 of exactly one record, when the size of every field is known."""
    yield from sized_base64_rules(value, _record_size(configuration, nested), exact=True)


def _record_size(configuration: ZarrV2StructConfiguration, nested: Nested) -> int | None:
    """Return the number of bytes that one record of `configuration` takes. Each field takes its item size in bytes times the product of its subarray shape, and a nested struct takes the size of its own record. Return None when the size of a field is not known (as for `|O`) or its dtype was not read."""
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
    """Return the number of bytes that one item of the dtype read as `field` takes, or None when that is not known. A typestr gives the size of its type, which for `U` counts code points of four bytes each."""
    if field.name == STRUCT_NAME or "fields" in field.configuration:
        return _record_size(cast("ZarrV2StructConfiguration", field.configuration), field.nested)
    return item_bytes(field.name)


STRUCT_V2: Final = ZarrV2DataTypeDefinition(
    name=STRUCT_NAME,
    configuration=ZarrV2StructConfiguration,
    rules=_rules,
    fill_value=ZarrV2Base64FillValue,
    fill_value_rules=_fill_value_rules,
)
"""A structured type: an array of field records. The fill value is base64 of one record (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L191-L193)."""

__all__ = ["STRUCT_V2", "ZarrV2StructConfiguration", "ZarrV2StructRecord"]
