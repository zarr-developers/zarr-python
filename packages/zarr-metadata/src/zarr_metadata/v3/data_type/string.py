"""
Zarr `string` data type (variable-length utf-8, zarr-extensions).

See https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/string/README.md
"""

from dataclasses import dataclass, field
from typing import ClassVar, Final, Literal

from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.v3._entity import (
    Configuration,
    DataTypeEntity,
    Loc,
    StorageClass,
    problem,
)

STRING_DATA_TYPE_NAME: Final = "string"
"""The `data_type` value for the `string` type."""

StringDataTypeName = Literal["string"]
"""Literal type of the `data_type` field for `string`."""

StringFillValue = str
"""Permitted JSON shape of the `fill_value` field for `string`: a JSON unicode string."""


__all__ = [
    "STRING_DATA_TYPE_NAME",
    "StringDataType",
    "StringDataTypeName",
    "StringFillValue",
]


@dataclass(frozen=True)
class StringDataType(DataTypeEntity):
    """The `string` data type. The name says everything."""

    configuration: Configuration = field(default_factory=Configuration)

    scalar_storage: ClassVar[StorageClass] = "variable_length"
    identifier: ClassVar[str] = STRING_DATA_TYPE_NAME

    def fill_value_problems(self, value: object, loc: Loc = ()) -> tuple[ValidationProblem, ...]:
        if not isinstance(value, str):
            return problem(loc, f"expected a string, got {value!r}", "invalid_value")
        return ()
