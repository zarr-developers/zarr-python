"""How a data type's values are stored: in single bytes, in several bytes each, or each in as many bytes as it needs.

Each data type's definition declares its storage. `storage_of` returns
the storage of a data type field that a scope read. The `bytes` codec
asks for the storage of the data type it receives, and a struct asks for
the storage of each of its fields.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from zarr_metadata.definition import (
    EmptyConfiguration,
    Nested,
    resolve,
)
from zarr_metadata.v3.definition import (
    CORE_AND_EXTENSIONS_CONTEXT,
    DataTypeDefinition,
    StorageClass,
    storage_of,
)

if TYPE_CHECKING:
    from zarr_metadata import JSONValue


def _struct(*field_types: JSONValue) -> JSONValue:
    fields: list[JSONValue] = [
        {"name": f"f{index}", "data_type": dt} for index, dt in enumerate(field_types)
    ]
    return {"name": "struct", "configuration": {"fields": fields}}


DATETIME: JSONValue = {
    "name": "numpy.datetime64",
    "configuration": {"unit": "s", "scale_factor": 1},
}


@pytest.mark.parametrize(
    ("data_type", "storage"),
    [
        ("bool", "single_byte"),
        ("int8", "single_byte"),
        ("uint8", "single_byte"),
        ("r8", "single_byte"),
        ("int16", "multi_byte"),
        ("uint64", "multi_byte"),
        ("float16", "multi_byte"),
        ("float64", "multi_byte"),
        ("complex64", "multi_byte"),
        (DATETIME, "multi_byte"),
        ("string", "variable_length"),
        ("bytes", "variable_length"),
        # For raw bits wider than a byte, the spec does not say whether a
        # byte order applies.
        ("r16", None),
        # A struct's values are its fields' values, packed together.
        (_struct("int8", "uint8"), "single_byte"),
        (_struct("int8", "float32"), "multi_byte"),
        (_struct("int8", _struct("uint8", "int16")), "multi_byte"),
        # A field with unknown storage makes the struct's storage unknown,
        # unless another field decides it.
        (_struct("int8", "r16"), None),
        (_struct("float32", "r16"), "multi_byte"),
        (_struct("int8", "acme.decimal"), None),
        # A data type that nothing in scope claims, or one that is not read.
        ("acme.decimal", None),
        ({"name": "numpy.datetime64", "configuration": {"unit": "s", "scale_factor": 0}}, None),
    ],
)
def test_every_data_type_says_how_its_values_are_stored(
    data_type: JSONValue, storage: StorageClass | None
) -> None:
    resolved, _ = resolve(data_type, DataTypeDefinition, CORE_AND_EXTENSIONS_CONTEXT)
    assert storage_of(resolved) == storage


def test_a_data_type_that_says_nothing_of_its_storage_leaves_it_unknown() -> None:
    quiet = DataTypeDefinition(name="acme.quiet", configuration=EmptyConfiguration)
    resolved, _ = resolve(
        "acme.quiet",
        DataTypeDefinition,
        CORE_AND_EXTENSIONS_CONTEXT.update(
            data_types=[*CORE_AND_EXTENSIONS_CONTEXT.data_types.values(), quiet]
        ),
    )
    assert storage_of(resolved) is None


def test_error_a_storage_that_gives_something_else() -> None:
    lying = DataTypeDefinition(
        name="acme.lying",
        configuration=EmptyConfiguration,
        storage=lambda configuration, nested: "two bytes",  # pyright: ignore[reportArgumentType]
    )
    resolved, _ = resolve(
        "acme.lying",
        DataTypeDefinition,
        CORE_AND_EXTENSIONS_CONTEXT.update(
            data_types=[*CORE_AND_EXTENSIONS_CONTEXT.data_types.values(), lying]
        ),
    )
    with pytest.raises(TypeError, match="'acme.lying': its storage gives one of"):
        storage_of(resolved)


def test_error_a_storage_that_raises_says_whose_it_is() -> None:
    def storage(configuration: EmptyConfiguration, nested: Nested) -> StorageClass:
        raise KeyError("bits")

    raising = DataTypeDefinition(
        name="acme.raising", configuration=EmptyConfiguration, storage=storage
    )
    resolved, _ = resolve(
        "acme.raising",
        DataTypeDefinition,
        CORE_AND_EXTENSIONS_CONTEXT.update(
            data_types=[*CORE_AND_EXTENSIONS_CONTEXT.data_types.values(), raising]
        ),
    )
    with pytest.raises(KeyError) as raised:
        storage_of(resolved)
    assert raised.value.__notes__ == ["raised by the storage of 'acme.raising'"]
