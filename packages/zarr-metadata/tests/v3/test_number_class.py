"""Test the number class a data type declares, and the arithmetic codecs that read it.

`cast_value` accepts only data types that model real numbers, on either
side of the cast, and wraps only to an integral one. `scale_offset`
accepts only data types with arithmetic. Each reads the number class that
the data type's definition declares, and accepts a data type that
declares none, since nothing is known of its values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, get_args

import pytest

from zarr_metadata.definition import EmptyConfiguration, resolve
from zarr_metadata.v3._pipeline import read_pipeline
from zarr_metadata.v3.definition import (
    CORE_AND_EXTENSIONS_CONTEXT,
    ZARR_PYTHON_CONTEXT,
    Chunk,
    CodecDefinition,
    DataTypeDefinition,
    NumberClass,
    ZarrV3Context,
)

if TYPE_CHECKING:
    from zarr_metadata.typed_json import JSONValue

NUMBER_CLASSES = {
    **dict.fromkeys(
        [
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "numpy.datetime64",
            "numpy.timedelta64",
        ],
        "integral",
    ),
    **dict.fromkeys(["float16", "float32", "float64"], "floating_point"),
    **dict.fromkeys(["complex64", "complex128"], "complex"),
    **dict.fromkeys(
        [
            "bool",
            "bytes",
            "fixed_length_utf32",
            "null_terminated_bytes",
            "r*",
            "raw_bytes",
            "string",
            "struct",
            "variable_length_bytes",
        ],
        "non_numeric",
    ),
}
"""The number class of each data type this package defines, by the name its definition is filed under."""


def test_each_data_type_declares_its_number_class() -> None:
    defined = {**CORE_AND_EXTENSIONS_CONTEXT.data_types, **ZARR_PYTHON_CONTEXT.data_types}
    assert {name: definition.number_class for name, definition in defined.items()} == (
        NUMBER_CLASSES
    )


ACME_DATA_TYPES = [
    *(
        DataTypeDefinition(
            name=f"acme.{number_class}", configuration=EmptyConfiguration, number_class=number_class
        )
        for number_class in get_args(NumberClass)
    ),
    DataTypeDefinition(name="acme.unknown", configuration=EmptyConfiguration),
]
"""An extension data type of each number class, and one that declares none."""

CONTEXT = CORE_AND_EXTENSIONS_CONTEXT.union(ZarrV3Context(data_types=ACME_DATA_TYPES))

LITTLE: JSONValue = {"name": "bytes", "configuration": {"endian": "little"}}


def _cast(data_type: JSONValue, **members: JSONValue) -> JSONValue:
    return {"name": "cast_value", "configuration": {"data_type": data_type, **members}}


def _chunk(data_type: str) -> Chunk:
    return Chunk((frozenset({4}),), resolve(data_type, DataTypeDefinition, CONTEXT)[0])


_NO_REALS = 'expected a chunk of a data type that models real numbers, got "{}"'
_NO_REAL_TARGET = 'expected a data type that models real numbers, got "{}"'
_NO_ARITHMETIC = 'expected a chunk of a data type with arithmetic, got "{}"'
_NO_WRAP = 'expected an integral data_type to wrap to, got "{}"'


@pytest.mark.parametrize(
    ("codec", "data_type", "found"),
    [
        # cast_value takes real numbers, and a data type nothing is known of.
        (_cast("int8"), "acme.integral", []),
        (_cast("int8"), "acme.floating_point", []),
        (_cast("int8"), "acme.unknown", []),
        (_cast("int8"), "acme.complex", [((0,), _NO_REALS.format("acme.complex"))]),
        (_cast("int8"), "acme.non_numeric", [((0,), _NO_REALS.format("acme.non_numeric"))]),
        # It casts to the same data types it casts from.
        (_cast("acme.integral"), "int8", []),
        (_cast("acme.floating_point"), "int8", []),
        (_cast("acme.unknown"), "int8", []),
        (
            _cast("acme.complex"),
            "int8",
            [((0, "configuration", "data_type"), _NO_REAL_TARGET.format("acme.complex"))],
        ),
        (
            _cast("acme.non_numeric"),
            "int8",
            [((0, "configuration", "data_type"), _NO_REAL_TARGET.format("acme.non_numeric"))],
        ),
        # It wraps to an integral data type, and to one nothing is known of.
        (_cast("acme.integral", out_of_range="wrap"), "int8", []),
        (_cast("acme.unknown", out_of_range="wrap"), "int8", []),
        (
            _cast("acme.floating_point", out_of_range="wrap"),
            "int8",
            [((0, "configuration", "out_of_range"), _NO_WRAP.format("acme.floating_point"))],
        ),
        # scale_offset takes every kind of number.
        ("scale_offset", "acme.integral", []),
        ("scale_offset", "acme.floating_point", []),
        ("scale_offset", "acme.complex", []),
        ("scale_offset", "acme.unknown", []),
        ("scale_offset", "acme.non_numeric", [((0,), _NO_ARITHMETIC.format("acme.non_numeric"))]),
    ],
)
def test_the_arithmetic_codecs_read_the_number_class(
    codec: JSONValue, data_type: str, found: list[tuple[tuple[str | int, ...], str]]
) -> None:
    read = [
        resolve(each, CodecDefinition, CONTEXT, (index,))
        for index, each in enumerate([codec, LITTLE])
    ]
    _, in_pipeline = read_pipeline([each for each, _ in read], _chunk(data_type))
    problems = [*(problem for _, own in read for problem in own), *in_pipeline]
    assert [(problem.loc, problem.message) for problem in problems] == found


@pytest.mark.parametrize("number_class", ["real", 1, ("integral",)])
def test_error_a_number_class_that_is_not_one(number_class: object) -> None:
    with pytest.raises(TypeError, match="'acme.d': number_class is one of .* or None, got"):
        DataTypeDefinition(
            name="acme.d",
            configuration=EmptyConfiguration,
            number_class=number_class,  # pyright: ignore[reportArgumentType]
        )
