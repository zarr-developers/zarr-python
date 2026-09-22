"""`acme.decimal`, a third-party data type, written against the door alone.

A fixed-size decimal stored as a 16-byte integer. The configuration carries `precision` (1..38 significant digits) and
`scale` (0..precision digits after the point); a fill value is a JSON
string holding a decimal literal such as `"12.50"` whose digits fit both.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Final, Literal, NotRequired

import pytest
from typing_extensions import ReadOnly, TypedDict

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


from zarr_metadata.v3.entity import (
    Configuration,
    DataTypeEntity,
    Loc,
    MetadataValidationError,
    StorageClass,
    ValidationProblem,
    problem,
    resolve,
)

ACME_DECIMAL_DATA_TYPE_NAME: Final = "acme.decimal"
"""The `name` field value of the `acme.decimal` data type."""

AcmeDecimalDataTypeName = Literal["acme.decimal"]
"""Literal type of the `name` field of the `acme.decimal` data type."""

ACME_DECIMAL_MAX_PRECISION: Final = 38
"""The most significant digits a 16-byte integer holds in every case."""

DECIMAL_LITERAL: Final = re.compile(r"-?(?P<integer>[0-9]+)(?:\.(?P<fraction>[0-9]+))?")
"""A plain decimal literal: an optional sign, digits, an optional fraction."""


class AcmeDecimalConfiguration(TypedDict, closed=True):
    """Configuration for the `acme.decimal` data type."""

    precision: ReadOnly[int]
    scale: ReadOnly[int]


class AcmeDecimal(TypedDict, closed=True):
    """`acme.decimal` data type metadata."""

    name: AcmeDecimalDataTypeName
    configuration: AcmeDecimalConfiguration
    must_understand: NotRequired[bool]


AcmeDecimalFillValue = str
"""Permitted JSON shape of the `fill_value` field for `acme.decimal`: a decimal literal."""


__all__ = [
    "ACME_DECIMAL_DATA_TYPE_NAME",
    "ACME_DECIMAL_MAX_PRECISION",
    "AcmeDecimal",
    "AcmeDecimalConfiguration",
    "AcmeDecimalDataType",
    "AcmeDecimalDataTypeName",
    "AcmeDecimalFillValue",
]


@dataclass(frozen=True)
class AcmeDecimalOptions(Configuration):
    precision: int
    scale: int

    def problems(self) -> Iterator[ValidationProblem]:
        if not 1 <= self.precision <= ACME_DECIMAL_MAX_PRECISION:
            yield ValidationProblem(
                ("precision",),
                f"expected an integer in [1, {ACME_DECIMAL_MAX_PRECISION}], got {self.precision}",
                "invalid_value",
            )
        if self.scale < 0:
            yield ValidationProblem(
                ("scale",), f"expected an integer >= 0, got {self.scale}", "invalid_value"
            )
        elif self.scale > self.precision:
            yield ValidationProblem(
                ("scale",),
                f"expected an integer <= precision ({self.precision}), got {self.scale}",
                "invalid_value",
            )


@dataclass(frozen=True)
class AcmeDecimalDataType(DataTypeEntity):
    """The `acme.decimal` data type, coerced from its metadata."""

    configuration: AcmeDecimalOptions

    identifier: ClassVar[str] = ACME_DECIMAL_DATA_TYPE_NAME
    scalar_storage: ClassVar[StorageClass] = "multi_byte"
    twos_complement: ClassVar[bool] = False

    def fill_value_problems(self, value: object, loc: Loc = ()) -> tuple[ValidationProblem, ...]:
        """A decimal literal whose digits fit `precision` and `scale`.

        Judged as written: `"12.50"` has two fractional digits whatever
        its value, so it needs a scale of at least two.
        """
        if not isinstance(value, str):
            return problem(
                loc, f"expected a decimal literal string, got {value!r}", "invalid_value"
            )
        matched = DECIMAL_LITERAL.fullmatch(value)
        if matched is None:
            return problem(
                loc, f"expected a decimal literal like '12.50', got {value!r}", "invalid_value"
            )
        integer_digits = len(matched.group("integer").lstrip("0"))
        fraction = matched.group("fraction")
        fraction_digits = 0 if fraction is None else len(fraction)
        allowed_integer_digits = self.configuration.precision - self.configuration.scale
        found: list[ValidationProblem] = []
        if fraction_digits > self.configuration.scale:
            found.extend(
                problem(
                    loc,
                    f"{value!r} has {fraction_digits} fractional digits, but scale is {self.configuration.scale}",
                    "invalid_value",
                )
            )
        if integer_digits > allowed_integer_digits:
            found.extend(
                problem(
                    loc,
                    f"{value!r} has {integer_digits} integer digits, but precision {self.configuration.precision} "
                    f"with scale {self.configuration.scale} allows {allowed_integer_digits}",
                    "invalid_value",
                )
            )
        return tuple(found)


# --- the tests --------------------------------------------------------------

from zarr_metadata.rules import canonicalize_array_metadata_v3, validate_array_metadata_v3
from zarr_metadata.v3.entity import CORE_AND_EXTENSIONS, ArrayDocumentV3, Context, Opaque

SCOPE: Context = CORE_AND_EXTENSIONS.extended_with(AcmeDecimalDataType)

LITTLE_ENDIAN_BYTES = {"name": "bytes", "configuration": {"endian": "little"}}


def _data_type(precision: int, scale: int) -> AcmeDecimal:
    return {
        "name": ACME_DECIMAL_DATA_TYPE_NAME,
        "configuration": {"precision": precision, "scale": scale},
    }


def _document(**overrides: object) -> dict[str, object]:
    return {
        "zarr_format": 3,
        "node_type": "array",
        "shape": (8,),
        "data_type": _data_type(4, 2),
        "fill_value": "12.50",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (8,)}},
        "chunk_key_encoding": "default",
        "codecs": (LITTLE_ENDIAN_BYTES,),
        **overrides,
    }


def _locs(problems: Iterable[ValidationProblem]) -> list[Loc]:
    return [problem.loc for problem in problems]


# --- documents, judged in a scope that knows the type ---------------------


def test_a_valid_document_has_no_problems() -> None:
    assert validate_array_metadata_v3(_document(), context=SCOPE) == ()


def test_a_bare_bytes_codec_is_refused_for_a_sixteen_byte_type() -> None:
    # A 16-byte integer has a byte order, so the `bytes` codec needs to be
    # told which one: the same rule the core multi-byte types are held to.
    problems = validate_array_metadata_v3(_document(codecs=("bytes",)), context=SCOPE)
    assert [(problem.loc, problem.kind) for problem in problems] == [
        (("codecs", 0, "configuration", "endian"), "missing_key")
    ]


def test_error_scale_above_precision_is_located_at_the_member() -> None:
    document = _document(data_type=_data_type(4, 5))
    problems = validate_array_metadata_v3(document, context=SCOPE)
    assert _locs(problems) == [("data_type", "configuration", "scale")]


def test_error_a_fill_value_that_does_not_fit_is_located_at_fill_value() -> None:
    problems = validate_array_metadata_v3(_document(fill_value="1234.5"), context=SCOPE)
    assert _locs(problems) == [("fill_value",)]


def test_the_type_passes_through_a_sharding_codec() -> None:
    shard = {
        "name": "sharding_indexed",
        "configuration": {
            "chunk_shape": (4,),
            "codecs": (LITTLE_ENDIAN_BYTES,),
            "index_codecs": (LITTLE_ENDIAN_BYTES, "crc32c"),
        },
    }
    document = _document(shape=(16,), codecs=(shard,))
    assert validate_array_metadata_v3(document, context=SCOPE) == ()
    array = ArrayDocumentV3.from_json(document, context=SCOPE)
    assert isinstance(array.data_type, AcmeDecimalDataType)


def test_an_inner_bare_bytes_codec_is_refused_inside_a_shard_too() -> None:
    shard = {
        "name": "sharding_indexed",
        "configuration": {
            "chunk_shape": (4,),
            "codecs": ("bytes",),
            "index_codecs": (LITTLE_ENDIAN_BYTES, "crc32c"),
        },
    }
    problems = validate_array_metadata_v3(_document(shape=(16,), codecs=(shard,)), context=SCOPE)
    assert _locs(problems) == [
        ("codecs", 0, "configuration", "codecs", 0, "configuration", "endian")
    ]


def test_an_unregistered_scope_reads_it_as_opaque_and_judges_nothing() -> None:
    document = _document(fill_value="this is not judged")
    assert validate_array_metadata_v3(document) == ()
    array = ArrayDocumentV3.from_json(document)
    assert isinstance(array.data_type, Opaque)
    assert array.data_type.reason == "out_of_scope"


# --- round trip -------------------------------------------------------------


def test_to_json_writes_back_what_was_read() -> None:
    document = _document()
    array = ArrayDocumentV3.from_json(document, context=SCOPE)
    assert isinstance(array.data_type, DataTypeEntity)
    assert array.data_type.to_json() == document["data_type"]
    assert array.data_type.canonical() == array.data_type


def test_canonical_form_keeps_the_configuration() -> None:
    result = canonicalize_array_metadata_v3(_document(), context=SCOPE)
    assert result.valid is True
    assert result.document["data_type"] == _data_type(4, 2)


# --- hand construction ------------------------------------------------------


def test_hand_construction() -> None:
    entity = AcmeDecimalDataType(AcmeDecimalOptions(precision=4, scale=2))
    assert entity.storage_class() == "multi_byte"
    assert entity.to_json() == _data_type(4, 2)
    assert entity == AcmeDecimalDataType(AcmeDecimalOptions(4, 2))
    assert hash(entity) == hash(AcmeDecimalDataType(AcmeDecimalOptions(4, 2)))


@pytest.mark.parametrize(
    ("precision", "scale", "value"),
    [
        (4, 2, "12.50"),
        (4, 2, "-99.99"),
        (4, 2, "0"),
        (4, 2, "0.5"),
        (4, 2, "0012.5"),
        (4, 0, "1234"),
        (4, 4, "0.1234"),
        (1, 1, "0.0"),
        (38, 10, "1234567890123456789012345678.0123456789"),
    ],
)
def test_fill_values_that_fit_are_accepted(precision: int, scale: int, value: str) -> None:
    entity = AcmeDecimalDataType(AcmeDecimalOptions(precision=precision, scale=scale))
    assert entity.fill_value_problems(value, ("fill_value",)) == ()


@pytest.mark.parametrize("precision", [0, 39])
def test_error_precision_out_of_range(precision: int) -> None:
    with pytest.raises(MetadataValidationError) as caught:
        AcmeDecimalDataType(AcmeDecimalOptions(precision=precision, scale=0))
    assert _locs(caught.value.problems) == [("precision",)]


def test_error_scale_negative() -> None:
    with pytest.raises(MetadataValidationError) as caught:
        AcmeDecimalDataType(AcmeDecimalOptions(precision=4, scale=-1))
    assert _locs(caught.value.problems) == [("scale",)]


def test_error_scale_above_precision() -> None:
    with pytest.raises(MetadataValidationError) as caught:
        AcmeDecimalDataType(AcmeDecimalOptions(precision=4, scale=5))
    assert [(problem.loc, problem.message) for problem in caught.value.problems] == [
        (("scale",), "expected an integer <= precision (4), got 5")
    ]


def test_error_the_constructor_stops_at_the_first_problem_and_coerce_reports_every_one() -> None:
    with pytest.raises(MetadataValidationError) as caught:
        AcmeDecimalDataType(AcmeDecimalOptions(precision=0, scale=-1))
    assert _locs(caught.value.problems) == [("precision",)]
    _, problems = resolve(
        {"name": "acme.decimal", "configuration": {"precision": 0, "scale": -1}},
        DataTypeEntity,
        SCOPE,
    )
    assert _locs(problems) == [("configuration", "precision"), ("configuration", "scale")]


def test_error_fill_value_must_be_a_string() -> None:
    entity = AcmeDecimalDataType(AcmeDecimalOptions(precision=4, scale=2))
    assert _locs(entity.fill_value_problems(12.5, ("fill_value",))) == [("fill_value",)]


@pytest.mark.parametrize("value", ["", "1e2", " 12.5", "12.", ".5", "abc", "1,5", "NaN"])
def test_error_fill_value_must_be_a_decimal_literal(value: str) -> None:
    entity = AcmeDecimalDataType(AcmeDecimalOptions(precision=4, scale=2))
    assert _locs(entity.fill_value_problems(value, ("fill_value",))) == [("fill_value",)]


def test_error_fill_value_with_too_many_fraction_digits() -> None:
    entity = AcmeDecimalDataType(AcmeDecimalOptions(precision=4, scale=2))
    problems = entity.fill_value_problems("1.234", ("fill_value",))
    assert [(problem.loc, problem.message) for problem in problems] == [
        (("fill_value",), "'1.234' has 3 fractional digits, but scale is 2")
    ]


def test_error_fill_value_with_too_many_integer_digits() -> None:
    entity = AcmeDecimalDataType(AcmeDecimalOptions(precision=4, scale=2))
    problems = entity.fill_value_problems("1234.5", ("fill_value",))
    assert [(problem.loc, problem.message) for problem in problems] == [
        (
            ("fill_value",),
            "'1234.5' has 4 integer digits, but precision 4 with scale 2 allows 2",
        )
    ]
