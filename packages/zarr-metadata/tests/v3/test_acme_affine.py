"""`acme.affine`, a third-party `array_array` codec, written against the door alone.

Every element becomes `x * scale + offset`, and the result may be stored
as a different data type. Every import is from `zarr_metadata.v3.entity`
or another public module; this is the complete example the door's
docstring points at.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal, NotRequired, Self

import pytest
from typing_extensions import TypedDict

from zarr_metadata.rules import (
    Canonical,
    canonicalize_array_metadata_v3,
    validate_array_metadata_v3,
)
from zarr_metadata.v3.data_type.float32 import Float32DataType
from zarr_metadata.v3.entity import (
    CORE_AND_EXTENSIONS,
    UNSET,
    ArrayArrayCodec,
    ArrayDocumentV3,
    ArrayParts,
    Configuration,
    Configured,
    DataTypeEntity,
    MetadataValidationError,
    Opaque,
    ValidationProblem,
    ZarrV3MetadataFieldJSON,
    problem,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


class AcmeAffineConfiguration(TypedDict, closed=True):
    scale: float
    offset: NotRequired[float]
    dtype: NotRequired[ZarrV3MetadataFieldJSON]


class AcmeAffineObject(TypedDict, closed=True):
    name: Literal["acme.affine"]
    configuration: AcmeAffineConfiguration
    must_understand: NotRequired[bool]


@dataclass(frozen=True)
class AcmeAffineOptions(Configuration):
    scale: float
    offset: float | UNSET = UNSET
    dtype: DataTypeEntity | Opaque | UNSET = UNSET

    def problems(self) -> Iterator[ValidationProblem]:
        if self.scale == 0:
            yield ValidationProblem(
                ("scale",), "expected a non-zero number, got 0", "invalid_value"
            )
        if (
            isinstance(self.dtype, DataTypeEntity)
            and self.dtype.storage_class() == "variable_length"
        ):
            yield ValidationProblem(
                ("dtype",),
                f"expected a fixed-size data type, got {type(self.dtype).identifier!r}",
                "invalid_value",
            )


@dataclass(frozen=True)
class AcmeAffineCodec(ArrayArrayCodec, Configured):
    """`x * scale + offset`, stored as `dtype` if one is named."""

    configuration: AcmeAffineOptions

    identifier: ClassVar[str] = "acme.affine"
    variable_size: ClassVar[bool] = False

    @property
    def scale(self) -> float:
        return self.configuration.scale

    @property
    def offset(self) -> float | UNSET:
        return self.configuration.offset

    @property
    def dtype(self) -> DataTypeEntity | Opaque | UNSET:
        return self.configuration.dtype

    def canonical(self) -> Self:
        """An offset of 0 is the identity, and absent says the same; `dtype` in its own form."""
        return self.with_configuration(
            offset=UNSET if self.offset == 0 else self.offset,
            dtype=UNSET if self.dtype is UNSET else self.dtype.canonical(),
        )

    def incoming_problems(self, incoming: ArrayParts | None) -> tuple[ValidationProblem, ...]:
        data_type = incoming.data_type if incoming is not None else None
        if data_type is None or data_type.storage_class() != "variable_length":
            return ()
        return problem(
            (),
            f"acme.affine cannot scale variable-length data_type {type(data_type).identifier!r}",
            "invalid_value",
        )

    def transition(self, incoming: ArrayParts) -> ArrayParts | None:
        if self.dtype is UNSET:
            return incoming
        return incoming.with_data_type(
            self.dtype if isinstance(self.dtype, DataTypeEntity) else None
        )


SCOPE = CORE_AND_EXTENSIONS.extended_with(AcmeAffineCodec)
BYTES_LE = {"name": "bytes", "configuration": {"endian": "little"}}


def _affine(**configuration: object) -> dict[str, object]:
    return {"name": "acme.affine", "configuration": configuration}


def _document(**overrides: object) -> dict[str, object]:
    return {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [8],
        "data_type": "float32",
        "fill_value": 0,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [8]}},
        "chunk_key_encoding": "default",
        "codecs": [_affine(scale=2), BYTES_LE],
        **overrides,
    }


@pytest.mark.parametrize(
    "configuration",
    [
        {"scale": 2},
        {"scale": 2.5, "offset": -1},
        {"scale": -0.5, "offset": 0.25, "dtype": "float64"},
        {"scale": 3, "offset": 0, "dtype": {"name": "float32"}},
    ],
)
def test_a_valid_document_has_no_problems(configuration: dict[str, object]) -> None:
    document = _document(codecs=[_affine(**configuration), BYTES_LE])
    assert validate_array_metadata_v3(document, context=SCOPE) == ()
    assert isinstance(ArrayDocumentV3.from_json(document, context=SCOPE).codecs[0], AcmeAffineCodec)


def test_error_problems_are_located_at_their_members() -> None:
    document = _document(
        codecs=[
            {"name": "transpose", "configuration": {"order": [0]}},
            _affine(scale=0, offset=None),
            BYTES_LE,
        ]
    )
    problems = validate_array_metadata_v3(document, context=SCOPE)
    # A null offset fails the type check, so the value rules do not run
    # on the hole: one problem, at the member that has it.
    assert [(found.loc, found.kind) for found in problems] == [
        (("codecs", 1, "configuration", "offset"), "invalid_type")
    ]
    problems = validate_array_metadata_v3(
        _document(codecs=[_affine(scale=0), BYTES_LE]), context=SCOPE
    )
    assert [(found.loc, found.kind) for found in problems] == [
        (("codecs", 0, "configuration", "scale"), "invalid_value")
    ]


def test_error_a_variable_length_type_is_refused_in_and_out() -> None:
    document = _document(data_type="string", fill_value="", codecs=[_affine(scale=2), "vlen-utf8"])
    problems = validate_array_metadata_v3(document, context=SCOPE)
    assert [(found.loc, found.kind) for found in problems] == [(("codecs", 0), "invalid_value")]
    document = _document(codecs=[_affine(scale=2, dtype="string"), BYTES_LE])
    problems = validate_array_metadata_v3(document, context=SCOPE)
    assert [(found.loc, found.kind) for found in problems] == [
        (("codecs", 0, "configuration", "dtype"), "invalid_value")
    ]


def test_dtype_changes_what_the_next_codec_sees() -> None:
    # uint8 needs no endianness and float32 does, so a bare `bytes` codec
    # is fine before the transition and wrong after it.
    without = _document(data_type="uint8", codecs=[_affine(scale=2), "bytes"])
    assert validate_array_metadata_v3(without, context=SCOPE) == ()
    retyped = _document(data_type="uint8", codecs=[_affine(scale=2, dtype="float32"), "bytes"])
    problems = validate_array_metadata_v3(retyped, context=SCOPE)
    assert [(found.loc, found.kind) for found in problems] == [
        (("codecs", 1, "configuration", "endian"), "missing_key")
    ]


def test_an_out_of_scope_dtype_is_kept_and_not_judged() -> None:
    entry = _affine(scale=2, dtype="mycorp.decimal")
    document = _document(codecs=[entry, "bytes"])
    assert validate_array_metadata_v3(document, context=SCOPE) == ()
    codec = ArrayDocumentV3.from_json(document, context=SCOPE).codecs[0]
    assert isinstance(codec, AcmeAffineCodec)
    assert isinstance(codec.dtype, Opaque)
    assert codec.dtype.reason == "out_of_scope"
    assert codec.to_json() == entry


def test_round_trip_and_canonical() -> None:
    entry = _affine(scale=2, offset=1, dtype="float64")
    codec = ArrayDocumentV3.from_json(_document(codecs=[entry, BYTES_LE]), context=SCOPE).codecs[0]
    assert isinstance(codec, AcmeAffineCodec)
    assert codec.to_json() == entry
    assert AcmeAffineCodec(AcmeAffineOptions(scale=2, offset=0)).canonical() == AcmeAffineCodec(
        AcmeAffineOptions(scale=2)
    )
    document = _document(codecs=[_affine(scale=2, offset=0.0), BYTES_LE])
    result = canonicalize_array_metadata_v3(document, context=SCOPE)
    assert isinstance(result, Canonical)
    assert result.document["codecs"][0] == _affine(scale=2)


def test_constructed_by_hand() -> None:
    codec = AcmeAffineCodec(AcmeAffineOptions(scale=2.5, offset=-1, dtype=Float32DataType()))
    assert codec.to_json() == {
        "name": "acme.affine",
        "configuration": {"scale": 2.5, "offset": -1, "dtype": "float32"},
    }
    with pytest.raises(MetadataValidationError) as caught:
        AcmeAffineCodec(AcmeAffineOptions(scale=0))
    assert [(found.loc, found.kind) for found in caught.value.problems] == [
        (("scale",), "invalid_value")
    ]
