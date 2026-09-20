"""A third party registering its own entity, through public API only.

Every import here is from a module without a leading underscore. If this
file has to reach into a private one, the extension surface is not real.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pytest

from zarr_metadata.model import UNSET, MetadataValidationError, ValidationProblem
from zarr_metadata.rules import (
    canonicalize_array_metadata_v3,
    validate_array_metadata_v3,
)
from zarr_metadata.v3.entity import (
    CORE_AND_EXTENSIONS,
    ArrayDocumentV3,
    ArrayParts,
    ChunkGridEntity,
    CodecEntity,
    CodecKind,
    Context,
    DataTypeEntity,
    MemberTypes,
    MetadataEntity,
    Opaque,
    StorageClass,
    is_int,
    problem,
)

ACME_MAX_ACCELERATION = 65537


@dataclass(frozen=True)
class AcmeLz4Codec(CodecEntity):
    """A third-party compressor."""

    acceleration: int | UNSET = UNSET

    identifier: ClassVar[str] = "acme.lz4"
    kind: ClassVar[CodecKind] = "bytes_bytes"
    variable_size: ClassVar[bool] = True
    member_types: ClassVar[MemberTypes] = {"acceleration": (False, is_int)}

    def problems(self) -> tuple[ValidationProblem, ...]:
        if self.acceleration is UNSET:
            return ()
        if not 1 <= self.acceleration <= ACME_MAX_ACCELERATION:
            return problem(
                ("acceleration",),
                f"expected an integer in [1, {ACME_MAX_ACCELERATION}], got {self.acceleration}",
                "invalid_value",
            )
        return ()


@dataclass(frozen=True)
class AcmeFloat8DataType(DataTypeEntity):
    """A third-party one-byte float."""

    identifier: ClassVar[str] = "acme.float8"
    scalar_storage: ClassVar[StorageClass] = "single_byte"


def _scope() -> Context:
    entities = dict(CORE_AND_EXTENSIONS.entities)
    return Context(
        {
            **entities,
            "codecs": {**entities["codecs"], AcmeLz4Codec.identifier: AcmeLz4Codec},
            "data_type": {
                **entities["data_type"],
                AcmeFloat8DataType.identifier: AcmeFloat8DataType,
            },
        }
    )


SCOPE = _scope()


def _document(**overrides: object) -> dict[str, object]:
    return {
        "zarr_format": 3,
        "node_type": "array",
        "shape": (8,),
        "data_type": "uint8",
        "fill_value": 0,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (8,)}},
        "chunk_key_encoding": "default",
        "codecs": ({"name": "bytes", "configuration": {"endian": "little"}},),
        **overrides,
    }


def test_an_unregistered_name_is_not_judged() -> None:
    # Extension openness: out of scope means unjudged, not invalid.
    document = _document(
        codecs=(
            {"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "acme.lz4", "configuration": {"acceleration": 999999}},
        )
    )
    assert validate_array_metadata_v3(document) == ()  # type: ignore[arg-type]


def test_a_registered_entity_is_judged() -> None:
    document = _document(
        codecs=(
            {"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "acme.lz4", "configuration": {"acceleration": 999999}},
        )
    )
    problems = validate_array_metadata_v3(document, context=SCOPE)  # type: ignore[arg-type]
    assert [problem.loc for problem in problems] == [("codecs", 1, "configuration", "acceleration")]


def test_a_registered_entity_joins_the_pipeline_rules() -> None:
    # Declared `bytes_bytes`, so it may not precede the array->bytes codec,
    # and it is variable-size, so it may not encode a shard index.
    document = _document(
        codecs=("acme.lz4", {"name": "bytes", "configuration": {"endian": "little"}})
    )
    problems = validate_array_metadata_v3(document, context=SCOPE)  # type: ignore[arg-type]
    assert [problem.loc for problem in problems] == [("codecs", 1)]


def test_a_registered_data_type_drives_the_codecs_around_it() -> None:
    # Single-byte, so the `bytes` codec needs no endianness for it.
    document = _document(data_type="acme.float8", fill_value=0, codecs=("bytes",))
    assert validate_array_metadata_v3(document, context=SCOPE) == ()  # type: ignore[arg-type]


def test_a_registered_entity_canonicalizes_itself() -> None:
    document = _document(
        codecs=(
            {"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "acme.lz4", "configuration": {}},
        )
    )
    result = canonicalize_array_metadata_v3(document, context=SCOPE)  # type: ignore[arg-type]
    assert result.valid is True
    assert result.document["codecs"][1] == "acme.lz4"  # type: ignore[index]


def test_error_an_entity_must_say_what_it_is() -> None:
    with pytest.raises(TypeError, match="does not declare identifier"):
        # Never bound: the guard raises while the class is being created,
        # which is the whole point -- so pyright cannot see it used.
        @dataclass(frozen=True)
        class Nameless(CodecEntity):  # pyright: ignore[reportUnusedClass]
            kind: ClassVar[CodecKind] = "bytes_bytes"


def test_error_a_registry_key_must_be_the_identifier() -> None:
    # Otherwise `resolve` never finds it and the document is silently
    # waved through, indistinguishable from openness.
    with pytest.raises(ValueError, match="registered at 'codecs' under 'acme.lz-4'"):
        Context({"codecs": {"acme.lz-4": AcmeLz4Codec}})


def test_the_entity_layer_answers_what_a_reader_needs() -> None:
    # The questions zarr-python asks before it can read a chunk.
    data_type, problems = CORE_AND_EXTENSIONS.coerce("data_type", "int32")
    assert problems == ()
    assert isinstance(data_type, DataTypeEntity)
    assert data_type.storage_class() == "multi_byte"

    grid, problems = CORE_AND_EXTENSIONS.coerce(
        "chunk_grid", {"name": "regular", "configuration": {"chunk_shape": (32, 32)}}
    )
    assert problems == ()
    assert isinstance(grid, MetadataEntity)
    parts = ArrayParts(grid.grid((64, 64)), data_type)  # type: ignore[attr-defined]
    assert parts.grid.rank == 2
    assert parts.grid.axis(0) == frozenset({32})


def test_error_an_optional_member_defaults_to_unset() -> None:
    # Otherwise every instance emits it, the bare-name spelling becomes
    # unreachable, and a canonicalized document gains a member the writer
    # never wrote.
    with pytest.raises(TypeError, match="a default other than UNSET"):

        @dataclass(frozen=True)
        class Inventive(CodecEntity):  # pyright: ignore[reportUnusedClass]
            level: int = 3

            identifier: ClassVar[str] = "acme.inventive"
            kind: ClassVar[CodecKind] = "bytes_bytes"
            member_types: ClassVar[MemberTypes] = {"level": (False, is_int)}


def test_a_reader_gets_entities_or_an_exception() -> None:
    array = ArrayDocumentV3.from_json(_document())
    assert isinstance(array.data_type, DataTypeEntity)
    assert array.data_type.storage_class() == "single_byte"
    assert array.parts.grid.rank == 1
    assert [type(codec).identifier for codec in array.codecs if isinstance(codec, CodecEntity)] == [
        "bytes"
    ]


def test_error_a_reader_gets_every_reason_at_once() -> None:
    document = _document(fill_value=-1, dimension_names=("x", "y"))
    with pytest.raises(MetadataValidationError) as raised:
        ArrayDocumentV3.from_json(document)
    assert {problem.loc for problem in raised.value.problems} == {
        ("fill_value",),
        ("dimension_names",),
    }


def test_an_unmodelled_extension_is_read_not_refused() -> None:
    # Openness: a name this reader does not model is not a failure. It
    # arrives as `Opaque`, saying which kind of not-an-entity it is, so
    # the reader can resolve it elsewhere instead of guessing.
    document = _document(
        codecs=(
            {"name": "numcodecs.bitround", "configuration": {"keepbits": 9}},
            {"name": "bytes", "configuration": {"endian": "little"}},
        )
    )
    array = ArrayDocumentV3.from_json(document)
    first = array.codecs[0]
    assert isinstance(first, Opaque)
    assert first.reason == "out_of_scope"
    assert first.json == {"name": "numcodecs.bitround", "configuration": {"keepbits": 9}}
    assert isinstance(array.codecs[1], CodecEntity)


def test_every_extension_point_is_an_exhaustive_two_case_union() -> None:
    # The property that makes the fields narrowable: an entity of the
    # right kind, or an `Opaque`. Never a bare `object`.
    array = ArrayDocumentV3.from_json(_document(data_type="mycorp.decimal", fill_value=0))
    assert isinstance(array.data_type, (DataTypeEntity, Opaque))
    assert isinstance(array.chunk_grid, (ChunkGridEntity, Opaque))
    assert isinstance(array.chunk_key_encoding, (MetadataEntity, Opaque))
    assert all(isinstance(codec, (CodecEntity, Opaque)) for codec in array.codecs)


def test_a_reader_can_choose_its_own_scope() -> None:
    document = _document(
        codecs=(
            {"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "acme.lz4", "configuration": {"acceleration": 4}},
        )
    )
    assert isinstance(ArrayDocumentV3.from_json(document).codecs[1], Opaque)
    in_scope = ArrayDocumentV3.from_json(document, context=SCOPE).codecs[1]
    assert isinstance(in_scope, AcmeLz4Codec)
    assert in_scope.acceleration == 4
