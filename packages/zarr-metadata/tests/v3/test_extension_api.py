"""A third party registering its own entity, through public API only.

Every import here is from a module without a leading underscore. If this
file has to reach into a private one, the extension surface is not real.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Annotated, ClassVar, Literal, NotRequired, Self, cast

import pytest
from typing_extensions import TypedDict

from zarr_metadata.model import UNSET, MetadataValidationError
from zarr_metadata.rules import (
    canonicalize_array_metadata_v3,
    validate_array_metadata_v3,
)
from zarr_metadata.v3._common import ZarrV3MetadataFieldJSON
from zarr_metadata.v3._entity import json_type_of
from zarr_metadata.v3.codec.blosc import BloscCodec
from zarr_metadata.v3.codec.gzip import GzipCodec, GzipCodecObject
from zarr_metadata.v3.entity import (
    CORE,
    CORE_AND_EXTENSIONS,
    FROM_NAME,
    ArrayDocumentV3,
    ArrayParts,
    ChunkGridEntity,
    CodecEntity,
    CodecKind,
    Context,
    DataTypeEntity,
    IntegerDataType,
    MetadataEntity,
    Opaque,
    StorageClass,
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

    def __post_init__(self) -> None:
        if self.acceleration is not UNSET and not 1 <= self.acceleration <= ACME_MAX_ACCELERATION:
            raise MetadataValidationError(
                problem(
                    ("acceleration",),
                    f"expected an integer in [1, {ACME_MAX_ACCELERATION}], got {self.acceleration}",
                    "invalid_value",
                )
            )


@dataclass(frozen=True)
class AcmeFloat8DataType(DataTypeEntity):
    """A third-party one-byte float."""

    identifier: ClassVar[str] = "acme.float8"
    scalar_storage: ClassVar[StorageClass] = "single_byte"


def _scope() -> Context:
    return CORE_AND_EXTENSIONS.extended_with(
        codecs={AcmeLz4Codec.identifier: AcmeLz4Codec},
        data_type={AcmeFloat8DataType.identifier: AcmeFloat8DataType},
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
        CORE.extended_with(codecs={"acme.lz-4": AcmeLz4Codec})


def test_error_an_entity_cannot_be_registered_at_the_wrong_point() -> None:
    # `EntityTables` says so to the type checker, which settles a scope
    # written out in source. A scope assembled at run time -- from an
    # entry point, from configuration -- had no type to check, and a
    # codec under `data_type` would resolve and then be asked for a
    # storage class it has no answer to.
    with pytest.raises(TypeError, match="registered at 'data_type', which takes DataTypeEntity"):
        CORE.extended_with(data_type={AcmeLz4Codec.identifier: AcmeLz4Codec})  # type: ignore[dict-item]


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
    assert isinstance(grid, ChunkGridEntity)
    parts = ArrayParts(grid.grid((64, 64)), data_type)
    assert parts.grid.rank == 2
    assert parts.grid.axis(0) == frozenset({32})


def test_error_an_optional_member_defaults_to_unset() -> None:
    # Otherwise every instance emits it, the bare-name spelling becomes
    # unreachable, and a canonicalized document gains a member the writer
    # never wrote.
    with pytest.raises(TypeError, match="a default other than UNSET"):

        @dataclass(frozen=True)
        class Inventive(CodecEntity):  # pyright: ignore[reportUnusedClass]
            # Optional by its type, so the annotation and the default agree
            # on that much; it is the default's value that is wrong.
            level: int | UNSET = 3  # pyright: ignore[reportAssignmentType]

            identifier: ClassVar[str] = "acme.inventive"
            kind: ClassVar[CodecKind] = "bytes_bytes"


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


def test_error_a_field_may_not_shadow_a_class_variable() -> None:
    # A field of that name goes into the configuration and into the JSON,
    # while the class variable it shadows is what the rest of the layer
    # reads -- so the entity would claim one thing and behave as another.
    with pytest.raises(TypeError, match="shadowing a class variable"):

        @dataclass(frozen=True)
        class Negotiable(CodecEntity):  # pyright: ignore[reportUnusedClass]
            must_understand: bool = True  # pyright: ignore[reportIncompatibleVariableOverride]

            identifier: ClassVar[str] = "acme.negotiable"
            kind: ClassVar[CodecKind] = "bytes_bytes"


def test_error_a_family_member_must_declare_what_the_family_left_open() -> None:
    # `bounds` is annotated on `IntegerDataType` and bound by none of it,
    # so every concrete integer type owes one. Nothing lists it: the
    # requirement is read off the annotation.
    with pytest.raises(TypeError, match="does not declare bounds"):

        @dataclass(frozen=True)
        class Int24DataType(IntegerDataType):  # pyright: ignore[reportUnusedClass]
            identifier: ClassVar[str] = "acme.int24"


# A third-party *family*: one class covering a parameterized set of names,
# the way `r<N>` covers every raw-byte width.
ACME_FIXED_PATTERN = re.compile(r"acme\.fixed(\d+)")


@dataclass(frozen=True)
class AcmeFixedDataType(DataTypeEntity):
    """`acme.fixedN`, a fixed-width type for every N."""

    data_type_name: Annotated[str, FROM_NAME]

    identifier: ClassVar[str] = "acme.fixed<N>"
    scalar_storage: ClassVar[StorageClass] = "multi_byte"

    @classmethod
    def accepts(cls, name: str) -> bool:
        return ACME_FIXED_PATTERN.fullmatch(name) is not None

    def to_json(self) -> ZarrV3MetadataFieldJSON:
        return cast("ZarrV3MetadataFieldJSON", self.data_type_name)


def test_a_third_party_can_register_a_family() -> None:
    # One class for an unbounded set of names. Nothing in the package
    # holds a table of spellings: the entity registers under an invented
    # identifier and `resolve` asks it, so a family is registered exactly
    # like a single name.
    scope = CORE_AND_EXTENSIONS.extended_with(
        data_type={AcmeFixedDataType.identifier: AcmeFixedDataType}
    )
    for name in ("acme.fixed8", "acme.fixed128"):
        assert scope.resolve("data_type", name) is AcmeFixedDataType
        entity, problems = scope.coerce("data_type", name)
        assert problems == ()
        assert isinstance(entity, AcmeFixedDataType)
        assert entity.to_json() == name
    # The invented identifier is not a name a document may write, and a
    # near-miss is still nobody's.
    assert scope.resolve("data_type", AcmeFixedDataType.identifier) is None
    assert scope.resolve("data_type", "acme.fixed") is None


def test_error_a_member_needs_a_check_from_somewhere() -> None:
    # An annotation outside the shapes `check_for` compiles implies no
    # check, so the entity owes one. Silently skipping the member would
    # let anything through where the field promised a type.
    with pytest.raises(TypeError, match="inner is annotated .*, which is not a shape JSON takes"):

        @dataclass(frozen=True)
        class Structured(CodecEntity):  # pyright: ignore[reportUnusedClass]
            inner: object

            identifier: ClassVar[str] = "acme.structured"
            kind: ClassVar[CodecKind] = "bytes_bytes"


@pytest.mark.parametrize(
    "name",
    ["member_types", "configuration_required", "nested_members"],
)
def test_error_a_derived_class_variable_may_not_be_declared(name: str) -> None:
    # Each is read off the fields at class creation, and a declaration
    # would be silently overwritten by that reading. Built with `type`,
    # since a class body cannot spell a name from a parameter.
    with pytest.raises(TypeError, match=f"declares {name}, which is derived from the fields"):
        type(
            "Opinionated",
            (CodecEntity,),
            {"identifier": "acme.opinionated", "kind": "bytes_bytes", name: {}},
        )


# A third-party codec that contains another codec: the case that used to
# need `prepare`, `configuration` and `canonical` written by hand.
@dataclass(frozen=True)
class AcmeWrapperCodec(CodecEntity):
    """A codec that applies another codec after its own step."""

    inner: CodecEntity | Opaque

    identifier: ClassVar[str] = "acme.wrapper"
    kind: ClassVar[CodecKind] = "bytes_bytes"


def test_a_third_party_entity_containing_entities_writes_nothing_for_it() -> None:
    # `inner: CodecEntity | Opaque` is the whole declaration. Reading it
    # in scope, writing it back, and canonicalizing through it all follow
    # from the annotation, so a wrapper is as short to write as a leaf.
    scope = CORE_AND_EXTENSIONS.extended_with(
        codecs={AcmeWrapperCodec.identifier: AcmeWrapperCodec}
    )
    entry = {
        "name": "acme.wrapper",
        "configuration": {"inner": {"name": "gzip", "configuration": {"level": 5}}},
    }
    codec, problems = scope.coerce("codecs", entry)
    assert problems == ()
    assert isinstance(codec, AcmeWrapperCodec)
    assert isinstance(codec.inner, GzipCodec)
    assert codec.inner.level == 5
    assert codec.to_json() == entry

    # An inner codec the scope does not model stays verbatim, as anywhere.
    unknown = {"name": "acme.wrapper", "configuration": {"inner": "acme.unknown"}}
    codec, problems = scope.coerce("codecs", unknown)
    assert problems == ()
    assert isinstance(codec, AcmeWrapperCodec)
    assert isinstance(codec.inner, Opaque)
    assert codec.to_json() == unknown

    # A problem inside is located inside.
    bad = {
        "name": "acme.wrapper",
        "configuration": {"inner": {"name": "gzip", "configuration": {"level": 99}}},
    }
    _, problems = scope.coerce("codecs", bad)
    assert [problem.loc for problem in problems] == [
        ("configuration", "inner", "configuration", "level")
    ]

    # Canonical form reaches the contained entity.
    verbose = {
        "name": "acme.wrapper",
        "configuration": {
            "inner": {
                "name": "blosc",
                "configuration": {
                    "cname": "zstd",
                    "clevel": 5,
                    "shuffle": "noshuffle",
                    "typesize": 4,
                    "blocksize": 0,
                },
            }
        },
    }
    codec, _ = scope.coerce("codecs", verbose)
    assert isinstance(codec, AcmeWrapperCodec)
    inner = codec.canonical().inner
    assert isinstance(inner, BloscCodec)
    assert inner.typesize is UNSET


def test_error_an_entity_may_not_override_canonical() -> None:
    # `canonical` is the walk into contained entities, read off the
    # annotations; an override could lose it. The entity's own rewrite
    # goes in `simplified`.
    with pytest.raises(TypeError, match="put the entity's own rewrite in `simplified`"):

        @dataclass(frozen=True)
        class Rewriter(CodecEntity):  # pyright: ignore[reportUnusedClass]
            identifier: ClassVar[str] = "acme.rewriter"
            kind: ClassVar[CodecKind] = "bytes_bytes"

            def canonical(self) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
                return self


def test_simplified_composes_with_the_walk_into_contained_entities() -> None:
    # An entity that contains an entity and rewrites its own members gets
    # both from `canonical` -- the contained blosc loses the `typesize`
    # that `noshuffle` ignores, and the frame of 0 that means "unframed"
    # is dropped -- with nothing to call `super()` for.
    @dataclass(frozen=True)
    class AcmeFramedCodec(CodecEntity):
        inner: CodecEntity | Opaque
        frame: int | UNSET = UNSET

        identifier: ClassVar[str] = "acme.framed"
        kind: ClassVar[CodecKind] = "bytes_bytes"

        def simplified(self) -> Self:
            return self if self.frame != 0 else replace(self, frame=UNSET)

    blosc = BloscCodec(cname="zstd", clevel=5, shuffle="noshuffle", typesize=4, blocksize=0)
    framed = AcmeFramedCodec(inner=blosc, frame=0)
    assert framed.canonical() == AcmeFramedCodec(inner=replace(blosc, typesize=UNSET))
    assert framed.inner is blosc  # a transformation, not a mutation


def test_error_a_nested_field_needs_an_entity_kind_with_a_point() -> None:
    # `MetadataEntity` is registered at no single point, so a field typed
    # as one could not be resolved through any scope.
    with pytest.raises(TypeError, match="has no `extension_point`"):

        @dataclass(frozen=True)
        class Vague(CodecEntity):  # pyright: ignore[reportUnusedClass]
            inner: MetadataEntity | Opaque

            identifier: ClassVar[str] = "acme.vague"
            kind: ClassVar[CodecKind] = "bytes_bytes"


# The JSON types third-party entities name, at module level so their
# annotations resolve.
class AcmeLvlConfiguration(TypedDict, closed=True):
    lvl: int


class AcmeLvlObject(TypedDict, closed=True):
    name: Literal["acme.lvl"]
    configuration: AcmeLvlConfiguration
    must_understand: NotRequired[bool]


class AcmeBlockConfiguration(TypedDict, closed=True):
    block: int


class AcmeBlockObject(TypedDict, closed=True):
    name: Literal["acme.block"]
    configuration: AcmeBlockConfiguration
    must_understand: NotRequired[bool]


# A third-party rule about a member, written in `__post_init__`.
@dataclass(frozen=True)
class AcmeBlockCodec(CodecEntity):
    """A codec whose block size must be a power of two."""

    block: int

    identifier: ClassVar[str] = "acme.block"
    kind: ClassVar[CodecKind] = "bytes_bytes"

    def __post_init__(self) -> None:
        if self.block < 1 or self.block & (self.block - 1) != 0:
            raise MetadataValidationError(
                problem(("block",), f"expected a power of two, got {self.block}", "invalid_value")
            )


def test_a_rule_about_a_member_is_post_init() -> None:
    # The rule runs on the typed members and reports relative to the
    # configuration; `coerce` catches what it raises and locates it in
    # the document, and the constructor raises it as it is.
    scope = CORE_AND_EXTENSIONS.extended_with(codecs={AcmeBlockCodec.identifier: AcmeBlockCodec})
    codec, problems = scope.coerce("codecs", {"name": "acme.block", "configuration": {"block": 64}})
    assert problems == ()
    assert isinstance(codec, AcmeBlockCodec)
    _, problems = scope.coerce("codecs", {"name": "acme.block", "configuration": {"block": 6}})
    assert [(p.loc, p.message) for p in problems] == [
        (("configuration", "block"), "expected a power of two, got 6")
    ]
    # And on the constructor, the same rule.
    with pytest.raises(MetadataValidationError) as caught:
        AcmeBlockCodec(block=6)
    assert [p.loc for p in caught.value.problems] == [("block",)]
    # A member that failed its type check never reaches the rule.
    _, problems = scope.coerce("codecs", {"name": "acme.block", "configuration": {"block": "x"}})
    assert [p.kind for p in problems] == ["invalid_type"]


def test_a_slotted_entity_is_compiled_once() -> None:
    # `@dataclass(slots=True)` builds the class twice; the second pass
    # arrives with the derived tables already on it and must not be
    # refused as having declared them.
    @dataclass(frozen=True, slots=True)
    class AcmeSlotted(CodecEntity):
        level: int

        identifier: ClassVar[str] = "acme.slotted"
        kind: ClassVar[CodecKind] = "bytes_bytes"

    assert list(AcmeSlotted.member_types) == ["level"]
    assert AcmeSlotted(level=1).to_json() == {
        "name": "acme.slotted",
        "configuration": {"level": 1},
    }


def test_a_bare_class_var_is_a_class_variable() -> None:
    @dataclass(frozen=True)
    class AcmeNoted(CodecEntity):
        identifier: ClassVar[str] = "acme.noted"
        kind: ClassVar[CodecKind] = "bytes_bytes"
        note: ClassVar = "not a member"

    assert AcmeNoted.member_types == {}


def test_a_number_member_is_a_float_field() -> None:
    # JSON has one number type; `float` admits an int spelled without a
    # point and refuses a bool, which is what a document's `2` and `true`
    # deserve.
    @dataclass(frozen=True)
    class AcmeScaled(CodecEntity):
        scale: float

        identifier: ClassVar[str] = "acme.scaled"
        kind: ClassVar[CodecKind] = "array_array"

        def transition(self, incoming: ArrayParts) -> ArrayParts | None:
            return incoming

    scope = CORE_AND_EXTENSIONS.extended_with(codecs={AcmeScaled.identifier: AcmeScaled})
    for written in (2, 2.5):
        codec, problems = scope.coerce(
            "codecs", {"name": "acme.scaled", "configuration": {"scale": written}}
        )
        assert problems == ()
        assert isinstance(codec, AcmeScaled)
    _, problems = scope.coerce("codecs", {"name": "acme.scaled", "configuration": {"scale": True}})
    assert [(p.loc, p.message) for p in problems] == [
        (("configuration", "scale"), "expected a number, got True")
    ]


def test_error_an_entity_must_be_a_dataclass() -> None:
    # Class creation runs before `@dataclass` and cannot see it missing;
    # registration can, and says so instead of the first `coerce` failing
    # with the base class's `__init__`.
    class Undecorated(CodecEntity):
        level: int

        identifier: ClassVar[str] = "acme.undecorated"
        kind: ClassVar[CodecKind] = "bytes_bytes"

    with pytest.raises(TypeError, match="not a dataclass; decorate it with @dataclass"):
        CORE_AND_EXTENSIONS.extended_with(codecs={Undecorated.identifier: Undecorated})


def test_error_a_nested_field_admits_opaque() -> None:
    # What the field holds when the inner name is out of scope.
    with pytest.raises(TypeError, match="inner holds an entity but does not admit Opaque"):

        @dataclass(frozen=True)
        class Closed(CodecEntity):  # pyright: ignore[reportUnusedClass]
            inner: CodecEntity

            identifier: ClassVar[str] = "acme.closed"
            kind: ClassVar[CodecKind] = "bytes_bytes"


def test_error_an_array_array_codec_defines_transition() -> None:
    # Left at the default, every rule after the codec would go silent.
    with pytest.raises(TypeError, match="array_array codec and does not define transition"):

        @dataclass(frozen=True)
        class Silent(CodecEntity):  # pyright: ignore[reportUnusedClass]
            identifier: ClassVar[str] = "acme.silent"
            kind: ClassVar[CodecKind] = "array_array"


def test_error_a_literal_class_variable_holds_a_listed_value() -> None:
    # `bytes` asks a data type's storage class and has nothing to say
    # about a fourth value: the endian rule would silently not apply.
    with pytest.raises(
        TypeError, match="sets scalar_storage = 'sixteen_bytes', which is not one of"
    ):

        @dataclass(frozen=True)
        class Wide(DataTypeEntity):  # pyright: ignore[reportUnusedClass]
            identifier: ClassVar[str] = "acme.wide"
            scalar_storage: ClassVar[StorageClass] = "sixteen_bytes"  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]


def test_error_a_list_of_problem_tuples_is_refused() -> None:
    # `problem()` returns a one-element tuple; a list of those would pass
    # the constructor and fail inside `coerce`, far from the mistake.
    with pytest.raises(TypeError, match="collect with `extend`, not `append`"):
        MetadataValidationError([problem(("a",), "bad a")])  # type: ignore[list-item]  # pyright: ignore[reportArgumentType]


def test_error_the_named_json_type_must_match_what_the_entity_writes() -> None:
    # A required member means the entity is always written as an object,
    # so naming a bare-name type for it is a promise `to_json` would break.
    with pytest.raises(
        TypeError, match="admits a bare name, which the entity never writes; lacks the object"
    ):

        @dataclass(frozen=True)
        class Misnamed(CodecEntity[Literal["acme.misnamed"]]):  # pyright: ignore[reportUnusedClass]
            level: int

            identifier: ClassVar[str] = "acme.misnamed"
            kind: ClassVar[CodecKind] = "bytes_bytes"


def test_error_the_named_json_type_must_name_what_the_entity_accepts() -> None:
    # `GzipCodecObject` spells `name: Literal["gzip"]`; an entity that
    # accepts only its own name cannot write that.
    with pytest.raises(TypeError, match="names 'gzip', which the entity does not accept"):

        @dataclass(frozen=True)
        class Impostor(CodecEntity[GzipCodecObject]):  # pyright: ignore[reportUnusedClass]
            level: int

            identifier: ClassVar[str] = "acme.impostor"
            kind: ClassVar[CodecKind] = "bytes_bytes"


def test_error_the_named_json_type_must_have_the_members_as_keys() -> None:
    with pytest.raises(
        TypeError, match=r"configuration keys \['lvl'\] where the members are \['level'\]"
    ):

        @dataclass(frozen=True)
        class Mismatched(CodecEntity[AcmeLvlObject]):  # pyright: ignore[reportUnusedClass]
            level: int

            identifier: ClassVar[str] = "acme.lvl"
            kind: ClassVar[CodecKind] = "bytes_bytes"


def test_a_third_party_entity_may_name_its_json_type_or_not() -> None:
    # Left defaulted, `to_json` is typed as any metadata field; named, as
    # the entity's own type -- and either way the same dict comes back.
    assert json_type_of(AcmeLz4Codec) is ZarrV3MetadataFieldJSON

    @dataclass(frozen=True)
    class AcmeTypedBlockCodec(CodecEntity[AcmeBlockObject]):
        block: int

        identifier: ClassVar[str] = "acme.block"
        kind: ClassVar[CodecKind] = "bytes_bytes"

    assert json_type_of(AcmeTypedBlockCodec) is AcmeBlockObject
    assert AcmeTypedBlockCodec(block=8).to_json() == {
        "name": "acme.block",
        "configuration": {"block": 8},
    }
