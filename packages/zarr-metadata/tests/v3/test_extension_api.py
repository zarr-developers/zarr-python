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
from zarr_metadata.v3.codec.blosc import BloscCodec
from zarr_metadata.v3.codec.gzip import GzipCodec
from zarr_metadata.v3.entity import (
    CORE_AND_EXTENSIONS,
    FROM_NAME,
    ArrayArrayCodec,
    ArrayDocumentV3,
    ArrayParts,
    BytesBytesCodec,
    ChunkGridEntity,
    ChunkKeyEncodingEntity,
    CodecEntity,
    CodecKind,
    Context,
    DataTypeEntity,
    IntegerDataType,
    JSONValue,
    Loc,
    MetadataEntity,
    Opaque,
    StorageClass,
    ValidationProblem,
    ZarrV3MetadataFieldJSON,
    problem,
    written,
)

ACME_MAX_ACCELERATION = 65537


@dataclass(frozen=True)
class AcmeLz4Codec(BytesBytesCodec):
    """A third-party compressor."""

    acceleration: int | UNSET = UNSET

    identifier: ClassVar[str] = "acme.lz4"
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

    def to_json(self) -> ZarrV3MetadataFieldJSON:
        if self.acceleration is UNSET:
            return "acme.lz4"
        return {"name": "acme.lz4", "configuration": {"acceleration": self.acceleration}}


@dataclass(frozen=True)
class AcmeFloat8DataType(DataTypeEntity):
    """A third-party one-byte float."""

    identifier: ClassVar[str] = "acme.float8"
    scalar_storage: ClassVar[StorageClass] = "single_byte"

    def fill_value_problems(self, value: object, loc: Loc = ()) -> tuple[ValidationProblem, ...]:
        return ()

    def to_json(self) -> ZarrV3MetadataFieldJSON:
        return "acme.float8"


def _scope() -> Context:
    return CORE_AND_EXTENSIONS.extended_with(AcmeLz4Codec, AcmeFloat8DataType)


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
    assert validate_array_metadata_v3(document) == ()


def test_a_registered_entity_is_judged() -> None:
    document = _document(
        codecs=(
            {"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "acme.lz4", "configuration": {"acceleration": 999999}},
        )
    )
    problems = validate_array_metadata_v3(document, context=SCOPE)
    assert [problem.loc for problem in problems] == [("codecs", 1, "configuration", "acceleration")]


def test_a_registered_entity_joins_the_pipeline_rules() -> None:
    # Declared `bytes_bytes`, so it may not precede the array->bytes codec,
    # and it is variable-size, so it may not encode a shard index.
    document = _document(
        codecs=("acme.lz4", {"name": "bytes", "configuration": {"endian": "little"}})
    )
    problems = validate_array_metadata_v3(document, context=SCOPE)
    assert [problem.loc for problem in problems] == [("codecs", 1)]


def test_a_registered_data_type_drives_the_codecs_around_it() -> None:
    # Single-byte, so the `bytes` codec needs no endianness for it.
    document = _document(data_type="acme.float8", fill_value=0, codecs=("bytes",))
    assert validate_array_metadata_v3(document, context=SCOPE) == ()


def test_a_registered_entity_canonicalizes_itself() -> None:
    document = _document(
        codecs=(
            {"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "acme.lz4", "configuration": {}},
        )
    )
    result = canonicalize_array_metadata_v3(document, context=SCOPE)
    assert result.valid is True
    assert result.document["codecs"][1] == "acme.lz4"


def test_error_an_entity_must_say_what_it_is() -> None:
    with pytest.raises(TypeError, match="does not declare identifier"):
        # Never bound: the guard raises while the class is being created,
        # which is the whole point -- so pyright cannot see it used.
        @dataclass(frozen=True)
        class Nameless(BytesBytesCodec):
            """A codec that forgot to say what it is."""


def test_the_entity_layer_answers_what_a_reader_needs() -> None:
    # The questions zarr-python asks before it can read a chunk.
    data_type, problems = CORE_AND_EXTENSIONS.coerce(DataTypeEntity, "int32")
    assert problems == ()
    assert isinstance(data_type, DataTypeEntity)
    assert data_type.storage_class() == "multi_byte"

    grid, problems = CORE_AND_EXTENSIONS.coerce(
        ChunkGridEntity, {"name": "regular", "configuration": {"chunk_shape": (32, 32)}}
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
        class Inventive(BytesBytesCodec):
            # Optional by its type, so the annotation and the default agree
            # on that much; it is the default's value that is wrong.
            level: int | UNSET = 3

            identifier: ClassVar[str] = "acme.inventive"


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
    assert isinstance(array.chunk_key_encoding, (ChunkKeyEncodingEntity, Opaque))
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
        class Negotiable(BytesBytesCodec):
            kind: str = "bytes_bytes"  # pyright: ignore[reportIncompatibleVariableOverride]

            identifier: ClassVar[str] = "acme.negotiable"


def test_error_a_family_member_must_declare_what_the_family_left_open() -> None:
    # `bounds` is annotated on `IntegerDataType` and bound by none of it,
    # so every concrete integer type owes one. Nothing lists it: the
    # requirement is read off the annotation.
    with pytest.raises(TypeError, match="does not declare bounds"):

        @dataclass(frozen=True)
        class Int24DataType(IntegerDataType):
            identifier: ClassVar[str] = "acme.int24"

            def to_json(self) -> ZarrV3MetadataFieldJSON:
                return "acme.int24"


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

    def fill_value_problems(self, value: object, loc: Loc = ()) -> tuple[ValidationProblem, ...]:
        return ()


def test_a_third_party_can_register_a_family() -> None:
    # One class for an unbounded set of names. Nothing in the package
    # holds a table of spellings: the entity registers under an invented
    # identifier and `resolve` asks it, so a family is registered exactly
    # like a single name.
    scope = CORE_AND_EXTENSIONS.extended_with(AcmeFixedDataType)
    for name in ("acme.fixed8", "acme.fixed128"):
        assert scope.resolve(DataTypeEntity, name) is AcmeFixedDataType
        entity, problems = scope.coerce(DataTypeEntity, name)
        assert problems == ()
        assert isinstance(entity, AcmeFixedDataType)
        assert entity.to_json() == name
    # The invented identifier is not a name a document may write, and a
    # near-miss is still nobody's.
    assert scope.resolve(DataTypeEntity, AcmeFixedDataType.identifier) is None
    assert scope.resolve(DataTypeEntity, "acme.fixed") is None


def test_error_a_member_needs_a_check_from_somewhere() -> None:
    # An annotation outside the shapes `check_for` compiles implies no
    # check, so the entity owes one. Silently skipping the member would
    # let anything through where the field promised a type.
    with pytest.raises(TypeError, match="inner is annotated .*, which is not a shape JSON takes"):

        @dataclass(frozen=True)
        class Structured(BytesBytesCodec):
            inner: object

            identifier: ClassVar[str] = "acme.structured"


# A third-party codec that contains another codec: the case that used to
# need `prepare`, `configuration` and `canonical` written by hand.
@dataclass(frozen=True)
class AcmeWrapperCodec(BytesBytesCodec):
    """A codec that applies another codec after its own step."""

    inner: CodecEntity | Opaque

    identifier: ClassVar[str] = "acme.wrapper"

    def to_json(self) -> ZarrV3MetadataFieldJSON:
        return {"name": "acme.wrapper", "configuration": {"inner": written(self.inner)}}


def test_a_third_party_entity_containing_entities_writes_nothing_for_it() -> None:
    # `inner: CodecEntity | Opaque` is the whole declaration. Reading it
    # in scope, writing it back, and canonicalizing through it all follow
    # from the annotation, so a wrapper is as short to write as a leaf.
    scope = CORE_AND_EXTENSIONS.extended_with(AcmeWrapperCodec)
    entry = {
        "name": "acme.wrapper",
        "configuration": {"inner": {"name": "gzip", "configuration": {"level": 5}}},
    }
    codec, problems = scope.coerce(CodecEntity, entry)
    assert problems == ()
    assert isinstance(codec, AcmeWrapperCodec)
    assert isinstance(codec.inner, GzipCodec)
    assert codec.inner.level == 5
    assert codec.to_json() == entry

    # An inner codec the scope does not model stays verbatim, as anywhere.
    unknown = {"name": "acme.wrapper", "configuration": {"inner": "acme.unknown"}}
    codec, problems = scope.coerce(CodecEntity, unknown)
    assert problems == ()
    assert isinstance(codec, AcmeWrapperCodec)
    assert isinstance(codec.inner, Opaque)
    assert codec.to_json() == unknown

    # A problem inside is located inside.
    bad = {
        "name": "acme.wrapper",
        "configuration": {"inner": {"name": "gzip", "configuration": {"level": 99}}},
    }
    _, problems = scope.coerce(CodecEntity, bad)
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
    codec, _ = scope.coerce(CodecEntity, verbose)
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
        class Rewriter(BytesBytesCodec):
            identifier: ClassVar[str] = "acme.rewriter"

            def canonical(self) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
                return self


def test_simplified_composes_with_the_walk_into_contained_entities() -> None:
    # An entity that contains an entity and rewrites its own members gets
    # both from `canonical` -- the contained blosc loses the `typesize`
    # that `noshuffle` ignores, and the frame of 0 that means "unframed"
    # is dropped -- with nothing to call `super()` for.
    @dataclass(frozen=True)
    class AcmeFramedCodec(BytesBytesCodec):
        inner: CodecEntity | Opaque
        frame: int | UNSET = UNSET

        identifier: ClassVar[str] = "acme.framed"

        def simplified(self) -> Self:
            return self if self.frame != 0 else replace(self, frame=UNSET)

        def to_json(self) -> ZarrV3MetadataFieldJSON:
            configuration: dict[str, JSONValue] = {"inner": written(self.inner)}
            if self.frame is not UNSET:
                configuration["frame"] = self.frame
            return {"name": "acme.framed", "configuration": configuration}

    blosc = BloscCodec(cname="zstd", clevel=5, shuffle="noshuffle", typesize=4, blocksize=0)
    framed = AcmeFramedCodec(inner=blosc, frame=0)
    assert framed.canonical() == AcmeFramedCodec(inner=replace(blosc, typesize=UNSET))
    assert framed.inner is blosc  # a transformation, not a mutation


def test_error_a_nested_field_needs_an_entity_kind_with_a_point() -> None:
    # `MetadataEntity` is registered at no single point, so a field typed
    # as one could not be resolved through any scope.
    with pytest.raises(TypeError, match="is of no kind; annotate it with a codec kind"):

        @dataclass(frozen=True)
        class Vague(BytesBytesCodec):
            inner: MetadataEntity | Opaque

            identifier: ClassVar[str] = "acme.vague"


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
class AcmeBlockCodec(BytesBytesCodec):
    """A codec whose block size must be a power of two."""

    block: int

    identifier: ClassVar[str] = "acme.block"

    def __post_init__(self) -> None:
        if self.block < 1 or self.block & (self.block - 1) != 0:
            raise MetadataValidationError(
                problem(("block",), f"expected a power of two, got {self.block}", "invalid_value")
            )

    def to_json(self) -> ZarrV3MetadataFieldJSON:
        return {"name": "acme.block", "configuration": {"block": self.block}}


def test_a_rule_about_a_member_is_post_init() -> None:
    # The rule runs on the typed members and reports relative to the
    # configuration; `coerce` catches what it raises and locates it in
    # the document, and the constructor raises it as it is.
    scope = CORE_AND_EXTENSIONS.extended_with(AcmeBlockCodec)
    codec, problems = scope.coerce(
        CodecEntity, {"name": "acme.block", "configuration": {"block": 64}}
    )
    assert problems == ()
    assert isinstance(codec, AcmeBlockCodec)
    _, problems = scope.coerce(CodecEntity, {"name": "acme.block", "configuration": {"block": 6}})
    assert [(p.loc, p.message) for p in problems] == [
        (("configuration", "block"), "expected a power of two, got 6")
    ]
    # And on the constructor, the same rule.
    with pytest.raises(MetadataValidationError) as caught:
        AcmeBlockCodec(block=6)
    assert [p.loc for p in caught.value.problems] == [("block",)]
    # A member that failed its type check never reaches the rule.
    _, problems = scope.coerce(CodecEntity, {"name": "acme.block", "configuration": {"block": "x"}})
    assert [p.kind for p in problems] == ["invalid_type"]


def test_a_slotted_entity_is_compiled_once() -> None:
    # `@dataclass(slots=True)` builds the class twice; the second pass
    # arrives with the derived tables already on it and must not be
    # refused as having declared them.
    @dataclass(frozen=True, slots=True)
    class AcmeSlotted(BytesBytesCodec):
        level: int

        identifier: ClassVar[str] = "acme.slotted"

        def to_json(self) -> ZarrV3MetadataFieldJSON:
            return {"name": "acme.slotted", "configuration": {"level": self.level}}

    assert AcmeSlotted(level=1).to_json() == {
        "name": "acme.slotted",
        "configuration": {"level": 1},
    }


def test_a_bare_class_var_is_a_class_variable() -> None:
    @dataclass(frozen=True)
    class AcmeNoted(BytesBytesCodec):
        identifier: ClassVar[str] = "acme.noted"
        note: ClassVar = "not a member"

        def to_json(self) -> ZarrV3MetadataFieldJSON:
            return "acme.noted"

    assert AcmeNoted().to_json() == "acme.noted"


def test_a_number_member_is_a_float_field() -> None:
    # JSON has one number type; `float` admits an int spelled without a
    # point and refuses a bool, which is what a document's `2` and `true`
    # deserve.
    @dataclass(frozen=True)
    class AcmeScaled(ArrayArrayCodec):
        scale: float

        identifier: ClassVar[str] = "acme.scaled"

        def transition(self, incoming: ArrayParts) -> ArrayParts | None:
            return incoming

        def to_json(self) -> ZarrV3MetadataFieldJSON:
            return {"name": "acme.scaled", "configuration": {"scale": self.scale}}

    scope = CORE_AND_EXTENSIONS.extended_with(AcmeScaled)
    for spelled in (2, 2.5):
        codec, problems = scope.coerce(
            CodecEntity, {"name": "acme.scaled", "configuration": {"scale": spelled}}
        )
        assert problems == ()
        assert isinstance(codec, AcmeScaled)
    _, problems = scope.coerce(
        CodecEntity, {"name": "acme.scaled", "configuration": {"scale": True}}
    )
    assert [(p.loc, p.message) for p in problems] == [
        (("configuration", "scale"), "expected a number, got True")
    ]


def test_error_an_entity_must_be_a_dataclass() -> None:
    # Class creation runs before `@dataclass` and cannot see it missing;
    # registration can, and says so instead of the first `coerce` failing
    # with the base class's `__init__`.
    class Undecorated(BytesBytesCodec):
        level: int

        identifier: ClassVar[str] = "acme.undecorated"

        def to_json(self) -> ZarrV3MetadataFieldJSON:
            return {"name": "acme.undecorated", "configuration": {"level": self.level}}

    with pytest.raises(TypeError, match="not a dataclass; decorate it with @dataclass"):
        CORE_AND_EXTENSIONS.extended_with(Undecorated)


def test_error_a_nested_field_admits_opaque() -> None:
    # What the field holds when the inner name is out of scope.
    with pytest.raises(TypeError, match="inner holds an entity but does not admit Opaque"):

        @dataclass(frozen=True)
        class Closed(BytesBytesCodec):
            inner: CodecEntity

            identifier: ClassVar[str] = "acme.closed"


def test_error_an_array_array_codec_defines_transition() -> None:
    # `transition` is abstract on the kind; a codec that leaves it so is
    # refused where it is first used, with what to write.
    @dataclass(frozen=True)
    class Silent(ArrayArrayCodec):
        identifier: ClassVar[str] = "acme.silent"

        def to_json(self) -> ZarrV3MetadataFieldJSON:
            return "acme.silent"

    with pytest.raises(
        TypeError, match="does not define transition, which its base leaves abstract"
    ):
        CORE_AND_EXTENSIONS.extended_with(Silent)


def test_error_a_codec_is_of_a_kind() -> None:
    with pytest.raises(
        TypeError, match="subclasses CodecEntity directly; subclass ArrayArrayCodec"
    ):

        @dataclass(frozen=True)
        class Kindless(CodecEntity):
            identifier: ClassVar[str] = "acme.kindless"
            kind: ClassVar[CodecKind] = "bytes_bytes"


def test_error_a_data_type_judges_its_fill_values() -> None:
    # Abstract, so that accepting every fill value is said, not defaulted.
    @dataclass(frozen=True)
    class Lax(DataTypeEntity):
        identifier: ClassVar[str] = "acme.lax"
        scalar_storage: ClassVar[StorageClass] = "single_byte"

        def to_json(self) -> ZarrV3MetadataFieldJSON:
            return "acme.lax"

    with pytest.raises(TypeError, match="does not define fill_value_problems"):
        CORE_AND_EXTENSIONS.extended_with(Lax)


def test_error_a_literal_class_variable_holds_a_listed_value() -> None:
    # `bytes` asks a data type's storage class and has nothing to say
    # about a fourth value: the endian rule would silently not apply.
    with pytest.raises(
        TypeError, match="sets scalar_storage = 'sixteen_bytes', which is not one of"
    ):

        @dataclass(frozen=True)
        class Wide(DataTypeEntity):
            identifier: ClassVar[str] = "acme.wide"
            scalar_storage: ClassVar[StorageClass] = "sixteen_bytes"  # pyright: ignore[reportAssignmentType]


def test_error_a_list_of_problem_tuples_is_refused() -> None:
    # `problem()` returns a one-element tuple; a list of those would pass
    # the constructor and fail inside `coerce`, far from the mistake.
    with pytest.raises(TypeError, match="collect with `extend`, not `append`"):
        MetadataValidationError([problem(("a",), "bad a")])  # pyright: ignore[reportArgumentType]
