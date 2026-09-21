"""A third party registering its own entity, through public API only.

Every import here is from a module without a leading underscore. If this
file has to reach into a private one, the extension surface is not real.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal, NotRequired, Self, cast

import pytest
from typing_extensions import TypedDict

from zarr_metadata.model import UNSET, MetadataValidationError, ValidationProblem
from zarr_metadata.rules import (
    canonicalize_array_metadata_v3,
    validate_array_metadata_v3,
)
from zarr_metadata.v3._common import ZarrV3MetadataFieldJSON
from zarr_metadata.v3._compile import _CHECK_COMPILERS
from zarr_metadata.v3._entity import json_type_of
from zarr_metadata.v3.codec.blosc import BloscCodec
from zarr_metadata.v3.codec.gzip import GzipCodec
from zarr_metadata.v3.entity import (
    CORE,
    CORE_AND_EXTENSIONS,
    ArrayDocumentV3,
    ArrayParts,
    ChunkGridEntity,
    CodecEntity,
    CodecKind,
    Coerced,
    Context,
    DataTypeEntity,
    IntegerDataType,
    Interval,
    Loc,
    MemberTypes,
    MetadataEntity,
    Opaque,
    StorageClass,
    is_int,
    named_configuration,
    problem,
    register_check,
    validates,
)

ACME_MAX_ACCELERATION = 65537


@dataclass(frozen=True)
class AcmeLz4Codec(CodecEntity):
    """A third-party compressor."""

    acceleration: Annotated[int, Interval(ge=1, le=ACME_MAX_ACCELERATION)] | UNSET = UNSET

    identifier: ClassVar[str] = "acme.lz4"
    kind: ClassVar[CodecKind] = "bytes_bytes"
    variable_size: ClassVar[bool] = True


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


def test_error_value_rules_must_be_value_problems() -> None:
    # `problems` was the old name and takes an entity; an override using
    # it would never run, and nothing else would notice.
    with pytest.raises(TypeError, match="none of them takes an entity"):

        @dataclass(frozen=True)
        class Stale(CodecEntity):  # pyright: ignore[reportUnusedClass]
            identifier: ClassVar[str] = "acme.stale"
            kind: ClassVar[CodecKind] = "bytes_bytes"

            def problems(self) -> tuple[ValidationProblem, ...]:
                return ()


def test_error_an_entity_may_not_validate_in_post_init() -> None:
    # `coerce` builds through `unchecked`, which never reaches
    # `__post_init__`, so a rule there holds for a hand-built entity and
    # is silently absent for every entity read from a document.
    with pytest.raises(TypeError, match="`unchecked` does not reach"):

        @dataclass(frozen=True)
        class Eager(CodecEntity):  # pyright: ignore[reportUnusedClass]
            identifier: ClassVar[str] = "acme.eager"
            kind: ClassVar[CodecKind] = "bytes_bytes"

            def __post_init__(self) -> None:
                raise AssertionError


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

    data_type_name: str

    identifier: ClassVar[str] = "acme.fixed<N>"
    scalar_storage: ClassVar[StorageClass] = "multi_byte"

    @classmethod
    def accepts(cls, name: str) -> bool:
        return ACME_FIXED_PATTERN.fullmatch(name) is not None

    @classmethod
    def coerce(cls, value: object, context: object) -> Coerced[Self]:
        name, _, _ = named_configuration(value)
        if name is None or not cls.accepts(name):
            return None, problem((), "expected an 'acme.fixedN' data type")
        return cls.unchecked(data_type_name=name), ()

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


def test_error_requiredness_may_not_be_restated() -> None:
    # It is the field's to say. A declared entry exists for the check,
    # which the annotation does not imply; saying the member is required
    # as well is the drift the derivation removes.
    with pytest.raises(TypeError, match="requiredness its field does not give it"):

        @dataclass(frozen=True)
        class Insistent(CodecEntity):  # pyright: ignore[reportUnusedClass]
            acceleration: int | UNSET = UNSET

            identifier: ClassVar[str] = "acme.insistent"
            kind: ClassVar[CodecKind] = "bytes_bytes"
            member_types: ClassVar[MemberTypes] = {"acceleration": (True, is_int)}


def test_error_a_member_needs_a_check_from_somewhere() -> None:
    # An annotation outside the shapes `check_for` compiles implies no
    # check, so the entity owes one. Silently skipping the member would
    # let anything through where the field promised a type.
    with pytest.raises(TypeError, match="no check can be read off the annotation of inner"):

        @dataclass(frozen=True)
        class Structured(CodecEntity):  # pyright: ignore[reportUnusedClass]
            inner: object

            identifier: ClassVar[str] = "acme.structured"
            kind: ClassVar[CodecKind] = "bytes_bytes"


def test_error_a_bare_name_rule_may_not_be_restated() -> None:
    # Whether the bare spelling is legal follows from whether any member
    # is required, which the fields already say.
    with pytest.raises(TypeError, match="declares `configuration_required`"):

        @dataclass(frozen=True)
        class Opinionated(CodecEntity):  # pyright: ignore[reportUnusedClass]
            acceleration: int | UNSET = UNSET

            identifier: ClassVar[str] = "acme.opinionated"
            kind: ClassVar[CodecKind] = "bytes_bytes"
            configuration_required: ClassVar[bool] = True


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


def test_error_an_entity_may_not_define_prepare() -> None:
    # A member that is an entity is read from its annotation; an override
    # named `prepare` is resolution that would never run.
    with pytest.raises(TypeError, match="nothing calls `prepare`"):

        @dataclass(frozen=True)
        class Preparer(CodecEntity):  # pyright: ignore[reportUnusedClass]
            identifier: ClassVar[str] = "acme.preparer"
            kind: ClassVar[CodecKind] = "bytes_bytes"

            @classmethod
            def prepare(cls, members: object, context: object) -> object:
                return members


def test_error_a_nested_field_needs_an_entity_kind_with_a_point() -> None:
    # `MetadataEntity` is registered at no single point, so a field typed
    # as one could not be resolved through any scope.
    with pytest.raises(TypeError, match="has no `extension_point`"):

        @dataclass(frozen=True)
        class Vague(CodecEntity):  # pyright: ignore[reportUnusedClass]
            inner: MetadataEntity | Opaque

            identifier: ClassVar[str] = "acme.vague"
            kind: ClassVar[CodecKind] = "bytes_bytes"


# A third-party rule about one member, written as a `@validates` rule.
@dataclass(frozen=True)
class AcmeBlockCodec(CodecEntity):
    """A codec whose block size must be a power of two."""

    block: int

    identifier: ClassVar[str] = "acme.block"
    kind: ClassVar[CodecKind] = "bytes_bytes"

    @staticmethod
    @validates("block")
    def _block_is_a_power_of_two(block: int) -> tuple[ValidationProblem, ...]:
        if block < 1 or block & (block - 1) != 0:
            return problem((), f"expected a power of two, got {block}", "invalid_value")
        return ()


def test_a_rule_about_one_member_is_a_validates_rule() -> None:
    # The rule receives the typed member, only when present, and reports
    # relative to it: the location is supplied, and no `**members` is
    # unpacked by hand. The declared signature survives, so a call by
    # name is checked.
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


def test_error_a_validates_rule_must_name_a_field() -> None:
    with pytest.raises(TypeError, match="`@validates\\('blocc'\\)` names no field"):

        @dataclass(frozen=True)
        class Misspelt(CodecEntity):  # pyright: ignore[reportUnusedClass]
            block: int

            identifier: ClassVar[str] = "acme.misspelt"
            kind: ClassVar[CodecKind] = "bytes_bytes"

            @staticmethod
            @validates("blocc")
            def _rule(block: int) -> tuple[ValidationProblem, ...]:
                return ()


# An annotation shape the compiler does not read, taught to it from outside.
class Hex(str):
    """A hex digest: a `str` to the type checker, its own class at run time."""

    __slots__ = ()


def _is_hex(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
    if not isinstance(value, str) or any(c not in "0123456789abcdef" for c in value):
        return problem(loc, f"expected lowercase hex digits, got {value!r}")
    return ()


def test_a_third_party_can_teach_the_compiler_a_shape() -> None:
    # A `str` subclass is a real case: to the type checker `Hex` is a
    # `str`, but at run time it is a class `check_for` has no registration
    # for, so an entity using it is refused -- until one is registered,
    # through the same door the built-in shapes came through.
    with pytest.raises(TypeError, match="no check can be read off the annotation of digest"):

        @dataclass(frozen=True)
        class Unregistered(CodecEntity):  # pyright: ignore[reportUnusedClass]
            digest: Hex

            identifier: ClassVar[str] = "acme.unregistered"
            kind: ClassVar[CodecKind] = "bytes_bytes"

    def is_hex_annotation(annotation: object) -> bool:
        return annotation is Hex

    register_check(is_hex_annotation, lambda annotation: _is_hex)
    try:

        @dataclass(frozen=True)
        class AcmeDigestCodec(CodecEntity):
            digest: Hex

            identifier: ClassVar[str] = "acme.digest"
            kind: ClassVar[CodecKind] = "bytes_bytes"

        scope = CORE_AND_EXTENSIONS.extended_with(
            codecs={AcmeDigestCodec.identifier: AcmeDigestCodec}
        )
        codec, problems = scope.coerce(
            "codecs", {"name": "acme.digest", "configuration": {"digest": "c0ffee"}}
        )
        assert problems == ()
        assert isinstance(codec, AcmeDigestCodec)
        _, problems = scope.coerce(
            "codecs", {"name": "acme.digest", "configuration": {"digest": "C0FFEE"}}
        )
        assert [(p.loc, p.kind) for p in problems] == [
            (("configuration", "digest"), "invalid_type")
        ]
    finally:
        # A registration is process-wide; leave the compiler as it was found.
        _CHECK_COMPILERS[:] = [
            entry for entry in _CHECK_COMPILERS if entry.predicate is not is_hex_annotation
        ]


def test_error_the_named_json_type_must_match_what_the_entity_writes() -> None:
    # A required member means the entity is always written as an object,
    # so naming a bare-name type for it is a promise `to_json` would break.
    with pytest.raises(TypeError, match="lacks an object, but the entity never writes a bare name"):

        @dataclass(frozen=True)
        class Misnamed(CodecEntity[Literal["acme.misnamed"]]):  # pyright: ignore[reportUnusedClass]
            level: int

            identifier: ClassVar[str] = "acme.misnamed"
            kind: ClassVar[CodecKind] = "bytes_bytes"


def test_a_third_party_entity_may_name_its_json_type_or_not() -> None:
    # Left defaulted, `to_json` is typed as any metadata field; named, as
    # the entity's own type -- and either way the same dict comes back.
    assert json_type_of(AcmeLz4Codec) is ZarrV3MetadataFieldJSON

    class AcmeBlockConfiguration(TypedDict, closed=True):
        block: int

    class AcmeBlockObject(TypedDict, closed=True):
        name: Literal["acme.block"]
        configuration: AcmeBlockConfiguration
        must_understand: NotRequired[bool]

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
