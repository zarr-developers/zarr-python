"""A third party registering its own entity, through public API only.

Every import here is from a module without a leading underscore. If this
file has to reach into a private one, the extension surface is not real.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, NotRequired, Self

import pytest
from typing_extensions import TypedDict

from zarr_metadata.model import UNSET, MetadataValidationError
from zarr_metadata.rules import (
    canonicalize_array_metadata_v3,
    validate_array_metadata_v3,
)
from zarr_metadata.v3.codec.blosc import BloscCodec, BloscOptions
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
    Configuration,
    Context,
    DataTypeEntity,
    IntegerDataType,
    Loc,
    MetadataEntity,
    Opaque,
    StorageClass,
    ValidationProblem,
    problem,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

ACME_MAX_ACCELERATION = 65537


@dataclass(frozen=True)
class AcmeLz4Options(Configuration):
    acceleration: int | UNSET = UNSET

    def problems(self) -> Iterator[ValidationProblem]:
        if self.acceleration is not UNSET and not 1 <= self.acceleration <= ACME_MAX_ACCELERATION:
            yield ValidationProblem(
                ("acceleration",),
                f"expected an integer in [1, {ACME_MAX_ACCELERATION}], got {self.acceleration}",
                "invalid_value",
            )


@dataclass(frozen=True)
class AcmeLz4Codec(BytesBytesCodec):
    """A third-party compressor."""

    configuration: AcmeLz4Options

    identifier: ClassVar[str] = "acme.lz4"
    variable_size: ClassVar[bool] = True

    @property
    def acceleration(self) -> int | UNSET:
        return self.configuration.acceleration


@dataclass(frozen=True)
class AcmeFloat8DataType(DataTypeEntity):
    """A third-party one-byte float."""

    identifier: ClassVar[str] = "acme.float8"
    scalar_storage: ClassVar[StorageClass] = "single_byte"

    def fill_value_problems(self, value: object, loc: Loc = ()) -> tuple[ValidationProblem, ...]:
        return ()


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
    @dataclass(frozen=True)
    class Nameless(BytesBytesCodec):
        """A codec that forgot to say what it is."""

        variable_size: ClassVar[bool] = True

    with pytest.raises(TypeError, match="does not declare identifier"):
        CORE_AND_EXTENSIONS.extended_with(Nameless)


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


@dataclass(frozen=True)
class DefaultedOptions(Configuration):
    level: int | UNSET = 3


def test_an_absent_optional_member_is_read_as_unset_whatever_its_default() -> None:
    # A default is for hand construction; what a document left out is
    # `UNSET` in the record, so no field's default decides what a
    # document said.
    @dataclass(frozen=True)
    class Defaulted(BytesBytesCodec):
        configuration: DefaultedOptions

        identifier: ClassVar[str] = "acme.defaulted"

        variable_size: ClassVar[bool] = False

        @property
        def level(self) -> int | UNSET:
            return self.configuration.level

    assert Defaulted(DefaultedOptions()).level == 3
    codec, problems = CORE_AND_EXTENSIONS.extended_with(Defaulted).coerce(
        CodecEntity, "acme.defaulted"
    )
    assert problems == ()
    assert isinstance(codec, Defaulted)
    assert codec.level is UNSET


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


def test_error_a_family_member_must_declare_what_the_family_left_open() -> None:
    # `bounds` is annotated on `IntegerDataType` and bound by none of it,
    # so every concrete integer type owes one. Nothing lists it: the
    # requirement is read off the annotation.
    @dataclass(frozen=True)
    class Int24DataType(IntegerDataType):
        identifier: ClassVar[str] = "acme.int24"

    with pytest.raises(TypeError, match="does not declare bounds"):
        CORE_AND_EXTENSIONS.extended_with(Int24DataType)


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
    def name_problems(cls, name: str) -> Iterator[ValidationProblem]:
        match = ACME_FIXED_PATTERN.fullmatch(name)
        if match is not None and int(match.group(1)) % 8 != 0:
            yield ValidationProblem((), "expected a width that is a multiple of 8", "invalid_value")

    @classmethod
    def accepts(cls, name: str) -> bool:
        return ACME_FIXED_PATTERN.fullmatch(name) is not None

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
    # A rule about the name lands on the field: the document has no
    # configuration to locate it under.
    problems = validate_array_metadata_v3(
        _document(data_type="acme.fixed12", fill_value=0), context=scope
    )
    assert [(p.loc, p.kind) for p in problems] == [(("data_type",), "invalid_value")]


@dataclass(frozen=True)
class StructuredOptions(Configuration):
    inner: object


def test_error_a_member_needs_a_check_from_somewhere() -> None:
    # An annotation outside the shapes the parser reads implies no
    # parser, so the entity owes one. Silently skipping the member would
    # let anything through where the field promised a type.
    @dataclass(frozen=True)
    class Structured(BytesBytesCodec):
        configuration: StructuredOptions

        identifier: ClassVar[str] = "acme.structured"
        variable_size: ClassVar[bool] = False

        @property
        def inner(self) -> object:
            return self.configuration.inner

    with pytest.raises(TypeError, match="inner is annotated .*, which is not a shape JSON takes"):
        CORE_AND_EXTENSIONS.extended_with(Structured)


# A third-party codec that contains another codec.
@dataclass(frozen=True)
class AcmeWrapperOptions(Configuration):
    inner: CodecEntity | Opaque


@dataclass(frozen=True)
class AcmeWrapperCodec(BytesBytesCodec):
    """A codec that applies another codec after its own step."""

    configuration: AcmeWrapperOptions

    identifier: ClassVar[str] = "acme.wrapper"

    variable_size: ClassVar[bool] = False

    @property
    def inner(self) -> CodecEntity | Opaque:
        return self.configuration.inner

    def canonical(self) -> Self:
        return self.with_configuration(inner=self.inner.canonical())


def test_a_third_party_entity_containing_entities_reads_them_in_scope() -> None:
    # `inner: CodecEntity | Opaque` is the whole declaration of the
    # reading: the inner codec is resolved in the scope the wrapper is
    # read in, and its problems are located inside. Writing and
    # canonicalizing it are the wrapper's own two lines.
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


@dataclass(frozen=True)
class AcmeFramedOptions(Configuration):
    inner: CodecEntity | Opaque
    frame: int | UNSET = UNSET


def test_canonical_is_the_entity_s_own_and_reaches_what_it_contains() -> None:
    # An entity that contains an entity and rewrites its own members does
    # both in one `canonical` -- the contained blosc loses the `typesize`
    # that `noshuffle` ignores, and the frame of 0 that means "unframed"
    # is dropped -- with nothing to call `super()` for.
    @dataclass(frozen=True)
    class AcmeFramedCodec(BytesBytesCodec):
        configuration: AcmeFramedOptions

        identifier: ClassVar[str] = "acme.framed"

        variable_size: ClassVar[bool] = False

        @property
        def inner(self) -> CodecEntity | Opaque:
            return self.configuration.inner

        @property
        def frame(self) -> int | UNSET:
            return self.configuration.frame

        def canonical(self) -> Self:
            return self.with_configuration(
                inner=self.inner.canonical(),
                frame=UNSET if self.frame == 0 else self.frame,
            )

    blosc = BloscCodec(
        BloscOptions(cname="zstd", clevel=5, shuffle="noshuffle", typesize=4, blocksize=0)
    )
    framed = AcmeFramedCodec(AcmeFramedOptions(inner=blosc, frame=0))
    assert framed.canonical() == AcmeFramedCodec(
        AcmeFramedOptions(inner=blosc.with_configuration(typesize=UNSET))
    )
    assert framed.inner is blosc  # a transformation, not a mutation


@dataclass(frozen=True)
class VagueOptions(Configuration):
    inner: MetadataEntity | Opaque


def test_error_a_nested_field_names_a_kind() -> None:
    # `MetadataEntity` is of no kind, so a field typed as one could not
    # be resolved through any scope.
    @dataclass(frozen=True)
    class Vague(BytesBytesCodec):
        configuration: VagueOptions

        identifier: ClassVar[str] = "acme.vague"
        variable_size: ClassVar[bool] = False

        @property
        def inner(self) -> MetadataEntity | Opaque:
            return self.configuration.inner

    with pytest.raises(TypeError, match="inner holds an entity but is not written as its kind"):
        CORE_AND_EXTENSIONS.extended_with(Vague)


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


# A third-party rule about a member: the record's own.


@dataclass(frozen=True)
class AcmeBlockOptions(Configuration):
    block: int

    def problems(self) -> Iterator[ValidationProblem]:
        if self.block < 1 or self.block & (self.block - 1) != 0:
            yield ValidationProblem(
                ("block",), f"expected a power of two, got {self.block}", "invalid_value"
            )


@dataclass(frozen=True)
class AcmeBlockCodec(BytesBytesCodec):
    """A codec whose block size must be a power of two."""

    configuration: AcmeBlockOptions

    identifier: ClassVar[str] = "acme.block"

    variable_size: ClassVar[bool] = False

    @property
    def block(self) -> int:
        return self.configuration.block


def test_a_rule_about_a_member_is_the_record_s_own() -> None:
    # The rule runs on the typed record and reports relative to the
    # configuration; `coerce` runs it to the end and locates what it
    # yields in the document; the constructor stops at the first.
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
        AcmeBlockCodec(AcmeBlockOptions(block=6))
    assert [p.loc for p in caught.value.problems] == [("block",)]
    # A member that failed its type check never reaches the rule.
    _, problems = scope.coerce(CodecEntity, {"name": "acme.block", "configuration": {"block": "x"}})
    assert [p.kind for p in problems] == ["invalid_type"]


@dataclass(frozen=True)
class AcmeRangeOptions(Configuration):
    low: int
    high: int

    def problems(self) -> Iterator[ValidationProblem]:
        if self.low < 0:
            yield ValidationProblem(
                ("low",), f"expected an integer >= 0, got {self.low}", "invalid_value"
            )
        if self.high < self.low:
            yield ValidationProblem(
                ("high",), f"expected an integer >= low, got {self.high}", "invalid_value"
            )


@dataclass(frozen=True)
class AcmeRangeCodec(BytesBytesCodec):
    """A codec with two rules, so that one can fail after another."""

    configuration: AcmeRangeOptions

    identifier: ClassVar[str] = "acme.range"

    variable_size: ClassVar[bool] = False

    @property
    def low(self) -> int:
        return self.configuration.low

    @property
    def high(self) -> int:
        return self.configuration.high


def test_the_constructor_stops_at_the_first_problem_and_coerce_reports_every_one() -> None:
    # One method, three consumers: the entity's constructor takes the
    # first problem it yields, `coerce` runs it to the end, and a reader
    # with a record asks it directly, and stops or collects.
    with pytest.raises(MetadataValidationError) as caught:
        AcmeRangeCodec(AcmeRangeOptions(low=-1, high=-2))
    assert [p.loc for p in caught.value.problems] == [("low",)]
    scope = CORE_AND_EXTENSIONS.extended_with(AcmeRangeCodec)
    _, problems = scope.coerce(
        CodecEntity, {"name": "acme.range", "configuration": {"low": -1, "high": -2}}
    )
    assert [p.loc for p in problems] == [("configuration", "low"), ("configuration", "high")]
    assert list(AcmeRangeOptions(low=0, high=1).problems()) == []
    # A reader that wants every problem of a record asks the record.
    assert [p.loc for p in AcmeRangeOptions(low=-1, high=-2).problems()] == [("low",), ("high",)]


def test_error_an_entity_may_not_define_post_init() -> None:
    # `coerce` never runs it, so a rule written there would judge a
    # hand-built entity and no document; the rules go on the record.
    @dataclass(frozen=True)
    class Checked(BytesBytesCodec):
        identifier: ClassVar[str] = "acme.checked"
        variable_size: ClassVar[bool] = False

        def __post_init__(self) -> None:
            return None

    with pytest.raises(TypeError, match="defines __post_init__; write its rules as `problems`"):
        CORE_AND_EXTENSIONS.extended_with(Checked)


@dataclass(frozen=True)
class LocalizedOptions(Configuration):
    # `Local` is defined inside the test, so it is not here, where this
    # class's annotations resolve: the case the message is for.
    inner: Local  # noqa: F821  # pyright: ignore[reportUndefinedVariable]


def test_error_a_field_annotation_names_what_is_not_defined() -> None:
    # Annotations are resolved where the class is, at registration; a
    # type defined inside a function is not there.
    class Local(TypedDict, closed=True):
        depth: int

    @dataclass(frozen=True)
    class Localized(BytesBytesCodec):
        configuration: LocalizedOptions

        identifier: ClassVar[str] = "acme.localized"
        variable_size: ClassVar[bool] = False

        @property
        def inner(self) -> Local:
            return self.configuration.inner

    with pytest.raises(TypeError, match="a field annotation names 'Local', which is not defined"):
        CORE_AND_EXTENSIONS.extended_with(Localized)


def test_error_a_codec_says_whether_its_output_size_is_fixed() -> None:
    # A default in either direction is a verdict: a compressor that
    # said nothing would be accepted as a shard-index codec.
    @dataclass(frozen=True)
    class Sizeless(BytesBytesCodec):
        identifier: ClassVar[str] = "acme.sizeless"

    with pytest.raises(TypeError, match="does not declare variable_size"):
        CORE_AND_EXTENSIONS.extended_with(Sizeless)


def test_a_reader_gets_structural_and_semantic_reasons_together() -> None:
    with pytest.raises(MetadataValidationError) as caught:
        ArrayDocumentV3.from_json(_document(attributes=5, fill_value=-1))
    assert {problem.loc for problem in caught.value.problems} == {("attributes",), ("fill_value",)}


def test_a_malformed_envelope_is_one_problem() -> None:
    # The envelope is judged once, by the scope; the entity is not asked
    # to read what is not a metadata field.
    _, problems = CORE_AND_EXTENSIONS.coerce(CodecEntity, 5, ("codecs", 0))
    assert [(p.loc, p.kind) for p in problems] == [(("codecs", 0), "invalid_type")]
    _, problems = CORE_AND_EXTENSIONS.coerce(
        CodecEntity, {"name": "gzip", "configuration": 42}, ("codecs", 0)
    )
    assert [(p.loc, p.kind) for p in problems] == [(("codecs", 0, "configuration"), "invalid_type")]


@dataclass(frozen=True)
class AcmeSlottedOptions(Configuration):
    level: int


def test_a_slotted_entity_is_accepted() -> None:
    # `@dataclass(slots=True)` builds the class twice; registration sees
    # the second, whose members are slot descriptors.
    @dataclass(frozen=True, slots=True)
    class AcmeSlotted(BytesBytesCodec):
        configuration: AcmeSlottedOptions

        identifier: ClassVar[str] = "acme.slotted"

        variable_size: ClassVar[bool] = False

        @property
        def level(self) -> int:
            return self.configuration.level

    assert AcmeSlotted(AcmeSlottedOptions(level=1)).to_json() == {
        "name": "acme.slotted",
        "configuration": {"level": 1},
    }


def test_a_bare_class_var_is_a_class_variable() -> None:
    @dataclass(frozen=True)
    class AcmeNoted(BytesBytesCodec):
        identifier: ClassVar[str] = "acme.noted"
        variable_size: ClassVar[bool] = False
        note: ClassVar = "not a member"

    assert AcmeNoted().to_json() == "acme.noted"


@dataclass(frozen=True)
class AcmeScaledOptions(Configuration):
    scale: float


def test_a_number_member_is_a_float_field() -> None:
    # JSON has one number type; `float` admits an int spelled without a
    # point and refuses a bool, which is what a document's `2` and `true`
    # deserve.
    @dataclass(frozen=True)
    class AcmeScaled(ArrayArrayCodec):
        configuration: AcmeScaledOptions

        identifier: ClassVar[str] = "acme.scaled"

        variable_size: ClassVar[bool] = False

        @property
        def scale(self) -> float:
            return self.configuration.scale

        def transition(self, incoming: ArrayParts) -> ArrayParts | None:
            return incoming

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


@dataclass(frozen=True)
class UndecoratedOptions(Configuration):
    level: int


def test_error_an_entity_must_be_a_dataclass() -> None:
    # Class creation runs before `@dataclass` and cannot see it missing;
    # registration can, and says so instead of the first `coerce` failing
    # with the base class's `__init__`.
    class Undecorated(BytesBytesCodec):
        configuration: UndecoratedOptions

        identifier: ClassVar[str] = "acme.undecorated"

        @property
        def level(self) -> int:
            return self.configuration.level

    with pytest.raises(TypeError, match="not a dataclass; decorate it with @dataclass"):
        CORE_AND_EXTENSIONS.extended_with(Undecorated)


@dataclass(frozen=True)
class ClosedOptions(Configuration):
    inner: CodecEntity


def test_error_a_nested_field_admits_opaque() -> None:
    # What the field holds when the inner name is out of scope.
    @dataclass(frozen=True)
    class Closed(BytesBytesCodec):
        configuration: ClosedOptions

        identifier: ClassVar[str] = "acme.closed"
        variable_size: ClassVar[bool] = False

        @property
        def inner(self) -> CodecEntity:
            return self.configuration.inner

    with pytest.raises(TypeError, match="inner holds an entity but is not written as its kind"):
        CORE_AND_EXTENSIONS.extended_with(Closed)


def test_error_an_array_array_codec_defines_transition() -> None:
    # `transition` is abstract on the kind; a codec that leaves it so is
    # refused where it is first used, with what to write.
    @dataclass(frozen=True)
    class Silent(ArrayArrayCodec):
        identifier: ClassVar[str] = "acme.silent"
        variable_size: ClassVar[bool] = False

    with pytest.raises(
        TypeError, match="does not define transition, which its base leaves abstract"
    ):
        CORE_AND_EXTENSIONS.extended_with(Silent)


def test_error_a_codec_is_of_a_kind() -> None:
    # The kind classes say what a codec does to the array; registration
    # refuses one that skipped them.
    @dataclass(frozen=True)
    class Kindless(CodecEntity):
        identifier: ClassVar[str] = "acme.kindless"

    with pytest.raises(
        TypeError, match="subclasses CodecEntity directly; subclass ArrayArrayCodec"
    ):
        CORE_AND_EXTENSIONS.extended_with(Kindless)


def test_error_a_data_type_judges_its_fill_values() -> None:
    # Abstract, so that accepting every fill value is said, not defaulted.
    @dataclass(frozen=True)
    class Lax(DataTypeEntity):
        identifier: ClassVar[str] = "acme.lax"
        scalar_storage: ClassVar[StorageClass] = "single_byte"

    with pytest.raises(TypeError, match="does not define fill_value_problems"):
        CORE_AND_EXTENSIONS.extended_with(Lax)


def test_error_a_list_of_problem_tuples_is_refused() -> None:
    # `problem()` returns a one-element tuple; a list of those would pass
    # the constructor and fail inside `coerce`, far from the mistake.
    with pytest.raises(TypeError, match="collect with `extend`, not `append`"):
        MetadataValidationError([problem(("a",), "bad a")])  # pyright: ignore[reportArgumentType]
