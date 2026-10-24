"""Kinds of definition: a class declared with `kind=True`. The mechanism also supports kinds of other Zarr formats."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, TypeAlias, TypeVar, cast

import pytest
from annotated_types import Ge
from typing_extensions import TypedDict

from zarr_metadata.definition import (
    AcceptedField,
    Context,
    Definition,
    EmptyConfiguration,
    RefusedField,
    UnclaimedField,
    WithFillValue,
    canonical_fill_value,
    fill_value_problems,
    resolve,
)
from zarr_metadata.definition._definition import as_kind, kind_of
from zarr_metadata.definition._scope import kind_name
from zarr_metadata.json_schema import field_json_schema_v3
from zarr_metadata.problems import ValidationProblem
from zarr_metadata.v3.codec.gzip import GZIP_CODEC
from zarr_metadata.v3.definition import CodecDefinition, DataTypeDefinition

if TYPE_CHECKING:
    from zarr_metadata.typed_json import JSONValue

C = TypeVar("C")


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class MyCodec(CodecDefinition[Any]):
    """A codec definition with an extra member of its own. It is still a codec."""

    note: str = ""


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class Tag(Definition[C], kind=True, format=3):
    """A separate kind, filed apart from every v3 kind."""

    label: ClassVar[str] = "tag"


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class NoKind(Definition[Any]):
    """A definition subclass that declares no kind."""


def test_a_subclass_of_a_kind_is_a_definition_of_that_kind() -> None:
    """A definition built as a subclass of `CodecDefinition` is filed, read and named as a codec: the kind is the nearest class in the MRO declared with `kind=True`, not the class of the definition itself."""
    mine = MyCodec(name="mine", configuration=EmptyConfiguration, kind="bytes_bytes", size="static")
    assert kind_of(mine) is CodecDefinition
    context = Context.of(mine)
    assert context.claimant(CodecDefinition, "mine") is mine
    assert isinstance(resolve({"name": "mine"}, CodecDefinition, context)[0], AcceptedField)


def test_a_kind_of_its_own_is_filed_apart() -> None:
    """A class declared with `kind=True` is a kind. `as_kind` accepts it with or without type arguments. A scope files its definitions apart from every other kind's definitions. Scopes compare equal by the definitions they file. Messages name the kind by its label."""
    tag = Tag(name="tag1", configuration=EmptyConfiguration)
    context = Context.of(tag, GZIP_CODEC)
    assert as_kind(Tag) is Tag
    assert as_kind(Tag[Any]) is Tag
    assert context.claimant(Tag, "tag1") is tag
    assert context.claimant(CodecDefinition, "tag1") is None
    assert context == Context.of(GZIP_CODEC, tag)
    assert kind_name(Tag) == "tag"
    assert kind_name(CodecDefinition) == "codec"


def test_error_a_class_that_declares_no_kind_is_of_none() -> None:
    """A `Definition` subclass declared without `kind=True` has no kind: `kind_of` returns None, `Context.of` rejects a definition of that class, and `as_kind` rejects the class."""
    none = NoKind(name="nokind", configuration=EmptyConfiguration)
    assert kind_of(none) is None
    with pytest.raises(TypeError, match="a definition of no kind"):
        Context.of(none)
    with pytest.raises(TypeError, match="is not a kind of metadata"):
        as_kind(NoKind)


class Params(TypedDict, closed=True):
    level: Annotated[int, Ge(0)]


Loc = tuple[str | int, ...]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class Flat(Definition[C], kind=True, format=3):
    """A kind whose format writes the parameters next to the name: `{"id": name, **parameters}`."""

    label: ClassVar[str] = "flat"

    @classmethod
    def named_configuration(
        cls, value: object
    ) -> tuple[str | None, Mapping[str, object] | None, tuple[ValidationProblem, ...]]:
        if not isinstance(value, Mapping):
            return None, None, ()
        entry = cast("Mapping[str, object]", value)
        name = entry.get("id")
        if not isinstance(name, str):
            return None, None, ()
        return name, {key: item for key, item in entry.items() if key != "id"}, ()

    @classmethod
    def envelope_problems(cls, value: object) -> tuple[ValidationProblem, ...]:
        if cls.named_configuration(value)[0] is None:
            return (ValidationProblem((), "expected an object with a string 'id'", "invalid_type"),)
        return ()

    @classmethod
    def envelope_json(cls, name: str, configuration: Mapping[str, JSONValue]) -> JSONValue:
        return {"id": name, **configuration}

    @classmethod
    def configuration_loc(cls, loc: Loc) -> Loc:
        return loc

    @classmethod
    def name_loc(cls, loc: Loc) -> Loc:
        return (*loc, "id")


FLAT = Flat(name="flat", configuration=Params)
FLAT_SCOPE = Context.of(FLAT)


@pytest.mark.parametrize(
    ("field", "kind", "problems", "written"),
    [
        ({"id": "flat", "level": 1}, AcceptedField, [], {"id": "flat", "level": 1}),
        ({"id": "other", "x": 1}, UnclaimedField, [], {"id": "other", "x": 1}),
        ({"id": "flat", "level": -1}, RefusedField, [(("c", "level"), "invalid_value")], None),
        (
            {"id": "flat", "payload": object()},
            RefusedField,
            [(("c", "payload"), "invalid_type")],
            None,
        ),
        ("flat", RefusedField, [(("c",), "invalid_type")], None),
    ],
    ids=["read", "unclaimed", "out-of-range", "not-json", "not-an-object"],
)
def test_a_kind_reads_the_envelope_its_format_writes(
    field: object, kind: type, problems: list[tuple[Loc, str]], written: object
) -> None:
    """`resolve` reads a field in the format that the kind's classmethods describe. The kind splits the name from the parameters. Problems are located where the kind puts the configuration. An accepted or unclaimed field is written back in the kind's envelope."""
    resolved, found = resolve(field, Flat, FLAT_SCOPE, ("c",))
    assert type(resolved) is kind
    assert [(problem.loc, problem.kind) for problem in found] == problems
    if written is not None:
        assert not isinstance(resolved, RefusedField)
        assert resolved.to_json() == written


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class Typed(WithFillValue[C], kind=True, format=3):
    """A test kind, other than the data type kind, whose definitions take a fill value."""

    label: ClassVar[str] = "typed"


Count: TypeAlias = Annotated[int, Ge(0)] | None

TYPED = Typed(name="typed", configuration=EmptyConfiguration, fill_value=Count)


def test_a_fill_value_is_judged_by_any_kind_with_one() -> None:
    """`fill_value_problems` and `canonical_fill_value` check a fill value with the definition's `fill_value` members, whatever the definition's kind. This works because the members are defined on `WithFillValue`, which every data type kind derives from."""
    resolved, _ = resolve("typed", Typed, Context.of(TYPED))
    assert fill_value_problems(resolved, 3) == ()
    assert fill_value_problems(resolved, None) == ()
    assert [(p.loc, p.kind) for p in fill_value_problems(resolved, -1, ("fill_value",))] == [
        (("fill_value",), "invalid_value")
    ]
    assert canonical_fill_value(resolved, 3) == 3


def test_error_the_json_schema_writer_writes_v3_fields_only() -> None:
    """`field_json_schema_v3` writes only the v3 envelope. For a kind of another format, it raises a `TypeError` that names this limit instead of writing a wrong schema."""
    from zarr_metadata.v2.definition import CORE_V2, ZarrV2DataTypeDefinition

    with pytest.raises(TypeError, match="Zarr v3"):
        field_json_schema_v3(ZarrV2DataTypeDefinition, context=CORE_V2)
    with pytest.raises(TypeError, match="Zarr v3"):
        field_json_schema_v3(
            Tag, context=Context.of(Tag(name="tag1", configuration=EmptyConfiguration))
        )


def test_a_v3_value_that_is_no_field_is_shown_in_the_message() -> None:
    """A nested value that is not a metadata field at all is reported with the value in the message, as it was before the kind's envelope checked it."""
    _, problems = resolve(
        {"name": "sharding_indexed", "configuration": {"chunk_shape": [1], "codecs": [3]}},
        CodecDefinition,
        Context.of(*__import__("zarr_metadata.v3.definition", fromlist=["CORE"]).CORE.definitions),
    )
    assert any(p.message.endswith("got 3") for p in problems), [p.message for p in problems]


def test_error_a_kind_declares_its_format() -> None:
    """A kind declares which Zarr format its fields belong to, `format=2` or `format=3`, in the class header next to `kind=True`. Declaring a kind without a format, or with any other format, raises `TypeError` at class creation."""
    with pytest.raises(TypeError, match="format"):
        type("NoFormat", (Definition,), {}, kind=True)
    with pytest.raises(TypeError, match="format"):
        type("WrongFormat", (Definition,), {}, kind=True, format=4)
    with pytest.raises(TypeError, match="format"):
        type("FormatOnADefinition", (CodecDefinition,), {}, format=3)


TAG = Tag(name="tag", configuration=EmptyConfiguration)


def test_a_scope_has_the_format_of_the_kinds_it_files() -> None:
    """A scope's `format` is the format shared by every kind it files: 3 for `CORE`, 2 for `CORE_V2`, and None for a scope that files nothing, which can read in either format."""
    from zarr_metadata.v2.definition import CORE_V2
    from zarr_metadata.v3.definition import CORE

    assert CORE.format == 3
    assert CORE_V2.format == 2
    assert Context.of().format is None
    assert CORE.extended_with(TAG).format == 3


def test_error_a_scope_files_one_format() -> None:
    """Definitions of two formats cannot share a scope: `Context.of`, `extended_with` and `joined` each raise `TypeError` naming both formats."""
    from zarr_metadata.v2.data_type.scalar import UINT_V2
    from zarr_metadata.v2.definition import CORE_V2
    from zarr_metadata.v3.definition import CORE

    with pytest.raises(TypeError, match="format"):
        Context.of(TAG, UINT_V2)
    with pytest.raises(TypeError, match="format"):
        CORE.extended_with(UINT_V2)
    with pytest.raises(TypeError, match="format"):
        Context.joined(CORE, CORE_V2)


def test_error_a_field_is_read_in_a_scope_of_its_format() -> None:
    """`resolve` raises `TypeError` for a scope whose format differs from the kind's. It reads in a scope with no format, which claims nothing."""
    from zarr_metadata.v2.definition import CORE_V2, ZarrV2DataTypeDefinition
    from zarr_metadata.v3.definition import CORE

    with pytest.raises(TypeError, match="format"):
        resolve("<u1", ZarrV2DataTypeDefinition, CORE)
    with pytest.raises(TypeError, match="format"):
        resolve("int32", DataTypeDefinition, CORE_V2)
    field, _ = resolve("<u1", ZarrV2DataTypeDefinition, Context.of())
    assert isinstance(field, UnclaimedField)


def test_error_what_nothing_hashes_is_not_a_kind() -> None:
    """A kind must be a class. A list or a dict passed as a kind is rejected with the same error as any other non-kind, not with the `TypeError` that hashing it would raise."""
    with pytest.raises(TypeError, match="is not a kind of metadata"):
        as_kind(["codec"])
    with pytest.raises(TypeError, match="is not a kind of metadata"):
        as_kind({"kind": CodecDefinition})


class _LazyEnvelope(CodecDefinition[Any], kind=True, format=3):
    """A kind whose envelope rule yields its problems instead of returning a tuple of them."""

    label: ClassVar[str] = "lazy codec"

    @classmethod
    def envelope_problems(cls, value: object) -> Any:  # noqa: ANN401 - what a kind's author may write
        yield from CodecDefinition.envelope_problems(value)


def test_a_kind_may_yield_its_envelope_problems() -> None:
    """A kind's author writes `envelope_problems`. One that yields its problems works the same as one that returns them, both with and without a problem to yield."""
    lazy = _LazyEnvelope(
        name="gzip",
        configuration=GZIP_CODEC.configuration,
        kind=GZIP_CODEC.kind,
        size=GZIP_CODEC.size,
    )
    context = Context.of(lazy)
    read, problems = resolve(
        {"name": "gzip", "configuration": {"level": 1}}, _LazyEnvelope, context
    )
    assert isinstance(read, AcceptedField)
    assert problems == ()
    refused, problems = resolve(
        {"name": "gzip", "configuration": 5}, _LazyEnvelope, context, ("codecs", 0)
    )
    assert isinstance(refused, RefusedField)
    assert [(found.loc, found.kind) for found in problems] == [
        (("codecs", 0, "configuration"), "invalid_type")
    ]
