"""Kinds of definition: the class that declares `is_kind`, open to kinds of another format."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, TypeAlias, TypeVar, cast

import pytest
from annotated_types import Ge
from typing_extensions import TypedDict

from zarr_metadata._json import ValidationProblem
from zarr_metadata.v3._definition import (
    WithFillValue,
    as_kind,
    field_json_schema,
    kind_of,
)
from zarr_metadata.v3._scope import kind_name
from zarr_metadata.v3.codec.gzip import GZIP_CODEC
from zarr_metadata.v3.definition import (
    CodecDefinition,
    Context,
    Definition,
    EmptyConfiguration,
    Read,
    Refused,
    Unclaimed,
    canonical_fill_value,
    fill_value_problems,
    resolve,
)

if TYPE_CHECKING:
    from zarr_metadata._common import JSONValue

C = TypeVar("C")


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class MyCodec(CodecDefinition[Any]):
    """A codec definition with a member of its own: still a codec."""

    note: str = ""


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class Tag(Definition[C]):
    """A kind of its own, filed apart from every v3 kind."""

    is_kind: ClassVar[bool] = True
    label: ClassVar[str] = "tag"


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class NoKind(Definition[Any]):
    """A definition subclass that declares no kind."""


def test_a_subclass_of_a_kind_is_a_definition_of_that_kind() -> None:
    """A definition built as a subclass of `CodecDefinition` is filed, read and named as a codec: the kind is the nearest class in the MRO that declares `is_kind`, not the class of the definition."""
    mine = MyCodec(name="mine", configuration=EmptyConfiguration, kind="bytes_bytes", size="static")
    assert kind_of(mine) is CodecDefinition
    scope = Context.of(mine)
    assert scope.claimant(CodecDefinition, "mine") is mine
    assert isinstance(resolve({"name": "mine"}, CodecDefinition, scope)[0], Read)


def test_a_kind_of_its_own_is_filed_apart() -> None:
    """A class that sets `is_kind` in its body is a kind: `as_kind` accepts it with or without type arguments, a scope files its definitions apart from every other kind's, scopes compare by what each files, and messages name the kind by its label."""
    tag = Tag(name="tag1", configuration=EmptyConfiguration)
    scope = Context.of(tag, GZIP_CODEC)
    assert as_kind(Tag) is Tag
    assert as_kind(Tag[Any]) is Tag
    assert scope.claimant(Tag, "tag1") is tag
    assert scope.claimant(CodecDefinition, "tag1") is None
    assert scope == Context.of(GZIP_CODEC, tag)
    assert kind_name(Tag) == "tag"
    assert kind_name(CodecDefinition) == "codec"


def test_error_a_class_that_declares_no_kind_is_of_none() -> None:
    """A `Definition` subclass that does not set `is_kind` is of no kind: `kind_of` is None, `Context.of` refuses a definition of it, and `as_kind` refuses the class."""
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
class Flat(Definition[C]):
    """A kind whose format writes the parameters beside the name: `{"id": name, **parameters}`."""

    is_kind: ClassVar[bool] = True
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
        ({"id": "flat", "level": 1}, Read, [], {"id": "flat", "level": 1}),
        ({"id": "other", "x": 1}, Unclaimed, [], {"id": "other", "x": 1}),
        ({"id": "flat", "level": -1}, Refused, [(("c", "level"), "invalid_value")], None),
        ({"id": "flat", "payload": object()}, Refused, [(("c", "payload"), "invalid_type")], None),
        ("flat", Refused, [(("c",), "invalid_type")], None),
    ],
    ids=["read", "unclaimed", "out-of-range", "not-json", "not-an-object"],
)
def test_a_kind_reads_the_envelope_its_format_writes(
    field: object, kind: type, problems: list[tuple[Loc, str]], written: object
) -> None:
    """`resolve` reads a field as its kind's classmethods say the format writes one: the name and parameters are split as the kind splits them, problems sit where the kind puts the configuration, and a field read or unclaimed is written back in the kind's envelope."""
    resolved, found = resolve(field, Flat, FLAT_SCOPE, ("c",))
    assert type(resolved) is kind
    assert [(problem.loc, problem.kind) for problem in found] == problems
    if written is not None:
        assert not isinstance(resolved, Refused)
        assert resolved.to_json() == written


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class Typed(WithFillValue[C]):
    """A kind of another format whose definitions take a fill value."""

    is_kind: ClassVar[bool] = True
    label: ClassVar[str] = "typed"


Count: TypeAlias = Annotated[int, Ge(0)] | None

TYPED = Typed(name="typed", configuration=EmptyConfiguration, fill_value=Count)


def test_a_fill_value_is_judged_by_any_kind_with_one() -> None:
    """`fill_value_problems` and `canonical_fill_value` judge a fill value by the definition's `fill_value` members whatever kind it is, since the members live on `WithFillValue`, which every data type kind derives from."""
    resolved, _ = resolve("typed", Typed, Context.of(TYPED))
    assert fill_value_problems(resolved, 3) == ()
    assert fill_value_problems(resolved, None) == ()
    assert [(p.loc, p.kind) for p in fill_value_problems(resolved, -1, ("fill_value",))] == [
        (("fill_value",), "invalid_value")
    ]
    assert canonical_fill_value(resolved, 3) == 3


def test_error_the_json_schema_writer_writes_v3_fields_only() -> None:
    """`field_json_schema` writes the v3 envelope, so a kind of another format is refused with a `TypeError` naming the limit rather than a wrong schema."""
    from zarr_metadata.v2.definition import CORE_V2, ZarrV2DataTypeDefinition

    with pytest.raises(TypeError, match="Zarr v3"):
        field_json_schema(ZarrV2DataTypeDefinition, CORE_V2)
    with pytest.raises(TypeError, match="Zarr v3"):
        field_json_schema(Tag, Context.of(Tag(name="tag1", configuration=EmptyConfiguration)))


def test_a_v3_value_that_is_no_field_is_shown_in_the_message() -> None:
    """A nested value that is not a metadata field at all is reported with the value shown, as it was before the kind's envelope judged it."""
    _, problems = resolve(
        {"name": "sharding_indexed", "configuration": {"chunk_shape": [1], "codecs": [3]}},
        CodecDefinition,
        Context.of(
            *__import__("zarr_metadata.v3.definition", fromlist=["CORE"]).CORE.definitions()
        ),
    )
    assert any(p.message.endswith("got 3") for p in problems), [p.message for p in problems]
