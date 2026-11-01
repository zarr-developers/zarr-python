"""Kinds of definition: the class a definition's kind is, and the scope of each format, which holds one slot per kind."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from zarr_metadata.definition import (
    AcceptedField,
    Definition,
    EmptyConfiguration,
    UnclaimedField,
    resolve,
)
from zarr_metadata.definition._definition import as_kind, kind_of
from zarr_metadata.definition._scope import kind_name
from zarr_metadata.json_schema import field_json_schema_v3
from zarr_metadata.v2.definition import CORE_V2_CONTEXT, ZarrV2Context, ZarrV2DataTypeDefinition
from zarr_metadata.v3.definition import (
    CORE_CONTEXT,
    CodecDefinition,
    DataTypeDefinition,
    ZarrV3Context,
)


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class MyCodec(CodecDefinition[Any]):
    """A codec definition with an extra member of its own. It is still a codec."""

    note: str = ""


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class NoKind(Definition[Any]):
    """A definition subclass that declares no kind."""


def test_a_subclass_of_a_kind_is_a_definition_of_that_kind() -> None:
    """A definition built as a subclass of `CodecDefinition` is filed, read and named as a codec: the kind is the nearest class in the MRO declared with `kind=True`, not the class of the definition itself."""
    mine = MyCodec(name="mine", configuration=EmptyConfiguration, kind="bytes_bytes", size="static")
    assert kind_of(mine) is CodecDefinition
    context = ZarrV3Context(codecs=[mine])
    assert context.claimant(CodecDefinition, "mine") is mine
    assert isinstance(resolve({"name": "mine"}, CodecDefinition, context)[0], AcceptedField)


def test_a_kind_is_named_by_its_class_and_its_label() -> None:
    """`as_kind` accepts a kind with or without type arguments, and messages name a kind by its label."""
    assert as_kind(CodecDefinition) is CodecDefinition
    assert as_kind(CodecDefinition[Any]) is CodecDefinition
    assert kind_name(CodecDefinition) == "codec"
    assert kind_name(ZarrV2DataTypeDefinition) == "v2 data type"


def test_error_a_class_that_declares_no_kind_is_of_none() -> None:
    """A `Definition` subclass declared without `kind=True` has no kind: `kind_of` returns None, no slot of a scope takes a definition of that class, and `as_kind` rejects the class."""
    none = NoKind(name="nokind", configuration=EmptyConfiguration)
    assert kind_of(none) is None
    with pytest.raises(TypeError, match="a definition of no kind"):
        ZarrV3Context(codecs=[none])  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="is not a kind of metadata"):
        as_kind(NoKind)


def test_error_the_json_schema_writer_writes_v3_fields_only() -> None:
    """`field_json_schema_v3` writes only the v3 envelope. For a kind of another format, it raises a `TypeError` that names this limit instead of writing a wrong schema."""
    with pytest.raises(TypeError, match="Zarr v3"):
        field_json_schema_v3(ZarrV2DataTypeDefinition, context=CORE_V2_CONTEXT)  # pyright: ignore[reportArgumentType]


def test_a_v3_value_that_is_no_field_is_shown_in_the_message() -> None:
    """A nested value that is not a metadata field at all is reported with the value in the message, as it was before the kind's envelope checked it."""
    _, problems = resolve(
        {"name": "sharding_indexed", "configuration": {"chunk_shape": [1], "codecs": [3]}},
        CodecDefinition,
        CORE_CONTEXT,
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


def test_a_scope_has_the_format_of_its_class() -> None:
    """A scope's `zarr_format` is the format of every kind its class holds: 3 for a `ZarrV3Context` and 2 for a `ZarrV2Context`, including one that files nothing."""
    assert CORE_CONTEXT.zarr_format == 3
    assert CORE_V2_CONTEXT.zarr_format == 2
    assert ZarrV3Context().zarr_format == 3
    assert ZarrV2Context().zarr_format == 2


def test_error_a_scope_files_one_format() -> None:
    """Definitions of two formats cannot share a scope: a slot refuses a definition of the other format's kind, in the constructor and in `update`, and `union` refuses scopes of two formats. Each raises `TypeError`."""
    from zarr_metadata.v2.data_type.scalar import UINT_DATA_TYPE_V2

    with pytest.raises(TypeError, match="v2 data type"):
        ZarrV3Context(data_types=[UINT_DATA_TYPE_V2])  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="v2 data type"):
        CORE_CONTEXT.update(data_types=[*CORE_CONTEXT.data_types.values(), UINT_DATA_TYPE_V2])  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="format"):
        CORE_CONTEXT.union(CORE_V2_CONTEXT)  # pyright: ignore[reportArgumentType]


def test_error_a_field_is_read_in_a_scope_of_its_format() -> None:
    """`resolve` raises `TypeError` for a scope whose format differs from the kind's. In an empty scope of its format, it reads a field as unclaimed."""
    with pytest.raises(TypeError, match="format"):
        resolve("<u1", ZarrV2DataTypeDefinition, CORE_CONTEXT)
    with pytest.raises(TypeError, match="format"):
        resolve("int32", DataTypeDefinition, CORE_V2_CONTEXT)
    field, _ = resolve("<u1", ZarrV2DataTypeDefinition, ZarrV2Context())
    assert isinstance(field, UnclaimedField)


def test_error_what_nothing_hashes_is_not_a_kind() -> None:
    """A kind must be a class. A list or a dict passed as a kind is rejected with the same error as any other non-kind, not with the `TypeError` that hashing it would raise."""
    with pytest.raises(TypeError, match="is not a kind of metadata"):
        as_kind(["codec"])
    with pytest.raises(TypeError, match="is not a kind of metadata"):
        as_kind({"kind": CodecDefinition})
