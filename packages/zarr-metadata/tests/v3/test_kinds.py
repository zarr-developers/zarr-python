"""Kinds of definition: the class that declares `is_kind`, open to kinds of another format."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, TypeVar

import pytest

from zarr_metadata.v3._definition import as_kind, kind_of
from zarr_metadata.v3._scope import kind_name
from zarr_metadata.v3.codec.gzip import GZIP_CODEC
from zarr_metadata.v3.definition import (
    CodecDefinition,
    Context,
    Definition,
    EmptyConfiguration,
    Read,
    resolve,
)

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
