"""The scope of each format: `ZarrV3Context` and `ZarrV2Context` are separate classes that implement the `Context` protocol, and behave alike."""

from __future__ import annotations

import copy
import dataclasses
import pickle
from collections.abc import Iterable
from typing import Any

import pytest
from typing_extensions import TypedDict

from zarr_metadata.definition import Context, ScopeConflictError, UnclaimedField, resolve
from zarr_metadata.definition._context import slot_kinds
from zarr_metadata.v2 import CORE_V2, GZIP_V2, ZarrV2Context
from zarr_metadata.v2.definition import ZarrV2CodecDefinition, ZarrV2ContextSlots
from zarr_metadata.v3 import CORE, GZIP_CODEC, ZarrV3Context
from zarr_metadata.v3._context import is_v3_kind
from zarr_metadata.v3.definition import CodecDefinition, ZarrV3ContextSlots


def _other(definition: Any) -> Any:  # noqa: ANN401 - a definition of either format
    """A second definition under the name of `definition`, with other rules."""
    return dataclasses.replace(definition, rules=lambda configuration, nested: iter(()))


FORMATS = [
    pytest.param(ZarrV3Context, CORE, CodecDefinition, GZIP_CODEC, 3, id="v3"),
    pytest.param(ZarrV2Context, CORE_V2, ZarrV2CodecDefinition, GZIP_V2, 2, id="v2"),
]


@pytest.mark.parametrize(("cls", "core", "kind", "gzip", "zarr_format"), FORMATS)
def test_the_scope_of_each_format_behaves_alike(
    cls: Any,  # noqa: ANN401 - either scope class
    core: Context,
    kind: Any,  # noqa: ANN401 - either codec kind
    gzip: Any,  # noqa: ANN401 - either gzip definition
    zarr_format: int,
) -> None:
    """Both scope classes implement `Context` in the same way: the format they read, `claimant`, `definitions_of`, a view per slot, `extended_with` replacing a definition by name, `joined`, equality and hashing by what each slot files, a short repr, and pickling and copying. A field is read in either one with `resolve`."""
    built = cls(codecs=[gzip])
    assert built.zarr_format == zarr_format
    assert built.claimant(kind, "gzip") is gzip
    assert dict(built.definitions_of(kind)) == {"gzip": gzip}
    assert dict(built.codecs) == {"gzip": gzip}
    assert built == cls(codecs=[gzip, gzip])
    assert hash(built) == hash(cls(codecs=[gzip]))
    assert built != cls()
    assert repr(built) == f"{cls.__name__}(<1 definitions>)"
    replaced = built.extended_with(codecs=[_other(gzip)])
    assert replaced.claimant(kind, "gzip") is not gzip
    assert cls.joined(built, cls()) == built
    assert cls.joined() == cls()
    assert pickle.loads(pickle.dumps(core)) == core
    assert copy.copy(core) is core
    assert copy.deepcopy(core) is core
    field, _ = resolve({"id": "acme"} if zarr_format == 2 else "acme", kind, cls())
    assert isinstance(field, UnclaimedField)


@pytest.mark.parametrize(("cls", "core", "kind", "gzip", "zarr_format"), FORMATS)
def test_error_joined_refuses_one_name_filed_two_ways(
    cls: Any,  # noqa: ANN401 - either scope class
    core: Context,
    kind: Any,  # noqa: ANN401 - either codec kind
    gzip: Any,  # noqa: ANN401 - either gzip definition
    zarr_format: int,
) -> None:
    """Scopes of either format that file different definitions under one name in a slot have no join: `joined` raises `ScopeConflictError` naming the kind and name."""
    with pytest.raises(ScopeConflictError) as raised:
        cls.joined(cls(codecs=[gzip]), cls(codecs=[_other(gzip)]))
    (conflict,) = raised.value.conflicts
    assert conflict.key == (kind, "gzip")


@pytest.mark.parametrize(("cls", "core", "kind", "gzip", "zarr_format"), FORMATS)
def test_error_a_slot_refuses_a_definition_of_the_other_format(
    cls: Any,  # noqa: ANN401 - either scope class
    core: Context,
    kind: Any,  # noqa: ANN401 - either codec kind
    gzip: Any,  # noqa: ANN401 - either gzip definition
    zarr_format: int,
) -> None:
    """A scope's `codecs` slot holds its own format's codecs only: the other format's gzip raises `TypeError`."""
    other = GZIP_V2 if zarr_format == 3 else GZIP_CODEC
    with pytest.raises(TypeError, match="codecs holds"):
        cls(codecs=[other])


@pytest.mark.parametrize(("cls", "core", "kind", "gzip", "zarr_format"), FORMATS)
def test_error_a_scope_refuses_a_kind_of_the_other_format(
    cls: Any,  # noqa: ANN401 - either scope class
    core: Any,  # noqa: ANN401 - either core scope
    kind: Any,  # noqa: ANN401 - either codec kind
    gzip: Any,  # noqa: ANN401 - either gzip definition
    zarr_format: int,
) -> None:
    """A scope holds the kinds of its own format only, so asking it for the definitions of the other format's codec kind, or for the one that claims a name, raises `TypeError` rather than answering that there are none."""
    other = ZarrV2CodecDefinition if zarr_format == 3 else CodecDefinition
    with pytest.raises(TypeError, match="kinds of its format"):
        core.definitions_of(other)
    with pytest.raises(TypeError, match="kinds of its format"):
        core.claimant(other, "gzip")


@pytest.mark.parametrize("cls", [ZarrV3Context, ZarrV2Context], ids=["v3", "v2"])
def test_each_declared_slot_has_a_view(cls: Any) -> None:  # noqa: ANN401 - either scope class
    """Each scope class reads its slots from its `...Slots` TypedDict, so the slots are declared once. Each slot has a view of the same name, which is the slot that holds its kind: a type checker cannot match a view's name to a declared slot, so this test does."""
    slots_type = ZarrV3ContextSlots if cls is ZarrV3Context else ZarrV2ContextSlots
    assert list(cls.kinds) == list(slots_type.__annotations__)
    scope: Any = CORE if cls is ZarrV3Context else CORE_V2
    for slot, kind in cls.kinds.items():
        assert getattr(scope, slot) == scope.definitions_of(kind)


def test_error_a_slot_is_declared_as_an_iterable_of_a_kind() -> None:
    """`slot_kinds` reads a slot's kind from its annotation, `Iterable[K]`. Any other annotation raises `TypeError` when the scope class is defined, so a wrong declaration cannot pass unnoticed."""

    class Wrong(TypedDict, total=False):
        codecs: list[CodecDefinition[Any]]

    with pytest.raises(TypeError, match=r"Wrong\.codecs: a slot is annotated Iterable\[K\]"):
        slot_kinds(Wrong, is_v3_kind)


def test_error_a_slot_holds_a_kind_of_its_format() -> None:
    """`slot_kinds` checks each slot's kind with the format's own check, so a v3 scope that declared a slot of v2 codecs raises `TypeError` when the scope class is defined."""

    class Mixed(TypedDict, total=False):
        codecs: Iterable[ZarrV2CodecDefinition[Any]]

    with pytest.raises(TypeError, match=r"Mixed\.codecs: ZarrV2CodecDefinition is not a kind"):
        slot_kinds(Mixed, is_v3_kind)
