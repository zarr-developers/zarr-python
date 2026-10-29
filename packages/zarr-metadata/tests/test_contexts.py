"""The scope of each format: `ZarrV3Context` and `ZarrV2Context` are separate classes that implement the `Context` protocol, and behave alike."""

from __future__ import annotations

import copy
import dataclasses
import pickle
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import pytest
from typing_extensions import TypedDict

from zarr_metadata.definition import Context, ScopeConflictError, UnclaimedField, resolve
from zarr_metadata.definition._context import slot_kinds
from zarr_metadata.model import (
    ZarrV2ArrayMetadata,
    ZarrV3ArrayMetadata,
    read_array_metadata_v2,
    read_array_metadata_v3,
)
from zarr_metadata.v2 import CORE_V2, GZIP_CODEC_V2, ZarrV2Context
from zarr_metadata.v2._context import _KINDS as V2_KINDS
from zarr_metadata.v2.definition import (
    ZarrV2CodecDefinition,
    ZarrV2ContextSlots,
    ZarrV2DataTypeDefinition,
)
from zarr_metadata.v3 import CORE_CONTEXT, GZIP_CODEC, ZarrV3Context
from zarr_metadata.v3._context import _KINDS as V3_KINDS
from zarr_metadata.v3._context import is_v3_kind
from zarr_metadata.v3.definition import CodecDefinition, DataTypeDefinition, ZarrV3ContextSlots


def _other(definition: Any) -> Any:  # noqa: ANN401 - a definition of either format
    """A second definition under the name of `definition`, with other rules."""
    return dataclasses.replace(definition, rules=lambda configuration, nested: iter(()))


ARRAY = ZarrV3ArrayMetadata.create_default().to_json()
V2_ARRAY = ZarrV2ArrayMetadata.create_default().to_json()

FORMATS = [
    pytest.param(ZarrV3Context, CORE_CONTEXT, CodecDefinition, GZIP_CODEC, 3, id="v3"),
    pytest.param(ZarrV2Context, CORE_V2, ZarrV2CodecDefinition, GZIP_CODEC_V2, 2, id="v2"),
]


@pytest.mark.parametrize(("cls", "core", "kind", "gzip", "zarr_format"), FORMATS)
def test_the_scope_of_each_format_behaves_alike(
    cls: Any,  # noqa: ANN401 - either scope class
    core: Context,
    kind: Any,  # noqa: ANN401 - either codec kind
    gzip: Any,  # noqa: ANN401 - either gzip definition
    zarr_format: int,
) -> None:
    """Both scope classes implement `Context` in the same way: the format they read, `claimant`, `definitions_of`, a view per slot, `update` replacing a slot, `union`, equality and hashing by what each slot files, a short repr, and pickling and copying. A field is read in either one with `resolve`."""
    built = cls(codecs=[gzip])
    assert built.zarr_format == zarr_format
    assert built.claimant(kind, "gzip") is gzip
    assert dict(built.definitions_of(kind)) == {"gzip": gzip}
    assert dict(built.codecs) == {"gzip": gzip}
    assert built == cls(codecs=[gzip, gzip])
    assert hash(built) == hash(cls(codecs=[gzip]))
    assert built != cls()
    assert repr(built) == f"{cls.__name__}(<1 definitions>)"
    replaced = built.update(codecs=[_other(gzip)])
    assert replaced.claimant(kind, "gzip") is not gzip
    assert built.union(cls()) == built
    assert built.union() == built
    assert pickle.loads(pickle.dumps(core)) == core
    assert copy.copy(core) is core
    assert copy.deepcopy(core) is core
    field, _ = resolve({"id": "acme"} if zarr_format == 2 else "acme", kind, cls())
    assert isinstance(field, UnclaimedField)


@pytest.mark.parametrize(("cls", "core", "kind", "gzip", "zarr_format"), FORMATS)
def test_update_replaces_each_slot_it_is_given_and_keeps_the_rest(
    cls: Any,  # noqa: ANN401 - either scope class
    core: Any,  # noqa: ANN401 - either core scope
    kind: Any,  # noqa: ANN401 - either codec kind
    gzip: Any,  # noqa: ANN401 - either gzip definition
    zarr_format: int,
) -> None:
    """`update` returns the scope with each slot it is given replaced whole, and every other slot kept. A slot is given as definitions, or as a mapping from each definition's name to it, so a slot of a scope passes back in as it is, or changed by name. The scope it is called on is unchanged."""
    only = core.update(codecs=[gzip])
    assert dict(only.codecs) == {"gzip": gzip}
    assert only.data_types == core.data_types
    assert core.update() == core
    assert core.update(codecs=core.codecs) == core
    assert cls(codecs={"gzip": gzip}) == cls(codecs=[gzip])
    other = _other(gzip)
    replaced = core.update(codecs={**core.codecs, "gzip": other})
    assert replaced.codecs["gzip"] is other
    assert {**replaced.codecs, "gzip": gzip} == dict(core.codecs)
    assert dict(cls().update(codecs=[*cls().codecs.values(), gzip]).codecs) == {"gzip": gzip}
    assert core.codecs["gzip"] is gzip


@pytest.mark.parametrize(("cls", "core", "kind", "gzip", "zarr_format"), FORMATS)
def test_error_a_slot_given_as_a_mapping_files_each_definition_under_its_name(
    cls: Any,  # noqa: ANN401 - either scope class
    core: Any,  # noqa: ANN401 - either core scope
    kind: Any,  # noqa: ANN401 - either codec kind
    gzip: Any,  # noqa: ANN401 - either gzip definition
    zarr_format: int,
) -> None:
    """A slot given as a mapping keys each definition by its own name, the name the scope files it under. Another key raises `TypeError` naming both, in the constructor and in `update`, rather than filing the definition under a name it does not read, or ignoring a misspelled key."""
    with pytest.raises(
        TypeError, match="codecs files each definition under its name, got 'gzip' under 'zstd'"
    ):
        cls(codecs={"zstd": gzip})
    with pytest.raises(TypeError, match="got 'gzip' under 'gzipp'"):
        core.update(codecs={**core.codecs, "gzipp": _other(gzip)})


@pytest.mark.parametrize(("cls", "core", "kind", "gzip", "zarr_format"), FORMATS)
def test_error_a_scope_has_only_the_slots_it_declares(
    cls: Any,  # noqa: ANN401 - either scope class
    core: Any,  # noqa: ANN401 - either core scope
    kind: Any,  # noqa: ANN401 - either codec kind
    gzip: Any,  # noqa: ANN401 - either gzip definition
    zarr_format: int,
) -> None:
    """A slot the scope does not declare raises `TypeError` naming it, in the constructor and in `update`, as a type checker reports it."""
    with pytest.raises(TypeError, match=f"a v{zarr_format} scope has no slot 'codec'"):
        cls(codec=[gzip])
    with pytest.raises(TypeError, match=f"a v{zarr_format} scope has no slot 'codec'"):
        core.update(codec=[gzip])


@pytest.mark.parametrize(("cls", "core", "kind", "gzip", "zarr_format"), FORMATS)
def test_error_union_refuses_one_name_filed_two_ways(
    cls: Any,  # noqa: ANN401 - either scope class
    core: Context,
    kind: Any,  # noqa: ANN401 - either codec kind
    gzip: Any,  # noqa: ANN401 - either gzip definition
    zarr_format: int,
) -> None:
    """Scopes of either format that file different definitions under one name in a slot have no union: `union` raises `ScopeConflictError` naming the kind and name. The two definitions read the same TypedDict, so the message says they differ rather than showing two identical descriptions."""
    with pytest.raises(ScopeConflictError) as raised:
        cls(codecs=[gzip]).union(cls(codecs=[_other(gzip)]))
    (conflict,) = raised.value.conflicts
    assert conflict.key == (kind, "gzip")
    assert "'gzip': two different definitions, each " in str(raised.value)


@pytest.mark.parametrize(("cls", "core", "kind", "gzip", "zarr_format"), FORMATS)
def test_error_a_slot_refuses_a_definition_of_the_other_format(
    cls: Any,  # noqa: ANN401 - either scope class
    core: Context,
    kind: Any,  # noqa: ANN401 - either codec kind
    gzip: Any,  # noqa: ANN401 - either gzip definition
    zarr_format: int,
) -> None:
    """A scope's `codecs` slot holds its own format's codecs only: the other format's gzip raises `TypeError`."""
    other = GZIP_CODEC_V2 if zarr_format == 3 else GZIP_CODEC
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
    other_data_type = ZarrV2DataTypeDefinition if zarr_format == 3 else DataTypeDefinition
    with pytest.raises(TypeError, match="kinds of its format"):
        core.claimant(other_data_type, "r*")


class _LooksLikeAScope:
    """A value with the `Context` protocol's members that is not one of this package's scope classes."""

    zarr_format = 3

    def claimant(self, kind: object, name: str) -> None:
        return None


class _BrokenScope:
    """A value whose `zarr_format` raises."""

    @property
    def zarr_format(self) -> int:
        raise RuntimeError


class _BooleanFormat:
    """A value whose `zarr_format` is a boolean, which no format is."""

    zarr_format = True


@pytest.mark.parametrize(
    ("given", "named"),
    [
        ("nope", "got 'nope'"),
        (_LooksLikeAScope(), "got a _LooksLikeAScope, a v3 scope"),
        (_BrokenScope(), "got <"),
        (_BooleanFormat(), "got <"),
    ],
    ids=["not-a-scope", "a-look-alike", "a-broken-format", "a-boolean-format"],
)
@pytest.mark.parametrize(
    ("read", "document", "cls"),
    [
        (read_array_metadata_v3, ARRAY, "ZarrV3Context"),
        (read_array_metadata_v2, V2_ARRAY, "ZarrV2Context"),
    ],
    ids=["v3", "v2"],
)
def test_error_a_reader_refuses_what_is_not_a_scope_of_its_format(
    read: Callable[..., object],
    document: object,
    cls: str,
    given: object,
    named: str,
) -> None:
    """A reader of either format takes a scope of its format's class, or None. Anything else raises `TypeError` naming the class the reader takes and what it was given, rather than failing inside the read or contradicting itself."""
    with pytest.raises(TypeError, match=f"a {cls}, {named}"):
        read(document, context=given)


@pytest.mark.parametrize(
    ("core", "given", "zarr_format"),
    [
        (CORE_CONTEXT, "nope", 3),
        (CORE_CONTEXT, CORE_V2, 3),
        (CORE_V2, "nope", 2),
        (CORE_V2, CORE_CONTEXT, 2),
    ],
    ids=["v3-not-a-scope", "v3-a-v2-scope", "v2-not-a-scope", "v2-a-v3-scope"],
)
def test_error_union_refuses_what_is_not_a_scope_of_its_format(
    core: Any,  # noqa: ANN401 - either core scope
    given: object,
    zarr_format: int,
) -> None:
    """A scope's `union` takes scopes of its own format only. A scope of the other format, or a value that is not a scope, raises `TypeError` naming what was given, for a v3 and a v2 scope."""
    with pytest.raises(
        TypeError,
        match=f"a union of v{zarr_format} scopes takes v{zarr_format} scopes, got",
    ):
        core.union(given)


@pytest.mark.parametrize("cls", [ZarrV3Context, ZarrV2Context], ids=["v3", "v2"])
def test_each_declared_slot_has_a_view(cls: Any) -> None:  # noqa: ANN401 - either scope class
    """Each scope class reads its slots from its `...Slots` TypedDict, so the slots are declared once. The slots hold every kind of the format, as its own check of a kind knows them. Each slot has a view of the same name, which is the slot that holds its kind: a type checker cannot match a view's name to a declared slot, so this test does."""
    slots_type = ZarrV3ContextSlots if cls is ZarrV3Context else ZarrV2ContextSlots
    assert list(cls.kinds) == list(slots_type.__annotations__)
    assert set(cls.kinds.values()) == (V3_KINDS if cls is ZarrV3Context else V2_KINDS)
    scope: Any = CORE_CONTEXT if cls is ZarrV3Context else CORE_V2
    for slot, kind in cls.kinds.items():
        assert getattr(scope, slot) == scope.definitions_of(kind)


@pytest.mark.parametrize(
    "annotation",
    [
        list[CodecDefinition[Any]],
        Iterable[CodecDefinition[Any]],
        Iterable[CodecDefinition[Any]] | Mapping[str, DataTypeDefinition[Any]],
        Iterable[CodecDefinition[Any]] | Mapping[int, CodecDefinition[Any]],
        Iterable[CodecDefinition[Any]] | Mapping[str, CodecDefinition[Any]] | None,
    ],
    ids=["a-list", "no-mapping", "two-kinds", "keyed-by-integers", "or-none"],
)
def test_error_a_slot_is_declared_as_definitions_of_a_kind_or_a_mapping_of_them(
    annotation: object,
) -> None:
    """`slot_kinds` reads a slot's kind from its annotation, `Iterable[K] | Mapping[str, K]`, with one kind `K` in both. Any other annotation raises `TypeError` when the scope class is defined, so a wrong declaration cannot pass unnoticed."""
    wrong = type("Wrong", (), {"__annotations__": {"codecs": annotation}})
    with pytest.raises(
        TypeError,
        match=r"Wrong\.codecs: a slot is annotated Iterable\[K\] \| Mapping\[str, K\]",
    ):
        slot_kinds(wrong, is_v3_kind)


def test_error_a_slot_holds_a_kind_of_its_format() -> None:
    """`slot_kinds` checks each slot's kind with the format's own check, so a v3 scope that declared a slot of v2 codecs raises `TypeError` when the scope class is defined."""

    class Mixed(TypedDict, total=False):
        codecs: Iterable[ZarrV2CodecDefinition[Any]] | Mapping[str, ZarrV2CodecDefinition[Any]]

    with pytest.raises(TypeError, match=r"Mixed\.codecs: ZarrV2CodecDefinition is not a kind"):
        slot_kinds(Mixed, is_v3_kind)
