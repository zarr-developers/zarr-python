"""The scope of each format: `ZarrV3Context` and `ZarrV2Context` are separate classes that implement the `Context` protocol, and behave alike."""

from __future__ import annotations

import copy
import dataclasses
import pickle
from typing import Any

import pytest

from zarr_metadata.definition import Context, ScopeConflictError, UnclaimedField, resolve
from zarr_metadata.v2 import CORE_V2, GZIP_V2, ZarrV2Context
from zarr_metadata.v2.definition import ZarrV2CodecDefinition
from zarr_metadata.v3 import CORE, GZIP_CODEC, ZarrV3Context
from zarr_metadata.v3.definition import CodecDefinition


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
