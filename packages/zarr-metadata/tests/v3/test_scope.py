"""The algebra of scopes: value semantics for `Context`, claims, refinement, disagreements and joins."""

from __future__ import annotations

import pickle
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from zarr_metadata.v3.codec.bytes import BYTES_CODEC
from zarr_metadata.v3.codec.crc32c import CRC32C_CODEC, Empty
from zarr_metadata.v3.codec.gzip import GZIP_CODEC
from zarr_metadata.v3.codec.sharding_indexed import SHARDING_INDEXED_CODEC
from zarr_metadata.v3.codec.zstd import ZSTD_CODEC
from zarr_metadata.v3.data_type.raw import RAW_BYTES_DATA_TYPE
from zarr_metadata.v3.definition import (
    CORE,
    CORE_AND_EXTENSIONS,
    CodecDefinition,
    Conflict,
    Context,
    DataTypeDefinition,
    Definition,
    Refused,
    Resolved,
    ScopeConflictError,
    claims_of,
    fields_of,
    refines,
    resolve,
)

SHARD = {
    "name": "sharding_indexed",
    "configuration": {
        "chunk_shape": [1],
        "codecs": ["bytes"],
        "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}, "crc32c"],
    },
}

MY_GZIP = CodecDefinition(name="gzip", configuration=Empty, kind="bytes_bytes", size="dynamic")
"""A private reading of the name `gzip`: another definition under one name, which takes no configuration.

Definitions compare by what they hold, so one rebuilt from the core
TypedDict with the core rules would be the core definition; this one
reads `gzip` otherwise.
"""


def _read(data: object, kind: type[Definition[Any]], scope: Context) -> Resolved[Any]:
    return resolve(data, kind, scope)[0]


@pytest.mark.parametrize(
    ("left", "right", "equal"),
    [
        (Context.of(GZIP_CODEC, BYTES_CODEC), Context.of(BYTES_CODEC, GZIP_CODEC), True),
        (CORE, pickle.loads(pickle.dumps(CORE)), True),
        (CORE, CORE_AND_EXTENSIONS, False),
        (Context.of(GZIP_CODEC), Context.of(GZIP_CODEC, ZSTD_CODEC), False),
        (Context.of(), Context.of(), True),
    ],
    ids=["order", "pickle", "core-vs-extensions", "subset", "empty"],
)
def test_a_scope_is_equal_to_another_by_the_definitions_it_files(
    left: Context, right: Context, equal: bool
) -> None:
    """Two scopes are one when they file the same definitions under the same names, however they were built; equal scopes hash alike."""
    assert (left == right) is equal
    if equal:
        assert hash(left) == hash(right)


def test_a_scope_is_a_set_member() -> None:
    """A scope hashes, so it can key a dict or sit in a set, which it could not when its tables were mapping proxies."""
    assert len({CORE, CORE_AND_EXTENSIONS, pickle.loads(pickle.dumps(CORE))}) == 2


def test_error_a_scope_conflict_says_each_disagreement() -> None:
    """A scope conflict lists each `(kind, name)` with what was claimed and what was found, and where, so a caller can see every disagreement at once."""
    error = ScopeConflictError(
        (
            Conflict((CodecDefinition, "bytes"), BYTES_CODEC, None, ("codecs", 0)),
            Conflict((CodecDefinition, "gzip"), None, GZIP_CODEC),
        )
    )
    assert error.conflicts[0].key == (CodecDefinition, "bytes")
    assert str(error) == (
        "codec 'bytes' at ('codecs', 0): claimed CodecDefinition(name='bytes'), found None; "
        "codec 'gzip': claimed None, found CodecDefinition(name='gzip')"
    )


@pytest.mark.parametrize(
    ("field", "claims"),
    [
        (_read("gzip", CodecDefinition, CORE), {(CodecDefinition, "gzip"): GZIP_CODEC}),
        (_read("zstd", CodecDefinition, CORE), {(CodecDefinition, "zstd"): None}),
        (
            _read("r16", DataTypeDefinition, CORE),
            {(DataTypeDefinition, "r*"): RAW_BYTES_DATA_TYPE},
        ),
        (
            _read(SHARD, CodecDefinition, CORE),
            {
                (CodecDefinition, "sharding_indexed"): SHARDING_INDEXED_CODEC,
                (CodecDefinition, "bytes"): BYTES_CODEC,
                (CodecDefinition, "crc32c"): CRC32C_CODEC,
            },
        ),
        (
            _read({"name": "gzip", "configuration": {"level": 12}}, CodecDefinition, CORE),
            {(CodecDefinition, "gzip"): GZIP_CODEC},
        ),
        (Refused(json=3, name=None, read_as=CodecDefinition), {}),
    ],
    ids=["read", "unclaimed", "raw-bits", "nested", "refused-claimed", "refused-nameless"],
)
def test_claims_of_says_what_a_reading_claimed_of_each_name(
    field: Resolved[Any], claims: dict[object, object]
) -> None:
    """A reading's claims name the definition that read each name the field and the fields it holds write, keyed as the scope files it -- raw bits under `r*` -- and None where nothing claimed one; a field refused by a definition still claims it, and one that names nothing claims nothing."""
    assert claims_of(fields_of(field)) == claims


def test_error_claims_of_refuses_one_name_read_two_ways() -> None:
    """Fields read in two scopes that give one name two definitions have no single set of claims: a `ScopeConflictError` naming the key."""
    fields = [
        *fields_of(_read("gzip", CodecDefinition, CORE), ("codecs", 0)),
        *fields_of(_read("gzip", CodecDefinition, Context.of(MY_GZIP)), ("codecs", 1)),
    ]
    with pytest.raises(ScopeConflictError) as raised:
        claims_of(fields)
    (conflict,) = raised.value.conflicts
    assert (conflict.key, conflict.claimed, conflict.found, conflict.loc) == (
        (CodecDefinition, "gzip"),
        GZIP_CODEC,
        MY_GZIP,
        ("codecs", 1),
    )


GZIP_FIELD = {"name": "gzip", "configuration": {"level": 5}}
ZSTD_FIELD = {"name": "zstd", "configuration": {"level": 3}}
NESTED_ZSTD = {
    "name": "sharding_indexed",
    "configuration": {
        "chunk_shape": [1],
        "codecs": ["bytes", ZSTD_FIELD],
        "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}, "crc32c"],
    },
}


@pytest.mark.parametrize(
    ("field", "other", "expected"),
    [
        (_read(GZIP_FIELD, CodecDefinition, CORE), _read(GZIP_FIELD, CodecDefinition, CORE), True),
        (
            _read({"name": "crc32c"}, CodecDefinition, CORE),
            _read("crc32c", CodecDefinition, CORE),
            True,
        ),
        (
            _read(ZSTD_FIELD, CodecDefinition, CORE_AND_EXTENSIONS),
            _read(ZSTD_FIELD, CodecDefinition, CORE),
            True,
        ),
        (
            _read(ZSTD_FIELD, CodecDefinition, CORE),
            _read(ZSTD_FIELD, CodecDefinition, CORE_AND_EXTENSIONS),
            False,
        ),
        (
            _read(GZIP_FIELD, CodecDefinition, CORE),
            _read(GZIP_FIELD, CodecDefinition, Context.of(MY_GZIP)),
            False,
        ),
        (
            _read(NESTED_ZSTD, CodecDefinition, CORE_AND_EXTENSIONS),
            _read(NESTED_ZSTD, CodecDefinition, CORE),
            True,
        ),
        (
            _read(NESTED_ZSTD, CodecDefinition, CORE),
            _read(NESTED_ZSTD, CodecDefinition, CORE_AND_EXTENSIONS),
            False,
        ),
        (
            _read({"name": "gzip", "configuration": {"level": 12}}, CodecDefinition, CORE),
            _read("gzip", CodecDefinition, CORE),
            False,
        ),
        (_read("zstd", CodecDefinition, CORE), _read("zstd", CodecDefinition, CORE), True),
        (
            _read({"name": "zstd", "configuration": {"level": 1}}, CodecDefinition, CORE),
            _read({"name": "zstd", "configuration": {"level": 2}}, CodecDefinition, CORE),
            False,
        ),
    ],
    ids=[
        "same",
        "same-spelled-otherwise",
        "gain",
        "loss",
        "conflict",
        "nested-gain",
        "nested-loss",
        "refused",
        "both-unclaimed",
        "unclaimed-differ",
    ],
)
def test_refines_orders_readings_by_information(
    field: Resolved[Any], other: Resolved[Any], expected: bool
) -> None:
    """`field` refines `other` when it reads the same where both read and gains where `other` left a name unclaimed -- in the fields it holds too; a loss, a conflict, a refused field, or two unclaimed fields written differently do not."""
    assert refines(field, other) is expected


@given(st.sampled_from([GZIP_FIELD, "crc32c", {"name": "crc32c"}, ZSTD_FIELD, NESTED_ZSTD]))
def test_refines_is_reflexive_and_mutual_refinement_is_equality(data: object) -> None:
    """Every field a scope reads refines itself, and two such fields that refine each other are equal: the order's bottom is equality. A refused field is outside the order, so it is not sampled."""
    for scope in (CORE, CORE_AND_EXTENSIONS):
        field = _read(data, CodecDefinition, scope)
        assert refines(field, field)
    low = _read(data, CodecDefinition, CORE)
    high = _read(data, CodecDefinition, CORE_AND_EXTENSIONS)
    assert (refines(low, high) and refines(high, low)) is (low == high)
