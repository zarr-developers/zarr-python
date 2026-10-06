"""The algebra of scopes: value semantics for `Context`, claims, refinement, disagreements and joins."""

from __future__ import annotations

import pickle
from typing import Any

import pytest

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
