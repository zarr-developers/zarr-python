"""The algebra of scopes: value semantics for `Context`, claims, refinement, disagreements and joins."""

from __future__ import annotations

import pickle
from typing import Any, get_type_hints

import pytest
from hypothesis import given
from hypothesis import strategies as st

from zarr_metadata.definition import (
    Claims,
    Conflict,
    Definition,
    Disagreements,
    EmptyConfiguration,
    RefusedField,
    ResolvedField,
    ScopeConflictError,
    resolve,
)
from zarr_metadata.definition._definition import fields_of
from zarr_metadata.definition._scope import claims_of, kind_name, refines
from zarr_metadata.v3.codec.bytes import BYTES_CODEC
from zarr_metadata.v3.codec.crc32c import CRC32C_CODEC
from zarr_metadata.v3.codec.gzip import GZIP_CODEC
from zarr_metadata.v3.codec.sharding_indexed import SHARDING_INDEXED_CODEC
from zarr_metadata.v3.codec.zstd import ZSTD_CODEC
from zarr_metadata.v3.data_type.bytes import BYTES_DATA_TYPE
from zarr_metadata.v3.data_type.raw import RAW_BYTES_DATA_TYPE
from zarr_metadata.v3.definition import (
    CORE,
    CORE_AND_EXTENSIONS,
    ArrayExtraFieldDefinition,
    ChunkGridDefinition,
    ChunkKeyEncodingDefinition,
    CodecDefinition,
    DataTypeDefinition,
    GroupExtraFieldDefinition,
    StorageTransformerDefinition,
    ZarrV3Context,
)

SHARD = {
    "name": "sharding_indexed",
    "configuration": {
        "chunk_shape": [1],
        "codecs": ["bytes"],
        "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}, "crc32c"],
    },
}

MY_GZIP = CodecDefinition(
    name="gzip", configuration=EmptyConfiguration, kind="bytes_bytes", size="dynamic"
)
"""A private definition of the name `gzip`: a second definition under the same name, which takes no configuration.

Definitions compare by their contents, so a definition rebuilt from the
core TypedDict and the core rules would equal the core definition. This
one reads `gzip` differently.
"""


def _read(data: object, kind: type[Definition[Any]], context: ZarrV3Context) -> ResolvedField[Any]:
    return resolve(data, kind, context)[0]


@pytest.mark.parametrize(
    ("left", "right", "equal"),
    [
        (
            ZarrV3Context(codecs=[GZIP_CODEC, BYTES_CODEC]),
            ZarrV3Context(codecs=[BYTES_CODEC, GZIP_CODEC]),
            True,
        ),
        (CORE, pickle.loads(pickle.dumps(CORE)), True),
        (CORE, CORE_AND_EXTENSIONS, False),
        (ZarrV3Context(codecs=[GZIP_CODEC]), ZarrV3Context(codecs=[GZIP_CODEC, ZSTD_CODEC]), False),
        (ZarrV3Context(), ZarrV3Context(), True),
    ],
    ids=["order", "pickle", "core-vs-extensions", "subset", "empty"],
)
def test_a_scope_is_equal_to_another_by_the_definitions_it_files(
    left: ZarrV3Context, right: ZarrV3Context, equal: bool
) -> None:
    """Two scopes are equal when they file the same definitions under the same names, however they were built. Equal scopes have equal hashes."""
    assert (left == right) is equal
    if equal:
        assert hash(left) == hash(right)


def test_a_scope_is_a_set_member() -> None:
    """A scope is hashable, so it can be a dict key or a set member. It was not hashable when its tables were mapping proxies."""
    assert len({CORE, CORE_AND_EXTENSIONS, pickle.loads(pickle.dumps(CORE))}) == 2


def test_error_a_scope_conflict_says_each_disagreement() -> None:
    """A scope conflict lists each `(kind, name)` with the claimed definition, the found definition, and the location, so a caller can see every disagreement at once."""
    error = ScopeConflictError(
        (
            Conflict((CodecDefinition, "bytes"), BYTES_CODEC, None, ("codecs", 0)),
            Conflict((CodecDefinition, "gzip"), None, GZIP_CODEC),
        )
    )
    assert error.conflicts[0].key == (CodecDefinition, "bytes")
    assert str(error) == (
        "codec 'bytes' at ('codecs', 0): claimed CodecDefinition(name='bytes') of "
        "BytesCodecConfiguration, found no definition; "
        "codec 'gzip': claimed no definition, found CodecDefinition(name='gzip') of "
        "GzipCodecConfiguration"
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
        (RefusedField(json=3, name=None, read_as=CodecDefinition), {}),
    ],
    ids=["read", "unclaimed", "raw-bits", "nested", "refused-claimed", "refused-nameless"],
)
def test_claims_of_says_what_a_reading_claimed_of_each_name(
    field: ResolvedField[Any], claims: dict[object, object]
) -> None:
    """A reading's claims map each name written by the field, and by the fields it holds, to the definition that read it. Each name is keyed as the scope files it (raw bits under `r*`), with None where nothing claimed the name. A field refused by a definition still claims that definition. A field with no name claims nothing."""
    assert claims_of(fields_of(field)) == claims


def test_error_claims_of_refuses_one_name_read_two_ways() -> None:
    """Fields read in two scopes that give one name two different definitions have no single set of claims. `claims_of` raises a `ScopeConflictError` that names the key."""
    fields = [
        *fields_of(_read("gzip", CodecDefinition, CORE), ("codecs", 0)),
        *fields_of(_read("gzip", CodecDefinition, ZarrV3Context(codecs=[MY_GZIP])), ("codecs", 1)),
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
BLOSC = {"cname": "zstd", "shuffle": "noshuffle", "blocksize": 0}
BLOSC_LEVEL_ONE = {"name": "blosc", "configuration": {**BLOSC, "clevel": 1}}
BLOSC_LEVEL_TRUE = {"name": "blosc", "configuration": {**BLOSC, "clevel": True}}
"""Two documents that Python's `==` treats as equal, but that `json_text` tells apart."""
SHARD_HOLDING_REFUSED = {
    "name": "sharding_indexed",
    "configuration": {
        "chunk_shape": [1],
        "codecs": ["bytes", {"name": "gzip", "configuration": {"level": 12}}],
        "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}, "crc32c"],
    },
}
"""A shard that is read, holding a gzip that its definition refuses."""
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
            _read(GZIP_FIELD, CodecDefinition, ZarrV3Context(codecs=[MY_GZIP])),
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
        (
            _read("crc32c", CodecDefinition, CORE),
            _read({"name": "crc32c"}, CodecDefinition, ZarrV3Context()),
            True,
        ),
        (
            _read(BLOSC_LEVEL_ONE, CodecDefinition, CORE),
            _read(BLOSC_LEVEL_TRUE, CodecDefinition, ZarrV3Context()),
            False,
        ),
        (
            _read(SHARD_HOLDING_REFUSED, CodecDefinition, CORE),
            _read(SHARD_HOLDING_REFUSED, CodecDefinition, CORE),
            True,
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
        "gain-over-another-spelling",
        "no-gain-over-true-for-one",
        "holding-a-refused-field",
    ],
)
def test_refines_orders_readings_by_information(
    field: ResolvedField[Any], other: ResolvedField[Any], expected: bool
) -> None:
    """`field` refines `other` when it reads the same where both read, and gains where `other` left a name unclaimed, including in the fields it holds. A loss, a conflict, a refused field, or two unclaimed fields written differently do not refine."""
    assert refines(field, other) is expected


DOCUMENTS = [
    GZIP_FIELD,
    "crc32c",
    {"name": "crc32c"},
    {"name": "crc32c", "configuration": {}},
    ZSTD_FIELD,
    NESTED_ZSTD,
    BLOSC_LEVEL_ONE,
    SHARD_HOLDING_REFUSED,
]
CONTEXTS = [ZarrV3Context(), CORE, CORE_AND_EXTENSIONS]
READINGS = [
    _read(document, CodecDefinition, context) for document in DOCUMENTS for context in CONTEXTS
]


@given(st.sampled_from(READINGS), st.sampled_from(READINGS), st.sampled_from(READINGS))
def test_refines_is_a_partial_order_whose_bottom_is_equality(
    a: ResolvedField[Any], b: ResolvedField[Any], c: ResolvedField[Any]
) -> None:
    """Over readings of documents in several scopes and spellings, `refines` is reflexive and transitive. Two fields that refine each other are equal, and equal fields are refined by the same fields."""
    assert refines(a, a)
    if refines(a, b) and refines(b, c):
        assert refines(a, c)
    assert (refines(a, b) and refines(b, a)) is (a == b)
    if a == b:
        assert refines(c, a) is refines(c, b)


@pytest.mark.parametrize(
    ("context", "claims", "gains", "conflicts"),
    [
        (CORE, {(CodecDefinition, "gzip"): GZIP_CODEC}, (), ()),
        (
            CORE_AND_EXTENSIONS,
            {(CodecDefinition, "zstd"): None},
            ((CodecDefinition, "zstd"),),
            (),
        ),
        (
            CORE,
            {(CodecDefinition, "zstd"): ZSTD_CODEC},
            (),
            (Conflict((CodecDefinition, "zstd"), ZSTD_CODEC, None),),
        ),
        (
            ZarrV3Context(codecs=[MY_GZIP]),
            {(CodecDefinition, "gzip"): GZIP_CODEC},
            (),
            (Conflict((CodecDefinition, "gzip"), GZIP_CODEC, MY_GZIP),),
        ),
        (CORE, {(CodecDefinition, "acme.x"): None}, (), ()),
        (CORE, {(DataTypeDefinition, "r*"): RAW_BYTES_DATA_TYPE}, (), ()),
    ],
    ids=["agrees", "gain", "loss", "conflict", "unclaimed-both", "raw-bits"],
)
def test_disagreements_says_where_a_scope_reads_claims_otherwise(
    context: ZarrV3Context,
    claims: dict[Any, Any],
    gains: tuple[Any, ...],
    conflicts: tuple[Conflict, ...],
) -> None:
    """A scope agrees with claims that it reads the same way. It gains where it claims a name the claims left unclaimed. It conflicts where it reads a name with another definition or with none."""
    found = context.disagreements(claims)
    assert isinstance(found, Disagreements)
    assert (found.gains, found.conflicts) == (gains, conflicts)
    assert found.agrees is (len(gains) == 0 and len(conflicts) == 0)


@pytest.mark.parametrize(
    ("contexts", "joined"),
    [
        ((CORE, CORE_AND_EXTENSIONS), CORE_AND_EXTENSIONS),
        (
            (ZarrV3Context(codecs=[GZIP_CODEC]), ZarrV3Context(codecs=[ZSTD_CODEC])),
            ZarrV3Context(codecs=[GZIP_CODEC, ZSTD_CODEC]),
        ),
        ((), ZarrV3Context()),
        ((CORE, CORE), CORE),
        (
            (ZarrV3Context(codecs=[BYTES_CODEC]), ZarrV3Context(data_types=[BYTES_DATA_TYPE])),
            ZarrV3Context(data_types=[BYTES_DATA_TYPE], codecs=[BYTES_CODEC]),
        ),
    ],
    ids=["subset", "disjoint", "none", "same", "same-name-two-kinds"],
)
def test_joined_is_the_least_scope_above_each(
    contexts: tuple[ZarrV3Context, ...], joined: ZarrV3Context
) -> None:
    """The join of scopes files every definition that any of them files, once. The same name in two kinds counts as two names."""
    assert ZarrV3Context.joined(*contexts) == joined


def test_error_joined_refuses_one_name_filed_two_ways() -> None:
    """Scopes that file different definitions under one name have no join. `ZarrV3Context.joined` raises a `ScopeConflictError` that names the key and both definitions."""
    with pytest.raises(ScopeConflictError) as raised:
        ZarrV3Context.joined(CORE, ZarrV3Context(codecs=[MY_GZIP]))
    (conflict,) = raised.value.conflicts
    assert (conflict.key, conflict.claimed, conflict.found) == (
        (CodecDefinition, "gzip"),
        GZIP_CODEC,
        MY_GZIP,
    )


def test_claims_is_a_type_a_signature_can_hold() -> None:
    """`Claims` resolves as an annotation at run time, when a caller's `get_type_hints` reads it: it is a type, not a string."""

    def read(claims: Claims) -> None:
        pass

    assert "claims" in get_type_hints(read)


@pytest.mark.parametrize(
    ("kind", "said"),
    [
        (CodecDefinition, "codec"),
        (DataTypeDefinition, "data type"),
        (ChunkGridDefinition, "chunk grid"),
        (ChunkKeyEncodingDefinition, "chunk key encoding"),
        (StorageTransformerDefinition, "storage transformer"),
    ],
    ids=["codec", "data-type", "chunk-grid", "chunk-key-encoding", "storage-transformer"],
)
def test_a_kind_is_named_in_words(kind: type[Definition[Any]], said: str) -> None:
    """`kind_name` names each kind of definition in the words a message uses: `ChunkKeyEncodingDefinition` is "chunk key encoding"."""
    assert kind_name(kind) == said
    assert said in str(Conflict((kind, "x"), None, None))


def test_a_gain_is_judged_by_what_the_definition_reads_not_by_spelling() -> None:
    """An accepted field refines an unclaimed one when its definition, reading the JSON of the unclaimed field, produces the accepted field. Two spellings of one configuration count as the same gain, so `refines` is transitive through `==`. A spelling that the definition reads as something else is not a gain."""
    from zarr_metadata.definition import (
        resolve,
    )
    from zarr_metadata.v3.definition import (
        CORE,
        ChunkKeyEncodingDefinition,
    )

    nothing = ZarrV3Context()
    spelled_out, _ = resolve(
        {"name": "default", "configuration": {"separator": "/"}}, ChunkKeyEncodingDefinition, CORE
    )
    bare, _ = resolve({"name": "default"}, ChunkKeyEncodingDefinition, CORE)
    unclaimed, _ = resolve({"name": "default"}, ChunkKeyEncodingDefinition, nothing)
    assert spelled_out == bare
    assert refines(bare, unclaimed)
    assert refines(spelled_out, unclaimed)
    other, _ = resolve(
        {"name": "default", "configuration": {"separator": "."}}, ChunkKeyEncodingDefinition, CORE
    )
    assert not refines(other, unclaimed)
    shard = {
        "name": "sharding_indexed",
        "configuration": {
            "chunk_shape": [2],
            "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
            "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}, "crc32c"],
            "index_location": "end",
        },
    }
    read, _ = resolve(shard, CodecDefinition, CORE)
    unread, _ = resolve({**shard, "must_understand": True}, CodecDefinition, nothing)
    assert refines(read, unread)


OTHER_GZIP = CodecDefinition(
    name="gzip", configuration=EmptyConfiguration, kind="bytes_bytes", size="static"
)
"""A second definition under the core gzip's name, which makes a join conflict."""


def test_a_scope_conflict_error_pickles_and_copies_with_its_conflicts() -> None:
    """`ScopeConflictError` keeps its conflicts and message through pickling and copying, as `MetadataValidationError` does. So a conflict reported in another process reads the same here."""
    import copy

    with pytest.raises(ScopeConflictError) as raised:
        ZarrV3Context.joined(CORE, ZarrV3Context(codecs=[OTHER_GZIP]))
    error = raised.value
    error.add_note("seen in a join")
    for again in (pickle.loads(pickle.dumps(error)), copy.copy(error), copy.deepcopy(error)):
        assert again.conflicts == error.conflicts
        assert str(again) == str(error)
        assert str(again).startswith("codec 'gzip'")
        assert again.__notes__ == ["seen in a join"]


def test_a_conflict_names_the_field_as_the_document_writes_it_and_tells_the_definitions_apart() -> (
    None
):
    """A conflict found in a document reports the name the document writes, `r16`, not the name its definition is filed under, `r*`. It tells two definitions of one name apart by the configuration each declares, as the messages for consolidated entries do."""
    import dataclasses

    from zarr_metadata.model import ZarrV3ArrayMetadata
    from zarr_metadata.v3.definition import CORE_AND_EXTENSIONS

    document = dict(
        ZarrV3ArrayMetadata.create_default(shape=(2,), data_type="r16", fill_value=[0, 0]).to_json()
    )
    model = ZarrV3ArrayMetadata(document, CORE_AND_EXTENSIONS)
    other = dataclasses.replace(RAW_BYTES_DATA_TYPE, configuration=EmptyConfiguration)
    with pytest.raises(ScopeConflictError) as raised:
        model.refined_in(
            CORE_AND_EXTENSIONS.update(data_types={**CORE_AND_EXTENSIONS.data_types, "r*": other})
        )
    message = str(raised.value)
    assert message.startswith("data type 'r16'")
    assert "RawBytesConfiguration" in message
    assert "EmptyConfiguration" in message
    with pytest.raises(ScopeConflictError) as joined:
        ZarrV3Context.joined(ZarrV3Context(codecs=[GZIP_CODEC]), ZarrV3Context(codecs=[OTHER_GZIP]))
    assert str(joined.value).count("gzip") >= 2
    assert "EmptyConfiguration" in str(joined.value)


def test_each_slot_is_a_read_only_view_by_name() -> None:
    """A v3 scope has one slot per kind, as `ZarrV3Context.kinds` lists them: the five extension points, and the extra fields of arrays and of groups. Each slot maps the name each definition is filed under to the definition, and cannot be changed in place."""
    assert dict(ZarrV3Context.kinds) == {
        "data_types": DataTypeDefinition,
        "chunk_grids": ChunkGridDefinition,
        "chunk_key_encodings": ChunkKeyEncodingDefinition,
        "codecs": CodecDefinition,
        "storage_transformers": StorageTransformerDefinition,
        "array_extra_fields": ArrayExtraFieldDefinition,
        "group_extra_fields": GroupExtraFieldDefinition,
    }
    assert CORE.codecs["gzip"] is GZIP_CODEC
    assert CORE.data_types["r*"] is RAW_BYTES_DATA_TYPE
    assert dict(CORE.storage_transformers) == {}
    with pytest.raises(TypeError):
        CORE.codecs["gzip"] = MY_GZIP  # pyright: ignore[reportIndexIssue]


def test_error_a_slot_refuses_a_definition_of_another_kind() -> None:
    """A definition of another kind than its slot holds raises `TypeError` naming the slot, the definition and its kind, at run time as a type checker does."""
    with pytest.raises(
        TypeError, match="codecs holds codec definitions, got 'bytes', a data type definition"
    ):
        ZarrV3Context(codecs=[BYTES_DATA_TYPE])  # pyright: ignore[reportArgumentType]


def test_error_a_slot_refuses_two_definitions_under_one_name() -> None:
    """Two different definitions under one name in a slot raise `ScopeConflictError` naming both, because the scope could not tell which one reads the name. `update` is the way to replace a definition."""
    with pytest.raises(ScopeConflictError) as raised:
        ZarrV3Context(codecs=[GZIP_CODEC, OTHER_GZIP])
    (conflict,) = raised.value.conflicts
    assert (conflict.key, conflict.claimed, conflict.found) == (
        (CodecDefinition, "gzip"),
        GZIP_CODEC,
        OTHER_GZIP,
    )
