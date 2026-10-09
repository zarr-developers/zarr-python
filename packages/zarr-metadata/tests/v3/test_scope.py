"""The algebra of scopes: value semantics for `Context`, claims, refinement, disagreements and joins."""

from __future__ import annotations

import pickle
from typing import Any, get_type_hints

import pytest
from hypothesis import given
from hypothesis import strategies as st

from zarr_metadata.v3._definition import (
    fields_of,
)
from zarr_metadata.v3._scope import (
    Claims,
    Disagreements,
    claims_of,
    kind_name,
    refines,
)
from zarr_metadata.v3.codec.bytes import BYTES_CODEC
from zarr_metadata.v3.codec.crc32c import CRC32C_CODEC, Empty
from zarr_metadata.v3.codec.gzip import GZIP_CODEC
from zarr_metadata.v3.codec.sharding_indexed import SHARDING_INDEXED_CODEC
from zarr_metadata.v3.codec.zstd import ZSTD_CODEC
from zarr_metadata.v3.data_type.bytes import BYTES_DATA_TYPE
from zarr_metadata.v3.data_type.raw import RAW_BYTES_DATA_TYPE
from zarr_metadata.v3.definition import (
    CORE,
    CORE_AND_EXTENSIONS,
    ChunkGridDefinition,
    ChunkKeyEncodingDefinition,
    CodecDefinition,
    Conflict,
    Context,
    DataTypeDefinition,
    Definition,
    RefusedField,
    ResolvedField,
    ScopeConflictError,
    StorageTransformerDefinition,
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


def _read(data: object, kind: type[Definition[Any]], scope: Context) -> ResolvedField[Any]:
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
BLOSC = {"cname": "zstd", "shuffle": "noshuffle", "blocksize": 0}
BLOSC_LEVEL_ONE = {"name": "blosc", "configuration": {**BLOSC, "clevel": 1}}
BLOSC_LEVEL_TRUE = {"name": "blosc", "configuration": {**BLOSC, "clevel": True}}
"""Two documents Python's `==` takes for one, which `json_text` tells apart."""
SHARD_HOLDING_REFUSED = {
    "name": "sharding_indexed",
    "configuration": {
        "chunk_shape": [1],
        "codecs": ["bytes", {"name": "gzip", "configuration": {"level": 12}}],
        "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}, "crc32c"],
    },
}
"""A shard read, holding a gzip its definition refuses."""
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
        (
            _read("crc32c", CodecDefinition, CORE),
            _read({"name": "crc32c"}, CodecDefinition, Context.of()),
            True,
        ),
        (
            _read(BLOSC_LEVEL_ONE, CodecDefinition, CORE),
            _read(BLOSC_LEVEL_TRUE, CodecDefinition, Context.of()),
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
    """`field` refines `other` when it reads the same where both read and gains where `other` left a name unclaimed -- in the fields it holds too; a loss, a conflict, a refused field, or two unclaimed fields written differently do not."""
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
SCOPES = [Context.of(), CORE, CORE_AND_EXTENSIONS]
READINGS = [_read(document, CodecDefinition, scope) for document in DOCUMENTS for scope in SCOPES]


@given(st.sampled_from(READINGS), st.sampled_from(READINGS), st.sampled_from(READINGS))
def test_refines_is_a_partial_order_whose_bottom_is_equality(
    a: ResolvedField[Any], b: ResolvedField[Any], c: ResolvedField[Any]
) -> None:
    """Over readings of documents in several scopes and spellings, `refines` is reflexive and transitive, two fields that refine each other are equal, and equal fields refine the same fields."""
    assert refines(a, a)
    if refines(a, b) and refines(b, c):
        assert refines(a, c)
    assert (refines(a, b) and refines(b, a)) is (a == b)
    if a == b:
        assert refines(c, a) is refines(c, b)


@pytest.mark.parametrize(
    ("scope", "claims", "gains", "conflicts"),
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
            Context.of(MY_GZIP),
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
    scope: Context,
    claims: dict[Any, Any],
    gains: tuple[Any, ...],
    conflicts: tuple[Conflict, ...],
) -> None:
    """A scope agrees with claims it reads identically, gains where it claims what the claims left unclaimed, and conflicts where it reads a name by another definition or by none."""
    found = scope.disagreements(claims)
    assert isinstance(found, Disagreements)
    assert (found.gains, found.conflicts) == (gains, conflicts)
    assert found.agrees is (len(gains) == 0 and len(conflicts) == 0)


@pytest.mark.parametrize(
    ("scopes", "joined"),
    [
        ((CORE, CORE_AND_EXTENSIONS), CORE_AND_EXTENSIONS),
        ((Context.of(GZIP_CODEC), Context.of(ZSTD_CODEC)), Context.of(GZIP_CODEC, ZSTD_CODEC)),
        ((), Context.of()),
        ((CORE, CORE), CORE),
        (
            (Context.of(BYTES_CODEC), Context.of(BYTES_DATA_TYPE)),
            Context.of(BYTES_CODEC, BYTES_DATA_TYPE),
        ),
    ],
    ids=["subset", "disjoint", "none", "same", "same-name-two-kinds"],
)
def test_joined_is_the_least_scope_above_each(scopes: tuple[Context, ...], joined: Context) -> None:
    """The join of scopes files every definition any of them files, once; one kind's name is not another's."""
    assert Context.joined(*scopes) == joined


def test_error_joined_refuses_one_name_filed_two_ways() -> None:
    """Scopes that file different definitions under one name have no join: a `ScopeConflictError` naming the key and both definitions."""
    with pytest.raises(ScopeConflictError) as raised:
        Context.joined(CORE, Context.of(MY_GZIP))
    (conflict,) = raised.value.conflicts
    assert (conflict.key, conflict.claimed, conflict.found) == (
        (CodecDefinition, "gzip"),
        GZIP_CODEC,
        MY_GZIP,
    )


def test_claims_is_a_type_a_signature_can_hold() -> None:
    """`Claims` resolves as an annotation at run time, as a caller's `get_type_hints` reads one: it is a type, not a string."""

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
    """`kind_name` names each kind of definition as a message does: `ChunkKeyEncodingDefinition` is "chunk key encoding"."""
    assert kind_name(kind) == said
    assert said in str(Conflict((kind, "x"), None, None))


def test_a_gain_is_judged_by_what_the_definition_reads_not_by_spelling() -> None:
    """An accepted field refines an unclaimed one when its definition, reading what the unclaimed field wrote, reads the accepted field: two spellings of one configuration are one gain, so `refines` is transitive through `==`, and a spelling the definition reads otherwise is no gain."""
    from zarr_metadata.v3.definition import CORE, ChunkKeyEncodingDefinition, Context, resolve

    nothing = Context.of()
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


OTHER_GZIP = CodecDefinition(name="gzip", configuration=Empty, kind="bytes_bytes", size="static")
"""A second definition under the core gzip's name: what a join conflicts on."""


def test_a_scope_conflict_error_pickles_and_copies_with_its_conflicts() -> None:
    """`ScopeConflictError` pickles and copies as it was raised: the same conflicts, the same message, as `MetadataValidationError` does, so a conflict reported in another process reads the same here."""
    import copy

    with pytest.raises(ScopeConflictError) as raised:
        Context.joined(CORE, Context.of(OTHER_GZIP))
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
    """A conflict found in a document says the name the document writes, `r16`, not the name its definition is filed under, `r*`, and tells two definitions of one name apart by the configuration each declares, as the consolidated entries' messages do."""
    import dataclasses

    from zarr_metadata.model import ZarrV3ArrayMetadata
    from zarr_metadata.v3.definition import CORE_AND_EXTENSIONS

    document = dict(
        ZarrV3ArrayMetadata.create_default(shape=(2,), data_type="r16", fill_value=[0, 0]).to_json()
    )
    model = ZarrV3ArrayMetadata(document, CORE_AND_EXTENSIONS)
    other = dataclasses.replace(RAW_BYTES_DATA_TYPE, configuration=Empty)
    with pytest.raises(ScopeConflictError) as raised:
        model.refined_in(CORE_AND_EXTENSIONS.extended_with(other))
    message = str(raised.value)
    assert message.startswith("data type 'r16'")
    assert "RawBytesConfiguration" in message
    assert "Empty" in message
    with pytest.raises(ScopeConflictError) as joined:
        Context.joined(Context.of(GZIP_CODEC), Context.of(OTHER_GZIP))
    assert str(joined.value).count("gzip") >= 2
    assert "Empty" in str(joined.value)


def test_error_a_scope_built_from_tables_keeps_the_invariants_of() -> None:
    """`Context(tables)`, the constructor, refuses what `Context.of` refuses: a key that is no kind, and kinds of two formats, so no scope is built that `of` could not build."""
    from types import MappingProxyType

    from zarr_metadata.v2.data_type.scalar import UINT_V2
    from zarr_metadata.v2.definition import ZarrV2DataTypeDefinition

    with pytest.raises(TypeError, match="one Zarr format"):
        Context(
            MappingProxyType(
                {CodecDefinition: {"gzip": GZIP_CODEC}, ZarrV2DataTypeDefinition: {"uint": UINT_V2}}
            )
        )
    with pytest.raises(TypeError, match="kind"):
        Context(MappingProxyType({int: {"gzip": GZIP_CODEC}}))  # pyright: ignore[reportArgumentType]
