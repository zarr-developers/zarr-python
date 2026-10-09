"""A v3 array document, read: each field as a scope read it, and its codecs as a pipeline.

`read_array_metadata_v3` returns everything one read of a document finds:
each field with where it sits, the kind it was read as and what the scope
made of it, the chunks the codecs are handed, each codec with the chunk it
is handed, every problem, and the document's model when there is none.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

from zarr_metadata._json import arrays_to_tuples
from zarr_metadata.model import (
    UNSET,
    ValidationProblem,
    ZarrV3ArrayMetadata,
    ZarrV3ArrayMetadataReading,
    read_array_metadata_v3,
    read_group_metadata_v3,
)
from zarr_metadata.v3._definition import (
    fields_of,
    with_problems,
)
from zarr_metadata.v3.definition import (
    CORE,
    CORE_AND_EXTENSIONS,
    AcceptedField,
    ChunkGridDefinition,
    ChunkKeyEncodingDefinition,
    CodecDefinition,
    Context,
    DataTypeDefinition,
    Definition,
    RefusedField,
    StorageTransformerDefinition,
    UnclaimedField,
    resolve,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from zarr_metadata.v3.definition import Lengths, Loc, ResolvedField

LITTLE = {"name": "bytes", "configuration": {"endian": "little"}}
ZSTD = {"name": "zstd", "configuration": {"level": 1}}

POINTS: list[tuple[Loc, type[Definition[Any]], type]] = [
    (("data_type",), DataTypeDefinition, AcceptedField),
    (("chunk_grid",), ChunkGridDefinition, AcceptedField),
    (("chunk_key_encoding",), ChunkKeyEncodingDefinition, AcceptedField),
]
"""A default document's single extension points, each read where it sits."""


def _document(shape: tuple[int, ...] = (4,), **fields: object) -> dict[str, Any]:
    """A default v3 array document of `shape` with `fields` in it, arrays as tuples as a reader's are."""
    document = {**ZarrV3ArrayMetadata.create_default(shape=shape).to_json(), **fields}
    return cast("dict[str, Any]", arrays_to_tuples(document))


def _codec(loc: Loc, variant: type = AcceptedField) -> tuple[Loc, type[Definition[Any]], type]:
    return (loc, CodecDefinition, variant)


def _shard(chunk_shape: list[int], codecs: list[object] | None = None, **more: object) -> object:
    configuration = {
        "chunk_shape": chunk_shape,
        "codecs": [LITTLE] if codecs is None else codecs,
        "index_codecs": [LITTLE],
        **more,
    }
    return {"name": "sharding_indexed", "configuration": configuration}


@pytest.mark.parametrize(
    ("document", "context", "fields", "handed"),
    [
        (_document(), CORE_AND_EXTENSIONS, [*POINTS, _codec(("codecs", 0))], [(frozenset({4}),)]),
        # Each codec is handed the chunk the one before it hands on.
        (
            _document(
                (4, 6), codecs=[{"name": "transpose", "configuration": {"order": [1, 0]}}, "bytes"]
            ),
            CORE_AND_EXTENSIONS,
            [*POINTS, _codec(("codecs", 0)), _codec(("codecs", 1))],
            [(frozenset({4}), frozenset({6})), (frozenset({6}), frozenset({4}))],
        ),
        # The fields a shard holds come after it, where they sit in its
        # configuration; read in the core spec's scope, an extension inside
        # it is out of scope where it sits.
        (
            _document(
                (16, 16),
                chunk_grid={"name": "regular", "configuration": {"chunk_shape": [8, 8]}},
                codecs=[_shard([4, 4], codecs=[LITTLE, ZSTD], index_codecs=[LITTLE, "crc32c"])],
            ),
            CORE,
            [
                *POINTS,
                _codec(("codecs", 0)),
                _codec(("codecs", 0, "configuration", "codecs", 0)),
                _codec(("codecs", 0, "configuration", "codecs", 1), UnclaimedField),
                _codec(("codecs", 0, "configuration", "index_codecs", 0)),
                _codec(("codecs", 0, "configuration", "index_codecs", 1)),
            ],
            [(frozenset({8}), frozenset({8}))],
        ),
        # A shard within a shard: each field, then the fields it holds.
        (
            _document(
                (16, 16),
                chunk_grid={"name": "regular", "configuration": {"chunk_shape": [8, 8]}},
                codecs=[_shard([4, 4], codecs=[_shard([2, 2])])],
            ),
            CORE_AND_EXTENSIONS,
            [
                *POINTS,
                _codec(("codecs", 0)),
                _codec(("codecs", 0, "configuration", "codecs", 0)),
                _codec(("codecs", 0, "configuration", "codecs", 0, "configuration", "codecs", 0)),
                _codec(
                    ("codecs", 0, "configuration", "codecs", 0, "configuration", "index_codecs", 0)
                ),
                _codec(("codecs", 0, "configuration", "index_codecs", 0)),
            ],
            [(frozenset({8}), frozenset({8}))],
        ),
        # A shard its check refuses keeps the fields it read inside it.
        (
            _document(
                (16, 16),
                chunk_grid={"name": "regular", "configuration": {"chunk_shape": [8, 8]}},
                codecs=[_shard([4, 4], index_location="middle", codecs=[LITTLE, ZSTD])],
            ),
            CORE,
            [
                *POINTS,
                _codec(("codecs", 0), RefusedField),
                _codec(("codecs", 0, "configuration", "codecs", 0)),
                _codec(("codecs", 0, "configuration", "codecs", 1), UnclaimedField),
                _codec(("codecs", 0, "configuration", "index_codecs", 0)),
            ],
            [(frozenset({8}), frozenset({8}))],
        ),
        # A struct's field types are data types, whether or not anything
        # in scope claims them.
        (
            _document(
                data_type={
                    "name": "struct",
                    "configuration": {
                        "fields": [
                            {"name": "a", "data_type": "int8"},
                            {"name": "b", "data_type": "acme.t"},
                        ]
                    },
                },
                fill_value={"a": 0, "b": "anything"},
                codecs=[LITTLE],
            ),
            CORE_AND_EXTENSIONS,
            [
                POINTS[0],
                (
                    ("data_type", "configuration", "fields", 0, "data_type"),
                    DataTypeDefinition,
                    AcceptedField,
                ),
                (
                    ("data_type", "configuration", "fields", 1, "data_type"),
                    DataTypeDefinition,
                    UnclaimedField,
                ),
                *POINTS[1:],
                _codec(("codecs", 0)),
            ],
            [(frozenset({4}),)],
        ),
        # A field its definition refuses, or nothing claims, is read as
        # the kind it sits in; a grid nothing claims says no lengths, and
        # a codec after the array -> bytes codec is handed bytes.
        (
            _document(
                chunk_grid={"name": "acme.grid", "configuration": {}},
                codecs=["bytes", {"name": "gzip", "configuration": {"level": 99}}],
                storage_transformers=[{"name": "acme.transformer"}],
            ),
            CORE_AND_EXTENSIONS,
            [
                POINTS[0],
                (("chunk_grid",), ChunkGridDefinition, UnclaimedField),
                POINTS[2],
                _codec(("codecs", 0)),
                _codec(("codecs", 1), RefusedField),
                (("storage_transformers", 0), StorageTransformerDefinition, UnclaimedField),
            ],
            [(None,), None],
        ),
        # A data type that is not JSON is refused.
        (
            _document(data_type=float("nan")),
            CORE_AND_EXTENSIONS,
            [
                (("data_type",), DataTypeDefinition, RefusedField),
                *POINTS[1:],
                _codec(("codecs", 0)),
            ],
            [(frozenset({4}),)],
        ),
    ],
    ids=[
        "default",
        "transposed",
        "a-shard-holding-an-extension-in-the-core-scope",
        "a-shard-within-a-shard",
        "a-shard-its-check-refuses",
        "a-struct-s-field-types",
        "fields-read-as-nothing",
        "a-data-type-that-is-not-json",
    ],
)
def test_a_document_reads_as_each_field_where_it_sits_and_its_codecs_as_a_pipeline(
    document: dict[str, Any],
    context: Context,
    fields: list[tuple[Loc, type[Definition[Any]], type]],
    handed: list[Lengths | None],
) -> None:
    reading = read_array_metadata_v3(document, context=context)
    # A model only of a document with no problem.
    assert (reading.metadata is None) is (reading.problems != ())
    assert [(loc, field.read_as, type(field)) for loc, field in reading.fields()] == fields
    # The chunk's vocabulary for what nothing says is None, the reading's
    # for a key the document lacks is `UNSET`.
    if reading.data_type is UNSET:
        assert reading.chunk.data_type is None
    else:
        assert reading.chunk.data_type is reading.data_type
    assert reading.pipeline[0].incoming == reading.chunk
    assert [None if s.incoming is None else s.incoming.lengths for s in reading.pipeline] == handed


_INNER_GZIP = {"name": "gzip", "configuration": {"level": 12}}
_WITH_PROBLEMS = _document(
    (4, 4),
    chunk_grid={"name": "regular", "configuration": {"chunk_shape": [4]}},
    fill_value="high",
    codecs=[
        _shard([2, 2], [LITTLE, _INNER_GZIP]),
        ZSTD,
        {"name": "transpose", "configuration": {"order": [1, 0]}},
    ],
)
"""An array document with a problem in each place a field can have one, and one in no field."""


def _array_fields(
    document: object,
) -> Iterator[tuple[Loc, ResolvedField[Any], tuple[ValidationProblem, ...]]]:
    reading = read_array_metadata_v3(document)
    return with_problems(reading.fields(), reading.problems)


def _group_fields(
    document: object,
) -> Iterator[tuple[Loc, ResolvedField[Any], tuple[ValidationProblem, ...]]]:
    group = {
        "zarr_format": 3,
        "node_type": "group",
        "consolidated_metadata": {
            "kind": "inline",
            "must_understand": False,
            "metadata": {"a": document},
        },
    }
    reading = read_group_metadata_v3(group)
    return with_problems(reading.fields(), reading.problems)


def _field_fields(
    document: object,
) -> Iterator[tuple[Loc, ResolvedField[Any], tuple[ValidationProblem, ...]]]:
    resolved, problems = resolve(
        cast("dict[str, Any]", document)["codecs"][0],
        CodecDefinition,
        CORE_AND_EXTENSIONS,
        ("codecs", 0),
    )
    return with_problems(fields_of(resolved, ("codecs", 0)), problems)


_INNER_LEVEL = ("codecs", 0, "configuration", "codecs", 1, "configuration", "level")
_EACH_FIELD_S = {
    ("data_type",): [],
    ("chunk_grid",): [("chunk_grid", "configuration", "chunk_shape")],
    ("chunk_key_encoding",): [],
    ("codecs", 0): [_INNER_LEVEL],
    ("codecs", 0, "configuration", "codecs", 0): [],
    ("codecs", 0, "configuration", "codecs", 1): [_INNER_LEVEL],
    ("codecs", 0, "configuration", "index_codecs", 0): [],
    ("codecs", 1): [],
    ("codecs", 2): [("codecs", 2)],
}
"""Each field of `_WITH_PROBLEMS`, and where each of its problems is."""
_IN_A_GROUP = ("consolidated_metadata", "metadata", "a")


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        (_array_fields, _EACH_FIELD_S),
        (
            _group_fields,
            {
                (*_IN_A_GROUP, *loc): [(*_IN_A_GROUP, *problem) for problem in problems]
                for loc, problems in _EACH_FIELD_S.items()
            },
        ),
        (
            _field_fields,
            {loc: problems for loc, problems in _EACH_FIELD_S.items() if loc[:2] == ("codecs", 0)},
        ),
    ],
    ids=["an-array", "a-document-a-group-holds", "one-field"],
)
def test_each_field_comes_with_the_problems_located_in_it(
    fields: Callable[
        [object], Iterator[tuple[Loc, ResolvedField[Any], tuple[ValidationProblem, ...]]]
    ],
    expected: dict[Loc, list[Loc]],
) -> None:
    # Those it was read with and those the document found with it where
    # it stands -- a transpose out of the pipeline's order, at its own
    # place, and the chunk grid over the shape -- the fields it holds
    # too: a shard with a bad inner codec has that problem as well. A
    # problem in no field -- the fill value -- is in none's.
    assert {
        loc: [p.loc for p in problems] for loc, _, problems in fields(_WITH_PROBLEMS)
    } == expected


def test_a_document_with_no_problem_reads_as_its_model_holding_the_fields_read() -> None:
    # The fields the read made, not a second reading of them.
    document = _document(codecs=[{"name": "transpose", "configuration": {"order": [0]}}, LITTLE])
    reading = read_array_metadata_v3(document)
    model = reading.metadata
    assert reading.problems == ()
    assert model is not None
    assert model == ZarrV3ArrayMetadata.from_json(document)
    assert model.data_type is reading.data_type
    assert model.chunk_grid is reading.chunk_grid
    assert model.chunk_key_encoding is reading.chunk_key_encoding
    assert all(
        codec is stage.codec for codec, stage in zip(model.codecs, reading.pipeline, strict=True)
    )


def test_error_a_value_that_is_not_a_mapping_reads_as_nothing() -> None:
    reading = read_array_metadata_v3(["not", "a", "document"])
    not_an_object = ValidationProblem((), "expected an object", "invalid_type")
    assert reading == ZarrV3ArrayMetadataReading(problems=(not_an_object,))
    assert list(reading.fields()) == []


@pytest.mark.parametrize(
    ("member", "attribute"),
    [("codecs", "pipeline"), ("storage_transformers", "storage_transformers")],
)
def test_error_a_list_of_fields_that_is_not_a_list_reads_as_empty(
    member: str, attribute: str
) -> None:
    document = _document()
    document[member] = "bytes"
    reading = read_array_metadata_v3(document)
    assert getattr(reading, attribute) == ()
    assert [(p.loc, p.kind) for p in reading.problems] == [((member,), "invalid_type")]


@pytest.mark.parametrize("member", ["data_type", "chunk_grid", "chunk_key_encoding"])
def test_error_a_document_without_a_field_reads_it_as_unset(member: str) -> None:
    document = _document()
    del document[member]
    reading = read_array_metadata_v3(document)
    assert getattr(reading, member) is UNSET
    assert (member,) not in [loc for loc, _ in reading.fields()]
    assert ValidationProblem((member,), "missing required key", "missing_key") in reading.problems


def test_error_a_document_without_a_shape_hands_its_codecs_chunks_of_no_known_rank() -> None:
    document = _document()
    del document["shape"]
    reading = read_array_metadata_v3(document)
    assert reading.chunk.lengths is None
