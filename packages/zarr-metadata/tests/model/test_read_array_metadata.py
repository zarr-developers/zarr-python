"""A v3 array document, read: each field as a scope read it, and its codecs as a pipeline.

`read_array_metadata_v3` returns what the validator read to find what is
wrong with a document, beside the problems it found: each field with
where it sits and the kind it was read as, the chunks the codecs are
handed, and each codec with the chunk it is handed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

from zarr_metadata._json import arrays_to_tuples
from zarr_metadata.model import (
    ValidationProblem,
    ZarrV3ArrayMetadata,
    ZarrV3ArrayMetadataReading,
    read_array_metadata_v3,
    validate_array_metadata_v3,
)
from zarr_metadata.v3.definition import (
    CORE,
    CORE_AND_EXTENSIONS,
    ChunkGridDefinition,
    ChunkKeyEncodingDefinition,
    CodecDefinition,
    Context,
    DataTypeDefinition,
    Definition,
    StorageTransformerDefinition,
)

if TYPE_CHECKING:
    from zarr_metadata.v3.definition import Lengths, Loc

LITTLE = {"name": "bytes", "configuration": {"endian": "little"}}
ZSTD = {"name": "zstd", "configuration": {"level": 1}}

POINTS: list[tuple[Loc, type[Definition[Any]], str]] = [
    (("data_type",), DataTypeDefinition, "read"),
    (("chunk_grid",), ChunkGridDefinition, "read"),
    (("chunk_key_encoding",), ChunkKeyEncodingDefinition, "read"),
]
"""A default document's single extension points, each read where it sits."""


def _document(shape: tuple[int, ...] = (4,), **fields: object) -> dict[str, Any]:
    """A default v3 array document of `shape` with `fields` in it, arrays as tuples as a reader's are."""
    document = {**ZarrV3ArrayMetadata.create_default(shape=shape).to_json(), **fields}
    return cast("dict[str, Any]", arrays_to_tuples(document))


def _codec(loc: Loc, resolution: str = "read") -> tuple[Loc, type[Definition[Any]], str]:
    return (loc, CodecDefinition, resolution)


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
                _codec(("codecs", 0, "configuration", "codecs", 1), "out_of_scope"),
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
                _codec(("codecs", 0), "invalid"),
                _codec(("codecs", 0, "configuration", "codecs", 0)),
                _codec(("codecs", 0, "configuration", "codecs", 1), "out_of_scope"),
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
                    "read",
                ),
                (
                    ("data_type", "configuration", "fields", 1, "data_type"),
                    DataTypeDefinition,
                    "out_of_scope",
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
                (("chunk_grid",), ChunkGridDefinition, "out_of_scope"),
                POINTS[2],
                _codec(("codecs", 0)),
                _codec(("codecs", 1), "invalid"),
                (("storage_transformers", 0), StorageTransformerDefinition, "out_of_scope"),
            ],
            [(None,), None],
        ),
        # A data type that is not JSON is read as nothing.
        (
            _document(data_type=float("nan")),
            CORE_AND_EXTENSIONS,
            [(("data_type",), DataTypeDefinition, "invalid"), *POINTS[1:], _codec(("codecs", 0))],
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
    fields: list[tuple[Loc, type[Definition[Any]], str]],
    handed: list[Lengths | None],
) -> None:
    reading, problems = read_array_metadata_v3(document, context=context)
    assert problems == validate_array_metadata_v3(document, context=context)
    assert [(loc, field.read_as, field.resolution) for loc, field in reading.fields()] == fields
    assert reading.chunk.data_type is reading.data_type
    assert reading.pipeline[0].incoming == reading.chunk
    assert [None if s.incoming is None else s.incoming.lengths for s in reading.pipeline] == handed


def test_error_a_value_that_is_not_a_mapping_reads_as_nothing() -> None:
    reading, problems = read_array_metadata_v3(["not", "a", "document"])
    assert reading == ZarrV3ArrayMetadataReading()
    assert list(reading.fields()) == []
    assert problems == (ValidationProblem((), "expected an object", "invalid_type"),)


@pytest.mark.parametrize(
    ("member", "attribute"),
    [("codecs", "pipeline"), ("storage_transformers", "storage_transformers")],
)
def test_error_a_list_of_fields_that_is_not_a_list_reads_as_empty(
    member: str, attribute: str
) -> None:
    document = _document()
    document[member] = "bytes"
    reading, problems = read_array_metadata_v3(document)
    assert getattr(reading, attribute) == ()
    assert [(p.loc, p.kind) for p in problems] == [((member,), "invalid_type")]


@pytest.mark.parametrize("member", ["data_type", "chunk_grid", "chunk_key_encoding"])
def test_error_a_document_without_a_field_reads_it_as_none(member: str) -> None:
    document = _document()
    del document[member]
    reading, problems = read_array_metadata_v3(document)
    assert getattr(reading, member) is None
    assert (member,) not in [loc for loc, _ in reading.fields()]
    assert ValidationProblem((member,), "missing required key", "missing_key") in problems


def test_error_a_document_without_a_shape_hands_its_codecs_chunks_of_no_known_rank() -> None:
    document = _document()
    del document["shape"]
    reading, _ = read_array_metadata_v3(document)
    assert reading.chunk.lengths is None
