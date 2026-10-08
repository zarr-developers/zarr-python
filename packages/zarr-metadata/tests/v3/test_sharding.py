"""Sharding, read: the shard it is handed, and the two pipelines it holds.

`sharding_indexed` divides the chunk it is handed into inner chunks, which
its inner codecs are handed, and indexes them in a shard index, which its
index codecs are handed; each pipeline is read as the array's is.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from zarr_metadata.model import (
    ZarrV3ArrayMetadata,
    validate_array_metadata_v3,
)
from zarr_metadata.v3._pipeline import (
    read_pipeline,
)
from zarr_metadata.v3.definition import (
    CORE_AND_EXTENSIONS,
    Chunk,
    CodecDefinition,
    DataTypeDefinition,
    JSONValue,
    Resolved,
    resolve,
)

if TYPE_CHECKING:
    from zarr_metadata.v3.definition import Stage, ValidationProblem

Where = tuple[str | int, ...]

LITTLE: JSONValue = {"name": "bytes", "configuration": {"endian": "little"}}
INDEX: list[JSONValue] = [LITTLE, "crc32c"]


def _dt(name: JSONValue) -> Resolved[DataTypeDefinition[Any]]:
    return resolve(name, DataTypeDefinition, CORE_AND_EXTENSIONS)[0]


FLOAT32 = _dt("float32")
UINT64 = _dt("uint64")


def _chunk(*axes: set[int] | None, data_type: Resolved[DataTypeDefinition[Any]] = FLOAT32) -> Chunk:
    return Chunk(tuple(None if axis is None else frozenset(axis) for axis in axes), data_type)


def _index(*axes: set[int] | None) -> Chunk:
    return _chunk(*axes, data_type=UINT64)


def _shard(
    chunk_shape: list[int],
    codecs: list[JSONValue] | None = None,
    index: list[JSONValue] | None = None,
) -> JSONValue:
    configuration: dict[str, JSONValue] = {
        "chunk_shape": list[JSONValue](chunk_shape),
        "codecs": [LITTLE] if codecs is None else codecs,
        "index_codecs": INDEX if index is None else index,
    }
    return {"name": "sharding_indexed", "configuration": configuration}


def _transpose(*order: int) -> JSONValue:
    return {"name": "transpose", "configuration": {"order": list(order)}}


def _read(
    codecs: list[JSONValue], chunk: Chunk
) -> tuple[tuple[Stage, ...], tuple[ValidationProblem, ...]]:
    """The stages, and every problem: each codec's own, where it sits, then the pipeline's."""
    read = [
        resolve(codec, CodecDefinition, CORE_AND_EXTENSIONS, (index,))
        for index, codec in enumerate(codecs)
    ]
    stages, found = read_pipeline([resolved for resolved, _ in read], chunk)
    return stages, (*(problem for _, own in read for problem in own), *found)


def _problems(codecs: list[JSONValue], chunk: Chunk) -> list[tuple[Where, str]]:
    return [(problem.loc, problem.kind) for problem in _read(codecs, chunk)[1]]


def _handed(stages: tuple[Stage, ...], at: Where = ()) -> dict[Where, Chunk | None]:
    """What each codec of each pipeline the codecs hold is handed, by where it sits."""
    found: dict[Where, Chunk | None] = {}
    for index, stage in enumerate(stages):
        for member, held in stage.inner.items():
            found |= {(*at, index, member, place): s.incoming for place, s in enumerate(held)}
            found |= _handed(held, (*at, index, member))
    return found


@pytest.mark.parametrize(
    ("codecs", "chunk", "handed"),
    [
        # Chunks of 8 by 8, shards of 2 by 2 inner chunks.
        (
            [_shard([4, 4])],
            _chunk({8}, {8}),
            {
                (0, "codecs", 0): _chunk({4}, {4}),
                (0, "index_codecs", 0): _index({2}, {2}, {2}),
                (0, "index_codecs", 1): None,
            },
        ),
        # The shard is the chunk it is handed: transposed, 4 by 6 divides
        # into inner chunks of 4 by 3 though 6 by 4 would not.
        (
            [_transpose(1, 0), _shard([4, 3])],
            _chunk({6}, {4}),
            {
                (1, "codecs", 0): _chunk({4}, {3}),
                (1, "index_codecs", 0): _index({1}, {2}, {2}),
                (1, "index_codecs", 1): None,
            },
        ),
        # A rectilinear grid's shards differ; so do their counts of inner
        # chunks.
        (
            [_shard([4])],
            _chunk({8, 4}),
            {
                (0, "codecs", 0): _chunk({4}),
                (0, "index_codecs", 0): _index({2, 1}, {2}),
                (0, "index_codecs", 1): None,
            },
        ),
        # Handed a chunk nothing is known of, a shard still has the inner
        # chunks it declares, and an index of uint64.
        (
            [_shard([4])],
            Chunk(),
            {
                (0, "codecs", 0): Chunk((frozenset({4}),)),
                (0, "index_codecs", 0): _index(None, {2}),
                (0, "index_codecs", 1): None,
            },
        ),
        # A shard within a shard is read the same way.
        (
            [_shard([4, 4], codecs=[_shard([2, 2])])],
            _chunk({8}, {8}),
            {
                (0, "codecs", 0): _chunk({4}, {4}),
                (0, "codecs", 0, "codecs", 0): _chunk({2}, {2}),
                (0, "codecs", 0, "index_codecs", 0): _index({2}, {2}, {2}),
                (0, "codecs", 0, "index_codecs", 1): None,
                (0, "index_codecs", 0): _index({2}, {2}, {2}),
                (0, "index_codecs", 1): None,
            },
        ),
    ],
)
def test_every_shard_hands_its_inner_codecs_its_inner_chunks_and_its_index_codecs_its_index(
    codecs: list[JSONValue], chunk: Chunk, handed: dict[Where, Chunk | None]
) -> None:
    stages, problems = _read(codecs, chunk)
    assert problems == ()
    assert _handed(stages) == handed


def test_error_a_shard_whose_inner_chunks_have_another_number_of_axes() -> None:
    # One problem: which of the two is wrong is not known, so what the
    # pipelines are handed is not either, and they are judged against
    # nothing -- these transposes fit once `chunk_shape` has two lengths.
    shard = _shard([4], codecs=[_transpose(1, 0), LITTLE], index=[_transpose(2, 1, 0), LITTLE])
    assert _problems([shard], _chunk({8}, {8})) == [
        ((0, "configuration", "chunk_shape"), "invalid_value")
    ]


@pytest.mark.parametrize(
    ("codecs", "chunk", "axis"),
    [
        ([_shard([4, 4])], _chunk({8}, {6}), 1),
        ([_shard([4])], _chunk({8, 6}), 0),
        # Divided along its axes as it is handed them: transposed, a shard
        # of 6 by 2 does not divide into inner chunks of 2 by 3, though 2
        # by 6 would.
        ([_transpose(1, 0), _shard([2, 3])], _chunk({2}, {6}), 1),
    ],
)
def test_error_an_inner_chunk_length_that_does_not_divide_the_shard(
    codecs: list[JSONValue], chunk: Chunk, axis: int
) -> None:
    at = len(codecs) - 1
    assert _problems(codecs, chunk) == [
        ((at, "configuration", "chunk_shape", axis), "invalid_value")
    ]


@pytest.mark.parametrize(
    ("shard", "found"),
    [
        # Each pipeline is read as the array's is, where it sits.
        (_shard([4, 4], codecs=[]), [((0, "configuration", "codecs"), "invalid_value")]),
        (
            _shard([4, 4], codecs=["crc32c", LITTLE]),
            [((0, "configuration", "codecs", 1), "invalid_value")],
        ),
        (
            _shard([4, 4], codecs=["bytes"]),
            [((0, "configuration", "codecs", 0, "configuration", "endian"), "missing_key")],
        ),
        # The index is of uint64, whose values take several bytes, and has
        # an axis more than the shard.
        (
            _shard([4, 4], index=["bytes"]),
            [((0, "configuration", "index_codecs", 0, "configuration", "endian"), "missing_key")],
        ),
        (
            _shard([4, 4], index=[_transpose(1, 0), LITTLE]),
            [((0, "configuration", "index_codecs", 0, "configuration", "order"), "invalid_value")],
        ),
    ],
)
def test_error_a_pipeline_a_shard_holds_that_does_not_fit_what_it_is_handed(
    shard: JSONValue, found: list[tuple[Where, str]]
) -> None:
    assert _problems([shard], _chunk({8}, {8})) == found


def test_error_an_array_document_s_shard_is_read_where_it_sits() -> None:
    document = dict(ZarrV3ArrayMetadata.create_default(shape=(16, 16)).to_json())
    document["chunk_grid"] = {"name": "regular", "configuration": {"chunk_shape": [8, 6]}}
    document["codecs"] = [_shard([4, 4], index=["bytes"])]
    assert [(p.loc, p.kind) for p in validate_array_metadata_v3(document)] == [
        (("codecs", 0, "configuration", "chunk_shape", 1), "invalid_value"),
        (
            ("codecs", 0, "configuration", "index_codecs", 0, "configuration", "endian"),
            "missing_key",
        ),
    ]
