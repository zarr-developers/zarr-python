"""The v2 group reader and the v2 node reader, which reads a node's files from a store."""

from __future__ import annotations

import json
from typing import Any, cast

import pytest

from zarr_metadata.model import (
    ZarrV2ArrayMetadata,
    ZarrV2ArrayMetadataReading,
    ZarrV2GroupMetadata,
    ZarrV2GroupMetadataReading,
    ZarrV2UnknownNodeMetadataReading,
    node_metadata_from_key_value_v2,
    read_group_metadata_v2,
    read_node_metadata_v2,
    validate_node_metadata_v2,
)
from zarr_metadata.problems import MetadataValidationError

ZARRAY: dict[str, Any] = {
    "zarr_format": 2,
    "shape": [4],
    "chunks": [2],
    "dtype": "<f4",
    "compressor": None,
    "fill_value": 0,
    "order": "C",
    "filters": None,
}


def _store(**files: object) -> dict[str, bytes]:
    return {key.replace("_", "."): json.dumps(value).encode() for key, value in files.items()}


def test_read_group_metadata_v2_reads_a_group() -> None:
    """`read_group_metadata_v2` returns a reading with no problems, whose model equals the group read from the same document, and the model's `reading` is that reading. A group has no fields for a scope to read."""
    document = {"zarr_format": 2, "attributes": {"a": 1}}
    reading = read_group_metadata_v2(document)
    assert reading.problems == ()
    assert reading.metadata == ZarrV2GroupMetadata(document)
    assert reading.metadata is not None
    assert reading.metadata.reading == reading
    assert list(reading.fields()) == []


def test_error_read_group_metadata_v2_reports_the_problems_of_a_bad_group() -> None:
    """A group document with a problem gives a reading that holds the problems and no model."""
    reading = read_group_metadata_v2({"zarr_format": 2, "extra": 1})
    assert [(p.loc, p.kind) for p in reading.problems] == [(("extra",), "unknown_key")]
    assert reading.metadata is None


@pytest.mark.parametrize(
    ("files", "kind", "model"),
    [
        ({"_zarray": ZARRAY}, ZarrV2ArrayMetadataReading, ZarrV2ArrayMetadata),
        (
            {"_zarray": ZARRAY, "_zattrs": {"a": 1}},
            ZarrV2ArrayMetadataReading,
            ZarrV2ArrayMetadata,
        ),
        ({"_zgroup": {"zarr_format": 2}}, ZarrV2GroupMetadataReading, ZarrV2GroupMetadata),
        (
            {"_zgroup": {"zarr_format": 2}, "_zattrs": {"a": 1}},
            ZarrV2GroupMetadataReading,
            ZarrV2GroupMetadata,
        ),
    ],
    ids=["array", "array-with-attributes", "group", "group-with-attributes"],
)
def test_a_v2_node_is_read_as_the_file_its_store_holds(
    files: dict[str, object], kind: type, model: object
) -> None:
    """A v2 node is an array when the store holds `.zarray` and a group when it holds `.zgroup`, with the attributes at `.zattrs`. The reader gives the reading of that node type, whose model is what the model's `from_key_value` builds; `node_metadata_from_key_value_v2` returns that model, and `validate_node_metadata_v2` finds no problems."""
    store = _store(**files)
    reading = read_node_metadata_v2(store)
    assert type(reading) is kind
    assert reading.problems == ()
    assert reading.metadata == cast("Any", model).from_key_value(store)
    assert node_metadata_from_key_value_v2(store) == reading.metadata
    assert validate_node_metadata_v2(store) == ()


def test_error_a_v2_store_with_no_node_file_is_no_node() -> None:
    """A store with neither `.zarray` nor `.zgroup` holds no node: the reading is a `ZarrV2UnknownNodeMetadataReading` with one `missing_key` problem, and building a model raises."""
    store = _store(_zattrs={"a": 1})
    reading = read_node_metadata_v2(store)
    assert type(reading) is ZarrV2UnknownNodeMetadataReading
    assert [(p.loc, p.kind) for p in reading.problems] == [((), "missing_key")]
    assert reading.metadata is None
    with pytest.raises(MetadataValidationError):
        node_metadata_from_key_value_v2(store)


def test_error_a_v2_store_with_both_node_files_is_no_node() -> None:
    """A node is an array or a group, not both: a store with `.zarray` and `.zgroup` gives a `ZarrV2UnknownNodeMetadataReading` with an `invalid_value` problem at `.zgroup`."""
    reading = read_node_metadata_v2(_store(_zarray=ZARRAY, _zgroup={"zarr_format": 2}))
    assert type(reading) is ZarrV2UnknownNodeMetadataReading
    assert [(p.loc, p.kind) for p in reading.problems] == [((".zgroup",), "invalid_value")]


def test_error_a_v2_node_file_that_is_not_json_is_a_problem_of_its_node() -> None:
    """A node file whose bytes are not JSON gives the reading of its node type, with an `invalid_json` problem at the file and no model."""
    reading = read_node_metadata_v2({".zarray": b"{not json"})
    assert type(reading) is ZarrV2ArrayMetadataReading
    assert [(p.loc, p.kind) for p in reading.problems] == [((".zarray",), "invalid_json")]
    assert reading.metadata is None


def test_error_a_v2_node_with_a_bad_document_reports_its_problems() -> None:
    """A node whose document has a problem gives the reading of its node type with the problems, as the model's reader reports them, and no model."""
    reading = read_node_metadata_v2(_store(_zarray={**ZARRAY, "shape": [-1]}))
    assert type(reading) is ZarrV2ArrayMetadataReading
    assert [(p.loc, p.kind) for p in reading.problems] == [(("shape",), "invalid_value")]
    assert validate_node_metadata_v2(_store(_zarray={**ZARRAY, "shape": [-1]})) == reading.problems
