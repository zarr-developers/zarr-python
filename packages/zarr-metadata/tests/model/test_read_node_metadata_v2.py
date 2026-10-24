"""The v2 readers that return problems instead of raising: the group reader, the consolidated-metadata reader, and the node reader, which reads a node's files from a store."""

from __future__ import annotations

import json
from typing import Any, cast

import pytest

from zarr_metadata.model import (
    ZarrV2ArrayMetadata,
    ZarrV2ArrayMetadataReading,
    ZarrV2ConsolidatedMetadata,
    ZarrV2GroupMetadata,
    ZarrV2GroupMetadataReading,
    ZarrV2UnknownNodeMetadataReading,
    node_metadata_from_key_value_v2,
    read_consolidated_metadata_v2,
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


def _files(**files: object) -> dict[str, object]:
    """A node's files as decoded JSON, keyed by file name: `_zarray=...` is `.zarray`."""
    return {key.replace("_", "."): value for key, value in files.items()}


def _store(**files: object) -> dict[str, bytes]:
    """The same files as a store holds them: keys and bytes."""
    return {key: json.dumps(value).encode() for key, value in _files(**files).items()}


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
def test_a_v2_node_is_read_as_the_file_it_has(
    files: dict[str, object], kind: type, model: object
) -> None:
    """A v2 node is an array when its files include `.zarray` and a group when they include `.zgroup`, with the attributes at `.zattrs`. `read_node_metadata_v2` reads the decoded files and gives the reading of that node type, whose model is what the model's `from_key_value` builds from the same files in a store; `node_metadata_from_key_value_v2` returns that model from the store, and `validate_node_metadata_v2` finds no problems."""
    reading = read_node_metadata_v2(_files(**files))
    assert type(reading) is kind
    assert reading.problems == ()
    assert reading.metadata == cast("Any", model).from_key_value(_store(**files))
    assert node_metadata_from_key_value_v2(_store(**files)) == reading.metadata
    assert validate_node_metadata_v2(_files(**files)) == ()


def test_a_v2_node_reads_the_same_from_zmetadata_entries_as_from_a_store() -> None:
    """The entries of a `.zmetadata` hold the same JSON as the node's own files, so the slice of entries for one path is a node's files: `read_node_metadata_v2` reads it as it reads the decoded store files, and builds the same model."""
    entries = {"a/.zarray": ZARRAY, "a/.zattrs": {"x": 1}, "b/.zgroup": {"zarr_format": 2}}
    node_files = {
        key.removeprefix("a/"): value for key, value in entries.items() if key.startswith("a/")
    }
    reading = read_node_metadata_v2(node_files)
    assert reading.metadata == node_metadata_from_key_value_v2(
        _store(_zarray=ZARRAY, _zattrs={"x": 1})
    )


def test_error_files_with_no_node_file_are_no_node() -> None:
    """Files with neither `.zarray` nor `.zgroup` describe no node: the reading is a `ZarrV2UnknownNodeMetadataReading` with one `missing_key` problem, and building a model from such a store raises."""
    reading = read_node_metadata_v2(_files(_zattrs={"a": 1}))
    assert type(reading) is ZarrV2UnknownNodeMetadataReading
    assert [(p.loc, p.kind) for p in reading.problems] == [((), "missing_key")]
    assert reading.metadata is None
    with pytest.raises(MetadataValidationError):
        node_metadata_from_key_value_v2(_store(_zattrs={"a": 1}))


def test_error_files_with_both_node_files_are_no_node() -> None:
    """A node is an array or a group, not both: files with `.zarray` and `.zgroup` give a `ZarrV2UnknownNodeMetadataReading` with an `invalid_value` problem at `.zgroup`."""
    reading = read_node_metadata_v2(_files(_zarray=ZARRAY, _zgroup={"zarr_format": 2}))
    assert type(reading) is ZarrV2UnknownNodeMetadataReading
    assert [(p.loc, p.kind) for p in reading.problems] == [((".zgroup",), "invalid_value")]


def test_error_a_store_file_that_is_not_json_raises_when_building_a_node() -> None:
    """Decoding is the store's step: a store file whose bytes are not JSON makes `node_metadata_from_key_value_v2` raise with an `invalid_json` problem at the file."""
    with pytest.raises(MetadataValidationError) as raised:
        node_metadata_from_key_value_v2({".zarray": b"{not json"})
    assert [(p.loc, p.kind) for p in raised.value.problems] == [((".zarray",), "invalid_json")]


def test_error_a_v2_node_with_a_bad_document_reports_its_problems() -> None:
    """A node whose document has a problem gives the reading of its node type with the problems, as the model's reader reports them, and no model."""
    files = _files(_zarray={**ZARRAY, "shape": [-1]})
    reading = read_node_metadata_v2(files)
    assert type(reading) is ZarrV2ArrayMetadataReading
    assert [(p.loc, p.kind) for p in reading.problems] == [(("shape",), "invalid_value")]
    assert validate_node_metadata_v2(files) == reading.problems


CONSOLIDATED: dict[str, Any] = {
    "zarr_consolidated_format": 1,
    "metadata": {
        ".zgroup": {"zarr_format": 2},
        "a/.zarray": ZARRAY,
        "a/.zattrs": {"x": 1},
    },
}


def test_read_consolidated_metadata_v2_reads_a_zmetadata() -> None:
    """`read_consolidated_metadata_v2` returns a reading with no problems whose model equals the one the constructor builds, and the model's `reading` is that reading. `fields()` gives each node's fields located in the `.zmetadata`."""
    reading = read_consolidated_metadata_v2(CONSOLIDATED)
    assert reading.problems == ()
    assert reading.metadata == ZarrV2ConsolidatedMetadata(CONSOLIDATED)
    assert reading.metadata is not None
    assert reading.metadata.reading == reading
    assert [loc for loc, _ in reading.fields()] == [("metadata", "a/.zarray", "dtype")]


def test_error_read_consolidated_metadata_v2_reports_problems_instead_of_raising() -> None:
    """A `.zmetadata` with a problem gives a reading with every problem and no model, where the constructor raises."""
    bad = {
        **CONSOLIDATED,
        "metadata": {**CONSOLIDATED["metadata"], "a/.zarray": {**ZARRAY, "shape": [-1]}},
    }
    reading = read_consolidated_metadata_v2(bad)
    assert [(p.loc, p.kind) for p in reading.problems] == [
        (("metadata", "a/.zarray", "shape"), "invalid_value")
    ]
    assert reading.metadata is None
    with pytest.raises(MetadataValidationError):
        ZarrV2ConsolidatedMetadata(bad)
