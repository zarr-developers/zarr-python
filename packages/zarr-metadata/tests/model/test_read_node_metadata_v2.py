"""The v2 readers that return problems instead of raising: the group reader, the consolidated-metadata reader, and the node reader, which reads a node's files from a store."""

from __future__ import annotations

import json
import pickle
from typing import Any, cast

import pytest

from zarr_metadata.definition import with_problems
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
    """A v2 node is an array when its files include `.zarray` and a group when they include `.zgroup`, with the attributes at `.zattrs`. `read_node_metadata_v2` reads the decoded files and gives the reading of that node type, whose model is what the model's `from_key_value` builds from the same files in a store. `node_metadata_from_key_value_v2` returns that model from the store."""
    reading = read_node_metadata_v2(_files(**files))
    assert type(reading) is kind
    assert reading.problems == ()
    assert reading.metadata == cast("Any", model).from_key_value(_store(**files))
    assert node_metadata_from_key_value_v2(_store(**files)) == reading.metadata


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


@pytest.mark.parametrize(
    ("name", "document"),
    [(".zarray", ZARRAY), (".zgroup", {"zarr_format": 2})],
    ids=["array", "group"],
)
def test_error_a_node_file_that_holds_attributes_is_refused_alike_in_a_store_and_a_zmetadata(
    name: str, document: dict[str, Any]
) -> None:
    """A `.zarray` or `.zgroup` that holds `attributes` itself is refused, with a message that says the attributes belong in `.zattrs`. `read_node_metadata_v2` and `read_consolidated_metadata_v2` report the same problem at the member in the file and in its entry. The kind is `invalid_value` for both files: the member is taken by the attributes, not unknown, and a caller who tolerates an unknown key would lose the attributes it holds."""
    held = {**document, "attributes": {}}
    from_files = read_node_metadata_v2({name: held}).problems
    from_entries = read_consolidated_metadata_v2(
        {
            "zarr_consolidated_format": 1,
            "metadata": {".zgroup": {"zarr_format": 2}, f"a/{name}": held},
        }
    ).problems
    assert [(p.loc, p.kind) for p in from_files] == [((name, "attributes"), "invalid_value")]
    assert [(p.loc, p.kind) for p in from_entries] == [
        (("metadata", f"a/{name}", "attributes"), "invalid_value")
    ]
    assert from_files[0].message == from_entries[0].message
    assert "belong in .zattrs" in from_files[0].message


@pytest.mark.parametrize(
    ("name", "document", "found"),
    [
        (
            ".zarray",
            {**ZARRAY, "dtype": "bogus", "order": "Q"},
            [
                ((".zarray", "attributes"), "invalid_value"),
                ((".zarray", "dtype"), "invalid_value"),
                ((".zarray", "order"), "invalid_value"),
            ],
        ),
        (
            ".zgroup",
            {"zarr_format": 3},
            [
                ((".zgroup", "attributes"), "invalid_value"),
                ((".zgroup", "zarr_format"), "invalid_value"),
            ],
        ),
    ],
    ids=["array", "group"],
)
def test_error_a_node_file_that_holds_attributes_still_reports_its_other_problems(
    name: str, document: dict[str, Any], found: list[tuple[tuple[str, ...], str]]
) -> None:
    """A `.zarray` or `.zgroup` that holds `attributes` does not hide its other problems: the rest of the file is still read, and its problems follow the one about `attributes`. `read_node_metadata_v2`, `read_consolidated_metadata_v2` and the model's `from_key_value` report the same problems, each in the file that holds it, or in its entry."""
    held = {**document, "attributes": {}}
    model = ZarrV2ArrayMetadata if name == ".zarray" else ZarrV2GroupMetadata
    from_entries = read_consolidated_metadata_v2(
        {
            "zarr_consolidated_format": 1,
            "metadata": {".zgroup": {"zarr_format": 2}, f"a/{name}": held},
        }
    ).problems
    with pytest.raises(MetadataValidationError) as raised:
        model.from_key_value({name: json.dumps(held).encode()})
    assert [(p.loc, p.kind) for p in read_node_metadata_v2({name: held}).problems] == found
    assert [(p.loc, p.kind) for p in from_entries] == [
        (("metadata", f"a/{loc[0]}", *loc[1:]), kind) for loc, kind in found
    ]
    assert [(p.loc, p.kind) for p in raised.value.problems] == found


def test_error_a_node_file_that_holds_attributes_next_to_a_bad_zattrs_reports_both() -> None:
    """A `.zarray` that holds `attributes`, next to a `.zattrs` that is not an object, has two problems, one in each file, and neither hides the other. `read_node_metadata_v2`, the model's `from_key_value` and `read_consolidated_metadata_v2` each report them in the file, or the entry, that holds each."""
    held = {**ZARRAY, "attributes": {}}
    found = [((".zarray", "attributes"), "invalid_value"), ((".zattrs",), "invalid_type")]
    assert [
        (p.loc, p.kind) for p in read_node_metadata_v2(_files(_zarray=held, _zattrs=5)).problems
    ] == found
    with pytest.raises(MetadataValidationError) as raised:
        ZarrV2ArrayMetadata.from_key_value(_store(_zarray=held, _zattrs=5))
    assert [(p.loc, p.kind) for p in raised.value.problems] == found
    from_entries = read_consolidated_metadata_v2(
        {
            "zarr_consolidated_format": 1,
            "metadata": {".zgroup": {"zarr_format": 2}, "a/.zarray": held, "a/.zattrs": 5},
        }
    ).problems
    assert [(p.loc, p.kind) for p in from_entries] == [
        (("metadata", "a/.zarray", "attributes"), "invalid_value"),
        (("metadata", "a/.zattrs"), "invalid_type"),
    ]


def test_error_a_store_file_that_is_not_json_raises_when_building_a_node() -> None:
    """Decoding is the store's step: a store file whose bytes are not JSON makes `node_metadata_from_key_value_v2` raise with an `invalid_json` problem at the file."""
    with pytest.raises(MetadataValidationError) as raised:
        node_metadata_from_key_value_v2({".zarray": b"{not json"})
    assert [(p.loc, p.kind) for p in raised.value.problems] == [((".zarray",), "invalid_json")]


def test_a_v2_node_reading_locates_its_fields_in_the_files_it_read() -> None:
    """A node's reading is of its files, so each field is under `.zarray`, as each problem is, and `with_problems` pairs a field with the problems in it. The model reads the merged document, so its own reading locates the same field at `dtype`. A pickled node reading keeps where its fields are."""
    struct = {**ZARRAY, "dtype": [["a", "<i4"]], "fill_value": None}
    valid = read_node_metadata_v2(_files(_zarray=struct))
    assert isinstance(valid, ZarrV2ArrayMetadataReading)
    assert valid.metadata is not None
    assert [loc for loc, _ in valid.fields()] == [(".zarray", "dtype"), (".zarray", "dtype", 0, 1)]
    assert [loc for loc, _ in valid.metadata.reading.fields()] == [("dtype",), ("dtype", 0, 1)]
    assert pickle.loads(pickle.dumps(valid)) == valid
    refused = read_node_metadata_v2(_files(_zarray={**struct, "dtype": [["a", "float32"]]}))
    paired = {
        loc: [p.loc for p in problems]
        for loc, _, problems in with_problems(refused.fields(), refused.problems)
    }
    assert paired[(".zarray", "dtype", 0, 1)] == [(".zarray", "dtype", 0, 1)]


def test_error_a_v2_node_with_a_bad_document_reports_its_problems() -> None:
    """A node whose document has a problem gives the reading of its node type with the problems, located in the file that holds each, and no model."""
    files = _files(_zarray={**ZARRAY, "shape": [-1]})
    reading = read_node_metadata_v2(files)
    assert type(reading) is ZarrV2ArrayMetadataReading
    assert [(p.loc, p.kind) for p in reading.problems] == [((".zarray", "shape"), "invalid_value")]
    assert reading.metadata is None


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


@pytest.mark.parametrize(
    ("entries", "key"),
    [
        ({".zgroup": {"zarr_format": 2}, "a/.zarray": ZARRAY}, "a/.zarray"),
        (
            {
                ".zgroup": {"zarr_format": 2},
                "a/.zgroup": {"zarr_format": 2},
                "a/b/.zarray": ZARRAY,
            },
            "a/b/.zarray",
        ),
        ({".zarray": ZARRAY}, ".zarray"),
    ],
    ids=["path", "nested-path", "root"],
)
def test_consolidated_fields_are_located_at_the_keys_the_document_writes(
    entries: dict[str, Any], key: str
) -> None:
    """`fields()` locates each node's fields under the key that the `.zmetadata` writes for the node's `.zarray`, so each location leads into the document."""
    reading = read_consolidated_metadata_v2({"zarr_consolidated_format": 1, "metadata": entries})
    assert reading.problems == ()
    assert [loc for loc, _ in reading.fields()] == [("metadata", key, "dtype")]


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
