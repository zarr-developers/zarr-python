"""Tests for comparing stored metadata documents with the documents metadata would store,
and for storing only the documents that differ."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal

import pytest

import zarr
from zarr.core.buffer import cpu
from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata
from zarr.core.metadata.io import (
    ABSENT,
    ARRAY_DOCUMENTS,
    DocumentChange,
    diff_documents,
    read_documents,
    upsert_metadata,
)
from zarr.core.sync import sync
from zarr.storage import MemoryStore, StorePath

if TYPE_CHECKING:
    from zarr.abc.store import Store
    from zarr.core.buffer import Buffer
    from zarr.core.common import JSON
    from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata

V3_DOC: dict[str, JSON] = {
    "zarr_format": 3,
    "node_type": "array",
    "shape": [3],
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [3]}},
    "fill_value": 0,
}


def _with(doc: dict[str, JSON], **changes: JSON) -> dict[str, JSON]:
    return {**doc, **changes}


@pytest.mark.parametrize(
    ("stored", "new", "expected"),
    [
        ({"zarr.json": V3_DOC}, {"zarr.json": V3_DOC}, ()),
        (
            {"zarr.json": V3_DOC},
            {"zarr.json": _with(V3_DOC, fill_value=1)},
            ((("zarr.json", "fill_value"), 0, 1),),
        ),
        (
            {
                "zarr.json": _with(
                    V3_DOC, chunk_grid={"name": "regular", "configuration": {"chunk_shape": [0]}}
                )
            },
            {"zarr.json": V3_DOC},
            ((("zarr.json", "chunk_grid", "configuration", "chunk_shape", 0), 0, 3),),
        ),
        (
            {"zarr.json": V3_DOC},
            {"zarr.json": _with(V3_DOC, shape=[3, 4])},
            ((("zarr.json", "shape", 1), ABSENT, 4),),
        ),
        (
            {"zarr.json": _with(V3_DOC, attributes={"a": 1})},
            {"zarr.json": _with(V3_DOC, dimension_names=["x"])},
            (
                (("zarr.json", "attributes"), {"a": 1}, ABSENT),
                (("zarr.json", "dimension_names"), ABSENT, ["x"]),
            ),
        ),
        (
            {"zarr.json": _with(V3_DOC, fill_value=True)},
            {"zarr.json": _with(V3_DOC, fill_value=1)},
            ((("zarr.json", "fill_value"), True, 1),),
        ),
        (
            {"zarr.json": _with(V3_DOC, fill_value=1.0)},
            {"zarr.json": _with(V3_DOC, fill_value=1)},
            ((("zarr.json", "fill_value"), 1.0, 1),),
        ),
        (
            {"zarr.json": _with(V3_DOC, fill_value=float("nan"))},
            {"zarr.json": _with(V3_DOC, fill_value=float("nan"))},
            (),
        ),
        (
            {"zarr.json": _with(V3_DOC, shape={"0": 3})},
            {"zarr.json": V3_DOC},
            ((("zarr.json", "shape"), {"0": 3}, [3]),),
        ),
        (
            {".zarray": {"shape": [3], "chunks": [0]}},
            {".zarray": {"shape": [3], "chunks": [3]}, ".zattrs": {}},
            (((".zarray", "chunks", 0), 0, 3), ((".zattrs",), ABSENT, {})),
        ),
    ],
    ids=[
        "identical",
        "changed-scalar",
        "changed-nested-list",
        "longer-list",
        "removed-and-added-key",
        "bool-is-not-int",
        "float-is-not-int",
        "nan-is-nan",
        "object-is-not-array",
        "v2-two-documents",
    ],
)
def test_diff_documents(
    stored: dict[str, JSON], new: dict[str, JSON], expected: tuple[Any, ...]
) -> None:
    """Documents are compared value by value, each change named by its JSON path under
    the document's key; values other than objects and arrays are identical only if
    their JSON encodings are. `diff_documents` is total over JSON: it raises no error."""
    assert diff_documents(stored, new) == tuple(DocumentChange(*change) for change in expected)


class _CountingStore(MemoryStore):
    """A memory store that counts the values set in it."""

    sets = 0

    async def set(self, key: str, value: Buffer, byte_range: tuple[int, int] | None = None) -> None:
        self.sets += 1
        await super().set(key, value, byte_range)


def _documents(store: Store) -> dict[str, Any]:
    assert isinstance(store, MemoryStore)
    return {key: json.loads(value.to_bytes()) for key, value in store._store_dict.items()}


def _legacy(zarr_format: Literal[2, 3]) -> tuple[StorePath, ArrayV2Metadata | ArrayV3Metadata]:
    """An array stored with chunk shape `[0]`, and the metadata its upgrade reads."""
    store = _CountingStore()
    array = zarr.create_array(
        store, shape=(3,), chunks=(3,), dtype="int16", zarr_format=zarr_format
    )
    key = ".zarray" if zarr_format == 2 else "zarr.json"
    doc = _documents(store)[key]
    if zarr_format == 2:
        doc["chunks"] = [0]
    else:
        doc["chunk_grid"]["configuration"]["chunk_shape"] = [0]
    sync(store.set(key, cpu.Buffer.from_bytes(json.dumps(doc).encode())))
    return StorePath(store), array.metadata


def _upsert(
    store_path: StorePath, metadata: ArrayV2Metadata | ArrayV3Metadata
) -> tuple[DocumentChange, ...]:
    """Upsert `metadata` against the documents stored at `store_path`."""
    stored = sync(read_documents(store_path, ARRAY_DOCUMENTS[metadata.zarr_format]))
    return sync(upsert_metadata(store_path, metadata, stored))


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_upsert_metadata_stores_documents_that_differ(zarr_format: Literal[2, 3]) -> None:
    """The documents that differ from the stored ones are stored, and the changes are
    returned."""
    store_path, metadata = _legacy(zarr_format)
    assert isinstance(store_path.store, _CountingStore)
    store_path.store.sets = 0
    key = ".zarray" if zarr_format == 2 else "zarr.json"
    chunk_path = (
        ("chunks", 0) if zarr_format == 2 else ("chunk_grid", "configuration", "chunk_shape", 0)
    )

    changes = _upsert(store_path, metadata)

    assert changes == (DocumentChange((key, *chunk_path), 0, 3),)
    assert store_path.store.sets == 1
    stored = _documents(store_path.store)
    assert stored[key] == json.loads(metadata.to_buffer_dict(cpu.buffer_prototype)[key].to_bytes())


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_upsert_metadata_identical_stores_nothing(zarr_format: Literal[2, 3]) -> None:
    """Metadata identical to what is stored stores nothing."""
    store = _CountingStore()
    array = zarr.create_array(
        store, shape=(3,), chunks=(3,), dtype="int16", zarr_format=zarr_format
    )
    store.sets = 0

    assert _upsert(StorePath(store), array.metadata) == ()
    assert store.sets == 0


def test_upsert_metadata_unstorable_leaves_store_untouched() -> None:
    """Metadata that may not be stored (a rectilinear chunk grid without the rectilinear
    chunks flag) fails before the store is read or written, naming the array."""
    store_path, _ = _legacy(3)
    before = _documents(store_path.store)
    with zarr.config.set({"array.rectilinear_chunks": True}):
        metadata = zarr.create_array(
            MemoryStore(), shape=(3,), chunks=[[1, 2]], dtype="int16"
        ).metadata
    with (
        zarr.config.set({"array.rectilinear_chunks": False}),
        pytest.raises(ValueError, match="experimental and disabled") as info,
    ):
        _upsert(store_path, metadata)
    assert info.value.__notes__ == [f"Array {str(store_path)!r}: nothing was stored."]
    assert _documents(store_path.store) == before


def test_upsert_metadata_stored_document_not_an_object() -> None:
    """A stored document that is not a JSON object is not overwritten."""
    store_path, metadata = _legacy(3)
    sync(store_path.store.set("zarr.json", cpu.Buffer.from_bytes(b"[]")))
    with pytest.raises(TypeError, match="Expected a JSON object, got list"):
        _upsert(store_path, metadata)
    assert _documents(store_path.store) == {"zarr.json": []}
