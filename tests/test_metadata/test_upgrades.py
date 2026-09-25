"""Tests for the upgrades that read invalid stored array metadata documents."""

from __future__ import annotations

import json
import re
import warnings
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pytest

import zarr
from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata
from zarr.core.metadata.upgrades import (
    V2_ARRAY_UPGRADES,
    V3_ARRAY_UPGRADES,
    upgrade_array_document,
)
from zarr.core.metadata.v3 import RectilinearChunkGridMetadata, RegularChunkGridMetadata
from zarr.dtype import Int16
from zarr.errors import ZarrUserWarning

if TYPE_CHECKING:
    from pathlib import Path

    from zarr.core.common import JSON


def _v2_doc(shape: list[int], chunks: list[Any]) -> dict[str, JSON]:
    return {
        "zarr_format": 2,
        "shape": shape,
        "chunks": chunks,
        "dtype": "<i2",
        "fill_value": 0,
        "order": "C",
        "filters": None,
        "compressor": None,
    }


def _v3_doc(
    shape: list[int], chunk_shape: list[Any], inner: list[int] | None = None
) -> dict[str, JSON]:
    bytes_codec: dict[str, JSON] = {"name": "bytes", "configuration": {"endian": "little"}}
    codecs: list[JSON] = [bytes_codec]
    if inner is not None:
        codecs = [
            {
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": inner,
                    "codecs": [bytes_codec],
                    "index_codecs": [bytes_codec, {"name": "crc32c"}],
                    "index_location": "end",
                },
            }
        ]
    return {
        "zarr_format": 3,
        "node_type": "array",
        "shape": shape,
        "data_type": "int16",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": chunk_shape}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": 0,
        "codecs": codecs,
    }


def _stored_chunks(doc: dict[str, Any]) -> Any:
    return (
        doc["chunks"]
        if doc["zarr_format"] == 2
        else doc["chunk_grid"]["configuration"]["chunk_shape"]
    )


@pytest.mark.parametrize(
    ("doc", "expected", "warning"),
    [
        (_v2_doc([10, 10], [4, 5]), [4, 5], None),
        (_v3_doc([0, 0], [1, 1]), [1, 1], None),
        (_v3_doc([10], [4], inner=[2]), [4], None),
        (_v2_doc([0, 4], [0, 4]), [1, 4], "0 on axis 0 as one chunk spanning the axis"),
        (_v3_doc([0], [False]), [1], "false on axis 0 as one chunk spanning the axis"),
        (_v2_doc([5], [True]), [1], "true on axis 0 as 1"),
        (_v3_doc([5, 4], [True, 4]), [1, 4], "true on axis 0 as 1"),
        (_v2_doc([3], [0]), [3], "holds only its fill value"),
        (_v3_doc([4, 3], [4, 0]), [4, 3], "holds only its fill value"),
        (_v3_doc([0], [0], inner=[4]), [4], "spanning the axis \\(4\\)"),
        (_v3_doc([10], [0], inner=[4]), [12], "spanning the axis \\(12\\)"),
        (_v3_doc([0, 3], [0, 3], inner=[2, 3]), [2, 3], "spanning the axis \\(2\\)"),
    ],
    ids=[
        "v2-valid",
        "v3-valid-empty-axes",
        "v3-valid-sharded",
        "v2-zero-empty-axis",
        "v3-false-empty-axis",
        "v2-true",
        "v3-true",
        "v2-zero-grown-axis",
        "v3-zero-grown-axis",
        "v3-sharded-zero-empty-axis",
        "v3-sharded-zero-grown-axis",
        "v3-sharded-zero-2d",
    ],
)
def test_upgrade_array_document(
    doc: dict[str, JSON], expected: list[int], warning: str | None
) -> None:
    """Valid documents pass unchanged and silently. A stored chunk size of 0 or `false`
    is read as one chunk spanning the axis (a multiple of the inner chunk when sharded)
    and `true` as 1; `from_dict` warns once with the reading and how to re-save."""
    upgrades = V2_ARRAY_UPGRADES if doc["zarr_format"] == 2 else V3_ARRAY_UPGRADES
    upgraded, readings = upgrade_array_document(doc, upgrades)
    assert _stored_chunks(dict(upgraded)) == expected
    assert {k: v for k, v in upgraded.items() if k not in ("chunks", "chunk_grid")} == {
        k: v for k, v in doc.items() if k not in ("chunks", "chunk_grid")
    }
    metadata_cls = ArrayV2Metadata if doc["zarr_format"] == 2 else ArrayV3Metadata
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        metadata_cls.from_dict(dict(doc))
    messages = [str(w.message) for w in record]
    if warning is None:
        assert upgraded is doc
        assert readings == []
        assert messages == []
    else:
        assert len(readings) == 1
        assert re.search(warning, readings[0])
        assert len(messages) == 1
        assert messages[0].startswith(readings[0])
        assert "update_attributes({})" in messages[0]
        assert "zarr.consolidate_metadata" in messages[0]


@pytest.mark.parametrize(
    "doc",
    [_v2_doc([4], [0]) | {"dtype": "<x2"}, _v3_doc([4], [0]) | {"data_type": "x"}],
    ids=["v2", "v3"],
)
def test_invalid_upgraded_document_raises_without_warning(doc: dict[str, JSON]) -> None:
    """A document that is invalid after its upgrade raises its own error, without first
    warning how it was read."""
    metadata_cls = ArrayV2Metadata if doc["zarr_format"] == 2 else ArrayV3Metadata
    with warnings.catch_warnings():
        warnings.simplefilter("error", ZarrUserWarning)
        with pytest.raises(ValueError):
            metadata_cls.from_dict(doc)


@pytest.mark.parametrize("doc", [_v2_doc([4], [-1]), _v3_doc([4], [-1])], ids=["v2", "v3"])
def test_stored_negative_chunk_size_rejected(doc: dict[str, JSON]) -> None:
    metadata_cls = ArrayV2Metadata if doc["zarr_format"] == 2 else ArrayV3Metadata
    with pytest.raises(ValueError, match="chunk edge length must be an integer >= 1, got -1"):
        metadata_cls.from_dict(doc)


@pytest.mark.parametrize("doc", [_v2_doc([4, 4], [0]), _v3_doc([4, 4], [0])], ids=["v2", "v3"])
def test_stored_chunk_shape_ndim_mismatch_rejected(doc: dict[str, JSON]) -> None:
    """A chunk shape with the wrong number of axes is not upgraded, only rejected."""
    metadata_cls = ArrayV2Metadata if doc["zarr_format"] == 2 else ArrayV3Metadata
    with warnings.catch_warnings():
        warnings.simplefilter("error", ZarrUserWarning)
        with pytest.raises(ValueError, match="chunk edge length|same length|same number"):
            metadata_cls.from_dict(doc)


@pytest.mark.parametrize("size", [0, True], ids=["zero", "true"])
def test_constructor_rejects_invalid_chunk_size(size: int) -> None:
    """Metadata built in code is strict: 0 and `True` are rejected, without a warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ZarrUserWarning)
        with pytest.raises(ValueError, match=f"got {size!r}"):
            RegularChunkGridMetadata(chunk_shape=(size,))
        with pytest.raises(ValueError, match=f"got {size!r}"):
            ArrayV2Metadata(
                shape=(0,),
                chunks=(size,),
                dtype=Int16(),
                fill_value=0,
                order="C",
            )


def _rewrite_doc(path: Path, zarr_format: Literal[2, 3], edit: Any) -> None:
    doc_path = path / (".zarray" if zarr_format == 2 else "zarr.json")
    doc = json.loads(doc_path.read_text())
    edit(doc)
    doc_path.write_text(json.dumps(doc))


@pytest.mark.parametrize(
    ("zarr_format", "shape", "stored", "inner", "expected"),
    [
        (2, (0, 4), [0, 4], None, (1, 4)),
        (2, (3,), [0], None, (3,)),
        (2, (5,), [True], None, (1,)),
        (3, (0,), [False], None, (1,)),
        (3, (3,), [0], None, (3,)),
        (3, (5,), [True], None, (1,)),
        (3, (0,), [0], (4,), (4,)),
        (3, (10,), [0], (4,), (12,)),
        (3, (0, 3), [0, 3], (2, 3), (2, 3)),
    ],
    ids=[
        "v2-empty-2d",
        "v2-grown",
        "v2-true",
        "v3-false-empty",
        "v3-grown",
        "v3-true",
        "v3-sharded-empty",
        "v3-sharded-grown",
        "v3-sharded-2d",
    ],
)
def test_legacy_chunk_size_round_trip(
    tmp_path: Path,
    zarr_format: Literal[2, 3],
    shape: tuple[int, ...],
    stored: list[Any],
    inner: tuple[int, ...] | None,
    expected: tuple[int, ...],
) -> None:
    """A store whose metadata holds a chunk size written by older software opens with a
    warning, reads and appends under the upgraded grid, and re-saves valid metadata."""
    path = tmp_path / "legacy.zarr"
    arr = zarr.create_array(
        store=path,
        shape=shape,
        chunks=inner or expected,
        shards=expected if inner else None,
        dtype="int16",
        fill_value=0,
        zarr_format=zarr_format,
    )
    data = np.arange(np.prod(shape), dtype="int16").reshape(shape)
    arr[...] = data

    def store_legacy(doc: dict[str, Any]) -> None:
        if zarr_format == 2:
            doc["chunks"] = stored
        else:
            doc["chunk_grid"]["configuration"]["chunk_shape"] = stored

    _rewrite_doc(path, zarr_format, store_legacy)

    with pytest.warns(ZarrUserWarning, match="is read as"):
        arr = zarr.open_array(store=path, mode="a")
    assert (arr.shards or arr.chunks) == expected
    np.testing.assert_array_equal(arr[...], data)

    block = np.full((2, *shape[1:]), 7, dtype="int16")
    arr.append(block)
    arr.update_attributes({})
    with warnings.catch_warnings():
        warnings.simplefilter("error", ZarrUserWarning)
        reopened = zarr.open_array(store=path)
    np.testing.assert_array_equal(reopened[...], np.concatenate([data, block]))


@pytest.mark.filterwarnings("ignore:Consolidated metadata is currently not part:UserWarning")
@pytest.mark.parametrize("zarr_format", [2, 3])
def test_legacy_chunk_size_consolidated(tmp_path: Path, zarr_format: Literal[2, 3]) -> None:
    """Consolidated metadata goes through the same upgrade; re-saving the array and
    consolidating again leaves a group that opens without a warning."""
    path = tmp_path / "group.zarr"
    group = zarr.open_group(path, mode="w", zarr_format=zarr_format)
    group.create_array("a", shape=(0,), chunks=(1,), dtype="int32")
    zarr.consolidate_metadata(path)

    # What zarr-python wrote for `chunks=(0,)` on an empty array, in both copies.
    if zarr_format == 2:
        _rewrite_doc(path / "a", 2, lambda doc: doc.update(chunks=[0]))
        zmetadata = json.loads((path / ".zmetadata").read_text())
        zmetadata["metadata"]["a/.zarray"]["chunks"] = [0]
        (path / ".zmetadata").write_text(json.dumps(zmetadata))
    else:
        _rewrite_doc(
            path / "a", 3, lambda doc: doc["chunk_grid"]["configuration"].update(chunk_shape=[0])
        )
        _rewrite_doc(
            path,
            3,
            lambda doc: doc["consolidated_metadata"]["metadata"]["a"]["chunk_grid"][
                "configuration"
            ].update(chunk_shape=[0]),
        )

    with pytest.warns(ZarrUserWarning, match="zarr.consolidate_metadata"):
        group = zarr.open_group(path, mode="r+")
    array = group["a"]
    assert isinstance(array, zarr.Array)
    assert array.chunks == (1,)

    array.update_attributes({})
    zarr.consolidate_metadata(path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ZarrUserWarning)
        warnings.filterwarnings("ignore", "Consolidated metadata is currently not part")
        for use_consolidated in (True, False):
            reopened = zarr.open_group(path, mode="r", use_consolidated=use_consolidated)["a"]
            assert isinstance(reopened, zarr.Array)
            assert reopened.chunks == (1,)


# A document copied verbatim from a store that zarr 3.2.1 wrote for
# `create_array(shape=(6, 20), chunks=(2, (5, 10, 5)), dtype="float32")`.
MIXED_REGULAR_GRID_DOC = """{
  "shape": [6, 20],
  "data_type": "float32",
  "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [2, [5, 10, 5]]}},
  "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
  "fill_value": 0.0,
  "codecs": [
    {"name": "bytes", "configuration": {"endian": "little"}},
    {"name": "zstd", "configuration": {"level": 0, "checksum": false}}
  ],
  "attributes": {},
  "zarr_format": 3,
  "node_type": "array",
  "storage_transformers": []
}"""


def _mixed_doc(shape: list[int], chunk_shape: list[Any]) -> dict[str, JSON]:
    doc: dict[str, JSON] = json.loads(MIXED_REGULAR_GRID_DOC)
    doc["shape"] = shape
    doc["chunk_grid"] = {"name": "regular", "configuration": {"chunk_shape": chunk_shape}}
    return doc


@pytest.mark.parametrize(
    ("doc", "expected", "axes"),
    [
        (json.loads(MIXED_REGULAR_GRID_DOC), (2, (5, 10, 5)), [1]),
        (_mixed_doc([6, 20, 4], [2, [5, 10, 5], [1, 3]]), (2, (5, 10, 5), (1, 3)), [1, 2]),
        (_mixed_doc([6, 20], [[1, 5], [5, 10, 5]]), ((1, 5), (5, 10, 5)), [0, 1]),
        (_mixed_doc([6, 12], [2, [5, 10, 5]]), (2, (5, 10, 5)), [1]),
        (_mixed_doc([0, 20], [2, [5, 10, 5]]), (2, (5, 10, 5)), [1]),
        (_mixed_doc([6, 0], [2, [5, 10, 5]]), (2, (5, 10, 5)), [1]),
        (_mixed_doc([4, 10_000], [2, [10] * 1000]), (2, (10,) * 1000), [1]),
    ],
    ids=["written", "3d", "every-axis", "shrunk", "empty-regular-axis", "empty-edge-axis", "long"],
)
def test_read_edge_lists_in_regular_grid(
    doc: dict[str, JSON], expected: tuple[int | tuple[int, ...], ...], axes: list[int]
) -> None:
    """A `regular` chunk grid whose `chunk_shape` lists chunk edges is read as the
    rectilinear grid it describes, without the rectilinear chunks flag, with one
    warning that names the axes, quotes at most a bounded part of the chunk shape and
    says how to re-save."""
    with (
        zarr.config.set({"array.rectilinear_chunks": False}),
        warnings.catch_warnings(record=True) as record,
    ):
        warnings.simplefilter("always")
        metadata = ArrayV3Metadata.from_dict(doc)
    assert metadata.chunk_grid == RectilinearChunkGridMetadata(chunk_shapes=expected)
    [message] = [str(w.message) for w in record]
    assert f"lists chunk edge lengths on axes {axes}" in message
    assert "array.rectilinear_chunks" in message
    assert "update_attributes({})" in message
    assert len(message) < 1000


def test_edge_lists_in_regular_grid_rle_rejected() -> None:
    """Run-length encoded edges were never written inside a `regular` chunk_shape, so
    such a document is not read as rectilinear."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ZarrUserWarning)
        with pytest.raises(TypeError, match="Dimension 1: a regular chunk grid requires"):
            ArrayV3Metadata.from_dict(_mixed_doc([6, 20], [2, [[5, 2], 10]]))


@pytest.mark.parametrize("edges", [[10.0, 10.0], [True, True]], ids=["float", "bool"])
def test_edge_lists_in_regular_grid_non_integer_edges_rejected(edges: list[Any]) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", ZarrUserWarning)
        with pytest.raises(TypeError, match="Dimension 1: a regular chunk grid requires"):
            ArrayV3Metadata.from_dict(_mixed_doc([6, 20], [2, edges]))


def test_edge_lists_in_regular_grid_nonpositive_edge_rejected() -> None:
    """The upgraded grid is validated before any warning, so the real error surfaces."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ZarrUserWarning)
        with pytest.raises(ValueError, match="must be >= 1, got 0"):
            ArrayV3Metadata.from_dict(_mixed_doc([6, 20], [2, [0, 20]]))


def test_edge_lists_in_regular_grid_short_edges_rejected() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", ZarrUserWarning)
        with pytest.raises(ValueError, match="sum to 15 but array shape extent is 20"):
            ArrayV3Metadata.from_dict(_mixed_doc([6, 20], [2, [5, 10]]))


def test_edge_lists_in_regular_grid_round_trip(tmp_path: Path) -> None:
    """A store holding the verbatim document opens without the rectilinear chunks flag,
    reads its data, re-saves as rectilinear with the flag, and then opens cleanly."""
    path = tmp_path / "mixed.zarr"
    data = np.arange(120, dtype="float32").reshape(6, 20)
    with zarr.config.set({"array.rectilinear_chunks": True}):
        arr = zarr.create_array(path, shape=data.shape, chunks=(2, (5, 10, 5)), dtype="float32")
    arr[...] = data
    (path / "zarr.json").write_text(MIXED_REGULAR_GRID_DOC)

    with (
        zarr.config.set({"array.rectilinear_chunks": False}),
        pytest.warns(ZarrUserWarning, match="read as that rectilinear chunk grid"),
    ):
        arr = zarr.open_array(path, mode="a")
    np.testing.assert_array_equal(arr[...], data)

    with zarr.config.set({"array.rectilinear_chunks": True}):
        arr.update_attributes({})
        assert json.loads((path / "zarr.json").read_text())["chunk_grid"] == {
            "name": "rectilinear",
            "configuration": {"kind": "inline", "chunk_shapes": [2, [5, 10, 5]]},
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error", ZarrUserWarning)
            reopened = zarr.open_array(path)
    np.testing.assert_array_equal(reopened[...], data)
