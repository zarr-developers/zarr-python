"""Tests for the upgrades that read invalid stored array metadata documents."""

from __future__ import annotations

import asyncio
import json
import re
import warnings
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pytest

import zarr
from zarr.codecs import ShardingCodec
from zarr.core.array import AsyncArray
from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata
from zarr.core.metadata.upgrades import (
    RESAVE_HINT,
    upgrade_array_document,
)
from zarr.core.metadata.v3 import RectilinearChunkGridMetadata, RegularChunkGridMetadata
from zarr.core.sync import sync
from zarr.dtype import Int16
from zarr.errors import ZarrUserWarning
from zarr.storage import MemoryStore, StorePath
from zarr.storage._common import make_store_path

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from zarr.core.common import JSON, ZarrFormat
    from zarr.types import AnyArray


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
    shape: list[int], chunk_shape: list[Any], inner: list[Any] | None = None
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


def _chunk_shapes(metadata: ArrayV2Metadata | ArrayV3Metadata) -> tuple[Any, ...]:
    """The chunk shape of `metadata`, then the inner chunk shape of each sharding codec,
    from the outermost in."""
    if isinstance(metadata, ArrayV2Metadata):
        return (metadata.chunks,)
    assert isinstance(metadata.chunk_grid, RegularChunkGridMetadata)
    shapes: list[Any] = [metadata.chunk_grid.chunk_shape]
    codecs: tuple[Any, ...] = metadata.codecs
    while sharding := next((c for c in codecs if isinstance(c, ShardingCodec)), None):
        shapes.append(sharding.chunk_shape)
        codecs = sharding.codecs
    return tuple(shapes)


def _nested_sharded_doc(inner: list[Any], nested: list[Any]) -> dict[str, JSON]:
    """A Zarr format 3 document of shape `[8]` in one shard of chunk shape `inner`, whose
    codecs shard each chunk again, in chunks of shape `nested`."""
    doc = _v3_doc([8], [8], inner=inner)
    outer = cast("dict[str, Any]", cast("list[JSON]", doc["codecs"])[0])
    configuration = outer["configuration"]
    configuration["codecs"] = [{**outer, "configuration": {**configuration, "chunk_shape": nested}}]
    return doc


@pytest.mark.parametrize(
    ("doc", "expected", "upgraded", "warning"),
    [
        (_v2_doc([10, 10], [4, 5]), ((4, 5),), False, None),
        (_v3_doc([0, 0], [1, 1]), ((1, 1),), False, None),
        (_v3_doc([10], [4], inner=[2]), ((4,), (2,)), False, None),
        (_v2_doc([0, 4], [0, 4]), ((1, 4),), True, None),
        (_v3_doc([0], [False]), ((1,),), True, None),
        (_v2_doc([5], [True]), ((1,),), True, None),
        (_v3_doc([5, 4], [True, 4]), ((1, 4),), True, None),
        (
            _v2_doc([3], [0]),
            ((3,),),
            True,
            (
                r"^The stored chunk shape \[0\] is invalid: .* read as \[3\], reading 0 in "
                r"dimension 0 as one chunk spanning the dimension \(3\), and .* holds only "
                r"its fill value\.$"
            ),
        ),
        (
            _v3_doc([4, 3], [4, 0]),
            ((4, 3),),
            True,
            r"reading 0 in dimension 1 as .* holds only its fill value\.$",
        ),
        (
            _v2_doc([0, 3], [0, 0]),
            ((1, 3),),
            True,
            r"read as \[1, 3\], reading 0 in dimension 1 as .* holds only its fill value\.$",
        ),
        (_v3_doc([0], [0], inner=[4]), ((4,), (4,)), True, None),
        (
            _v3_doc([10], [0], inner=[4]),
            ((12,), (4,)),
            True,
            r"spanning the dimension \(12\), and .* holds only its fill value",
        ),
        (_v3_doc([0, 3], [0, 3], inner=[2, 3]), ((2, 3), (2, 3)), True, None),
        (_v3_doc([5], [True], inner=[True]), ((1,), (1,)), True, None),
        (_nested_sharded_doc([4], [2]), ((8,), (4,), (2,)), False, None),
        (_nested_sharded_doc([4], [True]), ((8,), (4,), (1,)), True, None),
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
        "v2-zero-empty-and-grown-axes",
        "v3-sharded-zero-empty-axis",
        "v3-sharded-zero-grown-axis",
        "v3-sharded-zero-2d",
        "v3-sharded-true-inner-and-outer",
        "v3-nested-sharded-valid",
        "v3-nested-sharded-true",
    ],
)
def test_upgrade_array_document(
    doc: dict[str, JSON], expected: tuple[Any, ...], upgraded: bool, warning: str | None
) -> None:
    """Valid documents pass unchanged. A stored chunk size of 0 or `false` is read as one
    chunk spanning the axis (a multiple of the inner chunk when sharded) and `true` as 1,
    in the chunk shape and in the inner chunk shape of every sharding codec, nested or
    not. `from_dict` marks the metadata of an upgraded document; it warns once, naming
    the array, only where a chunk size of 0 was stored for a non-empty axis (which then
    holds only its fill value), saying how that part was read and how to re-save. The
    other readings give what zarr read before, so they are silent."""
    upgraded_doc, readings = upgrade_array_document(doc, cast("ZarrFormat", doc["zarr_format"]))
    assert {
        k: v for k, v in upgraded_doc.items() if k not in ("chunks", "chunk_grid", "codecs")
    } == {k: v for k, v in doc.items() if k not in ("chunks", "chunk_grid", "codecs")}
    assert bool(readings) is upgraded
    if not upgraded:
        assert upgraded_doc is doc
    metadata_cls = ArrayV2Metadata if doc["zarr_format"] == 2 else ArrayV3Metadata
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        metadata = metadata_cls.from_dict(dict(doc), path="group/array")
    assert _chunk_shapes(metadata) == expected
    assert metadata._stored_document_upgraded is upgraded
    messages = [str(w.message) for w in record]
    if warning is None:
        assert messages == []
    else:
        [message] = messages
        assert message.startswith("Array 'group/array': ")
        assert message.endswith(RESAVE_HINT)
        assert re.search(
            warning, message.removeprefix("Array 'group/array': ").removesuffix(f" {RESAVE_HINT}")
        )


@pytest.mark.parametrize(
    ("doc", "error"),
    [
        (_v2_doc([4], [0]) | {"order": "Z"}, "Failed to parse input for 'order'"),
        (_v3_doc([4], [0], inner=[2, 2]), "need to have the same number of dimensions"),
    ],
    ids=["v2", "v3-sharded"],
)
def test_invalid_upgraded_document_raises_without_warning(doc: dict[str, JSON], error: str) -> None:
    """A document the upgrades read that the metadata constructor then rejects raises
    that error, without first warning how it was read."""
    metadata_cls = ArrayV2Metadata if doc["zarr_format"] == 2 else ArrayV3Metadata
    assert upgrade_array_document(doc, cast("ZarrFormat", doc["zarr_format"]))[1]
    with warnings.catch_warnings():
        warnings.simplefilter("error", ZarrUserWarning)
        with pytest.raises(ValueError, match=error):
            metadata_cls.from_dict(doc)


def _read_strictly(doc: dict[str, JSON]) -> ArrayV2Metadata | ArrayV3Metadata:
    """Read `doc`, failing on any warning that it was upgraded."""
    metadata_cls = ArrayV2Metadata if doc["zarr_format"] == 2 else ArrayV3Metadata
    with warnings.catch_warnings():
        warnings.simplefilter("error", ZarrUserWarning)
        return metadata_cls.from_dict(doc)


def _open_strictly(path: Path, mode: Literal["r", "a", "r+"] = "r") -> AnyArray:
    """Open the array at `path`, failing on any warning that its document was upgraded."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ZarrUserWarning)
        array = zarr.open_array(store=path, mode=mode)
    assert isinstance(array, zarr.Array)
    return array


def test_stored_negative_chunk_size_rejected() -> None:
    """No known writer stored a negative chunk size: it is rejected, not upgraded."""
    with pytest.raises(ValueError, match="^Expected all values to be non-negative"):
        _read_strictly(_v2_doc([4], [-1]))


def test_stored_chunk_shape_ndim_mismatch_rejected() -> None:
    """A chunk shape with the wrong number of dimensions is not upgraded, so its 0 is
    rejected."""
    with pytest.raises(ValueError, match="^Dimension 0: chunk edge length must be >= 1, got 0$"):
        _read_strictly(_v3_doc([4, 4], [0]))


def _rectilinear_doc(shape: list[int], chunk_shapes: list[Any]) -> dict[str, JSON]:
    return _v3_doc(shape, [1] * len(shape)) | {
        "chunk_grid": {
            "name": "rectilinear",
            "configuration": {"kind": "inline", "chunk_shapes": chunk_shapes},
        }
    }


@pytest.mark.parametrize(
    ("chunk_shapes", "expected", "upgraded"),
    [
        ([[[4, 2]], [[5, 2]]], ((4, 4), (5, 5)), False),
        ([[[4.0, 2]], [[5, 2]]], ((4, 4), (5, 5)), True),
        ([[3.0, 5.0], [[5, 2]]], ((3, 5), (5, 5)), True),
        ([[[4.0, 2], 2.0], [[5, 2]]], ((4, 4, 2), (5, 5)), True),
        ([[[4.0, 2]], [10]], ((4, 4), (10,)), True),
        ([[[4.0, 2]], [5.0, 5]], ((4, 4), (5, 5)), True),
        ([[True, 4], [[5, 2]]], ((1, 4), (5, 5)), True),
        ([[[True, 2], 3], [[5, 2]]], ((1, 1, 3), (5, 5)), True),
    ],
    ids=[
        "valid",
        "float-rle-size",
        "float-edges",
        "float-rle-size-and-edge",
        "float-sharded-outer",
        "float-2d",
        "true-edge",
        "true-rle-size",
    ],
)
def test_read_invalid_edges_in_rectilinear_grid(
    chunk_shapes: list[Any], expected: tuple[tuple[int, ...], ...], upgraded: bool
) -> None:
    """A stored rectilinear chunk grid whose explicit edges or run-length encoded sizes
    are integral floats or JSON `true`, as zarr-python wrote them when given float or
    `True` edges, is read with those edges as the `int`s they equal. `from_dict` marks
    the metadata as upgraded, silently: zarr read these edges so before."""
    shape = [sum(edges) for edges in expected]
    doc = _rectilinear_doc(shape, chunk_shapes)
    with zarr.config.set({"array.rectilinear_chunks": True}):
        metadata = _read_strictly(doc)
        assert metadata.chunk_grid == RectilinearChunkGridMetadata(chunk_shapes=expected)
    assert metadata._stored_document_upgraded is upgraded


@pytest.mark.parametrize(
    ("doc", "error"),
    [
        (_v3_doc([20], [10.0]), "Dimension 0: chunk edge length must be an int, got 10.0"),
        (_v3_doc([4], [4.5]), "Dimension 0: chunk edge length must be an int, got 4.5"),
        (
            _v3_doc([8], [4], inner=[2.0]),
            "Expected an iterable of integers. Got [2.0] instead.",
        ),
        (_v2_doc([20], [10.0]), "Expected an iterable of integers. Got [10.0] instead."),
        (
            _rectilinear_doc([8], [[[4, 2.0]]]),
            "Dimension 0: RLE repeat count must be an int, got 2.0",
        ),
        (
            _rectilinear_doc([8], [4.0]),
            "Dimension 0: chunk edge length must be an int, got 4.0",
        ),
        (
            _rectilinear_doc([8], [[0.0, 8]]),
            "Dimension 0: chunk edge length must be an int, got 0.0",
        ),
    ],
    ids=[
        "regular",
        "regular-fractional",
        "sharding-inner",
        "v2",
        "rle-count",
        "rectilinear-bare",
        "rectilinear-zero",
    ],
)
def test_stored_float_chunk_size_rejected(doc: dict[str, JSON], error: str) -> None:
    """A float chunk size is read only where zarr-python stored one, as an edge of at
    least 1 of a rectilinear chunk grid. Anywhere else it is rejected, as zarr 3.4.0
    rejected a stored regular chunk size of `10.0`."""
    with zarr.config.set({"array.rectilinear_chunks": True}), pytest.raises(TypeError) as info:
        _read_strictly(doc)
    assert info.match(re.escape(error))


@pytest.mark.parametrize(
    ("chunks", "stored", "resaved"),
    [
        ([[4, 4], [5, 5]], [[[4.0, 2]], [[5, 2]]], "[[[4, 2]], [[5, 2]]]"),
        ([[1, 3, 4], [5, 5]], [[True, 3, 4], [[5, 2]]], "[[1, 3, 4], [[5, 2]]]"),
    ],
    ids=["float", "true"],
)
def test_invalid_edges_round_trip(
    tmp_path: Path, chunks: list[list[int]], stored: list[Any], resaved: str
) -> None:
    """A store whose rectilinear chunk grid holds the float or `true` edges zarr-python
    wrote opens silently, reads its data, and stores its edges as `int`s before the
    first write."""
    path = tmp_path / "rectilinear.zarr"
    data = np.arange(80, dtype="int16").reshape(8, 10)
    with zarr.config.set({"array.rectilinear_chunks": True}):
        zarr.create_array(path, shape=data.shape, chunks=chunks, dtype="int16")[...] = data
        _rewrite_doc(
            path, 3, lambda doc: doc["chunk_grid"]["configuration"].update(chunk_shapes=stored)
        )
        arr = _open_strictly(path, mode="a")
        np.testing.assert_array_equal(arr[...], data)
        arr[0, 0] = -1
        written = json.loads((path / "zarr.json").read_text())["chunk_grid"]["configuration"]
        assert json.dumps(written["chunk_shapes"]) == resaved
        data[0, 0] = -1
        np.testing.assert_array_equal(_open_strictly(path)[...], data)


def test_v2_constructor_rejects_chunks_of_wrong_length() -> None:
    with pytest.raises(ValueError, match="`chunks` has length 1, but `shape` has length 2"):
        ArrayV2Metadata(shape=(4, 4), chunks=(2,), dtype=Int16(), fill_value=0, order="C")


def _v2_metadata(chunks: Any) -> ArrayV2Metadata:
    return ArrayV2Metadata(shape=(4,), chunks=chunks, dtype=Int16(), fill_value=0, order="C")


def _rectilinear(chunk_shapes: tuple[Any, ...]) -> RectilinearChunkGridMetadata:
    with zarr.config.set({"array.rectilinear_chunks": True}):
        return RectilinearChunkGridMetadata(chunk_shapes=chunk_shapes)


def _rectilinear_from_dict(chunk_shapes: list[Any]) -> RectilinearChunkGridMetadata:
    with zarr.config.set({"array.rectilinear_chunks": True}):
        return RectilinearChunkGridMetadata.from_dict(
            {
                "name": "rectilinear",
                "configuration": {"kind": "inline", "chunk_shapes": chunk_shapes},
            }
        )


def _second_edge(grid: RectilinearChunkGridMetadata) -> int:
    edges = grid.chunk_shapes[0]
    assert isinstance(edges, tuple)
    return edges[1]


CHUNK_EDGE_SITES: dict[str, Callable[[Any], object]] = {
    "regular": lambda size: RegularChunkGridMetadata(chunk_shape=(size,)).chunk_shape[0],
    "rectilinear-bare": lambda size: _rectilinear((size,)).chunk_shapes[0],
    "rectilinear-edge": lambda size: _second_edge(_rectilinear(((4, size),))),
    "rectilinear-bare-json": lambda size: _rectilinear_from_dict([size]).chunk_shapes[0],
    "rectilinear-edge-json": lambda size: _second_edge(_rectilinear_from_dict([[4, size]])),
    "rectilinear-rle-json": lambda size: _second_edge(_rectilinear_from_dict([[[size, 2]]])),
}
"""Each place chunk grid metadata reads a chunk edge length, returning the edge it read."""


@pytest.mark.parametrize("site", CHUNK_EDGE_SITES)
@pytest.mark.parametrize(
    ("size", "expected"),
    [(4, 4), (True, 1)],
    ids=["int", "bool"],
)
def test_metadata_reads_integer_chunk_edge(site: str, size: object, expected: int) -> None:
    """Chunk grid metadata reads an `int` or a `bool` as the `int` chunk edge length it
    equals."""
    edge = CHUNK_EDGE_SITES[site](size)
    assert type(edge) is int
    assert edge == expected


@pytest.mark.parametrize("site", CHUNK_EDGE_SITES)
@pytest.mark.parametrize(
    "size",
    [4.0, np.float64(4.0), 4.5, float("inf"), "4", None, np.int64(4)],
    ids=["float", "numpy-float", "fractional", "inf", "str", "none", "numpy-int"],
)
def test_metadata_rejects_non_integer_chunk_edge(site: str, size: object) -> None:
    """A chunk edge length in metadata built in code is an `int`: a float is rejected,
    even an integral one (stored documents with integral floats are read by the
    upgrades), and so is a NumPy integer, as zarr 3.4.0 rejected one."""
    with pytest.raises(
        TypeError,
        match=re.escape(f"Dimension 0: chunk edge length must be an int, got {size!r}"),
    ):
        CHUNK_EDGE_SITES[site](size)


@pytest.mark.parametrize("site", CHUNK_EDGE_SITES)
@pytest.mark.parametrize("size", [0, False, -1])
def test_metadata_rejects_chunk_edge_below_one(site: str, size: int) -> None:
    """Chunk grid metadata built in code is strict: a chunk edge length below 1 is
    rejected, without a warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ZarrUserWarning)
        with pytest.raises(
            ValueError, match=f"Dimension 0: chunk edge length must be >= 1, got {size}"
        ):
            CHUNK_EDGE_SITES[site](size)


@pytest.mark.parametrize("chunk_shape", [4, np.int64(4), None, "44", {"4": 4}])
def test_regular_chunk_grid_rejects_chunk_shape_not_a_sequence(chunk_shape: Any) -> None:
    """A regular chunk shape is an iterable of chunk edge lengths, but not a string or a
    mapping, which is rejected as a whole, not entry by entry."""
    with pytest.raises(
        TypeError,
        match=re.escape(
            f"A chunk shape must be an iterable of chunk edge lengths, got {chunk_shape!r}"
        ),
    ):
        RegularChunkGridMetadata(chunk_shape=chunk_shape)


def _sharding_chunk_shape(chunks: Any) -> tuple[tuple[int, ...], object]:
    codec = ShardingCodec(chunk_shape=chunks)
    configuration = cast("dict[str, JSON]", codec.to_dict()["configuration"])
    return codec.chunk_shape, configuration["chunk_shape"]


CHUNK_SHAPE_SITES: dict[str, Callable[[Any], tuple[tuple[int, ...], object]]] = {
    "v2": lambda chunks: ((md := _v2_metadata(chunks)).chunks, md.to_dict()["chunks"]),
    "sharding-inner": _sharding_chunk_shape,
}
"""`ArrayV2Metadata` and `ShardingCodec` read a chunk shape as an array shape, returning
the chunk shape and the value `to_dict` writes for it."""


@pytest.mark.parametrize("site", CHUNK_SHAPE_SITES)
@pytest.mark.parametrize(
    ("chunks", "expected"),
    [
        ((4,), (4,)),
        ([4], (4,)),
        (4, (4,)),
        (np.int64(4), (4,)),
        ((np.int64(4),), (4,)),
        (np.array([4]), (4,)),
        ((True,), (1,)),
        ((0,), (0,)),
        ((False,), (0,)),
        (range(4, 5), (4,)),
    ],
)
def test_chunk_shape_read_as_array_shape(
    site: str, chunks: object, expected: tuple[int, ...]
) -> None:
    """`ArrayV2Metadata` and `ShardingCodec` read their chunk shape as `parse_shapelike`
    reads an array shape: an integer or an iterable of non-negative integers, including
    NumPy integers and bools. A chunk size of 0 is written back as given; reading a
    stored 0 is `zarr.core.metadata.upgrades`' business."""
    parsed, written = CHUNK_SHAPE_SITES[site](chunks)
    assert parsed == expected
    assert all(type(size) is int for size in parsed)
    assert written == expected


@pytest.mark.parametrize("site", CHUNK_SHAPE_SITES)
def test_chunk_shape_read_as_array_shape_rejects_negative(site: str) -> None:
    with pytest.raises(ValueError, match="Expected all values to be non-negative"):
        CHUNK_SHAPE_SITES[site]((-1,))


@pytest.mark.parametrize("site", CHUNK_SHAPE_SITES)
@pytest.mark.parametrize("chunks", [(4.0,), "4", None])
def test_chunk_shape_read_as_array_shape_rejects_non_integer(site: str, chunks: object) -> None:
    with pytest.raises(TypeError, match="Expected an"):
        CHUNK_SHAPE_SITES[site](chunks)


def _rewrite_doc(path: Path, zarr_format: Literal[2, 3], edit: Any) -> None:
    doc_path = path / (".zarray" if zarr_format == 2 else "zarr.json")
    doc = json.loads(doc_path.read_text())
    edit(doc)
    doc_path.write_text(json.dumps(doc))


@pytest.mark.parametrize(
    ("zarr_format", "shape", "stored", "inner", "expected", "warns"),
    [
        (2, (0, 4), [0, 4], None, (1, 4), False),
        (3, (5,), [True], None, (1,), False),
        (3, (10,), [0], (4,), (12,), True),
    ],
    ids=["v2-empty-2d", "v3-true", "v3-sharded-grown"],
)
def test_legacy_chunk_size_round_trip(
    tmp_path: Path,
    zarr_format: Literal[2, 3],
    shape: tuple[int, ...],
    stored: list[Any],
    inner: tuple[int, ...] | None,
    expected: tuple[int, ...],
    warns: bool,
) -> None:
    """A store whose metadata holds a chunk size written by older software opens (with a
    warning where a non-empty axis was stored with chunk size 0), reads and appends under
    the upgraded grid, and re-saves valid metadata."""
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

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", ZarrUserWarning)
        arr = zarr.open_array(store=path, mode="a")
    assert [
        re.match(r"^Array '.*legacy\.zarr': .* is read as", str(w.message)) is not None
        for w in record
    ] == [True] * warns
    assert (arr.shards or arr.chunks) == expected
    np.testing.assert_array_equal(arr[...], data)

    block = np.full((2, *shape[1:]), 7, dtype="int16")
    arr.append(block)
    arr.update_attributes({})
    with warnings.catch_warnings():
        warnings.simplefilter("error", ZarrUserWarning)
        reopened = zarr.open_array(store=path)
    np.testing.assert_array_equal(reopened[...], np.concatenate([data, block]))


@pytest.mark.parametrize(
    ("zarr_format", "shape", "inner", "expected"),
    [(2, (3,), None, (3,)), (3, (3,), None, (3,)), (3, (10,), (4,), (12,))],
    ids=["v2", "v3", "v3-sharded"],
)
@pytest.mark.parametrize("api", ["sync", "async", "async-concurrent"])
def test_write_stores_upgraded_metadata_first(
    tmp_path: Path,
    zarr_format: Literal[2, 3],
    shape: tuple[int, ...],
    inner: tuple[int, ...] | None,
    expected: tuple[int, ...],
    api: str,
) -> None:
    """Writing chunks to an array read from an upgraded document first stores the
    upgraded metadata, so readers that do not upgrade (or read it differently) see
    the chunks the write stored."""
    path = tmp_path / "legacy.zarr"
    zarr.create_array(
        store=path,
        shape=shape,
        chunks=inner or expected,
        shards=expected if inner else None,
        dtype="int16",
        fill_value=0,
        zarr_format=zarr_format,
    )
    _rewrite_doc(
        path,
        zarr_format,
        lambda doc: (
            doc.update(chunks=[0])
            if zarr_format == 2
            else doc["chunk_grid"]["configuration"].update(chunk_shape=[0])
        ),
    )
    data = np.arange(1, shape[0] + 1, dtype="int16")
    with pytest.warns(ZarrUserWarning, match="is read as"):
        arr = zarr.open_array(store=path, mode="r+")
    if api == "sync":
        arr[:] = data
    elif api == "async":
        sync(arr.async_array.setitem(slice(None), data))
    else:

        async def write_twice() -> None:
            # Both writes find the metadata not yet stored, and both store it.
            await asyncio.gather(*(arr.async_array.setitem(slice(None), data) for _ in "ab"))

        sync(write_twice())

    assert not arr.metadata._stored_document_upgraded
    with warnings.catch_warnings():
        warnings.simplefilter("error", ZarrUserWarning)
        reopened = zarr.open_array(store=path, mode="r")
    assert (reopened.shards or reopened.chunks) == expected
    np.testing.assert_array_equal(reopened[...], data)


def _legacy_array(path: Path, zarr_format: Literal[2, 3]) -> None:
    """Store an array of shape (3,) whose stored chunk shape is `[0]`."""
    zarr.create_array(store=path, shape=(3,), chunks=(3,), dtype="int16", zarr_format=zarr_format)
    _rewrite_doc(
        path,
        zarr_format,
        lambda doc: (
            doc.update(chunks=[0])
            if zarr_format == 2
            else doc["chunk_grid"]["configuration"].update(chunk_shape=[0])
        ),
    )


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_stale_handle_write_keeps_newer_metadata(
    tmp_path: Path, zarr_format: Literal[2, 3]
) -> None:
    """A handle read from an upgraded document stores the upgrade of what the store
    holds when it first writes chunks; if another handle stored valid metadata since,
    it stores no metadata and writes only its chunks."""
    path = tmp_path / "legacy.zarr"
    _legacy_array(path, zarr_format)
    with pytest.warns(ZarrUserWarning, match="is read as"):
        stale = zarr.open_array(store=path, mode="r+")
    with pytest.warns(ZarrUserWarning, match="is read as"):
        other = zarr.open_array(store=path, mode="r+")
    other.append(np.arange(1, 7, dtype="int16"))
    other.attrs["x"] = 1

    stale[0] = 9

    assert not stale.metadata._stored_document_upgraded
    reopened = _open_strictly(path)
    assert reopened.shape == (9,)
    assert reopened.attrs.asdict() == {"x": 1}
    np.testing.assert_array_equal(reopened[...], [9, 0, 0, 1, 2, 3, 4, 5, 6])


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_stale_handle_write_keeps_valid_document_as_written(
    tmp_path: Path, zarr_format: Literal[2, 3]
) -> None:
    """If the document the store holds when a handle read from an upgraded document
    first writes chunks needs no upgrade, it is left as written, even where zarr would
    encode the same metadata differently (as another implementation may have written it)."""
    path = tmp_path / "legacy.zarr"
    _legacy_array(path, zarr_format)
    with pytest.warns(ZarrUserWarning, match="is read as"):
        stale = zarr.open_array(store=path, mode="r+")
    # Valid metadata for the same array, as another writer might store it: chunk size
    # 3, without the optional members zarr writes, as compact JSON.
    doc_path = path / (".zarray" if zarr_format == 2 else "zarr.json")
    doc = json.loads(doc_path.read_text())
    for optional in ("dimension_separator", "attributes", "storage_transformers"):
        doc.pop(optional, None)
    _stored_chunks(doc)[0] = 3
    doc_path.write_text(json.dumps(doc, separators=(",", ":")))
    written = doc_path.read_bytes()

    stale[0] = 9

    assert doc_path.read_bytes() == written
    np.testing.assert_array_equal(_open_strictly(path)[...], [9, 0, 0])


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_stale_handle_write_after_chunk_grid_change_raises(
    tmp_path: Path, zarr_format: Literal[2, 3]
) -> None:
    """If the document the store holds when a handle read from an upgraded document
    first writes chunks lays out chunks differently from the handle's metadata (here the
    array was resized by software that kept the stored chunk size of 0, which now reads
    as a larger chunk), the handle's chunks would not be found under it: the write
    raises and stores nothing."""
    path = tmp_path / "legacy.zarr"
    _legacy_array(path, zarr_format)
    with pytest.warns(ZarrUserWarning, match="is read as"):
        stale = zarr.open_array(store=path, mode="r+")
    _rewrite_doc(path, zarr_format, lambda doc: doc.update(shape=[10]))
    documents = {p.name: p.read_bytes() for p in path.iterdir()}

    with pytest.raises(ValueError, match="has changed since this array was opened: reopen"):
        stale[0:3] = [7, 8, 9]

    assert stale.metadata._stored_document_upgraded
    assert {p.name: p.read_bytes() for p in path.iterdir()} == documents


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_write_without_stored_document(zarr_format: Literal[2, 3]) -> None:
    """An array read from an upgraded document that no store holds (as
    `AsyncArray.from_dict` builds one) writes its chunks as any array does: there is no
    stored document to upgrade."""
    store = MemoryStore()
    doc = _v2_doc([3], [True]) if zarr_format == 2 else _v3_doc([3], [True])
    array = zarr.Array(AsyncArray.from_dict(StorePath(store), doc))
    upgraded = array.metadata._stored_document_upgraded

    array[:] = [1, 2, 3]

    assert (upgraded, array.metadata._stored_document_upgraded) == (True, False)
    np.testing.assert_array_equal(array[:], [1, 2, 3])
    assert not [key for key in store._store_dict if key.endswith((".zarray", "zarr.json"))]


@pytest.mark.parametrize(("shape", "expected"), [((0,), (1,)), ((3,), (3,))])
def test_array_from_metadata_with_chunk_size_zero(shape: tuple[int], expected: tuple[int]) -> None:
    """`ArrayV2Metadata` accepts a chunk size of 0, as a stored document may hold it. An
    array built from such metadata reads it as the upgrades read that document, silently
    (no data was read or written under it): `create_hierarchy` stores the metadata as
    given and yields such an array, which stores the upgrade before its first write."""
    metadata = ArrayV2Metadata(shape=shape, chunks=(0,), dtype=Int16(), fill_value=0, order="C")
    store = MemoryStore()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ZarrUserWarning)
        nodes = dict(zarr.create_hierarchy(store=store, nodes={"a": metadata}))
    array = nodes["a"]
    assert isinstance(array, zarr.Array)
    assert array.chunks == expected
    assert json.loads(store._store_dict["a/.zarray"].to_bytes())["chunks"] == [0]

    array[...] = 1

    # Writing an empty selection stores no chunks, so it stores no metadata either.
    resaved = list(expected) if array.size else [0]
    assert json.loads(store._store_dict["a/.zarray"].to_bytes())["chunks"] == resaved
    np.testing.assert_array_equal(zarr.open_array(store, path="a")[...], np.ones(shape))


def _store_zero(doc: dict[str, Any]) -> None:
    _stored_chunks(doc)[0] = 0


def _rewrite_consolidated(
    path: Path, zarr_format: Literal[2, 3], name: str, edit: Callable[[dict[str, Any]], None]
) -> None:
    """Edit the document of the member `name` in the consolidated metadata at `path`."""
    if zarr_format == 2:
        document = json.loads((path / ".zmetadata").read_text())
        edit(document["metadata"][f"{name}/.zarray"])
        (path / ".zmetadata").write_text(json.dumps(document))
    else:
        _rewrite_doc(path, 3, lambda doc: edit(doc["consolidated_metadata"]["metadata"][name]))


def _consolidated_member(path: Path, zarr_format: Literal[2, 3], name: str) -> Any:
    if zarr_format == 2:
        return json.loads((path / ".zmetadata").read_text())["metadata"][f"{name}/.zarray"]
    return json.loads((path / "zarr.json").read_text())["consolidated_metadata"]["metadata"][name]


@pytest.mark.filterwarnings("ignore:Consolidated metadata is currently not part:UserWarning")
@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("operation", ["attrs", "update_attributes_async", "delete-member"])
def test_group_write_refreshes_upgraded_consolidated_member(
    tmp_path: Path, zarr_format: Literal[2, 3], operation: str
) -> None:
    """Storing a group's metadata stores its consolidated metadata. Each member of it read
    from a document that had to be upgraded is first read again from the member's own
    document as it now is (here resized by software that kept the stored chunk size of
    0), whose upgrade is stored, so no group write stores a stale copy as if valid."""
    path = tmp_path / "group.zarr"
    group = zarr.open_group(path, mode="w", zarr_format=zarr_format)
    group.create_array("a", shape=(3,), chunks=(3,), dtype="int16")
    group.create_array("b", shape=(1,), chunks=(1,), dtype="int16")
    zarr.consolidate_metadata(path)
    _rewrite_consolidated(path, zarr_format, "a", _store_zero)

    def resize_keeping_zero(doc: dict[str, Any]) -> None:
        _store_zero(doc)
        doc["shape"] = [10]

    _rewrite_doc(path / "a", zarr_format, resize_keeping_zero)
    with pytest.warns(ZarrUserWarning, match="is read as"):
        group = zarr.open_group(path, mode="r+", use_consolidated=True)

    if operation == "attrs":
        group.attrs["x"] = 1
    elif operation == "update_attributes_async":
        sync(group.update_attributes_async({"x": 1}))
    else:
        del group["b"]

    assert _stored_chunks(_consolidated_member(path, zarr_format, "a")) == [10]
    reopened = zarr.open_group(path, mode="r", use_consolidated=True)
    member = reopened["a"]
    assert isinstance(member, zarr.Array)
    assert (member.shape, member.chunks) == ((10,), (10,))
    assert _open_strictly(path / "a").chunks == (10,)


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_empty_write_stores_no_metadata(tmp_path: Path, zarr_format: Literal[2, 3]) -> None:
    """A write of an empty selection stores no chunks, so it stores no metadata either."""
    path = tmp_path / "legacy.zarr"
    _legacy_array(path, zarr_format)
    documents = {p.name: p.read_bytes() for p in path.iterdir()}
    with pytest.warns(ZarrUserWarning, match="is read as"):
        array = zarr.open_array(store=path, mode="r+")

    array[0:0] = np.empty(0, dtype="int16")

    assert array.metadata._stored_document_upgraded
    assert {p.name: p.read_bytes() for p in path.iterdir()} == documents


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_async_array_from_dict_names_array(tmp_path: Path, zarr_format: Literal[2, 3]) -> None:
    """`AsyncArray.from_dict` names the array at its store path in the upgrade warning."""
    store_path = sync(make_store_path(tmp_path / "legacy.zarr"))
    doc = _v2_doc([4], [0]) if zarr_format == 2 else _v3_doc([4], [0])
    with pytest.warns(ZarrUserWarning, match=f"^Array {re.escape(repr(str(store_path)))}: "):
        AsyncArray.from_dict(store_path, doc)


@pytest.mark.filterwarnings("ignore:Consolidated metadata is currently not part:UserWarning")
@pytest.mark.parametrize("zarr_format", [2, 3])
def test_consolidate_stores_upgraded_members(tmp_path: Path, zarr_format: Literal[2, 3]) -> None:
    """Consolidating a group stores the upgrade of every member document that needed
    one before the consolidated document, so chunks written through the consolidated
    metadata are stored under a member document that agrees with it."""
    path = tmp_path / "group.zarr"
    zarr.open_group(path, mode="w", zarr_format=zarr_format)
    _legacy_array(path / "a", zarr_format)
    with pytest.warns(ZarrUserWarning, match="is read as"):
        zarr.consolidate_metadata(path)

    member = _open_strictly(path / "a")
    assert member.chunks == (3,)
    group = zarr.open_group(path, mode="r+", use_consolidated=True)
    array = group["a"]
    assert isinstance(array, zarr.Array)
    array[:] = [7, 8, 9]
    np.testing.assert_array_equal(_open_strictly(path / "a")[...], [7, 8, 9])


@pytest.mark.filterwarnings("ignore:Consolidated metadata is currently not part:UserWarning")
@pytest.mark.parametrize("zarr_format", [2, 3])
def test_legacy_chunk_size_consolidated(tmp_path: Path, zarr_format: Literal[2, 3]) -> None:
    """Consolidated metadata goes through the same upgrade as the arrays' own documents,
    with one warning naming each array by its path; re-saving the arrays and
    consolidating again leaves a group that opens without a warning."""
    path = tmp_path / "group.zarr"
    group = zarr.open_group(path, mode="w", zarr_format=zarr_format)
    names = ("a", "b")
    for name in names:
        group.create_array(name, shape=(2,), chunks=(2,), dtype="int32")
    zarr.consolidate_metadata(path)

    # What zarr-python wrote for `chunks=(0,)`, in both copies.
    for name in names:
        if zarr_format == 2:
            _rewrite_doc(path / name, 2, lambda doc: doc.update(chunks=[0]))
            zmetadata = json.loads((path / ".zmetadata").read_text())
            zmetadata["metadata"][f"{name}/.zarray"]["chunks"] = [0]
            (path / ".zmetadata").write_text(json.dumps(zmetadata))
        else:
            _rewrite_doc(
                path / name,
                3,
                lambda doc: doc["chunk_grid"]["configuration"].update(chunk_shape=[0]),
            )
            _rewrite_doc(
                path,
                3,
                lambda doc, name=name: doc["consolidated_metadata"]["metadata"][name]["chunk_grid"][
                    "configuration"
                ].update(chunk_shape=[0]),
            )

    for use_consolidated in (True, False):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always", ZarrUserWarning)
            group = zarr.open_group(path, mode="r+", use_consolidated=use_consolidated)
            arrays = [group[name] for name in names]
        assert all("zarr.consolidate_metadata" in str(w.message) for w in record)
        assert sorted(str(w.message).split(": ")[0] for w in record) == [
            f"Array '{group.store_path / name}'" for name in names
        ]
    for array in arrays:
        assert isinstance(array, zarr.Array)
        assert array.chunks == (2,)
        array.update_attributes({})
    zarr.consolidate_metadata(path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ZarrUserWarning)
        warnings.filterwarnings("ignore", "Consolidated metadata is currently not part")
        for use_consolidated in (True, False):
            reopened = zarr.open_group(path, mode="r", use_consolidated=use_consolidated)
            for name in names:
                array = reopened[name]
                assert isinstance(array, zarr.Array)
                assert array.chunks == (2,)
