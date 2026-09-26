"""Tests for the upgrades that read invalid stored array metadata documents."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import re
import warnings
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pytest

import zarr
from zarr.codecs import ShardingCodec
from zarr.codecs.numcodecs import Quantize
from zarr.core.array import AsyncArray
from zarr.core.group import ConsolidatedMetadata
from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata
from zarr.core.metadata.upgrades import (
    RESAVE_HINT,
    upgrade_array_document,
)
from zarr.core.metadata.v3 import RectilinearChunkGridMetadata, RegularChunkGridMetadata
from zarr.core.sync import sync
from zarr.dtype import Int16
from zarr.errors import ZarrUserWarning
from zarr.storage import LocalStore, MemoryStore, StorePath
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
    with pytest.raises(ValueError, match="^Dimension 0: chunk edge length must be >= 1, got -1$"):
        _read_strictly(_v2_doc([4], [-1]))


def test_stored_chunk_shape_ndim_mismatch_rejected() -> None:
    """A chunk shape with the wrong number of dimensions is not upgraded, so its 0 is
    rejected."""
    with pytest.raises(ValueError, match="^Dimension 0: chunk edge length must be >= 1, got 0$"):
        _read_strictly(_v3_doc([4, 4], [0]))


def test_stored_zero_inner_chunk_size_rejected() -> None:
    """No known writer stored an inner chunk size of 0, and no span defines one: it is
    rejected, not upgraded."""
    with pytest.raises(ValueError, match="^Dimension 0: chunk edge length must be >= 1, got 0$"):
        _read_strictly(_v3_doc([4], [4], inner=[0]))


@pytest.mark.parametrize("inner", [[0], [False]])
def test_stored_zero_chunk_size_of_shard_with_invalid_inner_chunk_shape_rejected(
    inner: list[Any],
) -> None:
    """A stored chunk size of 0 of a sharded array is read in multiples of the inner
    chunk size; if that is not an integer of at least 1, the 0 is not upgraded, so it
    is rejected."""
    with pytest.raises(ValueError, match="^Dimension 0: chunk edge length must be >= 1, got 0$"):
        _read_strictly(_v3_doc([4], [0], inner=inner))


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
            "Dimension 0: chunk edge length must be an int, got 2.0",
        ),
        (_v2_doc([20], [10.0]), "Dimension 0: chunk edge length must be an int, got 10.0"),
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
    "v2": lambda size: _v2_metadata((size,)).chunks[0],
    "rectilinear-bare": lambda size: _rectilinear((size,)).chunk_shapes[0],
    "rectilinear-edge": lambda size: _second_edge(_rectilinear(((4, size),))),
    "rectilinear-bare-json": lambda size: _rectilinear_from_dict([size]).chunk_shapes[0],
    "rectilinear-edge-json": lambda size: _second_edge(_rectilinear_from_dict([[4, size]])),
    "rectilinear-rle-json": lambda size: _second_edge(_rectilinear_from_dict([[[size, 2]]])),
    "sharding-inner": lambda size: ShardingCodec(chunk_shape=(size,)).chunk_shape[0],
}
"""Each place metadata built in code takes a chunk edge length, returning the edge it
took."""


@pytest.mark.parametrize("site", CHUNK_EDGE_SITES)
def test_metadata_takes_int_chunk_edge(site: str) -> None:
    """Metadata built in code takes an `int` chunk edge length of at least 1 as is."""
    edge = CHUNK_EDGE_SITES[site](4)
    assert type(edge) is int
    assert edge == 4


@pytest.mark.parametrize("site", CHUNK_EDGE_SITES)
@pytest.mark.parametrize(
    "size",
    [True, False, 4.0, np.float64(4.0), 4.5, float("inf"), "4", None, np.int64(4)],
    ids=["true", "false", "float", "numpy-float", "fractional", "inf", "str", "none", "numpy-int"],
)
def test_metadata_rejects_non_int_chunk_edge(site: str, size: object) -> None:
    """Metadata built in code takes chunk edge lengths as `int`s only, everywhere: not a
    `bool`, a float (stored documents with integral floats are read by the upgrades) or
    a NumPy integer."""
    with pytest.raises(
        TypeError, match=re.escape(f"Dimension 0: chunk edge length must be an int, got {size!r}")
    ):
        CHUNK_EDGE_SITES[site](size)


@pytest.mark.parametrize("site", CHUNK_EDGE_SITES)
@pytest.mark.parametrize("size", [0, -1])
def test_metadata_rejects_chunk_edge_below_one(site: str, size: int) -> None:
    """Metadata built in code is strict: a chunk edge length below 1 is rejected,
    without a warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ZarrUserWarning)
        with pytest.raises(
            ValueError, match=f"Dimension 0: chunk edge length must be >= 1, got {size}"
        ):
            CHUNK_EDGE_SITES[site](size)


CHUNK_SHAPE_SITES: dict[str, Callable[[Any], object]] = {
    "regular": lambda chunk_shape: RegularChunkGridMetadata(chunk_shape=chunk_shape),
    "v2": _v2_metadata,
    "sharding-inner": lambda chunk_shape: ShardingCodec(chunk_shape=chunk_shape),
}


@pytest.mark.parametrize("site", CHUNK_SHAPE_SITES)
@pytest.mark.parametrize("chunk_shape", [4, np.int64(4), "10", {"a": 1}, range(1, 2)])
def test_metadata_rejects_chunk_shape_not_list_or_tuple(site: str, chunk_shape: object) -> None:
    """A regular chunk shape is a list or tuple; anything else is rejected as a whole,
    not iterated as if its elements were chunk edge lengths."""
    with pytest.raises(
        TypeError,
        match=re.escape(
            f"A chunk shape must be a list or tuple of chunk edge lengths, got {chunk_shape!r}"
        ),
    ):
        CHUNK_SHAPE_SITES[site](chunk_shape)


def test_consolidated_member_error_names_member() -> None:
    """A group whose consolidated metadata holds a member that cannot be read does not
    open from it; the error names the member."""
    member = {**_v2_doc([5], [4]), "chunks": 4}
    with pytest.raises(TypeError, match="A chunk shape must be a list or tuple") as info:
        ConsolidatedMetadata.from_dict(
            {"kind": "inline", "must_understand": False, "metadata": {"sub/a": member}},
            path="group",
        )
    assert info.value.__notes__ == ["Member 'group/sub/a' of the consolidated metadata."]


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


def _store_zero(doc: dict[str, Any]) -> None:
    _stored_chunks(doc)[0] = 0


def _resize_to_10(doc: dict[str, Any]) -> None:
    doc["shape"] = [10]


def _halve_inner_chunk_shape(doc: dict[str, Any]) -> None:
    doc["codecs"][0]["configuration"]["chunk_shape"] = [2]


@pytest.mark.parametrize(
    ("zarr_format", "sharded", "change"),
    [(2, False, _resize_to_10), (3, False, _resize_to_10), (3, True, _halve_inner_chunk_shape)],
    ids=["v2-resized", "v3-resized", "v3-sharded-inner-chunk-shape"],
)
def test_stale_handle_write_after_chunk_grid_change_raises(
    tmp_path: Path,
    zarr_format: Literal[2, 3],
    sharded: bool,
    change: Callable[[dict[str, Any]], None],
) -> None:
    """If the document the store holds when a handle read from an upgraded document
    first writes chunks lays out chunks differently from the handle's metadata (the
    array was resized by software that kept the stored chunk size of 0, which now reads
    as a larger chunk, or its inner chunk shape changed), the handle's chunks would not
    be found under it: the write raises and stores nothing."""
    path = tmp_path / "legacy.zarr"
    if sharded:
        zarr.create_array(store=path, shape=(3,), chunks=(4,), shards=(4,), dtype="int16")
        _rewrite_doc(path, 3, _store_zero)
    else:
        _legacy_array(path, zarr_format)
    with pytest.warns(ZarrUserWarning, match="is read as"):
        stale = zarr.open_array(store=path, mode="r+")
    _rewrite_doc(path, zarr_format, change)
    documents = {p.name: p.read_bytes() for p in path.iterdir()}

    with pytest.raises(ValueError, match="has changed since this array was opened; reopen"):
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


def test_array_from_metadata_with_numpy_scalar_codec_configuration() -> None:
    """An array is built from metadata whose codec configuration holds NumPy scalars
    (which are not JSON values), as zarr always built one."""
    array = zarr.create_array(
        MemoryStore(), shape=(4,), chunks=(2,), dtype="f8", filters=[Quantize(digits=3, dtype="f8")]
    )
    assert isinstance(array.metadata, ArrayV3Metadata)
    # The codec keeps its configuration as given, though it is typed as JSON.
    digits = cast("JSON", np.int64(3))
    codecs = (Quantize(digits=digits, dtype="f8"), *array.metadata.codecs[1:])
    metadata = dataclasses.replace(array.metadata, codecs=codecs)

    assert AsyncArray(metadata, StorePath(MemoryStore())).metadata is metadata


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


def _flagged_consolidated_group(path: Path, zarr_format: Literal[2, 3], member: str) -> None:
    """A group whose consolidated copy of the array `member` holds the stored chunk size
    0, and whose array `b` is valid."""
    group = zarr.open_group(path, mode="w", zarr_format=zarr_format)
    parent, _, name = member.rpartition("/")
    (group.require_group(parent) if parent else group).create_array(
        name, shape=(3,), chunks=(3,), dtype="int16"
    )
    group.create_array("b", shape=(1,), chunks=(1,), dtype="int16")
    zarr.consolidate_metadata(path)
    _rewrite_consolidated(path, zarr_format, member, _store_zero)


@pytest.mark.filterwarnings("ignore:Consolidated metadata is currently not part:UserWarning")
@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("member", ["a", "g/a"])
@pytest.mark.parametrize("operation", ["attrs", "update_attributes_async", "delete-member"])
def test_group_write_refreshes_upgraded_consolidated_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    zarr_format: Literal[2, 3],
    member: str,
    operation: str,
) -> None:
    """Storing a group's metadata stores its consolidated metadata. Each member of it read
    from a document that had to be upgraded, at any depth, is first read again from the
    member's own document as it now is (here resized by software that kept the stored
    chunk size of 0), whose upgrade is stored, so no group write stores a stale copy as
    if valid. The group adopts the members it stored, so later writes read no member."""
    path = tmp_path / "group.zarr"
    _flagged_consolidated_group(path, zarr_format, member)

    def resize_keeping_zero(doc: dict[str, Any]) -> None:
        _store_zero(doc)
        doc["shape"] = [10]

    _rewrite_doc(path / member, zarr_format, resize_keeping_zero)
    with pytest.warns(ZarrUserWarning, match="is read as"):
        group = zarr.open_group(path, mode="r+", use_consolidated=True)

    if operation == "attrs":
        group.attrs["x"] = 1
    elif operation == "update_attributes_async":
        group = sync(group.update_attributes_async({"x": 1}))
    else:
        del group["b"]

    assert _stored_chunks(_consolidated_member(path, zarr_format, member)) == [10]
    reopened = zarr.open_group(path, mode="r", use_consolidated=True)
    array = reopened[member]
    assert isinstance(array, zarr.Array)
    assert (array.shape, array.chunks) == ((10,), (10,))
    assert _open_strictly(path / member).chunks == (10,)

    reads: list[str] = []
    get = LocalStore.get

    async def recording_get(self: LocalStore, key: str, *args: Any, **kwargs: Any) -> Any:
        reads.append(key)
        return await get(self, key, *args, **kwargs)

    monkeypatch.setattr(LocalStore, "get", recording_get)
    group.attrs["y"] = 2
    assert reads == []


@pytest.mark.filterwarnings("ignore:Consolidated metadata is currently not part:UserWarning")
@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("replacement", ["group", "invalid", "none"])
def test_group_write_keeps_upgraded_member_without_readable_document(
    tmp_path: Path, zarr_format: Literal[2, 3], replacement: str
) -> None:
    """A member of consolidated metadata read from a document that had to be upgraded,
    whose own document is no longer an array document that can be read, has no document
    to upgrade: a group write stores the consolidated copy as it is."""
    path = tmp_path / "group.zarr"
    _flagged_consolidated_group(path, zarr_format, "a")
    with pytest.warns(ZarrUserWarning, match="is read as"):
        group = zarr.open_group(path, mode="r+", use_consolidated=True)
    del zarr.open_group(path, mode="r+", use_consolidated=False)["a"]
    if replacement == "group":
        zarr.create_group(path / "a", zarr_format=zarr_format)
    elif replacement == "invalid":
        (path / "a").mkdir()
        document = path / "a" / (".zarray" if zarr_format == 2 else "zarr.json")
        document.write_text(json.dumps({"zarr_format": zarr_format, "node_type": "array"}))

    group.attrs["x"] = 1

    assert _stored_chunks(_consolidated_member(path, zarr_format, "a")) == [3]


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
    ("doc", "expected", "warning"),
    [
        (json.loads(MIXED_REGULAR_GRID_DOC), (2, (5, 10, 5)), r"^The stored chunk grid .* \[1\]"),
        (
            _mixed_doc([6, 20, 4], [2, [5, 10, 5], [1, 3]]),
            (2, (5, 10, 5), (1, 3)),
            r"^The stored chunk grid .* in dimensions \[1, 2\]",
        ),
        (_mixed_doc([6, 20], [2, [20]]), (2, (20,)), r"^The stored chunk grid .* \[1\]"),
        (_mixed_doc([6, 12], [2, [5, 10, 5]]), (2, (5, 10, 5)), r"^The stored chunk grid"),
        (_mixed_doc([0, 20], [2, [5, 10, 5]]), (2, (5, 10, 5)), r"^The stored chunk grid"),
        (_mixed_doc([6, 0], [2, [5, 10, 5]]), (2, (5, 10, 5)), r"^The stored chunk grid"),
        # JSON true is read as 1 silently; only the grid reading warns.
        (_mixed_doc([6, 20], [True, [5, 10, 5]]), (1, (5, 10, 5)), r"^The stored chunk grid"),
        (_mixed_doc([6, 20], [2, [5.0, 10.0, 5.0]]), (2, (5, 10, 5)), r"^The stored chunk grid"),
        (
            _mixed_doc([4, 10_000], [0, [10] * 1000]),
            (4, (10,) * 1000),
            r"^The stored chunk shape \[0, \[10, 10, .*\.\.\. is invalid: .* read as \[4, \[10, .*\.\.\.,",
        ),
    ],
    ids=[
        "written",
        "3d",
        "one-edge",
        "shrunk",
        "empty-int-axis",
        "empty-edge-axis",
        "true",
        "float-edges",
        "long",
    ],
)
def test_read_edge_lists_in_regular_grid(
    doc: dict[str, JSON], expected: tuple[int | tuple[int, ...], ...], warning: str
) -> None:
    """A `regular` chunk grid whose chunk shape mixes chunk sizes with lists of chunk
    edge lengths is read as the rectilinear chunk grid it describes, without the
    rectilinear chunks flag. `from_dict` warns once, naming the array and the axes,
    quoting a bounded part of the chunk shape, and saying that re-saving requires the
    flag."""
    with (
        zarr.config.set({"array.rectilinear_chunks": False}),
        warnings.catch_warnings(record=True) as record,
    ):
        warnings.simplefilter("always")
        metadata = ArrayV3Metadata.from_dict(doc, path="group/array")
    assert metadata.chunk_grid == RectilinearChunkGridMetadata(chunk_shapes=expected)
    [message] = [str(w.message) for w in record]
    assert re.search(warning, message.removeprefix("Array 'group/array': "))
    assert message.startswith("Array 'group/array': ")
    assert (
        "Re-saving the metadata stores that rectilinear chunk grid, so each step that "
        "follows requires `zarr.config.set({'array.rectilinear_chunks': True})`. "
    ) in message
    assert message.endswith(RESAVE_HINT)
    assert len(message) < 1000


def _rejected_without_warning(doc: dict[str, JSON]) -> pytest.ExceptionInfo[Exception]:
    with warnings.catch_warnings():
        warnings.simplefilter("error", ZarrUserWarning)
        with pytest.raises((TypeError, ValueError)) as info:
            ArrayV3Metadata.from_dict(doc)
    return info


def test_regular_grid_of_only_edge_lists_rejected() -> None:
    """A regular chunk shape made only of edge lists was never stored (a rectilinear
    chunk grid was), so it is not read as rectilinear."""
    info = _rejected_without_warning(_mixed_doc([6, 20], [[1, 5], [5, 10, 5]]))
    assert info.match(re.escape("Dimension 0: chunk edge length must be an int, got [1, 5]"))


def test_run_length_encoded_edges_in_regular_grid_rejected() -> None:
    """Run-length encoded edges were never stored in a regular chunk shape."""
    info = _rejected_without_warning(_mixed_doc([6, 20], [2, [[5, 2], 10]]))
    assert info.match(re.escape("Dimension 1: chunk edge length must be an int, got [[5, 2], 10]"))


@pytest.mark.parametrize("edge", [5.5, "5"], ids=["fractional", "string"])
def test_non_int_edge_in_regular_grid_rejected(edge: object) -> None:
    """An edge that is not an int is reported as such, not blamed on its list."""
    info = _rejected_without_warning(_mixed_doc([6, 20], [2, [edge, 15]]))
    assert info.match(re.escape(f"Dimension 1: chunk edge length must be an int, got {edge!r}"))


def test_edge_below_one_in_regular_grid_rejected() -> None:
    info = _rejected_without_warning(_mixed_doc([6, 20], [2, [0, 20]]))
    assert info.match("Dimension 1: chunk edge length must be >= 1, got 0")


def test_short_edges_in_regular_grid_rejected() -> None:
    info = _rejected_without_warning(_mixed_doc([6, 20], [2, [5, 10]]))
    assert info.match("sum to 15 but array shape extent is 20")


def _store_mixed_array(path: Path) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Write the chunks of the verbatim document and the document itself at `path`."""
    data = np.arange(120, dtype="float32").reshape(6, 20)
    with zarr.config.set({"array.rectilinear_chunks": True}):
        arr = zarr.create_array(path, shape=data.shape, chunks=(2, (5, 10, 5)), dtype="float32")
        arr[...] = data
    (path / "zarr.json").write_text(MIXED_REGULAR_GRID_DOC)
    return data


def _update_attributes(arr: zarr.Array[Any], data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    arr.update_attributes({})
    return data


def _write(arr: zarr.Array[Any], data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    arr[:2] = -data[:2]
    return np.concatenate([-data[:2], data[2:]])


@pytest.mark.parametrize("store_metadata", [_update_attributes, _write], ids=["re-save", "write"])
def test_edge_lists_in_regular_grid_round_trip(
    tmp_path: Path,
    store_metadata: Callable[[zarr.Array[Any], np.ndarray[Any, Any]], np.ndarray[Any, Any]],
) -> None:
    """A store holding the verbatim document opens without the rectilinear chunks flag
    and reads its data. With the flag, re-saving the metadata, or writing chunks (which
    stores the metadata first), stores the rectilinear chunk grid, which then opens
    cleanly."""
    path = tmp_path / "mixed.zarr"
    data = _store_mixed_array(path)

    with (
        zarr.config.set({"array.rectilinear_chunks": False}),
        pytest.warns(ZarrUserWarning, match="read as that rectilinear chunk grid"),
    ):
        arr = zarr.open_array(path, mode="a")
    np.testing.assert_array_equal(arr[...], data)

    with zarr.config.set({"array.rectilinear_chunks": True}):
        expected = store_metadata(arr, data)
        assert json.loads((path / "zarr.json").read_text())["chunk_grid"] == {
            "name": "rectilinear",
            "configuration": {"kind": "inline", "chunk_shapes": [2, [5, 10, 5]]},
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error", ZarrUserWarning)
            reopened = zarr.open_array(path)
    np.testing.assert_array_equal(reopened[...], expected)


def _store_mixed_group(path: Path) -> None:
    """A group at `path` holding the verbatim document at `mixed` and a regular array
    `n`, with consolidated metadata that quotes the verbatim document."""
    group = zarr.open_group(path, mode="w")
    _store_mixed_array(path / "mixed")
    group.create_array("n", data=np.arange(4), chunks=(2,))
    with zarr.config.set({"array.rectilinear_chunks": True}):
        zarr.consolidate_metadata(path)
    group_doc = json.loads((path / "zarr.json").read_text())
    group_doc["consolidated_metadata"]["metadata"]["mixed"] = json.loads(MIXED_REGULAR_GRID_DOC)
    (path / "zarr.json").write_text(json.dumps(group_doc))


def _resize(path: Path) -> None:
    zarr.open_array(path / "mixed", mode="a").resize((6, 5))


def _write_chunks(path: Path) -> None:
    zarr.open_array(path / "mixed", mode="a")[...] = 1


def _delete_member(path: Path) -> None:
    del zarr.open_group(path, mode="a")["n"]


def _overwrite_hierarchy(path: Path) -> None:
    mixed = zarr.open_array(path / "mixed")
    list(zarr.create_hierarchy(store=LocalStore(path), nodes={"n": mixed.metadata}, overwrite=True))


def _consolidate(path: Path) -> None:
    zarr.consolidate_metadata(path)


def _set_group_attribute(path: Path) -> None:
    zarr.open_group(path, mode="a").attrs["x"] = 1


@pytest.mark.filterwarnings(
    "ignore:.*read as that rectilinear chunk grid:zarr.errors.ZarrUserWarning"
)
@pytest.mark.filterwarnings("ignore:Consolidated metadata is currently not part:UserWarning")
@pytest.mark.parametrize(
    "action",
    [
        _resize,
        _write_chunks,
        _delete_member,
        _overwrite_hierarchy,
        _consolidate,
        _set_group_attribute,
    ],
    ids=["resize", "write", "delete-member", "overwrite-hierarchy", "consolidate", "group-attrs"],
)
def test_store_untouched_without_flag(tmp_path: Path, action: Callable[[Path], None]) -> None:
    """An operation that would store the rectilinear chunk grid read from the verbatim
    document fails without the flag before it deletes or writes anything."""
    path = tmp_path / "group.zarr"
    _store_mixed_group(path)
    stored = {p: p.read_bytes() for p in path.rglob("*") if p.is_file()}
    with (
        zarr.config.set({"array.rectilinear_chunks": False}),
        pytest.raises(ValueError, match="experimental and disabled by default"),
    ):
        action(path)
    assert {p: p.read_bytes() for p in path.rglob("*") if p.is_file()} == stored


@pytest.mark.filterwarnings("ignore:Consolidated metadata is currently not part:UserWarning")
@pytest.mark.parametrize("member", ["mixed", "sub/mixed"])
def test_consolidate_edge_lists_in_regular_grid(tmp_path: Path, member: str) -> None:
    """Consolidating a group holding the verbatim document stores the member's
    rectilinear chunk grid, so without the flag it fails, naming the array, and stores
    nothing, so the group still opens without the flag; with the flag, the member and
    the consolidated metadata store the rectilinear chunk grid."""
    path = tmp_path / "group.zarr"
    zarr.open_group(path, mode="w").create_group("sub")
    data = _store_mixed_array(path / member)
    group_path = str(sync(make_store_path(path)))
    with zarr.config.set({"array.rectilinear_chunks": False}):
        with (
            pytest.warns(ZarrUserWarning, match="read as that rectilinear chunk grid"),
            pytest.raises(ValueError, match="experimental and disabled") as info,
        ):
            zarr.consolidate_metadata(path)
        assert info.value.__notes__ == [
            f"Array {member!r} in the consolidated metadata.",
            f"Group {group_path!r}: nothing was stored.",
        ]
        with pytest.warns(ZarrUserWarning, match="read as that rectilinear chunk grid"):
            mixed = zarr.open_group(path, mode="r")[member]
    assert isinstance(mixed, zarr.Array)
    np.testing.assert_array_equal(mixed[...], data)

    rectilinear = {
        "name": "rectilinear",
        "configuration": {"kind": "inline", "chunk_shapes": [2, [5, 10, 5]]},
    }
    with zarr.config.set({"array.rectilinear_chunks": True}):
        with pytest.warns(ZarrUserWarning, match="read as that rectilinear chunk grid"):
            zarr.consolidate_metadata(path)
        group_doc = json.loads((path / "zarr.json").read_text())
        consolidated = group_doc["consolidated_metadata"]["metadata"][member]
        assert consolidated["chunk_grid"] == rectilinear
        assert json.loads((path / member / "zarr.json").read_text())["chunk_grid"] == rectilinear
        with warnings.catch_warnings():
            warnings.simplefilter("error", ZarrUserWarning)
            reopened = zarr.open_group(path, mode="r")[member]
    assert isinstance(reopened, zarr.Array)
    np.testing.assert_array_equal(reopened[...], data)


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
@pytest.mark.filterwarnings("ignore:Consolidated metadata is currently not part:UserWarning")
def test_group_write_without_flag_stores_no_member_upgrade(tmp_path: Path) -> None:
    """A group write refused without the flag stores no upgrade of another consolidated
    member either (here `a`, stored with chunk size 0), although that one alone could be
    stored: every member is refreshed and the group encoded before anything is stored."""
    path = tmp_path / "group.zarr"
    _store_mixed_group(path)
    zarr.open_group(path, mode="a").create_array("a", shape=(4,), chunks=(4,), dtype="int16")
    with zarr.config.set({"array.rectilinear_chunks": True}):
        zarr.consolidate_metadata(path)
    _rewrite_doc(path / "a", 3, _store_zero)
    _rewrite_consolidated(path, 3, "a", _store_zero)
    # Consolidating with the flag stored the rectilinear grid: restore the verbatim document.
    (path / "mixed" / "zarr.json").write_text(MIXED_REGULAR_GRID_DOC)
    _rewrite_consolidated(
        path, 3, "mixed", lambda doc: doc.update(json.loads(MIXED_REGULAR_GRID_DOC))
    )
    stored = {p: p.read_bytes() for p in path.rglob("*") if p.is_file()}
    with (
        zarr.config.set({"array.rectilinear_chunks": False}),
        pytest.raises(ValueError, match="experimental and disabled by default"),
    ):
        _set_group_attribute(path)
    assert {p: p.read_bytes() for p in path.rglob("*") if p.is_file()} == stored


@pytest.mark.filterwarnings(
    "ignore:.*read as that rectilinear chunk grid:zarr.errors.ZarrUserWarning"
)
@pytest.mark.filterwarnings("ignore:Consolidated metadata is currently not part:UserWarning")
@pytest.mark.parametrize("action", [_delete_member, _set_group_attribute], ids=["delete", "attrs"])
def test_group_write_stores_mixed_member_with_flag(
    tmp_path: Path, action: Callable[[Path], None]
) -> None:
    """With the flag, a group write stores the rectilinear chunk grid read from the
    verbatim document, in the member's own document and in the consolidated metadata,
    which then open cleanly."""
    path = tmp_path / "group.zarr"
    _store_mixed_group(path)
    rectilinear = {
        "name": "rectilinear",
        "configuration": {"kind": "inline", "chunk_shapes": [2, [5, 10, 5]]},
    }
    with zarr.config.set({"array.rectilinear_chunks": True}):
        action(path)
        group_doc = json.loads((path / "zarr.json").read_text())
        assert group_doc["consolidated_metadata"]["metadata"]["mixed"]["chunk_grid"] == rectilinear
        assert json.loads((path / "mixed" / "zarr.json").read_text())["chunk_grid"] == rectilinear
        with warnings.catch_warnings():
            warnings.simplefilter("error", ZarrUserWarning)
            for use_consolidated in (True, False):
                mixed = zarr.open_group(path, use_consolidated=use_consolidated)["mixed"]
                assert isinstance(mixed, zarr.Array)
                assert mixed.read_chunk_sizes == ((2, 2, 2), (5, 10, 5))


@pytest.mark.filterwarnings(
    "ignore:.*read as that rectilinear chunk grid:zarr.errors.ZarrUserWarning"
)
@pytest.mark.filterwarnings("ignore:Consolidated metadata is currently not part:UserWarning")
def test_delete_member_without_flag_keeps_group(tmp_path: Path) -> None:
    """A deletion that fails without the flag leaves the group listing the member."""
    path = tmp_path / "group.zarr"
    _store_mixed_group(path)
    with zarr.config.set({"array.rectilinear_chunks": False}):
        group = zarr.open_group(path, mode="a")
        with pytest.raises(ValueError, match="experimental and disabled"):
            del group["n"]
    assert group.metadata.consolidated_metadata is not None
    assert sorted(group.metadata.consolidated_metadata.metadata) == ["mixed", "n"]
