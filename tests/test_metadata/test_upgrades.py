"""Tests for the upgrades that read invalid stored array metadata documents."""

from __future__ import annotations

import json
import re
import warnings
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pytest

import zarr
from zarr.codecs import ShardingCodec
from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata
from zarr.core.metadata.upgrades import (
    RESAVE_HINT,
    V2_ARRAY_UPGRADES,
    V3_ARRAY_UPGRADES,
    upgrade_array_document,
)
from zarr.core.metadata.v3 import RectilinearChunkGridMetadata, RegularChunkGridMetadata
from zarr.dtype import Int16
from zarr.errors import ZarrUserWarning

if TYPE_CHECKING:
    from collections.abc import Callable
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


def _chunk_shapes(metadata: ArrayV2Metadata | ArrayV3Metadata) -> tuple[Any, Any]:
    """The chunk shape of `metadata` and, if it is sharded, its inner chunk shape."""
    if isinstance(metadata, ArrayV2Metadata):
        return metadata.chunks, None
    assert isinstance(metadata.chunk_grid, RegularChunkGridMetadata)
    inner = next((c.chunk_shape for c in metadata.codecs if isinstance(c, ShardingCodec)), None)
    return metadata.chunk_grid.chunk_shape, inner


@pytest.mark.parametrize(
    ("doc", "expected", "warning"),
    [
        (_v2_doc([10, 10], [4, 5]), ((4, 5), None), None),
        (_v3_doc([0, 0], [1, 1]), ((1, 1), None), None),
        (_v3_doc([10], [4], inner=[2]), ((4,), (2,)), None),
        (
            _v2_doc([0, 4], [0, 4]),
            ((1, 4), None),
            r"0 on axis 0 as one chunk spanning the axis \(1\)\.",
        ),
        (
            _v3_doc([0], [False]),
            ((1,), None),
            r"false on axis 0 as one chunk spanning the axis \(1\)\.",
        ),
        (_v2_doc([5], [True]), ((1,), None), "true on axis 0 as 1"),
        (
            _v3_doc([5, 4], [True, 4]),
            ((1, 4), None),
            r"\[true, 4\] is invalid.*true on axis 0 as 1",
        ),
        (
            _v2_doc([3], [0]),
            ((3,), None),
            r"spanning the axis \(3\), and .* holds only its fill value",
        ),
        (_v3_doc([4, 3], [4, 0]), ((4, 3), None), "0 on axis 1 as .* holds only its fill value"),
        (_v3_doc([0], [0], inner=[4]), ((4,), (4,)), r"spanning the axis \(4\)\."),
        (
            _v3_doc([10], [0], inner=[4]),
            ((12,), (4,)),
            r"spanning the axis \(12\), and .* holds only its fill value",
        ),
        (_v3_doc([0, 3], [0, 3], inner=[2, 3]), ((2, 3), (2, 3)), r"spanning the axis \(2\)\."),
        (
            _v3_doc([5], [True], inner=[True]),
            ((1,), (1,)),
            (
                r"^The stored inner chunk shape of the sharding codec \[true\] is invalid: .* "
                r"The stored chunk shape \[true\] is invalid: "
            ),
        ),
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
        "v3-sharded-true-inner-and-outer",
    ],
)
def test_upgrade_array_document(
    doc: dict[str, JSON], expected: tuple[Any, Any], warning: str | None
) -> None:
    """Valid documents pass unchanged and silently. A stored chunk size of 0 or `false`
    is read as one chunk spanning the axis (a multiple of the inner chunk when sharded)
    and `true` as 1, in the chunk shape and in a sharding codec's inner chunk shape;
    `from_dict` warns once for the document, naming the array, saying how each part was
    read (and that the array holds only its fill value where a chunk size of 0 was
    stored for a non-empty axis) and how to re-save."""
    upgrades = V2_ARRAY_UPGRADES if doc["zarr_format"] == 2 else V3_ARRAY_UPGRADES
    upgraded, readings = upgrade_array_document(doc, upgrades)
    assert {k: v for k, v in upgraded.items() if k not in ("chunks", "chunk_grid", "codecs")} == {
        k: v for k, v in doc.items() if k not in ("chunks", "chunk_grid", "codecs")
    }
    metadata_cls = ArrayV2Metadata if doc["zarr_format"] == 2 else ArrayV3Metadata
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        metadata = metadata_cls.from_dict(dict(doc), path="group/array")
    assert _chunk_shapes(metadata) == expected
    messages = [str(w.message) for w in record]
    if warning is None:
        assert upgraded is doc
        assert readings == []
        assert messages == []
    else:
        assert len(messages) == 1
        message = messages[0]
        assert message.startswith("Array 'group/array': ")
        assert re.search(warning, message.removeprefix("Array 'group/array': "))
        assert ("holds only its fill value" in message) == ("fill value" in warning)
        assert message.endswith(RESAVE_HINT)


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
    assert upgrade_array_document(doc, V2_ARRAY_UPGRADES + V3_ARRAY_UPGRADES)[1]
    with warnings.catch_warnings():
        warnings.simplefilter("error", ZarrUserWarning)
        with pytest.raises(ValueError, match=error):
            metadata_cls.from_dict(doc)


@pytest.mark.parametrize(
    ("doc", "error"),
    [
        (_v2_doc([4], [-1]), "Dimension 0: Chunk edge length must be >= 1, got -1"),
        (_v3_doc([4, 4], [0]), "Dimension 0: Chunk edge length must be >= 1, got 0"),
        (_v3_doc([4], [4.0]), "Dimension 0: Chunk edge length must be an int, got 4.0"),
        (_v3_doc([4], [0], inner=[0]), "Dimension 0: Chunk edge length must be >= 1, got 0"),
    ],
    ids=["negative", "ndim-mismatch", "float", "sharded-inner-zero"],
)
def test_stored_chunk_shape_not_upgraded(doc: dict[str, JSON], error: str) -> None:
    """Invalid chunk sizes no known writer stored, and chunk shapes with the wrong
    number of axes, are not upgraded, only rejected."""
    metadata_cls = ArrayV2Metadata if doc["zarr_format"] == 2 else ArrayV3Metadata
    with warnings.catch_warnings():
        warnings.simplefilter("error", ZarrUserWarning)
        with pytest.raises((TypeError, ValueError), match=re.escape(error)):
            metadata_cls.from_dict(doc)


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


CHUNK_EDGE_SITES: dict[str, Callable[[Any], object]] = {
    "regular": lambda size: RegularChunkGridMetadata(chunk_shape=(size,)),
    "v2": lambda size: _v2_metadata((size,)),
    "rectilinear-bare": lambda size: _rectilinear((size,)),
    "rectilinear-edge": lambda size: _rectilinear(((4, size),)),
    "rectilinear-bare-json": lambda size: _rectilinear_from_dict([size]),
    "rectilinear-edge-json": lambda size: _rectilinear_from_dict([[4, size]]),
    "rectilinear-rle-json": lambda size: _rectilinear_from_dict([[[size, 2]]]),
    "sharding-inner": lambda size: ShardingCodec(chunk_shape=(size,)),
}


@pytest.mark.parametrize("site", CHUNK_EDGE_SITES)
@pytest.mark.parametrize("size", [True, False, 4.0, np.int64(4), "4"])
def test_metadata_rejects_non_int_chunk_edge(site: str, size: object) -> None:
    """Metadata built in code takes chunk edge lengths as `int`s only, everywhere."""
    with pytest.raises(
        TypeError, match=re.escape(f"Chunk edge length must be an int, got {size!r}")
    ):
        CHUNK_EDGE_SITES[site](size)


@pytest.mark.parametrize("site", CHUNK_EDGE_SITES)
@pytest.mark.parametrize("size", [0, -1])
def test_metadata_rejects_chunk_edge_below_one(site: str, size: int) -> None:
    """Metadata built in code is strict: a chunk edge length below 1 is rejected,
    without a warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ZarrUserWarning)
        with pytest.raises(ValueError, match=f"Chunk edge length must be >= 1, got {size}"):
            CHUNK_EDGE_SITES[site](size)


@pytest.mark.parametrize("chunks", [4, np.int64(4)])
def test_v2_constructor_rejects_scalar_chunks(chunks: object) -> None:
    with pytest.raises(TypeError, match="A chunk shape must be a sequence of chunk edge lengths"):
        _v2_metadata(chunks)


def _rewrite_doc(path: Path, zarr_format: Literal[2, 3], edit: Any) -> None:
    doc_path = path / (".zarray" if zarr_format == 2 else "zarr.json")
    doc = json.loads(doc_path.read_text())
    edit(doc)
    doc_path.write_text(json.dumps(doc))


@pytest.mark.parametrize(
    ("zarr_format", "shape", "stored", "inner", "expected"),
    [
        (2, (0, 4), [0, 4], None, (1, 4)),
        (3, (5,), [True], None, (1,)),
        (3, (10,), [0], (4,), (12,)),
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

    with pytest.warns(ZarrUserWarning, match=r"^Array '.*legacy\.zarr': .* is read as"):
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
    """Consolidated metadata goes through the same upgrade, with one warning naming each
    array; re-saving the arrays and consolidating again leaves a group that opens
    without a warning."""
    path = tmp_path / "group.zarr"
    group = zarr.open_group(path, mode="w", zarr_format=zarr_format)
    names = ("a", "b")
    for name in names:
        group.create_array(name, shape=(0,), chunks=(1,), dtype="int32")
    zarr.consolidate_metadata(path)

    # What zarr-python wrote for `chunks=(0,)` on an empty array, in both copies.
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

    with pytest.warns(ZarrUserWarning, match="zarr.consolidate_metadata") as record:
        group = zarr.open_group(path, mode="r+")
    assert sorted(str(w.message).split(":")[0] for w in record) == ["Array 'a'", "Array 'b'"]
    for name in names:
        array = group[name]
        assert isinstance(array, zarr.Array)
        assert array.chunks == (1,)
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
                assert array.chunks == (1,)
