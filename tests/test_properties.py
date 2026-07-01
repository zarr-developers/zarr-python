import itertools
import json
import numbers
from collections.abc import Generator
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import zarr
from zarr.core.buffer import default_buffer_prototype

pytest.importorskip("hypothesis")

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
from hypothesis import assume, given, settings
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from tests.conftest import Expect
from zarr.abc.store import Store
from zarr.core.common import ZARR_JSON, ZARRAY_JSON, ZATTRS_JSON
from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata
from zarr.core.sync import sync
from zarr.testing.strategies import (
    IndexMode,
    array_metadata,
    arrays,
    basic_indices,
    block_indices,
    block_test_arrays,
    complex_rectilinear_arrays,
    numpy_array_indexers,
    numpy_arrays,
    rectilinear_arrays,
    simple_arrays,
    stores,
    zarr_formats,
)


@pytest.fixture(autouse=True)
def _enable_rectilinear_chunks() -> Generator[None, None, None]:
    """Enable rectilinear chunks for all property tests since strategies may generate them."""
    with zarr.config.set({"array.rectilinear_chunks": True}):
        yield


def deep_equal(a: Any, b: Any) -> bool:
    """Deep equality check with handling of special cases for array metadata classes"""
    if isinstance(a, (complex, np.complexfloating)) and isinstance(
        b, (complex, np.complexfloating)
    ):
        a_real, a_imag = float(a.real), float(a.imag)
        b_real, b_imag = float(b.real), float(b.imag)
        if np.isnan(a_real) and np.isnan(b_real):
            real_eq = True
        else:
            real_eq = a_real == b_real
        if np.isnan(a_imag) and np.isnan(b_imag):
            imag_eq = True
        else:
            imag_eq = a_imag == b_imag
        return real_eq and imag_eq

    if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
        if np.isnan(a) and np.isnan(b):
            return True
        return a == b

    if isinstance(a, np.datetime64) and isinstance(b, np.datetime64):
        if np.isnat(a) and np.isnat(b):
            return True
        return a == b

    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.shape != b.shape:
            return False
        return all(itertools.starmap(deep_equal, zip(a.flat, b.flat, strict=False)))

    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(deep_equal(a[k], b[k]) for k in a)

    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(itertools.starmap(deep_equal, zip(a, b, strict=False)))

    return a == b


@pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
@given(data=st.data())
def test_array_roundtrip(data: st.DataObject) -> None:
    nparray = data.draw(numpy_arrays())
    zarray = data.draw(arrays(arrays=st.just(nparray)))
    assert_array_equal(nparray, zarray[:])


@pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
@given(array=arrays())
def test_array_creates_implicit_groups(array):
    path = array.path
    ancestry = path.split("/")[:-1]
    for i in range(len(ancestry)):
        parent = "/".join(ancestry[: i + 1])
        if array.metadata.zarr_format == 2:
            assert (
                sync(array.store.get(f"{parent}/.zgroup", prototype=default_buffer_prototype()))
                is not None
            )
        elif array.metadata.zarr_format == 3:
            assert (
                sync(array.store.get(f"{parent}/zarr.json", prototype=default_buffer_prototype()))
                is not None
            )


# Eager basic/oindex/vindex/mask indexing is exercised by the per-mode read/write
# parity tests below (test_basic_read/test_basic_write, test_oindex_*, ...), each
# comparing against NumPy. Special array shapes those strategies don't cover keep
# dedicated tests (complex rectilinear arrays here; block indexing later).


@settings(deadline=None)
@pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
@given(data=st.data())
async def test_basic_indexing_complex_rectilinear(data: st.DataObject) -> None:
    nparray, zarray = data.draw(complex_rectilinear_arrays())
    indexer = data.draw(basic_indices(shape=nparray.shape))
    assert_array_equal(nparray[indexer], zarray[indexer])


@settings(deadline=None)
@pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
@given(data=st.data())
def test_block_indexing(data: st.DataObject) -> None:
    # Block indexing addresses whole inner chunks. block_indices() builds its
    # array-space oracle from cumulative chunk offsets, so it works for regular
    # (uniform), rectilinear, and sharded grids alike; block_test_arrays draws
    # across that matrix (rectilinear + sharded is unsupported and not drawn).
    zarray, nparray = data.draw(block_test_arrays())

    block_indexer, array_indexer = data.draw(block_indices(chunk_sizes=zarray.write_chunk_sizes))
    expected = nparray[array_indexer]

    # sync get, via both the .blocks interface and the dedicated method
    assert_array_equal(expected, zarray.blocks[block_indexer])
    assert_array_equal(expected, zarray.get_block_selection(block_indexer))

    # sync set, via both interfaces; sharded set is broken upstream (GH2834)
    assume(zarray.shards is None)
    new_data = data.draw(numpy_arrays(shapes=st.just(expected.shape), dtype=nparray.dtype))
    nparray[array_indexer] = new_data
    zarray.blocks[block_indexer] = new_data
    assert_array_equal(nparray, zarray[:])

    zarray.set_block_selection(block_indexer, new_data)
    assert_array_equal(nparray, zarray[:])


@given(store=stores, meta=array_metadata())  # type: ignore[misc]
@pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
async def test_roundtrip_array_metadata_from_store(
    store: Store, meta: ArrayV2Metadata | ArrayV3Metadata
) -> None:
    """
    Verify that the I/O for metadata in a store are lossless.

    This test serializes an ArrayV2Metadata or ArrayV3Metadata object to a dict
    of buffers via `to_buffer_dict`, writes each buffer to a store under keys
    prefixed with "0/", and then reads them back. The test asserts that each
    retrieved buffer exactly matches the original buffer.
    """
    asdict = meta.to_buffer_dict(prototype=default_buffer_prototype())
    for key, expected in asdict.items():
        await store.set(f"0/{key}", expected)
        actual = await store.get(f"0/{key}", prototype=default_buffer_prototype())
        assert actual == expected


@given(data=st.data(), zarr_format=zarr_formats)
@pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
def test_roundtrip_array_metadata_from_json(data: st.DataObject, zarr_format: int) -> None:
    """
    Verify that JSON serialization and deserialization of metadata is lossless.

    For Zarr v2:
      - The metadata is split into two JSON documents (one for array data and one
        for attributes). The test merges the attributes back before deserialization.
    For Zarr v3:
      - All metadata is stored in a single JSON document. No manual merger is necessary.

    The test then converts both the original and round-tripped metadata objects
    into dictionaries using `dataclasses.asdict` and uses a deep equality check
    to verify that the roundtrip has preserved all fields (including special
    cases like NaN, Infinity, complex numbers, and datetime values).
    """
    metadata = data.draw(array_metadata(zarr_formats=st.just(zarr_format)))
    buffer_dict = metadata.to_buffer_dict(prototype=default_buffer_prototype())

    if zarr_format == 2:
        zarray_dict = json.loads(buffer_dict[ZARRAY_JSON].to_bytes().decode())
        zattrs_dict = json.loads(buffer_dict[ZATTRS_JSON].to_bytes().decode())
        # zattrs and zarray are separate in v2, we have to add attributes back prior to `from_dict`
        zarray_dict["attributes"] = zattrs_dict
        metadata_roundtripped = ArrayV2Metadata.from_dict(zarray_dict)
    else:
        zarray_dict = json.loads(buffer_dict[ZARR_JSON].to_bytes().decode())
        metadata_roundtripped = ArrayV3Metadata.from_dict(zarray_dict)

    orig = metadata.to_dict()
    rt = metadata_roundtripped.to_dict()

    assert deep_equal(orig, rt), f"Roundtrip mismatch:\nOriginal: {orig}\nRoundtripped: {rt}"


# @st.composite
# def advanced_indices(draw, *, shape):
#     basic_idxr = draw(
#         basic_indices(
#             shape=shape, min_dims=len(shape), max_dims=len(shape), allow_ellipsis=False
#         ).filter(lambda x: isinstance(x, tuple))
#     )

#     int_idxr = draw(
#         npst.integer_array_indices(shape=shape, result_shape=npst.array_shapes(max_dims=1))
#     )
#     args = tuple(
#         st.sampled_from((l, r)) for l, r in zip_longest(basic_idxr, int_idxr, fillvalue=slice(None))
#     )
#     return draw(st.tuples(*args))


# @given(st.data())
# def test_roundtrip_object_array(data):
#     nparray = data.draw(np_arrays)
#     zarray = data.draw(arrays(arrays=st.just(nparray)))
#     assert_array_equal(nparray, zarray[:])


def serialized_complex_float_is_valid(
    serialized: tuple[numbers.Real | str, numbers.Real | str],
) -> bool:
    """
    Validate that the serialized representation of a complex float conforms to the spec.

    The specification requires that a serialized complex float must be either:
      - A JSON number, or
      - One of the strings "NaN", "Infinity", or "-Infinity".

    Args:
        serialized: The value produced by JSON serialization for a complex floating point number.

    Returns:
        bool: True if the serialized value is valid according to the spec, False otherwise.
    """
    return (
        isinstance(serialized, tuple)
        and len(serialized) == 2
        and all(serialized_float_is_valid(x) for x in serialized)
    )


def serialized_float_is_valid(serialized: numbers.Real | str) -> bool:
    """
    Validate that the serialized representation of a float conforms to the spec.

    The specification requires that a serialized float must be either:
      - A JSON number, or
      - One of the strings "NaN", "Infinity", or "-Infinity".

    Args:
        serialized: The value produced by JSON serialization for a floating point number.

    Returns:
        bool: True if the serialized value is valid according to the spec, False otherwise.
    """
    if isinstance(serialized, numbers.Real):
        return True
    return serialized in ("NaN", "Infinity", "-Infinity")


@given(meta=array_metadata())  # type: ignore[misc]
@pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
def test_array_metadata_meets_spec(meta: ArrayV2Metadata | ArrayV3Metadata) -> None:
    """
    Validate that the array metadata produced by the library conforms to the relevant spec (V2 vs V3).

    For ArrayV2Metadata:
      - Ensures that 'zarr_format' is 2.
      - Verifies that 'filters' is either None or a tuple (and not an empty tuple).
    For ArrayV3Metadata:
      - Ensures that 'zarr_format' is 3.

    For both versions:
      - If the dtype is a floating point of some kind, verifies of fill values:
          * NaN is serialized as the string "NaN"
          * Positive Infinity is serialized as the string "Infinity"
          * Negative Infinity is serialized as the string "-Infinity"
          * Other fill values are preserved as-is.
      - If the dtype is a complex number of some kind, verifies that each component of the fill
        value (real and imaginary) satisfies the serialization rules for floating point numbers.
      - If the dtype is a datetime of some kind, verifies that `NaT` values are serialized as "NaT".

    Note:
      This test validates spec-compliance for array metadata serialization.
      It is a work-in-progress and should be expanded as further edge cases are identified.
    """
    asdict_dict = meta.to_dict()

    # version-specific validations
    if isinstance(meta, ArrayV2Metadata):
        assert asdict_dict["filters"] != ()
        assert asdict_dict["filters"] is None or isinstance(asdict_dict["filters"], tuple)
        assert asdict_dict["zarr_format"] == 2
    else:
        assert asdict_dict["zarr_format"] == 3

    # version-agnostic validations
    dtype_native = meta.dtype.to_native_dtype()
    if dtype_native.kind == "f":
        assert serialized_float_is_valid(asdict_dict["fill_value"])
    elif dtype_native.kind == "c":
        # fill_value should be a two-element array [real, imag].
        assert serialized_complex_float_is_valid(asdict_dict["fill_value"])
    elif dtype_native.kind in ("M", "m") and np.isnat(meta.fill_value):
        assert asdict_dict["fill_value"] == -9223372036854775808


# The indexing modes and which Array method implements each. vindex/mask are
# "vectorized" — they scatter through a single flat index, so an out= buffer must
# be flat (number of selected points) rather than the multi-dimensional result.
_INDEX_MODES: tuple[IndexMode, ...] = ("basic", "oindex", "vindex", "mask")


# The mode-dispatch / eligibility / unambiguous-write helpers below exist for the
# stateful _IndexingStateMachine (further down), which picks a random mode per step.
# The per-mode parity tests are deliberately explicit and do not use them.
def _get(target: zarr.Array, mode: IndexMode, zsel: Any, *, out: Any = None) -> Any:
    """Read `zsel` from `target` via the get-method for `mode`."""
    if mode == "basic":
        return target.get_basic_selection(zsel, out=out)
    if mode == "oindex":
        return target.get_orthogonal_selection(zsel, out=out)
    if mode == "vindex":
        return target.get_coordinate_selection(zsel, out=out)
    if mode == "mask":
        return target.get_mask_selection(zsel, out=out)
    raise AssertionError(mode)


def _setitem(zarray: zarr.Array, mode: IndexMode, zsel: Any, value: Any) -> None:
    """Write `value` at `zsel` via the set-method for `mode`."""
    if mode == "basic":
        zarray[zsel] = value
    elif mode == "oindex":
        zarray.oindex[zsel] = value
    elif mode == "vindex":
        zarray.vindex[zsel] = value
    elif mode == "mask":
        zarray.set_mask_selection(zsel, value)
    else:
        raise AssertionError(mode)


def _write_is_unambiguous(mode: IndexMode, npsel: Any, shape: tuple[int, ...]) -> bool:
    """Whether a write to `npsel` targets each cell at most once.

    Negative indices are normalized first (`[2, -2]` on length 4 both target cell
    2), then duplicate targets are detected. A repeated target makes the surviving
    value depend on the order the writes are applied. NumPy happens to be
    deterministic here (last-write-wins), but only as an implementation detail of
    iterating the index serially in C order — it is not a guarantee. zarr writes
    scalars into chunks and cannot in general promise a write order, so it cannot
    be expected to match NumPy's choice. There is therefore no well-defined oracle
    for such a write, and we reject it (a fundamental limitation, not a bug).
    """
    sel = npsel if isinstance(npsel, tuple) else (npsel,)
    if mode == "oindex":
        # Independent axes: each fancy axis must select each index at most once.
        for axis, s in enumerate(sel):
            if isinstance(s, np.ndarray):
                norm = np.where(s < 0, s + shape[axis], s)
                if norm.size != np.unique(norm).size:
                    return False
        return True
    if mode == "vindex":
        # Correlated coordinate arrays: a target is the tuple of normalized coords,
        # so the write is well-defined iff no coordinate tuple repeats.
        norm = [
            np.where(s < 0, s + shape[axis], s)
            for axis, s in enumerate(sel)
            if isinstance(s, np.ndarray)
        ]
        if not norm:
            return True
        coords = np.stack([a.ravel() for a in np.broadcast_arrays(*norm)], axis=-1)
        return len(coords) == len(np.unique(coords, axis=0))
    return True  # basic / mask never target a cell twice


def _n_array_axes(npsel: Any) -> int:
    """Number of integer-array (fancy) axes in a selection."""
    sel = npsel if isinstance(npsel, tuple) else (npsel,)
    return sum(isinstance(s, np.ndarray) for s in sel)


def _eligible(mode: IndexMode, shape: tuple[int, ...]) -> bool:
    """Whether `mode` can be exercised on `shape`.

    Rank-0 arrays have no interesting selections; the fancy modes
    (oindex/vindex/mask via integer/boolean arrays) can't handle zero-size axes.
    """
    if len(shape) == 0:
        return False
    return mode == "basic" or all(s > 0 for s in shape)


def _indexing_array(data: st.DataObject) -> zarr.Array:
    """An eager array (regular or rectilinear) for the indexing-parity tests."""
    return data.draw(
        st.one_of(
            simple_arrays(shapes=npst.array_shapes(max_dims=4)),
            rectilinear_arrays(shapes=npst.array_shapes(max_dims=3, min_side=1, max_side=20)),
        )
    )


def _fancy_array(data: st.DataObject) -> zarr.Array:
    """An eager array with rank >= 1 and no zero-size axis.

    The fancy modes (oindex/vindex/mask) need at least one dimension and can't
    address a zero-length axis, so drawing eligible arrays directly avoids
    filtering selections after the fact.
    """
    return data.draw(
        st.one_of(
            simple_arrays(shapes=npst.array_shapes(min_dims=1, max_dims=4, min_side=1)),
            rectilinear_arrays(
                shapes=npst.array_shapes(min_dims=1, max_dims=3, min_side=1, max_side=20)
            ),
        )
    )


def _unique_indices(data: st.DataObject, size: int) -> np.ndarray[Any, np.dtype[np.intp]]:
    """A non-empty array of distinct, non-negative indices into an axis of `size`."""
    values = data.draw(st.lists(st.integers(0, size - 1), unique=True, min_size=1, max_size=size))
    return np.array(values, dtype=np.intp)


# Per-mode read/write parity against NumPy. Each mode gets its own explicit test
# (readability over DRY): the shared, consumer-agnostic oracle lives with the lazy
# PR, which reuses these same selections against a view. `numpy_array_indexers`
# returns a (zarr_selection, numpy_selection) pair — they differ only for oindex,
# where NumPy needs the np.ix_ spelling.

_RD = pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")


@settings(deadline=None)
@_RD
@given(data=st.data())
async def test_basic_read(data: st.DataObject) -> None:
    """Basic-indexing reads match NumPy: sync, async, and into an out= buffer."""
    arr = _indexing_array(data)
    ref = arr[:]
    idx = data.draw(basic_indices(shape=ref.shape))
    expected = np.asarray(ref[idx])
    actual = np.asarray(arr[idx])
    assert_array_equal(actual, expected)
    assert_array_equal(np.asarray(await arr._async_array.getitem(idx)), expected)
    if expected.ndim >= 1 and expected.size > 0:
        assert actual.dtype == expected.dtype
        buf = default_buffer_prototype().nd_buffer.empty(shape=expected.shape, dtype=expected.dtype)
        arr.get_basic_selection(idx, out=buf)
        assert_array_equal(np.asarray(buf.as_ndarray_like()), expected)


@settings(deadline=None)
@_RD
@given(data=st.data())
async def test_oindex_read(data: st.DataObject) -> None:
    """Orthogonal (outer-product) reads match NumPy: sync, async, and out=."""
    arr = _fancy_array(data)
    ref = arr[:]
    zsel, npsel = data.draw(numpy_array_indexers(mode="oindex", shape=ref.shape))
    expected = np.asarray(ref[npsel])
    actual = np.asarray(arr.oindex[zsel])
    assert actual.dtype == expected.dtype
    assert_array_equal(actual, expected)
    assert_array_equal(np.asarray(await arr._async_array.oindex.getitem(zsel)), expected)
    if expected.size > 0:
        buf = default_buffer_prototype().nd_buffer.empty(shape=expected.shape, dtype=expected.dtype)
        arr.get_orthogonal_selection(zsel, out=buf)
        assert_array_equal(np.asarray(buf.as_ndarray_like()), expected)


@settings(deadline=None)
@_RD
@given(data=st.data())
async def test_vindex_read(data: st.DataObject) -> None:
    """Coordinate (vectorized) reads match NumPy: sync, async, and out= (flat buffer)."""
    arr = _fancy_array(data)
    ref = arr[:]
    zsel, npsel = data.draw(numpy_array_indexers(mode="vindex", shape=ref.shape))
    expected = np.asarray(ref[npsel])
    actual = np.asarray(arr.vindex[zsel])
    assert actual.dtype == expected.dtype
    assert_array_equal(actual, expected)
    assert_array_equal(np.asarray(await arr._async_array.vindex.getitem(zsel)), expected)
    if expected.size > 0:  # the vectorized out= buffer is flat
        buf = default_buffer_prototype().nd_buffer.empty(
            shape=(expected.size,), dtype=expected.dtype
        )
        arr.get_coordinate_selection(zsel, out=buf)
        assert_array_equal(np.asarray(buf.as_ndarray_like()).reshape(expected.shape), expected)


@settings(deadline=None)
@_RD
@given(data=st.data())
async def test_mask_read(data: st.DataObject) -> None:
    """Boolean-mask reads match NumPy, and agree with the vindex[mask] spelling."""
    arr = _fancy_array(data)
    ref = arr[:]
    mask = data.draw(npst.arrays(dtype=np.bool_, shape=st.just(ref.shape)))
    expected = np.asarray(ref[mask])
    actual = np.asarray(arr.get_mask_selection(mask))
    assert actual.dtype == expected.dtype
    assert_array_equal(actual, expected)
    assert_array_equal(np.asarray(arr.vindex[mask]), expected)
    if expected.size > 0:
        buf = default_buffer_prototype().nd_buffer.empty(
            shape=(expected.size,), dtype=expected.dtype
        )
        arr.get_mask_selection(mask, out=buf)
        assert_array_equal(np.asarray(buf.as_ndarray_like()).reshape(expected.shape), expected)


@settings(deadline=None)
@_RD
@given(data=st.data())
async def test_basic_write(data: st.DataObject) -> None:
    """Basic-indexing writes match NumPy (slices/ints never target a cell twice)."""
    arr = _indexing_array(data)
    ref = arr[:]
    idx = data.draw(basic_indices(shape=ref.shape))
    value = data.draw(numpy_arrays(shapes=st.just(np.asarray(ref[idx]).shape), dtype=ref.dtype))
    arr[idx] = value
    ref[idx] = value
    assert_array_equal(ref, arr[:])


@settings(deadline=None)
@_RD
@given(data=st.data())
async def test_oindex_write(data: st.DataObject) -> None:
    """Orthogonal writes to distinct per-axis indices match NumPy.

    Indices are drawn unique per axis so no cell is written twice (a repeated target
    is order-dependent and has no well-defined oracle). Multi-axis orthogonal writes
    to *sharded* arrays are unsupported (see
    `test_oindex_sharded_multiaxis_write_xfail`), so this uses unsharded arrays.
    """
    arr = _fancy_array(data)
    assume(arr.shards is None)
    ref = arr[:]
    idx = tuple(_unique_indices(data, s) for s in ref.shape)
    value = data.draw(numpy_arrays(shapes=st.just(ref[np.ix_(*idx)].shape), dtype=ref.dtype))
    arr.oindex[idx] = value
    ref[np.ix_(*idx)] = value
    assert_array_equal(ref, arr[:])


@settings(deadline=None)
@_RD
@given(data=st.data())
async def test_vindex_write(data: st.DataObject) -> None:
    """Coordinate writes to distinct coordinates match NumPy.

    Coordinates come from distinct flat indices, so every target tuple is unique and
    the write is well-defined regardless of the order chunks are visited.
    """
    arr = _fancy_array(data)
    ref = arr[:]
    flat = _unique_indices(data, ref.size)
    coords = tuple(c.astype(np.intp) for c in np.unravel_index(flat, ref.shape))
    value = data.draw(numpy_arrays(shapes=st.just(flat.shape), dtype=ref.dtype))
    arr.vindex[coords] = value
    ref[coords] = value
    assert_array_equal(ref, arr[:])


@settings(deadline=None)
@_RD
@given(data=st.data())
async def test_mask_write(data: st.DataObject) -> None:
    """Boolean-mask writes match NumPy, via set_mask_selection and vindex[mask]."""
    arr = _fancy_array(data)
    ref = arr[:]
    mask = data.draw(npst.arrays(dtype=np.bool_, shape=st.just(ref.shape)))
    value = data.draw(numpy_arrays(shapes=st.just(np.asarray(ref[mask]).shape), dtype=ref.dtype))
    arr.set_mask_selection(mask, value)
    ref[mask] = value
    assert_array_equal(ref, arr[:])
    arr.vindex[mask] = value  # the two spellings must agree
    assert_array_equal(ref, arr[:])


@pytest.mark.xfail(
    reason="GH2834: the sharding partial-write codec does not support orthogonal "
    "writes spanning two or more array axes",
    strict=True,
)
def test_oindex_sharded_multiaxis_write_xfail() -> None:
    """Pin the one known-unsupported write: oindex across >= 2 array axes on a shard.

    Single-axis oindex, vindex, and mask writes to sharded arrays all work; only
    the multi-array-axis orthogonal case is broken. Strict xfail so this flips to a
    failure (prompting removal) if/when GH2834 is fixed.
    """
    a = zarr.create_array({}, shape=(12, 12), chunks=(3, 3), shards=(6, 6), dtype="i4")
    a[...] = np.arange(144, dtype="i4").reshape(12, 12)
    i0, i1 = np.array([0, 5, 9]), np.array([1, 6, 10])
    a.set_orthogonal_selection((i0, i1), np.zeros((3, 3), dtype="i4"))
    assert_array_equal(a.oindex[i0, i1], np.zeros((3, 3), dtype="i4"))


@pytest.mark.parametrize(
    "case",
    [
        Expect(
            input=("vindex", (np.array([2, -2, 0, 1]),), (4,)),
            output=False,
            id="vindex-neg-collision",
        ),
        Expect(input=("vindex", (np.array([0, 1, 2]),), (4,)), output=True, id="vindex-distinct"),
        Expect(
            input=("oindex", (np.array([1, -3]),), (4,)), output=False, id="oindex-neg-collision"
        ),
        Expect(
            input=("oindex", (np.array([0, 2]), np.array([1, 1])), (4, 4)),
            output=False,
            id="oindex-repeat",
        ),
        Expect(
            input=("oindex", (np.array([0, 2]), np.array([1, 3])), (4, 4)),
            output=True,
            id="oindex-distinct",
        ),
        Expect(input=("basic", (slice(None),), (4,)), output=True, id="basic-always"),
    ],
    ids=lambda c: c.id,
)
def test_write_is_unambiguous(
    case: Expect[tuple[IndexMode, Any, tuple[int, ...]], bool],
) -> None:
    """A write is well-defined iff each target cell is hit at most once *after*
    negative indices are normalized.

    Regression anchor for the CI fix: vindex `[2, -2, 0, 1]` on length 4 maps
    `-2 -> 2`, so cell 2 is written twice (order-dependent) and must be rejected,
    even though the raw values are all distinct.
    """
    mode, npsel, shape = case.input
    assert _write_is_unambiguous(mode, npsel, shape) is case.output


@st.composite
def _bounds_selection(draw: st.DrawFn, mode: IndexMode, shape: tuple[int, ...]) -> Any:
    """An integer selection whose components may fall outside `[-size, size)`, to
    probe out-of-bounds behavior. Returns `(zarr_sel, numpy_sel)`."""
    wide = {s: st.integers(-2 * s - 1, 2 * s) for s in set(shape)}
    if mode == "basic":
        sel = tuple(draw(wide[s]) for s in shape)
        return sel, sel
    n = draw(st.integers(1, 4))
    arrs = tuple(
        np.array(draw(st.lists(wide[s], min_size=n, max_size=n)), dtype=np.intp) for s in shape
    )
    if mode == "oindex":
        return arrs, np.ix_(*arrs)  # numpy outer-product oracle
    return arrs, arrs  # vindex: pointwise


@pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
@given(data=st.data())
def test_indexing_bounds_error_parity(data: st.DataObject) -> None:
    """Out-of-bounds integer indexing raises in zarr iff it raises in NumPy.

    zarr's bounds errors (BoundsCheckError etc.) subclass IndexError, so the eager
    methods must agree with NumPy on *whether* an index is in bounds. This is the
    ndindex `check_same` error-parity idea, and it catches the silent-wrong-data
    class — returning garbage where NumPy would raise.
    """
    zarray = data.draw(simple_arrays(shapes=npst.array_shapes(max_dims=3, min_side=1, max_side=6)))
    nparray = zarray[:]
    mode = data.draw(st.sampled_from(("basic", "oindex", "vindex")))
    zsel, npsel = data.draw(_bounds_selection(mode, nparray.shape))
    try:
        expected = np.asarray(nparray[npsel])
    except IndexError:
        with pytest.raises(IndexError):
            _get(zarray, mode, zsel)
        return
    assert_array_equal(np.asarray(_get(zarray, mode, zsel)), expected)


class _IndexingStateMachine(RuleBasedStateMachine):
    """Apply many random indexing writes to one array while a NumPy model tracks it
    in lockstep; the model must match after every step.

    This is the TensorStore `driver_testutil` pattern with NumPy as the oracle:
    a single array accumulates many writes, so it catches read-modify-write,
    chunk-boundary, and shard-merge/persistence bugs that the single-shot
    `test_indexing_write_parity` cannot see. Concrete geometries are defined by
    subclasses; the rules are dtype-agnostic (fixed i4 — dtype breadth lives in the
    property tests).
    """

    shape: tuple[int, ...] = ()
    chunks: tuple[int, ...] = ()
    shards: tuple[int, ...] | None = None

    def __init__(self) -> None:
        super().__init__()
        n = int(np.prod(self.shape, dtype=int))
        self.zarray = zarr.create_array(
            {}, shape=self.shape, chunks=self.chunks, shards=self.shards, dtype="i4"
        )
        self.zarray[...] = np.arange(n, dtype="i4").reshape(self.shape)
        self.model = self.zarray[:]  # mirror the array; kept in lockstep by the rules

    @rule(data=st.data(), mode=st.sampled_from(_INDEX_MODES))
    def write(self, data: st.DataObject, mode: IndexMode) -> None:
        if not _eligible(mode, self.shape):
            return
        zsel, npsel = data.draw(numpy_array_indexers(mode=mode, shape=self.shape))
        if mode in ("oindex", "vindex") and not _write_is_unambiguous(mode, npsel, self.shape):
            return
        if mode == "oindex" and self.shards is not None and _n_array_axes(npsel) >= 2:
            return  # GH2834, see test_oindex_sharded_multiaxis_write_xfail
        expected = np.asarray(self.model[npsel])
        value = data.draw(numpy_arrays(shapes=st.just(expected.shape), dtype=self.model.dtype))
        self.model[npsel] = value
        _setitem(self.zarray, mode, zsel, value)

    @rule(data=st.data(), mode=st.sampled_from(_INDEX_MODES))
    def read(self, data: st.DataObject, mode: IndexMode) -> None:
        if not _eligible(mode, self.shape):
            return
        zsel, npsel = data.draw(numpy_array_indexers(mode=mode, shape=self.shape))
        assert_array_equal(np.asarray(_get(self.zarray, mode, zsel)), np.asarray(self.model[npsel]))

    @invariant()
    def matches_model(self) -> None:
        assert_array_equal(self.zarray[:], self.model)


class _RegularStateMachine(_IndexingStateMachine):
    shape, chunks, shards = (12,), (3,), None


class _Sharded2DStateMachine(_IndexingStateMachine):
    shape, chunks, shards = (8, 9), (2, 3), (4, 9)


class _ThreeDStateMachine(_IndexingStateMachine):
    shape, chunks, shards = (4, 5, 6), (2, 3, 3), None


# Run these under the slow-hypothesis CI job (like the store stateful tests): a
# stateful run is a sequence of steps, so it is heavier than the single-shot tests.
TestIndexingStateMachineRegular = pytest.mark.slow_hypothesis(_RegularStateMachine.TestCase)
TestIndexingStateMachineSharded = pytest.mark.slow_hypothesis(_Sharded2DStateMachine.TestCase)
TestIndexingStateMachine3D = pytest.mark.slow_hypothesis(_ThreeDStateMachine.TestCase)
