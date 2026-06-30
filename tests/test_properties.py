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

from zarr.abc.store import Store
from zarr.core.common import ZARR_JSON, ZARRAY_JSON, ZATTRS_JSON
from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata
from zarr.core.sync import sync
from zarr.testing.strategies import (
    array_metadata,
    arrays,
    basic_indices,
    block_indices,
    block_test_arrays,
    complex_rectilinear_arrays,
    indexers,
    numpy_arrays,
    orthogonal_indices,
    rectilinear_arrays,
    simple_arrays,
    stores,
    windows,
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


# this decorator removes timeout; not ideal but it should avoid intermittent CI failures


@settings(deadline=None)
@pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
@given(data=st.data())
async def test_basic_indexing(data: st.DataObject) -> None:
    zarray = data.draw(st.one_of(simple_arrays(), rectilinear_arrays()))
    nparray = zarray[:]
    indexer = data.draw(basic_indices(shape=nparray.shape))

    # sync get
    actual = zarray[indexer]
    assert_array_equal(nparray[indexer], actual)

    # async get
    async_zarray = zarray._async_array
    actual = await async_zarray.getitem(indexer)
    assert_array_equal(nparray[indexer], actual)

    # sync set
    new_data = data.draw(numpy_arrays(shapes=st.just(actual.shape), dtype=nparray.dtype))
    zarray[indexer] = new_data
    nparray[indexer] = new_data
    assert_array_equal(nparray, zarray[:])

    # TODO test async setitem?


@settings(deadline=None)
@pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
@given(data=st.data())
async def test_basic_indexing_complex_rectilinear(data: st.DataObject) -> None:
    nparray, zarray = data.draw(complex_rectilinear_arrays())
    indexer = data.draw(basic_indices(shape=nparray.shape))
    assert_array_equal(nparray[indexer], zarray[indexer])


@given(data=st.data())
@pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
async def test_oindex(data: st.DataObject) -> None:
    # integer_array_indices can't handle 0-size dimensions.
    zarray = data.draw(
        st.one_of(
            simple_arrays(shapes=npst.array_shapes(max_dims=4, min_side=1)),
            rectilinear_arrays(shapes=npst.array_shapes(max_dims=4, min_side=1, max_side=20)),
        )
    )
    nparray = zarray[:]
    zindexer, npindexer = data.draw(orthogonal_indices(shape=nparray.shape))

    # sync get
    actual = zarray.oindex[zindexer]
    assert_array_equal(nparray[npindexer], actual)

    # async get
    async_zarray = zarray._async_array
    actual = await async_zarray.oindex.getitem(zindexer)
    assert_array_equal(nparray[npindexer], actual)

    # sync get
    assume(zarray.shards is None)  # GH2834
    for idxr in npindexer:
        if isinstance(idxr, np.ndarray) and idxr.size != np.unique(idxr).size:
            # behaviour of setitem with repeated indices is not guaranteed in practice
            assume(False)
    new_data = data.draw(numpy_arrays(shapes=st.just(actual.shape), dtype=nparray.dtype))
    nparray[npindexer] = new_data
    zarray.oindex[zindexer] = new_data
    assert_array_equal(nparray, zarray[:])

    # note: async oindex setitem not yet implemented


@given(data=st.data())
@pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
async def test_vindex(data: st.DataObject) -> None:
    # integer_array_indices can't handle 0-size dimensions.
    zarray = data.draw(
        st.one_of(
            simple_arrays(shapes=npst.array_shapes(max_dims=4, min_side=1)),
            rectilinear_arrays(shapes=npst.array_shapes(max_dims=3, min_side=1, max_side=20)),
        )
    )
    nparray = zarray[:]
    indexer = data.draw(
        npst.integer_array_indices(
            shape=nparray.shape, result_shape=npst.array_shapes(min_side=1, max_dims=None)
        )
    )

    # sync get
    actual = zarray.vindex[indexer]
    assert_array_equal(nparray[indexer], actual)

    # async get
    async_zarray = zarray._async_array
    actual = await async_zarray.vindex.getitem(indexer)
    assert_array_equal(nparray[indexer], actual)

    # sync set
    # FIXME!
    # when the indexer is such that a value gets overwritten multiple times,
    # I think the output depends on chunking.
    # new_data = data.draw(npst.arrays(shape=st.just(actual.shape), dtype=nparray.dtype))
    # nparray[indexer] = new_data
    # zarray.vindex[indexer] = new_data
    # assert_array_equal(nparray, zarray[:])

    # note: async vindex setitem not yet implemented


@settings(deadline=None)
@pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
@given(data=st.data())
def test_mask_indexing(data: st.DataObject) -> None:
    zarray = data.draw(st.one_of(simple_arrays(), rectilinear_arrays()))
    nparray = zarray[:]
    mask = data.draw(npst.arrays(dtype=np.bool_, shape=st.just(nparray.shape)))

    expected = nparray[mask]

    # sync get, via both the dedicated method and the vindex interface
    assert_array_equal(expected, zarray.get_mask_selection(mask))
    assert_array_equal(expected, zarray.vindex[mask])

    # sync set, via both interfaces
    assume(zarray.shards is None)  # GH2834
    new_data = data.draw(numpy_arrays(shapes=st.just(expected.shape), dtype=nparray.dtype))
    nparray[mask] = new_data
    zarray.set_mask_selection(mask, new_data)
    assert_array_equal(nparray, zarray[:])

    zarray.vindex[mask] = new_data
    assert_array_equal(nparray, zarray[:])


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
_INDEX_MODES = ("basic", "oindex", "vindex", "mask")
_VECTORIZED_MODES = frozenset({"vindex", "mask"})


def _get(target: zarr.Array, mode: str, zsel: Any, *, out: Any = None) -> Any:
    """Read ``zsel`` from ``target`` via the get-method for ``mode``."""
    if mode == "basic":
        return target.get_basic_selection(zsel, out=out)
    if mode == "oindex":
        return target.get_orthogonal_selection(zsel, out=out)
    if mode == "vindex":
        return target.get_coordinate_selection(zsel, out=out)
    if mode == "mask":
        return target.get_mask_selection(zsel, out=out)
    raise AssertionError(mode)


def assert_indexing_matches_numpy(
    target: zarr.Array, ref: np.ndarray[Any, Any], mode: str, data: st.DataObject
) -> None:
    """Assert ``target``'s ``mode`` indexing matches NumPy on ``ref``.

    Consumer-agnostic: ``target`` is any ``zarr.Array`` (an ordinary eager array
    or a lazy view) and ``ref`` is the NumPy array it should behave like. Draws a
    selection for ``mode`` and checks both a plain read and a read into an ``out=``
    buffer (flat for the vectorized vindex/mask modes, full-shape otherwise).
    """
    if ref.ndim == 0:
        return
    # integer_array_indices / orthogonal strategies can't handle 0-size dims.
    if mode != "basic" and not all(s > 0 for s in ref.shape):
        return
    zsel, npsel = data.draw(indexers(mode=mode, shape=ref.shape))
    expected = np.asarray(ref[npsel])

    # 1) plain read
    assert_array_equal(np.asarray(_get(target, mode, zsel)), expected)

    # 2) read into an out= buffer (flat for vectorized modes, full otherwise)
    if expected.ndim >= 1 and expected.size > 0:
        buf_shape = (expected.size,) if mode in _VECTORIZED_MODES else expected.shape
        buf = default_buffer_prototype().nd_buffer.empty(shape=buf_shape, dtype=expected.dtype)
        _get(target, mode, zsel, out=buf)
        assert_array_equal(np.asarray(buf.as_ndarray_like()).reshape(expected.shape), expected)


@settings(deadline=None)
@pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
@given(data=st.data())
async def test_indexing_parity(data: st.DataObject) -> None:
    """Every indexing method on an eager array matches NumPy (all modes, with/without out=).

    Comprehensive oracle over {basic, oindex, vindex, mask} by {out=None, out=}.
    Deliberately independent of lazy indexing so it can guard the eager API on its
    own; ``test_lazy_view_indexing_parity`` reuses the same oracle for the lazy
    consumer (a view is just another ``zarr.Array``).
    """
    zarray = data.draw(simple_arrays(shapes=npst.array_shapes(max_dims=4, min_side=1)))
    nparray = zarray[:]
    mode = data.draw(st.sampled_from(_INDEX_MODES))
    assert_indexing_matches_numpy(zarray, nparray, mode, data)


@pytest.mark.skipif(
    not hasattr(zarr.Array, "lazy"), reason="lazy indexing (Array.lazy) not available"
)
@settings(deadline=None)
@pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
@given(data=st.data())
async def test_lazy_view_indexing_parity(data: st.DataObject) -> None:
    """The eager indexing oracle, applied to a lazy view (the lazy consumer).

    A view (built from a non-negative window) is just another ``zarr.Array``, so
    it flows through the same ``assert_indexing_matches_numpy`` harness. The lazy
    indexing bugs found in review were all "the view path diverges from NumPy for
    some (method, parameter) combination" — enumerating that surface here catches
    the class. This test is skipped until ``Array.lazy`` exists, so the harness
    above can merge ahead of the lazy feature.
    """
    zarray = data.draw(simple_arrays(shapes=npst.array_shapes(max_dims=4, min_side=1)))
    nparray = zarray[:]
    window = data.draw(windows(shape=nparray.shape))
    view = zarray.lazy[window]
    vref = nparray[window]
    mode = data.draw(st.sampled_from(_INDEX_MODES))
    assert_indexing_matches_numpy(view, vref, mode, data)
