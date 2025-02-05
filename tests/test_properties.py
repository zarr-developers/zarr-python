import dataclasses
import json

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from zarr.core.buffer import default_buffer_prototype

pytest.importorskip("hypothesis")

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
from hypothesis import assume, given

from zarr.abc.store import Store
from zarr.core.common import ZARRAY_JSON, ZATTRS_JSON
from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata
from zarr.core.sync import sync
from zarr.testing.strategies import (
    array_metadata,
    array_metadata_v2,
    arrays,
    basic_indices,
    numpy_arrays,
    orthogonal_indices,
    stores,
    zarr_formats,
)


def deep_equal(a, b):
    """Deep equality check w/ NaN e to handle array metadata serialization and deserialization behaviors"""
    if isinstance(a, (complex, np.complexfloating)) and isinstance(
        b, (complex, np.complexfloating)
    ):
        # Convert to Python float to force standard NaN handling.
        a_real, a_imag = float(a.real), float(a.imag)
        b_real, b_imag = float(b.real), float(b.imag)
        # If both parts are NaN, consider them equal.
        if np.isnan(a_real) and np.isnan(b_real):
            real_eq = True
        else:
            real_eq = a_real == b_real
        if np.isnan(a_imag) and np.isnan(b_imag):
            imag_eq = True
        else:
            imag_eq = a_imag == b_imag
        return real_eq and imag_eq

    # Handle floats (including numpy floating types) and treat NaNs as equal.
    if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
        if np.isnan(a) and np.isnan(b):
            return True
        return a == b

    # Handle numpy.datetime64 values, treating NaT as equal.
    if isinstance(a, np.datetime64) and isinstance(b, np.datetime64):
        if np.isnat(a) and np.isnat(b):
            return True
        return a == b

    # Handle numpy arrays.
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.shape != b.shape:
            return False
        # Compare elementwise.
        return all(deep_equal(x, y) for x, y in zip(a.flat, b.flat, strict=False))

    # Handle dictionaries.
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(deep_equal(a[k], b[k]) for k in a)

    # Handle lists and tuples.
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(deep_equal(x, y) for x, y in zip(a, b, strict=False))

    # Fallback to default equality.
    return a == b


@given(data=st.data(), zarr_format=zarr_formats)
def test_roundtrip(data: st.DataObject, zarr_format: int) -> None:
    nparray = data.draw(numpy_arrays(zarr_formats=st.just(zarr_format)))
    zarray = data.draw(arrays(arrays=st.just(nparray), zarr_formats=st.just(zarr_format)))
    assert_array_equal(nparray, zarray[:])


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


@given(data=st.data())
def test_basic_indexing(data: st.DataObject) -> None:
    zarray = data.draw(arrays())
    nparray = zarray[:]
    indexer = data.draw(basic_indices(shape=nparray.shape))
    actual = zarray[indexer]
    assert_array_equal(nparray[indexer], actual)

    new_data = data.draw(npst.arrays(shape=st.just(actual.shape), dtype=nparray.dtype))
    zarray[indexer] = new_data
    nparray[indexer] = new_data
    assert_array_equal(nparray, zarray[:])


@given(data=st.data())
def test_oindex(data: st.DataObject) -> None:
    # integer_array_indices can't handle 0-size dimensions.
    zarray = data.draw(arrays(shapes=npst.array_shapes(max_dims=4, min_side=1)))
    nparray = zarray[:]

    zindexer, npindexer = data.draw(orthogonal_indices(shape=nparray.shape))
    actual = zarray.oindex[zindexer]
    assert_array_equal(nparray[npindexer], actual)

    assume(zarray.shards is None)  # GH2834
    new_data = data.draw(npst.arrays(shape=st.just(actual.shape), dtype=nparray.dtype))
    nparray[npindexer] = new_data
    zarray.oindex[zindexer] = new_data
    assert_array_equal(nparray, zarray[:])


@given(data=st.data())
def test_vindex(data: st.DataObject) -> None:
    # integer_array_indices can't handle 0-size dimensions.
    zarray = data.draw(arrays(shapes=npst.array_shapes(max_dims=4, min_side=1)))
    nparray = zarray[:]

    indexer = data.draw(
        npst.integer_array_indices(
            shape=nparray.shape, result_shape=npst.array_shapes(min_side=1, max_dims=None)
        )
    )
    actual = zarray.vindex[indexer]
    assert_array_equal(nparray[indexer], actual)

    # FIXME!
    # when the indexer is such that a value gets overwritten multiple times,
    # I think the output depends on chunking.
    # new_data = data.draw(npst.arrays(shape=st.just(actual.shape), dtype=nparray.dtype))
    # nparray[indexer] = new_data
    # zarray.vindex[indexer] = new_data
    # assert_array_equal(nparray, zarray[:])


@given(store=stores, meta=array_metadata())  # type: ignore[misc]
async def test_roundtrip_array_metadata(
    store: Store, meta: ArrayV2Metadata | ArrayV3Metadata
) -> None:
    asdict = meta.to_buffer_dict(prototype=default_buffer_prototype())
    for key, expected in asdict.items():
        await store.set(f"0/{key}", expected)
        actual = await store.get(f"0/{key}", prototype=default_buffer_prototype())
        assert actual == expected


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


@given(array_metadata_v2())
def test_v2meta_roundtrip(metadata):
    buffer_dict = metadata.to_buffer_dict(prototype=default_buffer_prototype())
    zarray_dict = json.loads(buffer_dict[ZARRAY_JSON].to_bytes().decode())
    zattrs_dict = json.loads(buffer_dict[ZATTRS_JSON].to_bytes().decode())

    # zattrs and zarray are separate in v2, we have to add attributes back prior to `from_dict`
    zarray_dict["attributes"] = zattrs_dict

    metadata_roundtripped = ArrayV2Metadata.from_dict(zarray_dict)

    # Convert both metadata instances to dictionaries.
    orig = dataclasses.asdict(metadata)
    rt = dataclasses.asdict(metadata_roundtripped)

    assert deep_equal(orig, rt), f"Roundtrip mismatch:\nOriginal: {orig}\nRoundtripped: {rt}"


@given(npst.from_dtype(dtype=np.dtype("float64"), allow_nan=True, allow_infinity=True))
def test_v2meta_nan_and_infinity(fill_value):
    metadata = ArrayV2Metadata(
        shape=[10],
        dtype=np.dtype("float64"),
        chunks=[5],
        fill_value=fill_value,
        order="C",
    )

    buffer_dict = metadata.to_buffer_dict(prototype=default_buffer_prototype())
    zarray_dict = json.loads(buffer_dict[ZARRAY_JSON].to_bytes().decode())

    if np.isnan(fill_value):
        assert zarray_dict["fill_value"] == "NaN"
    elif np.isinf(fill_value) and fill_value > 0:
        assert zarray_dict["fill_value"] == "Infinity"
    elif np.isinf(fill_value):
        assert zarray_dict["fill_value"] == "-Infinity"
    else:
        assert zarray_dict["fill_value"] == fill_value
