import json
import numpy as np
import pytest
from numpy.testing import assert_array_equal

pytest.importorskip("hypothesis")

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
from hypothesis import given

from zarr.core.buffer import default_buffer_prototype
from zarr.core.common import ZARRAY_JSON, parse_shapelike
from zarr.core.metadata.v2 import ArrayV2Metadata, parse_fill_value, parse_dtype, parse_shapelike
from zarr.testing.strategies import arrays, basic_indices, numpy_arrays, zarr_formats
from zarr.testing.v2metadata import array_metadata_v2_inputs


@given(data=st.data(), zarr_format=zarr_formats)
def test_roundtrip(data: st.DataObject, zarr_format: int) -> None:
    nparray = data.draw(numpy_arrays(zarr_formats=st.just(zarr_format)))
    zarray = data.draw(arrays(arrays=st.just(nparray), zarr_formats=st.just(zarr_format)))
    assert_array_equal(nparray, zarray[:])


@given(data=st.data())
def test_basic_indexing(data: st.DataObject) -> None:
    zarray = data.draw(arrays())
    nparray = zarray[:]
    indexer = data.draw(basic_indices(shape=nparray.shape))
    actual = zarray[indexer]
    assert_array_equal(nparray[indexer], actual)

    new_data = np.ones_like(actual)
    zarray[indexer] = new_data
    nparray[indexer] = new_data
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


@given(array_metadata_v2_inputs())
def test_v2meta_fill_value_serialization(inputs):
    metadata = ArrayV2Metadata(**inputs)
    buffer_dict = metadata.to_buffer_dict(prototype=default_buffer_prototype())
    zarray_dict = json.loads(buffer_dict[ZARRAY_JSON].to_bytes().decode())

    if isinstance(inputs["fill_value"], (float, np.floating)) and np.isnan(inputs["fill_value"]):
        assert zarray_dict["fill_value"] == "NaN"
    else:
        assert zarray_dict["fill_value"] == inputs["fill_value"]


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