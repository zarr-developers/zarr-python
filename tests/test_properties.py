import numpy as np
import pytest
from numpy.testing import assert_array_equal

pytest.importorskip("hypothesis")

import hypothesis.extra.numpy as npst  # noqa: E402
import hypothesis.strategies as st  # noqa: E402
from hypothesis import given  # noqa: E402

from zarr.testing.strategies import arrays, basic_indices, numpy_arrays, zarr_formats  # noqa: E402


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
    zarray = data.draw(arrays())
    nparray = zarray[:]

    indexer = data.draw(
        npst.integer_array_indices(
            shape=nparray.shape, result_shape=npst.array_shapes(max_dims=None)
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
