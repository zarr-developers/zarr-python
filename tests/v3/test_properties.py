import numpy as np
import pytest
from numpy.testing import assert_array_equal

pytest.importorskip("hypothesis")

import hypothesis.extra.numpy as npst  # noqa
import hypothesis.strategies as st  # noqa
from hypothesis import given, settings  # noqa
from zarr.testing.strategies import arrays, np_arrays, basic_indices  # noqa


@given(st.data())
def test_roundtrip(data):
    nparray = data.draw(np_arrays)
    zarray = data.draw(arrays(arrays=st.just(nparray)))
    assert_array_equal(nparray, zarray[:])


@given(data=st.data())
# The filter warning here is to silence an occasional warning in NDBuffer.all_equal
# See https://github.com/zarr-developers/zarr-python/pull/2118#issuecomment-2310280899
# Uncomment the next line to reproduce the original failure.
# @reproduce_failure('6.111.2', b'AXicY2FgZGRAB/8/ndR2z7nkDZEDADWpBL4=')
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_basic_indexing(data):
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
# The filter warning here is to silence an occasional warning in NDBuffer.all_equal
# See https://github.com/zarr-developers/zarr-python/pull/2118#issuecomment-2310280899
# Uncomment the next line to reproduce the original failure.
# @reproduce_failure('6.111.2', b'AXicY2FgZGRAB/8/eLmF7qr/C5EDADZUBRM=')
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_vindex(data):
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
