import numpy as np
import pytest
from numpy.testing import assert_array_equal

pytest.importorskip("hypothesis")

import hypothesis.extra.numpy as npst  # noqa
import hypothesis.strategies as st  # noqa
from hypothesis import given, settings  # noqa
from zarr.strategies import arrays, np_arrays  # noqa


# @pytest.mark.slow
@settings(max_examples=300)
@given(st.data())
def test_roundtrip(data):
    nparray = data.draw(np_arrays)
    zarray = data.draw(arrays(arrays=st.just(nparray)))
    assert_array_equal(nparray, zarray[:])


@given(st.data())
def test_roundtrip_object_array(data):
    nparray = data.draw(np_arrays)
    zarray = data.draw(arrays(arrays=st.just(nparray)))
    assert_array_equal(nparray, zarray[:])


# @pytest.mark.slow
@settings(max_examples=500)
@given(data=st.data())
def test_basic_indexing(data):
    def is_negative_slice(idx):
        return isinstance(idx, slice) and idx.step is not None and idx.step < 0

    zarray = data.draw(arrays())
    nparray = zarray[:]
    indexer = data.draw(
        npst.basic_indices(shape=nparray.shape).filter(
            lambda idxr: (
                not (
                    is_negative_slice(idxr)
                    or (isinstance(idxr, tuple) and any(is_negative_slice(idx) for idx in idxr))
                )
            )
        )
    )
    actual = zarray[indexer]
    assert_array_equal(nparray[indexer], actual)

    new_data = np.ones_like(actual)
    zarray[indexer] = new_data
    nparray[indexer] = new_data
    assert_array_equal(nparray, zarray[:])
