import pytest
from numpy.testing import assert_array_equal

from zarr.core.buffer import default_buffer_prototype

pytest.importorskip("hypothesis")

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
from hypothesis import assume, given

from zarr.abc.store import Store
from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata
from zarr.testing.strategies import (
    array_metadata,
    arrays,
    basic_indices,
    numpy_arrays,
    orthogonal_indices,
    stores,
    zarr_formats,
)


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
