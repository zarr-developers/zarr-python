from typing import Any

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
from hypothesis import given, settings  # noqa

from .array import Array
from .group import Group
from .store import MemoryStore, StoreLike

# Copied from Xarray
_attr_keys = st.text(st.characters(), min_size=1)
_attr_values = st.recursive(
    st.none() | st.booleans() | st.text(st.characters(), max_size=5),
    lambda children: st.lists(children) | st.dictionaries(_attr_keys, children),
    max_leaves=3,
)

# No '/' in array names?
# No '.' in paths?
zarr_key_chars = st.sampled_from("-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz")

# The following should be public strategies
attrs = st.none() | st.dictionaries(_attr_keys, _attr_values)
paths = st.none() | st.text(zarr_key_chars, min_size=1) | st.just("/")
array_names = st.text(zarr_key_chars | st.just("."), min_size=1).filter(
    lambda t: not t.startswith((".", ".."))
)
np_arrays = npst.arrays(
    # FIXME: re-enable timedeltas once we figure out the fill_value issue.
    dtype=npst.scalar_dtypes().filter(lambda x: x.kind != "m"),
    shape=npst.array_shapes(max_dims=4),
)
stores = st.builds(MemoryStore, st.just({}), mode=st.just("w"))
compressors = st.sampled_from([None, "default"])


@st.composite  # type: ignore[misc]
def np_array_and_chunks(
    draw: st.DrawFn, *, arrays: st.SearchStrategy[np.ndarray] = np_arrays
) -> tuple[np.ndarray, tuple[int]]:
    """A hypothesis strategy to generate small sized random arrays.

    Returns: a tuple of the array and a suitable random chunking for it.
    """
    array = draw(arrays)
    # We want this strategy to shrink towards arrays with smaller number of chunks
    # 1. st.integers() shrinks towards smaller values. So we use that to generate number of chunks
    numchunks = draw(st.tuples(*[st.integers(min_value=1, max_value=size) for size in array.shape]))
    # 2. and now generate the chunks tuple
    chunks = tuple(size // nchunks for size, nchunks in zip(array.shape, numchunks, strict=True))
    return (array, chunks)


@st.composite  # type: ignore[misc]
def arrays(
    draw: st.DrawFn,
    *,
    compressors: st.SearchStrategy = compressors,
    stores: st.SearchStrategy[StoreLike] = stores,
    arrays: st.SearchStrategy[np.ndarray] = np_arrays,
    paths: st.SearchStrategy[None | str] = paths,
    array_names: st.SearchStrategy = array_names,
    attrs: st.SearchStrategy = attrs,
) -> Array:
    store = draw(stores)
    nparray, chunks = draw(np_array_and_chunks(arrays=arrays))
    path = draw(paths)
    name = draw(array_names)
    attributes = draw(attrs)
    # compressor = draw(compressors)

    # TODO: clean this up
    if path is None and name is None:
        array_path = None
        array_name = None
    elif path is None and name is not None:
        array_path = f"{name}"
        array_name = f"/{name}"
    elif path is not None and name is None:
        array_path = path
        array_name = None
    elif path == "/":
        assert name is not None
        array_path = name
        array_name = "/" + name
    else:
        assert name is not None
        array_path = f"{path}/{name}"
        array_name = "/" + array_path

    expected_attrs = {} if attributes is None else attributes

    root = Group.create(store)
    a = root.create_array(
        array_path,
        shape=nparray.shape,
        chunks=chunks,
        dtype=nparray.dtype.str,
        attributes=attributes,
        # compressor=compressor,  # TODO: FIXME
        # TODO: FIXME seems to break with booleans and timedelta
        # fill_value=nparray.dtype.type(0),
    )

    assert isinstance(a, Array)
    assert nparray.shape == a.shape
    # assert chunks == a.chunks  # TODO: adapt for v2, v3
    assert array_path == a.path
    assert array_name == a.name
    # assert a.basename is None  # TODO
    # assert a.store == normalize_store_arg(store)
    assert dict(a.attrs) == expected_attrs

    a[:] = nparray

    store.close()

    return a


def is_negative_slice(idx: Any) -> bool:
    return isinstance(idx, slice) and idx.step is not None and idx.step < 0


@st.composite  # type: ignore[misc]
def basic_indices(draw: st.DrawFn, *, shape: tuple[int]):
    """Basic indices without unsupported negative slices."""
    return draw(
        npst.basic_indices(shape=shape).filter(
            lambda idxr: (
                not (
                    is_negative_slice(idxr)
                    or (isinstance(idxr, tuple) and any(is_negative_slice(idx) for idx in idxr))
                )
            )
        )
    )
