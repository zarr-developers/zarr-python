import re
from typing import Any

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
from hypothesis import given, settings  # noqa
from hypothesis.strategies import SearchStrategy

from zarr.core.array import Array
from zarr.core.group import Group
from zarr.store import MemoryStore, StoreLike

# Copied from Xarray
_attr_keys = st.text(st.characters(), min_size=1)
_attr_values = st.recursive(
    st.none() | st.booleans() | st.text(st.characters(), max_size=5),
    lambda children: st.lists(children) | st.dictionaries(_attr_keys, children),
    max_leaves=3,
)

# From https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html#node-names
# 1. must not be the empty string ("")
# 2. must not include the character "/"
# 3. must not be a string composed only of period characters, e.g. "." or ".."
# 4. must not start with the reserved prefix "__"
zarr_key_chars = st.sampled_from(
    ".-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz"
)
node_names = st.text(zarr_key_chars, min_size=1).filter(
    lambda t: t not in (".", "..") and not t.startswith("__")
)
array_names = node_names
attrs = st.none() | st.dictionaries(_attr_keys, _attr_values)
paths = st.lists(node_names, min_size=1).map(lambda x: "/".join(x)) | st.just("/")
np_arrays = npst.arrays(
    # TODO: re-enable timedeltas once they are supported
    dtype=npst.scalar_dtypes().filter(lambda x: x.kind != "m"),
    shape=npst.array_shapes(max_dims=4),
)
stores = st.builds(MemoryStore, st.just({}), mode=st.just("w"))
compressors = st.sampled_from([None, "default"])


@st.composite  # type: ignore[misc]
def np_array_and_chunks(
    draw: st.DrawFn, *, arrays: st.SearchStrategy[np.ndarray] = np_arrays
) -> tuple[np.ndarray, tuple[int]]:  # type: ignore[type-arg]
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
    # if path is None and name is None:
    #     array_path = None
    #     array_name = None
    # elif path is None and name is not None:
    #     array_path = f"{name}"
    #     array_name = f"/{name}"
    # elif path is not None and name is None:
    #     array_path = path
    #     array_name = None
    # elif path == "/":
    #     assert name is not None
    #     array_path = name
    #     array_name = "/" + name
    # else:
    #     assert name is not None
    #     array_path = f"{path}/{name}"
    #     array_name = "/" + array_path

    expected_attrs = {} if attributes is None else attributes

    array_path = path + ("/" if not path.endswith("/") else "") + name
    root = Group.from_store(store)
    fill_value_args: tuple[Any, ...] = tuple()
    if nparray.dtype.kind == "M":
        m = re.search(r"\[(.+)\]", nparray.dtype.str)
        if not m:
            raise ValueError(f"Couldn't find precision for dtype '{nparray.dtype}.")

        fill_value_args = (
            # e.g. ns, D
            m.groups()[0],
        )

    a = root.create_array(
        array_path,
        shape=nparray.shape,
        chunks=chunks,
        dtype=nparray.dtype.str,
        attributes=attributes,
        # compressor=compressor,  # TODO: FIXME
        fill_value=nparray.dtype.type(0, *fill_value_args),
    )

    assert isinstance(a, Array)
    assert nparray.shape == a.shape
    assert chunks == a.chunks
    assert array_path == a.path, (path, name, array_path, a.name, a.path)
    # assert array_path == a.name, (path, name, array_path, a.name, a.path)
    # assert a.basename is None  # TODO
    # assert a.store == normalize_store_arg(store)
    assert dict(a.attrs) == expected_attrs

    a[:] = nparray

    return a


def is_negative_slice(idx: Any) -> bool:
    return isinstance(idx, slice) and idx.step is not None and idx.step < 0


@st.composite  # type: ignore[misc]
def basic_indices(draw: st.DrawFn, *, shape: tuple[int], **kwargs):  # type: ignore[no-untyped-def]
    """Basic indices without unsupported negative slices."""
    return draw(
        npst.basic_indices(shape=shape, **kwargs).filter(
            lambda idxr: (
                not (
                    is_negative_slice(idxr)
                    or (isinstance(idxr, tuple) and any(is_negative_slice(idx) for idx in idxr))
                )
            )
        )
    )


def key_ranges(keys: SearchStrategy = node_names) -> SearchStrategy[list]:
    """
    Function to generate key_ranges strategy for get_partial_values()
    returns list strategy w/ form::

        [(key, (range_start, range_step)),
         (key, (range_start, range_step)),...]
    """
    byte_ranges = st.tuples(
        st.none() | st.integers(min_value=0), st.none() | st.integers(min_value=0)
    )
    key_tuple = st.tuples(keys, byte_ranges)
    key_range_st = st.lists(key_tuple, min_size=1, max_size=10)
    return key_range_st
