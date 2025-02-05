import sys
from typing import Any, Literal

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numcodecs
import numpy as np
from hypothesis import assume, given, settings  # noqa: F401
from hypothesis.strategies import SearchStrategy

import zarr
from zarr.abc.store import RangeByteRequest, Store
from zarr.codecs.bytes import BytesCodec
from zarr.core.array import Array
from zarr.core.chunk_grids import RegularChunkGrid
from zarr.core.chunk_key_encodings import DefaultChunkKeyEncoding
from zarr.core.common import ZarrFormat
from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata
from zarr.core.sync import sync
from zarr.storage import MemoryStore, StoreLike
from zarr.storage._common import _dereference_path

# Copied from Xarray
_attr_keys = st.text(st.characters(), min_size=1)
_attr_values = st.recursive(
    st.none() | st.booleans() | st.text(st.characters(), max_size=5),
    lambda children: st.lists(children) | st.dictionaries(_attr_keys, children),
    max_leaves=3,
)


def v3_dtypes() -> st.SearchStrategy[np.dtype]:
    return (
        npst.boolean_dtypes()
        | npst.integer_dtypes(endianness="=")
        | npst.unsigned_integer_dtypes(endianness="=")
        | npst.floating_dtypes(endianness="=")
        | npst.complex_number_dtypes(endianness="=")
        # | npst.byte_string_dtypes(endianness="=")
        # | npst.unicode_string_dtypes()
        # | npst.datetime64_dtypes()
        # | npst.timedelta64_dtypes()
    )


def v2_dtypes() -> st.SearchStrategy[np.dtype]:
    return (
        npst.boolean_dtypes()
        | npst.integer_dtypes(endianness="=")
        | npst.unsigned_integer_dtypes(endianness="=")
        | npst.floating_dtypes(endianness="=")
        | npst.complex_number_dtypes(endianness="=")
        | npst.byte_string_dtypes(endianness="=")
        | npst.unicode_string_dtypes(endianness="=")
        | npst.datetime64_dtypes(endianness="=")
        # | npst.timedelta64_dtypes()
    )


def safe_unicode_for_dtype(dtype: np.dtype[np.str_]) -> st.SearchStrategy[str]:
    """Generate UTF-8-safe text constrained to max_len of dtype."""
    # account for utf-32 encoding (i.e. 4 bytes/character)
    max_len = max(1, dtype.itemsize // 4)

    return st.text(
        alphabet=st.characters(
            blacklist_categories=["Cs"],  # Avoid *technically allowed* surrogates
            min_codepoint=32,
        ),
        min_size=1,
        max_size=max_len,
    )


def clear_store(x: Store) -> Store:
    sync(x.clear())
    return x


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
keys = st.lists(node_names, min_size=1).map("/".join)
paths = st.just("/") | keys
# st.builds will only call a new store constructor for different keyword arguments
# i.e. stores.examples() will always return the same object per Store class.
# So we map a clear to reset the store.
stores = st.builds(MemoryStore, st.just({})).map(clear_store)
compressors = st.sampled_from([None, "default"])
zarr_formats: st.SearchStrategy[ZarrFormat] = st.sampled_from([3, 2])
array_shapes = npst.array_shapes(max_dims=4, min_side=0)


@st.composite  # type: ignore[misc]
def dimension_names(draw: st.DrawFn, *, ndim: int | None = None) -> list[None | str] | None:
    simple_text = st.text(zarr_key_chars, min_size=0)
    return draw(st.none() | st.lists(st.none() | simple_text, min_size=ndim, max_size=ndim))  # type: ignore[no-any-return]


@st.composite  # type: ignore[misc]
def array_metadata(
    draw: st.DrawFn,
    *,
    array_shapes: st.SearchStrategy[tuple[int, ...]] = npst.array_shapes,
    zarr_formats: st.SearchStrategy[Literal[2, 3]] = zarr_formats,
    attributes: st.SearchStrategy[dict[str, Any]] = attrs,
) -> ArrayV2Metadata | ArrayV3Metadata:
    zarr_format = draw(zarr_formats)
    # separator = draw(st.sampled_from(['/', '\\']))
    shape = draw(array_shapes())
    ndim = len(shape)
    chunk_shape = draw(array_shapes(min_dims=ndim, max_dims=ndim))
    dtype = draw(v3_dtypes())
    fill_value = draw(npst.from_dtype(dtype))
    if zarr_format == 2:
        return ArrayV2Metadata(
            shape=shape,
            chunks=chunk_shape,
            dtype=dtype,
            fill_value=fill_value,
            order=draw(st.sampled_from(["C", "F"])),
            attributes=draw(attributes),
            dimension_separator=draw(st.sampled_from([".", "/"])),
            filters=None,
            compressor=None,
        )
    else:
        return ArrayV3Metadata(
            shape=shape,
            data_type=dtype,
            chunk_grid=RegularChunkGrid(chunk_shape=chunk_shape),
            fill_value=fill_value,
            attributes=draw(attributes),
            dimension_names=draw(dimension_names(ndim=ndim)),
            chunk_key_encoding=DefaultChunkKeyEncoding(separator="/"),  # FIXME
            codecs=[BytesCodec()],
            storage_transformers=(),
        )


@st.composite  # type: ignore[misc]
def numpy_arrays(
    draw: st.DrawFn,
    *,
    shapes: st.SearchStrategy[tuple[int, ...]] = array_shapes,
    zarr_formats: st.SearchStrategy[ZarrFormat] = zarr_formats,
) -> Any:
    """
    Generate numpy arrays that can be saved in the provided Zarr format.
    """
    zarr_format = draw(zarr_formats)
    dtype = draw(v3_dtypes() if zarr_format == 3 else v2_dtypes())
    if np.issubdtype(dtype, np.str_):
        safe_unicode_strings = safe_unicode_for_dtype(dtype)
        return draw(npst.arrays(dtype=dtype, shape=shapes, elements=safe_unicode_strings))

    return draw(npst.arrays(dtype=dtype, shape=shapes))


@st.composite  # type: ignore[misc]
def chunk_shapes(draw: st.DrawFn, *, shape: tuple[int, ...]) -> tuple[int, ...]:
    # We want this strategy to shrink towards arrays with smaller number of chunks
    # 1. st.integers() shrinks towards smaller values. So we use that to generate number of chunks
    numchunks = draw(
        st.tuples(*[st.integers(min_value=0 if size == 0 else 1, max_value=size) for size in shape])
    )
    # 2. and now generate the chunks tuple
    return tuple(
        size // nchunks if nchunks > 0 else 0
        for size, nchunks in zip(shape, numchunks, strict=True)
    )


@st.composite  # type: ignore[misc]
def shard_shapes(
    draw: st.DrawFn, *, shape: tuple[int, ...], chunk_shape: tuple[int, ...]
) -> tuple[int, ...]:
    # We want this strategy to shrink towards arrays with smaller number of shards
    # shards must be an integral number of chunks
    assert all(c != 0 for c in chunk_shape)
    numchunks = tuple(s // c for s, c in zip(shape, chunk_shape, strict=True))
    multiples = tuple(draw(st.integers(min_value=1, max_value=nc)) for nc in numchunks)
    return tuple(m * c for m, c in zip(multiples, chunk_shape, strict=True))


@st.composite  # type: ignore[misc]
def np_array_and_chunks(
    draw: st.DrawFn, *, arrays: st.SearchStrategy[np.ndarray] = numpy_arrays
) -> tuple[np.ndarray, tuple[int, ...]]:  # type: ignore[type-arg]
    """A hypothesis strategy to generate small sized random arrays.

    Returns: a tuple of the array and a suitable random chunking for it.
    """
    array = draw(arrays)
    return (array, draw(chunk_shapes(shape=array.shape)))


@st.composite  # type: ignore[misc]
def arrays(
    draw: st.DrawFn,
    *,
    shapes: st.SearchStrategy[tuple[int, ...]] = array_shapes,
    compressors: st.SearchStrategy = compressors,
    stores: st.SearchStrategy[StoreLike] = stores,
    paths: st.SearchStrategy[str | None] = paths,
    array_names: st.SearchStrategy = array_names,
    arrays: st.SearchStrategy | None = None,
    attrs: st.SearchStrategy = attrs,
    zarr_formats: st.SearchStrategy = zarr_formats,
) -> Array:
    store = draw(stores)
    path = draw(paths)
    name = draw(array_names)
    attributes = draw(attrs)
    zarr_format = draw(zarr_formats)
    if arrays is None:
        arrays = numpy_arrays(shapes=shapes, zarr_formats=st.just(zarr_format))
    nparray = draw(arrays)
    chunk_shape = draw(chunk_shapes(shape=nparray.shape))
    if zarr_format == 3 and all(c > 0 for c in chunk_shape):
        shard_shape = draw(st.none() | shard_shapes(shape=nparray.shape, chunk_shape=chunk_shape))
    else:
        shard_shape = None
    # test that None works too.
    fill_value = draw(st.one_of([st.none(), npst.from_dtype(nparray.dtype)]))
    # compressor = draw(compressors)

    expected_attrs = {} if attributes is None else attributes

    array_path = _dereference_path(path, name)
    root = zarr.open_group(store, mode="w", zarr_format=zarr_format)

    a = root.create_array(
        array_path,
        shape=nparray.shape,
        chunks=chunk_shape,
        shards=shard_shape,
        dtype=nparray.dtype,
        attributes=attributes,
        # compressor=compressor,  # FIXME
        fill_value=fill_value,
    )

    assert isinstance(a, Array)
    if a.metadata.zarr_format == 3:
        assert a.fill_value is not None
    assert a.name is not None
    assert isinstance(root[array_path], Array)
    assert nparray.shape == a.shape
    assert chunk_shape == a.chunks
    assert shard_shape == a.shards
    assert array_path == a.path, (path, name, array_path, a.name, a.path)
    assert a.basename == name, (a.basename, name)
    assert dict(a.attrs) == expected_attrs

    a[:] = nparray

    return a


def is_negative_slice(idx: Any) -> bool:
    return isinstance(idx, slice) and idx.step is not None and idx.step < 0


@st.composite  # type: ignore[misc]
def basic_indices(draw: st.DrawFn, *, shape: tuple[int], **kwargs: Any) -> Any:
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


@st.composite  # type: ignore[misc]
def orthogonal_indices(
    draw: st.DrawFn, *, shape: tuple[int]
) -> tuple[tuple[np.ndarray[Any, Any], ...], tuple[np.ndarray[Any, Any], ...]]:
    """
    Strategy that returns
    (1) a tuple of integer arrays used for orthogonal indexing of Zarr arrays.
    (2) an tuple of integer arrays that can be used for equivalent indexing of numpy arrays
    """
    zindexer = []
    npindexer = []
    ndim = len(shape)
    for axis, size in enumerate(shape):
        val = draw(
            npst.integer_array_indices(
                shape=(size,), result_shape=npst.array_shapes(min_side=1, max_side=size, max_dims=1)
            )
            | basic_indices(min_dims=1, shape=(size,), allow_ellipsis=False)
            .map(lambda x: (x,) if not isinstance(x, tuple) else x)  # bare ints, slices
            .filter(bool)  # skip empty tuple
        )
        (idxr,) = val
        if isinstance(idxr, int):
            idxr = np.array([idxr])
        zindexer.append(idxr)
        if isinstance(idxr, slice):
            idxr = np.arange(*idxr.indices(size))
        elif isinstance(idxr, (tuple, int)):
            idxr = np.array(idxr)
        newshape = [1] * ndim
        newshape[axis] = idxr.size
        npindexer.append(idxr.reshape(newshape))

    # casting the output of broadcast_arrays is needed for numpy 1.25
    return tuple(zindexer), tuple(np.broadcast_arrays(*npindexer))


def key_ranges(
    keys: SearchStrategy = node_names, max_size: int = sys.maxsize
) -> SearchStrategy[list[int]]:
    """
    Function to generate key_ranges strategy for get_partial_values()
    returns list strategy w/ form::

        [(key, (range_start, range_end)),
         (key, (range_start, range_end)),...]
    """

    def make_request(start: int, length: int) -> RangeByteRequest:
        return RangeByteRequest(start, end=min(start + length, max_size))

    byte_ranges = st.builds(
        make_request,
        start=st.integers(min_value=0, max_value=max_size),
        length=st.integers(min_value=0, max_value=max_size),
    )
    key_tuple = st.tuples(keys, byte_ranges)
    return st.lists(key_tuple, min_size=1, max_size=10)


def simple_text():
    """A strategy for generating simple text strings."""
    return st.text(st.characters(min_codepoint=32, max_codepoint=126), min_size=1, max_size=10)


def simple_attrs():
    """A strategy for generating simple attribute dictionaries."""
    return st.dictionaries(
        simple_text(),
        st.one_of(
            st.integers(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.booleans(),
            simple_text(),
        ),
    )


def array_shapes(min_dims=1, max_dims=3, max_len=100):
    """A strategy for generating array shapes."""
    return st.lists(
        st.integers(min_value=1, max_value=max_len), min_size=min_dims, max_size=max_dims
    )


# def zarr_compressors():
#     """A strategy for generating Zarr compressors."""
#     return st.sampled_from([None, Blosc(), GZip(), Zstd(), LZ4()])


# def zarr_codecs():
#     """A strategy for generating Zarr codecs."""
#     return st.sampled_from([BytesCodec(), Blosc(), GZip(), Zstd(), LZ4()])


def zarr_filters():
    """A strategy for generating Zarr filters."""
    return st.lists(
        st.just(numcodecs.Delta(dtype="i4")), min_size=0, max_size=2
    )  # Example filter, expand as needed


def zarr_storage_transformers():
    """A strategy for generating Zarr storage transformers."""
    return st.lists(
        st.dictionaries(
            simple_text(), st.one_of(st.integers(), st.floats(), st.booleans(), simple_text())
        ),
        min_size=0,
        max_size=2,
    )


@st.composite
def array_metadata_v2(draw: st.DrawFn) -> ArrayV2Metadata:
    """Generates valid ArrayV2Metadata objects for property-based testing."""
    dims = draw(st.integers(min_value=1, max_value=3))  # Limit dimensions for complexity
    shape = tuple(draw(array_shapes(min_dims=dims, max_dims=dims, max_len=100)))
    max_chunk_len = max(shape) if shape else 100
    chunks = tuple(
        draw(
            st.lists(
                st.integers(min_value=1, max_value=max_chunk_len), min_size=dims, max_size=dims
            )
        )
    )

    # Validate shape and chunks relationship
    assume(all(c <= s for s, c in zip(shape, chunks, strict=False)))  # Chunk size must be <= shape

    dtype = draw(v2_dtypes())
    fill_value = draw(st.one_of([st.none(), npst.from_dtype(dtype)]))
    order = draw(st.sampled_from(["C", "F"]))
    dimension_separator = draw(st.sampled_from([".", "/"]))
    # compressor = draw(zarr_compressors())
    filters = tuple(draw(zarr_filters())) if draw(st.booleans()) else None
    attributes = draw(simple_attrs())

    # Construct the metadata object.  Type hints are crucial here for correctness.
    return ArrayV2Metadata(
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        fill_value=fill_value,
        order=order,
        dimension_separator=dimension_separator,
        #    compressor=compressor,
        filters=filters,
        attributes=attributes,
    )


@st.composite
def array_metadata_v3(draw: st.DrawFn) -> ArrayV3Metadata:
    """Generates valid ArrayV3Metadata objects for property-based testing."""
    dims = draw(st.integers(min_value=1, max_value=3))
    shape = tuple(draw(array_shapes(min_dims=dims, max_dims=dims, max_len=100)))
    max_chunk_len = max(shape) if shape else 100
    chunks = tuple(
        draw(
            st.lists(
                st.integers(min_value=1, max_value=max_chunk_len), min_size=dims, max_size=dims
            )
        )
    )
    assume(all(c <= s for s, c in zip(shape, chunks, strict=False)))

    dtype = draw(v3_dtypes())
    fill_value = draw(npst.from_dtype(dtype))
    chunk_grid = RegularChunkGrid(chunks)  # Ensure chunks is passed as tuple.
    chunk_key_encoding = DefaultChunkKeyEncoding(separator="/")  # Or st.sampled_from(["/", "."])
    # codecs = tuple(draw(st.lists(zarr_codecs(), min_size=0, max_size=3)))
    attributes = draw(simple_attrs())
    dimension_names = (
        tuple(draw(st.lists(st.one_of(st.none(), simple_text()), min_size=dims, max_size=dims)))
        if draw(st.booleans())
        else None
    )
    storage_transformers = tuple(draw(zarr_storage_transformers()))

    return ArrayV3Metadata(
        shape=shape,
        data_type=dtype,
        chunk_grid=chunk_grid,
        chunk_key_encoding=chunk_key_encoding,
        fill_value=fill_value,
        #    codecs=codecs,
        attributes=attributes,
        dimension_names=dimension_names,
        storage_transformers=storage_transformers,
    )
