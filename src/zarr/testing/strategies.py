import itertools
import math
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
import numpy.typing as npt
from hypothesis import event
from hypothesis.strategies import SearchStrategy

import zarr
from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    Store,
    SuffixByteRequest,
)
from zarr.codecs.bytes import BytesCodec
from zarr.codecs.crc32c_ import Crc32cCodec
from zarr.codecs.sharding import SUBCHUNK_WRITE_ORDER, ShardingCodec, SubchunkWriteOrder
from zarr.codecs.zstd import ZstdCodec
from zarr.core.array import Array, CompressorsLike, SerializerLike
from zarr.core.chunk_key_encodings import DefaultChunkKeyEncoding
from zarr.core.common import JSON, AccessModeLiteral, ZarrFormat
from zarr.core.dtype import get_data_type_from_native_dtype
from zarr.core.indexing import Selection
from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata
from zarr.core.metadata.v3 import RectilinearChunkGridMetadata, RegularChunkGridMetadata
from zarr.core.sync import sync
from zarr.storage import MemoryStore, StoreLike
from zarr.storage._utils import _join_paths, normalize_path
from zarr.types import AnyArray

TrueOrFalse = Literal[True, False]

# Copied from Xarray
_attr_keys = st.text(st.characters(), min_size=1)
_attr_values = st.recursive(
    st.none() | st.booleans() | st.text(st.characters(), max_size=5),
    lambda children: st.lists(children) | st.dictionaries(_attr_keys, children),
    max_leaves=3,
)


@st.composite
def keys(draw: st.DrawFn, *, max_num_nodes: int | None = None) -> str:
    return draw(st.lists(node_names, min_size=1, max_size=max_num_nodes).map("/".join))


@st.composite
def paths(draw: st.DrawFn, *, max_num_nodes: int | None = None) -> str:
    return draw(st.just("/") | keys(max_num_nodes=max_num_nodes))


def dtypes() -> st.SearchStrategy[np.dtype[Any]]:
    return (
        npst.boolean_dtypes()
        | npst.integer_dtypes(endianness="=")
        | npst.unsigned_integer_dtypes(endianness="=")
        | npst.floating_dtypes(endianness="=")
        | npst.complex_number_dtypes(endianness="=")
        | npst.byte_string_dtypes(endianness="=")
        | npst.unicode_string_dtypes(endianness="=")
        | npst.datetime64_dtypes(endianness="=")
        | npst.timedelta64_dtypes(endianness="=")
    )


def v3_dtypes() -> st.SearchStrategy[np.dtype[Any]]:
    return dtypes()


def v2_dtypes() -> st.SearchStrategy[np.dtype[Any]]:
    return dtypes()


def safe_unicode_for_dtype(dtype: np.dtype[np.str_]) -> st.SearchStrategy[str]:
    """Generate UTF-8-safe text constrained to max_len of dtype."""
    # account for utf-32 encoding (i.e. 4 bytes/character)
    max_len = max(1, dtype.itemsize // 4)

    return st.text(
        alphabet=st.characters(
            exclude_categories=["Cs"],  # Avoid *technically allowed* surrogates
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
node_names = (
    st.text(zarr_key_chars, min_size=1)
    .filter(lambda t: t not in (".", "..") and not t.startswith("__"))
    .filter(lambda name: name.lower() != "zarr.json")
)
short_node_names = (
    st.text(zarr_key_chars, max_size=3, min_size=1)
    .filter(lambda t: t not in (".", "..") and not t.startswith("__"))
    .filter(lambda name: name.lower() != "zarr.json")
)
array_names = node_names
attrs: st.SearchStrategy[Mapping[str, JSON] | None] = st.none() | st.dictionaries(
    _attr_keys, _attr_values
)
# st.builds will only call a new store constructor for different keyword arguments
# i.e. stores.examples() will always return the same object per Store class.
# So we map a clear to reset the store.
stores = st.builds(MemoryStore, st.just({})).map(clear_store)
compressors = st.sampled_from([None, "default"])
zarr_formats: st.SearchStrategy[ZarrFormat] = st.sampled_from([3, 2])
# We de-prioritize arrays having dim sizes 0, 1, 2
array_shapes = npst.array_shapes(max_dims=4, min_side=3, max_side=5) | npst.array_shapes(
    max_dims=4, min_side=0
)


@st.composite
def dimension_names(draw: st.DrawFn, *, ndim: int | None = None) -> list[None | str] | None:
    simple_text = st.text(zarr_key_chars, min_size=0)
    return draw(st.none() | st.lists(st.none() | simple_text, min_size=ndim, max_size=ndim))  # type: ignore[arg-type]


subchunk_write_orders: st.SearchStrategy[SubchunkWriteOrder] = st.sampled_from(SUBCHUNK_WRITE_ORDER)

# Inner codec chains for a ShardingCodec. We MUST sample the uncompressed,
# single-BytesCodec configuration (no Zstd) — that is the only configuration in
# which the FusedCodecPipeline's vectorized whole-shard "bulk decode" fast path
# engages, so it is the only one that can exercise (and regress-guard) that path
# against arbitrary indexing. Freezing the inner codecs to [BytesCodec, ZstdCodec]
# silently disables the fast path under every property test.
sharding_inner_codecs: st.SearchStrategy[list[BytesCodec | ZstdCodec]] = st.sampled_from(
    [
        [BytesCodec()],
        [BytesCodec(), ZstdCodec()],
    ]
)


@st.composite
def array_metadata(
    draw: st.DrawFn,
    *,
    array_shapes: Callable[..., st.SearchStrategy[tuple[int, ...]]] = npst.array_shapes,
    zarr_formats: st.SearchStrategy[ZarrFormat] = zarr_formats,
    attributes: SearchStrategy[Mapping[str, JSON] | None] = attrs,
) -> ArrayV2Metadata | ArrayV3Metadata:
    zarr_format = draw(zarr_formats)
    # separator = draw(st.sampled_from(['/', '\\']))
    shape = draw(array_shapes())
    ndim = len(shape)
    np_dtype = draw(dtypes())
    dtype = get_data_type_from_native_dtype(np_dtype)
    fill_value = draw(npst.from_dtype(np_dtype))
    if zarr_format == 2:
        chunk_shape = draw(array_shapes(min_dims=ndim, max_dims=ndim, min_side=1))
        return ArrayV2Metadata(
            shape=shape,
            chunks=chunk_shape,
            dtype=dtype,
            fill_value=fill_value,
            order=draw(st.sampled_from(["C", "F"])),
            attributes=draw(attributes),  # type: ignore[arg-type]
            dimension_separator=draw(st.sampled_from([".", "/"])),
            filters=None,
            compressor=None,
        )
    else:
        chunk_grid = draw(chunk_grids(shape=shape))
        return ArrayV3Metadata(
            shape=shape,
            data_type=dtype,
            chunk_grid=chunk_grid,
            fill_value=fill_value,
            attributes=draw(attributes),  # type: ignore[arg-type]
            dimension_names=draw(dimension_names(ndim=ndim)),
            chunk_key_encoding=DefaultChunkKeyEncoding(separator="/"),  # FIXME
            codecs=[BytesCodec()],
            storage_transformers=(),
        )


@st.composite
def numpy_arrays(
    draw: st.DrawFn,
    *,
    shapes: st.SearchStrategy[tuple[int, ...]] = array_shapes,
    dtype: np.dtype[Any] | None = None,
) -> npt.NDArray[Any]:
    """
    Generate numpy arrays that can be saved in the provided Zarr format.
    """
    if dtype is None:
        dtype = draw(dtypes())
    if np.issubdtype(dtype, np.str_):
        safe_unicode_strings = safe_unicode_for_dtype(dtype)
        return draw(npst.arrays(dtype=dtype, shape=shapes, elements=safe_unicode_strings))

    return draw(npst.arrays(dtype=dtype, shape=shapes))


@st.composite
def chunk_shapes(draw: st.DrawFn, *, shape: tuple[int, ...]) -> tuple[int, ...]:
    # We want this strategy to shrink towards arrays with smaller number of chunks
    # 1. st.integers() shrinks towards smaller values. So we use that to generate number of chunks
    numchunks = draw(
        st.tuples(
            *[
                st.integers(min_value=0 if size == 0 else 1, max_value=max(size, 1))
                for size in shape
            ]
        )
    )
    # 2. and now generate the chunks tuple
    # Chunk sizes must be >= 1 per spec; for zero-extent dimensions use 1.
    chunks = tuple(
        max(1, size // nchunks) if nchunks > 0 else 1
        for size, nchunks in zip(shape, numchunks, strict=True)
    )

    for c in chunks:
        event("chunk size", c)

    if any((c != 0 and s % c != 0) for s, c in zip(shape, chunks, strict=True)):
        event("smaller last chunk")

    return chunks


@st.composite
def shard_shapes(
    draw: st.DrawFn, *, shape: tuple[int, ...], chunk_shape: tuple[int, ...]
) -> tuple[int, ...]:
    # We want this strategy to shrink towards arrays with smaller number of shards
    # shards must be an integral number of chunks
    assert all(c != 0 for c in chunk_shape)
    numchunks = tuple(s // c for s, c in zip(shape, chunk_shape, strict=True))
    multiples = tuple(draw(st.integers(min_value=1, max_value=nc)) for nc in numchunks)
    return tuple(m * c for m, c in zip(multiples, chunk_shape, strict=True))


@st.composite
def np_array_and_chunks(
    draw: st.DrawFn,
    *,
    arrays: st.SearchStrategy[npt.NDArray[Any]] = numpy_arrays(),  # noqa: B008
) -> tuple[np.ndarray[Any, Any], tuple[int, ...]]:
    """A hypothesis strategy to generate small sized random arrays.

    Returns: a tuple of the array and a suitable random chunking for it.
    """
    array = draw(arrays)
    return (array, draw(chunk_shapes(shape=array.shape)))


@st.composite
def arrays(
    draw: st.DrawFn,
    *,
    shapes: st.SearchStrategy[tuple[int, ...]] = array_shapes,
    compressors: st.SearchStrategy = compressors,
    stores: st.SearchStrategy[StoreLike] = stores,
    paths: st.SearchStrategy[str] = paths(),  # noqa: B008
    array_names: st.SearchStrategy = array_names,
    arrays: st.SearchStrategy | None = None,
    attrs: st.SearchStrategy = attrs,
    zarr_formats: st.SearchStrategy = zarr_formats,
    subchunk_write_orders: SearchStrategy[SubchunkWriteOrder] = subchunk_write_orders,
    open_mode: AccessModeLiteral = "w",
) -> AnyArray:
    store = draw(stores, label="store")
    path = draw(paths, label="array parent")
    name = draw(array_names, label="array name")
    attributes = draw(attrs, label="attributes")
    zarr_format = draw(zarr_formats, label="zarr format")
    if arrays is None:
        arrays = numpy_arrays(shapes=shapes)
    nparray = draw(arrays, label="array data")
    dim_names: None | list[str | None] = None
    serializer: SerializerLike = "auto"
    compressors_unsearched: CompressorsLike = "auto"

    # For v3 arrays, optionally use RectilinearChunkGridMetadata
    chunk_grid_meta: RegularChunkGridMetadata | RectilinearChunkGridMetadata | None = None

    # test that None works too.
    fill_value = draw(st.one_of([st.none(), npst.from_dtype(nparray.dtype)]))
    # compressor = draw(compressors)

    expected_attrs = {} if attributes is None else attributes

    array_path = _join_paths([path, name])
    root = zarr.open_group(store, mode=open_mode, zarr_format=zarr_format)

    # Convert chunk grid metadata to a form create_array accepts:
    # - RegularChunkGridMetadata -> flat tuple of ints
    # - RectilinearChunkGridMetadata -> nested list of ints (triggers rectilinear path)
    # - v2 -> flat tuple of ints
    chunks_param: tuple[int, ...] | list[list[int]]
    shard_shape = None
    dim_names = None
    if zarr_format == 3:
        chunk_grid_meta = draw(st.none() | chunk_grids(shape=nparray.shape), label="chunk grid")
        dim_names = draw(dimension_names(ndim=nparray.ndim), label="dimension names")
        if isinstance(chunk_grid_meta, RectilinearChunkGridMetadata):
            chunks_param = [
                list(dim) if isinstance(dim, tuple) else [dim]
                for dim in chunk_grid_meta.chunk_shapes
            ]
        elif isinstance(chunk_grid_meta, RegularChunkGridMetadata):
            chunks_param = chunk_grid_meta.chunk_shape
        else:
            chunks_param = draw(chunk_shapes(shape=nparray.shape), label="chunk shape")

            if all(s > c and c > 1 for s, c in zip(nparray.shape, chunks_param, strict=True)):
                shard_shape = draw(
                    st.none() | shard_shapes(shape=nparray.shape, chunk_shape=chunks_param),
                    label="shard shape",
                )
                if shard_shape is not None:
                    subchunk_write_order = draw(subchunk_write_orders)
                    inner_codecs = draw(sharding_inner_codecs, label="sharding inner codecs")
                    serializer = ShardingCodec(
                        subchunk_write_order=subchunk_write_order,
                        codecs=inner_codecs,
                        index_codecs=[BytesCodec(), Crc32cCodec()],
                        chunk_shape=chunks_param,
                    )
                    compressors_unsearched = None
    else:
        chunks_param = draw(chunk_shapes(shape=nparray.shape), label="chunk shape")
    a = root.create_array(
        array_path,
        shape=nparray.shape,
        chunks=chunks_param,
        shards=shard_shape,
        dtype=nparray.dtype,
        attributes=attributes,
        compressors=compressors_unsearched,  # FIXME
        fill_value=fill_value,
        dimension_names=dim_names,
        serializer=serializer,
    )

    assert isinstance(a, Array)
    if a.metadata.zarr_format == 3:
        assert a.fill_value is not None
    assert a.name is not None
    assert a.path == normalize_path(array_path)
    assert a.name == f"/{a.path}"
    assert isinstance(root[array_path], Array)
    assert nparray.shape == a.shape

    # Verify chunks — for rectilinear grids, .chunks raises
    if zarr_format == 3:
        assert shard_shape == a.shards
        if isinstance(a.metadata.chunk_grid, RegularChunkGridMetadata):
            assert a.metadata.chunk_grid.chunk_shape == (
                a.shards if shard_shape is not None else a.chunks
            )
            assert shard_shape == a.shards
        else:
            assert isinstance(a.metadata.chunk_grid, RectilinearChunkGridMetadata)
            assert shard_shape is None

    assert a.basename == name, (a.basename, name)
    assert dict(a.attrs) == expected_attrs

    a[:] = nparray

    return a


@st.composite
def simple_arrays(
    draw: st.DrawFn,
    *,
    shapes: st.SearchStrategy[tuple[int, ...]] = array_shapes,
) -> Any:
    return draw(
        arrays(
            shapes=shapes,
            paths=paths(max_num_nodes=2),
            array_names=short_node_names,
            attrs=st.none(),
            compressors=st.sampled_from([None, "default"]),
        )
    )


@st.composite
def rectilinear_chunks(draw: st.DrawFn, *, shape: tuple[int, ...]) -> list[list[int]]:
    """Generate valid rectilinear chunk shapes for a given array shape.

    Uses two modes per dimension:
    - "expanded": random divider points create arbitrary chunk sizes
    - "rle": uniform chunks with optional remainder, optionally shuffled

    Keeps max chunks per dimension <= 20 to avoid performance issues
    in property tests. With higher dimensions, the total chunk count
    grows multiplicatively.
    """
    chunk_shapes: list[list[int]] = []
    for size in shape:
        assert size > 0
        if size > 1:
            mode = draw(st.sampled_from(["expanded", "rle"]))
            if mode == "expanded":
                event("rectilinear expanded")
                max_chunks = min(size - 1, 20)
                nchunks = draw(st.integers(min_value=1, max_value=max_chunks))
                dividers = sorted(
                    draw(
                        st.lists(
                            st.integers(min_value=1, max_value=size - 1),
                            min_size=nchunks - 1,
                            max_size=nchunks - 1,
                            unique=True,
                        )
                    )
                )
                chunk_shapes.append(
                    [a - b for a, b in zip(dividers + [size], [0] + dividers, strict=False)]
                )
            else:
                # RLE mode: uniform chunks with optional remainder
                max_chunk_size = min(size, 20)
                chunk_size = draw(st.integers(min_value=1, max_value=max_chunk_size))
                n_full = size // chunk_size
                remainder = size % chunk_size
                chunks_list = [chunk_size] * n_full
                if remainder > 0:
                    chunks_list.append(remainder)
                # Optionally shuffle to create non-contiguous duplicate patterns
                if draw(st.booleans()):
                    event("rectilinear rle shuffled")
                    chunks_list = draw(st.permutations(chunks_list))
                else:
                    event("rectilinear rle")
                chunk_shapes.append(list(chunks_list))
        else:
            chunk_shapes.append([1])
    return chunk_shapes


@st.composite
def chunk_grids(
    draw: st.DrawFn, *, shape: tuple[int, ...]
) -> RegularChunkGridMetadata | RectilinearChunkGridMetadata:
    """Generate either a RegularChunkGridMetadata or RectilinearChunkGridMetadata.

    This strategy depends on the global state of the config having rectilinear chunk grids enabled or not.
    This means that it may be a possible source of a hypothesis FlakyStrategy error due dependence
    on global state. However, in practice this seems unlikely to happen.

    This allows property tests to exercise both chunk grid types.
    """
    # RectilinearChunkGridMetadata doesn't support zero-sized dimensions,
    # so use RegularChunkGridMetadata if any dimension is 0
    if any(s == 0 for s in shape):
        event("using RegularChunkGridMetadata (zero-sized dimensions)")
        return RegularChunkGridMetadata(chunk_shape=draw(chunk_shapes(shape=shape)))

    if zarr.config.get("array.rectilinear_chunks") and draw(st.booleans()):
        chunks = draw(rectilinear_chunks(shape=shape))
        event("using RectilinearChunkGridMetadata")
        return RectilinearChunkGridMetadata(chunk_shapes=tuple(tuple(dim) for dim in chunks))
    else:
        event("using RegularChunkGridMetadata")
        return RegularChunkGridMetadata(chunk_shape=draw(chunk_shapes(shape=shape)))


# Rectilinear arrays need min_side >= 1 so every dimension has at least one element
_rectilinear_shapes = npst.array_shapes(max_dims=3, min_side=1, max_side=20)


@st.composite
def rectilinear_arrays(
    draw: st.DrawFn,
    *,
    shapes: st.SearchStrategy[tuple[int, ...]] = _rectilinear_shapes,
) -> Any:
    """Generate a zarr v3 array with rectilinear (variable) chunk grid."""
    shape = draw(shapes)
    chunk_shapes = draw(rectilinear_chunks(shape=shape))

    np_dtype = draw(dtypes())
    nparray = draw(numpy_arrays(shapes=st.just(shape), dtype=np_dtype))
    fill_value = draw(st.one_of([st.none(), npst.from_dtype(np_dtype)]))
    dim_names = draw(dimension_names(ndim=len(shape)))

    store = MemoryStore()
    with zarr.config.set({"array.rectilinear_chunks": True}):
        a = zarr.create_array(
            store=store,
            shape=shape,
            chunks=chunk_shapes,
            dtype=np_dtype,
            fill_value=fill_value,
            dimension_names=dim_names,
        )
        a[:] = nparray

    return a


def is_negative_slice(idx: Any) -> bool:
    return isinstance(idx, slice) and idx.step is not None and idx.step < 0


@st.composite
def end_slices(draw: st.DrawFn, *, shape: tuple[int, ...]) -> Any:
    """
    A strategy that slices ranges that include the last chunk.
    This is intended to stress-test handling of a possibly smaller last chunk.
    """
    slicers = []
    for size in shape:
        start = draw(st.integers(min_value=size // 2, max_value=size - 1))
        length = draw(st.integers(min_value=0, max_value=size - start))
        slicers.append(slice(start, start + length))
    event("drawing end slice")
    return tuple(slicers)


@st.composite
def basic_indices(
    draw: st.DrawFn,
    *,
    shape: tuple[int, ...],
    min_dims: int = 0,
    max_dims: int | None = None,
    allow_newaxis: TrueOrFalse = False,
    allow_ellipsis: TrueOrFalse = True,
) -> Any:
    """Basic indices without unsupported negative slices."""
    strategy = npst.basic_indices(
        shape=shape,
        min_dims=min_dims,
        max_dims=max_dims,
        allow_newaxis=allow_newaxis,
        allow_ellipsis=allow_ellipsis,
    ).filter(
        lambda idxr: (
            not (
                is_negative_slice(idxr)
                or (isinstance(idxr, tuple) and any(is_negative_slice(idx) for idx in idxr))
            )
        )
    )
    if math.prod(shape) >= 3:
        strategy = end_slices(shape=shape) | strategy
    return draw(strategy)


@st.composite
def orthogonal_indices(
    draw: st.DrawFn, *, shape: tuple[int, ...]
) -> tuple[tuple[np.ndarray[Any, Any], ...], tuple[np.ndarray[Any, Any], ...]]:
    """
    Strategy that returns
    (1) a tuple of integer arrays used for orthogonal indexing of Zarr arrays.
    (2) a tuple of integer arrays that can be used for equivalent indexing of numpy arrays
    """
    zindexer = []
    npindexer = []
    ndim = len(shape)
    for axis, size in enumerate(shape):
        if size != 0:
            strategy = npst.integer_array_indices(
                shape=(size,), result_shape=npst.array_shapes(min_side=1, max_side=size, max_dims=1)
            ) | basic_indices(min_dims=1, shape=(size,), allow_ellipsis=False)
        else:
            strategy = basic_indices(min_dims=1, shape=(size,), allow_ellipsis=False)

        val = draw(
            strategy
            # bare ints, slices
            .map(lambda x: (x,) if not isinstance(x, tuple) else x)
            # skip empty tuple
            .filter(bool)
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

    # casting the output of broadcast_arrays is needed for numpy < 2
    return tuple(zindexer), tuple(np.broadcast_arrays(*npindexer))


IndexMode = Literal["basic", "oindex", "vindex", "mask"]


@st.composite
def windows(draw: st.DrawFn, *, shape: tuple[int, ...]) -> tuple[slice, ...]:
    """A non-negative, full-rank tuple of slice windows — one per axis.

    A rank-preserving sub-region selection: each axis gets `start:stop` with
    `0 <= start < stop <= size` (an empty `0:0` slice for a zero-length axis).
    Bounds stay non-negative so the window is valid for any consumer, including
    those that treat negative indices as literal coordinates rather than
    from-the-end (e.g. building a sub-array view).
    """
    out: list[slice] = []
    for size in shape:
        if size == 0:
            out.append(slice(0, 0))
            continue
        start = draw(st.integers(min_value=0, max_value=size - 1))
        stop = draw(st.integers(min_value=start + 1, max_value=size))
        out.append(slice(start, stop))
    return tuple(out)


@st.composite
def numpy_array_indexers(
    draw: st.DrawFn, *, mode: IndexMode, shape: tuple[int, ...]
) -> tuple[Selection, Selection]:
    """A `(zarr_selection, numpy_selection)` pair for `mode` on `shape`.

    Scope: the *element-space* indexing modes that have a direct NumPy-array
    equivalent, so a NumPy array of `shape` can serve as the correctness oracle.
    One strategy covers them all, so a test can be written once and parametrized
    over mode instead of re-deriving selection setup per test:

    - `"basic"`  — slices / ints / ellipsis (no newaxis, no negative slices)
    - `"oindex"` — per-axis integer arrays or slices (orthogonal / outer product)
    - `"vindex"` — broadcast integer coordinate arrays (vectorized)
    - `"mask"`   — a boolean array of `shape`

    The two returned selections differ only for `oindex` (zarr's per-axis
    spelling vs the `np.ix_`-style spelling numpy needs); for the other modes
    the same object indexes both a zarr array and its numpy reference. The
    array-based modes (`oindex`/`vindex`/`mask`) need `shape` to have no
    zero-length axis; `basic` has no such requirement.

    Deliberately excluded is **block** indexing (`Array.blocks` /
    `get_block_selection`): it addresses the *chunk grid*, not array elements,
    so it is parametrized by the chunk grid rather than `shape` and has no
    NumPy-array equivalent to compare against — its oracle is a coordinate
    translation built by the separate `block_indices` strategy.
    """
    if mode == "basic":
        sel = draw(basic_indices(shape=shape))
        return sel, sel
    if mode == "oindex":
        return draw(orthogonal_indices(shape=shape))
    if mode == "vindex":
        idx = draw(
            npst.integer_array_indices(
                shape=shape, result_shape=npst.array_shapes(min_side=1, max_dims=None)
            )
        )
        return idx, idx
    if mode == "mask":
        m = draw(npst.arrays(dtype=np.bool_, shape=st.just(shape)))
        return m, m
    raise ValueError(f"unknown indexing mode: {mode!r}")


# --- Composed indexing programs -------------------------------------------------
#
# A *program* is a short sequence of indexing operations composed on a single
# array, plus an execution recipe describing how to realize the composed
# selection (read whole, read a sub-selection eagerly, read into an ``out=``
# buffer, or write). It exists to differentially test the index-transform layer:
# every operation composes a real transform, and the runner (in
# ``tests/test_properties.py``) checks the composed result against a NumPy oracle
# that applies the equivalent selections in lockstep.
#
# The mode vocabulary here is the *public dialect* ("orthogonal"/"vectorized"),
# not the internal accessor names ("oindex"/"vindex"); the runner maps between
# them.
ProgramMode = Literal["basic", "orthogonal", "vectorized"]
ProgramExecution = Literal["materialize", "eager_on_lazy", "out", "set_scalar", "set_array"]


@dataclass(frozen=True)
class IndexOperation:
    """One indexing step in an :class:`IndexingProgram`.

    ``selection`` carries exactly what the runner needs to reproduce the step on
    *both* a zarr array and its NumPy reference:

    - ``"basic"`` — a single object used verbatim for both (slice / int /
      ellipsis / tuple thereof).
    - ``"orthogonal"`` — a ``(zarr_selection, numpy_selection)`` pair, where the
      NumPy side is the ``np.ix_`` outer-product spelling of the per-axis
      selection.
    - ``"vectorized"`` — a single coordinate selection used verbatim for both
      (plain NumPy fancy-indexing semantics).
    """

    mode: ProgramMode
    selection: Any


@dataclass(frozen=True)
class IndexingProgram:
    """A composed sequence of :class:`IndexOperation` plus an execution recipe."""

    operations: tuple[IndexOperation, ...]
    execution: ProgramExecution


# Small arrays keep 300 Hypothesis examples cheap while still crossing chunk
# boundaries (the runner chunks each axis into ~2 pieces). ``min_side=0`` lets a
# program start from — or a basic step produce — a zero-extent axis, which is the
# required "empty result" coverage class.
program_shapes = npst.array_shapes(min_dims=1, max_dims=3, min_side=0, max_side=4)


def _basic_result_shape(shape: tuple[int, ...], selection: Any) -> tuple[int, ...]:
    """The NumPy result shape of applying a *basic* ``selection`` to ``shape``.

    Used to carry the visible shape forward while generating a program, so each
    step's selection is drawn against the shape the previous steps leave behind.
    Uses a tiny placeholder array (program shapes are capped small).
    """
    result_shape: tuple[int, ...] = np.empty(shape, dtype=np.int8)[selection].shape
    return result_shape


@st.composite
def indexing_programs(draw: st.DrawFn, *, shape: tuple[int, ...]) -> IndexingProgram:
    """Generate a composable indexing program over an array of ``shape``.

    Structure (one to three operations): a prefix of zero-to-two **basic** steps,
    optionally followed by a single **fancy** (orthogonal/vectorized) step as the
    *last* operation. The visible shape is carried forward so every step is valid
    against what its predecessors produce.

    **Why fancy is confined to the last position** — the exclusions here are
    deliberate and documented so they can shrink as the underlying bugs are
    fixed:

    - *Fancy-after-fancy* composition (applying oindex/vindex to a view that
      already carries an orthogonal ArrayMap axis) is unsupported: it raises a
      clear ``NotImplementedError`` at composition time rather than resolving, so
      at most one fancy step is generated.
    - *Basic-after-fancy* is excluded wholesale. Integer basic indexing on an
      oindex-picked axis is a strict xfail
      (``TestKnownFancyIntBugs::test_int_read_on_oindex_view``); rather than track
      which composed axes became "fancy-picked" and admit only the working
      basic-on-non-fancy-axis subset, we keep fancy last and let the basic prefix
      cover multi-step basic composition. (When those bugs are fixed this
      restriction can be relaxed to interleave basic and fancy freely.)

    A fancy step is only emitted when the current shape has rank ``>= 1`` and no
    zero-length axis (integer/orthogonal selections cannot address an empty axis);
    otherwise the program stays basic-only. If nothing else was generated, a
    single basic step is appended so a program always has at least one operation.
    """
    n_basic = draw(st.integers(min_value=0, max_value=2))
    want_fancy = draw(st.booleans())

    operations: list[IndexOperation] = []
    cur = shape
    for _ in range(n_basic):
        if len(cur) == 0:
            break  # a rank-0 (scalar) view has no further indexing to compose
        sel = draw(basic_indices(shape=cur))
        operations.append(IndexOperation("basic", sel))
        cur = _basic_result_shape(cur, sel)

    fancy_ok = len(cur) > 0 and all(s > 0 for s in cur)
    if want_fancy and fancy_ok:
        mode = draw(st.sampled_from(("orthogonal", "vectorized")))
        if mode == "orthogonal":
            zsel, npsel = draw(orthogonal_indices(shape=cur))
            operations.append(IndexOperation("orthogonal", (zsel, npsel)))
        else:
            idx = draw(
                npst.integer_array_indices(
                    shape=cur,
                    result_shape=npst.array_shapes(min_side=1, max_side=4, max_dims=2),
                )
            )
            operations.append(IndexOperation("vectorized", idx))

    if len(operations) == 0:
        operations.append(IndexOperation("basic", draw(basic_indices(shape=cur))))

    executions: tuple[ProgramExecution, ...] = (
        "materialize",
        "eager_on_lazy",
        "out",
        "set_scalar",
        "set_array",
    )
    execution = draw(st.sampled_from(executions))
    return IndexingProgram(tuple(operations), execution)


@st.composite
def block_indices(
    draw: st.DrawFn, *, chunk_sizes: tuple[tuple[int, ...], ...]
) -> tuple[tuple[int | slice, ...], tuple[slice, ...]]:
    """
    Strategy for block-selection indexers over a chunk grid.

    Block indexing is basic indexing applied to the block grid (the grid of
    chunks), so each axis is drawn with `basic_indices` over that axis's chunk
    count, mirroring how `orthogonal_indices` reuses `basic_indices` per
    axis. `chunk_sizes` gives the per-chunk data sizes of the array's *outer*
    (block) grid for every axis — i.e. `Array.write_chunk_sizes`, the grid that
    `Array.blocks` addresses (the shard grid when sharding is used). For
    example `(3, 3, 3, 1)` for a length-10 axis with a regular chunk size of 3,
    or the explicit edges of a rectilinear axis; `nchunks` for an axis is
    `len(chunk_sizes[axis])`.

    The array-space translation uses the cumulative sum of those sizes, matching
    `BlockIndexer`'s use of `dim_grid.chunk_offset`. Because the sizes are
    clipped to the array extent, the final offset equals the extent and the
    translation is exact for regular (uniform), rectilinear, and sharded grids
    alike.

    Block indexing only supports integers and step-1 slices whose start
    references an existing chunk, so strided slices and slices starting at the
    grid edge are filtered out.

    Returns
    -------
    block_indexer
        A per-axis tuple of ints / step-1 slices addressing whole chunks,
        suitable for `Array.blocks` / `get_block_selection` / `set_block_selection`.
    array_indexer
        The equivalent array-space selection (a tuple of slices) for indexing
        the corresponding numpy array, used as the comparison oracle.
    """

    def supported(nchunks: int) -> Callable[[tuple[Any, ...]], bool]:
        # Block indexing only accepts step-1 slices whose start references an
        # existing chunk (a slice starting at nchunks raises, unlike numpy).
        def predicate(value: tuple[Any, ...]) -> bool:
            dim_sel = value[0]
            if isinstance(dim_sel, slice):
                if dim_sel.step not in (None, 1):
                    return False
                start = dim_sel.start or 0
                return 0 <= (start + nchunks if start < 0 else start) < nchunks
            return True

        return predicate

    block_indexer: list[int | slice] = []
    array_indexer: list[slice] = []
    for sizes in chunk_sizes:
        nchunks = len(sizes)
        # offsets[i] is the array-space start of chunk i; length nchunks + 1.
        offsets = list(itertools.accumulate(sizes, initial=0))
        dim_strategy = (
            basic_indices(min_dims=1, shape=(nchunks,), allow_ellipsis=False)
            # normalize bare ints / slices to a 1-tuple, skip the empty tuple
            .map(lambda x: (x,) if not isinstance(x, tuple) else x)
            .filter(bool)
            .filter(supported(nchunks))
        )
        # basic_indices draws slices far more often than bare integers, so the
        # integer (single-block) branch below would only be hit on rare draws.
        # Union in an explicit integer so it is reliably exercised — keeping
        # coverage deterministic under the derandomized `ci` Hypothesis profile.
        (dim_sel,) = draw(
            dim_strategy | st.integers(min_value=0, max_value=nchunks - 1).map(lambda i: (i,))
        )
        block_indexer.append(dim_sel)
        if isinstance(dim_sel, slice):
            start, stop, _ = dim_sel.indices(nchunks)
            array_indexer.append(slice(offsets[start], offsets[stop]))
        else:
            block = dim_sel % nchunks
            array_indexer.append(slice(offsets[block], offsets[block + 1]))
    return tuple(block_indexer), tuple(array_indexer)


@st.composite
def block_test_arrays(
    draw: st.DrawFn,
) -> tuple[Array[Any], np.ndarray[Any, Any]]:
    """Draw an array for block-indexing property tests, with its source contents.

    Two arms, selected with equal probability:

    - **regular**: a regular chunk grid, optionally wrapped in sharding.
    - **rectilinear**: a variable (rectilinear) chunk grid, always unsharded.

    Returns `(zarray, nparray)`. The per-axis block sizes the oracle needs are
    `zarray.write_chunk_sizes` — the array's *outer* (block / shard) grid, which
    is exactly the grid `Array.blocks` addresses; the caller reads it directly.
    """
    chunks: tuple[int, ...] | list[list[int]]
    if draw(st.booleans()):
        # regular arm, optionally sharded
        nparray, chunks = draw(
            np_array_and_chunks(
                arrays=numpy_arrays(shapes=npst.array_shapes(max_dims=4, min_side=1))
            )
        )
        # min_side=1 chunking guarantees shape // chunk >= 1 on every axis, which
        # shard_shapes requires.
        shards = draw(st.none() | shard_shapes(shape=nparray.shape, chunk_shape=chunks))
        event("block regular sharded" if shards is not None else "block regular unsharded")
        rectilinear = False
    else:
        # rectilinear arm, always unsharded
        event("block rectilinear")
        shape = draw(_rectilinear_shapes)
        chunks = draw(rectilinear_chunks(shape=shape))
        nparray = draw(numpy_arrays(shapes=st.just(shape), dtype=draw(dtypes())))
        shards, rectilinear = None, True

    store = draw(stores)
    with zarr.config.set({"array.rectilinear_chunks": rectilinear}):
        zarray = zarr.create_array(
            store=store,
            shape=nparray.shape,
            chunks=chunks,
            shards=shards,
            dtype=nparray.dtype,
        )
    zarray[...] = nparray
    return zarray, nparray


def key_ranges(
    keys: SearchStrategy[str] = node_names, max_size: int = sys.maxsize
) -> SearchStrategy[list[tuple[str, ByteRequest | None]]]:
    """
    Function to generate key_ranges strategy for get_partial_values()
    returns list strategy w/ form::

        [(key, byte_request),
         (key, byte_request),...]

    where `byte_request` is `None` or any of the concrete `ByteRequest`
    subtypes. The bounds are drawn independently of each value's length, so the
    offsets/suffixes routinely exceed the data and exercise the clamping logic
    in `_normalize_byte_range_index`.
    """

    def make_range(start: int, length: int) -> RangeByteRequest:
        return RangeByteRequest(start, end=min(start + length, max_size))

    bound = st.integers(min_value=0, max_value=max_size)
    byte_ranges: SearchStrategy[ByteRequest | None] = st.one_of(
        st.none(),
        st.builds(make_range, start=bound, length=bound),
        st.builds(OffsetByteRequest, offset=bound),
        st.builds(SuffixByteRequest, suffix=bound),
    )
    key_tuple = st.tuples(keys, byte_ranges)
    return st.lists(key_tuple, min_size=1, max_size=10)


@st.composite
def complex_rectilinear_arrays(
    draw: st.DrawFn,
    *,
    stores: st.SearchStrategy[StoreLike] = stores,
    paths: st.SearchStrategy[str] = paths(),  # noqa: B008
    array_names: st.SearchStrategy = array_names,
    attrs: st.SearchStrategy = attrs,
) -> tuple[npt.NDArray[Any], AnyArray]:
    """Generate a rectilinear array with many small chunks.

    The shape is derived from the chunk edges (5-10 chunks per dim,
    sizes 1-5), exercising higher chunk counts than `rectilinear_arrays`.
    """
    ndim = draw(st.integers(min_value=1, max_value=3))
    nchunks = draw(st.integers(min_value=5, max_value=10))
    dim_chunks = st.lists(st.integers(min_value=1, max_value=5), min_size=nchunks, max_size=nchunks)
    chunk_shapes = draw(st.lists(dim_chunks, min_size=ndim, max_size=ndim))

    shape = tuple(sum(dim) for dim in chunk_shapes)
    nparray = draw(numpy_arrays(shapes=st.just(shape)))
    dim_names = draw(dimension_names(ndim=ndim))
    fill_value = draw(st.one_of([st.none(), npst.from_dtype(nparray.dtype)]))
    attributes = draw(attrs)

    store = draw(stores, label="store")
    path = draw(paths, label="array parent")
    name = draw(array_names, label="array name")
    array_path = _join_paths([path, name])

    root = zarr.open_group(store, mode="w", zarr_format=3)
    with zarr.config.set({"array.rectilinear_chunks": True}):
        a = root.create_array(
            array_path,
            shape=shape,
            chunks=chunk_shapes,
            dtype=nparray.dtype,
            fill_value=fill_value,
            dimension_names=dim_names,
            attributes=attributes,
        )
    a[:] = nparray
    return nparray, a


@st.composite
def chunk_paths(draw: st.DrawFn, ndim: int, numblocks: tuple[int, ...], subset: bool = True) -> str:
    blockidx = draw(
        st.tuples(*tuple(st.integers(min_value=0, max_value=max(0, b - 1)) for b in numblocks))
    )
    subset_slicer = slice(draw(st.integers(min_value=0, max_value=ndim))) if subset else slice(None)
    return "/".join(map(str, blockidx[subset_slicer]))
