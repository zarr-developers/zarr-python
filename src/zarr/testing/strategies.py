import dataclasses
import itertools
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
import numpy.typing as npt
from hypothesis import event
from hypothesis.strategies import SearchStrategy

import zarr
from zarr.abc.codec import Codec
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
from zarr.core.dtype import data_type_registry, get_data_type_from_native_dtype
from zarr.core.dtype.common import HasItemSize
from zarr.core.dtype.npy.common import DATETIME_UNIT
from zarr.core.dtype.npy.structured import Struct
from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType
from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata
from zarr.core.metadata.v3 import (
    RectilinearChunkGridMetadata,
    RectilinearDimSpecJSON,
    RegularChunkGridMetadata,
)
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


_field_names = st.text(
    alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=4
)
_field_titles = st.text(
    alphabet=st.characters(min_codepoint=65, max_codepoint=90), min_size=1, max_size=4
)


def _leaf_zdtypes(cls: type[ZDType[TBaseDType, TBaseScalar]]) -> SearchStrategy[ZDType[Any, Any]]:
    """
    A strategy for instances of a single non-struct `ZDType` class, drawing each constructor
    parameter the class declares from its valid range.
    """
    params = {f.name for f in dataclasses.fields(cls)}
    kwargs: dict[str, SearchStrategy[Any]] = {}
    if "endianness" in params:
        kwargs["endianness"] = st.sampled_from(["little", "big"])
    if "length" in params:
        kwargs["length"] = st.integers(min_value=1, max_value=16)
    if "unit" in params:
        # The constructor normalizes the microsecond alias; all units accept a scale.
        kwargs["unit"] = st.sampled_from(DATETIME_UNIT)
        kwargs["scale_factor"] = st.integers(min_value=1, max_value=2**31 - 1)
    return st.builds(cls, **kwargs)


def _struct_zdtypes(
    children: SearchStrategy[ZDType[Any, Any]],
) -> SearchStrategy[ZDType[Any, Any]]:
    """A strategy for `Struct` instances whose field data types are drawn from `children`."""

    @st.composite
    def _draw(draw: st.DrawFn) -> ZDType[Any, Any]:
        num_fields = draw(st.integers(min_value=1, max_value=4))
        # suffix with the index so that names are unique without filtering
        names = [f"{draw(_field_names)}{i}" for i in range(num_fields)]
        return Struct(fields=tuple((name, draw(children)) for name in names))

    return _draw()


def zdtypes(*, max_leaves: int = 6) -> SearchStrategy[ZDType[Any, Any]]:
    """
    Generate instances of the built-in registered `ZDType` classes, including nested `Struct`.

    Struct fields are restricted to fixed-size data types, as required by the V3 `struct`
    extension. This strategy samples bounded lengths and normalized datetime units/scales;
    it does not cover every valid instance or arbitrary third-party dtype constructors.
    """
    leaf_classes = [cls for cls in data_type_registry.contents.values() if cls is not Struct]
    leaves = st.one_of([_leaf_zdtypes(cls) for cls in leaf_classes])
    fixed_size_leaves = st.one_of(
        [_leaf_zdtypes(cls) for cls in leaf_classes if issubclass(cls, HasItemSize)]
    )
    structs = st.recursive(fixed_size_leaves, _struct_zdtypes, max_leaves=max_leaves).filter(
        lambda dt: isinstance(dt, Struct)
    )
    return leaves | structs


@st.composite
def structured_dtypes(
    draw: st.DrawFn, *, allow_extended: bool = False, max_depth: int = 3
) -> np.dtype[np.void]:
    """
    A strategy for native NumPy structured dtypes, flat or nested.

    With `allow_extended=False` (the default), generate packed fields without titles or
    subarray shapes. With `allow_extended=True`, also generate field titles, subarray fields,
    and `align=True` layouts, independently. The current native dtype conversion rejects
    titles and subarray fields and accepts padding with a warning. These are implementation
    behaviors, not restrictions imposed by the V2 format.
    """
    fixed_size_leaves = st.one_of(
        [
            _leaf_zdtypes(cls)
            for cls in data_type_registry.contents.values()
            if cls is not Struct and issubclass(cls, HasItemSize)
        ]
    )

    def build(depth: int) -> np.dtype[np.void]:
        num_fields = draw(st.integers(min_value=1, max_value=4))
        # suffix with the index so that names and titles are unique without filtering; titles
        # draw from a different alphabet so they never collide with names either
        names = [f"{draw(_field_names)}{i}" for i in range(num_fields)]
        titles = [f"{draw(_field_titles)}{i}" for i in range(num_fields)]
        specs: list[tuple[Any, Any]] = []
        for name, title in zip(names, titles, strict=True):
            field_dtype: Any
            if depth < max_depth and draw(st.booleans()):
                field_dtype = build(depth + 1)
            else:
                field_dtype = draw(fixed_size_leaves).to_native_dtype()
            key: Any = name
            if allow_extended and draw(st.booleans()):
                key = (title, name)
            if allow_extended and draw(st.booleans()):
                field_dtype = (field_dtype, draw(npst.array_shapes(max_dims=2, max_side=3)))
            specs.append((key, field_dtype))
        align = allow_extended and draw(st.booleans())
        return np.dtype(specs, align=align)

    return build(0)


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
def dimension_names(draw: st.DrawFn, *, ndim: int | None = None) -> list[str | None] | None:
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
def _sharding_codecs(
    draw: st.DrawFn,
    *,
    chunk_shape: tuple[int, ...],
    codecs: Sequence[Codec] | None = None,
) -> ShardingCodec:
    """A ``ShardingCodec`` over ``chunk_shape`` with a drawn subchunk write order.

    The inner codec chain is drawn from ``sharding_inner_codecs`` unless ``codecs``
    is given, which lets a caller nest another ``ShardingCodec`` inside.
    """
    subchunk_write_order = draw(subchunk_write_orders)
    inner_codecs: Sequence[Codec] = (
        draw(sharding_inner_codecs, label="sharding inner codecs") if codecs is None else codecs
    )
    return ShardingCodec(
        subchunk_write_order=subchunk_write_order,
        codecs=inner_codecs,
        index_codecs=[BytesCodec(), Crc32cCodec()],
        chunk_shape=chunk_shape,
    )


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
    dim_names: list[str | None] | None = None
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

    chunks_param: tuple[int, ...] | list[int | list[int]] | RectilinearChunkGridMetadata
    shard_shape = None
    dim_names = None
    if zarr_format == 3:
        chunk_grid_meta = draw(st.none() | chunk_grids(shape=nparray.shape), label="chunk grid")
        dim_names = draw(dimension_names(ndim=nparray.ndim), label="dimension names")
        if isinstance(chunk_grid_meta, RectilinearChunkGridMetadata):
            # A rectilinear grid is passed either as the metadata object or in
            # the list form of `chunks=`, drawn from its own strategy. A 0-d
            # array has no dimension to hold an edge list.
            if nparray.ndim > 0 and draw(st.booleans(), label="chunks as lists"):
                event("rectilinear chunks= as lists")
                chunks_param = draw(rectilinear_chunks(shape=nparray.shape), label="chunks")
                chunk_grid_meta = RectilinearChunkGridMetadata(
                    chunk_shapes=tuple(
                        dim if isinstance(dim, int) else tuple(dim) for dim in chunks_param
                    )
                )
            else:
                event("rectilinear chunks= as metadata")
                chunks_param = chunk_grid_meta
        elif isinstance(chunk_grid_meta, RegularChunkGridMetadata):
            chunks_param = chunk_grid_meta.chunk_shape
        else:
            chunks_param = draw(chunk_shapes(shape=nparray.shape), label="chunk shape")

            # Any chunk that fits the array can be sharded: shard_shapes draws a
            # whole number of chunks per axis, one inner chunk included.
            if all(s >= c >= 1 for s, c in zip(nparray.shape, chunks_param, strict=True)):
                shard_shape = draw(
                    st.none() | shard_shapes(shape=nparray.shape, chunk_shape=chunks_param),
                    label="shard shape",
                )
                event("sharded" if shard_shape is not None else "unsharded")
                if shard_shape is not None:
                    serializer = draw(_sharding_codecs(chunk_shape=chunks_param))
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
            # The stored grid is exactly the declared one: bare ints stay bare
            # ints, edge lists keep their edges.
            assert a.metadata.chunk_grid == chunk_grid_meta
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


def chunks_param_from_rectilinear(
    meta: RectilinearChunkGridMetadata,
) -> list[int | list[int]]:
    """Convert rectilinear chunk grid metadata into a `chunks=` argument.

    Explicit edge tuples become lists. Bare ints — the spec's step-size
    shorthand meaning "repeat to cover the axis" — pass through unchanged;
    wrapping one in a single-element list would instead declare exactly one
    chunk, which fails normalization whenever the axis needs more than one.
    """
    return [list(dim) if isinstance(dim, tuple) else dim for dim in meta.chunk_shapes]


@st.composite
def rectilinear_dim_edges(draw: st.DrawFn, *, extent: int) -> list[int]:
    """Explicit chunk edge lengths summing exactly to `extent`.

    A zero `extent` has no chunks for edges to cover, and no non-empty list
    of positive edges sums to 0; its edges are the chunks the axis grows into
    on `append` or `resize`, so any non-empty list of positive edges is drawn.

    Two modes: "uneven" cuts the extent at random dividers; "uniform" repeats
    one size with an optional remainder, optionally shuffled so equal edges
    are not all adjacent. At most 20 chunks per dimension keeps property
    tests fast.
    """
    assert extent >= 0
    if extent == 0:
        event("rectilinear edges: zero extent")
        return draw(st.lists(st.integers(min_value=1, max_value=10), min_size=1, max_size=5))
    if extent == 1:
        return [1]
    if draw(st.booleans(), label="uneven edges"):
        nchunks = draw(st.integers(min_value=1, max_value=min(extent, 20)))
        # Draw distinct dividers by index into the unused positions: no
        # rejection, unlike `st.lists(..., unique=True)`.
        positions = list(range(1, extent))
        dividers = sorted(
            positions.pop(draw(st.integers(min_value=0, max_value=len(positions) - 1)))
            for _ in range(nchunks - 1)
        )
        return [b - a for a, b in zip([0, *dividers], [*dividers, extent], strict=True)]
    size = draw(st.integers(min_value=math.ceil(extent / 20), max_value=extent))
    edges = [size] * (extent // size)
    if extent % size:
        edges.append(extent % size)
    if draw(st.booleans(), label="shuffle uniform edges"):
        return list(draw(st.permutations(edges)))
    return edges


def _rectilinear_step(draw: st.DrawFn, *, extent: int) -> int:
    """A bare-int chunk size for one dimension: a step that repeats to cover
    the extent, with the last chunk possibly smaller. A step larger than the
    extent (one overhanging chunk) is allowed, as for a regular grid."""
    step = draw(st.integers(min_value=max(1, math.ceil(extent / 20)), max_value=extent + 3))
    if step > extent:
        event("rectilinear step: larger than extent")
    return step


@st.composite
def rectilinear_chunks(draw: st.DrawFn, *, shape: tuple[int, ...]) -> list[int | list[int]]:
    """A `chunks=` specification declaring a rectilinear grid over `shape`.

    Each dimension is either a bare int (a step size; the last chunk may be
    smaller) or an explicit edge list summing to the extent (any edges, for a
    zero extent). At least one dimension is an edge list, since bare ints
    alone declare a regular grid.
    Run-length encoding is not part of the `chunks=` syntax; it belongs to
    stored metadata, see `rectilinear_chunk_shape_declarations`.

    `shape` must have at least one dimension: a 0-d array has no dimension
    to hold an edge list, so it cannot have a rectilinear grid.
    """
    assert shape, "a rectilinear grid needs at least one dimension"
    forced_list = draw(st.integers(min_value=0, max_value=len(shape) - 1))
    chunks: list[int | list[int]] = []
    for i, extent in enumerate(shape):
        if i != forced_list and draw(st.booleans(), label="bare int"):
            chunks.append(_rectilinear_step(draw, extent=extent))
        else:
            chunks.append(draw(rectilinear_dim_edges(extent=extent)))
    return chunks


def _rle_encode(draw: st.DrawFn, edges: list[int]) -> list[int | list[int]]:
    """Run-length encode `edges` as the spec allows: a mix of bare ints and
    `[size, count]` pairs. Either the canonical form (each run as one pair,
    a run of one as a bare int) or an arbitrary grouping, which may split a
    run across pairs and use `count == 1`."""
    canonical = draw(st.booleans(), label="canonical rle")
    if not canonical:
        event("rectilinear rle: arbitrary grouping")
    encoded: list[int | list[int]] = []
    i = 0
    while i < len(edges):
        run = 1
        while i + run < len(edges) and edges[i + run] == edges[i]:
            run += 1
        if canonical:
            count = run
            bare = run == 1
        else:
            count = draw(st.integers(min_value=1, max_value=run))
            bare = count == 1 and draw(st.booleans(), label="bare edge")
        encoded.append(edges[i] if bare else [edges[i], count])
        i += count
    return encoded


def _rectilinear_chunk_shape(draw: st.DrawFn, *, extent: int) -> int | tuple[int, ...]:
    """One dimension of a stored rectilinear grid's `chunk_shapes`: a bare-int
    step, or explicit edges. Edges may sum beyond the extent, which the spec
    allows and a shrinking resize produces."""
    if draw(st.booleans(), label="bare int"):
        return _rectilinear_step(draw, extent=extent)
    edges = draw(rectilinear_dim_edges(extent=extent))
    if extent > 0 and draw(st.booleans(), label="overhang"):
        event("rectilinear edges: overhang")
        if draw(st.booleans(), label="trailing edge"):
            edges = [*edges, draw(st.integers(min_value=1, max_value=5))]
        else:
            edges[-1] += draw(st.integers(min_value=1, max_value=5))
    return tuple(edges)


@st.composite
def rectilinear_chunk_shape_declarations(
    draw: st.DrawFn, *, shape: tuple[int, ...]
) -> tuple[list[RectilinearDimSpecJSON], tuple[int | tuple[int, ...], ...]]:
    """The `chunk_shapes` of a stored rectilinear chunk grid, with its meaning.

    Samples the whole declaration space of the spec. Per dimension: a bare
    int step, or an edge list written in full or run-length encoded
    (canonically, or with arbitrary grouping). Edge lists may sum beyond the
    extent.

    Returns `(declaration, chunk_shapes)`: the JSON value to store, and the
    `chunk_shapes` that parsing it must produce.
    """
    chunk_shapes = tuple(_rectilinear_chunk_shape(draw, extent=extent) for extent in shape)
    declaration: list[RectilinearDimSpecJSON] = [
        dim
        if isinstance(dim, int)
        else list(dim)
        if draw(st.booleans(), label="write edges in full")
        else _rle_encode(draw, list(dim))
        for dim in chunk_shapes
    ]
    return declaration, chunk_shapes


@st.composite
def rectilinear_chunk_grids(
    draw: st.DrawFn, *, shape: tuple[int, ...]
) -> RectilinearChunkGridMetadata:
    """A `RectilinearChunkGridMetadata` over `shape`, per dimension a bare-int
    step or explicit edges, which may sum beyond the extent."""
    return RectilinearChunkGridMetadata(
        chunk_shapes=tuple(_rectilinear_chunk_shape(draw, extent=extent) for extent in shape)
    )


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
    if zarr.config.get("array.rectilinear_chunks") and draw(st.booleans()):
        event("using RectilinearChunkGridMetadata")
        return draw(rectilinear_chunk_grids(shape=shape))
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


# Sharded arrays need min_side >= 1: a shard must hold at least one chunk on every axis.
_sharded_shapes = npst.array_shapes(max_dims=4, min_side=1, max_side=8)


@st.composite
def sharded_arrays(
    draw: st.DrawFn,
    *,
    shapes: st.SearchStrategy[tuple[int, ...]] = _sharded_shapes,
    nested: bool | None = None,
) -> Any:
    """Generate a zarr v3 array whose chunks are grouped into shards.

    ``arrays`` shards only a small fraction of its draws (a v3 array with a
    regular chunk grid, every axis larger than a chunk that is itself larger
    than 1, and then only half the time), so a property test that must
    exercise the sharding codec should draw from this strategy directly. Every
    draw is sharded: the chunk shape and the shard shape (an integral number of
    chunks per axis, possibly a single chunk) are drawn from ``shapes``, and
    the codec's subchunk write order and inner codec chain are drawn as in
    ``arrays``. ``shapes`` must generate shapes with at least one element on
    every axis.

    ``nested`` selects one level of recursive sharding: the drawn chunks are
    grouped into inner shards, which are themselves grouped into the shards
    stored in the array, so the outer ``ShardingCodec`` wraps an inner one with
    its own subchunk write order. ``None`` (the default) draws it, so half the
    examples nest. For a nested array ``Array.chunks`` is the inner shard shape
    (the outer codec's chunk shape); the innermost chunk shape is the inner
    codec's ``chunk_shape``.
    """
    shape = draw(shapes)
    chunk_shape = draw(chunk_shapes(shape=shape), label="chunk shape")
    serializer = draw(_sharding_codecs(chunk_shape=chunk_shape))
    nest = draw(st.booleans(), label="nested sharding") if nested is None else nested
    if nest:
        # Each level's shard is an integral number of the level below's chunks.
        codec_chunk_shape = draw(
            shard_shapes(shape=shape, chunk_shape=chunk_shape), label="inner shard shape"
        )
        serializer = draw(_sharding_codecs(chunk_shape=codec_chunk_shape, codecs=[serializer]))
    else:
        codec_chunk_shape = chunk_shape
    shard_shape = draw(
        shard_shapes(shape=shape, chunk_shape=codec_chunk_shape), label="shard shape"
    )
    event("nested sharding" if nest else "single-level sharding")

    nparray = draw(numpy_arrays(shapes=st.just(shape)), label="array data")
    fill_value = draw(st.one_of([st.none(), npst.from_dtype(nparray.dtype)]))
    dim_names = draw(dimension_names(ndim=len(shape)), label="dimension names")

    # The shard is the array's chunk grid and the drawn codec is its serializer.
    # Passing ``shards=`` instead would make ``create_array`` wrap the codec in a
    # second ``ShardingCodec`` of the same chunk shape, hiding the drawn write
    # order behind a default outer one.
    a = zarr.create_array(
        store=MemoryStore(),
        shape=shape,
        chunks=shard_shape,
        dtype=nparray.dtype,
        fill_value=fill_value,
        dimension_names=dim_names,
        serializer=serializer,
        filters=None,
        compressors=None,
    )
    assert a.shards == shard_shape
    assert a.chunks == codec_chunk_shape
    assert isinstance(a.metadata, ArrayV3Metadata)
    (codec,) = a.metadata.codecs
    assert isinstance(codec, ShardingCodec)
    assert codec.subchunk_write_order == serializer.subchunk_write_order
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
) -> tuple[tuple[int | slice | np.ndarray[Any, Any], ...], tuple[np.ndarray[Any, Any], ...]]:
    """
    Strategy that returns
    (1) a tuple of per-axis selectors (integer array, slice, or bare integer) for
        orthogonal indexing of Zarr arrays.
    (2) a tuple of broadcast integer arrays that index a numpy array to the same
        result. A bare integer drops its axis, as ``oindex`` does, so it is
        given as a 0-d array and does not contribute a result dimension.
    """
    zindexer: list[int | slice | np.ndarray[Any, Any]] = []
    kept: list[tuple[int, np.ndarray[Any, Any]]] = []
    npindexer: dict[int, np.ndarray[Any, Any]] = {}
    for axis, size in enumerate(shape):
        if size != 0:
            strategy = (
                npst.integer_array_indices(
                    shape=(size,),
                    result_shape=npst.array_shapes(min_side=1, max_side=size, max_dims=1),
                )
                | basic_indices(min_dims=1, shape=(size,), allow_ellipsis=False)
                # basic_indices(min_dims=1) never yields a bare integer, so draw
                # one explicitly: it is the only selector that drops an axis.
                | st.integers(min_value=-size, max_value=size - 1)
            )
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
        zindexer.append(idxr)
        if isinstance(idxr, int):
            npindexer[axis] = np.array(idxr)
            continue
        if isinstance(idxr, slice):
            idxr = np.arange(*idxr.indices(size))
        elif isinstance(idxr, tuple):
            idxr = np.array(idxr)
        kept.append((axis, idxr))

    for pos, (axis, idxr) in enumerate(kept):
        newshape = [1] * len(kept)
        newshape[pos] = idxr.size
        npindexer[axis] = idxr.reshape(newshape)

    # casting the output of broadcast_arrays is needed for numpy < 2
    return tuple(zindexer), tuple(
        np.broadcast_arrays(*(npindexer[axis] for axis in range(len(shape))))
    )


@st.composite
def block_indices(
    draw: st.DrawFn, *, chunk_sizes: tuple[tuple[int, ...], ...]
) -> tuple[tuple[int | slice, ...], tuple[slice, ...]]:
    """
    Strategy for block-selection indexers over a chunk grid.

    Block indexing is basic indexing applied to the block grid (the grid of
    chunks), so each axis is drawn with ``basic_indices`` over that axis's chunk
    count, mirroring how ``orthogonal_indices`` reuses ``basic_indices`` per
    axis. ``chunk_sizes`` gives the per-chunk data sizes of the array's *outer*
    (block) grid for every axis — i.e. ``Array.write_chunk_sizes``, the grid that
    ``Array.blocks`` addresses (the shard grid when sharding is used). For
    example ``(3, 3, 3, 1)`` for a length-10 axis with a regular chunk size of 3,
    or the explicit edges of a rectilinear axis; ``nchunks`` for an axis is
    ``len(chunk_sizes[axis])``.

    The array-space translation uses the cumulative sum of those sizes, matching
    ``BlockIndexer``'s use of ``dim_grid.chunk_offset``. Because the sizes are
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
        suitable for ``Array.blocks`` / ``get_block_selection`` / ``set_block_selection``.
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
        # coverage deterministic under the derandomized ``ci`` Hypothesis profile.
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

    Returns ``(zarray, nparray)``. The per-axis block sizes the oracle needs are
    ``zarray.write_chunk_sizes`` — the array's *outer* (block / shard) grid, which
    is exactly the grid ``Array.blocks`` addresses; the caller reads it directly.
    """
    chunks: tuple[int, ...] | list[int | list[int]]
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

    where ``byte_request`` is ``None`` or any of the concrete ``ByteRequest``
    subtypes. The bounds are drawn independently of each value's length, so the
    offsets/suffixes routinely exceed the data and exercise the clamping logic
    in ``_normalize_byte_range_index``.
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
    sizes 1-5), exercising higher chunk counts than ``rectilinear_arrays``.
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
