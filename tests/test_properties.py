import itertools
import json
import numbers
import warnings
from collections.abc import Generator
from typing import Any

import numpy as np
import pytest
from numpy.lib.recfunctions import repack_fields
from numpy.testing import assert_array_equal

import zarr
from zarr.core.buffer import default_buffer_prototype

pytest.importorskip("hypothesis")

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
from hypothesis import assume, event, given, settings

from tests.conftest import declared_chunk_data_sizes
from zarr.abc.store import Store
from zarr.core.common import ZARR_JSON, ZARRAY_JSON, ZATTRS_JSON
from zarr.core.dtype import get_data_type_from_json, get_data_type_from_native_dtype
from zarr.core.dtype.common import HasItemSize
from zarr.core.dtype.npy.structured import Struct
from zarr.core.dtype.wrapper import ZDType
from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata
from zarr.core.metadata.v3 import RectilinearChunkGridMetadata, RectilinearChunkGridMetadataJSON
from zarr.core.sync import sync
from zarr.errors import ZarrUserWarning
from zarr.storage import MemoryStore
from zarr.testing.strategies import (
    _rectilinear_chunks,
    array_metadata,
    arrays,
    basic_indices,
    block_indices,
    block_test_arrays,
    complex_rectilinear_arrays,
    numpy_arrays,
    orthogonal_indices,
    rectilinear_arrays,
    rectilinear_chunk_shape_declarations,
    rectilinear_chunks,
    sharded_arrays,
    simple_arrays,
    stores,
    structured_dtypes,
    zarr_formats,
    zdtypes,
)


@pytest.fixture(autouse=True)
def _enable_rectilinear_chunks() -> Generator[None, None, None]:
    """Enable rectilinear chunks for all property tests since strategies may generate them."""
    with zarr.config.set({"array.rectilinear_chunks": True}):
        yield


def deep_equal(a: Any, b: Any) -> bool:
    """Deep equality check with handling of special cases for array metadata classes"""
    if isinstance(a, (complex, np.complexfloating)) and isinstance(
        b, (complex, np.complexfloating)
    ):
        a_real, a_imag = float(a.real), float(a.imag)
        b_real, b_imag = float(b.real), float(b.imag)
        if np.isnan(a_real) and np.isnan(b_real):
            real_eq = True
        else:
            real_eq = a_real == b_real
        if np.isnan(a_imag) and np.isnan(b_imag):
            imag_eq = True
        else:
            imag_eq = a_imag == b_imag
        return real_eq and imag_eq

    if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
        if np.isnan(a) and np.isnan(b):
            return True
        return a == b

    if isinstance(a, np.datetime64) and isinstance(b, np.datetime64):
        if np.isnat(a) and np.isnat(b):
            return True
        return a == b

    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.shape != b.shape:
            return False
        return all(itertools.starmap(deep_equal, zip(a.flat, b.flat, strict=False)))

    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(deep_equal(a[k], b[k]) for k in a)

    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(itertools.starmap(deep_equal, zip(a, b, strict=False)))

    return a == b


@pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
@given(data=st.data())
def test_array_roundtrip(data: st.DataObject) -> None:
    nparray = data.draw(numpy_arrays())
    zarray = data.draw(arrays(arrays=st.just(nparray)))
    assert_array_equal(nparray, zarray[:])


@pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
@given(array=arrays())
def test_array_creates_implicit_groups(array):
    path = array.path
    ancestry = path.split("/")[:-1]
    for i in range(len(ancestry)):
        parent = "/".join(ancestry[: i + 1])
        if array.metadata.zarr_format == 2:
            assert (
                sync(array.store.get(f"{parent}/.zgroup", prototype=default_buffer_prototype()))
                is not None
            )
        elif array.metadata.zarr_format == 3:
            assert (
                sync(array.store.get(f"{parent}/zarr.json", prototype=default_buffer_prototype()))
                is not None
            )


# this decorator removes timeout; not ideal but it should avoid intermittent CI failures


@settings(deadline=None)
@pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
@given(data=st.data())
async def test_basic_indexing(data: st.DataObject) -> None:
    zarray = data.draw(st.one_of(simple_arrays(), rectilinear_arrays()))
    nparray = zarray[:]
    indexer = data.draw(basic_indices(shape=nparray.shape))

    # sync get
    actual = zarray[indexer]
    assert_array_equal(nparray[indexer], actual)

    # async get
    async_zarray = zarray._async_array
    actual = await async_zarray.getitem(indexer)
    assert_array_equal(nparray[indexer], actual)

    # sync set
    new_data = data.draw(numpy_arrays(shapes=st.just(actual.shape), dtype=nparray.dtype))
    zarray[indexer] = new_data
    nparray[indexer] = new_data
    assert_array_equal(nparray, zarray[:])

    # TODO test async setitem?


@settings(deadline=None)
@pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
@given(data=st.data())
async def test_basic_indexing_complex_rectilinear(data: st.DataObject) -> None:
    nparray, zarray = data.draw(complex_rectilinear_arrays())
    indexer = data.draw(basic_indices(shape=nparray.shape))
    assert_array_equal(nparray[indexer], zarray[indexer])


@given(data=st.data())
@pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
async def test_oindex(data: st.DataObject) -> None:
    # integer_array_indices can't handle 0-size dimensions.
    # A sharded array is drawn as its own arm: simple_arrays shards only a few
    # percent of its draws, and the sharding codec's write path for a selection
    # with two or more array-indexed axes (GH4284) needs real weight here. That
    # path only exists for a value with two or more axes, hence min_dims=2.
    zarray = data.draw(
        st.one_of(
            simple_arrays(shapes=npst.array_shapes(max_dims=4, min_side=1)),
            rectilinear_arrays(shapes=npst.array_shapes(max_dims=4, min_side=1, max_side=20)),
            sharded_arrays(
                shapes=npst.array_shapes(min_dims=2, max_dims=4, min_side=1, max_side=8)
            ),
        )
    )
    nparray = zarray[:]
    zindexer, npindexer = data.draw(orthogonal_indices(shape=nparray.shape))

    # sync get
    actual = zarray.oindex[zindexer]
    assert_array_equal(nparray[npindexer], actual)

    # async get
    async_zarray = zarray._async_array
    actual = await async_zarray.oindex.getitem(zindexer)
    assert_array_equal(nparray[npindexer], actual)

    # sync set
    for idxr, size in zip(zindexer, nparray.shape, strict=True):
        if isinstance(idxr, np.ndarray) and idxr.size != np.unique(idxr % size).size:
            # behaviour of setitem with repeated indices is not guaranteed in practice
            # Negative and positive spellings of the same index are duplicates too.
            assume(False)
    # The sharding codec sees a coordinate selection (the GH4284 path) when the
    # chunk selection has more than one array axis or drops an integer axis.
    n_array_axes = sum(isinstance(idxr, np.ndarray) for idxr in zindexer)
    coordinate_path = n_array_axes > 1 or any(isinstance(idxr, int) for idxr in zindexer)
    event(
        f"oindex write: {'sharded' if zarray.shards is not None else 'unsharded'}, "
        f"{'coordinate' if coordinate_path else 'orthogonal'} chunk selection"
    )
    new_data = data.draw(numpy_arrays(shapes=st.just(actual.shape), dtype=nparray.dtype))
    nparray[npindexer] = new_data
    zarray.oindex[zindexer] = new_data
    assert_array_equal(nparray, zarray[:])

    # note: async oindex setitem not yet implemented


@given(data=st.data())
@pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
async def test_vindex(data: st.DataObject) -> None:
    # integer_array_indices can't handle 0-size dimensions.
    zarray = data.draw(
        st.one_of(
            simple_arrays(shapes=npst.array_shapes(max_dims=4, min_side=1)),
            rectilinear_arrays(shapes=npst.array_shapes(max_dims=3, min_side=1, max_side=20)),
            sharded_arrays(),
        )
    )
    nparray = zarray[:]
    indexer = data.draw(
        npst.integer_array_indices(
            shape=nparray.shape, result_shape=npst.array_shapes(min_side=1, max_dims=None)
        )
    )

    # sync get
    actual = zarray.vindex[indexer]
    assert_array_equal(nparray[indexer], actual)

    # async get
    async_zarray = zarray._async_array
    actual = await async_zarray.vindex.getitem(indexer)
    assert_array_equal(nparray[indexer], actual)

    # sync set
    # Reads preserve the supplied indices; normalize negative indices explicitly when
    # detecting repeated points rather than relying on a read to mutate the indexer.
    points = np.stack(
        [
            (idxr % size).ravel()
            for idxr, size in zip(np.broadcast_arrays(*indexer), nparray.shape, strict=True)
        ],
        axis=-1,
    )
    if len(np.unique(points, axis=0)) != len(points):
        # behaviour of setitem with repeated coordinates is not guaranteed in practice
        assume(False)
    new_data = data.draw(numpy_arrays(shapes=st.just(actual.shape), dtype=nparray.dtype))
    nparray[indexer] = new_data
    zarray.vindex[indexer] = new_data
    assert_array_equal(nparray, zarray[:])

    # note: async vindex setitem not yet implemented


@settings(deadline=None)
@pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
@given(data=st.data())
def test_mask_indexing(data: st.DataObject) -> None:
    zarray = data.draw(st.one_of(simple_arrays(), rectilinear_arrays()))
    nparray = zarray[:]
    mask = data.draw(npst.arrays(dtype=np.bool_, shape=st.just(nparray.shape)))

    expected = nparray[mask]

    # sync get, via both the dedicated method and the vindex interface
    assert_array_equal(expected, zarray.get_mask_selection(mask))
    assert_array_equal(expected, zarray.vindex[mask])

    # sync set, via both interfaces
    new_data = data.draw(numpy_arrays(shapes=st.just(expected.shape), dtype=nparray.dtype))
    nparray[mask] = new_data
    zarray.set_mask_selection(mask, new_data)
    assert_array_equal(nparray, zarray[:])

    zarray.vindex[mask] = new_data
    assert_array_equal(nparray, zarray[:])


@settings(deadline=None)
@pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
@given(data=st.data())
def test_block_indexing(data: st.DataObject) -> None:
    # Block indexing addresses whole inner chunks. block_indices() builds its
    # array-space oracle from cumulative chunk offsets, so it works for regular
    # (uniform), rectilinear, and sharded grids alike; block_test_arrays draws
    # across that matrix (rectilinear + sharded is unsupported and not drawn).
    zarray, nparray = data.draw(block_test_arrays())

    # The block grid is worked out from the stored declaration, not by zarr's grid code.
    assert isinstance(zarray.metadata, ArrayV3Metadata)
    grid = zarray.metadata.chunk_grid
    declared = (
        grid.chunk_shapes if isinstance(grid, RectilinearChunkGridMetadata) else grid.chunk_shape
    )
    chunk_sizes = tuple(
        declared_chunk_data_sizes(d, n) for d, n in zip(declared, zarray.shape, strict=True)
    )
    assert zarray.write_chunk_sizes == chunk_sizes

    block_indexer, array_indexer = data.draw(block_indices(chunk_sizes=chunk_sizes))
    expected = nparray[array_indexer]

    # sync get, via both the .blocks interface and the dedicated method
    assert_array_equal(expected, zarray.blocks[block_indexer])
    assert_array_equal(expected, zarray.get_block_selection(block_indexer))

    # sync set, via both interfaces
    new_data = data.draw(numpy_arrays(shapes=st.just(expected.shape), dtype=nparray.dtype))
    nparray[array_indexer] = new_data
    zarray.blocks[block_indexer] = new_data
    assert_array_equal(nparray, zarray[:])

    zarray.set_block_selection(block_indexer, new_data)
    assert_array_equal(nparray, zarray[:])


@given(store=stores, meta=array_metadata())  # type: ignore[misc]
@pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
async def test_roundtrip_array_metadata_from_store(
    store: Store, meta: ArrayV2Metadata | ArrayV3Metadata
) -> None:
    """
    Verify that the I/O for metadata in a store are lossless.

    This test serializes an ArrayV2Metadata or ArrayV3Metadata object to a dict
    of buffers via `to_buffer_dict`, writes each buffer to a store under keys
    prefixed with "0/", and then reads them back. The test asserts that each
    retrieved buffer exactly matches the original buffer.
    """
    asdict = meta.to_buffer_dict(prototype=default_buffer_prototype())
    for key, expected in asdict.items():
        await store.set(f"0/{key}", expected)
        actual = await store.get(f"0/{key}", prototype=default_buffer_prototype())
        assert actual == expected


@given(data=st.data(), zarr_format=zarr_formats)
@pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
def test_roundtrip_array_metadata_from_json(data: st.DataObject, zarr_format: int) -> None:
    """
    Verify that JSON serialization and deserialization of metadata is lossless.

    For Zarr v2:
      - The metadata is split into two JSON documents (one for array data and one
        for attributes). The test merges the attributes back before deserialization.
    For Zarr v3:
      - All metadata is stored in a single JSON document. No manual merger is necessary.

    The test then converts both the original and round-tripped metadata objects
    into dictionaries using `dataclasses.asdict` and uses a deep equality check
    to verify that the roundtrip has preserved all fields (including special
    cases like NaN, Infinity, complex numbers, and datetime values).
    """
    metadata = data.draw(array_metadata(zarr_formats=st.just(zarr_format)))
    buffer_dict = metadata.to_buffer_dict(prototype=default_buffer_prototype())

    if zarr_format == 2:
        zarray_dict = json.loads(buffer_dict[ZARRAY_JSON].to_bytes().decode())
        zattrs_dict = json.loads(buffer_dict[ZATTRS_JSON].to_bytes().decode())
        # zattrs and zarray are separate in v2, we have to add attributes back prior to `from_dict`
        zarray_dict["attributes"] = zattrs_dict
        metadata_roundtripped = ArrayV2Metadata.from_dict(zarray_dict)
    else:
        zarray_dict = json.loads(buffer_dict[ZARR_JSON].to_bytes().decode())
        metadata_roundtripped = ArrayV3Metadata.from_dict(zarray_dict)

    orig = metadata.to_dict()
    rt = metadata_roundtripped.to_dict()

    assert deep_equal(orig, rt), f"Roundtrip mismatch:\nOriginal: {orig}\nRoundtripped: {rt}"


def _struct_depth(zdtype: ZDType[Any, Any]) -> int:
    if not isinstance(zdtype, Struct):
        return 0
    return 1 + max(_struct_depth(field_dtype) for _, field_dtype in zdtype.fields)


@given(zdtype=zdtypes(), zarr_format=zarr_formats)
@pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
def test_zdtype_json_roundtrip(zdtype: ZDType[Any, Any], zarr_format: int) -> None:
    """
    Generated built-in data types, including nested structs, round-trip through their JSON
    forms for both Zarr formats within the strategy's documented parameter ranges.

    Zarr format 3 data type names do not carry endianness (the bytes codec does), so for that
    format the JSON form is compared instead of the data type instance.
    """
    event(f"dtype={type(zdtype).__name__}")
    event(f"struct_depth={_struct_depth(zdtype)}")
    as_json = zdtype.to_json(zarr_format=zarr_format)  # type: ignore[arg-type]
    roundtripped = get_data_type_from_json(as_json, zarr_format=zarr_format)
    assert roundtripped.to_json(zarr_format=zarr_format) == as_json  # type: ignore[arg-type]
    if zarr_format == 2:
        assert roundtripped == zdtype


@given(zdtype=zdtypes())
@pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
def test_zdtype_native_roundtrip(zdtype: ZDType[Any, Any]) -> None:
    """
    Generated built-in data types round-trip through their native NumPy dtypes within the
    strategy's parameter ranges, and reported item sizes match native dtype itemsizes.

    The NumPy object dtype is shared by several Zarr data types, so resolving it is ambiguous by
    design and must raise instead.
    """
    event(f"dtype={type(zdtype).__name__}")
    native = zdtype.to_native_dtype()
    if native.kind == "O":
        event("native=object")
        with pytest.raises(ValueError, match="ambiguous"):
            get_data_type_from_native_dtype(native)
        return
    roundtripped = get_data_type_from_native_dtype(native)
    assert roundtripped == zdtype
    assert roundtripped.to_native_dtype() == native
    if isinstance(zdtype, HasItemSize):
        assert zdtype.item_size == native.itemsize


def _extended_descr_features(descr: list[Any]) -> set[str]:
    """Classify NumPy's serialized field records independently of Zarr's dtype conversion."""
    features = set()
    for field in descr:
        if isinstance(field[0], tuple):
            features.add("title")
        if len(field) == 3:
            features.add("subarray")
        if isinstance(field[1], list):
            features.update(_extended_descr_features(field[1]))
    return features


@given(dtype=structured_dtypes(allow_extended=True))
@pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
def test_structured_dtype_never_silently_changes(dtype: np.dtype[np.void]) -> None:
    """
    Generated structured dtypes with titles or subarrays are rejected by the current
    conversion. Other generated dtypes preserve their fields, warning on layout changes.
    """
    unsupported_features = _extended_descr_features(dtype.descr)
    if unsupported_features:
        with pytest.raises(ValueError, match="|".join(sorted(unsupported_features))):
            get_data_type_from_native_dtype(dtype)
        event("outcome=rejected")
        return
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ZarrUserWarning)
        zdtype = get_data_type_from_native_dtype(dtype)
    event("outcome=accepted")
    native = zdtype.to_native_dtype()
    assert native == repack_fields(dtype, recurse=True)
    layout_warnings = [
        w for w in caught if issubclass(w.category, ZarrUserWarning) and "packed" in str(w.message)
    ]
    assert bool(layout_warnings) == (native != dtype)


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


def serialized_complex_float_is_valid(
    serialized: tuple[numbers.Real | str, numbers.Real | str],
) -> bool:
    """
    Validate that the serialized representation of a complex float conforms to the spec.

    The specification requires that a serialized complex float must be either:
      - A JSON number, or
      - One of the strings "NaN", "Infinity", or "-Infinity".

    Args:
        serialized: The value produced by JSON serialization for a complex floating point number.

    Returns:
        bool: True if the serialized value is valid according to the spec, False otherwise.
    """
    return (
        isinstance(serialized, tuple)
        and len(serialized) == 2
        and all(serialized_float_is_valid(x) for x in serialized)
    )


def serialized_float_is_valid(serialized: numbers.Real | str) -> bool:
    """
    Validate that the serialized representation of a float conforms to the spec.

    The specification requires that a serialized float must be either:
      - A JSON number, or
      - One of the strings "NaN", "Infinity", or "-Infinity".

    Args:
        serialized: The value produced by JSON serialization for a floating point number.

    Returns:
        bool: True if the serialized value is valid according to the spec, False otherwise.
    """
    if isinstance(serialized, numbers.Real):
        return True
    return serialized in ("NaN", "Infinity", "-Infinity")


@given(meta=array_metadata())  # type: ignore[misc]
@pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
def test_array_metadata_meets_spec(meta: ArrayV2Metadata | ArrayV3Metadata) -> None:
    """
    Validate that the array metadata produced by the library conforms to the relevant spec (V2 vs V3).

    For ArrayV2Metadata:
      - Ensures that 'zarr_format' is 2.
      - Verifies that 'filters' is either None or a tuple (and not an empty tuple).
    For ArrayV3Metadata:
      - Ensures that 'zarr_format' is 3.

    For both versions:
      - If the dtype is a floating point of some kind, verifies of fill values:
          * NaN is serialized as the string "NaN"
          * Positive Infinity is serialized as the string "Infinity"
          * Negative Infinity is serialized as the string "-Infinity"
          * Other fill values are preserved as-is.
      - If the dtype is a complex number of some kind, verifies that each component of the fill
        value (real and imaginary) satisfies the serialization rules for floating point numbers.
      - If the dtype is a datetime of some kind, verifies that `NaT` values are serialized as "NaT".

    Note:
      This test validates spec-compliance for array metadata serialization.
      It is a work-in-progress and should be expanded as further edge cases are identified.
    """
    asdict_dict = meta.to_dict()

    # version-specific validations
    if isinstance(meta, ArrayV2Metadata):
        assert asdict_dict["filters"] != ()
        assert asdict_dict["filters"] is None or isinstance(asdict_dict["filters"], tuple)
        assert asdict_dict["zarr_format"] == 2
    else:
        assert asdict_dict["zarr_format"] == 3

    # version-agnostic validations
    dtype_native = meta.dtype.to_native_dtype()
    if dtype_native.kind == "f":
        assert serialized_float_is_valid(asdict_dict["fill_value"])
    elif dtype_native.kind == "c":
        # fill_value should be a two-element array [real, imag].
        assert serialized_complex_float_is_valid(asdict_dict["fill_value"])
    elif dtype_native.kind in ("M", "m") and np.isnat(meta.fill_value):
        assert asdict_dict["fill_value"] == -9223372036854775808


@given(data=st.data())
def test_rectilinear_chunks_declares_edges_per_dimension(data: st.DataObject) -> None:
    """`rectilinear_chunks` draws an explicit edge list for every dimension, summing
    to the extent (any positive edges, for a zero extent); a 0-d shape has none."""
    shape = data.draw(
        npst.array_shapes(min_dims=0, max_dims=3, min_side=0, max_side=20), label="shape"
    )
    chunks = data.draw(rectilinear_chunks(shape=shape), label="chunks")
    assert len(chunks) == len(shape)
    for edges, extent in zip(chunks, shape, strict=True):
        assert isinstance(edges, list)
        assert edges
        assert all(type(edge) is int and edge >= 1 for edge in edges)
        assert extent == 0 or sum(edges) == extent


def test_chunks_param_from_rectilinear_bare_int_roundtrip() -> None:
    """Bare-int dims in rectilinear metadata (the spec's step-size shorthand,
    produced by a scalar dimension of a mixed chunk spec) must pass
    through the `chunks=` conversion unchanged. Wrapping one in a
    single-element list turns "repeat to cover the axis" into "exactly one
    chunk" and re-creation fails the sum-to-span check. Edge tuples become lists,
    as zarr 3.4.0 returned them."""
    from zarr.core.metadata.v3 import RectilinearChunkGridMetadata
    from zarr.storage import MemoryStore
    from zarr.testing.strategies import chunks_param_from_rectilinear

    with zarr.config.set({"array.rectilinear_chunks": True}):
        src = zarr.create_array(MemoryStore(), shape=(3, 3), chunks=([1, 2], 1), dtype="uint8")
        grid = src.metadata.chunk_grid  # type: ignore[union-attr]
        assert isinstance(grid, RectilinearChunkGridMetadata)
        assert grid.chunk_shapes == ((1, 2), 1)
        chunks = chunks_param_from_rectilinear(grid)
        # a list of lists, not tuples (`[1, 2] != (1, 2)`)
        assert chunks == [[1, 2], 1]
        dst = zarr.create_array(
            MemoryStore(),
            shape=src.shape,
            chunks=chunks,
            dtype="uint8",
        )
        assert dst.metadata.chunk_grid == grid  # type: ignore[union-attr]


@given(data=st.data())
def test_rectilinear_chunk_grid_declarations(data: st.DataObject) -> None:
    """Every `chunk_shapes` declaration the rectilinear spec allows — bare-int
    steps, edge lists written in full or run-length encoded in any grouping,
    edges overhanging the extent — parses to its expanded edges, and the
    re-serialized form parses back to the same grid."""
    shape = data.draw(npst.array_shapes(max_dims=3, min_side=0, max_side=20), label="shape")
    declaration, chunk_shapes = data.draw(
        rectilinear_chunk_shape_declarations(shape=shape), label="declaration"
    )
    stored: RectilinearChunkGridMetadataJSON = {
        "name": "rectilinear",
        "configuration": {"kind": "inline", "chunk_shapes": declaration},
    }
    meta = RectilinearChunkGridMetadata.from_dict(stored)
    assert meta.chunk_shapes == chunk_shapes

    serialized = json.loads(json.dumps(meta.to_dict()))
    assert serialized["name"] == "rectilinear"
    assert RectilinearChunkGridMetadata.from_dict(serialized) == meta

    # The declaration is read correctly inside a whole stored metadata document.
    document = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": list(shape),
        "data_type": "uint8",
        "chunk_grid": stored,
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": 0,
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "attributes": {},
    }
    assert ArrayV3Metadata.from_dict(document).chunk_grid == meta  # type: ignore[arg-type]


def _rle_expand(dim: list[Any]) -> list[int]:
    """Expand one stored run-length encoded dimension: bare edges and
    `[size, count]` pairs."""
    edges: list[int] = []
    for item in dim:
        if type(item) is int:
            edges.append(item)
        else:
            size, count = item
            edges.extend([size] * count)
    return edges


@given(data=st.data())
def test_create_array_stores_declared_rectilinear_chunks(data: st.DataObject) -> None:
    """A `chunks=` specification mixing bare ints and edge lists in any
    arrangement is stored as a rectilinear grid whose `chunk_shapes` are
    exactly the specification: bare ints stay bare ints, and edge lists keep
    their edges (gh-4374, gh-4272). Checked on the stored JSON."""
    shape = data.draw(npst.array_shapes(max_dims=3, min_side=0, max_side=20), label="shape")
    chunks = data.draw(_rectilinear_chunks(shape=shape), label="chunks")
    arr = zarr.create_array(MemoryStore(), shape=shape, chunks=chunks, dtype="uint8")

    zarr_json = sync(arr.store.get(ZARR_JSON, prototype=default_buffer_prototype()))
    assert zarr_json is not None
    stored = json.loads(zarr_json.to_bytes())["chunk_grid"]
    assert stored["name"] == "rectilinear"
    assert stored["configuration"]["kind"] == "inline"
    stored_dims = stored["configuration"]["chunk_shapes"]
    assert [dim if type(dim) is int else _rle_expand(dim) for dim in stored_dims] == chunks


@given(data=st.data())
def test_rectilinear_zero_length_axis_round_trip(data: st.DataObject) -> None:
    """An array declared with rectilinear `chunks=` over a zero-length axis
    holds the data written after that axis grows, by `append` or by `resize`,
    also when it grows past the declared edges."""
    shape = list(data.draw(npst.array_shapes(max_dims=3, min_side=0, max_side=6), label="shape"))
    axis = data.draw(st.integers(0, len(shape) - 1), label="zero-length axis")
    shape[axis] = 0
    chunks = data.draw(_rectilinear_chunks(shape=tuple(shape)), label="chunks")
    store = MemoryStore()
    arr = zarr.create_array(store, shape=tuple(shape), chunks=chunks, dtype="int16", fill_value=-1)
    assert_array_equal(arr[...], np.full(shape, -1, dtype="int16"))

    declared = chunks[axis]
    declared_span = sum(declared) if isinstance(declared, list) else declared
    rows = data.draw(st.integers(1, 2 * declared_span + 2), label="rows")
    if isinstance(declared, list):
        event("grown axis declared as an edge list")
        if rows > declared_span:
            event("grown past the declared edges")
    grown = [*shape]
    grown[axis] = rows
    values = data.draw(npst.arrays(np.dtype("int16"), tuple(grown)), label="values")
    if data.draw(st.booleans(), label="append"):
        arr.append(values, axis=axis)
    else:
        arr.resize(tuple(grown))
        arr[...] = values
    assert arr.shape == tuple(grown)
    assert_array_equal(arr[...], values)
    assert_array_equal(zarr.open_array(store, mode="r")[...], values)
