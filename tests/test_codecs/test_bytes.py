"""Tests for `BytesCodec` and the deprecation of the `Endian` enum."""

from __future__ import annotations

import enum
import sys
import warnings
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pytest

import zarr
from tests.conftest import Expect, ExpectFail
from zarr.abc.codec import SupportsSyncCodec
from zarr.abc.store import OffsetByteRequest, RangeByteRequest
from zarr.codecs.bytes import (
    ENDIAN,
    BytesCodec,
    Endian,
    EndianLiteral,
    _row_window,
)
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import NDBuffer, default_buffer_prototype
from zarr.core.dtype import get_data_type_from_native_dtype
from zarr.core.dtype.npy.int import Int8, Int32
from zarr.core.dtype.npy.structured import Struct
from zarr.storage import StorePath

from .test_codecs import _AsyncArrayProxy

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from zarr.abc.store import Store


@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
@pytest.mark.parametrize(
    "input_dtype",
    [
        ">u2",
        "<u2",
        [("flux", ">f4"), ("mask", ">i4")],
        [("flux", "<f4"), ("mask", "<i4")],
    ],
    ids=["big-scalar", "little-scalar", "big-struct", "little-struct"],
)
@pytest.mark.parametrize("store_endian", ["big", "little"])
async def test_endian(
    store: Store,
    input_dtype: str | list[tuple[str, str]],
    store_endian: Literal["big", "little"],
) -> None:
    """
    The `bytes` codec stores multi-byte data in the byte order configured on the
    codec, regardless of the input array's byte order, and reads it back to the
    original values. For structured dtypes this applies to every multi-byte
    field, per the `struct` data type spec; the struct cases guard against the
    endianness bugs from
    https://github.com/zarr-developers/zarr-python/issues/4141, where the
    encode path never byte-swapped struct fields (numpy reports byteorder '|'
    for void dtypes) and the decode path ignored the codec's endian entirely.
    The input-dtype/store-endian cross-product exercises the encode-side
    byteswap (input byte order != store byte order) and the no-op case alike.
    Compression is disabled so the stored chunk is the codec's raw output and
    its byte layout can be asserted directly.
    """
    dtype = np.dtype(input_dtype)
    if dtype.fields is None:
        data = np.arange(0, 256, dtype=dtype).reshape((16, 16))
    else:
        data = np.zeros((16, 16), dtype=dtype)
        data["flux"] = np.arange(0, 256).reshape((16, 16))
        data["mask"] = np.arange(256, 512).reshape((16, 16))
    path = "endian"
    spath = StorePath(store, path)
    a = await zarr.api.asynchronous.create_array(
        spath,
        shape=data.shape,
        chunks=(16, 16),
        dtype=dtype,
        fill_value=0,
        compressors=None,
        serializer=BytesCodec(endian=store_endian),
    )

    await _AsyncArrayProxy(a)[:, :].set(data)

    # The stored chunk is laid out in the byte order configured on the codec.
    stored = await store.get(f"{path}/c/0/0", prototype=default_buffer_prototype())
    assert stored is not None
    assert stored.to_bytes() == data.astype(dtype.newbyteorder(store_endian)).tobytes()

    # ... and the data reads back to the original values.
    readback_data = await _AsyncArrayProxy(a)[:, :].get()
    assert np.array_equal(data, readback_data)


def test_bytes_codec_supports_sync() -> None:
    assert isinstance(BytesCodec(), SupportsSyncCodec)


@pytest.mark.parametrize("endian", ENDIAN)
@pytest.mark.parametrize(
    "native_dtype",
    [np.dtype("float64"), np.dtype(">u2"), np.dtype([("a", ">f4"), ("b", "<i4")])],
    ids=["native-scalar", "big-scalar", "mixed-endian-struct"],
)
def test_bytes_codec_sync_roundtrip(endian: EndianLiteral, native_dtype: np.dtype[Any]) -> None:
    """
    The synchronous encode/decode path round-trips data, and the two byte
    orders involved are independent: the codec's `endian` configuration governs
    only the stored byte layout (every multi-byte value, including struct
    fields, is laid out in the codec's byte order regardless of the input
    array's byte order), while the decoded buffer's byte order is governed by
    the array's data type regardless of the codec's. The mixed-endian struct
    case pins that per-field byte order of the in-memory dtype survives a
    roundtrip through a single stored byte order.
    """
    if native_dtype.fields is None:
        arr = np.arange(100, dtype=native_dtype)
    else:
        arr = np.array([(1.5, 2), (3.5, 4), (5.5, 6), (7.5, 8)], dtype=native_dtype)
    zdtype = get_data_type_from_native_dtype(arr.dtype)
    spec = ArraySpec(
        shape=arr.shape,
        dtype=zdtype,
        fill_value=zdtype.cast_scalar(0),
        config=ArrayConfig(order="C", write_empty_chunks=True),
        prototype=default_buffer_prototype(),
    )
    nd_buf: NDBuffer = default_buffer_prototype().nd_buffer.from_numpy_array(arr)

    codec = BytesCodec(endian=endian).evolve_from_array_spec(spec)

    encoded = codec._encode_sync(nd_buf, spec)
    assert encoded is not None
    assert encoded.to_bytes() == arr.astype(native_dtype.newbyteorder(endian)).tobytes()

    decoded = codec._decode_sync(encoded, spec)
    assert decoded.dtype == zdtype.to_native_dtype()
    np.testing.assert_array_equal(arr, decoded.as_numpy_array())


@pytest.mark.parametrize("endian", ENDIAN)
def test_bytes_codec_accepts_all_endians(endian: EndianLiteral) -> None:
    """
    Every endian value in ENDIAN is accepted by BytesCodec and round-trips
    to the same value on the stored attribute. Catches drift between the
    EndianLiteral type alias and the runtime ENDIAN tuple.
    """
    codec = BytesCodec(endian=endian)
    assert codec.endian == endian


@pytest.mark.parametrize("endian", ENDIAN)
def test_bytes_codec_json_roundtrip(endian: EndianLiteral) -> None:
    """
    BytesCodec.to_dict produces the spec-defined wire shape and the
    round-trip through from_dict preserves equality. Asserting the literal
    JSON shape catches drift between BytesCodec's runtime representation and
    the codec's V3 on-disk form.
    """
    codec = BytesCodec(endian=endian)
    assert codec.to_dict() == {"name": "bytes", "configuration": {"endian": endian}}
    restored = BytesCodec.from_dict(codec.to_dict())
    assert restored == codec


# to_dict and from_dict are inverses over this (endian setting, wire dict) mapping:
# to_dict turns the endian setting into the dict; from_dict recovers it.
_ENDIAN_DICT_CASES: list[Expect[EndianLiteral | None, dict[str, Any]]] = [
    Expect(
        input="little",
        output={"name": "bytes", "configuration": {"endian": "little"}},
        id="little",
    ),
    Expect(
        input="big",
        output={"name": "bytes", "configuration": {"endian": "big"}},
        id="big",
    ),
    Expect(input=None, output={"name": "bytes"}, id="missing"),
]


@pytest.mark.parametrize("case", _ENDIAN_DICT_CASES, ids=lambda c: c.id)
def test_to_dict(case: Expect[EndianLiteral | None, dict[str, Any]]) -> None:
    assert BytesCodec(endian=case.input).to_dict() == case.output


@pytest.mark.parametrize("case", _ENDIAN_DICT_CASES, ids=lambda c: c.id)
def test_from_dict(case: Expect[EndianLiteral | None, dict[str, Any]]) -> None:
    assert BytesCodec.from_dict(case.output).endian == case.input


@pytest.mark.parametrize("endian", ["little", "big", pytest.param(None, id="missing")])
def test_roundtrip(endian: EndianLiteral | None) -> None:
    codec = BytesCodec(endian=endian)

    encoded = codec.to_dict()
    roundtripped = BytesCodec.from_dict(encoded)

    assert codec == roundtripped


@pytest.mark.parametrize(
    ("member", "expected"),
    [("little", "little"), ("big", "big")],
)
def test_endian_member_access_warns(member: str, expected: str) -> None:
    """
    Accessing a member on the deprecated `Endian` class emits a
    `DeprecationWarning` and resolves to the equivalent literal string.
    """
    with pytest.warns(DeprecationWarning, match=rf"Endian\.{member}"):
        value = getattr(Endian, member)
    assert value == expected


def test_endian_class_imports_silently() -> None:
    """
    Importing the deprecated `Endian` class by name must not emit a warning;
    only member access does. Guards against `bytes.py` accidentally
    triggering its own deprecation warnings at import time.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        from zarr.codecs.bytes import Endian as _Endian  # noqa: F401


def test_bytes_codec_init_with_enum_instance_warns() -> None:
    """
    Passing a foreign `enum.Enum` instance to `BytesCodec.__init__` triggers
    the init-level deprecation warning (from `_coerce_enum_input`) and
    normalizes the value to the corresponding literal string. Covers the
    case where a downstream package defined its own enum-shaped class to
    bridge between zarr's old API and its own.
    """

    class LegacyEndian(enum.Enum):
        little = "little"

    with pytest.warns(DeprecationWarning, match=r"Passing an enum to BytesCodec"):
        codec = BytesCodec(endian=cast(Endian, LegacyEndian.little))
    assert codec.endian == "little"


def test_bytes_codec_init_with_deprecated_class_member() -> None:
    """
    The realistic legacy-upgrade idiom: `BytesCodec(endian=Endian.little)`.
    Member access on `Endian` emits one `DeprecationWarning` (from the
    metaclass) and resolves to the bare string, which `BytesCodec` then
    accepts without further warning. No second warning from
    `_coerce_enum_input` because the metaclass already produced a string.

    The `cast` is necessary because the metaclass `__getattr__` is typed
    as returning `str`, which does not statically match the codec's
    `EndianLiteral` parameter even though the runtime value does.
    """
    with pytest.warns(DeprecationWarning, match=r"Endian\.little"):
        codec = BytesCodec(endian=cast(EndianLiteral, Endian.little))
    assert codec.endian == "little"


@pytest.mark.parametrize(
    "case",
    [
        ExpectFail(
            input="north",
            exception=ValueError,
            id="unknown-string",
            msg="endian must be one of",
        ),
    ],
    ids=lambda c: c.id,
)
def test_bytes_codec_rejects_unknown_endian(case: ExpectFail[Any]) -> None:
    """
    `BytesCodec.__init__` raises `ValueError` when given a value outside
    `ENDIAN`, and the error message names the offending parameter.
    """
    with case.raises():
        BytesCodec(endian=case.input)


def test_endian_attribute_error_for_unknown_member() -> None:
    """
    Attribute access for a name that is not a known member of the
    deprecated `Endian` class falls through to `AttributeError`, matching
    the behavior of a regular class.
    """
    with pytest.raises(AttributeError):
        getattr(Endian, "not_a_member")  # noqa: B009


def test_bytes_codec_default_endian_matches_system() -> None:
    """
    Constructing `BytesCodec()` with no arguments yields a codec whose
    `endian` matches `sys.byteorder`. This replaces the previous
    `default_system_endian = Endian(sys.byteorder)` module-level binding.
    """
    codec = BytesCodec()
    assert codec.endian == sys.byteorder


def _make_array_spec(dtype: Any) -> ArraySpec:
    """Build a minimal ArraySpec around the given dtype for codec.evolve testing."""
    return ArraySpec(
        shape=(1,),
        dtype=dtype,
        fill_value=0,
        config=cast(ArrayConfig, {}),
        prototype=default_buffer_prototype(),
    )


def test_bytes_codec_evolve_structured_multi_byte_fields_warns_and_defaults() -> None:
    """
    BytesCodec(endian=None).evolve_from_array_spec(spec) with a structured dtype
    whose fields contain multi-byte members emits a UserWarning about the
    missing endian and returns a codec with endian set to "little" for legacy
    compatibility.
    """
    codec = BytesCodec(endian=None)
    dtype = Struct(fields=(("a", Int32()), ("b", Int32())))
    spec = _make_array_spec(dtype)
    with pytest.warns(UserWarning, match=r"Missing 'endian' for structured dtype"):
        evolved = codec.evolve_from_array_spec(spec)
    assert evolved.endian == "little"


def test_bytes_codec_evolve_structured_single_byte_fields_clears_endian() -> None:
    """
    For a structured dtype whose fields are all single-byte, BytesCodec drops
    its endian on evolve (endian is meaningless for single-byte content).
    """
    codec = BytesCodec(endian="little")
    dtype = Struct(fields=(("a", Int8()), ("b", Int8())))
    spec = _make_array_spec(dtype)
    evolved = codec.evolve_from_array_spec(spec)
    assert evolved.endian is None


@pytest.fixture(
    params=[
        "zarr.core.codec_pipeline.BatchedCodecPipeline",
        "zarr.core.codec_pipeline.FusedCodecPipeline",
    ]
)
def codec_pipeline(request: pytest.FixtureRequest) -> Iterator[None]:
    """Run a test once with each codec pipeline."""
    with zarr.config.set({"codec_pipeline.path": request.param}):
        yield


class _RangeLoggingStore(zarr.storage.MemoryStore):
    """A memory store that records the byte ranges it serves for chunk keys."""

    def __init__(self) -> None:
        super().__init__()
        self.reads: list[tuple[str, Any, int]] = []

    def _log(self, key: str, byte_range: Any, buf: Any) -> None:
        if buf is not None and not key.endswith("zarr.json"):
            self.reads.append((key, byte_range, len(buf)))

    async def get(self, key: str, prototype: Any = None, byte_range: Any = None) -> Any:
        buf = await super().get(key, prototype, byte_range)
        self._log(key, byte_range, buf)
        return buf

    def get_sync(self, key: str, *, prototype: Any = None, byte_range: Any = None) -> Any:
        buf = super().get_sync(key, prototype=prototype, byte_range=byte_range)
        self._log(key, byte_range, buf)
        return buf


def _single_chunk_array(
    data: np.ndarray[Any, Any], **kwargs: Any
) -> tuple[Any, _RangeLoggingStore]:
    store = _RangeLoggingStore()
    arr = zarr.create_array(
        store, shape=data.shape, dtype=data.dtype, chunks=data.shape, fill_value=0, **kwargs
    )
    arr[...] = data
    store.reads.clear()
    return arr, store


@pytest.mark.usefixtures("codec_pipeline")
@pytest.mark.parametrize(
    ("selection", "rows_read"),
    [
        (np.s_[500:600, 3], 100),
        (np.s_[500:600], 100),
        (np.s_[777, 5], 1),
        (np.s_[-1], 1),
        (np.s_[10:20:3, ::2], 10),  # rows 10 to 19 are fetched
        (np.s_[...], 10_000),
    ],
)
def test_uncompressed_partial_read(selection: Any, rows_read: int) -> None:
    """Reading part of an uncompressed chunk fetches only the rows it touches."""
    data = np.arange(100_000, dtype="int16").reshape(10_000, 10)
    arr, store = _single_chunk_array(data, compressors=None)
    np.testing.assert_array_equal(arr[selection], data[selection])
    assert sum(n for *_, n in store.reads) == rows_read * 10 * 2


@pytest.mark.usefixtures("codec_pipeline")
@pytest.mark.parametrize("dtype", [">u2", "<f8", "u1", "bool"])
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_uncompressed_partial_read_values(dtype: str, ndim: int) -> None:
    """Partial reads return the same values as numpy for several dtypes, byte orders, and shapes."""
    shape = {1: (1000,), 2: (100, 7), 3: (20, 5, 3)}[ndim]
    data = (np.arange(int(np.prod(shape))) % 200).astype(dtype).reshape(shape)
    arr, _ = _single_chunk_array(data, compressors=None)
    selections: list[Any] = [np.s_[3:9], np.s_[4], np.s_[-3:], np.s_[1:15:4]]
    for selection in selections:
        np.testing.assert_array_equal(arr[selection], data[selection])
    np.testing.assert_array_equal(arr.oindex[[1, 5, 2]], data[[1, 5, 2]])
    mask = np.zeros(shape[0], dtype=bool)
    mask[[2, 5, 11]] = True
    np.testing.assert_array_equal(arr.oindex[mask], data[mask])
    coords = tuple(np.array([0, 6, 3]) % n for n in shape)
    np.testing.assert_array_equal(arr.vindex[coords], data[coords])


@pytest.mark.usefixtures("codec_pipeline")
def test_uncompressed_partial_read_across_chunks() -> None:
    """A selection spanning several chunks reads only the touched rows of each."""
    data = np.arange(40_000, dtype="int32").reshape(4000, 10)
    store = _RangeLoggingStore()
    arr = zarr.create_array(
        store, shape=data.shape, dtype="int32", chunks=(1000, 10), compressors=None
    )
    arr[...] = data
    store.reads.clear()
    np.testing.assert_array_equal(arr[990:1010, 2], data[990:1010, 2])
    assert sorted((key, n) for key, _, n in store.reads) == [("c/0/0", 400), ("c/1/0", 400)]


@pytest.mark.usefixtures("codec_pipeline")
def test_compressed_chunks_are_read_whole() -> None:
    """With a compressor the chunk bytes cannot be split, so the whole chunk is fetched."""
    data = np.arange(100_000, dtype="int16").reshape(10_000, 10)
    arr, store = _single_chunk_array(data)  # default compressor
    np.testing.assert_array_equal(arr[500:600, 3], data[500:600, 3])
    ((_, byte_range, _),) = store.reads
    assert byte_range is None


@pytest.mark.usefixtures("codec_pipeline")
@pytest.mark.parametrize("selection", [np.s_[10:20], np.s_[...]])
def test_uncompressed_partial_read_missing_chunk(selection: Any) -> None:
    """An unwritten chunk reads as the fill value, for a partial and a whole-chunk read."""
    store = _RangeLoggingStore()
    arr = zarr.create_array(
        store, shape=(100, 4), dtype="int16", chunks=(100, 4), compressors=None, fill_value=7
    )
    expected = np.full((100, 4), 7, dtype="int16")[selection]
    np.testing.assert_array_equal(arr[selection], expected)


class _RewritesRangeStore(zarr.storage.MemoryStore):
    """A memory store that does not serve the byte range it is asked for.

    `rewrite` maps the requested range to the one that is served instead.
    """

    def __init__(self, rewrite: Callable[[RangeByteRequest], Any]) -> None:
        super().__init__()
        self.rewrite = rewrite

    def _served(self, byte_range: Any) -> Any:
        if isinstance(byte_range, RangeByteRequest):
            return self.rewrite(byte_range)
        return byte_range

    async def get(self, key: str, prototype: Any = None, byte_range: Any = None) -> Any:
        return await super().get(key, prototype, self._served(byte_range))

    def get_sync(self, key: str, *, prototype: Any = None, byte_range: Any = None) -> Any:
        return super().get_sync(key, prototype=prototype, byte_range=self._served(byte_range))


def _ignore_range(byte_range: RangeByteRequest) -> None:
    """Send the whole value, as an HTTP server that ignores the Range header does."""


def _ignore_range_end(byte_range: RangeByteRequest) -> OffsetByteRequest:
    """Send everything from the start of the range to the end of the value."""
    return OffsetByteRequest(byte_range.start)


def _one_byte_short(byte_range: RangeByteRequest) -> RangeByteRequest:
    return RangeByteRequest(byte_range.start, byte_range.end - 1)


def _rewriting_array(
    rewrite: Callable[[RangeByteRequest], Any],
) -> tuple[Any, np.ndarray[Any, Any]]:
    data = np.arange(400, dtype="<i2").reshape(100, 4)
    arr = zarr.create_array(
        _RewritesRangeStore(rewrite),
        shape=data.shape,
        dtype=data.dtype,
        chunks=data.shape,
        compressors=None,
    )
    arr[...] = data
    return arr, data


@pytest.mark.usefixtures("codec_pipeline")
@pytest.mark.parametrize("rewrite", [_ignore_range, _ignore_range_end])
@pytest.mark.parametrize("selection", [np.s_[10:12, 0], np.s_[50:52, 0], np.s_[7], np.s_[98:]])
def test_uncompressed_partial_read_store_sends_more_than_range(
    rewrite: Callable[[RangeByteRequest], Any], selection: Any
) -> None:
    """A store that sends more than the requested range still reads correctly."""
    arr, data = _rewriting_array(rewrite)
    np.testing.assert_array_equal(arr[selection], data[selection])


@pytest.mark.usefixtures("codec_pipeline")
def test_uncompressed_partial_read_store_sends_unexpected_length() -> None:
    """A response whose length matches no known byte range behavior is an error."""
    arr, _ = _rewriting_array(_one_byte_short)
    with pytest.raises(ValueError, match="the store returned 15 bytes"):
        arr[50:52, 0]


@pytest.mark.parametrize(
    ("selection", "shape"),
    [
        ((slice(5, 5), slice(None)), (10, 4)),  # no rows
        ((slice(8, 2, -1), slice(None)), (10, 4)),  # negative step
        ((np.array([], dtype=np.intp), slice(None)), (10, 4)),  # no rows
        ((slice(None), 2), (10, 4)),  # every row
        ((np.array([0, 9]), slice(None)), (10, 4)),  # first and last row
        ((..., 2), (10, 4)),  # rows not determined
        (slice(2, 4), (10,)),  # not a tuple
        ((), ()),  # zero-dimensional chunk
    ],
)
def test_row_window_reads_whole_chunk(selection: Any, shape: tuple[int, ...]) -> None:
    """Selections with no narrower row window than the whole chunk give None."""
    assert _row_window(selection, shape) is None
