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
from zarr.codecs.bytes import (
    ENDIAN,
    BytesCodec,
    Endian,
    EndianLiteral,
)
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import NDBuffer, default_buffer_prototype
from zarr.core.dtype import get_data_type_from_native_dtype
from zarr.core.dtype.npy.int import Int8, Int32
from zarr.core.dtype.npy.structured import Struct
from zarr.storage import StorePath

from .test_codecs import _AsyncArrayProxy

if TYPE_CHECKING:
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
