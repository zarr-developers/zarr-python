"""Tests for `BytesCodec` and the deprecation of the `Endian` enum."""

from __future__ import annotations

import enum
import sys
import warnings
from typing import Any, Literal, cast

import numpy as np
import pytest

import zarr
from zarr.abc.codec import SupportsSyncCodec
from zarr.abc.store import Store
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


@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
@pytest.mark.parametrize("input_dtype", [">u2", "<u2"])
@pytest.mark.parametrize("store_endian", ["big", "little"])
async def test_endian(
    store: Store,
    input_dtype: Literal[">u2", "<u2"],
    store_endian: Literal["big", "little"],
) -> None:
    """
    The `bytes` codec stores multi-byte data in the byte order configured on the
    codec, regardless of the input array's byte order, and reads it back to the
    original values. The input-dtype/store-endian cross-product exercises the
    encode-side byteswap (input byte order != store byte order) and the no-op
    case alike. Compression is disabled so the stored chunk is the codec's raw
    output and its byte layout can be asserted directly.
    """
    data = np.arange(0, 256, dtype=input_dtype).reshape((16, 16))
    path = "endian"
    spath = StorePath(store, path)
    a = await zarr.api.asynchronous.create_array(
        spath,
        shape=data.shape,
        chunks=(16, 16),
        dtype="uint16",
        fill_value=0,
        compressors=None,
        serializer=BytesCodec(endian=store_endian),
    )

    await _AsyncArrayProxy(a)[:, :].set(data)

    # The stored chunk is laid out in the byte order configured on the codec.
    stored = await store.get(f"{path}/c/0/0", prototype=default_buffer_prototype())
    assert stored is not None
    expected_dtype = ">u2" if store_endian == "big" else "<u2"
    assert stored.to_bytes() == data.astype(expected_dtype).tobytes()

    # ... and the data reads back to the original values.
    readback_data = await _AsyncArrayProxy(a)[:, :].get()
    assert np.array_equal(data, readback_data)


def test_bytes_codec_supports_sync() -> None:
    assert isinstance(BytesCodec(), SupportsSyncCodec)


def test_bytes_codec_sync_roundtrip() -> None:
    codec = BytesCodec()
    arr = np.arange(100, dtype="float64")
    zdtype = get_data_type_from_native_dtype(arr.dtype)
    spec = ArraySpec(
        shape=arr.shape,
        dtype=zdtype,
        fill_value=zdtype.cast_scalar(0),
        config=ArrayConfig(order="C", write_empty_chunks=True),
        prototype=default_buffer_prototype(),
    )
    nd_buf: NDBuffer = default_buffer_prototype().nd_buffer.from_numpy_array(arr)

    codec = codec.evolve_from_array_spec(spec)

    encoded = codec._encode_sync(nd_buf, spec)
    assert encoded is not None
    decoded = codec._decode_sync(encoded, spec)
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


@pytest.mark.parametrize(
    ("endian", "expected"),
    [
        pytest.param(
            "little", {"name": "bytes", "configuration": {"endian": "little"}}, id="little"
        ),
        pytest.param("big", {"name": "bytes", "configuration": {"endian": "big"}}, id="big"),
        pytest.param(None, {"name": "bytes"}, id="missing"),
    ],
)
def test_to_dict(endian: EndianLiteral | None, expected: dict[str, Any]) -> None:
    codec = BytesCodec(endian=endian)

    actual = codec.to_dict()

    assert actual == expected


@pytest.mark.parametrize(
    ("mapping", "expected"),
    [
        pytest.param(
            {"name": "bytes", "configuration": {"endian": "little"}}, "little", id="little"
        ),
        pytest.param({"name": "bytes", "configuration": {"endian": "big"}}, "big", id="big"),
        pytest.param({"name": "bytes"}, None, id="missing"),
    ],
)
def test_from_dict(mapping: dict[str, Any], expected: EndianLiteral | None) -> None:
    actual = BytesCodec.from_dict(mapping)

    assert actual.endian == expected


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


def test_bytes_codec_rejects_unknown_endian() -> None:
    """
    `BytesCodec.__init__` raises `ValueError` when given a string outside
    `ENDIAN`, and the error message names the offending parameter.
    """
    kwargs: dict[str, Any] = {"endian": "north"}
    with pytest.raises(ValueError, match="endian must be one of"):
        BytesCodec(**kwargs)


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
