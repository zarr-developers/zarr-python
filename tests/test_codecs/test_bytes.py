"""Tests for `BytesCodec` and the deprecation of the `Endian` enum."""

from __future__ import annotations

import enum
import sys
import warnings
from typing import Any, cast

import pytest

from zarr.codecs.bytes import (
    ENDIAN,
    BytesCodec,
    Endian,
    EndianLiteral,
)


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
    BytesCodec.to_dict / from_dict preserves every value in ENDIAN. Guards
    against drift in the codec's V3 JSON form.
    """
    codec = BytesCodec(endian=endian)
    restored = BytesCodec.from_dict(codec.to_dict())
    assert restored == codec


@pytest.mark.parametrize(
    ("member", "expected"),
    [("little", "little"), ("big", "big")],
)
def test_endian_member_access_warns(member: str, expected: str) -> None:
    """
    Accessing a member on the deprecated `Endian` class emits a
    `DeprecationWarning` and resolves to the equivalent literal string.
    """
    with pytest.warns(DeprecationWarning, match=f"Endian.{member}"):
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
    Passing a real `enum.Enum` instance to `BytesCodec.__init__` triggers
    the init-level deprecation warning and normalizes the value to the
    corresponding literal string.
    """

    class LegacyEndian(enum.Enum):
        little = "little"

    with pytest.warns(DeprecationWarning, match="enum"):
        codec = BytesCodec(endian=cast(Endian, LegacyEndian.little))
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
        _ = Endian.not_a_member


def test_bytes_codec_default_endian_matches_system() -> None:
    """
    Constructing `BytesCodec()` with no arguments yields a codec whose
    `endian` matches `sys.byteorder`. This replaces the previous
    `default_system_endian = Endian(sys.byteorder)` module-level binding.
    """
    codec = BytesCodec()
    assert codec.endian == sys.byteorder
