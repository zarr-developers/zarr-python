from __future__ import annotations

from numcodecs import GZip

from zarr.codecs._numcodecs import get_numcodec
from zarr.codecs._v2 import _is_numcodec, _is_numcodec_cls


def test_get_numcodec() -> None:
    assert get_numcodec({"id": "gzip", "level": 2}) == GZip(level=2)  # type: ignore[typeddict-unknown-key]


def test_is_numcodec() -> None:
    """
    Test the _is_numcodec function
    """
    assert _is_numcodec(GZip())


def test_is_numcodec_cls() -> None:
    """
    Test the _is_numcodec_cls function
    """
    assert _is_numcodec_cls(GZip)
