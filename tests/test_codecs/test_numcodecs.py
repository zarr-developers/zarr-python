from __future__ import annotations

import contextlib
import pickle
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from numcodecs import GZip

from zarr import config, create_array, open_array
from zarr.abc.numcodec import _is_numcodec, _is_numcodec_cls
from zarr.codecs import numcodecs as _numcodecs
from zarr.errors import ZarrUserWarning
from zarr.registry import get_numcodec

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextlib.contextmanager
def codec_conf() -> Iterator[Any]:
    base_conf = config.get("codecs")
    new_conf = {
        "numcodecs.bz2": "zarr.codecs.numcodecs.BZ2",
        "numcodecs.crc32": "zarr.codecs.numcodecs.CRC32",
        "numcodecs.crc32c": "zarr.codecs.numcodecs.CRC32C",
        "numcodecs.lz4": "zarr.codecs.numcodecs.LZ4",
        "numcodecs.lzma": "zarr.codecs.numcodecs.LZMA",
        "numcodecs.zfpy": "zarr.codecs.numcodecs.ZFPY",
        "numcodecs.adler32": "zarr.codecs.numcodecs.Adler32",
        "numcodecs.astype": "zarr.codecs.numcodecs.AsType",
        "numcodecs.bitround": "zarr.codecs.numcodecs.BitRound",
        "numcodecs.blosc": "zarr.codecs.numcodecs.Blosc",
        "numcodecs.delta": "zarr.codecs.numcodecs.Delta",
        "numcodecs.fixedscaleoffset": "zarr.codecs.numcodecs.FixedScaleOffset",
        "numcodecs.fletcher32": "zarr.codecs.numcodecs.Fletcher32",
        "numcodecs.gzip": "zarr.codecs.numcodecs.GZip",
        "numcodecs.jenkinslookup3": "zarr.codecs.numcodecs.JenkinsLookup3",
        "numcodecs.pcodec": "zarr.codecs.numcodecs.PCodec",
        "numcodecs.packbits": "zarr.codecs.numcodecs.PackBits",
        "numcodecs.shuffle": "zarr.codecs.numcodecs.Shuffle",
        "numcodecs.quantize": "zarr.codecs.numcodecs.Quantize",
        "numcodecs.zlib": "zarr.codecs.numcodecs.Zlib",
        "numcodecs.zstd": "zarr.codecs.numcodecs.Zstd",
    }

    yield config.set({"codecs": new_conf | base_conf})


if TYPE_CHECKING:
    from zarr.core.common import JSON


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


EXPECTED_WARNING_STR = "Numcodecs codecs are not in the Zarr version 3.*"

ALL_CODECS = tuple(
    filter(
        lambda v: isinstance(v, _numcodecs._NumcodecsCodec),
        tuple(getattr(_numcodecs, cls_name) for cls_name in _numcodecs.__all__),
    )
)


@pytest.mark.parametrize("codec_class", ALL_CODECS)
def test_docstring(codec_class: type[_numcodecs._NumcodecsCodec]) -> None:
    """
    Test that the docstring for the zarr.numcodecs codecs references the wrapped numcodecs class.
    """
    assert "See :class:`numcodecs." in codec_class.__doc__  # type: ignore[operator]


@pytest.mark.parametrize(
    "codec_class",
    [
        _numcodecs.Blosc,
        _numcodecs.LZ4,
        _numcodecs.Zstd,
        _numcodecs.Zlib,
        _numcodecs.GZip,
        _numcodecs.BZ2,
        _numcodecs.LZMA,
        _numcodecs.Shuffle,
    ],
)
def test_generic_compressor(codec_class: type[_numcodecs._NumcodecsBytesBytesCodec]) -> None:
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    with pytest.warns(ZarrUserWarning, match=EXPECTED_WARNING_STR):
        a = create_array(
            {},
            shape=data.shape,
            chunks=(16, 16),
            dtype=data.dtype,
            fill_value=0,
            compressors=[codec_class()],
        )

    a[:, :] = data.copy()
    np.testing.assert_array_equal(data, a[:, :])


@pytest.mark.parametrize(
    ("codec_class", "codec_config"),
    [
        (_numcodecs.Delta, {"dtype": "float32"}),
        (_numcodecs.FixedScaleOffset, {"offset": 0, "scale": 25.5}),
        (_numcodecs.FixedScaleOffset, {"offset": 0, "scale": 51, "astype": "uint16"}),
        (_numcodecs.AsType, {"encode_dtype": "float32", "decode_dtype": "float32"}),
    ],
    ids=[
        "delta",
        "fixedscaleoffset",
        "fixedscaleoffset2",
        "astype",
    ],
)
def test_generic_filter(
    codec_class: type[_numcodecs._NumcodecsArrayArrayCodec],
    codec_config: dict[str, JSON],
) -> None:
    data = np.linspace(0, 10, 256, dtype="float32").reshape((16, 16))

    with pytest.warns(ZarrUserWarning, match=EXPECTED_WARNING_STR):
        a = create_array(
            {},
            shape=data.shape,
            chunks=(16, 16),
            dtype=data.dtype,
            fill_value=0,
            filters=[
                codec_class(**codec_config),
            ],
        )

    a[:, :] = data.copy()
    with codec_conf():
        with pytest.warns(ZarrUserWarning, match=EXPECTED_WARNING_STR):
            b = open_array(a.store, mode="r")
    np.testing.assert_array_equal(data, b[:, :])


def test_generic_filter_bitround() -> None:
    data = np.linspace(0, 1, 256, dtype="float32").reshape((16, 16))

    with pytest.warns(ZarrUserWarning, match=EXPECTED_WARNING_STR):
        a = create_array(
            {},
            shape=data.shape,
            chunks=(16, 16),
            dtype=data.dtype,
            fill_value=0,
            filters=[_numcodecs.BitRound(keepbits=3)],
        )

    a[:, :] = data.copy()
    with pytest.warns(ZarrUserWarning, match=EXPECTED_WARNING_STR):
        b = open_array(a.store, mode="r")
    assert np.allclose(data, b[:, :], atol=0.1)


def test_generic_filter_quantize() -> None:
    data = np.linspace(0, 10, 256, dtype="float32").reshape((16, 16))

    with pytest.warns(ZarrUserWarning, match=EXPECTED_WARNING_STR):
        a = create_array(
            {},
            shape=data.shape,
            chunks=(16, 16),
            dtype=data.dtype,
            fill_value=0,
            filters=[_numcodecs.Quantize(digits=3)],
        )

    a[:, :] = data.copy()
    with pytest.warns(ZarrUserWarning, match=EXPECTED_WARNING_STR):
        b = open_array(a.store, mode="r")
    assert np.allclose(data, b[:, :], atol=0.001)


def test_generic_filter_packbits() -> None:
    data = np.zeros((16, 16), dtype="bool")
    data[0:4, :] = True

    with pytest.warns(ZarrUserWarning, match=EXPECTED_WARNING_STR):
        a = create_array(
            {},
            shape=data.shape,
            chunks=(16, 16),
            dtype=data.dtype,
            fill_value=0,
            filters=[_numcodecs.PackBits()],
        )

    a[:, :] = data.copy()
    with pytest.warns(ZarrUserWarning, match=EXPECTED_WARNING_STR):
        b = open_array(a.store, mode="r")
    np.testing.assert_array_equal(data, b[:, :])

    with pytest.warns(ZarrUserWarning, match=EXPECTED_WARNING_STR):
        with pytest.raises(ValueError, match=".*requires bool dtype.*"):
            create_array(
                {},
                shape=data.shape,
                chunks=(16, 16),
                dtype="uint32",
                fill_value=0,
                filters=[_numcodecs.PackBits()],
            )


@pytest.mark.parametrize(
    "codec_class",
    [
        _numcodecs.CRC32,
        _numcodecs.CRC32C,
        _numcodecs.Adler32,
        _numcodecs.Fletcher32,
        _numcodecs.JenkinsLookup3,
    ],
)
def test_generic_checksum(codec_class: type[_numcodecs._NumcodecsBytesBytesCodec]) -> None:
    data = np.linspace(0, 10, 256, dtype="float32").reshape((16, 16))

    with pytest.warns(ZarrUserWarning, match=EXPECTED_WARNING_STR):
        a = create_array(
            {},
            shape=data.shape,
            chunks=(16, 16),
            dtype=data.dtype,
            fill_value=0,
            compressors=[codec_class()],
        )

    a[:, :] = data.copy()
    with codec_conf():
        with pytest.warns(ZarrUserWarning, match=EXPECTED_WARNING_STR):
            b = open_array(a.store, mode="r")
    np.testing.assert_array_equal(data, b[:, :])


@pytest.mark.parametrize("codec_class", [_numcodecs.PCodec, _numcodecs.ZFPY])
def test_generic_bytes_codec(codec_class: type[_numcodecs._NumcodecsArrayBytesCodec]) -> None:
    try:
        with pytest.warns(ZarrUserWarning, match=EXPECTED_WARNING_STR):
            codec_class()._codec  # noqa: B018
    except ValueError as e:  # pragma: no cover
        if "codec not available" in str(e):
            pytest.xfail(f"{codec_class.codec_name} is not available: {e}")
        else:
            raise
    except ImportError as e:  # pragma: no cover
        pytest.xfail(f"{codec_class.codec_name} is not available: {e}")

    data = np.arange(0, 256, dtype="float32").reshape((16, 16))

    with pytest.warns(ZarrUserWarning, match=EXPECTED_WARNING_STR):
        a = create_array(
            {},
            shape=data.shape,
            chunks=(16, 16),
            dtype=data.dtype,
            fill_value=0,
            serializer=codec_class(),
        )

    a[:, :] = data.copy()
    np.testing.assert_array_equal(data, a[:, :])


def test_delta_astype() -> None:
    data = np.linspace(0, 10, 256, dtype="i8").reshape((16, 16))

    with pytest.warns(ZarrUserWarning, match=EXPECTED_WARNING_STR):
        a = create_array(
            {},
            shape=data.shape,
            chunks=(16, 16),
            dtype=data.dtype,
            fill_value=0,
            filters=[
                _numcodecs.Delta(dtype="i8", astype="i2"),
            ],
        )

    a[:, :] = data.copy()
    with codec_conf():
        with pytest.warns(ZarrUserWarning, match=EXPECTED_WARNING_STR):
            b = open_array(a.store, mode="r")
    np.testing.assert_array_equal(data, b[:, :])


def test_repr() -> None:
    with pytest.warns(ZarrUserWarning, match=EXPECTED_WARNING_STR):
        codec = _numcodecs.LZ4(level=5)
    assert repr(codec) == "LZ4(codec_name='numcodecs.lz4', codec_config={'level': 5})"


def test_to_dict() -> None:
    with pytest.warns(ZarrUserWarning, match=EXPECTED_WARNING_STR):
        codec = _numcodecs.LZ4(level=5)
    assert codec.to_dict() == {"name": "numcodecs.lz4", "configuration": {"level": 5}}


@pytest.mark.parametrize(
    "codec_cls",
    [
        _numcodecs.Blosc,
        _numcodecs.LZ4,
        _numcodecs.Zstd,
        _numcodecs.Zlib,
        _numcodecs.GZip,
        _numcodecs.BZ2,
        _numcodecs.LZMA,
        _numcodecs.Shuffle,
        _numcodecs.BitRound,
        _numcodecs.Delta,
        _numcodecs.FixedScaleOffset,
        _numcodecs.Quantize,
        _numcodecs.PackBits,
        _numcodecs.AsType,
        _numcodecs.CRC32,
        _numcodecs.CRC32C,
        _numcodecs.Adler32,
        _numcodecs.Fletcher32,
        _numcodecs.JenkinsLookup3,
        _numcodecs.PCodec,
        _numcodecs.ZFPY,
    ],
)
def test_codecs_pickleable(codec_cls: type[_numcodecs._NumcodecsCodec]) -> None:
    with pytest.warns(ZarrUserWarning, match=EXPECTED_WARNING_STR):
        codec = codec_cls()

    expected = codec

    p = pickle.dumps(codec)
    actual = pickle.loads(p)
    assert actual == expected
