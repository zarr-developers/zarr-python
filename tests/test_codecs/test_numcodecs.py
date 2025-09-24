from __future__ import annotations

import contextlib
import pickle
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pytest
from numcodecs import GZip

from zarr import config, create_array, open_array
from zarr.abc.numcodec import Numcodec, _is_numcodec_cls
from zarr.codecs import numcodecs as _numcodecs
from zarr.errors import ZarrUserWarning
from zarr.registry import get_codec_class, get_numcodec

if TYPE_CHECKING:
    from collections.abc import Iterator


CODECS_WITH_SPECS: Final = ("zstd", "gzip", "blosc", "crc32c")

PCODEC_MISSING: bool
ZFPY_MISSING: bool


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
    Test isinstance with a Numcodec
    """
    assert isinstance(GZip(), Numcodec)


def test_is_numcodec_cls() -> None:
    """
    Test the _is_numcodec_cls function
    """
    assert _is_numcodec_cls(GZip)


EXPECTED_WARNING_STR = (
    "Data saved with this codec may not be supported by other Zarr implementations. "
)

ALL_CODECS = tuple(
    filter(
        lambda v: issubclass(v, _numcodecs._NumcodecsCodec) and hasattr(v, "codec_name"),
        tuple(getattr(_numcodecs, cls_name) for cls_name in _numcodecs.__all__),
    )
)


@pytest.mark.parametrize(
    "codec_cls",
    [codec for codec in ALL_CODECS if codec.codec_name.split(".")[-1] not in CODECS_WITH_SPECS],
)
def test_get_codec_class(codec_cls: type[_numcodecs._NumcodecsCodec]) -> None:
    assert get_codec_class(codec_cls.codec_name) == codec_cls  # type: ignore[comparison-overlap]


@pytest.mark.parametrize("codec_class", ALL_CODECS)
def test_docstring(codec_class: type[_numcodecs._NumcodecsCodec]) -> None:
    """
    Test that the docstring for the zarr.numcodecs codecs references the wrapped numcodecs class.
    """
    # TODO: unskip or delete when we add docstrings
    pytest.skip(f"Skipping the docstring check for {codec_class}")


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
    compressors = [codec_class()]

    if codec_class._codec_id not in CODECS_WITH_SPECS:
        with pytest.warns(ZarrUserWarning, match=EXPECTED_WARNING_STR):
            a = create_array(
                {},
                shape=data.shape,
                chunks=(16, 16),
                dtype=data.dtype,
                fill_value=0,
                compressors=compressors,
            )
    else:
        a = create_array(
            {},
            shape=data.shape,
            chunks=(16, 16),
            dtype=data.dtype,
            fill_value=0,
            compressors=compressors,
        )
    a[:, :] = data.copy()
    np.testing.assert_array_equal(data, a[:, :])


@pytest.mark.parametrize(
    ("codec_class", "codec_config"),
    [
        (_numcodecs.Delta, {"dtype": "float32"}),
        (_numcodecs.FixedScaleOffset, {"offset": 0, "scale": 25.5, "dtype": "float32"}),
        (_numcodecs.FixedScaleOffset, {"offset": 0, "scale": 51, "dtype": "float32"}),
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
            filters=[_numcodecs.Quantize(digits=3, dtype="float32")],
        )

    a[:, :] = data.copy()
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
    b = open_array(a.store, mode="r")
    np.testing.assert_array_equal(data, b[:, :])

    with pytest.raises(ValueError, match=r".*requires bool dtype.*"):
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
        b = open_array(a.store, mode="r")
    np.testing.assert_array_equal(data, b[:, :])


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
        b = open_array(a.store, mode="r")
    np.testing.assert_array_equal(data, b[:, :])


def test_repr() -> None:
    codec = _numcodecs.LZ4(acceleration=5)
    assert (
        repr(codec)
        == "LZ4(codec_name='numcodecs.lz4', _codec=LZ4(acceleration=5), codec_config={'id': 'lz4', 'acceleration': 5})"
    )


def test_to_dict() -> None:
    codec = _numcodecs.LZ4(acceleration=5)
    with pytest.warns(ZarrUserWarning, match=EXPECTED_WARNING_STR):
        assert codec.to_dict() == {"name": "lz4", "configuration": {"acceleration": 5}}


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
        # BitRound, Delta, FixedScaleOffset, Quantize, AsType removed
        # because they require mandatory parameters
        _numcodecs.PackBits,
        _numcodecs.CRC32,
        _numcodecs.CRC32C,
        _numcodecs.Adler32,
        _numcodecs.Fletcher32,
        _numcodecs.JenkinsLookup3,
    ],
)
def test_codecs_pickleable(codec_cls: type[_numcodecs._NumcodecsCodec]) -> None:
    codec = codec_cls()
    expected = codec

    p = pickle.dumps(codec)
    actual = pickle.loads(p)
    assert actual == expected


def compare_json_dicts(actual: Any, expected: Any) -> bool:
    """Compare two dictionaries that may contain numpy arrays."""
    if set(actual.keys()) != set(expected.keys()):
        return False
    for key in actual:
        actual_val = actual[key]
        expected_val = expected[key]
        if isinstance(actual_val, np.ndarray) and isinstance(expected_val, np.ndarray):
            if not np.array_equal(actual_val, expected_val):
                return False
        elif isinstance(actual_val, dict) and isinstance(expected_val, dict):
            if not compare_json_dicts(actual_val, expected_val):
                return False
        elif (
            isinstance(actual_val, np.ndarray)
            or isinstance(expected_val, np.ndarray)
            or actual_val != expected_val
        ):
            return False
    return True
