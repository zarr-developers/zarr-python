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
from zarr.codecs._v2 import codec_json_v2_to_v3
from zarr.errors import ZarrUserWarning
from zarr.registry import get_codec_class, get_numcodec

if TYPE_CHECKING:
    from collections.abc import Iterator

    from zarr.core.common import CodecJSON, CodecJSON_V2, ZarrFormat

CODECS_WITH_SPECS: Final = ("zstd", "gzip", "blosc")


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


@pytest.mark.parametrize("codec_cls", ALL_CODECS)
def test_get_codec_class(codec_cls: type[_numcodecs._NumcodecsCodec]) -> None:
    assert get_codec_class(codec_cls.codec_name) == codec_cls  # type: ignore[comparison-overlap]


@pytest.mark.parametrize("codec_class", ALL_CODECS)
def test_docstring(codec_class: type[_numcodecs._NumcodecsCodec]) -> None:
    """
    Test that the docstring for the zarr.numcodecs codecs references the wrapped numcodecs class.
    """
    assert "See [numcodecs." in codec_class.__doc__  # type: ignore[operator]


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
        b = open_array(a.store, mode="r")
    np.testing.assert_array_equal(data, b[:, :])


@pytest.mark.parametrize("codec_class", [_numcodecs.PCodec, _numcodecs.ZFPY])
def test_generic_bytes_codec(codec_class: type[_numcodecs._NumcodecsArrayBytesCodec]) -> None:
    try:
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
        _numcodecs.PCodec,
        _numcodecs.ZFPY,
    ],
)
def test_codecs_pickleable(codec_cls: type[_numcodecs._NumcodecsCodec]) -> None:
    codec = codec_cls()
    expected = codec

    p = pickle.dumps(codec)
    actual = pickle.loads(p)
    assert actual == expected


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
@pytest.mark.parametrize(
    ("codec", "expected"),
    [
        # Default configurations
        (_numcodecs.LZ4(), {"id": "lz4", "acceleration": 1}),
        (_numcodecs.Zlib(), {"id": "zlib", "level": 1}),
        (_numcodecs.BZ2(), {"id": "bz2", "level": 1}),
        (
            _numcodecs.LZMA(),
            {"id": "lzma", "filters": None, "preset": None, "format": 1, "check": -1},
        ),
        (_numcodecs.Shuffle(), {"id": "shuffle", "elementsize": 4}),
        (_numcodecs.PackBits(), {"id": "packbits"}),
        (_numcodecs.CRC32(), {"id": "crc32"}),
        (_numcodecs.Adler32(), {"id": "adler32"}),
        (_numcodecs.Fletcher32(), {"id": "fletcher32"}),
        (_numcodecs.JenkinsLookup3(), {"id": "jenkins_lookup3", "initval": 0, "prefix": None}),
        (
            _numcodecs.PCodec(),
            {
                "id": "pcodec",
                "delta_encoding_order": None,
                "delta_spec": "auto",
                "equal_pages_up_to": 262144,
                "level": 8,
                "mode_spec": "auto",
                "paging_spec": "equal_pages_up_to",
            },
        ),
        (
            _numcodecs.ZFPY(),
            {
                "id": "zfpy",
                "compression_kwargs": {"tolerance": -1},
                "mode": 4,
                "precision": -1,
                "rate": -1,
                "tolerance": -1,
            },
        ),
        (_numcodecs.Delta(dtype="uint8"), {"id": "delta", "dtype": "|u1", "astype": "|u1"}),
        (_numcodecs.BitRound(keepbits=8), {"id": "bitround", "keepbits": 8}),
        (
            _numcodecs.FixedScaleOffset(offset=0.0, scale=1.0, dtype="uint8"),
            {
                "id": "fixedscaleoffset",
                "scale": 1.0,
                "offset": 0.0,
                "astype": "|u1",
                "dtype": "|u1",
            },
        ),
        (
            _numcodecs.Quantize(digits=3, dtype="float32"),
            {"id": "quantize", "digits": 3, "astype": "<f4", "dtype": "<f4"},
        ),
        (
            _numcodecs.AsType(encode_dtype="float32", decode_dtype="float32"),
            {"id": "astype", "encode_dtype": "<f4", "decode_dtype": "<f4"},
        ),
        (
            _numcodecs.Blosc(),
            {"id": "blosc", "clevel": 5, "shuffle": 1, "blocksize": 0, "cname": "lz4"},
        ),
        (_numcodecs.Zstd(), {"id": "zstd", "level": 0, "checksum": False}),
        (_numcodecs.GZip(), {"id": "gzip", "level": 1}),
        (_numcodecs.CRC32C(), {"id": "crc32c"}),
        # Custom configurations - covering important edge cases
        (_numcodecs.LZ4(acceleration=5), {"id": "lz4", "acceleration": 5}),
        (_numcodecs.Zlib(level=6), {"id": "zlib", "level": 6}),
        (_numcodecs.BZ2(level=9), {"id": "bz2", "level": 9}),
        (
            _numcodecs.LZMA(format=1, check=0, preset=6),
            {"id": "lzma", "format": 1, "check": 0, "preset": 6, "filters": None},
        ),
        (_numcodecs.Shuffle(elementsize=8), {"id": "shuffle", "elementsize": 8}),
        (_numcodecs.CRC32(location="start"), {"id": "crc32", "location": "start"}),
        (_numcodecs.CRC32(location="end"), {"id": "crc32", "location": "end"}),
        (_numcodecs.Adler32(location="start"), {"id": "adler32", "location": "start"}),
        (
            _numcodecs.JenkinsLookup3(initval=42, prefix=b"test"),
            {
                "id": "jenkins_lookup3",
                "initval": 42,
                "prefix": np.array([116, 101, 115, 116], dtype=np.uint8),
            },
        ),
        (
            _numcodecs.JenkinsLookup3(initval=0),
            {"id": "jenkins_lookup3", "initval": 0, "prefix": None},
        ),
        (
            _numcodecs.PCodec(level=8, delta_encoding_order=1),
            {
                "id": "pcodec",
                "level": 8,
                "mode_spec": "auto",
                "delta_spec": "auto",
                "paging_spec": "equal_pages_up_to",
                "delta_encoding_order": 1,
                "equal_pages_up_to": 262144,
            },
        ),
        (
            _numcodecs.ZFPY(mode=2, rate=16.0, precision=20, tolerance=0.001),
            {
                "id": "zfpy",
                "mode": 2,
                "compression_kwargs": {"rate": 16.0},
                "tolerance": 0.001,
                "rate": 16.0,
                "precision": 20,
            },
        ),
        (_numcodecs.Delta(dtype="int16"), {"id": "delta", "dtype": "<i2", "astype": "<i2"}),
        (_numcodecs.BitRound(keepbits=12), {"id": "bitround", "keepbits": 12}),
    ],
)
def test_json_roundtrip_default_config(
    codec: _numcodecs._NumcodecsCodec, expected: CodecJSON_V2, zarr_format: ZarrFormat
) -> None:
    """Test JSON serialization and roundtrip for all codecs with various configurations."""
    # Helper function to compare dictionaries with potential numpy arrays

    # Test serialization
    expected_transformed: CodecJSON
    if zarr_format == 3:
        expected_transformed = codec_json_v2_to_v3(expected)
    else:
        expected_transformed = expected

    json_output = codec.to_json(zarr_format=zarr_format)
    assert compare_json_dicts(json_output, expected_transformed), (
        f"JSON mismatch: {json_output} != {expected_transformed}"
    )

    codec_from_json = type(codec).from_json(json_output)

    original_config = codec.codec_config
    roundtrip_config = codec_from_json.codec_config

    assert compare_json_dicts(roundtrip_config, original_config), (
        f"Roundtrip config mismatch: {roundtrip_config} != {original_config}"
    )


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
