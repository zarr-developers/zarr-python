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
from zarr.core.common import CodecJSON_V2, ZarrFormat
from zarr.errors import ZarrUserWarning
from zarr.registry import get_numcodec

if TYPE_CHECKING:
    from collections.abc import Iterator

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
        "numcodecs.gZip": "zarr.codecs.numcodecs.GZip",
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
            filters=[_numcodecs.Quantize(digits=3)],
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
    codec = _numcodecs.LZ4(level=5)
    assert repr(codec) == "LZ4(codec_name='numcodecs.lz4', codec_config={'level': 5})"


def test_to_dict() -> None:
    codec = _numcodecs.LZ4(level=5)
    with pytest.warns(ZarrUserWarning, match=EXPECTED_WARNING_STR):
        assert codec.to_dict() == {"name": "lz4", "configuration": {"level": 5}}


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
    codec = codec_cls()
    expected = codec

    p = pickle.dumps(codec)
    actual = pickle.loads(p)
    assert actual == expected


# JSON Serialization/Deserialization Tests


@pytest.mark.parametrize(
    ("codec_class", "codec_config", "expected_v2", "expected_v3"),
    [
        # Bytes-to-bytes codecs
        (
            _numcodecs.LZ4,
            {"acceleration": 5},
            {"id": "lz4", "acceleration": 5},
            {"name": "lz4", "configuration": {"acceleration": 5}},
        ),
        (
            _numcodecs.LZ4,
            {},
            {"id": "lz4"},
            {"name": "lz4", "configuration": {}},
        ),
        (
            _numcodecs.Zlib,
            {"level": 6},
            {"id": "zlib", "level": 6},
            {"name": "zlib", "configuration": {"level": 6}},
        ),
        (
            _numcodecs.Zlib,
            {},
            {"id": "zlib"},
            {"name": "zlib", "configuration": {}},
        ),
        (
            _numcodecs.BZ2,
            {"level": 9},
            {"id": "bz2", "level": 9},
            {"name": "bz2", "configuration": {"level": 9}},
        ),
        (
            _numcodecs.BZ2,
            {},
            {"id": "bz2"},
            {"name": "bz2", "configuration": {}},
        ),
        (
            _numcodecs.LZMA,
            {"format": 1, "check": 0, "preset": 6},
            {"id": "lzma", "format": 1, "check": 0, "preset": 6},
            {"name": "lzma", "configuration": {"format": 1, "check": 0, "preset": 6}},
        ),
        (
            _numcodecs.LZMA,
            {},
            {"id": "lzma"},
            {"name": "lzma", "configuration": {}},
        ),
        (
            _numcodecs.Shuffle,
            {"elementsize": 4},
            {"id": "shuffle", "elementsize": 4},
            {"name": "shuffle", "configuration": {"elementsize": 4}},
        ),
        (
            _numcodecs.Shuffle,
            {},
            {"id": "shuffle"},
            {"name": "shuffle", "configuration": {}},
        ),
    ],
)
def test_bytes_to_bytes_codec_json_v2_v3(
    codec_class: type[_numcodecs._NumcodecsBytesBytesCodec],
    codec_config: dict[str, Any],
    expected_v2: dict[str, Any],
    expected_v3: dict[str, Any],
) -> None:
    """Test JSON serialization for bytes-to-bytes codecs in both V2 and V3 formats."""
    codec = codec_class(**codec_config)

    with pytest.warns(ZarrUserWarning, match=EXPECTED_WARNING_STR):
        # Test V2 serialization
        v2_json = codec.to_json(zarr_format=2)
        assert v2_json == expected_v2

        # Test V3 serialization
        v3_json = codec.to_json(zarr_format=3)
        assert v3_json == expected_v3

    # Test round-trip deserialization
    codec_from_v2 = codec_class.from_json(v2_json)
    codec_from_v3 = codec_class.from_json(v3_json)

    # Compare configs excluding the 'id' field added during serialization
    expected_config_v2 = {k: v for k, v in codec_from_v2.codec_config.items() if k != "id"}
    expected_config_v3 = {k: v for k, v in codec_from_v3.codec_config.items() if k != "id"}

    assert expected_config_v2 == codec.codec_config
    assert expected_config_v3 == codec.codec_config


@pytest.mark.parametrize(
    ("codec_class", "codec_config", "expected_v2", "expected_v3"),
    [
        # Array-to-array codecs
        (
            _numcodecs.Delta,
            {"dtype": "float32", "astype": "int16"},
            {"id": "delta", "dtype": "float32", "astype": "int16"},
            {"name": "delta", "configuration": {"dtype": "float32", "astype": "int16"}},
        ),
        (
            _numcodecs.Delta,
            {"dtype": "float64"},
            {"id": "delta", "dtype": "float64"},
            {"name": "delta", "configuration": {"dtype": "float64"}},
        ),
        (
            _numcodecs.BitRound,
            {"keepbits": 8},
            {"id": "bitround", "keepbits": 8},
            {"name": "bitround", "configuration": {"keepbits": 8}},
        ),
        (
            _numcodecs.FixedScaleOffset,
            {"dtype": "float32", "scale": 100.0, "offset": 10.0, "astype": "uint16"},
            {
                "id": "fixedscaleoffset",
                "dtype": "float32",
                "scale": 100.0,
                "offset": 10.0,
                "astype": "uint16",
            },
            {
                "name": "fixedscaleoffset",
                "configuration": {
                    "dtype": "float32",
                    "scale": 100.0,
                    "offset": 10.0,
                    "astype": "uint16",
                },
            },
        ),
        (
            _numcodecs.FixedScaleOffset,
            {},
            {"id": "fixedscaleoffset"},
            {"name": "fixedscaleoffset", "configuration": {}},
        ),
        (
            _numcodecs.Quantize,
            {"digits": 3, "dtype": "float32"},
            {"id": "quantize", "digits": 3, "dtype": "float32"},
            {"name": "quantize", "configuration": {"digits": 3, "dtype": "float32"}},
        ),
        (
            _numcodecs.Quantize,
            {"digits": 5},
            {"id": "quantize", "digits": 5},
            {"name": "quantize", "configuration": {"digits": 5}},
        ),
        (
            _numcodecs.PackBits,
            {},
            {"id": "packbits"},
            {"name": "packbits", "configuration": {}},
        ),
        (
            _numcodecs.AsType,
            {"encode_dtype": "float32", "decode_dtype": "int32"},
            {"id": "astype", "encode_dtype": "float32", "decode_dtype": "int32"},
            {
                "name": "astype",
                "configuration": {"encode_dtype": "float32", "decode_dtype": "int32"},
            },
        ),
        (
            _numcodecs.AsType,
            {"encode_dtype": "uint8"},
            {"id": "astype", "encode_dtype": "uint8"},
            {"name": "astype", "configuration": {"encode_dtype": "uint8"}},
        ),
    ],
)
def test_array_to_array_codec_json_v2_v3(
    codec_class: type[_numcodecs._NumcodecsArrayArrayCodec],
    codec_config: dict[str, Any],
    expected_v2: dict[str, Any],
    expected_v3: dict[str, Any],
) -> None:
    """Test JSON serialization for array-to-array codecs in both V2 and V3 formats."""
    codec = codec_class(**codec_config)

    # Many codecs emit warnings about unstable specifications
    with pytest.warns(ZarrUserWarning, match=EXPECTED_WARNING_STR):
        # Test V2 serialization
        v2_json = codec.to_json(zarr_format=2)
        assert v2_json == expected_v2

        # Test V3 serialization
        v3_json = codec.to_json(zarr_format=3)
        assert v3_json == expected_v3

    # Test round-trip deserialization
    codec_from_v2 = codec_class.from_json(v2_json)
    codec_from_v3 = codec_class.from_json(v3_json)

    # Compare configs excluding the 'id' field added during serialization
    expected_config_v2 = {k: v for k, v in codec_from_v2.codec_config.items() if k != "id"}
    expected_config_v3 = {k: v for k, v in codec_from_v3.codec_config.items() if k != "id"}

    assert expected_config_v2 == codec.codec_config
    assert expected_config_v3 == codec.codec_config


@pytest.mark.parametrize(
    ("codec_class", "codec_config", "expected_v2", "expected_v3"),
    [
        # Checksum codecs
        (
            _numcodecs.CRC32,
            {"location": "start"},
            {"id": "crc32", "location": "start"},
            {"name": "crc32", "configuration": {"location": "start"}},
        ),
        (
            _numcodecs.CRC32,
            {"location": "end"},
            {"id": "crc32", "location": "end"},
            {"name": "crc32", "configuration": {"location": "end"}},
        ),
        (
            _numcodecs.CRC32,
            {},
            {"id": "crc32"},
            {"name": "crc32", "configuration": {}},
        ),
        (
            _numcodecs.Adler32,
            {"location": "start"},
            {"id": "adler32", "location": "start"},
            {"name": "adler32", "configuration": {"location": "start"}},
        ),
        (
            _numcodecs.Adler32,
            {},
            {"id": "adler32"},
            {"name": "adler32", "configuration": {}},
        ),
        (
            _numcodecs.Fletcher32,
            {},
            {"id": "fletcher32"},
            {"name": "fletcher32", "configuration": {}},
        ),
        (
            _numcodecs.JenkinsLookup3,
            {"initval": 42, "prefix": b"test"},
            {"id": "jenkins_lookup3", "initval": 42, "prefix": b"test"},
            {"name": "jenkins_lookup3", "configuration": {"initval": 42, "prefix": b"test"}},
        ),
        (
            _numcodecs.JenkinsLookup3,
            {"initval": 0},
            {"id": "jenkins_lookup3", "initval": 0},
            {"name": "jenkins_lookup3", "configuration": {"initval": 0}},
        ),
        (
            _numcodecs.JenkinsLookup3,
            {},
            {"id": "jenkins_lookup3"},
            {"name": "jenkins_lookup3", "configuration": {}},
        ),
    ],
)
def test_checksum_codec_json_v2_v3(
    codec_class: type[_numcodecs._NumcodecsChecksumCodec],
    codec_config: dict[str, Any],
    expected_v2: dict[str, Any],
    expected_v3: dict[str, Any],
) -> None:
    """Test JSON serialization for checksum codecs in both V2 and V3 formats."""
    codec = codec_class(**codec_config)

    # Many codecs emit warnings about unstable specifications
    with pytest.warns(ZarrUserWarning, match=EXPECTED_WARNING_STR):
        # Test V2 serialization
        v2_json = codec.to_json(zarr_format=2)
        assert v2_json == expected_v2

        # Test V3 serialization
        v3_json = codec.to_json(zarr_format=3)
        assert v3_json == expected_v3

    # Test round-trip deserialization
    codec_from_v2 = codec_class.from_json(v2_json)
    codec_from_v3 = codec_class.from_json(v3_json)

    # Compare configs excluding the 'id' field added during serialization
    expected_config_v2 = {k: v for k, v in codec_from_v2.codec_config.items() if k != "id"}
    expected_config_v3 = {k: v for k, v in codec_from_v3.codec_config.items() if k != "id"}

    assert expected_config_v2 == codec.codec_config
    assert expected_config_v3 == codec.codec_config


@pytest.mark.parametrize(
    ("codec_class", "codec_config", "expected_v2", "expected_v3"),
    [
        # Array-to-bytes codecs
        (
            _numcodecs.PCodec,
            {"level": 8, "delta_encoding_order": 1},
            {"id": "pcodec", "level": 8, "delta_encoding_order": 1},
            {"name": "pcodec", "configuration": {"level": 8, "delta_encoding_order": 1}},
        ),
        (
            _numcodecs.PCodec,
            {},
            {"id": "pcodec"},
            {"name": "pcodec", "configuration": {}},
        ),
        (
            _numcodecs.ZFPY,
            {"mode": 2, "rate": 16.0, "precision": 20, "tolerance": 0.001},
            {"id": "zfpy", "mode": 2, "rate": 16.0, "precision": 20, "tolerance": 0.001},
            {
                "name": "zfpy",
                "configuration": {"mode": 2, "rate": 16.0, "precision": 20, "tolerance": 0.001},
            },
        ),
        (
            _numcodecs.ZFPY,
            {},
            {"id": "zfpy"},
            {"name": "zfpy", "configuration": {}},
        ),
    ],
)
def test_array_to_bytes_codec_json_v2_v3(
    codec_class: type[_numcodecs._NumcodecsArrayBytesCodec],
    codec_config: dict[str, Any],
    expected_v2: dict[str, Any],
    expected_v3: dict[str, Any],
) -> None:
    """Test JSON serialization for array-to-bytes codecs in both V2 and V3 formats."""
    try:
        codec = codec_class(**codec_config)
        codec._codec  # Try to access the underlying codec to check if it's available
    except (ValueError, ImportError) as e:
        if "codec not available" in str(e) or "not available" in str(e):
            pytest.skip(f"{codec_class.codec_name} is not available: {e}")
        else:
            raise

    # Many codecs emit warnings about unstable specifications
    with pytest.warns(ZarrUserWarning, match=EXPECTED_WARNING_STR):
        # Test V2 serialization
        v2_json = codec.to_json(zarr_format=2)
        assert v2_json == expected_v2

        # Test V3 serialization
        v3_json = codec.to_json(zarr_format=3)
        assert v3_json == expected_v3

    # Test round-trip deserialization
    codec_from_v2 = codec_class.from_json(v2_json)
    codec_from_v3 = codec_class.from_json(v3_json)

    # Compare configs excluding the 'id' field added during serialization
    expected_config_v2 = {k: v for k, v in codec_from_v2.codec_config.items() if k != "id"}
    expected_config_v3 = {k: v for k, v in codec_from_v3.codec_config.items() if k != "id"}

    assert expected_config_v2 == codec.codec_config
    assert expected_config_v3 == codec.codec_config


def test_json_v3_string_format() -> None:
    """Test that V3 codecs can be serialized and deserialized from string format."""
    # Test with a simple codec
    codec = _numcodecs.LZ4()

    # Test string-only V3 format (codec name without configuration)
    v3_string = "lz4"
    codec_from_string = _numcodecs.LZ4.from_json(v3_string)

    # Should be equivalent to default codec
    assert codec_from_string.codec_config == {}
    assert codec_from_string == _numcodecs.LZ4()


def test_json_mixed_format_compatibility() -> None:
    """Test that codecs can deserialize from both V2 and V3 JSON formats."""
    # Test a codec with configuration
    original_codec = _numcodecs.Zlib(level=9)

    # Create both formats
    with pytest.warns(ZarrUserWarning, match=EXPECTED_WARNING_STR):
        v2_json = original_codec.to_json(zarr_format=2)
        v3_json = original_codec.to_json(zarr_format=3)

    # Both should deserialize to the same codec
    codec_from_v2 = _numcodecs.Zlib.from_json(v2_json)
    codec_from_v3 = _numcodecs.Zlib.from_json(v3_json)

    # Compare configs excluding the 'id' field added during serialization
    original_config = original_codec.codec_config
    config_from_v2 = {k: v for k, v in codec_from_v2.codec_config.items() if k != "id"}
    config_from_v3 = {k: v for k, v in codec_from_v3.codec_config.items() if k != "id"}

    assert config_from_v2 == original_config
    assert config_from_v3 == original_config
    assert config_from_v2 == config_from_v3


def test_json_error_handling() -> None:
    """Test error handling for invalid JSON inputs."""
    # Test None input
    with pytest.raises(AttributeError):
        _numcodecs.LZ4.from_json(None)  # type: ignore[arg-type]

    # Test list input (doesn't have get method)
    with pytest.raises(AttributeError):
        _numcodecs.LZ4.from_json([])  # type: ignore[arg-type]


@pytest.mark.filterwarnings("ignore", category=ZarrUserWarning)
@pytest.mark.parametrize(
    ("codec", "expected"),
    [
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
    ],
)
def test_json_roundtrip_default_config(
    codec: _numcodecs._NumcodecsCodec, expected: CodecJSON_V2, zarr_format: ZarrFormat
) -> None:
    """Test JSON roundtrip for all codecs with minimal valid configuration."""
    if zarr_format == 3:
        expected_transformed = codec_json_v2_to_v3(expected)
    else:
        expected_transformed = expected
    assert codec.to_json(zarr_format=zarr_format) == expected_transformed
