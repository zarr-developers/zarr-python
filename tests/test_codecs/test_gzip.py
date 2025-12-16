from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numcodecs
import numpy as np
import pytest

import zarr
import zarr.codecs.numcodecs as znumcodecs
from tests.test_codecs.conftest import BaseTestCodec
from zarr.codecs import GzipCodec
from zarr.core.dtype.npy.int import UInt8
from zarr.core.metadata.common import _parse_codec
from zarr.errors import ZarrDeprecationWarning

if TYPE_CHECKING:
    from zarr.abc.codec import Codec
    from zarr.core.common import ZarrFormat


class TestGZipCodec(BaseTestCodec):
    test_cls = GzipCodec
    valid_json_v2 = (
        {
            "id": "gzip",
            "level": 1,
        },
    )
    valid_json_v3 = (
        {
            "name": "gzip",
            "configuration": {
                "level": 1,
            },
        },
    )

    @staticmethod
    def check_json_v2(data: object) -> bool:
        return GzipCodec._check_json_v2(data)

    @staticmethod
    def check_json_v3(data: object) -> bool:
        return GzipCodec._check_json_v3(data)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"id": "gzip", "level": 1}, GzipCodec(level=1)),
        ({"name": "gzip", "configuration": {"level": 1}}, GzipCodec(level=1)),
        (GzipCodec(level=1), GzipCodec(level=1)),
        (numcodecs.GZip(level=1), GzipCodec(level=1)),
        ("zarr3_gzip", GzipCodec(level=1)),
    ],
)
def test_parse_codec(data: Any, expected: Codec) -> None:
    """
    Ensure that the _parse_codec function properly resolves a codec-like input into the expected output
    """
    if data == "zarr3_gzip":
        with pytest.warns(ZarrDeprecationWarning):
            data = znumcodecs.GZip(level=1)
    assert _parse_codec(data, dtype=UInt8()) == expected


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("codec_type", ["legacy_zarr3", "numcodecs"])
def test_gzip_compression(
    zarr_format: ZarrFormat, codec_type: Literal["legacy_zarr3", "numcodecs"]
) -> None:
    """
    Test that any of the gzip-like codecs can be used for compression, and that
    reading the array back uses the primary gzip codec class.
    """
    store: dict[str, Any] = {}
    ref_codec = GzipCodec(level=1)

    if codec_type == "legacy_zarr3":
        with pytest.warns(ZarrDeprecationWarning):
            codec = znumcodecs.GZip(level=1)
    else:
        codec = numcodecs.GZip(level=1)

    z_w = zarr.create_array(
        store=store,
        dtype="int",
        shape=(1,),
        chunks=(10,),
        zarr_format=zarr_format,
        compressors=codec,
    )
    z_w[:] = 5

    z_r = zarr.open_array(store=store, zarr_format=zarr_format)
    assert np.all(z_r[:] == 5)
    assert z_r.compressors == (ref_codec,)


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrDeprecationWarning")
def test_v3_json_alias() -> None:
    from zarr.codecs import numcodecs as _numcodecs

    """
    Test that the default JSON output of the legacy numcodecs.zarr3.GZip codec is readable, even if it's
    underspecified.
    """
    assert _numcodecs.GZip.from_json(
        {"name": "numcodecs.gzip", "configuration": {}}
    ) == _numcodecs.GZip(level=1)

    assert _numcodecs.GZip.from_json({"name": "gzip", "configuration": {}}) == _numcodecs.GZip(
        level=1
    )
