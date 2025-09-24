from typing import Any

import numcodecs
import numpy as np
import pytest

import zarr
import zarr.codecs.numcodecs as znumcodecs
from tests.test_codecs.conftest import BaseTestCodec
from zarr.abc.codec import Codec
from zarr.abc.store import Store
from zarr.codecs import GzipCodec
from zarr.core.common import ZarrFormat
from zarr.core.dtype.npy.int import UInt8
from zarr.core.metadata.common import _parse_codec
from zarr.storage import StorePath


class TestGZipCodec(BaseTestCodec):
    test_cls = GzipCodec
    valid_json_v2 = (
        {  # type: ignore[typeddict-unknown-key]
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


class TestNumcodecsGZipCodec(TestGZipCodec):
    test_cls = znumcodecs.GZip  # type: ignore[assignment]


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"id": "gzip", "level": 1}, GzipCodec(level=1)),
        ({"name": "gzip", "configuration": {"level": 1}}, GzipCodec(level=1)),
        (GzipCodec(level=1), GzipCodec(level=1)),
        (numcodecs.GZip(level=1), GzipCodec(level=1)),
        (zarr.codecs.numcodecs.GZip(level=1), GzipCodec(level=1)),
    ],
)
def test_parse_codec(data: Any, expected: Codec) -> None:
    """
    Ensure that the _parse_codec function properly resolves a codec-like input into the expected output
    """
    # TODO: remove the dtype parameter when we fix the problem with the blosc codec.
    assert _parse_codec(data, dtype=UInt8()) == expected


@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
def test_gzip(store: Store) -> None:
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    a = zarr.create_array(
        StorePath(store),
        shape=data.shape,
        chunks=(16, 16),
        dtype=data.dtype,
        fill_value=0,
        compressors=GzipCodec(),
    )

    a[:, :] = data
    assert np.array_equal(data, a[:, :])


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("codec_cls", [GzipCodec, znumcodecs.GZip, numcodecs.GZip])
def test_gzip_compression(zarr_format: ZarrFormat, codec_cls: Any) -> None:
    store: dict[str, Any] = {}
    z_w = zarr.create_array(
        store=store,
        dtype="int",
        shape=(1,),
        chunks=(10,),
        zarr_format=zarr_format,
        compressors=codec_cls(),
    )
    z_w[:] = 5

    z_r = zarr.open_array(store=store, zarr_format=zarr_format)
    assert np.all(z_r[:] == 5)
