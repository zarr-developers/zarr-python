from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numcodecs
import numpy as np
import pytest

import zarr
import zarr.codecs.numcodecs
from tests.test_codecs.conftest import BaseTestCodec
from zarr.codecs import ZstdCodec
from zarr.codecs.zstd import check_json_v2, check_json_v3
from zarr.storage import StorePath

if TYPE_CHECKING:
    from zarr.abc.store import Store
    from zarr.codecs.zstd import ZstdJSON_V2, ZstdJSON_V3
    from zarr.core.common import ZarrFormat


class TestZstdCodec(BaseTestCodec):
    test_cls = ZstdCodec
    valid_json_v2 = (
        {
            "id": "zstd",
            "level": 0,
        },
    )
    valid_json_v3 = (
        {
            "name": "zstd",
            "configuration": {
                "level": 0,
                "checksum": False,
            },
        },
    )

    @staticmethod
    def check_json_v2(data: object) -> bool:
        return check_json_v2(data)

    @staticmethod
    def check_json_v3(data: object) -> bool:
        return check_json_v3(data)

    def test_checksum_removed(self) -> None:
        """
        Test that the checksum field is not serialized to Zarr V2 JSON when it is False
        """
        codec = self.test_cls(checksum=False)
        assert "checksum" not in codec.to_json(zarr_format=2)

        codec = self.test_cls(checksum=True)
        assert codec.to_json(zarr_format=2)["checksum"] is True


@pytest.mark.parametrize("level", [1, 5, 9])
@pytest.mark.parametrize("checksum", [True, False])
def test_json(level: int, checksum: bool) -> None:
    codec = ZstdCodec(level=level, checksum=checksum)
    expected_v2: ZstdJSON_V2

    if checksum:
        expected_v2 = {"id": "zstd", "level": level, "checksum": True}
    else:
        expected_v2 = {
            "id": "zstd",
            "level": level,
        }
    expected_v3: ZstdJSON_V3 = {
        "name": "zstd",
        "configuration": {"level": level, "checksum": checksum},
    }
    assert codec.to_json(zarr_format=2) == expected_v2
    assert codec.to_json(zarr_format=3) == expected_v3


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize(
    "codec",
    [
        ZstdCodec(level=1, checksum=False),
        zarr.codecs.numcodecs.Zstd(level=1, checksum=False),
        numcodecs.Zstd(level=1, checksum=False),
    ],
)
def test_zstd_compression(zarr_format: ZarrFormat, codec: Any) -> None:
    """
    Test that any of the zstd-like codecs can be used for compression, and that
    reading the array back uses the primary zstd codec class.
    """
    ref_codec = ZstdCodec(level=1, checksum=False)
    store: dict[str, Any] = {}
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


@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
@pytest.mark.parametrize("checksum", [True, False])
def test_zstd(store: Store, checksum: bool) -> None:
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    a = zarr.create_array(
        StorePath(store, path="zstd"),
        shape=data.shape,
        chunks=(16, 16),
        dtype=data.dtype,
        fill_value=0,
        compressors=ZstdCodec(level=0, checksum=checksum),
    )

    a[:, :] = data
    assert np.array_equal(data, a[:, :])


def test_v3_json_alias() -> None:
    from zarr.codecs import numcodecs as _numcodecs

    """
    Test that the default JSON output of the legacy numcodecs.zarr3.Zstd codec is readable, even if it's
    underspecified.
    """
    assert _numcodecs.Zstd.from_json(
        {"name": "numcodecs.zstd", "configuration": {}}
    ) == _numcodecs.Zstd(level=0, checksum=False)
