from typing import TYPE_CHECKING

import numpy as np
import pytest

import zarr
from tests.test_codecs.conftest import BaseTestCodec
from zarr.abc.store import Store
from zarr.codecs import ZstdCodec
from zarr.storage import StorePath

if TYPE_CHECKING:
    from zarr.codecs.zstd import ZstdJSON_V2, ZstdJSON_V3


class TestZstdCodec(BaseTestCodec):
    test_cls = ZstdCodec
    valid_json_v2 = (
        {  # type: ignore[typeddict-unknown-key]
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


@pytest.mark.parametrize("level", [1, 5, 9])
@pytest.mark.parametrize("checksum", [True, False])
def test_json(level: int, checksum: bool) -> None:
    codec = ZstdCodec(level=level, checksum=checksum)
    expected_v2: ZstdJSON_V2 = {
        "id": "zstd",
        "level": level,
    }
    expected_v3: ZstdJSON_V3 = {
        "name": "zstd",
        "configuration": {"level": level, "checksum": checksum},
    }
    assert codec.to_json(zarr_format=2) == expected_v2
    assert codec.to_json(zarr_format=3) == expected_v3


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
