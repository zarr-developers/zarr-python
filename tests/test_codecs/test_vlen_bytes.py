from __future__ import annotations

from tests.test_codecs.conftest import BaseTestCodec
from zarr.codecs.vlen_utf8 import VLenBytesCodec, VLenUTF8Codec, VLenUTF8JSON_V2, VLenUTF8JSON_V3


class TestVLenBytesCodec(BaseTestCodec):
    test_cls = VLenBytesCodec
    valid_json_v2 = ({"id": "vlen-bytes"},)
    valid_json_v3 = ({"name": "vlen-bytes"}, "vlen-bytes")

    @staticmethod
    def check_json_v2(data: object) -> bool:
        return VLenBytesCodec._check_json_v2(data)

    @staticmethod
    def check_json_v3(data: object) -> bool:
        return VLenBytesCodec._check_json_v3(data)


def test_vlen_utf8_to_json() -> None:
    codec = VLenUTF8Codec()
    expected_v2: VLenUTF8JSON_V2 = {"id": "vlen-utf8"}
    expected_v3: VLenUTF8JSON_V3 = {
        "name": "vlen-utf8",
    }
    assert codec.to_json(zarr_format=2) == expected_v2
    assert codec.to_json(zarr_format=3) == expected_v3
