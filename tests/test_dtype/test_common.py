from __future__ import annotations

import pytest

from zarr.dtype import check_dtype_spec_no_object_codec_v2


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"name": "|u1", "object_codec_id": None}, True),
        ({"name": [["a", "<f8"]], "object_codec_id": None}, True),
        # an object codec
        ({"name": "|O", "object_codec_id": "vlen-utf8"}, False),
        ({"name": "|u1", "object_codec_id": "vlen-utf8"}, False),
        # not a Zarr V2 data type
        ({"name": "|u1"}, False),
        ({"name": "|u1", "object_codec_id": None, "extra": 1}, False),
        ({"name": 1, "object_codec_id": None}, False),
        ("|u1", False),
    ],
    ids=str,
)
def test_check_dtype_spec_no_object_codec_v2(data: object, expected: bool) -> None:
    assert check_dtype_spec_no_object_codec_v2(data) is expected
