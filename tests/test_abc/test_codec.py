from __future__ import annotations

from zarr.core.common import check_codecjson_v2


def test_check_codecjson_v2_valid() -> None:
    """
    Test that the _check_codecjson_v2 function works
    """
    assert check_codecjson_v2({"id": "gzip"})
    assert not check_codecjson_v2({"id": 10})
    assert not check_codecjson_v2([10, 11])
