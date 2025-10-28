import pytest

from tests.test_codecs.conftest import BaseTestCodec
from zarr.codecs import numcodecs as _numcodecs

pytest.importorskip("zfpy")


class TestZFPYCodec(BaseTestCodec):
    test_cls = _numcodecs.ZFPY
    valid_json_v2 = (
        {
            "id": "zfpy",
            "mode": 4,
            "compression_kwargs": {"tolerance": -1},
            "precision": -1,
            "rate": -1,
            "tolerance": -1,
        },
    )
    valid_json_v3 = (
        {
            "name": "zfpy",
            "configuration": {
                "mode": 4,
                "compression_kwargs": {"tolerance": -1},
                "precision": -1,
                "rate": -1,
                "tolerance": -1,
            },
        },
        {
            "name": "numcodecs.zfpy",
            "configuration": {
                "mode": 4,
                "compression_kwargs": {"tolerance": -1},
                "precision": -1,
                "rate": -1,
                "tolerance": -1,
            },
        },
    )


def test_v3_json_alias() -> None:
    from zarr.codecs import numcodecs as _numcodecs

    """
    Test that the default JSON output of the legacy numcodecs.zarr3.ZFPY codec is readable, even if it's
    underspecified.
    """
    assert _numcodecs.ZFPY.from_json(
        {"name": "numcodecs.zfpy", "configuration": {}}
    ) == _numcodecs.ZFPY(mode=4, tolerance=-1, rate=-1, precision=-1)
