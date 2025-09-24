from tests.test_codecs.test_zstd import TestZstdCodec
from zarr.codecs import numcodecs as znumcodecs


class TestNumcodecsZstdCodec(TestZstdCodec):
    test_cls = znumcodecs.Zstd  # type: ignore[assignment]
