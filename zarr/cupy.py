from numcodecs.abc import Codec
from numcodecs.compat import ensure_contiguous_ndarray_like
from numcodecs.registry import get_codec, register_codec


class CuPyCPUCompressor(Codec):
    """CPU compressor for CuPy arrays

    This compressor converts CuPy arrays host memory before compressing
    the arrays using `compressor`.

    Parameters
    ----------
    compressor : numcodecs.abc.Codec
        The codec to use for compression and decompression.
    """

    codec_id = "cupy_cpu_compressor"

    def __init__(self, compressor: Codec = None):
        self.compressor = compressor

    def encode(self, buf):
        import cupy

        buf = cupy.asnumpy(ensure_contiguous_ndarray_like(buf))
        if self.compressor:
            buf = self.compressor.encode(buf)
        return buf

    def decode(self, chunk, out=None):
        import cupy

        if self.compressor:
            cpu_out = None if out is None else cupy.asnumpy(out)
            chunk = self.compressor.decode(chunk, cpu_out)

        chunk = cupy.asarray(ensure_contiguous_ndarray_like(chunk))
        if out is not None:
            cupy.copyto(out, chunk.view(dtype=out.dtype), casting="no")
            chunk = out
        return chunk

    def get_config(self):
        cc_config = self.compressor.get_config() if self.compressor else None
        return {
            "id": self.codec_id,
            "compressor_config": cc_config,
        }

    @classmethod
    def from_config(cls, config):
        cc_config = config.get("compressor_config", None)
        compressor = get_codec(cc_config) if cc_config else None
        return cls(compressor=compressor)


register_codec(CuPyCPUCompressor)
