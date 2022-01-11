from numcodecs.abc import Codec
from numcodecs.registry import get_codec, register_codec

from .util import ensure_contiguous_ndarray


class CuPyCPUCompressor(Codec):
    codec_id = "cupy_cpu_compressor"

    def __init__(self, cpu_compressor: Codec = None):
        self.cpu_compressor = cpu_compressor

    def encode(self, buf):
        import cupy

        buf = cupy.asnumpy(ensure_contiguous_ndarray(buf))
        if self.cpu_compressor:
            buf = self.cpu_compressor.encode(buf)
        return buf

    def decode(self, chunk, out=None):
        import cupy

        if out is not None:
            cpu_out = cupy.asnumpy(out)
        else:
            cpu_out = None
        if self.cpu_compressor:
            chunk = self.cpu_compressor.decode(chunk, cpu_out)
            if out is None:
                cpu_out = chunk
        chunk = cupy.asarray(ensure_contiguous_ndarray(chunk))
        if out is not None:
            cupy.copyto(out, chunk.view(dtype=out.dtype), casting="no")
            chunk = out
        return chunk

    def get_config(self):
        cc_config = self.cpu_compressor.get_config() if self.cpu_compressor else None
        return {
            "id": self.codec_id,
            "cpu_compressor_config": cc_config,
        }

    @classmethod
    def from_config(cls, config):
        cc_config = config.get("cpu_compressor_config", None)
        cpu_compressor = get_codec(cc_config) if cc_config else None
        return cls(cpu_compressor=cpu_compressor)


register_codec(CuPyCPUCompressor)
