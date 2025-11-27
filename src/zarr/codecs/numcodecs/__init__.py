from __future__ import annotations

from typing import Final

from zarr.codecs.numcodecs._codecs import (
    _NumcodecsArrayArrayCodec,
    _NumcodecsArrayBytesCodec,
    _NumcodecsBytesBytesCodec,
    _NumcodecsCodec,
)
from zarr.codecs.numcodecs.adler32 import Adler32
from zarr.codecs.numcodecs.astype import AsType
from zarr.codecs.numcodecs.bitround import BitRound
from zarr.codecs.numcodecs.blosc import Blosc
from zarr.codecs.numcodecs.bz2 import BZ2
from zarr.codecs.numcodecs.crc32 import CRC32
from zarr.codecs.numcodecs.crc32c import CRC32C
from zarr.codecs.numcodecs.delta import Delta
from zarr.codecs.numcodecs.fixed_scale_offset import ScaleOffset
from zarr.codecs.numcodecs.fletcher32 import Fletcher32
from zarr.codecs.numcodecs.gzip import GZip
from zarr.codecs.numcodecs.jenkins_lookup3 import JenkinsLookup3
from zarr.codecs.numcodecs.lz4 import LZ4
from zarr.codecs.numcodecs.lzma import LZMA
from zarr.codecs.numcodecs.packbits import PackBits
from zarr.codecs.numcodecs.pcodec import PCodec
from zarr.codecs.numcodecs.quantize import Quantize
from zarr.codecs.numcodecs.shuffle import Shuffle
from zarr.codecs.numcodecs.zfpy import ZFPY
from zarr.codecs.numcodecs.zlib import Zlib
from zarr.codecs.numcodecs.zstd import Zstd

# This is a fixed dictionary of numcodecs codecs for which we have pre-made Zarr V3 wrappers
NUMCODECS_WRAPPERS: Final[dict[str, type[_NumcodecsCodec]]] = {
    "bz2": BZ2,
    "crc32": CRC32,
    "crc32c": CRC32C,
    "lz4": LZ4,
    "lzma": LZMA,
    "zfpy": ZFPY,
    "adler32": Adler32,
    "astype": AsType,
    "bitround": BitRound,
    "blosc": Blosc,
    "delta": Delta,
    "fixedscaleoffset": ScaleOffset,
    "fletcher32": Fletcher32,
    "gzip": GZip,
    "jenkins_lookup3": JenkinsLookup3,
    "packbits": PackBits,
    "pcodec": PCodec,
    "quantize": Quantize,
    "shuffle": Shuffle,
    "zlib": Zlib,
    "zstd": Zstd,
}

__all__ = [
    "BZ2",
    "CRC32",
    "CRC32C",
    "LZ4",
    "LZMA",
    "ZFPY",
    "Adler32",
    "AsType",
    "BitRound",
    "Blosc",
    "Delta",
    "Fletcher32",
    "GZip",
    "JenkinsLookup3",
    "PCodec",
    "PackBits",
    "Quantize",
    "ScaleOffset",
    "Shuffle",
    "Zlib",
    "Zstd",
    "_NumcodecsArrayArrayCodec",
    "_NumcodecsArrayBytesCodec",
    "_NumcodecsBytesBytesCodec",
    "_NumcodecsCodec",
]
