from __future__ import annotations

import itertools
import subprocess
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import zarr
from zarr import Array
from zarr.codecs.blosc import CNAME, SHUFFLE
from zarr.core.common import CodecJSON_V3, ZarrFormat
from zarr.core.dtype import VariableLengthBytes, VariableLengthUTF8
from zarr.core.dtype.npy.bool import Bool
from zarr.core.dtype.npy.bytes import NullTerminatedBytes, RawBytes
from zarr.core.dtype.npy.complex import Complex64, Complex128
from zarr.core.dtype.npy.float import Float32, Float64
from zarr.core.dtype.npy.int import Int8, Int16, Int32
from zarr.core.dtype.npy.string import FixedLengthUTF32
from zarr.core.dtype.npy.time import DateTime64, TimeDelta64
from zarr.core.dtype.wrapper import ZDType
from zarr.storage import LocalStore

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

    import zarr
    from zarr.core.metadata.v2 import ArrayV2Metadata
    from zarr.core.metadata.v3 import ArrayV3Metadata


def runner_installed() -> bool:
    """
    Check if a PEP-723 compliant python script runner is installed.
    """
    try:
        subprocess.check_output(["uv", "--version"])
        return True  # noqa: TRY300
    except FileNotFoundError:
        return False


@dataclass(kw_only=True)
class ArrayParams:
    """
    We use this class to initialize array data for the regression tests, which rely on testing the
    ability to round-trip Zarr data through other zarr tools.
    """

    values: np.ndarray[tuple[int], np.dtype[np.generic]]
    fill_value: object
    dtype: ZDType[Any, Any]
    filters: tuple[CodecJSON_V3, ...] = ()
    serializer: CodecJSON_V3 | None = None
    compressor: CodecJSON_V3 | None = None


def uv_run_cmd(
    script_path: str,
    source_path: str,
    dest_path: str,
) -> subprocess.CompletedProcess[str]:
    """
    Use uv to run the script at script_path with source_path as its first argument and
    dest_path as its second argument.
    """
    return subprocess.run(
        [
            "uv",
            "run",
            script_path,
            source_path,
            dest_path,
        ],
        capture_output=True,
        text=True,
    )


@pytest.fixture
def source_array(
    tmp_path: Path, request: pytest.FixtureRequest
) -> Array[ArrayV3Metadata] | Array[ArrayV2Metadata]:
    """
    Create an array based on the provided ArrayParams and Zarr format.
    The array will be stored at tmp_path / "in" and returned.
    """
    dest = tmp_path / "in"

    store = LocalStore(dest)
    array_params: ArrayParams
    zarr_format: ZarrFormat
    array_params, zarr_format = request.param
    compressor = array_params.compressor
    filters = array_params.filters
    serializer = array_params.serializer

    z = zarr.create_array(
        store,
        dtype=array_params.dtype,
        shape=array_params.values.shape,
        chunks=array_params.values.shape,
        compressors=compressor,
        filters=filters,
        serializer=serializer,
        fill_value=array_params.fill_value,
        zarr_format=zarr_format,
        chunk_key_encoding={"name": "v2", "configuration": {"separator": "/"}},
    )
    z[:] = array_params.values
    return z


default_array_params = ArrayParams(
    values=np.arange(10, dtype="float64"),
    dtype=Float64(),
    fill_value=1.0,
    compressor=None,
    serializer="bytes",
    filters=(),
)

basic_codecs: tuple[CodecJSON_V3, ...] = (
    {"name": "lz4", "configuration": {}},
    {"name": "lzma", "configuration": {}},
    {"name": "zstd", "configuration": {"level": 1, "checksum": False}},
)

basic_dtypes = (
    Bool(),
    Int8(),
    Int16(endianness="big"),
    Int32(),
    Float32(),
    Float32(endianness="big"),
    Float64(endianness="little"),
    Float64(endianness="big"),
    Complex64(endianness="big"),
    Complex64(endianness="little"),
    Complex128(endianness="little"),
    Complex128(endianness="big"),
)
datetime_dtypes = (
    DateTime64(endianness="little", unit="ns", scale_factor=10),
    DateTime64(endianness="big", unit="us", scale_factor=10),
    TimeDelta64(endianness="little", unit="ms", scale_factor=2),
    TimeDelta64(endianness="big", unit="ps", scale_factor=4),
)
string_dtypes = (
    FixedLengthUTF32(length=1, endianness="little"),
    FixedLengthUTF32(length=4, endianness="big"),
)
bytes_dtypes = (NullTerminatedBytes(length=1), NullTerminatedBytes(length=4), RawBytes(length=10))

basic_array_cases = [
    replace(
        default_array_params,
        dtype=dtype,
        values=np.arange(10).view(dtype=dtype.to_native_dtype()),
        compressor=codec,
    )
    for codec, dtype in itertools.product(basic_codecs, basic_dtypes)
]
datetime_array_cases = [
    replace(
        default_array_params,
        values=np.ones((4,), dtype=dtype.to_native_dtype()),
        dtype=dtype,
        fill_value=1,
        compressor=codec,
    )
    for codec, dtype in itertools.product(basic_codecs, datetime_dtypes)
]
string_array_cases = [
    ArrayParams(
        values=np.array(["aaaa", "bbbb", "ccccc", "dddd"], dtype=dtype.to_native_dtype()),
        dtype=dtype,
        fill_value="foo",
        compressor=codec,
    )
    for codec, dtype in itertools.product(basic_codecs, string_dtypes)
]

bytes_array_cases = [
    ArrayParams(
        values=np.array([b"aaaa", b"bbbb", b"ccccc", b"dddd"], dtype=dtype.to_native_dtype()),
        dtype=dtype,
        fill_value=b"foo",
        compressor=codec,
    )
    for codec, dtype in itertools.product(basic_codecs, bytes_dtypes)
]

vlen_string_cases = [
    ArrayParams(
        values=np.array(["a", "bb", "ccc", "dddd"], dtype="O"),
        dtype=VariableLengthUTF8(),
        fill_value="1",
        serializer="vlen-utf8",
        compressor={"name": "gzip", "configuration": {"level": 1}},
    )
]
vlen_bytes_cases = [
    ArrayParams(
        dtype=VariableLengthBytes(),
        values=np.array([b"a", b"bb", b"ccc", b"dddd"], dtype="O"),
        fill_value=b"1",
        serializer="vlen-bytes",
        compressor={"name": "gzip", "configuration": {"level": 1}},
    )
]

# Snappy is not supported by numcodecs yet
blosc_cases = [
    replace(
        default_array_params,
        compressor={
            "name": "blosc",
            "configuration": {
                "clevel": 1,
                "shuffle": shuf,
                "cname": cname,
                "typesize": 1,
                "blocksize": 1,
            },
        },
    )
    for shuf, cname in itertools.product(SHUFFLE, CNAME)
    if cname != "snappy"
]

gzip_cases = [
    replace(
        default_array_params,
        compressor={"name": "gzip", "configuration": {"level": level}},
    )
    for level in [1, 2, 3]
]

numcodecs_codec_cases = [
    replace(
        default_array_params,
        compressor={"name": "bz2", "configuration": {"level": 5}},
    ),
    replace(
        default_array_params,
        compressor={"name": "zlib", "configuration": {"level": 5}},
    ),
    replace(
        default_array_params,
        compressor={"name": "adler32", "configuration": {"location": "end"}},
    ),
    replace(
        default_array_params,
        compressor={"name": "crc32", "configuration": {"location": "end"}},
    ),
    replace(
        default_array_params,
        compressor="fletcher32",
    ),
    replace(
        default_array_params,
        compressor={"name": "jenkins_lookup3", "configuration": {"initval": 1, "prefix": None}},
    ),
    # Array transformation filters
    replace(
        default_array_params,
        filters=(
            {
                "name": "astype",
                "configuration": {"encode_dtype": "int32", "decode_dtype": "float64"},
            },
        ),
        compressor=None,
    ),
    replace(
        default_array_params,
        filters=({"name": "bitround", "configuration": {"keepbits": 10}},),
        compressor=None,
    ),
    ArrayParams(
        values=np.arange(25, dtype="int32"),
        dtype=Int32(),
        fill_value=1,
        filters=({"name": "delta", "configuration": {"dtype": "int32", "astype": "int32"}},),
        compressor=None,
    ),
    replace(
        default_array_params,
        filters=(
            {
                "name": "scale_offset",
                "configuration": {
                    "dtype": "float64",
                    "astype": "int32",
                    "scale": 10,
                    "offset": 100,
                },
            },
        ),
        compressor=None,
    ),
    replace(
        default_array_params,
        filters=({"name": "quantize", "configuration": {"digits": 3, "dtype": "float64"}},),
        compressor=None,
    ),
    ArrayParams(
        values=np.arange(25, dtype="int32"),
        dtype=Int32(),
        fill_value=1,
        filters=(),
        compressor={"name": "shuffle", "configuration": {"elementsize": 4}},
    ),
    ArrayParams(
        values=np.array([True, False, True, False] * 5, dtype=bool),
        dtype=Bool(),
        fill_value=False,
        filters=("packbits",),
        compressor=None,
    ),
]
