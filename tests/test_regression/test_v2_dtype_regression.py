import itertools
import subprocess
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pytest
from numcodecs import (
    LZ4,
    LZMA,
    Blosc,
    GZip,
    VLenBytes,
    VLenUTF8,
    Zstd,
)

import zarr
import zarr.abc
import zarr.abc.codec
import zarr.codecs.numcodecs as znumcodecs
from zarr.abc.numcodec import Numcodec
from zarr.codecs import blosc
from zarr.codecs.gzip import GzipCodec
from zarr.codecs.vlen_utf8 import VLenBytesCodec, VLenUTF8Codec
from zarr.core.chunk_key_encodings import V2ChunkKeyEncoding
from zarr.core.dtype.npy.bytes import VariableLengthBytes
from zarr.core.dtype.npy.string import VariableLengthUTF8
from zarr.storage import LocalStore
from zarr.types import ArrayV2, ArrayV3

if TYPE_CHECKING:
    from zarr.core.dtype import ZDTypeLike

ZarrPythonVersion = Literal["2.18", "3.0.8"]


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
    values: np.ndarray[tuple[int], np.dtype[np.generic]]
    fill_value: Any
    filters: tuple[Numcodec | zarr.abc.codec.Codec, ...] = ()
    serializer: str | None = None
    compressor: Numcodec | zarr.abc.codec.Codec | None


basic_codecs: tuple[Numcodec, ...] = GZip(), Blosc(), LZ4(), LZMA(), Zstd()
basic_dtypes = "|b", ">i2", ">i4", ">f4", ">f8", "<f4", "<f8", ">c8", "<c8", ">c16", "<c16"
datetime_dtypes = "<M8[10ns]", ">M8[10us]", "<m8[2ms]", ">m8[4ps]"
string_dtypes = "<U1", ">U4"
bytes_dtypes = ">S1", "<S4", ">V10", "<V4"

basic_array_cases = [
    ArrayParams(values=np.arange(4, dtype=dtype), fill_value=1, compressor=codec)
    for codec, dtype in product(basic_codecs, basic_dtypes)
]
datetime_array_cases = [
    ArrayParams(values=np.ones((4,), dtype=dtype), fill_value=1, compressor=codec)
    for codec, dtype in product(basic_codecs, datetime_dtypes)
]
string_array_cases = [
    ArrayParams(
        values=np.array(["aaaa", "bbbb", "ccccc", "dddd"], dtype=dtype),
        fill_value="foo",
        compressor=codec,
    )
    for codec, dtype in product(basic_codecs, string_dtypes)
]

bytes_array_cases = [
    ArrayParams(
        values=np.array([b"aaaa", b"bbbb", b"ccccc", b"dddd"], dtype=dtype),
        fill_value=b"foo",
        compressor=codec,
    )
    for codec, dtype in product(basic_codecs, bytes_dtypes)
]

vlen_string_cases = [
    ArrayParams(
        values=np.array(["a", "bb", "ccc", "dddd"], dtype="O"),
        fill_value="1",
        serializer="vlen-utf8",
        compressor=GZip(),
    )
]
vlen_bytes_cases = [
    ArrayParams(
        values=np.array([b"a", b"bb", b"ccc", b"dddd"], dtype="O"),
        fill_value=b"1",
        serializer="vlen-bytes",
        compressor=GZip(),
    )
]
# Snappy is not supported by numcodecs yet
zarr_v3_blosc_cases = [
    ArrayParams(
        values=np.arange(4, dtype="float64"),
        fill_value=1,
        compressor=blosc.BloscCodec(clevel=1, shuffle=shuf, cname=cname),  # type: ignore[arg-type]
    )
    for shuf, cname in itertools.product(blosc.SHUFFLE, blosc.CNAME)
    if cname != "snappy"
]

zarr_v3_gzip_cases = [
    ArrayParams(
        values=np.arange(4, dtype="float64"),
        fill_value=1,
        compressor=GzipCodec(level=level),
    )
    for level in [1, 2, 3]
]

# Numcodecs codec test cases - testing all codecs from src/zarr/codecs/numcodecs
# Compression codecs
numcodecs_codec_cases = [
    ArrayParams(
        values=np.arange(100, dtype="float64"),
        fill_value=1.0,
        compressor=znumcodecs.BZ2(level=5),
    ),
    ArrayParams(
        values=np.arange(100, dtype="float64"),
        fill_value=1.0,
        compressor=znumcodecs.Zlib(level=5),
    ),
    ArrayParams(
        values=np.arange(100, dtype="float64"),
        fill_value=1.0,
        filters=(),
        compressor=znumcodecs.Adler32(),
    ),
    ArrayParams(
        values=np.arange(100, dtype="float64"),
        fill_value=1.0,
        filters=(),
        compressor=znumcodecs.CRC32(),
    ),
    ArrayParams(
        values=np.arange(100, dtype="float64"),
        fill_value=1.0,
        filters=(),
        compressor=znumcodecs.Fletcher32(),
    ),
    ArrayParams(
        values=np.arange(100, dtype="float64"),
        fill_value=1.0,
        filters=(),
        compressor=znumcodecs.JenkinsLookup3(),
    ),
    # Array transformation filters
    ArrayParams(
        values=np.arange(100, dtype="float64"),
        fill_value=1.0,
        filters=(znumcodecs.AsType(encode_dtype="<i4", decode_dtype="<f8"),),
        compressor=None,
    ),
    ArrayParams(
        values=np.arange(100, dtype="float64"),
        fill_value=1.0,
        filters=(znumcodecs.BitRound(keepbits=10),),
        compressor=None,
    ),
    ArrayParams(
        values=np.arange(100, dtype="int32"),
        fill_value=1,
        filters=(znumcodecs.Delta(dtype="<i4", astype="<i4"),),
        compressor=None,
    ),
    ArrayParams(
        values=np.arange(100, dtype="float64"),
        fill_value=1.0,
        filters=(znumcodecs.FixedScaleOffset(dtype="<f8", astype="<i4", scale=10.0, offset=100.0),),
        compressor=None,
    ),
    ArrayParams(
        values=np.arange(100, dtype="float64"),
        fill_value=1.0,
        filters=(znumcodecs.Quantize(digits=3, dtype="<f8"),),
        compressor=None,
    ),
    ArrayParams(
        values=np.arange(100, dtype="int32"),
        fill_value=1,
        filters=(),
        compressor=znumcodecs.Shuffle(elementsize=4),
    ),
    ArrayParams(
        values=np.array([True, False, True, False] * 25, dtype=bool),
        fill_value=False,
        filters=(znumcodecs.PackBits(),),
        compressor=None,
    ),
]

array_cases_v2_18 = (
    basic_array_cases
    + bytes_array_cases
    + datetime_array_cases
    + string_array_cases
    + vlen_string_cases
    + vlen_bytes_cases
    + zarr_v3_blosc_cases
    + zarr_v3_gzip_cases
)

array_cases_v3_08 = (
    vlen_string_cases + numcodecs_codec_cases + zarr_v3_blosc_cases + zarr_v3_gzip_cases
)


@pytest.fixture
def source_array_v2(tmp_path: Path, request: pytest.FixtureRequest) -> ArrayV2:
    """
    Writes a zarr array to a temporary directory based on the provided ArrayParams. The array is
    returned.
    """
    dest = tmp_path / "in"
    store = LocalStore(dest)
    array_params: ArrayParams = request.param
    compressor = array_params.compressor
    chunk_key_encoding = V2ChunkKeyEncoding(separator="/")
    dtype: ZDTypeLike
    if array_params.values.dtype == np.dtype("|O") and array_params.serializer == "vlen-utf8":
        dtype = VariableLengthUTF8()  # type: ignore[assignment]
        filters = array_params.filters + (VLenUTF8(),)
    elif array_params.values.dtype == np.dtype("|O") and array_params.serializer == "vlen-bytes":
        dtype = VariableLengthBytes()
        filters = array_params.filters + (VLenBytes(),)
    else:
        dtype = array_params.values.dtype
        filters = array_params.filters
    z = zarr.create_array(
        store,
        shape=array_params.values.shape,
        dtype=dtype,
        chunks=array_params.values.shape,
        compressors=compressor,  # type: ignore[arg-type]
        filters=filters,
        fill_value=array_params.fill_value,
        order="C",
        chunk_key_encoding=chunk_key_encoding,
        write_data=True,
        zarr_format=2,
    )
    z[:] = array_params.values
    return z


@pytest.fixture
def source_array_v3(tmp_path: Path, request: pytest.FixtureRequest) -> ArrayV3:
    """
    Writes a zarr array to a temporary directory based on the provided ArrayParams. The array is
    returned.
    """
    dest = tmp_path / "in"
    store = LocalStore(dest)
    array_params: ArrayParams = request.param
    chunk_key_encoding = V2ChunkKeyEncoding(separator="/")
    dtype: ZDTypeLike
    serializer: Literal["auto"] | zarr.abc.codec.Codec
    if array_params.values.dtype == np.dtype("|O") and array_params.serializer == "vlen-utf8":
        dtype = VariableLengthUTF8()  # type: ignore[assignment]
        serializer = VLenUTF8Codec()
    elif array_params.values.dtype == np.dtype("|O") and array_params.serializer == "vlen-bytes":
        dtype = VariableLengthBytes()
        serializer = VLenBytesCodec()
    else:
        dtype = array_params.values.dtype
        serializer = "auto"
    z = zarr.create_array(
        store,
        shape=array_params.values.shape,
        dtype=dtype,
        chunks=array_params.values.shape,
        compressors=array_params.compressor,  # type: ignore[arg-type]
        filters=array_params.filters,  # type: ignore[arg-type]
        serializer=serializer,
        fill_value=array_params.fill_value,
        chunk_key_encoding=chunk_key_encoding,
        write_data=True,
        zarr_format=3,
    )
    z[:] = array_params.values
    return z


# TODO: make this dynamic based on the installed scripts
script_paths = [Path(__file__).resolve().parent / "scripts" / "v2.18.py"]


@pytest.mark.skipif(not runner_installed(), reason="no python script runner installed")
@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
@pytest.mark.parametrize(
    "source_array_v2", array_cases_v2_18, indirect=True, ids=tuple(map(str, array_cases_v2_18))
)
@pytest.mark.parametrize("script_path", script_paths)
def test_roundtrip_v2(source_array_v2: ArrayV2, tmp_path: Path, script_path: Path) -> None:
    out_path = tmp_path / "out"
    copy_op = subprocess.run(
        [
            "uv",
            "run",
            str(script_path),
            str(source_array_v2.store).removeprefix("file://"),
            str(out_path),
        ],
        capture_output=True,
        text=True,
    )
    assert copy_op.returncode == 0
    out_array = zarr.open_array(store=out_path, mode="r", zarr_format=2)
    assert source_array_v2.metadata.to_dict() == out_array.metadata.to_dict()
    assert np.array_equal(source_array_v2[:], out_array[:])


@pytest.mark.skipif(not runner_installed(), reason="no python script runner installed")
@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
@pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
@pytest.mark.parametrize(
    "source_array_v3", array_cases_v3_08, indirect=True, ids=tuple(map(str, array_cases_v3_08))
)
def test_roundtrip_v3(source_array_v3: ArrayV3, tmp_path: Path) -> None:
    script_path = Path(__file__).resolve().parent / "scripts" / "v3.0.8.py"
    out_path = tmp_path / "out"
    copy_op = subprocess.run(
        [
            "uv",
            "run",
            str(script_path),
            str(source_array_v3.store).removeprefix("file://"),
            str(out_path),
        ],
        capture_output=True,
        text=True,
    )
    assert copy_op.returncode == 0
    out_array = zarr.open_array(store=out_path, mode="r", zarr_format=3)
    assert source_array_v3.metadata.to_dict() == out_array.metadata.to_dict()
    assert np.array_equal(source_array_v3[:], out_array[:])
