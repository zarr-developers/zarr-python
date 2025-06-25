import subprocess
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numcodecs
import numpy as np
import pytest
from numcodecs import LZ4, LZMA, Blosc, GZip, VLenBytes, VLenUTF8, Zstd

import zarr
import zarr.abc
import zarr.abc.codec
import zarr.codecs as zarrcodecs
from zarr.core.array import Array
from zarr.core.chunk_key_encodings import V2ChunkKeyEncoding
from zarr.core.dtype.npy.bytes import VariableLengthBytes
from zarr.core.dtype.npy.string import VariableLengthUTF8
from zarr.storage import LocalStore

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
    fill_value: np.generic | str | int | bytes
    filters: tuple[numcodecs.abc.Codec, ...] = ()
    serializer: str | None = None
    compressor: numcodecs.abc.Codec


basic_codecs = GZip(), Blosc(), LZ4(), LZMA(), Zstd()
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
array_cases_v2_18 = (
    basic_array_cases
    + bytes_array_cases
    + datetime_array_cases
    + string_array_cases
    + vlen_string_cases
    + vlen_bytes_cases
)

array_cases_v3_08 = vlen_string_cases


@pytest.fixture
def source_array_v2(tmp_path: Path, request: pytest.FixtureRequest) -> Array:
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
        compressors=compressor,
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
def source_array_v3(tmp_path: Path, request: pytest.FixtureRequest) -> Array:
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
        serializer = zarrcodecs.VLenUTF8Codec()
    elif array_params.values.dtype == np.dtype("|O") and array_params.serializer == "vlen-bytes":
        dtype = VariableLengthBytes()
        serializer = zarrcodecs.VLenBytesCodec()
    else:
        dtype = array_params.values.dtype
        serializer = "auto"
    if array_params.compressor == GZip():
        compressor = zarrcodecs.GzipCodec()
    else:
        msg = (
            "This test is only compatible with gzip compression at the moment, because the author"
            "did not want to implement a complete abstraction layer for v2 and v3 codecs in this test."
        )
        raise ValueError(msg)
    z = zarr.create_array(
        store,
        shape=array_params.values.shape,
        dtype=dtype,
        chunks=array_params.values.shape,
        compressors=compressor,
        filters=array_params.filters,
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
@pytest.mark.parametrize(
    "source_array_v2", array_cases_v2_18, indirect=True, ids=tuple(map(str, array_cases_v2_18))
)
@pytest.mark.parametrize("script_path", script_paths)
def test_roundtrip_v2(source_array_v2: Array, tmp_path: Path, script_path: Path) -> None:
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
@pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
@pytest.mark.parametrize(
    "source_array_v3", array_cases_v3_08, indirect=True, ids=tuple(map(str, array_cases_v3_08))
)
def test_roundtrip_v3(source_array_v3: Array, tmp_path: Path) -> None:
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
