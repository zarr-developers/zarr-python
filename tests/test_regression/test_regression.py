import subprocess
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

import numcodecs
import numpy as np
import pytest
from numcodecs import LZ4, LZMA, Blosc, GZip, VLenBytes, VLenUTF8, Zstd

import zarr
from zarr.core.array import Array
from zarr.core.chunk_key_encodings import V2ChunkKeyEncoding
from zarr.core.dtype.npy.bytes import VariableLengthBytes
from zarr.core.dtype.npy.string import VariableLengthUTF8
from zarr.storage import LocalStore

if TYPE_CHECKING:
    from zarr.core.dtype import ZDTypeLike


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
        filters=(VLenUTF8(),),
        compressor=GZip(),
    )
]
vlen_bytes_cases = [
    ArrayParams(
        values=np.array([b"a", b"bb", b"ccc", b"dddd"], dtype="O"),
        fill_value=b"1",
        filters=(VLenBytes(),),
        compressor=GZip(),
    )
]
array_cases = (
    basic_array_cases
    + bytes_array_cases
    + datetime_array_cases
    + string_array_cases
    + vlen_string_cases
    + vlen_bytes_cases
)


@pytest.fixture
def source_array(tmp_path: Path, request: pytest.FixtureRequest) -> Array:
    dest = tmp_path / "in"
    store = LocalStore(dest)
    array_params: ArrayParams = request.param
    compressor = array_params.compressor
    chunk_key_encoding = V2ChunkKeyEncoding(separator="/")
    dtype: ZDTypeLike
    if array_params.values.dtype == np.dtype("|O") and array_params.filters == (VLenUTF8(),):
        dtype = VariableLengthUTF8()  # type: ignore[assignment]
    elif array_params.values.dtype == np.dtype("|O") and array_params.filters == (VLenBytes(),):
        dtype = VariableLengthBytes()
    else:
        dtype = array_params.values.dtype
    z = zarr.create_array(
        store,
        shape=array_params.values.shape,
        dtype=dtype,
        chunks=array_params.values.shape,
        compressors=compressor,
        filters=array_params.filters,
        fill_value=array_params.fill_value,
        order="C",
        chunk_key_encoding=chunk_key_encoding,
        write_data=True,
        zarr_format=2,
    )
    z[:] = array_params.values
    return z


# TODO: make this dynamic based on the installed scripts
script_paths = [Path(__file__).resolve().parent / "scripts" / "v2.18.py"]


@pytest.mark.skipif(not runner_installed(), reason="no python script runner installed")
@pytest.mark.parametrize(
    "source_array", array_cases, indirect=True, ids=tuple(map(str, array_cases))
)
@pytest.mark.parametrize("script_path", script_paths)
def test_roundtrip(source_array: Array, tmp_path: Path, script_path: Path) -> None:
    out_path = tmp_path / "out"
    copy_op = subprocess.run(
        [
            "uv",
            "run",
            script_path,
            str(source_array.store).removeprefix("file://"),
            str(out_path),
        ],
        capture_output=True,
        text=True,
    )
    assert copy_op.returncode == 0
    out_array = zarr.open_array(store=out_path, mode="r", zarr_format=2)
    assert source_array.metadata.to_dict() == out_array.metadata.to_dict()
    assert np.array_equal(source_array[:], out_array[:])
