import subprocess
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numcodecs
import numpy as np
import pytest
from numcodecs import LZ4, LZMA, Blosc, GZip, VLenUTF8, Zstd

import zarr
from zarr.core.array import Array
from zarr.core.dtype.npy.string import VariableLengthString
from zarr.core.metadata.v2 import ArrayV2Metadata
from zarr.storage import LocalStore


def runner_installed() -> bool:
    try:
        subprocess.check_output(["uv", "--version"])
        return True
    except FileNotFoundError:
        return False


def array_metadata_equals(a: ArrayV2Metadata, b: ArrayV2Metadata) -> bool:
    dict_a, dict_b = asdict(a), asdict(b)
    fill_value_a, fill_value_b = dict_a.pop("fill_value"), dict_b.pop("fill_value")
    if (
        isinstance(fill_value_a, float)
        and isinstance(fill_value_b, float)
        and np.isnan(fill_value_a)
        and np.isnan(fill_value_b)
    ):
        return dict_a == dict_b
    else:
        return fill_value_a == fill_value_b and dict_a == dict_b


@dataclass(kw_only=True)
class ArrayParams:
    values: np.ndarray[tuple[int], np.dtype[np.generic]]
    fill_value: np.generic | str
    compressor: numcodecs.abc.Codec


basic_codecs = GZip(), Blosc(), LZ4(), LZMA(), Zstd()
basic_dtypes = "|b", ">i2", ">i4", ">f4", ">f8", "<f4", "<f8", ">c8", "<c8", ">c16", "<c16"
datetime_dtypes = "<M8[10ns]", ">M8[10us]", "<m8[2ms]", ">m8[4ps]"
string_dtypes = ">S1", "<S4", "<U1", ">U4"

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
vlen_string_cases = [
    ArrayParams(
        values=np.array(["a", "bb", "ccc", "dddd"], dtype="O"),
        fill_value="1",
        compressor=VLenUTF8(),
    )
]
array_cases = basic_array_cases + datetime_array_cases + string_array_cases + vlen_string_cases


@pytest.fixture
def source_array(tmp_path: Path, request: pytest.FixtureRequest) -> Array:
    dest = tmp_path / "in"
    store = LocalStore(dest)
    array_params: ArrayParams = request.param
    compressor = array_params.compressor
    if array_params.values.dtype == np.dtype("|O"):
        dtype = VariableLengthString()
    else:
        dtype = array_params.values.dtype
    z = zarr.create_array(
        store,
        shape=array_params.values.shape,
        dtype=dtype,
        chunks=array_params.values.shape,
        compressors=compressor,
        fill_value=array_params.fill_value,
        order="C",
        filters=None,
        chunk_key_encoding={"name": "v2", "configuration": {"separator": "/"}},
        write_data=True,
        zarr_format=2,
    )
    z[:] = array_params.values
    return z


@pytest.mark.skipif(not runner_installed(), reason="no python script runner installed")
@pytest.mark.parametrize(
    "source_array", array_cases, indirect=True, ids=tuple(map(str, array_cases))
)
def test_roundtrip(source_array: Array, tmp_path: Path) -> None:
    out_path = tmp_path / "out"
    copy_op = subprocess.run(
        [
            "uv",
            "run",
            Path(__file__).resolve().parent / "v2.18.py",
            str(source_array.store).removeprefix("file://"),
            str(out_path),
        ],
        capture_output=True,
        text=True,
    )
    assert copy_op.returncode == 0
    out_array = zarr.open_array(store=out_path, mode="r", zarr_format=2)
    assert array_metadata_equals(source_array.metadata, out_array.metadata)
    assert np.array_equal(source_array[:], out_array[:])
