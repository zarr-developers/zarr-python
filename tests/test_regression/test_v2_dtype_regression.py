from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import zarr
from tests.test_regression.conftest import runner_installed, uv_run_cmd
from zarr.types import ArrayV2

from .conftest import (
    basic_array_cases,
    blosc_cases,
    bytes_array_cases,
    datetime_array_cases,
    gzip_cases,
    numcodecs_codec_cases,
    string_array_cases,
    vlen_bytes_cases,
    vlen_string_cases,
)

# Numcodecs codec test cases - testing all codecs from src/zarr/codecs/numcodecs
# Compression codecs

array_cases = (
    basic_array_cases
    + bytes_array_cases
    + datetime_array_cases
    + string_array_cases
    + vlen_string_cases
    + vlen_bytes_cases
    + blosc_cases
    + gzip_cases
    + numcodecs_codec_cases
)


@pytest.mark.skipif(not runner_installed(), reason="no python script runner installed")
@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
@pytest.mark.parametrize(
    "source_array", [(array_case, 2) for array_case in array_cases], indirect=True, ids=str
)
def test_roundtrip_v2(source_array: ArrayV2, tmp_path: Path) -> None:
    script_path = Path(__file__).resolve().parent / "scripts" / "v2.18.py"
    out_path = tmp_path / "out"
    copy_op = uv_run_cmd(
        str(script_path), str(source_array.store).removeprefix("file://"), str(out_path)
    )
    assert copy_op.returncode == 0
    out_array = zarr.open_array(store=out_path, mode="r")
    assert source_array.metadata.to_dict() == out_array.metadata.to_dict()
    assert np.array_equal(source_array[:], out_array[:])
