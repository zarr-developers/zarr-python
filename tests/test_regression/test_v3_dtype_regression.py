from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import zarr
from zarr.types import ArrayV3

from .conftest import (
    blosc_cases,
    gzip_cases,
    runner_installed,
    uv_run_cmd,
    vlen_string_cases,
)

array_cases = vlen_string_cases + blosc_cases + gzip_cases


@pytest.mark.skipif(not runner_installed(), reason="no python script runner installed")
@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
@pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
@pytest.mark.parametrize(
    "source_array", [(array_case, 3) for array_case in array_cases], indirect=True, ids=str
)
def test_roundtrip_v3(source_array: ArrayV3, tmp_path: Path) -> None:
    script_path = Path(__file__).resolve().parent / "scripts" / "v3.0.8.py"
    out_path = tmp_path / "out"
    copy_op = uv_run_cmd(
        str(script_path), str(source_array.store).removeprefix("file://"), str(out_path)
    )
    assert copy_op.returncode == 0
    out_array = zarr.open_array(store=out_path, mode="r")
    assert source_array.metadata.to_dict() == out_array.metadata.to_dict()
    assert np.array_equal(source_array[:], out_array[:])
