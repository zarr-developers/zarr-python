import zarr.v3.array.v3 as v3
import zarr.v3.array.v2 as v2
import pytest
from typing import Any, Dict, Literal, Tuple, Union
import numpy as np

from zarr.v3.common import ChunkCoords

# todo: parametrize by chunks
@pytest.mark.parametrize("zarr_version", ("2", "3"))
@pytest.mark.parametrize(
    "shape",
    (
        (10,),
        (
            10,
            11,
        ),
        (
            10,
            11,
            12,
        ),
    ),
)
@pytest.mark.parametrize(
    "dtype", (np.dtype("uint8"), "uint8", np.dtype("float32"), "float32", "int64")
)
@pytest.mark.parametrize("attributes", ({}, dict(a=10, b=10)))
@pytest.mark.parametrize("fill_value", (0, 1, 2))
@pytest.mark.parametrize("dimension_separator", (".", "/"))
def test_array(
    tmpdir,
    zarr_version: Literal["2", "3"],
    shape: Tuple[int, ...],
    dtype: Union[str, np.dtype],
    attributes: Dict[str, Any],
    fill_value: float,
    dimension_separator: Literal[".", "/"],
):
    store_path = str(tmpdir)
    arr: Union[v2.Array, v3.Array]
    if zarr_version == "2":
        arr = v2.Array.create(
            store=store_path,
            shape=shape,
            dtype=dtype,
            chunks=shape,
            dimension_separator=dimension_separator,
            fill_value=fill_value,
            attributes=attributes,
            exists_ok=True,
        )
    else:
        arr = v3.Array.create(
            store=store_path,
            shape=shape,
            dtype=dtype,
            chunk_shape=shape,
            fill_value=fill_value,
            attributes=attributes,
            exists_ok=True,
        )
    fill_array = np.zeros(shape, dtype=dtype) + fill_value
    assert np.array_equal(arr[:], fill_array)

    data = np.arange(np.prod(shape)).reshape(shape).astype(dtype)

    # note: if we try to create a prefix called "0/0/0" but an object named "0" already
    # exists in the store, then we will get an unhandled exception
    arr[:] = data
    assert np.array_equal(arr[:], data)

    # partial write
    arr[slice(0, 1)] = data[slice(0, 1)]
