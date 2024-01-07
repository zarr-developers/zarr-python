import zarr.v3.array.v3 as v3
import zarr.v3.array.v2 as v2
import pytest
from typing import Any, Dict, Literal, Tuple, Union
import numpy as np

from zarr.v3.types import Attributes, ChunkCoords
from zarr.v3.metadata.v3 import DefaultChunkKeyEncoding, RegularChunkGrid, RegularChunkGridConfig

# todo: parametrize by chunks
@pytest.mark.asyncio
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
async def test_array(
    tmpdir,
    zarr_version: Literal["2", "3"],
    shape: Tuple[int, ...],
    dtype: Union[str, np.dtype],
    attributes: Attributes,
    fill_value: float,
    dimension_separator: Literal[".", "/"],
):
    store_path = str(tmpdir)
    arr: Union[v2.AsyncArray, v3.Array]
    if zarr_version == "2":
        arr = await v2.AsyncArray.create(
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
        arr = await v3.AsyncArray.create(
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


@pytest.mark.parametrize("zarr_format", (2, 3))
def test_init_format(zarr_format: Literal[2, 3]):
    dtype = "uint8"
    shape = (10,)
    if zarr_format == 2:
        with pytest.raises(ValueError):
            arr1 = v2.ArrayMetadata(shape=shape, dtype=dtype, chunks=shape, zarr_format=3)
    else:
        with pytest.raises(ValueError):
            arr2 = v3.ArrayMetadata(
                shape=shape,
                data_type=dtype,
                codecs=[],
                chunk_grid=RegularChunkGrid(
                    configuration=RegularChunkGridConfig(chunk_shape=shape)
                ),
                fill_value=0,
                chunk_key_encoding=DefaultChunkKeyEncoding(),
                zarr_format=2,
            )


@pytest.mark.parametrize("zarr_format", ("2", "3"))
def test_init_node_type(zarr_format: Literal["2", "3"]):
    dtype = "uint8"
    shape = (10,)
    if zarr_format == 2:
        with pytest.raises(ValueError):
            arr = v2.ArrayMetadata(shape=shape, dtype=dtype, chunks=shape, node_type="group")
    else:
        with pytest.raises(ValueError):
            arr = v3.ArrayMetadata(
                shape=shape,
                data_type=dtype,
                codecs=[],
                chunk_grid=RegularChunkGrid(
                    configuration=RegularChunkGridConfig(chunk_shape=shape)
                ),
                fill_value=0,
                chunk_key_encoding=DefaultChunkKeyEncoding(),
                node_type="group",
            )
