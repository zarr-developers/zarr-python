import string

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import zarr

_DEFAULT_TEST_DIM_SIZES = (8, 9, 10)


@pytest.fixture
def store() -> zarr.abc.store.Store:
    return zarr.storage.MemoryStore()


@pytest.fixture
def dataset(
    seed: int = 12345,
    add_attrs: bool = True,
    dim_sizes: tuple[int, int, int] = _DEFAULT_TEST_DIM_SIZES,
    use_extension_array: bool = False,
) -> xr.Dataset:
    rs = np.random.default_rng(seed)
    _vars = {
        "var1": ["dim1", "dim2"],
        "var2": ["dim1", "dim2"],
        "var3": ["dim3", "dim1"],
    }
    _dims = {"dim1": dim_sizes[0], "dim2": dim_sizes[1], "dim3": dim_sizes[2]}

    obj = xr.Dataset()
    obj["dim2"] = ("dim2", 0.5 * np.arange(_dims["dim2"]))
    if _dims["dim3"] > 26:
        raise RuntimeError(f'Not enough letters for filling this dimension size ({_dims["dim3"]})')
    obj["dim3"] = ("dim3", list(string.ascii_lowercase[0 : _dims["dim3"]]))
    obj["time"] = ("time", pd.date_range("2000-01-01", periods=20))
    for v, dims in sorted(_vars.items()):
        data = rs.normal(size=tuple(_dims[d] for d in dims))
        obj[v] = (dims, data)
        if add_attrs:
            obj[v].attrs = {"foo": "variable"}
    if use_extension_array:
        obj["var4"] = (
            "dim1",
            pd.Categorical(
                rs.choice(
                    list(string.ascii_lowercase[: rs.integers(1, 5)]),
                    size=dim_sizes[0],
                )
            ),
        )
    if dim_sizes == _DEFAULT_TEST_DIM_SIZES:
        numbers_values = np.array([0, 1, 2, 0, 0, 1, 1, 2, 2, 3], dtype="int64")
    else:
        numbers_values = rs.integers(0, 3, _dims["dim3"], dtype="int64")
    obj.coords["numbers"] = ("dim3", numbers_values)
    obj.encoding = {"foo": "bar"}
    return obj


def test_roundtrip(store: zarr.abc.store.Store, dataset: xr.Dataset) -> None:
    dataset.to_zarr(store)
    other_dataset = xr.open_dataset(store)
    assert dataset.identical(other_dataset)
