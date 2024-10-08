"""Tests for the strings module."""

import numpy as np
import pytest

from zarr.core.strings import _NUMPY_SUPPORTS_VLEN_STRING, _STRING_DTYPE, cast_to_string_dtype


def test_string_defaults() -> None:
    if _NUMPY_SUPPORTS_VLEN_STRING:
        assert _STRING_DTYPE == np.dtypes.StringDType()
    else:
        assert _STRING_DTYPE == np.dtypes.ObjectDType()


def test_cast_to_string_dtype() -> None:
    d1 = np.array(["a", "b", "c"])
    assert d1.dtype == np.dtype("<U1")
    d1s = cast_to_string_dtype(d1)
    assert d1s.dtype == _STRING_DTYPE

    with pytest.raises(ValueError, match="Cannot cast dtype |S1"):
        cast_to_string_dtype(d1.astype("|S1"))

    if _NUMPY_SUPPORTS_VLEN_STRING:
        assert cast_to_string_dtype(d1.astype("T")).dtype == _STRING_DTYPE
        assert cast_to_string_dtype(d1.astype("O")).dtype == _STRING_DTYPE
        with pytest.raises(ValueError, match="Cannot cast object dtype to string dtype"):
            cast_to_string_dtype(np.array([1, "b", "c"], dtype="O"))
    else:
        with pytest.warns():
            assert cast_to_string_dtype(d1.astype("O")).dtype == _STRING_DTYPE
        with pytest.warns():
            assert cast_to_string_dtype(np.array([1, "b", "c"], dtype="O")).dtype == _STRING_DTYPE
        assert cast_to_string_dtype(d1.astype("O"), safe=True).dtype == _STRING_DTYPE
