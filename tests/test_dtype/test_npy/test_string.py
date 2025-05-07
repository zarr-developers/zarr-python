from __future__ import annotations

import numpy as np

from tests.test_dtype.test_wrapper import _TestZDType
from zarr.core.dtype.npy.string import _NUMPY_SUPPORTS_VLEN_STRING, VariableLengthString

if _NUMPY_SUPPORTS_VLEN_STRING:

    class TestVariableLengthString(_TestZDType):
        test_cls = VariableLengthString
        valid_dtype = (np.dtypes.StringDType(),)
        invalid_dtype = (
            np.dtype(np.int8),
            np.dtype(np.float64),
            np.dtype("|S10"),
        )
        valid_json_v2 = ("|O",)
        valid_json_v3_cases = ({"name": "numpy.variable_length_utf8"},)
        invalid_json_v2 = (
            "|S10",
            "|f8",
            "invalid",
        )
        invalid_json_v3 = (
            {"name": "numpy.variable_length_utf8", "configuration": {"invalid_key": "value"}},
            {"name": "invalid_name"},
        )

else:

    class TestVariableLengthString(_TestZDType):
        test_cls = VariableLengthString
        valid_dtype = (np.dtype("O"),)
        invalid_dtype = (
            np.dtype(np.int8),
            np.dtype(np.float64),
            np.dtype("|S10"),
        )
        valid_json_v2 = ("|O",)
        valid_json_v3_cases = ({"name": "numpy.variable_length_utf8"},)
        invalid_json_v2 = (
            "|S10",
            "|f8",
            "invalid",
        )
        invalid_json_v3 = (
            {"name": "numpy.variable_length_utf8", "configuration": {"invalid_key": "value"}},
            {"name": "invalid_name"},
        )
