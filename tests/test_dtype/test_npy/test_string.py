from __future__ import annotations

import numpy as np

from tests.test_dtype.test_wrapper import _TestZDType
from zarr.core.dtype.npy.string import _NUMPY_SUPPORTS_VLEN_STRING, VariableLengthString

if _NUMPY_SUPPORTS_VLEN_STRING:

    class TestVariableLengthString(_TestZDType):
        test_cls = VariableLengthString  # type: ignore[assignment]
        valid_dtype = (np.dtypes.StringDType(),)  # type: ignore[assignment]
        invalid_dtype = (
            np.dtype(np.int8),
            np.dtype(np.float64),
            np.dtype("|S10"),
        )
        valid_json_v2 = ("|O",)
        valid_json_v3 = ("numpy.variable_length_utf8",)
        invalid_json_v2 = (
            "|S10",
            "|f8",
            "invalid",
        )
        invalid_json_v3 = (
            {"name": "numpy.variable_length_utf8", "configuration": {"invalid_key": "value"}},
            {"name": "invalid_name"},
        )

        scalar_v2_params = ((VariableLengthString(), ""), (VariableLengthString(), "hi"))
        scalar_v3_params = (
            (VariableLengthString(), ""),
            (VariableLengthString(), "hi"),
        )

        cast_value_params = (
            (VariableLengthString(), "", np.str_("")),
            (VariableLengthString(), "hi", np.str_("hi")),
        )
        item_size_params = (VariableLengthString(),)

else:

    class TestVariableLengthString(_TestZDType):  # type: ignore[no-redef]
        test_cls = VariableLengthString  # type: ignore[assignment]
        valid_dtype = (np.dtype("O"),)
        invalid_dtype = (
            np.dtype(np.int8),
            np.dtype(np.float64),
            np.dtype("|S10"),
        )
        valid_json_v2 = ("|O",)
        valid_json_v3 = ("numpy.variable_length_utf8",)
        invalid_json_v2 = (
            "|S10",
            "|f8",
            "invalid",
        )
        invalid_json_v3 = (
            {"name": "numpy.variable_length_utf8", "configuration": {"invalid_key": "value"}},
            {"name": "invalid_name"},
        )

        scalar_v2_params = ((VariableLengthString(), ""), (VariableLengthString(), "hi"))
        scalar_v3_params = (
            (VariableLengthString(), ""),
            (VariableLengthString(), "hi"),
        )

        cast_value_params = (
            (VariableLengthString(), "", np.str_("")),
            (VariableLengthString(), "hi", np.str_("hi")),
        )

        item_size_params = (VariableLengthString(),)
