from __future__ import annotations

import numpy as np

from tests.test_dtype.test_wrapper import BaseTestZDType
from zarr.core.dtype.npy.bool import Bool


class TestBool(BaseTestZDType):
    test_cls = Bool

    valid_dtype = (np.dtype(np.bool_),)
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.float64),
        np.dtype(np.uint16),
    )
    valid_json_v2 = ({"name": "|b1", "object_codec_id": None},)
    valid_json_v3 = ("bool",)
    invalid_json_v2 = (
        "|b1",
        "bool",
        "|f8",
    )
    invalid_json_v3 = (
        "|b1",
        "|f8",
        {"name": "bool", "configuration": {"endianness": "little"}},
    )

    scalar_v2_params = ((Bool(), True), (Bool(), False))
    scalar_v3_params = ((Bool(), True), (Bool(), False))

    cast_value_params = (
        (Bool(), "true", np.True_),
        (Bool(), True, np.True_),
        (Bool(), False, np.False_),
        (Bool(), np.True_, np.True_),
        (Bool(), np.False_, np.False_),
    )
    invalid_scalar_params = (None,)
    item_size_params = (Bool(),)
