from __future__ import annotations

import numpy as np
import pytest

from tests.test_dtype.test_wrapper import BaseTestZDType
from zarr.core.dtype import FixedLengthUTF32
from zarr.core.dtype.common import UnstableSpecificationWarning
from zarr.core.dtype.npy.string import _NUMPY_SUPPORTS_VLEN_STRING, VariableLengthUTF8

if _NUMPY_SUPPORTS_VLEN_STRING:

    class TestVariableLengthString(BaseTestZDType):
        test_cls = VariableLengthUTF8  # type: ignore[assignment]
        valid_dtype = (np.dtypes.StringDType(),)  # type: ignore[assignment]
        invalid_dtype = (
            np.dtype(np.int8),
            np.dtype(np.float64),
            np.dtype("|S10"),
        )
        valid_json_v2 = ({"name": "|O", "object_codec_id": "vlen-utf8"},)
        valid_json_v3 = ("string",)
        invalid_json_v2 = (
            "|S10",
            "|f8",
            "invalid",
        )
        invalid_json_v3 = (
            {"name": "variable_length_utf8", "configuration": {"invalid_key": "value"}},
            {"name": "invalid_name"},
        )

        scalar_v2_params = ((VariableLengthUTF8(), ""), (VariableLengthUTF8(), "hi"))
        scalar_v3_params = (
            (VariableLengthUTF8(), ""),
            (VariableLengthUTF8(), "hi"),
        )

        cast_value_params = (
            (VariableLengthUTF8(), "", np.str_("")),
            (VariableLengthUTF8(), "hi", np.str_("hi")),
        )
        # anything can become a string
        invalid_scalar_params = (None,)
        item_size_params = (VariableLengthUTF8(),)

else:

    class TestVariableLengthString(BaseTestZDType):  # type: ignore[no-redef]
        test_cls = VariableLengthUTF8  # type: ignore[assignment]
        valid_dtype = (np.dtype("O"),)
        invalid_dtype = (
            np.dtype(np.int8),
            np.dtype(np.float64),
            np.dtype("|S10"),
        )
        valid_json_v2 = ({"name": "|O", "object_codec_id": "vlen-utf8"},)
        valid_json_v3 = ("string",)
        invalid_json_v2 = (
            "|S10",
            "|f8",
            "invalid",
        )
        invalid_json_v3 = (
            {"name": "numpy.variable_length_utf8", "configuration": {"invalid_key": "value"}},
            {"name": "invalid_name"},
        )

        scalar_v2_params = ((VariableLengthUTF8(), ""), (VariableLengthUTF8(), "hi"))
        scalar_v3_params = (
            (VariableLengthUTF8(), ""),
            (VariableLengthUTF8(), "hi"),
        )

        cast_value_params = (
            (VariableLengthUTF8(), "", np.str_("")),
            (VariableLengthUTF8(), "hi", np.str_("hi")),
        )
        # anything can become a string
        invalid_scalar_params = (None,)
        item_size_params = (VariableLengthUTF8(),)


class TestFixedLengthUTF32(BaseTestZDType):
    test_cls = FixedLengthUTF32
    valid_dtype = (np.dtype(">U10"), np.dtype("<U10"))
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.float64),
        np.dtype("|S10"),
    )
    valid_json_v2 = (
        {"name": ">U10", "object_codec_id": None},
        {"name": "<U10", "object_codec_id": None},
    )
    valid_json_v3 = ({"name": "fixed_length_utf32", "configuration": {"length_bytes": 320}},)
    invalid_json_v2 = (
        "|U",
        "|S10",
        "|f8",
    )
    invalid_json_v3 = (
        {"name": "fixed_length_utf32", "configuration": {"length_bits": 0}},
        {"name": "numpy.fixed_length_utf32", "configuration": {"length_bits": "invalid"}},
    )

    scalar_v2_params = ((FixedLengthUTF32(length=1), ""), (FixedLengthUTF32(length=2), "hi"))
    scalar_v3_params = (
        (FixedLengthUTF32(length=1), ""),
        (FixedLengthUTF32(length=2), "hi"),
        (FixedLengthUTF32(length=4), "hihi"),
    )

    cast_value_params = (
        (FixedLengthUTF32(length=1), "", np.str_("")),
        (FixedLengthUTF32(length=2), "hi", np.str_("hi")),
        (FixedLengthUTF32(length=4), "hihi", np.str_("hihi")),
    )
    item_size_params = (
        FixedLengthUTF32(length=1),
        FixedLengthUTF32(length=4),
        FixedLengthUTF32(length=10),
    )
    # anything can become a string
    invalid_scalar_params = (None,)


@pytest.mark.parametrize(
    "zdtype",
    [
        FixedLengthUTF32(length=10),
    ],
)
def test_unstable_dtype_warning(zdtype: FixedLengthUTF32 | VariableLengthUTF8) -> None:
    """
    Test that we get a warning when serializing a dtype without a zarr v3 spec to json
    when zarr_format is 3
    """
    with pytest.raises(UnstableSpecificationWarning):
        zdtype.to_json(zarr_format=3)


def test_invalid_size() -> None:
    """
    Test that it's impossible to create a data type that has no length
    """
    length = 0
    msg = f"length must be >= 1, got {length}."
    with pytest.raises(ValueError, match=msg):
        FixedLengthUTF32(length=length)
