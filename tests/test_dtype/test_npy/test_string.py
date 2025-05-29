from __future__ import annotations

import numpy as np

from tests.test_dtype.test_wrapper import BaseTestZDType, V2JsonTestParams
from zarr.core.dtype import FixedLengthASCII, FixedLengthUTF32
from zarr.core.dtype.npy.string import _NUMPY_SUPPORTS_VLEN_STRING, VariableLengthString

if _NUMPY_SUPPORTS_VLEN_STRING:

    class TestVariableLengthString(BaseTestZDType):
        test_cls = VariableLengthString  # type: ignore[assignment]
        valid_dtype = (np.dtypes.StringDType(),)  # type: ignore[assignment]
        invalid_dtype = (
            np.dtype(np.int8),
            np.dtype(np.float64),
            np.dtype("|S10"),
        )
        valid_json_v2 = (V2JsonTestParams(dtype="|O", object_codec_id="vlen-utf8"),)
        valid_json_v3 = ("variable_length_utf8",)
        invalid_json_v2 = (
            "|S10",
            "|f8",
            "invalid",
        )
        invalid_json_v3 = (
            {"name": "variable_length_utf8", "configuration": {"invalid_key": "value"}},
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

    class TestVariableLengthString(BaseTestZDType):  # type: ignore[no-redef]
        test_cls = VariableLengthString  # type: ignore[assignment]
        valid_dtype = (np.dtype("O"),)
        invalid_dtype = (
            np.dtype(np.int8),
            np.dtype(np.float64),
            np.dtype("|S10"),
        )
        valid_json_v2 = (V2JsonTestParams(dtype="|O", object_codec_id="vlen-utf8"),)
        valid_json_v3 = ("variable_length_utf8",)
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


class TestFixedLengthAscii(BaseTestZDType):
    test_cls = FixedLengthASCII
    valid_dtype = (np.dtype("|S10"), np.dtype("|S4"))
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.float64),
        np.dtype("|U10"),
    )
    valid_json_v2 = (
        V2JsonTestParams(dtype="|S0"),
        V2JsonTestParams(dtype="|S2"),
        V2JsonTestParams(dtype="|S4"),
    )
    valid_json_v3 = ({"name": "fixed_length_ascii", "configuration": {"length_bytes": 10}},)
    invalid_json_v2 = (
        "|S",
        "|U10",
        "|f8",
    )
    invalid_json_v3 = (
        {"name": "fixed_length_ascii", "configuration": {"length_bits": 0}},
        {"name": "numpy.fixed_length_ascii", "configuration": {"length_bits": "invalid"}},
    )

    scalar_v2_params = (
        (FixedLengthASCII(length=0), ""),
        (FixedLengthASCII(length=2), "YWI="),
        (FixedLengthASCII(length=4), "YWJjZA=="),
    )
    scalar_v3_params = (
        (FixedLengthASCII(length=0), ""),
        (FixedLengthASCII(length=2), "YWI="),
        (FixedLengthASCII(length=4), "YWJjZA=="),
    )
    cast_value_params = (
        (FixedLengthASCII(length=0), "", np.bytes_("")),
        (FixedLengthASCII(length=2), "ab", np.bytes_("ab")),
        (FixedLengthASCII(length=4), "abcd", np.bytes_("abcd")),
    )
    item_size_params = (
        FixedLengthASCII(length=0),
        FixedLengthASCII(length=4),
        FixedLengthASCII(length=10),
    )


class TestFixedLengthUTF32(BaseTestZDType):
    test_cls = FixedLengthUTF32
    valid_dtype = (np.dtype(">U10"), np.dtype("<U10"))
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.float64),
        np.dtype("|S10"),
    )
    valid_json_v2 = (V2JsonTestParams(dtype=">U10"), V2JsonTestParams(dtype="<U10"))
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

    scalar_v2_params = ((FixedLengthUTF32(length=0), ""), (FixedLengthUTF32(length=2), "hi"))
    scalar_v3_params = (
        (FixedLengthUTF32(length=0), ""),
        (FixedLengthUTF32(length=2), "hi"),
        (FixedLengthUTF32(length=4), "hihi"),
    )

    cast_value_params = (
        (FixedLengthUTF32(length=0), "", np.str_("")),
        (FixedLengthUTF32(length=2), "hi", np.str_("hi")),
        (FixedLengthUTF32(length=4), "hihi", np.str_("hihi")),
    )
    item_size_params = (
        FixedLengthUTF32(length=0),
        FixedLengthUTF32(length=4),
        FixedLengthUTF32(length=10),
    )
