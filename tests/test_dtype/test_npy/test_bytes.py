import numpy as np

from tests.test_dtype.test_wrapper import BaseTestZDType, V2JsonTestParams
from zarr.core.dtype.npy.bytes import NullTerminatedBytes, RawBytes, VariableLengthBytes


class TestNullTerminatedBytes(BaseTestZDType):
    test_cls = NullTerminatedBytes
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
    valid_json_v3 = ({"name": "null_terminated_bytes", "configuration": {"length_bytes": 10}},)
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
        (NullTerminatedBytes(length=0), ""),
        (NullTerminatedBytes(length=2), "YWI="),
        (NullTerminatedBytes(length=4), "YWJjZA=="),
    )
    scalar_v3_params = (
        (NullTerminatedBytes(length=0), ""),
        (NullTerminatedBytes(length=2), "YWI="),
        (NullTerminatedBytes(length=4), "YWJjZA=="),
    )
    cast_value_params = (
        (NullTerminatedBytes(length=0), "", np.bytes_("")),
        (NullTerminatedBytes(length=2), "ab", np.bytes_("ab")),
        (NullTerminatedBytes(length=4), "abcdefg", np.bytes_("abcd")),
    )
    item_size_params = (
        NullTerminatedBytes(length=0),
        NullTerminatedBytes(length=4),
        NullTerminatedBytes(length=10),
    )


class TestRawBytes(BaseTestZDType):
    test_cls = RawBytes
    valid_dtype = (np.dtype("|V10"),)
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.float64),
        np.dtype("|S10"),
    )
    valid_json_v2 = (V2JsonTestParams(dtype="|V10"),)
    valid_json_v3 = (
        {"name": "raw_bytes", "configuration": {"length_bytes": 0}},
        {"name": "raw_bytes", "configuration": {"length_bytes": 8}},
    )

    invalid_json_v2 = (
        "|V",
        "|S10",
        "|f8",
    )
    invalid_json_v3 = (
        {"name": "r10"},
        {"name": "r-80"},
    )

    scalar_v2_params = (
        (RawBytes(length=0), ""),
        (RawBytes(length=2), "YWI="),
        (RawBytes(length=4), "YWJjZA=="),
    )
    scalar_v3_params = (
        (RawBytes(length=0), ""),
        (RawBytes(length=2), "YWI="),
        (RawBytes(length=4), "YWJjZA=="),
    )
    cast_value_params = (
        (RawBytes(length=0), b"", np.void(b"")),
        (RawBytes(length=2), b"ab", np.void(b"ab")),
        (RawBytes(length=4), b"abcd", np.void(b"abcd")),
    )
    item_size_params = (
        RawBytes(length=0),
        RawBytes(length=4),
        RawBytes(length=10),
    )


class TestVariableLengthBytes(BaseTestZDType):
    test_cls = VariableLengthBytes
    valid_dtype = (np.dtype("|O"),)
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.float64),
        np.dtype("|U10"),
    )
    valid_json_v2 = (V2JsonTestParams(dtype="|O", object_codec_id="vlen-bytes"),)
    valid_json_v3 = ("variable_length_bytes",)
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
        (VariableLengthBytes(), ""),
        (VariableLengthBytes(), "YWI="),
        (VariableLengthBytes(), "YWJjZA=="),
    )
    scalar_v3_params = (
        (VariableLengthBytes(), ""),
        (VariableLengthBytes(), "YWI="),
        (VariableLengthBytes(), "YWJjZA=="),
    )
    cast_value_params = (
        (VariableLengthBytes(), "", b""),
        (VariableLengthBytes(), "ab", b"ab"),
        (VariableLengthBytes(), "abcdefg", b"abcdefg"),
    )
    item_size_params = (
        VariableLengthBytes(),
        VariableLengthBytes(),
        VariableLengthBytes(),
    )
