import numpy as np
import pytest

from tests.test_dtype.test_wrapper import BaseTestZDType
from zarr.core.dtype.common import UnstableSpecificationWarning
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
        {"name": "|S1", "object_codec_id": None},
        {"name": "|S2", "object_codec_id": None},
        {"name": "|S4", "object_codec_id": None},
    )
    valid_json_v3 = ({"name": "null_terminated_bytes", "configuration": {"length_bytes": 10}},)
    invalid_json_v2 = (
        "|S",
        "|U10",
        "|f8",
        {"name": "|S4", "object_codec_id": "vlen-bytes"},
    )
    invalid_json_v3 = (
        {"name": "fixed_length_ascii", "configuration": {"length_bits": 0}},
        {"name": "numpy.fixed_length_ascii", "configuration": {"length_bits": "invalid"}},
    )

    scalar_v2_params = (
        (NullTerminatedBytes(length=1), "MA=="),
        (NullTerminatedBytes(length=2), "YWI="),
        (NullTerminatedBytes(length=4), "YWJjZA=="),
    )
    scalar_v3_params = (
        (NullTerminatedBytes(length=1), "MA=="),
        (NullTerminatedBytes(length=2), "YWI="),
        (NullTerminatedBytes(length=4), "YWJjZA=="),
    )
    cast_value_params = (
        (NullTerminatedBytes(length=1), "", np.bytes_("")),
        (NullTerminatedBytes(length=2), "ab", np.bytes_("ab")),
        (NullTerminatedBytes(length=4), "abcdefg", np.bytes_("abcd")),
    )
    invalid_scalar_params = ((NullTerminatedBytes(length=1), 1.0),)
    item_size_params = (
        NullTerminatedBytes(length=1),
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
    valid_json_v2 = ({"name": "|V10", "object_codec_id": None},)
    valid_json_v3 = (
        {"name": "raw_bytes", "configuration": {"length_bytes": 1}},
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
        (RawBytes(length=1), "AA=="),
        (RawBytes(length=2), "YWI="),
        (RawBytes(length=4), "YWJjZA=="),
    )
    scalar_v3_params = (
        (RawBytes(length=1), "AA=="),
        (RawBytes(length=2), "YWI="),
        (RawBytes(length=4), "YWJjZA=="),
    )
    cast_value_params = (
        (RawBytes(length=1), b"\x00", np.void(b"\x00")),
        (RawBytes(length=2), b"ab", np.void(b"ab")),
        (RawBytes(length=4), b"abcd", np.void(b"abcd")),
    )
    invalid_scalar_params = ((RawBytes(length=1), 1.0),)
    item_size_params = (
        RawBytes(length=1),
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
    valid_json_v2 = ({"name": "|O", "object_codec_id": "vlen-bytes"},)
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
    invalid_scalar_params = ((VariableLengthBytes(), 1.0),)
    item_size_params = (VariableLengthBytes(),)


@pytest.mark.parametrize(
    "zdtype", [NullTerminatedBytes(length=10), RawBytes(length=10), VariableLengthBytes()]
)
def test_unstable_dtype_warning(
    zdtype: NullTerminatedBytes | RawBytes | VariableLengthBytes,
) -> None:
    """
    Test that we get a warning when serializing a dtype without a zarr v3 spec to json
    when zarr_format is 3
    """
    with pytest.raises(UnstableSpecificationWarning):
        zdtype.to_json(zarr_format=3)


@pytest.mark.parametrize("zdtype_cls", [NullTerminatedBytes, RawBytes])
def test_invalid_size(zdtype_cls: type[NullTerminatedBytes] | type[RawBytes]) -> None:
    """
    Test that it's impossible to create a data type that has no length
    """
    length = 0
    msg = f"length must be >= 1, got {length}."
    with pytest.raises(ValueError, match=msg):
        zdtype_cls(length=length)
