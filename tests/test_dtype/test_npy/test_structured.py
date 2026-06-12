from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.test_dtype.test_wrapper import BaseTestZDType
from zarr.core.dtype import (
    Float16,
    Float64,
    Int32,
    Int64,
    Struct,
    Structured,
    UInt8,
    get_data_type_from_json,
)


class TestStruct(BaseTestZDType):
    """Test the canonical 'struct' dtype format."""

    test_cls = Struct
    valid_dtype = (
        np.dtype([("field1", np.int32), ("field2", np.float64)]),
        np.dtype([("field1", np.int64), ("field2", np.int32)]),
    )
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.float64),
        np.dtype("|S10"),
    )
    valid_json_v2 = (
        {"name": [["field1", ">i4"], ["field2", ">f8"]], "object_codec_id": None},
        {"name": [["field1", ">i8"], ["field2", ">i4"]], "object_codec_id": None},
    )
    valid_json_v3 = (
        {
            "name": "struct",
            "configuration": {
                "fields": [
                    {"name": "field1", "data_type": "int32"},
                    {"name": "field2", "data_type": "float64"},
                ]
            },
        },
        {
            "name": "struct",
            "configuration": {
                "fields": [
                    {
                        "name": "field1",
                        "data_type": {
                            "name": "numpy.datetime64",
                            "configuration": {"unit": "s", "scale_factor": 1},
                        },
                    },
                    {
                        "name": "field2",
                        "data_type": {
                            "name": "fixed_length_utf32",
                            "configuration": {"length_bytes": 32},
                        },
                    },
                ]
            },
        },
    )
    invalid_json_v2 = (
        [("field1", "|i1"), ("field2", "|f8")],
        [("field1", "|S10"), ("field2", "|f8")],
    )
    invalid_json_v3 = (
        {
            "name": "struct",
            "configuration": {
                "fields": [
                    ("field1", {"name": "int32", "configuration": {"endianness": "invalid"}}),
                    ("field2", {"name": "float64", "configuration": {"endianness": "big"}}),
                ]
            },
        },
        {"name": "invalid_name"},
    )

    scalar_v2_params = (
        (Struct(fields=(("field1", Int32()), ("field2", Float64()))), "AQAAAAAAAAAAAPA/"),
        (Struct(fields=(("field1", Float16()), ("field2", Int32()))), "AQAAAAAA"),
    )
    scalar_v3_params = (
        (
            Struct(fields=(("field1", Int32()), ("field2", Float64()))),
            {"field1": 1, "field2": 1.0},
        ),
        (Struct(fields=(("field1", Int64()), ("field2", Int32()))), {"field1": 1, "field2": 1}),
    )

    cast_value_params = (
        (
            Struct(fields=(("field1", Int32()), ("field2", Float64()))),
            (1, 2.0),
            np.array((1, 2.0), dtype=[("field1", np.int32), ("field2", np.float64)]),
        ),
        (
            Struct(fields=(("field1", Int64()), ("field2", Int32()))),
            (3, 4.5),
            np.array((3, 4.5), dtype=[("field1", np.int64), ("field2", np.int32)]),
        ),
    )

    item_size_params = (
        Struct(fields=(("field1", Int32()), ("field2", Float64()))),
        Struct(fields=(("field1", Int64()), ("field2", Int32()))),
    )

    invalid_scalar_params = (
        (Struct(fields=(("field1", Int32()), ("field2", Float64()))), "i am a string"),
        (Struct(fields=(("field1", Int32()), ("field2", Float64()))), {"type": "dict"}),
    )

    def scalar_equals(self, scalar1: Any, scalar2: Any) -> bool:
        if hasattr(scalar1, "shape") and hasattr(scalar2, "shape"):
            return np.array_equal(scalar1, scalar2)
        return super().scalar_equals(scalar1, scalar2)


class TestStructured:
    """Test the legacy 'structured' dtype format."""

    def test_invalid_size(self) -> None:
        """Test that it's impossible to create a data type that has no fields."""
        fields = ()
        msg = f"must have at least one field. Got {fields!r}"
        with pytest.raises(ValueError, match=msg):
            Structured(fields=fields)

    def test_structured_legacy_name_with_tuple_format(self) -> None:
        """Test that the legacy 'structured' name with tuple field format is accepted."""
        json_v3 = {
            "name": "structured",
            "configuration": {
                "fields": [
                    ["field1", "int32"],
                    ["field2", "float64"],
                ]
            },
        }
        dtype = Structured.from_json(json_v3, zarr_format=3)
        assert dtype.fields[0][0] == "field1"
        assert dtype.fields[1][0] == "field2"

    @pytest.mark.filterwarnings("ignore::zarr.errors.UnstableSpecificationWarning")
    def test_structured_writes_tuple_format(self) -> None:
        """Test that 'structured' writes the tuple field format."""
        dtype = Structured(fields=(("field1", Int32()), ("field2", Float64())))
        json_v3 = dtype.to_json(zarr_format=3)
        assert json_v3["name"] == "structured"
        assert json_v3["configuration"]["fields"][0] == ["field1", "int32"]


def test_invalid_size() -> None:
    """Test that it's impossible to create a data type that has no fields."""
    fields = ()
    msg = f"must have at least one field. Got {fields!r}"
    with pytest.raises(ValueError, match=msg):
        Struct(fields=fields)


@pytest.mark.filterwarnings("ignore::zarr.errors.UnstableSpecificationWarning")
def test_struct_name_is_primary() -> None:
    """Test that 'struct' is the primary name written to JSON."""
    dtype = Struct(fields=(("field1", Int32()), ("field2", Float64())))
    json_v3 = dtype.to_json(zarr_format=3)
    assert json_v3["name"] == "struct"


def test_struct_reads_legacy_tuple_format() -> None:
    """Test that 'struct' dtype reads the legacy tuple field format."""
    json_v3 = {
        "name": "struct",
        "configuration": {
            "fields": [
                ["field1", "int32"],
                ["field2", "float64"],
            ]
        },
    }
    dtype = Struct.from_json(json_v3, zarr_format=3)
    assert isinstance(dtype, Struct)
    assert dtype.fields[0][0] == "field1"
    assert dtype.fields[1][0] == "field2"


def test_struct_reads_canonical_object_format() -> None:
    """Test that 'struct' dtype reads the new object field format."""
    json_v3 = {
        "name": "struct",
        "configuration": {
            "fields": [
                {"name": "field1", "data_type": "int32"},
                {"name": "field2", "data_type": "float64"},
            ]
        },
    }
    dtype = Struct.from_json(json_v3, zarr_format=3)
    assert isinstance(dtype, Struct)
    assert dtype.fields[0][0] == "field1"
    assert dtype.fields[1][0] == "field2"


def test_fill_value_dict_form() -> None:
    """Test that dict form fill values are properly parsed."""
    dtype = Struct(fields=(("x", Int32()), ("y", Float64())))
    fill_value = dtype.from_json_scalar({"x": 42, "y": 3.14}, zarr_format=3)
    assert fill_value["x"] == 42
    assert fill_value["y"] == 3.14


def test_fill_value_dict_form_missing_fields() -> None:
    """Test that missing fields in dict form fill values use defaults."""
    dtype = Struct(fields=(("x", Int32()), ("y", Float64())))
    fill_value = dtype.from_json_scalar({"x": 42}, zarr_format=3)
    assert fill_value["x"] == 42
    assert fill_value["y"] == 0.0


def test_fill_value_legacy_base64() -> None:
    """Test that legacy base64-encoded fill values are still readable."""
    dtype = Struct(fields=(("field1", Int32()), ("field2", Float64())))
    fill_value = dtype.from_json_scalar("AQAAAAAAAAAAAPA/", zarr_format=3)
    assert fill_value["field1"] == 1
    assert fill_value["field2"] == 1.0


def test_fill_value_to_json_dict_form() -> None:
    """Test that fill values are serialized as dict form."""
    dtype = Struct(fields=(("x", Int32()), ("y", Float64())))
    scalar = np.array((42, 3.14), dtype=[("x", np.int32), ("y", np.float64)])[()]
    json_val = dtype.to_json_scalar(scalar, zarr_format=3)
    assert isinstance(json_val, dict)
    assert json_val["x"] == 42
    assert json_val["y"] == 3.14


def test_has_multi_byte_fields_true() -> None:
    """Test that has_multi_byte_fields returns True for dtypes with multi-byte fields."""
    dtype = Struct(fields=(("field1", Int32()), ("field2", Float64())))
    assert dtype.has_multi_byte_fields() is True


def test_has_multi_byte_fields_false() -> None:
    """Test that has_multi_byte_fields returns False for dtypes with only single-byte fields."""
    dtype = Struct(fields=(("field1", UInt8()), ("field2", UInt8())))
    assert dtype.has_multi_byte_fields() is False


def test_struct_from_native_dtype() -> None:
    """Test that Struct can be created from native numpy dtype."""
    dtype = np.dtype([("field1", np.int32), ("field2", np.float64)])
    struct = Struct.from_native_dtype(dtype)
    assert struct.fields[0][0] == "field1"
    assert struct.fields[1][0] == "field2"


@pytest.mark.filterwarnings("ignore::zarr.errors.UnstableSpecificationWarning")
@pytest.mark.parametrize(
    "dtype",
    [
        # one level of nesting
        np.dtype([("a", "<i4"), ("nested", [("x", "<f4"), ("y", "<f8")])]),
        # two levels of nesting
        np.dtype(
            [
                ("a", "<i4"),
                ("lvl1", [("b", "<i2"), ("lvl2", [("x", "<f4"), ("y", "<f8")])]),
            ]
        ),
    ],
)
def test_nested_structured_v2_round_trip(dtype: np.dtype[np.void]) -> None:
    """
    Regression test for nested structured dtypes failing the Zarr V2 JSON round-trip.

    ``Struct.to_json(zarr_format=2)`` emits a nested field as ``[name, [[sub, dt], ...]]``, and
    ``get_data_type_from_json`` must be able to read that form back. Previously the inner type
    guard recursed as a single ``[name, dtype]`` pair and rejected the list-of-fields form,
    so Zarr wrote V2 metadata it could not read back.
    """
    zdtype = Struct.from_native_dtype(dtype)
    json_v2 = zdtype.to_json(zarr_format=2)
    recovered = get_data_type_from_json(json_v2, zarr_format=2)
    assert recovered == zdtype
    assert recovered.to_native_dtype() == dtype


@pytest.mark.filterwarnings("ignore::zarr.errors.UnstableSpecificationWarning")
def test_nested_structured_v2_array_round_trip() -> None:
    """End-to-end test: write and read a Zarr V2 array with a nested structured dtype."""
    import zarr

    dtype = np.dtype([("a", "<i4"), ("nested", [("x", "<f4"), ("y", "<f8")])])
    store = zarr.storage.MemoryStore()
    arr = zarr.create_array(store, shape=(3,), chunks=(2,), dtype=dtype, zarr_format=2)
    data = np.zeros((3,), dtype=dtype)
    data["a"] = [1, 2, 3]
    data["nested"]["x"] = [1.5, 2.5, 3.5]
    data["nested"]["y"] = [10.0, 20.0, 30.0]
    arr[:] = data

    reopened = zarr.open_array(store, zarr_format=2)
    out = np.asarray(reopened[:])
    assert out.dtype == dtype
    assert np.array_equal(out, data)


@pytest.mark.parametrize(
    "dtype",
    [
        # top-level padding from align=True
        np.dtype([("a", "i1"), ("b", "i8")], align=True),
        # outer layout is packed, but a nested field dtype carries padding
        np.dtype([("a", "i8"), ("nested", np.dtype([("x", "i1"), ("y", "i8")], align=True))]),
    ],
)
def test_padded_structured_dtype_raises(dtype: np.dtype[np.void]) -> None:
    """
    Regression test: structured dtypes with non-default (padded / aligned) field layouts must
    fail loudly rather than silently dropping the padding.

    The Zarr structured metadata records only ``(name, dtype)`` pairs and re-packs fields
    contiguously on read, so an aligned dtype would round-trip to a smaller itemsize, silently
    corrupting stored chunk bytes. ``from_native_dtype`` detects this and raises instead.
    """
    with pytest.raises(ValueError, match="non-default field layout"):
        Struct.from_native_dtype(dtype)


def test_packed_structured_dtype_round_trips() -> None:
    """
    A packed (default-layout) structured dtype, including nested ones, must continue to round-trip
    unchanged. This guards the loud-failure path for aligned dtypes against false positives, and
    ensures existing data written with packed layouts keeps working.
    """
    for dtype in (
        np.dtype([("a", "i1"), ("b", "i8")]),
        np.dtype([("a", "<i4"), ("nested", [("x", "<f4"), ("y", "<f8")])]),
    ):
        recovered = Struct.from_native_dtype(dtype).to_native_dtype()
        assert recovered == dtype
        assert recovered.itemsize == dtype.itemsize
