# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "zarr @ git+https://github.com/zarr-developers/zarr-python.git@main",
#   "ml_dtypes==0.5.1",
#   "pytest==8.4.1"
# ]
# ///
#

"""
Demonstrate how to extend Zarr Python by defining a new data type
"""

import json
import sys
from pathlib import Path
from typing import ClassVar, Literal, Self, TypeGuard, overload

import ml_dtypes  # necessary to add extra dtypes to NumPy
import numpy as np
import pytest

import zarr
from zarr.core.common import JSON, ZarrFormat
from zarr.core.dtype import ZDType, data_type_registry
from zarr.core.dtype.common import (
    DataTypeValidationError,
    DTypeConfig_V2,
    DTypeJSON,
    check_dtype_spec_v2,
)

# This is the int2 array data type
int2_dtype_cls = type(np.dtype("int2"))

# This is the int2 scalar type
int2_scalar_cls = ml_dtypes.int2


class Int2(ZDType[int2_dtype_cls, int2_scalar_cls]):
    """
    This class provides a Zarr compatibility layer around the int2 data type (the ``dtype`` of a
    NumPy array of type int2) and the int2 scalar type (the ``dtype`` of the scalar value inside an int2 array).
    """

    # This field is as the key for the data type in the internal data type registry, and also
    # as the identifier for the data type when serializaing the data type to disk for zarr v3
    _zarr_v3_name: ClassVar[Literal["int2"]] = "int2"
    # this field will be used internally
    _zarr_v2_name: ClassVar[Literal["int2"]] = "int2"

    # we bind a class variable to the native data type class so we can create instances of it
    dtype_cls = int2_dtype_cls

    @classmethod
    def from_native_dtype(cls, dtype: np.dtype) -> Self:
        """Create an instance of this ZDType from a native dtype."""
        if cls._check_native_dtype(dtype):
            return cls()
        raise DataTypeValidationError(
            f"Invalid data type: {dtype}. Expected an instance of {cls.dtype_cls}"
        )

    def to_native_dtype(self: Self) -> int2_dtype_cls:
        """Create an int2 dtype instance from this ZDType"""
        return self.dtype_cls()

    @classmethod
    def _check_json_v2(cls, data: DTypeJSON) -> TypeGuard[DTypeConfig_V2[Literal["|b1"], None]]:
        """
        Type check for Zarr v2-flavored JSON.

        This will check that the input is a dict like this:
        .. code-block:: json

        {
            "name": "int2",
            "object_codec_id": None
        }

        Note that this representation differs from the ``dtype`` field looks like in zarr v2 metadata.
        Specifically, whatever goes into the ``dtype`` field in metadata is assigned to the ``name`` field here.

        See the Zarr docs for more information about the JSON encoding for data types.
        """
        return (
            check_dtype_spec_v2(data) and data["name"] == "int2" and data["object_codec_id"] is None
        )

    @classmethod
    def _check_json_v3(cls, data: DTypeJSON) -> TypeGuard[Literal["int2"]]:
        """
        Type check for Zarr V3-flavored JSON.

        Checks that the input is the string "int2".
        """
        return data == cls._zarr_v3_name

    @classmethod
    def _from_json_v2(cls, data: DTypeJSON) -> Self:
        """
        Create an instance of this ZDType from Zarr V3-flavored JSON.
        """
        if cls._check_json_v2(data):
            return cls()
        #  This first does a type check on the input, and if that passes we create an instance of the ZDType.
        msg = f"Invalid JSON representation of {cls.__name__}. Got {data!r}, expected the string {cls._zarr_v2_name!r}"
        raise DataTypeValidationError(msg)

    @classmethod
    def _from_json_v3(cls: type[Self], data: DTypeJSON) -> Self:
        """
        Create an instance of this ZDType from Zarr V3-flavored JSON.

        This first does a type check on the input, and if that passes we create an instance of the ZDType.
        """
        if cls._check_json_v3(data):
            return cls()
        msg = f"Invalid JSON representation of {cls.__name__}. Got {data!r}, expected the string {cls._zarr_v3_name!r}"
        raise DataTypeValidationError(msg)

    @overload  # type: ignore[override]
    def to_json(self, zarr_format: Literal[2]) -> DTypeConfig_V2[Literal["int2"], None]: ...

    @overload
    def to_json(self, zarr_format: Literal[3]) -> Literal["int2"]: ...

    def to_json(
        self, zarr_format: ZarrFormat
    ) -> DTypeConfig_V2[Literal["int2"], None] | Literal["int2"]:
        """
        Serialize this ZDType to v2- or v3-flavored JSON

        If the zarr_format is 2, then return a dict like this:
        .. code-block:: json

        {
            "name": "int2",
            "object_codec_id": None
        }

        If the zarr_format is 3, then return the string "int2"

        """
        if zarr_format == 2:
            return {"name": "int2", "object_codec_id": None}
        elif zarr_format == 3:
            return self._zarr_v3_name
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    def _check_scalar(self, data: object) -> TypeGuard[int | ml_dtypes.int2]:
        """
        Check if a python object is a valid int2-compatible scalar

        The strictness of this type check is an implementation degree of freedom.
        You could be strict here, and only accept int2 values, or be open and accept any integer
        or any object and rely on exceptions from the int2 constructor that will be called in
        cast_scalar.
        """
        return isinstance(data, (int, int2_scalar_cls))

    def cast_scalar(self, data: object) -> ml_dtypes.int2:
        """
        Attempt to cast a python object to an int2.

        We first perform a type check to ensure that the input type is appropriate, and if that
        passes we call the int2 scalar constructor.
        """
        if self._check_scalar(data):
            return ml_dtypes.int2(data)
        msg = (
            f"Cannot convert object {data!r} with type {type(data)} to a scalar compatible with the "
            f"data type {self}."
        )
        raise TypeError(msg)

    def default_scalar(self) -> ml_dtypes.int2:
        """
        Get the default scalar value. This will be used when automatically selecting a fill value.
        """
        return ml_dtypes.int2(0)

    def to_json_scalar(self, data: object, *, zarr_format: ZarrFormat) -> int:
        """
        Convert a python object to a JSON representation of an int2 scalar.
        This is necessary for taking user input for the ``fill_value`` attribute in array metadata.

        In this implementation, we optimistically convert the input to an int,
        and then check that it lies in the acceptable range for this data type.
        """
        # We could add a type check here, but we don't need to for this example
        val: int = int(data)  # type: ignore[call-overload]
        if val not in (-2, -1, 0, 1):
            raise ValueError("Invalid value. Expected -2, -1, 0, or 1.")
        return val

    def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> ml_dtypes.int2:
        """
        Read a JSON-serializable value as an int2 scalar.

        We first perform a type check to ensure that the JSON value is well-formed, then call the
        int2 scalar constructor.

        The base definition of this method requires that it take a zarr_format parameter because
        other data types serialize scalars differently in zarr v2 and v3, but we don't use this here.

        """
        if self._check_scalar(data):
            return ml_dtypes.int2(data)
        raise TypeError(f"Invalid type: {data}. Expected an int.")


# after defining dtype class, it must be registered with the data type registry so zarr can use it
data_type_registry.register(Int2._zarr_v3_name, Int2)


# this parametrized function will create arrays in zarr v2 and v3 using our new data type
@pytest.mark.parametrize("zarr_format", [2, 3])
def test_custom_dtype(tmp_path: Path, zarr_format: Literal[2, 3]) -> None:
    # create array and write values
    z_w = zarr.create_array(
        store=tmp_path, shape=(4,), dtype="int2", zarr_format=zarr_format, compressors=None
    )
    z_w[:] = [-1, -2, 0, 1]

    # open the array
    z_r = zarr.open_array(tmp_path, mode="r")

    print(z_r.info_complete())

    # look at the array metadata
    if zarr_format == 2:
        meta_file = tmp_path / ".zarray"
    else:
        meta_file = tmp_path / "zarr.json"
    print(json.dumps(json.loads(meta_file.read_text()), indent=2))


if __name__ == "__main__":
    # Run the example with printed output, and a dummy pytest configuration file specified.
    # Without the dummy configuration file, at test time pytest will attempt to use the
    # configuration file in the project root, which will error because Zarr is using some
    # plugins that are not installed in this example.
    sys.exit(pytest.main(["-s", __file__, f"-c {__file__}"]))
