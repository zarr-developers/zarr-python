# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "zarr @ {root}",
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
from typing import ClassVar, Literal, Self, TypeGuard

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

int2_dtype_cls = type(np.dtype("int2"))
int2_scalar_cls = ml_dtypes.int2


class Int2(ZDType[int2_dtype_cls, int2_scalar_cls]):
    """
    This class provides a Zarr compatibility layer around the int2 data type and the int2
    scalar type.
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
        """Type check for Zarr v2-flavored JSON"""
        return (
            check_dtype_spec_v2(data) and data["name"] == "int2" and data["object_codec_id"] is None
        )

    @classmethod
    def _check_json_v3(cls, data: DTypeJSON) -> TypeGuard[Literal["int2"]]:
        """Type check for Zarr v3-flavored JSON"""
        return data == cls._zarr_v3_name

    @classmethod
    def _from_json_v2(cls, data: DTypeJSON) -> Self:
        """
        Create an instance of this ZDType from zarr v3-flavored JSON.
        """
        if cls._check_json_v2(data):
            return cls()
        msg = f"Invalid JSON representation of {cls.__name__}. Got {data!r}, expected the string {cls._zarr_v2_name!r}"
        raise DataTypeValidationError(msg)

    @classmethod
    def _from_json_v3(cls: type[Self], data: DTypeJSON) -> Self:
        """
        Create an instance of this ZDType from zarr v3-flavored JSON.
        """
        if cls._check_json_v3(data):
            return cls()
        msg = f"Invalid JSON representation of {cls.__name__}. Got {data!r}, expected the string {cls._zarr_v3_name!r}"
        raise DataTypeValidationError(msg)

    def to_json(
        self, zarr_format: ZarrFormat
    ) -> DTypeConfig_V2[Literal["int2"], None] | Literal["int2"]:
        """Serialize this ZDType to v2- or v3-flavored JSON"""
        if zarr_format == 2:
            return {"name": "int2", "object_codec_id": None}
        elif zarr_format == 3:
            return self._zarr_v3_name
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    def _check_scalar(self, data: object) -> TypeGuard[int]:
        """Check if a python object is a valid scalar"""
        return isinstance(data, (int, int2_scalar_cls))

    def cast_scalar(self, data: object) -> ml_dtypes.int2:
        """
        Attempt to cast a python object to an int2. Might fail pending a type check.
        """
        if self._check_scalar(data):
            return ml_dtypes.int2(data)
        msg = f"Cannot convert object with type {type(data)} to a 2-bit integer."
        raise TypeError(msg)

    def default_scalar(self) -> ml_dtypes.int2:
        """Get the default scalar value"""
        return ml_dtypes.int2(0)

    def to_json_scalar(self, data: object, *, zarr_format: ZarrFormat) -> int:
        """Convert a python object to a scalar."""
        return int(data)

    def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> ml_dtypes.int2:
        """
        Read a JSON-serializable value as a scalar. The base definition of this method
        requires that it take a zarr_format parameter, because some data types serialize scalars
        differently in zarr v2 and v3
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
    sys.exit(pytest.main(["-s", __file__, f"-c {__file__}"]))
