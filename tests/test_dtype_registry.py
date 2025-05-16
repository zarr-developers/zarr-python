from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args

import numpy as np
import pytest

import zarr
from zarr.core.config import config
from zarr.core.dtype import (
    AnyDType,
    Bool,
    DataTypeRegistry,
    DateTime64,
    FixedLengthUnicode,
    Int8,
    Int16,
    TBaseDType,
    TBaseScalar,
    ZDType,
    data_type_registry,
    get_data_type_from_json,
    parse_data_type,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from zarr.core.common import ZarrFormat

from .test_dtype.conftest import zdtype_examples


@pytest.fixture
def data_type_registry_fixture() -> DataTypeRegistry:
    return DataTypeRegistry()


class TestRegistry:
    @staticmethod
    def test_register(data_type_registry_fixture: DataTypeRegistry) -> None:
        """
        Test that registering a dtype in a data type registry works.
        """
        data_type_registry_fixture.register(Bool._zarr_v3_name, Bool)
        assert data_type_registry_fixture.get(Bool._zarr_v3_name) == Bool
        assert isinstance(data_type_registry_fixture.match_dtype(np.dtype("bool")), Bool)

    @staticmethod
    def test_override(data_type_registry_fixture: DataTypeRegistry) -> None:
        """
        Test that registering a new dtype with the same name works (overriding the previous one).
        """
        data_type_registry_fixture.register(Bool._zarr_v3_name, Bool)

        class NewBool(Bool):
            def default_value(self) -> np.bool_:
                return np.True_

        data_type_registry_fixture.register(NewBool._zarr_v3_name, NewBool)
        assert isinstance(data_type_registry_fixture.match_dtype(np.dtype("bool")), NewBool)

    @staticmethod
    @pytest.mark.parametrize(
        ("wrapper_cls", "dtype_str"), [(Bool, "bool"), (FixedLengthUnicode, "|U4")]
    )
    def test_match_dtype(
        data_type_registry_fixture: DataTypeRegistry,
        wrapper_cls: type[ZDType[TBaseDType, TBaseScalar]],
        dtype_str: str,
    ) -> None:
        """
        Test that match_dtype resolves a numpy dtype into an instance of the correspond wrapper for that dtype.
        """
        data_type_registry_fixture.register(wrapper_cls._zarr_v3_name, wrapper_cls)
        assert isinstance(data_type_registry_fixture.match_dtype(np.dtype(dtype_str)), wrapper_cls)

    @staticmethod
    def test_unregistered_dtype(data_type_registry_fixture: DataTypeRegistry) -> None:
        """
        Test that match_dtype raises an error if the dtype is not registered.
        """
        outside_dtype = "int8"
        with pytest.raises(
            ValueError, match=f"No data type wrapper found that matches dtype '{outside_dtype}'"
        ):
            data_type_registry_fixture.match_dtype(np.dtype(outside_dtype))

        with pytest.raises(KeyError):
            data_type_registry_fixture.get(outside_dtype)

    @staticmethod
    @pytest.mark.parametrize("zdtype", zdtype_examples)
    def test_registered_dtypes(
        zdtype: ZDType[TBaseDType, TBaseScalar], zarr_format: ZarrFormat
    ) -> None:
        """
        Test that the registered dtypes can be retrieved from the registry.
        """

        assert data_type_registry.match_dtype(zdtype.to_dtype()) == zdtype
        assert (
            data_type_registry.match_json(
                zdtype.to_json(zarr_format=zarr_format), zarr_format=zarr_format
            )
            == zdtype
        )

    @staticmethod
    @pytest.mark.parametrize("zdtype", zdtype_examples)
    def test_match_dtype_unique(
        zdtype: ZDType[Any, Any],
        data_type_registry_fixture: DataTypeRegistry,
        zarr_format: ZarrFormat,
    ) -> None:
        """
        Test that the match_dtype method uniquely specifies a registered data type. We create a local registry
        that excludes the data type class being tested, and ensure that an instance of the wrapped data type
        fails to match anything in the registry
        """
        for _cls in get_args(AnyDType):
            if _cls is not type(zdtype):
                data_type_registry_fixture.register(_cls._zarr_v3_name, _cls)

        dtype_instance = zdtype.to_dtype()

        msg = f"No data type wrapper found that matches dtype '{dtype_instance}'"
        with pytest.raises(ValueError, match=re.escape(msg)):
            data_type_registry_fixture.match_dtype(dtype_instance)

        instance_dict = zdtype.to_json(zarr_format=zarr_format)
        msg = f"No data type wrapper found that matches {instance_dict}"
        with pytest.raises(ValueError, match=re.escape(msg)):
            data_type_registry_fixture.match_json(instance_dict, zarr_format=zarr_format)


# this is copied from the registry tests -- we should deduplicate
here = str(Path(__file__).parent.absolute())


@pytest.fixture
def set_path() -> Generator[None, None, None]:
    sys.path.append(here)
    zarr.registry._collect_entrypoints()
    yield
    sys.path.remove(here)
    registries = zarr.registry._collect_entrypoints()
    for registry in registries:
        registry.lazy_load_list.clear()
    config.reset()


@pytest.mark.usefixtures("set_path")
def test_entrypoint_dtype(zarr_format: ZarrFormat) -> None:
    from package_with_entrypoint import TestDataType

    instance = TestDataType()
    dtype_json = instance.to_json(zarr_format=zarr_format)
    assert get_data_type_from_json(dtype_json, zarr_format=zarr_format) == instance


@pytest.mark.parametrize(
    ("dtype_params", "expected", "zarr_format"),
    [
        ("int8", Int8(), 3),
        (Int8(), Int8(), 3),
        (">i2", Int16(endianness="big"), 2),
        ("datetime64[10s]", DateTime64(unit="s", scale_factor=10), 2),
        (
            {"name": "numpy.datetime64", "configuration": {"unit": "s", "scale_factor": 10}},
            DateTime64(unit="s", scale_factor=10),
            3,
        ),
    ],
)
def test_parse_data_type(
    dtype_params: Any, expected: ZDType[Any, Any], zarr_format: ZarrFormat
) -> None:
    """
    Test that parse_data_type accepts alternative representations of ZDType instances, and resolves
    those inputs to the expected ZDType instance.
    """
    observed = parse_data_type(dtype_params, zarr_format=zarr_format)
    assert observed == expected
