from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, get_args

import numpy as np
import pytest

import zarr
from tests.conftest import skip_object_dtype
from zarr.core.config import config
from zarr.core.dtype import (
    AnyDType,
    DataTypeRegistry,
    TBaseDType,
    TBaseScalar,
    get_data_type_from_json,
)
from zarr.core.dtype.common import unpack_dtype_json
from zarr.dtype import (  # type: ignore[attr-defined]
    Bool,
    FixedLengthUTF32,
    ZDType,
    data_type_registry,
    parse_data_type,
    parse_dtype,
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
            def default_scalar(self) -> np.bool_:
                return np.True_

        data_type_registry_fixture.register(NewBool._zarr_v3_name, NewBool)
        assert isinstance(data_type_registry_fixture.match_dtype(np.dtype("bool")), NewBool)

    @staticmethod
    @pytest.mark.parametrize(
        ("wrapper_cls", "dtype_str"), [(Bool, "bool"), (FixedLengthUTF32, "|U4")]
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
        outside_dtype_name = "int8"
        outside_dtype = np.dtype(outside_dtype_name)
        msg = f"No Zarr data type found that matches dtype '{outside_dtype!r}'"
        with pytest.raises(ValueError, match=re.escape(msg)):
            data_type_registry_fixture.match_dtype(outside_dtype)

        with pytest.raises(KeyError):
            data_type_registry_fixture.get(outside_dtype_name)

    @staticmethod
    @pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
    @pytest.mark.parametrize("zdtype", zdtype_examples)
    def test_registered_dtypes_match_dtype(zdtype: ZDType[TBaseDType, TBaseScalar]) -> None:
        """
        Test that the registered dtypes can be retrieved from the registry.
        """
        skip_object_dtype(zdtype)
        assert data_type_registry.match_dtype(zdtype.to_native_dtype()) == zdtype

    @staticmethod
    @pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
    @pytest.mark.parametrize("zdtype", zdtype_examples)
    def test_registered_dtypes_match_json(
        zdtype: ZDType[TBaseDType, TBaseScalar], zarr_format: ZarrFormat
    ) -> None:
        assert (
            data_type_registry.match_json(
                zdtype.to_json(zarr_format=zarr_format), zarr_format=zarr_format
            )
            == zdtype
        )

    @staticmethod
    @pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
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
        skip_object_dtype(zdtype)
        for _cls in get_args(AnyDType):
            if _cls is not type(zdtype):
                data_type_registry_fixture.register(_cls._zarr_v3_name, _cls)

        dtype_instance = zdtype.to_native_dtype()

        msg = f"No Zarr data type found that matches dtype '{dtype_instance!r}'"
        with pytest.raises(ValueError, match=re.escape(msg)):
            data_type_registry_fixture.match_dtype(dtype_instance)

        instance_dict = zdtype.to_json(zarr_format=zarr_format)
        msg = f"No Zarr data type found that matches {instance_dict!r}"
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

    data_type_registry._lazy_load()
    instance = TestDataType()
    dtype_json = instance.to_json(zarr_format=zarr_format)
    assert get_data_type_from_json(dtype_json, zarr_format=zarr_format) == instance
    data_type_registry.unregister(TestDataType._zarr_v3_name)


@pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
@pytest.mark.parametrize("data_type", zdtype_examples, ids=str)
@pytest.mark.parametrize("json_style", [(2, "internal"), (2, "metadata"), (3, None)], ids=str)
@pytest.mark.parametrize(
    "dtype_parser_func", [parse_dtype, parse_data_type], ids=["parse_dtype", "parse_data_type"]
)
def test_parse_data_type(
    data_type: ZDType[Any, Any],
    json_style: tuple[ZarrFormat, None | Literal["internal", "metadata"]],
    dtype_parser_func: Any,
) -> None:
    """
    Test the parsing of data types into ZDType instances.

    This function tests the ability of `dtype_parser_func` to correctly
    interpret and parse data type specifications into `ZDType` instances
    according to the specified Zarr format and JSON style.

    Parameters
    ----------
    data_type : ZDType[Any, Any]
        The data type to be tested for parsing.
    json_style : tuple[ZarrFormat, None or Literal["internal", "metadata"]]
        A tuple specifying the Zarr format version and the JSON style
        for Zarr V2 2. For Zarr V2 there are 2 JSON styles: "internal", and
        "metadata". The internal style takes the form {"name": <data type identifier>, "object_codec_id": <object codec id>},
        while the metadata style is just <data type identifier>.
    dtype_parser_func : Any
        The function to be tested for parsing the data type. This is necessary for compatibility
        reasons, as we support multiple functions that perform the same data type parsing operation.
    """
    zarr_format, style = json_style
    dtype_spec: Any

    if zarr_format == 2:
        dtype_spec = data_type.to_json(zarr_format=zarr_format)
        if style == "internal":
            pass
        elif style == "metadata":
            dtype_spec = unpack_dtype_json(dtype_spec)
        else:
            raise ValueError(f"Invalid zarr v2 json style: {style}")
    else:
        dtype_spec = data_type.to_json(zarr_format=zarr_format)

    if dtype_spec == "|O":
        # The object data type on its own is ambiguous and should fail to resolve.
        msg = "Zarr data type resolution from object failed."
        with pytest.raises(ValueError, match=msg):
            dtype_parser_func(dtype_spec, zarr_format=zarr_format)
    else:
        observed = dtype_parser_func(dtype_spec, zarr_format=zarr_format)
        assert observed == data_type
