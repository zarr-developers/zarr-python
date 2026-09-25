from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Literal, Self, get_args

import numpy as np
import pytest

from tests.conftest import skip_object_dtype
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
    Int8,
    Int16,
    RawBytes,
    Struct,
    UInt8,
    VariableLengthUTF8,
    ZDType,
    data_type_registry,
    parse_data_type,
    parse_dtype,
)
from zarr.errors import DataTypeValidationError

if TYPE_CHECKING:
    from zarr.core.common import ZarrFormat
    from zarr.core.dtype.common import DTypeJSON, DTypeName_V2

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
    def test_match_dtype_string_na_object_error(
        data_type_registry_fixture: DataTypeRegistry,
    ) -> None:
        data_type_registry_fixture.register(VariableLengthUTF8._zarr_v3_name, VariableLengthUTF8)  # type: ignore[arg-type]
        dtype: np.dtype[Any] = np.dtypes.StringDType(na_object=None)
        with pytest.raises(ValueError, match=r"Zarr data type resolution from StringDType.*failed"):
            data_type_registry_fixture.match_dtype(dtype)

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


@pytest.mark.usefixtures("set_path")
def test_entrypoint_dtype(zarr_format: ZarrFormat) -> None:
    from package_with_entrypoint import TestDataType

    # the registry loads entry point data types when it is first used
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
    json_style: tuple[ZarrFormat, Literal["internal", "metadata"] | None],
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


class _LittleEndianUInt8(UInt8):
    """A data type that declares the non-canonical spelling "<u1" as its own Zarr V2 name."""

    _zarr_v3_name = "test.little_endian_uint8"  # type: ignore[assignment]
    _zarr_v2_names = ("<u1",)  # type: ignore[assignment]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        *((name, Bool()) for name in ("|b1", "<b1", ">b1")),
        *((name, Int8()) for name in ("|i1", "<i1", ">i1")),
        *((name, UInt8()) for name in ("|u1", "<u1", ">u1")),
        *((name, RawBytes(length=4)) for name in ("|V4", "<V4", ">V4")),
        ("<i2", Int16(endianness="little")),
        (">i2", Int16(endianness="big")),
        ([["a", "<i1"], ["b", ">b1"]], Struct(fields=(("a", Int8()), ("b", Bool())))),
    ],
    ids=str,
)
def test_match_json_v2_byte_order(name: DTypeName_V2, expected: ZDType[Any, Any]) -> None:
    """
    A Zarr V2 data type name whose byte order is not relevant matches the same data type for any
    byte order character, and serializes with the canonical "|".
    """
    observed = data_type_registry.match_json({"name": name, "object_codec_id": None}, zarr_format=2)
    assert observed == expected
    assert observed.to_json(zarr_format=2) == expected.to_json(zarr_format=2)


def test_match_json_v2_declared_spelling_takes_precedence(
    data_type_registry_fixture: DataTypeRegistry,
) -> None:
    """
    A data type that declares a non-canonical spelling matches that spelling, and the other
    spellings still match the canonical data type.
    """
    data_type_registry_fixture.register(UInt8._zarr_v3_name, UInt8)
    data_type_registry_fixture.register(_LittleEndianUInt8._zarr_v3_name, _LittleEndianUInt8)
    for name, expected in (("<u1", _LittleEndianUInt8()), (">u1", UInt8()), ("|u1", UInt8())):
        observed = data_type_registry_fixture.match_json(
            {"name": name, "object_codec_id": None}, zarr_format=2
        )
        assert type(observed) is type(expected)
        assert observed.to_json(zarr_format=2)["name"] == ("<u1" if name == "<u1" else "|u1")


@pytest.mark.parametrize("name", ["|i2", "|f4", "|U1", "|M8[ns]"])
def test_match_json_v2_byte_order_relevant(name: str) -> None:
    """The byte order of a multi-byte data type is part of its name, so "|" is rejected."""
    with pytest.raises(ValueError, match="No Zarr data type found"):
        data_type_registry.match_json({"name": name, "object_codec_id": None}, zarr_format=2)


@pytest.mark.parametrize("name", ["=u1", "=i2"])
def test_match_json_v2_native_byte_order(name: str) -> None:
    """The Zarr V2 byte order character is one of "<", ">", or "|"; NumPy's "=" is not allowed."""
    with pytest.raises(ValueError, match="No Zarr data type found"):
        data_type_registry.match_json({"name": name, "object_codec_id": None}, zarr_format=2)


def test_match_json_v2_byte_order_alias_object_codec() -> None:
    """Normalizing the byte order does not relax the other members of the data type."""
    with pytest.raises(ValueError, match="No Zarr data type found"):
        data_type_registry.match_json(
            {"name": "<u1", "object_codec_id": "vlen-utf8"}, zarr_format=2
        )


@pytest.mark.parametrize("name", ["<V", ">V", "<V-1", "<V4x", "<V٤"])
def test_match_json_v2_byte_order_alias_malformed_length(name: str) -> None:
    """A fixed-length bytes name needs a length of ASCII digits to have a canonical alias."""
    with pytest.raises(ValueError, match="No Zarr data type found"):
        data_type_registry.match_json({"name": name, "object_codec_id": None}, zarr_format=2)


class _Byte(UInt8):
    """A data type that is not in the default registry, spelled "<u1" in Zarr V2."""

    _zarr_v3_name = "test.byte"  # type: ignore[assignment]
    _zarr_v2_names = ("<u1",)  # type: ignore[assignment]


def _registry_with(*classes: type[ZDType[Any, Any]]) -> DataTypeRegistry:
    registry = DataTypeRegistry()
    for cls in classes:
        registry.register(cls._zarr_v3_name, cls)
    return registry


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_resolve_struct_fields_with_registry(zarr_format: ZarrFormat) -> None:
    """
    The fields of a structured data type are resolved from the registry that resolves the
    structured data type.
    """
    registry = _registry_with(Struct, _Byte, Int8)
    expected = Struct(fields=(("a", _Byte()), ("b", Int8())))
    observed = registry.match_json(
        expected.to_json(zarr_format=zarr_format), zarr_format=zarr_format
    )
    assert observed == expected
    assert type(observed.fields[0][1]) is _Byte


def test_resolve_struct_fields_default_registry() -> None:
    """Resolved from the default registry, which lacks _Byte, the field matches nothing."""
    data = Struct(fields=(("a", _Byte()),)).to_json(zarr_format=3)
    with pytest.raises(DataTypeValidationError, match="at /configuration/fields/0/data_type$"):
        get_data_type_from_json(data, zarr_format=3)


def test_resolve_struct_fields_missing_from_registry() -> None:
    """A registry that lacks a field's data type rejects the struct, although the default has it."""
    data = Struct(fields=(("a", UInt8()),)).to_json(zarr_format=3)
    with pytest.raises(DataTypeValidationError, match="at /configuration/fields/0/data_type$"):
        _registry_with(Struct).match_json(data, zarr_format=3)


class _OverridesFromJSON(Int8):
    """A data type that overrides `from_json` itself, as data types that contain none may."""

    _zarr_v3_name = "test.overrides_from_json"  # type: ignore[assignment]

    @classmethod
    def from_json(cls, data: DTypeJSON, *, zarr_format: ZarrFormat) -> Self:
        if zarr_format == 3 and data == cls._zarr_v3_name:
            return cls()
        raise DataTypeValidationError(f"Invalid JSON representation of {cls.__name__}: {data!r}")


def test_resolve_data_type_overriding_from_json() -> None:
    """A registry creates a data type that contains no other data types with its `from_json`."""
    registry = _registry_with(Struct, _OverridesFromJSON)
    expected = Struct(fields=(("a", _OverridesFromJSON()),))
    observed = registry.match_json(expected.to_json(zarr_format=3), zarr_format=3)
    assert observed == expected


@pytest.mark.parametrize(
    ("zarr_format", "zdtype", "pointer"),
    [
        (2, Struct(fields=(("x", Int8()), ("y", _Byte()))), "/1/1"),
        # Zarr V2 structured data types nested in structured data types do not round trip
        (
            3,
            Struct(fields=(("outer", Struct(fields=(("x", Int8()), ("y", _Byte())))),)),
            "/configuration/fields/0/data_type/configuration/fields/1/data_type",
        ),
    ],
    ids=str,
)
def test_resolve_struct_field_missing_location(
    zarr_format: ZarrFormat, zdtype: Struct, pointer: str
) -> None:
    """
    A field data type that matches no data type is reported with its location in the JSON, through
    the structured data types that contain it, rather than as a mismatch of the outermost data type.
    """
    registry = _registry_with(Struct, Int8)
    with pytest.raises(DataTypeValidationError, match=f"at {re.escape(pointer)}$"):
        registry.match_json(zdtype.to_json(zarr_format=zarr_format), zarr_format=zarr_format)
