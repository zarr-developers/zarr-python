from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import pytest

from zarr.core.dtype.common import HasItemSize

if TYPE_CHECKING:
    from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType


"""
class _TestZDTypeSchema:
    # subclasses define the URL for the schema, if available
    schema_url: ClassVar[str] = ""

    @pytest.fixture(scope="class")
    def get_schema(self) -> object:
        response = requests.get(self.schema_url)
        response.raise_for_status()
        return json_schema.loads(response.text)

    def test_schema(self, schema: json_schema.Schema) -> None:
        assert schema.is_valid(self.test_cls.to_json(zarr_format=2))
"""


class _TestZDType:
    """
    A base class for testing ZDType subclasses. This class works in conjunction with the custom
    pytest collection function ``pytest_generate_tests`` defined in conftest.py, which applies the
    following procedure when generating tests:

    At test generation time, for each test fixture referenced by a method on this class
    pytest will look for an attribute with the same name as that fixture. Pytest will assume that
    this class attribute is a tuple of values to be used for generating a parametrized test fixture.

    This means that child classes can, by using different values for these class attributes, have
    customized test parametrization.

    Attributes
    ----------
    test_cls : type[ZDType[TBaseDType, TBaseScalar]]
        The ZDType subclass being tested.
    scalar_type : ClassVar[type[TBaseScalar]]
        The expected scalar type for the ZDType.
    valid_dtype : ClassVar[tuple[TBaseDType, ...]]
        A tuple of valid numpy dtypes for the ZDType.
    invalid_dtype : ClassVar[tuple[TBaseDType, ...]]
        A tuple of invalid numpy dtypes for the ZDType.
    valid_json_v2 : ClassVar[tuple[str | dict[str, object] | list[object], ...]]
        A tuple of valid JSON representations for Zarr format version 2.
    invalid_json_v2 : ClassVar[tuple[str | dict[str, object] | list[object], ...]]
        A tuple of invalid JSON representations for Zarr format version 2.
    valid_json_v3 : ClassVar[tuple[str | dict[str, object], ...]]
        A tuple of valid JSON representations for Zarr format version 3.
    invalid_json_v3 : ClassVar[tuple[str | dict[str, object], ...]]
        A tuple of invalid JSON representations for Zarr format version 3.
    cast_value_params : ClassVar[tuple[tuple[Any, Any, Any], ...]]
        A tuple of (dtype, value, expected) tuples for testing ZDType.cast_value.
    """

    test_cls: type[ZDType[TBaseDType, TBaseScalar]]
    scalar_type: ClassVar[type[TBaseScalar]]
    valid_dtype: ClassVar[tuple[TBaseDType, ...]] = ()
    invalid_dtype: ClassVar[tuple[TBaseDType, ...]] = ()

    valid_json_v2: ClassVar[tuple[str | dict[str, object] | list[object], ...]] = ()
    invalid_json_v2: ClassVar[tuple[str | dict[str, object] | list[object], ...]] = ()

    valid_json_v3: ClassVar[tuple[str | dict[str, object], ...]] = ()
    invalid_json_v3: ClassVar[tuple[str | dict[str, object], ...]] = ()

    # for testing scalar round-trip serialization, we need a tuple of (data type json, scalar json)
    # pairs. the first element of the pair is used to create a dtype instance, and the second
    # element is the json serialization of the scalar that we want to round-trip.

    scalar_v2_params: ClassVar[tuple[tuple[Any, Any], ...]] = ()
    scalar_v3_params: ClassVar[tuple[tuple[Any, Any], ...]] = ()
    cast_value_params: ClassVar[tuple[tuple[Any, Any, Any], ...]]
    item_size_params: ClassVar[tuple[ZDType[Any, Any], ...]]

    def json_scalar_equals(self, scalar1: object, scalar2: object) -> bool:
        # An equality check for json-encoded scalars. This defaults to regular equality,
        # but some classes may need to override this for special cases
        return scalar1 == scalar2

    def scalar_equals(self, scalar1: object, scalar2: object) -> bool:
        # An equality check for scalars. This defaults to regular equality,
        # but some classes may need to override this for special cases
        return scalar1 == scalar2

    def test_check_dtype_valid(self, valid_dtype: object) -> None:
        assert self.test_cls.check_dtype(valid_dtype)  # type: ignore[arg-type]

    def test_check_dtype_invalid(self, invalid_dtype: object) -> None:
        assert not self.test_cls.check_dtype(invalid_dtype)  # type: ignore[arg-type]

    def test_from_dtype_roundtrip(self, valid_dtype: Any) -> None:
        zdtype = self.test_cls.from_dtype(valid_dtype)
        assert zdtype.to_dtype() == valid_dtype

    def test_from_json_roundtrip_v2(self, valid_json_v2: Any) -> None:
        zdtype = self.test_cls.from_json(valid_json_v2, zarr_format=2)
        assert zdtype.to_json(zarr_format=2) == valid_json_v2

    @pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
    def test_from_json_roundtrip_v3(self, valid_json_v3: Any) -> None:
        zdtype = self.test_cls.from_json(valid_json_v3, zarr_format=3)
        assert zdtype.to_json(zarr_format=3) == valid_json_v3

    def test_scalar_roundtrip_v2(self, scalar_v2_params: tuple[Any, Any]) -> None:
        zdtype, scalar_json = scalar_v2_params
        scalar = zdtype.from_json_value(scalar_json, zarr_format=2)
        assert self.json_scalar_equals(scalar_json, zdtype.to_json_value(scalar, zarr_format=2))

    def test_scalar_roundtrip_v3(self, scalar_v3_params: tuple[Any, Any]) -> None:
        zdtype, scalar_json = scalar_v3_params
        scalar = zdtype.from_json_value(scalar_json, zarr_format=3)
        assert self.json_scalar_equals(scalar_json, zdtype.to_json_value(scalar, zarr_format=3))

    def test_cast_value(self, cast_value_params: tuple[Any, Any, Any]) -> None:
        zdtype, value, expected = cast_value_params
        observed = zdtype.cast_value(value)
        assert self.scalar_equals(expected, observed)

    def test_item_size(self, item_size_params: ZDType[Any, Any]) -> None:
        """
        Test that the item_size attribute matches the numpy dtype itemsize attribute, for dtypes
        with a fixed scalar size.
        """
        if isinstance(item_size_params, HasItemSize):
            assert item_size_params.item_size == item_size_params.to_dtype().itemsize
        else:
            pytest.skip(f"Dtype {item_size_params} does not implement HasItemSize")
