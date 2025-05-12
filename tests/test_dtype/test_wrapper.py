from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType

import pytest
import requests


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


class _TestZDType:
    test_cls: type[ZDType[TBaseDType, TBaseScalar]]

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

    def test_from_json_roundtrip_v3(self, valid_json_v3: Any) -> None:
        zdtype = self.test_cls.from_json(valid_json_v3, zarr_format=3)
        assert zdtype.to_json(zarr_format=3) == valid_json_v3

    def test_scalar_roundtrip_v2(self, scalar_v2_params: Any) -> None:
        dtype_json, scalar_json = scalar_v2_params
        zdtype = self.test_cls.from_json(dtype_json, zarr_format=2)
        scalar = zdtype.from_json_value(scalar_json, zarr_format=2)
        assert scalar_json == zdtype.to_json_value(scalar, zarr_format=2)

    def test_scalar_roundtrip_v3(self, scalar_v3_params: Any) -> None:
        dtype_json, scalar_json = scalar_v3_params
        zdtype = self.test_cls.from_json(dtype_json, zarr_format=3)
        scalar = zdtype.from_json_value(scalar_json, zarr_format=3)
        assert scalar_json == zdtype.to_json_value(scalar, zarr_format=3)
