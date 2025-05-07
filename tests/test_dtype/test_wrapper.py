from __future__ import annotations

from typing import Any, ClassVar

import hypothesis.strategies as st
import numpy as np
from hypothesis.extra import numpy as npst

from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType


def all_dtypes() -> st.SearchStrategy[np.dtype[np.generic]]:
    return (
        npst.boolean_dtypes()
        | npst.integer_dtypes(endianness="=")
        | npst.unsigned_integer_dtypes(endianness="=")
        | npst.floating_dtypes(endianness="=")
        | npst.complex_number_dtypes(endianness="=")
        | npst.byte_string_dtypes(endianness="=")
        | npst.unicode_string_dtypes(endianness="=")
        | npst.datetime64_dtypes(endianness="=")
        | npst.timedelta64_dtypes(endianness="=")
    )


def get_classvar_attributes(cls: type) -> dict[str, Any]:
    classvar_attributes = {}
    for name, annotation in cls.__annotations__.items():
        if getattr(annotation, "__origin__", None) is ClassVar:
            classvar_attributes[name] = getattr(cls, name)
    return classvar_attributes


class _TestZDType:
    test_cls: type[ZDType[TBaseDType, TBaseScalar]]

    valid_dtype: ClassVar[tuple[TBaseDType, ...]] = ()
    invalid_dtype: ClassVar[tuple[TBaseDType, ...]] = ()

    valid_json_v2: ClassVar[tuple[str | dict[str, Any], ...]] = ()
    invalid_json_v2: ClassVar[tuple[str | dict[str, Any], ...]] = ()

    valid_json_v3: ClassVar[tuple[str | dict[str, Any], ...]] = ()
    invalid_json_v3: ClassVar[tuple[str | dict[str, Any], ...]] = ()

    def test_check_dtype_valid(self, valid_dtype: Any) -> None:
        assert self.test_cls.check_dtype(valid_dtype)

    def test_check_dtype_invalid(self, invalid_dtype: Any) -> None:
        assert not self.test_cls.check_dtype(invalid_dtype)

    def test_from_dtype_roundtrip(self, valid_dtype: Any) -> None:
        zdtype = self.test_cls.from_dtype(valid_dtype)
        assert zdtype.to_dtype() == valid_dtype

    """ @abc.abstractmethod
    def test_cast_value(self, value: Any) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def test_check_value(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def test_default_value(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def test_check_json(self, value: Any) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def test_from_json_roundtrip_v2(self, value: Any) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def test_from_json_roundtrip_v3(self, value: Any) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def test_from_json_value_roundtrip_v2(self, value: Any) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def test_from_json_value_roundtrip_v3(self, value: Any) -> None:
        raise NotImplementedError """
