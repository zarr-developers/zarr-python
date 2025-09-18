# Generate a collection of zdtype instances for use in testing.
import warnings
from typing import Any

import numpy as np

from zarr.core.dtype import data_type_registry
from zarr.core.dtype.common import HasLength
from zarr.core.dtype.npy.structured import Structured
from zarr.core.dtype.npy.time import DateTime64, TimeDelta64
from zarr.core.dtype.wrapper import ZDType

zdtype_examples: tuple[ZDType[Any, Any], ...] = ()
for wrapper_cls in data_type_registry.contents.values():
    # The Structured dtype has to be constructed with some actual fields
    if wrapper_cls is Structured:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            zdtype_examples += (
                wrapper_cls.from_native_dtype(np.dtype([("a", np.float64), ("b", np.int8)])),
            )
    elif issubclass(wrapper_cls, HasLength):
        zdtype_examples += (wrapper_cls(length=1),)
    elif issubclass(wrapper_cls, DateTime64 | TimeDelta64):
        zdtype_examples += (wrapper_cls(unit="s", scale_factor=10),)
    else:
        zdtype_examples += (wrapper_cls(),)


def pytest_generate_tests(metafunc: Any) -> None:
    """
    This is a pytest hook to parametrize class-scoped fixtures.

    This hook allows us to define class-scoped fixtures as class attributes and then
    generate the parametrize calls for pytest. This allows the fixtures to be
    reused across multiple tests within the same class.

    For example, if you had a regular pytest class like this:

    class TestClass:
       @pytest.mark.parametrize("param_a", [1, 2, 3])
        def test_method(self, param_a):
            ...

    Child classes inheriting from ``TestClass`` would not be able to override the ``param_a`` fixture

    this implementation of ``pytest_generate_tests`` allows you to define class-scoped fixtures as
    class attributes, which allows the following to work:

    class TestExample:
        param_a = [1, 2, 3]

        def test_example(self, param_a):
            ...

    # this class will have its test_example method parametrized with the values of TestB.param_a
    class TestB(TestExample):
        param_a = [1, 2, 100, 10]

    """
    # Iterate over all the fixtures defined in the class
    # and parametrize them with the values defined in the class
    # This allows us to define class-scoped fixtures as class attributes
    # and then generate the parametrize calls for pytest
    for fixture_name in metafunc.fixturenames:
        if hasattr(metafunc.cls, fixture_name):
            params = getattr(metafunc.cls, fixture_name)
            metafunc.parametrize(fixture_name, params, scope="class", ids=str)
