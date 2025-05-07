# Generate a collection of zdtype instances for use in testing.
from typing import Any

import numpy as np

from zarr.core.dtype import data_type_registry
from zarr.core.dtype.common import HasLength
from zarr.core.dtype.npy.sized import Structured
from zarr.core.dtype.npy.time import DateTime64, TimeDelta64
from zarr.core.dtype.wrapper import ZDType

zdtype_examples: tuple[ZDType[Any, Any], ...] = ()
for wrapper_cls in data_type_registry.contents.values():
    # The Structured dtype has to be constructed with some actual fields
    if wrapper_cls is Structured:
        zdtype_examples += (wrapper_cls.from_dtype(np.dtype([("a", np.float64), ("b", np.int8)])),)
    elif issubclass(wrapper_cls, HasLength):
        zdtype_examples += (wrapper_cls(length=1),)
    elif issubclass(wrapper_cls, DateTime64 | TimeDelta64):
        zdtype_examples += (wrapper_cls(unit="s", interval=10),)
    else:
        zdtype_examples += (wrapper_cls(),)


def pytest_generate_tests(metafunc):
    for fixture_name in metafunc.fixturenames:
        if hasattr(metafunc.cls, fixture_name):
            metafunc.parametrize(fixture_name, getattr(metafunc.cls, fixture_name), scope="class")
