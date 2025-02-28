"""
# Overview

This module provides a proof-of-concept standalone interface for managing dtypes in the zarr-python codebase.

The `ZarrDType` class introduced in this module effectively acts as a replacement for `np.dtype` throughout the
zarr-python codebase. It attempts to encapsulate all relevant runtime information necessary for working with
dtypes in the context of the Zarr V3 specification (e.g. is this a core dtype or not, how many bytes and what
endianness is the dtype etc). By providing this abstraction, the module aims to:

- Simplify dtype management within zarr-python
- Support runtime flexibility and custom extensions
- Remove unnecessary dependencies on the numpy API

## Extensibility

The module attempts to support user-driven extensions, allowing developers to introduce custom dtypes
without requiring immediate changes to zarr-python. Extensions can leverage the current entrypoint mechanism,
enabling integration of experimental features. Over time, widely adopted extensions may be formalized through
inclusion in zarr-python or standardized via a Zarr Enhancement Proposal (ZEP), but this is not essential.

## Examples

### Core `dtype` Registration

The following example demonstrates how to register a built-in `dtype` in the core codebase:

```python
from zarr.core.dtype import ZarrDType
from zarr.registry import register_v3dtype

class Float16(ZarrDType):
    zarr_spec_format = "3"
    experimental = False
    endianness = "little"
    byte_count = 2
    to_numpy = np.dtype('float16')

register_v3dtype(Float16)
```

### Entrypoint Extension

The following example demonstrates how users can register a new `bfloat16` dtype for Zarr.
This approach adheres to the existing Zarr entrypoint pattern as much as possible, ensuring
consistency with other extensions. The code below would typically be part of a Python package
that specifies the entrypoints for the extension:

```python
import ml_dtypes
from zarr.core.dtype import ZarrDType  # User inherits from ZarrDType when creating their dtype

class Bfloat16(ZarrDType):
    zarr_spec_format = "3"
    experimental = True
    endianness = "little"
    byte_count = 2
    to_numpy = np.dtype('bfloat16')  # Enabled by importing ml_dtypes
    configuration_v3 = {
        "version": "example_value",
        "author": "example_value",
        "ml_dtypes_version": "example_value"
    }
```

### dtype lookup

The following examples demonstrate how to perform a lookup for the relevant ZarrDType, given
a string that matches the dtype Zarr specification ID, or a numpy dtype object:

```
from zarr.registry import get_v3dtype_class, get_v3dtype_class_from_numpy

get_v3dtype_class('complex64')  # returns little-endian Complex64 ZarrDType
get_v3dtype_class('not_registered_dtype')  # ValueError

get_v3dtype_class_from_numpy('>i2')  # returns big-endian Int16 ZarrDType
get_v3dtype_class_from_numpy(np.dtype('float32'))  # returns little-endian Float32 ZarrDType
get_v3dtype_class_from_numpy('i10')  # ValueError
```

### String dtypes

The following indicates one possibility for supporting variable-length strings. It is via the
entrypoint mechanism as in a previous example. The Apache Arrow specification does not currently
include a dtype for fixed-length strings (only for fixed-length bytes) and so I am using string
here to implicitly refer to a variable-length string data (there may be some subtleties with codecs
that means this needs to be refined further):

```python
import numpy as np
from zarr.core.dtype import ZarrDType  # User inherits from ZarrDType when creating their dtype

try:
    to_numpy = np.dtypes.StringDType()
except AttributeError:
    to_numpy = np.dtypes.ObjectDType()

class String(ZarrDType):
    zarr_spec_format = "3"
    experimental = True
    endianness = 'little'
    byte_count = None  # None is defined to mean variable
    to_numpy = to_numpy
```

### int4 dtype

There is currently considerable interest in the AI community in 'quantising' models - storing
models at reduced precision, while minimising loss of information content. There are a number
of sub-byte dtypes that the community are using e.g. int4. Unfortunately numpy does not
currently have support for handling such sub-byte dtypes in an easy way. However, they can
still be held in a numpy array and then passed (in a zero-copy way) to something like pytorch
which can handle appropriately:

```python
import numpy as np
from zarr.core.dtype import ZarrDType  # User inherits from ZarrDType when creating their dtype

class Int4(ZarrDType):
    zarr_spec_format = "3"
    experimental = True
    endianness = 'little'
    byte_count = 1  # this is ugly, but I could change this from byte_count to bit_count if there was consensus
    to_numpy = np.dtype('B')  # could also be np.dtype('V1'), but this would prevent bit-twiddling
    configuration_v3 = {
        "version": "example_value",
        "author": "example_value",
    }
```
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np


class FrozenClassVariables(type):
    def __setattr__(cls, attr: str, value: object) -> None:
        if hasattr(cls, attr):
            raise ValueError(f"Attribute {attr} on ZarrDType class can not be changed once set.")
        else:
            raise AttributeError(f"'{cls}' object has no attribute '{attr}'")


class ZarrDType(metaclass=FrozenClassVariables):
    zarr_spec_format: Literal["2", "3"]  # the version of the zarr spec used
    experimental: bool  # is this in the core spec or not
    endianness: Literal[
        "big", "little", None
    ]  # None indicates not defined i.e. single byte or byte strings
    byte_count: int | None  # None indicates variable count
    to_numpy: np.dtype[Any]  # may involve installing a a numpy extension e.g. ml_dtypes;

    configuration_v3: dict | None  # TODO: understand better how this is recommended by the spec

    _zarr_spec_identifier: str  # implementation detail used to map to core spec

    def __init_subclass__(  # enforces all required fields are set and basic sanity checks
        cls,
        **kwargs,
    ) -> None:
        required_attrs = [
            "zarr_spec_format",
            "experimental",
            "endianness",
            "byte_count",
            "to_numpy",
        ]
        for attr in required_attrs:
            if not hasattr(cls, attr):
                raise ValueError(f"{attr} is a required attribute for a Zarr dtype.")

        if not hasattr(cls, "configuration_v3"):
            cls.configuration_v3 = None

        cls._zarr_spec_identifier = (
            "big_" + cls.__qualname__.lower()
            if cls.endianness == "big"
            else cls.__qualname__.lower()
        )  # how this dtype is identified in core spec; convention is prefix with big_ for big-endian

        cls._validate()  # sanity check on basic requirements

        super().__init_subclass__(**kwargs)

    # TODO: add further checks
    @classmethod
    def _validate(cls):
        if cls.byte_count is not None and cls.byte_count <= 0:
            raise ValueError("byte_count must be a positive integer.")

        if cls.byte_count == 1 and cls.endianness is not None:
            raise ValueError("Endianness must be None for single-byte types.")
