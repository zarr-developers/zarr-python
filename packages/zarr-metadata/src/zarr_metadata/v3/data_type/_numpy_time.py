"""How the two numpy time types store their values, defined once so that neither type module imports it from the other.

The encodings both types share with the v2 time dtypes are in
`zarr_metadata._numpy`.
"""

from typing import Final

from zarr_metadata.v3._definition import multi_byte

numpy_time_storage: Final = multi_byte
"""The storage of the numpy time types: signed 64-bit integers, in the byte order that the codecs specify.

Each type "is compatible with any codec that supports arrays of signed
64-bit integers"
(https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/numpy.datetime64/README.md?plain=1#L120),
and "the endianness of numpy.datetime64 arrays is determined by the
configuration of the codecs defined in metadata"
(https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/numpy.datetime64/README.md?plain=1#L83-L85).
The `numpy.timedelta64` spec says the same.
"""


__all__ = ["numpy_time_storage"]
