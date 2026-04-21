"""
Bytes data type configuration.
"""

from typing import TypedDict

from typing_extensions import ReadOnly


class FixedLengthBytesConfig(TypedDict):
    """
    Configuration for a fixed-length bytes data type in Zarr v3.

    Attributes
    ----------
    length_bytes
        The length in bytes of the data associated with this configuration.
    """

    length_bytes: ReadOnly[int]


__all__ = [
    "FixedLengthBytesConfig",
]
