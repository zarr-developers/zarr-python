"""
Gzip codec types.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/gzip/index.html
"""

from typing import Literal, NotRequired, TypedDict

GzipCodecName = Literal["gzip"]
"""The ``name`` field value of a ``gzip`` codec envelope."""


class GzipCodecConfiguration(TypedDict):
    """
    Configuration for the Zarr v3 ``gzip`` codec.

    ``level`` is an integer in the range 0-9; 0 disables compression and 9
    is slowest with the best compression ratio. The spec does not mandate
    a default.
    """

    level: NotRequired[int]


class GzipCodec(TypedDict):
    """Full ``gzip`` codec named-config envelope."""

    name: GzipCodecName
    configuration: GzipCodecConfiguration


__all__ = [
    "GzipCodec",
    "GzipCodecConfiguration",
    "GzipCodecName",
]
