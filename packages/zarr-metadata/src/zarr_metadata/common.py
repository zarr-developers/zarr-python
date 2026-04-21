"""
Top-level cross-version primitives for Zarr metadata.

Version-specific types live under ``zarr_metadata.v2`` and ``zarr_metadata.v3``.
Codec and dtype spec types live under ``zarr_metadata.codec`` and
``zarr_metadata.dtype``.
"""

from collections.abc import Mapping, Sequence
from typing import NotRequired, TypedDict

from typing_extensions import ReadOnly

JSON = str | int | float | bool | Mapping[str, "JSON"] | Sequence["JSON"] | None
"""Any valid JSON value."""


class NamedConfig[TName: str, TConfig: Mapping[str, object]](TypedDict):
    """
    Named-config envelope with optional configuration.

    Generic with two parameters: name literal and configuration mapping.
    """

    name: ReadOnly[TName]
    configuration: NotRequired[ReadOnly[TConfig]]


class NamedRequiredConfig[TName: str, TConfig: Mapping[str, object]](TypedDict):
    """
    Named-config envelope with required configuration.

    Generic with two parameters: name literal and configuration mapping.
    """

    name: ReadOnly[TName]
    configuration: ReadOnly[TConfig]