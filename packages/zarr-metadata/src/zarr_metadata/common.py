"""
Top-level cross-version primitives for Zarr metadata.

Version-specific types live under `zarr_metadata.v2` and `zarr_metadata.v3`.
Codec and dtype spec types live under `zarr_metadata.v3.codec` and
`zarr_metadata.v3.data_type`.
"""

from collections.abc import Mapping, Sequence
from typing import Generic, NotRequired, TypedDict, TypeVar

from typing_extensions import ReadOnly

JSON = str | int | float | bool | Mapping[str, "JSON"] | Sequence["JSON"] | None
"""Any valid JSON value."""


TName = TypeVar("TName", bound=str)
TConfig = TypeVar("TConfig", bound=Mapping[str, object])


class NamedConfig(TypedDict, Generic[TName, TConfig]):  # noqa: UP046
    """
    Named-config envelope with optional configuration.

    Generic with two parameters: name literal and configuration mapping.

    Uses the PEP 484 ``Generic[T]`` form rather than PEP 695 ``[T]`` syntax
    so the package supports Python 3.11.
    """

    name: ReadOnly[TName]
    configuration: NotRequired[ReadOnly[TConfig]]


class NamedRequiredConfig(TypedDict, Generic[TName, TConfig]):  # noqa: UP046
    """
    Named-config envelope with required configuration.

    Generic with two parameters: name literal and configuration mapping.

    Uses the PEP 484 ``Generic[T]`` form rather than PEP 695 ``[T]`` syntax
    so the package supports Python 3.11.
    """

    name: ReadOnly[TName]
    configuration: ReadOnly[TConfig]
