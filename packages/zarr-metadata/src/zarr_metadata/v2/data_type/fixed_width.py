"""The v2 families with a fixed width per item, of any size: `bytes` (`S`), `str` (`U`) and `void` (`V`)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from zarr_metadata.v2._definition import ZarrV2DataTypeDefinition
from zarr_metadata.v2.data_type._rules import orderless_at, sized, sized_base64_rules
from zarr_metadata.v2.data_type.scalar import ZarrV2ScalarConfiguration

if TYPE_CHECKING:
    from collections.abc import Iterator

    from zarr_metadata._json import ValidationProblem
    from zarr_metadata.definition._definition import Nested

ZarrV2Base64FillValue = str | None
"""A fill value that the v2 spec encodes as base64, for a fixed-length byte string, void, or a structured type (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L191-L193), or null."""


def _void_fill_value_rules(
    configuration: ZarrV2ScalarConfiguration, nested: Nested, value: ZarrV2Base64FillValue
) -> Iterator[ValidationProblem]:
    """Check that the fill value is base64 of exactly the item's size in bytes."""
    yield from sized_base64_rules(value, configuration["itemsize"], exact=True)


def _bytes_fill_value_rules(
    configuration: ZarrV2ScalarConfiguration, nested: Nested, value: ZarrV2Base64FillValue
) -> Iterator[ValidationProblem]:
    """Check that the fill value is base64 of at most the item's size in bytes. NumPy pads a shorter byte string with zeros."""
    yield from sized_base64_rules(value, configuration["itemsize"], exact=False)


BYTES_V2: Final = ZarrV2DataTypeDefinition(
    name="bytes",
    configuration=ZarrV2ScalarConfiguration,
    rules=sized(None, None),
    canonical=orderless_at(None),
    fill_value=ZarrV2Base64FillValue,
    fill_value_rules=_bytes_fill_value_rules,
)
"""`|S<n>`: byte strings of `n` bytes. The fill value is base64 of at most `n` bytes."""

STR_V2: Final = ZarrV2DataTypeDefinition(
    name="str",
    configuration=ZarrV2ScalarConfiguration,
    rules=sized(None, frozenset()),
    fill_value=str | None,
)
"""`<U<n>`: strings of `n` code points. Each code point takes four bytes, in the written byte order. The fill value is a string."""

VOID_V2: Final = ZarrV2DataTypeDefinition(
    name="void",
    configuration=ZarrV2ScalarConfiguration,
    rules=sized(None, None),
    canonical=orderless_at(None),
    fill_value=ZarrV2Base64FillValue,
    fill_value_rules=_void_fill_value_rules,
)
"""`|V<n>`: `n` untyped bytes. The fill value is base64 of exactly `n` bytes."""

__all__ = [
    "BYTES_V2",
    "STR_V2",
    "VOID_V2",
    "ZarrV2Base64FillValue",
]
