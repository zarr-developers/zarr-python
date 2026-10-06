"""The v2 families of a fixed width per item: `bytes` (`S`), `str` (`U`) and `void` (`V`), of any size."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from zarr_metadata._json import ValidationProblem, shown
from zarr_metadata.v2._definition import ZarrV2DataTypeDefinition
from zarr_metadata.v2.data_type.scalar import ZarrV2ScalarConfiguration, orderless_at, sized
from zarr_metadata.v3.data_type.bytes import base64_bytes

if TYPE_CHECKING:
    from collections.abc import Iterator

    from zarr_metadata.v3._definition import Nested

ZarrV2Base64FillValue = str | None
"""A fill value the v2 spec encodes as base64: for a fixed-length byte string, void, or a structured type (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L191-L193); or null."""


def base64_fill_value_rules(
    configuration: object, nested: Nested, value: ZarrV2Base64FillValue
) -> Iterator[ValidationProblem]:
    """A string of standard-alphabet base64, when it is not null."""
    if value is None:
        return
    try:
        base64_bytes(value)
    except ValueError:
        yield ValidationProblem(
            (), f"expected standard-alphabet base64, got {shown(value)}", "invalid_value"
        )


BYTES_V2: Final = ZarrV2DataTypeDefinition(
    name="bytes",
    configuration=ZarrV2ScalarConfiguration,
    rules=sized(None, None),
    canonical=orderless_at(None),
    fill_value=ZarrV2Base64FillValue,
    fill_value_rules=base64_fill_value_rules,
)
"""`|S<n>`: byte strings of `n` bytes; the fill value base64 of one."""

STR_V2: Final = ZarrV2DataTypeDefinition(
    name="str",
    configuration=ZarrV2ScalarConfiguration,
    rules=sized(None, frozenset()),
    fill_value=str | None,
)
"""`<U<n>`: strings of `n` code points, each four bytes in the byte order written; the fill value a string."""

VOID_V2: Final = ZarrV2DataTypeDefinition(
    name="void",
    configuration=ZarrV2ScalarConfiguration,
    rules=sized(None, None),
    canonical=orderless_at(None),
    fill_value=ZarrV2Base64FillValue,
    fill_value_rules=base64_fill_value_rules,
)
"""`|V<n>`: `n` bytes of no type; the fill value base64 of them."""

__all__ = ["BYTES_V2", "STR_V2", "VOID_V2", "ZarrV2Base64FillValue", "base64_fill_value_rules"]
