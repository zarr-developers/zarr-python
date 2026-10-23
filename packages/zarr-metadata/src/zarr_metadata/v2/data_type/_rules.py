"""Rules that several v2 dtype families share: the sizes and byte orders a family allows, and a base64 fill value of the item's size."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from zarr_metadata._json import ValidationProblem, shown
from zarr_metadata._numpy import base64_bytes

if TYPE_CHECKING:
    from collections.abc import Iterator

    from zarr_metadata.definition._definition import Nested
    from zarr_metadata.v2.data_type.fixed_width import ZarrV2Base64FillValue
    from zarr_metadata.v2.data_type.scalar import ZarrV2ScalarConfiguration


@dataclass(frozen=True, slots=True)
class _Sized:
    """The rules of a family whose types are `sizes` bytes wide. This is a value rather than a closure, so a scope that holds it can be pickled."""

    sizes: tuple[int, ...] | None
    orderless: frozenset[int] | None

    def __call__(
        self, configuration: ZarrV2ScalarConfiguration, nested: Nested
    ) -> Iterator[ValidationProblem]:
        size = configuration["itemsize"]
        if self.sizes is not None and size not in self.sizes:
            yield ValidationProblem(
                ("itemsize",),
                f"expected a size of {', '.join(map(str, self.sizes))} bytes, got {size}",
                "invalid_value",
            )
            return
        if (
            configuration["byteorder"] == "|"
            and self.orderless is not None
            and size not in self.orderless
        ):
            yield ValidationProblem(
                ("byteorder",),
                f"expected a byte order '<' or '>' for a type of {size} bytes, got '|'",
                "invalid_value",
            )


def sized(sizes: tuple[int, ...] | None, orderless: frozenset[int] | None) -> _Sized:
    """Return the rules of a family whose types are `sizes` bytes wide. If `sizes` is None, any width is allowed.

    `orderless` is the set of sizes at which the type has no byte order,
    which NumPy writes as `|`. This is one byte for an integer, every size
    for bytes and void, and no size for a float. None means every size.
    At any other size, the typestr must state which end comes first, `<`
    or `>`.
    """
    return _Sized(sizes, orderless)


@dataclass(frozen=True, slots=True)
class _OrderlessAt:
    """The canonical spelling for a family whose types have no byte order at `sizes` bytes."""

    sizes: frozenset[int] | None

    def __call__(self, configuration: ZarrV2ScalarConfiguration) -> ZarrV2ScalarConfiguration:
        orderless = self.sizes is None or configuration["itemsize"] in self.sizes
        if orderless and configuration["byteorder"] != "|":
            return cast("ZarrV2ScalarConfiguration", {**configuration, "byteorder": "|"})
        return configuration


def orderless_at(sizes: frozenset[int] | None) -> _OrderlessAt:
    """Return the canonical spelling for a family whose types have no byte order at `sizes` bytes. At those sizes the byte order is `|`, as NumPy writes it. None means every size."""
    return _OrderlessAt(sizes)


def sized_base64_rules(
    value: ZarrV2Base64FillValue, size: int | None, *, exact: bool
) -> Iterator[ValidationProblem]:
    """Check that a fill value that is not null is standard-alphabet base64 of exactly `size` bytes, or of at most `size` bytes when `exact` is False. Any size passes when `size` is None, which happens only when the item size is unknown."""
    if value is None:
        return
    try:
        base64_bytes(value)
    except ValueError:
        yield ValidationProblem(
            (), f"expected standard-alphabet base64, got {shown(value)}", "invalid_value"
        )
        return
    if size is None:
        return
    held = base64.b64decode(value)
    if (len(held) != size) if exact else (len(held) > size):
        bound = "" if exact else "at most "
        yield ValidationProblem(
            (),
            f"expected base64 of {bound}{size} bytes, the item's size, got {len(held)} bytes",
            "invalid_value",
        )


__all__ = ["orderless_at", "sized", "sized_base64_rules"]
