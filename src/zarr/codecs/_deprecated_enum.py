"""Helpers for deprecating string-valued enums in favor of literal strings.

See PR #3963 for context on the deprecation pattern.
"""

from __future__ import annotations

import warnings
from enum import Enum


class _DeprecatedStrEnumMeta(type):
    """
    Metaclass for legacy enum-like classes. Accessing a member name on the
    class (e.g. `LegacyShim.foo`) emits a `DeprecationWarning` and returns
    the equivalent string. Members are declared by setting a `_members`
    class attribute mapping each member name to its string value.
    """

    _members: dict[str, str]

    def __getattr__(cls, name: str) -> str:
        members: dict[str, str] = type.__getattribute__(cls, "_members")
        if name in members:
            warnings.warn(
                f"{cls.__name__}.{name} is deprecated; pass the string {members[name]!r} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return members[name]
        raise AttributeError(name)


def _coerce_enum_input(value: object, param_name: str, codec_name: str) -> object:
    """
    If `value` is a real `enum.Enum` instance, emit a deprecation warning
    naming `codec_name` and return `value.value`. Otherwise return `value`
    unchanged. The third argument lets the warning text name the actual
    codec (e.g. `BloscCodec`, `BytesCodec`, `ShardingCodec`).

    Note that zarr's own legacy classes (e.g. `ShardingCodecIndexLocation`)
    never reach the `Enum` branch here: they no longer inherit from `Enum`,
    and member access on them already returns a plain string (with its own
    warning) via `_DeprecatedStrEnumMeta`. This branch exists for enum
    instances defined *outside* zarr — in particular `str`-mixin enums that
    downstream code defined to mirror zarr's old enums, which the old
    `parse_enum`-based codepath accepted because they are `str` instances.
    Coercing them to `value.value` keeps the stored attribute a plain string
    and gives those callers a migration warning.
    """
    if isinstance(value, Enum):
        warnings.warn(
            f"Passing an enum to {codec_name}(..., {param_name}=...) is deprecated; "
            "pass the equivalent literal string instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return value.value
    return value
