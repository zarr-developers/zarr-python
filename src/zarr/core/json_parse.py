"""Helpers for validating JSON-decoded metadata.

Most JSON metadata validation is delegated to
[`msgspec.convert`][msgspec.convert], which handles the type coercions Zarr
needs (``Literal`` membership, ``int``/``bool`` strictness, list-to-tuple,
``TypedDict`` with ``NotRequired``). ``convert`` is a thin wrapper that
translates [`msgspec.ValidationError`][msgspec.ValidationError] into the
``ValueError`` the rest of the codebase already raises.

msgspec cannot handle two things in Zarr's metadata types:

* the recursive ``JSON`` / ``JSONValue`` aliases, which it rejects at
  schema-build time, and
* PEP 728 ``extra_items=`` extension fields, which it silently drops.

User-defined attributes are left to the JSON reader and writer rather than
recursively validated here. See https://github.com/zarr-developers/zarr-python/issues/3285.
"""

from __future__ import annotations

from typing import Any, get_origin

import msgspec

__all__ = ["convert", "parse_field"]


def _type_name(type_: Any) -> str:
    """Render ``type_`` for an error message.

    Parameterized types keep their arguments, so a ``Literal`` reports its
    members (``Literal[2, 3]``) rather than the bare origin name. ``__name__``
    would drop them, which loses the most useful part of the message.
    """
    if get_origin(type_) is not None:
        return str(type_).replace("typing.", "")
    return getattr(type_, "__name__", None) or str(type_).replace("typing.", "")


def convert(value: object, type_: Any, *, strict: bool = True) -> Any:
    """Validate and coerce ``value`` against ``type_`` via [`msgspec.convert`][msgspec.convert].

    On a mismatch msgspec raises
    [`msgspec.ValidationError`][msgspec.ValidationError]; this re-raises
    a plain, field-agnostic ``ValueError`` naming the expected type, so callers
    can add their own field context (see ``parse_field``).
    """
    try:
        return msgspec.convert(value, type_, strict=strict)
    except msgspec.ValidationError as exc:
        raise ValueError(f"Expected instance of {_type_name(type_)}, got {value!r}.") from exc


def parse_field(
    data: object, type_: Any, field: str, *, error: type[Exception] = ValueError
) -> Any:
    """Validate ``data`` for metadata field ``field`` against ``type_``.

    Wraps ``convert`` and, on failure, re-raises ``error`` with field
    context, chaining the underlying type error. This keeps the
    ``convert``-then-re-raise pattern in one place rather than repeating it in
    every per-field parser.
    """
    try:
        return convert(data, type_)
    except ValueError as exc:
        raise error(
            f"Failed to parse input for {field!r}: expected {_type_name(type_)}, got {data!r}."
        ) from exc
