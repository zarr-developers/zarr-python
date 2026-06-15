"""Unified runtime type checking for JSON-decoded metadata.

This module provides a single entry point, :func:`parse_json`, that validates a
JSON-decoded ``value`` against a Python ``type_annotation`` and returns data
assignable to that annotation (coercing where sensible, e.g. a list is coerced
to a ``tuple`` for ``Sequence[T]`` annotations), or raises a useful exception.

It is intended to consolidate the many hand-written ``parse_*`` helpers spread
across the codebase (see issue #3285). The scope is deliberately limited to the
JSON-shaped types that appear in Zarr metadata:

* primitives: ``None``, ``str``, ``int``, ``float``, ``bool``
* ``Literal[...]``
* unions, including ``Optional[T]`` and ``X | Y``
* ``tuple[...]`` (fixed-length and variadic)
* ``Sequence[T]`` / ``list[T]`` (coerced to ``tuple``)
* ``Mapping[str, T]`` / ``dict[str, T]``
* :class:`~typing.TypedDict`

Anything outside this scope raises a :class:`TypeError`.
"""

from __future__ import annotations

import types
from collections.abc import Mapping, Sequence
from typing import Any, Literal, NotRequired, Required, Union, get_args, get_origin

from typing_extensions import get_type_hints, is_typeddict

__all__ = ["parse_json"]

# Primitive types handled by a plain ``isinstance`` check. ``bool`` deliberately
# comes before ``int`` because ``bool`` is a subclass of ``int`` and the two
# must not be confused (see ``_parse_primitive``).
_PRIMITIVES: tuple[type, ...] = (bool, int, float, str)


def parse_json(value: object, type_annotation: object) -> Any:
    """Validate ``value`` against ``type_annotation`` and return assignable data.

    Parameters
    ----------
    value : object
        A JSON-decoded value (``str``, ``int``, ``float``, ``bool``, ``None``,
        ``list``/``Sequence`` or ``dict``/``Mapping``).
    type_annotation : object
        The expected type. One of the categories listed in the module
        docstring.

    Returns
    -------
    object
        ``value``, possibly coerced (e.g. a list coerced to a ``tuple`` for a
        ``Sequence[T]`` annotation, or a TypedDict-shaped ``dict``).

    Raises
    ------
    ValueError
        If ``value`` does not satisfy a primitive, literal, or union
        annotation.
    TypeError
        If ``value`` has the wrong container shape, or if
        ``type_annotation`` is not within the supported scope.
    """
    # 1. None / type(None)
    if type_annotation is None or type_annotation is type(None):
        if value is None:
            return None
        raise ValueError(f"Expected None, got {value!r} instead.")

    origin = get_origin(type_annotation)

    # 3. Literal[...]
    if origin is Literal:
        return _parse_literal(value, type_annotation)

    # 4. Union / Optional / types.UnionType (X | Y)
    if origin is Union or origin is types.UnionType:
        return _parse_union(value, type_annotation)

    # 8. TypedDict
    if is_typeddict(type_annotation):
        return _parse_typeddict(value, type_annotation)

    # 5. tuple[...]
    if origin is tuple:
        return _parse_tuple(value, type_annotation)

    # 7. Mapping[str, T] / dict[str, T]
    if origin is not None and isinstance(origin, type) and issubclass(origin, Mapping):
        return _parse_mapping(value, type_annotation)

    # 6. Sequence[T] / list[T] (after tuple/Mapping so those take precedence)
    if origin is not None and isinstance(origin, type) and issubclass(origin, Sequence):
        return _parse_sequence(value, type_annotation)

    # 2. Primitives (str, int, float, bool)
    if isinstance(type_annotation, type) and issubclass(type_annotation, _PRIMITIVES):
        return _parse_primitive(value, type_annotation)

    # 9. Fallback
    raise TypeError(
        f"Cannot parse value {value!r} against unsupported type annotation {type_annotation!r}."
    )


def _parse_primitive(value: object, type_annotation: type) -> Any:
    """Validate ``value`` against a primitive type via ``isinstance``.

    The critical edge case is that ``bool`` is a subclass of ``int``. When an
    ``int`` is expected a ``bool`` must be rejected, and when a ``bool`` is
    expected an ``int`` must be rejected.
    """
    if type_annotation is bool:
        if isinstance(value, bool):
            return value
        raise ValueError(f"Expected bool, got {value!r} instead.")

    if type_annotation is int:
        # Reject bool, which is an ``int`` subclass.
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        raise ValueError(f"Expected int, got {value!r} instead.")

    if type_annotation is float:
        # Reject bool, which is an ``int`` (and thus a ``float``-compatible) value.
        if isinstance(value, float) and not isinstance(value, bool):
            return value
        raise ValueError(f"Expected float, got {value!r} instead.")

    # str
    if isinstance(value, type_annotation):
        return value
    raise ValueError(f"Expected {type_annotation.__name__}, got {value!r} instead.")


def _parse_literal(value: object, type_annotation: object) -> Any:
    """Validate that ``value`` is one of the literal members."""
    choices = get_args(type_annotation)
    # ``True == 1`` in Python, so guard against a bool being accepted for an int
    # literal (and vice versa) by also comparing types for the matched member.
    for choice in choices:
        if value == choice and type(value) is type(choice):
            return value
    raise ValueError(f"Expected one of {choices!r}, got {value!r} instead.")


def _parse_union(value: object, type_annotation: object) -> Any:
    """Try each union member; return the first that parses, else aggregate."""
    members = get_args(type_annotation)
    errors: list[str] = []
    for member in members:
        try:
            return parse_json(value, member)
        except (ValueError, TypeError) as exc:
            errors.append(f"  - against {member!r}: {exc}")
    joined = "\n".join(errors)
    raise ValueError(
        f"Expected a value matching one of {members!r}, got {value!r} instead. "
        f"Tried each union member:\n{joined}"
    )


def _parse_tuple(value: object, type_annotation: object) -> tuple[Any, ...]:
    """Validate a fixed-length or variadic ``tuple[...]`` annotation.

    ``tuple[int, str]`` is fixed-length; ``tuple[int, ...]`` is variadic. Each
    element is recursively parsed against its corresponding element type.
    """
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TypeError(f"Expected a sequence, got {value!r} instead.")

    args = get_args(type_annotation)

    # Bare ``tuple`` with no parameters: accept any sequence of elements.
    if not args:
        return tuple(value)

    # Variadic ``tuple[T, ...]``.
    if len(args) == 2 and args[1] is Ellipsis:
        element_type = args[0]
        return tuple(parse_json(item, element_type) for item in value)

    # Special-case the empty tuple ``tuple[()]``.
    if args == ((),):
        args = ()

    # Fixed-length ``tuple[T1, T2, ...]``.
    if len(value) != len(args):
        raise TypeError(
            f"Expected a sequence of length {len(args)}, got {value!r} of "
            f"length {len(value)} instead."
        )
    return tuple(
        parse_json(item, element_type) for item, element_type in zip(value, args, strict=True)
    )


def _parse_sequence(value: object, type_annotation: object) -> tuple[Any, ...]:
    """Validate ``Sequence[T]`` / ``list[T]``, returning a ``tuple``.

    Each element is recursively parsed against ``T``. A ``str`` is *not* a valid
    sequence here, even though it is technically a ``Sequence[str]``, because in
    JSON terms a string is a primitive, not an array.
    """
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TypeError(f"Expected a sequence, got {value!r} instead.")

    args = get_args(type_annotation)
    if not args:
        # Bare ``list`` / ``Sequence`` with no element type: accept any element.
        return tuple(value)
    element_type = args[0]
    return tuple(parse_json(item, element_type) for item in value)


def _parse_mapping(value: object, type_annotation: object) -> dict[str, Any]:
    """Validate ``Mapping[str, T]`` / ``dict[str, T]``, returning a ``dict``.

    Keys must be ``str`` and each value is recursively parsed against ``T``.
    """
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected a mapping, got {value!r} instead.")

    args = get_args(type_annotation)
    if args:
        key_type, value_type = args[0], args[1]
    else:
        key_type, value_type = str, object

    result: dict[str, Any] = {}
    for key, val in value.items():
        if key_type is str and not isinstance(key, str):
            raise TypeError(f"Expected mapping key to be str, got {key!r} instead.")
        result[key] = parse_json(val, value_type) if value_type is not object else val
    return result


def _parse_typeddict(value: object, type_annotation: Any) -> dict[str, Any]:
    """Validate a :class:`~typing.TypedDict` annotation.

    Each required key must be present and is parsed against its annotation;
    optional keys are parsed when present. Whether a key is required is derived
    from the *resolved* annotations -- their ``Required`` / ``NotRequired``
    wrappers combined with the TypedDict's totality (``__total__``).

    This deliberately does not use ``__required_keys__`` / ``__optional_keys__``:
    under ``from __future__ import annotations`` those are computed from
    stringized annotations at class-creation time, so a ``NotRequired[...]``
    wrapper is invisible to them. Resolving the hints with
    ``include_extras=True`` evaluates the strings and makes the wrappers visible.
    """
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected a mapping, got {value!r} instead.")

    hints = get_type_hints(type_annotation, include_extras=True)
    total = getattr(type_annotation, "__total__", True)

    required_keys: set[str] = set()
    field_types: dict[str, Any] = {}
    for key, hint in hints.items():
        # Annotated as ``Any`` so mypy does not narrow the ``is`` comparisons
        # against the ``Required`` / ``NotRequired`` special forms.
        origin: Any = get_origin(hint)
        if origin is Required:
            required_keys.add(key)
            field_types[key] = get_args(hint)[0]
        elif origin is NotRequired:
            field_types[key] = get_args(hint)[0]
        else:
            if total:
                required_keys.add(key)
            field_types[key] = hint

    missing = [key for key in required_keys if key not in value]
    if missing:
        raise ValueError(
            f"Expected required key(s) {sorted(missing)!r} for "
            f"{type_annotation.__name__}, got {value!r} instead."
        )

    result: dict[str, Any] = {}
    for key, field_type in field_types.items():
        if key in value:
            result[key] = parse_json(value[key], field_type)
    # Preserve any extra keys not declared on the TypedDict.
    for key, val in value.items():
        if key not in result:
            result[key] = val
    return result
