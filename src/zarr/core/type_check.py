from __future__ import annotations

import sys
import types
import typing
from dataclasses import dataclass
from typing import (
    Any,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

try:
    # typing_extensions.ReadOnly if available; otherwise a sentinel wrapper behavior
    from typing_extensions import ReadOnly
except Exception:
    class ReadOnly:  # type: ignore
        def __class_getitem__(cls, item: Any) -> Any:
            return item


# ---------- result dataclass ----------
@dataclass(frozen=True)
class TypeCheckResult:
    success: bool
    errors: list[str]


# ---------- helpers ----------
def _type_name(tp: Any) -> str:
    """Return a human-friendly type name (int, float, str) when possible."""
    try:
        if isinstance(tp, type):
            return tp.__name__
    except Exception:
        pass
    # For typing constructs, show a compact representation
    return getattr(tp, "__qualname__", None) or str(tp)


def _is_typeddict_class(tp: Any) -> bool:
    """Safe predicate: is tp a TypedDict class (non-subscripted)?"""
    return isinstance(tp, type) and hasattr(tp, "__annotations__") and hasattr(tp, "__total__")


def _strip_readonly(tp: Any) -> Any:
    """If tp is ReadOnly[T], return T, else return tp."""
    origin = get_origin(tp)
    if origin is ReadOnly:
        args = get_args(tp)
        return args[0] if args else Any
    return tp


def _substitute_typevars(tp: Any, type_map: dict[TypeVar, Any]) -> Any:
    """Recursively substitute TypeVars (if any) according to type_map."""
    from typing import TypeVar as _TypeVar

    if isinstance(tp, _TypeVar):
        return type_map.get(tp, tp)

    origin = get_origin(tp)
    if origin is None:
        return tp

    args = get_args(tp)
    if not args:
        return tp

    # Substitute each arg
    new_args = tuple(_substitute_typevars(a, type_map) for a in args)
    try:
        return origin[new_args]  # reconstruct parameterized type if possible
    except Exception:
        # Fallback: if single-arg wrapper like ReadOnly[T], return inner substituted
        if len(new_args) == 1:
            return new_args[0]
        return tp


def _resolved_typedict_hints(td_cls: type, type_map: dict[TypeVar, Any] | None = None) -> dict[str, Any]:
    """Return resolved annotations for a TypedDict class, substituting TypeVars."""
    try:
        mod = sys.modules.get(td_cls.__module__)
        globalns = vars(mod) if mod else None
        localns = dict(vars(td_cls))
        hints = get_type_hints(td_cls, globalns=globalns, localns=localns, include_extras=True)
    except Exception:
        hints = getattr(td_cls, "__annotations__", {}).copy()

    if type_map:
        for k, v in list(hints.items()):
            hints[k] = _substitute_typevars(v, type_map)
    return hints


def _find_generic_typedict_base(cls: type) -> tuple[type | None, tuple[Any, ...] | None]:
    """If cls inherits from a generic TypedDict base, return (base_origin, args)."""
    for base in getattr(cls, "__orig_bases__", ()):
        origin = get_origin(base)
        if origin is None:
            continue
        if isinstance(origin, type) and hasattr(origin, "__annotations__"):
            return origin, get_args(base)
    return None, None


# ---------- core checker ----------
def check_type(obj: Any, expected_type: Any, path: str = "value") -> TypeCheckResult:
    origin = get_origin(expected_type)
    args = get_args(expected_type)

    # Any
    if expected_type is Any:
        return TypeCheckResult(True, [])

    # PEP 604 unions (types.UnionType) OR typing.Union
    if origin is typing.Union or isinstance(expected_type, types.UnionType):
        errors: list[str] = []
        union_args = args or (get_args(expected_type) if args else ())
        # on PEP604, get_args still works; just ensure we have union_args
        for arg in union_args:
            res = check_type(obj, arg, path)
            if res.success:
                return TypeCheckResult(True, [])
            errors.extend(res.errors)
        return TypeCheckResult(False, errors or [f"{path} did not match any type in {expected_type}"])

    # Literal
    if origin is typing.Literal:
        allowed = args
        if obj in allowed:
            return TypeCheckResult(True, [])
        return TypeCheckResult(False, [f"{path} expected literal in {allowed} but got {obj!r}"])

    # None
    if expected_type is None or expected_type is type(None):
        if obj is None:
            return TypeCheckResult(True, [])
        return TypeCheckResult(False, [f"{path} expected None but got {type(obj).__name__}"])

    # Primitives
    if expected_type in (int, float, str, bool):
        if isinstance(obj, expected_type):
            return TypeCheckResult(True, [])
        return TypeCheckResult(False, [f"{path} expected {_type_name(expected_type)} but got {type(obj).__name__}"])

    # If expected_type is a subscripted TypedDict, origin will be the base TD class
    if origin and isinstance(origin, type) and hasattr(origin, "__annotations__"):
        # generic typed dict path
        base_origin = origin
        base_args = args
        return _check_generic_typeddict(obj, base_origin, base_args, path)

    # Non-subscripted TypedDict class
    if _is_typeddict_class(expected_type):
        # special-case: class may itself inherit a generic base with concrete args
        base_origin, base_args = _find_generic_typedict_base(expected_type)
        if base_origin is not None:
            # build map
            type_vars = getattr(base_origin, "__parameters__", ())
            type_map = dict(zip(type_vars, base_args))
            return _check_generic_typeddict(obj, base_origin, base_args, path, type_map=type_map)
        return _check_typeddict(obj, expected_type, path)

    # list[T]
    if origin is list or origin is list:
        if not isinstance(obj, list):
            return TypeCheckResult(False, [f"{path} expected list but got {type(obj).__name__}"])
        elem_type = args[0] if args else Any
        errors: list[str] = []
        for i, item in enumerate(obj):
            res = check_type(item, elem_type, f"{path}[{i}]")
            if not res.success:
                errors.extend(res.errors)
        return TypeCheckResult(not errors, errors)

    # tuple[...] (fixed or variadic)
    if origin is tuple:
        if not isinstance(obj, tuple):
            return TypeCheckResult(False, [f"{path} expected tuple but got {type(obj).__name__}"])
        targs = args
        errors: list[str] = []
        if len(targs) == 2 and targs[1] is Ellipsis:
            elem_t = targs[0]
            for i, item in enumerate(obj):
                res = check_type(item, elem_t, f"{path}[{i}]")
                if not res.success:
                    errors.extend(res.errors)
            return TypeCheckResult(not errors, errors)
        if len(obj) != len(targs):
            return TypeCheckResult(False, [f"{path} expected tuple of length {len(targs)} but got {len(obj)}"])
        for i, (item, tp) in enumerate(zip(obj, targs)):
            res = check_type(item, tp, f"{path}[{i}]")
            if not res.success:
                errors.extend(res.errors)
        return TypeCheckResult(not errors, errors)

    # set[T]
    if origin is set:
        if not isinstance(obj, set):
            return TypeCheckResult(False, [f"{path} expected set but got {type(obj).__name__}"])
        item_t = args[0] if args else Any
        errors: list[str] = []
        for i, item in enumerate(obj):
            res = check_type(item, item_t, f"{path}[{i}]")
            if not res.success:
                errors.extend(res.errors)
        return TypeCheckResult(not errors, errors)

    # Mapping / dict[K, V] (accept both builtin generics and abc.Mapping)
    if origin in (dict, typing.Mapping) or expected_type in (dict, typing.Mapping):
        if not isinstance(obj, dict) and not isinstance(obj, typing.Mapping):
            return TypeCheckResult(False, [f"{path} expected dict/mapping but got {type(obj).__name__}"])
        key_t = args[0] if args else Any
        val_t = args[1] if len(args) > 1 else Any
        errors: list[str] = []
        for k, v in obj.items():
            rk = check_type(k, key_t, f"{path}[key {k!r}]")
            rv = check_type(v, val_t, f"{path}[{k!r}]")
            if not rk.success:
                errors.extend(rk.errors)
            if not rv.success:
                errors.extend(rv.errors)
        return TypeCheckResult(not errors, errors)

    # Fallback: try isinstance, but guard against TypeError for typing constructs
    try:
        if isinstance(obj, expected_type):
            return TypeCheckResult(True, [])
        # make a nicer message for named types
        tn = _type_name(expected_type)
        return TypeCheckResult(False, [f"{path} expected {tn} but got {type(obj).__name__}"])
    except TypeError:
        return TypeCheckResult(False, [f"{path} cannot be checked against {expected_type}"])


# ---------- TypedDict handling ----------
def _check_typeddict(obj: Any, td_cls: type, path: str) -> TypeCheckResult:
    if not isinstance(obj, dict):
        return TypeCheckResult(False, [f"{path} expected dict for TypedDict but got {type(obj).__name__}"])
    annotations = _resolved_typedict_hints(td_cls)
    total = getattr(td_cls, "__total__", True)
    required_keys = getattr(td_cls, "__required_keys__", set())
    errors: list[str] = []
    for key, typ in annotations.items():
        eff = _strip_readonly(typ)
        if key not in obj:
            if total or key in required_keys:
                errors.append(f"{path} missing required key '{key}'")
            continue
        res = check_type(obj[key], eff, f"{path}['{key}']")
        if not res.success:
            errors.extend(res.errors)
    for key in obj:
        if key not in annotations:
            errors.append(f"{path} has unexpected key '{key}'")
    return TypeCheckResult(not errors, errors)


def _check_generic_typeddict(
    obj: Any,
    origin: type,
    args: tuple,
    path: str,
    type_map: dict[TypeVar, Any] | None = None,
) -> TypeCheckResult:
    if not isinstance(obj, dict):
        return TypeCheckResult(False, [f"{path} expected dict for generic TypedDict but got {type(obj).__name__}"])
    if type_map is None:
        tvars = getattr(origin, "__parameters__", ())
        type_map = dict(zip(tvars, args))
    annotations = _resolved_typedict_hints(origin, type_map)
    total = getattr(origin, "__total__", True)
    required_keys = getattr(origin, "__required_keys__", set())
    errors: list[str] = []
    for key, typ in annotations.items():
        eff = _strip_readonly(typ)
        if key not in obj:
            if total or key in required_keys:
                errors.append(f"{path} missing required key '{key}'")
            continue
        res = check_type(obj[key], eff, f"{path}['{key}']")
        if not res.success:
            errors.extend(res.errors)
    for key in obj:
        if key not in annotations:
            errors.append(f"{path} has unexpected key '{key}'")
    return TypeCheckResult(not errors, errors)
