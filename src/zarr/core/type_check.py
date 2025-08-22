from __future__ import annotations
import collections.abc
import types  # NEW
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, TypedDict, get_origin, get_args, Union, Literal, Mapping
from typing_extensions import ReadOnly as TE_ReadOnly  # optional, for robust detection

# add imports
import sys
from typing import get_type_hints

# --- helper: resolve annotations for (generic) TypedDicts ------------------

def _resolved_typedict_hints(td_cls: type) -> dict[str, Any]:
    """Return fully-resolved annotations for a TypedDict class.

    Works with deferred annotations (from __future__ import annotations)
    and preserves extras like ReadOnly[...] for later stripping.
    """
    try:
        mod = sys.modules.get(td_cls.__module__)
        globalns = vars(mod) if mod else None
        localns = dict(vars(td_cls))
        return get_type_hints(td_cls, globalns=globalns, localns=localns, include_extras=True)
    except Exception:
        # Fall back to raw annotations if resolution fails for any reason
        return getattr(td_cls, "__annotations__", {})

@dataclass(frozen=True)
class TypeCheckResult:
    success: bool
    errors: list[str]

def _is_readonly_origin(origin: Any) -> bool:
    """Return True if origin refers to typing_extensions.ReadOnly."""
    if origin is None:
        return False
    if TE_ReadOnly is not None and origin is TE_ReadOnly:
        return True
    # Fallback: compare by name/module to avoid hard dependency on typing_extensions
    return getattr(origin, "__name__", "") == "ReadOnly" or str(origin).endswith("ReadOnly")

def _strip_readonly(tp: Any) -> Any:
    """If tp is ReadOnly[T], return T; otherwise return tp."""
    origin = get_origin(tp)
    if _is_readonly_origin(origin):
        args = get_args(tp)
        return args[0] if args else Any
    return tp

def _substitute_typevars(tp: Any, type_map: dict[TypeVar, Any]) -> Any:
    """Substitute a TypeVar with its concrete type if present in type_map."""
    if isinstance(tp, TypeVar):
        return type_map.get(tp, tp)
    # If tp is ReadOnly[TVar], unwrap then substitute
    origin = get_origin(tp)
    if _is_readonly_origin(origin):
        inner = _strip_readonly(tp)
        return _substitute_typevars(inner, type_map)
    return tp

def _is_typeddict_class(tp: Any) -> bool:
    """Return True iff tp is a (possibly generic) TypedDict class."""
    # TypedDict subclasses have these runtime attributes; using issubclass(...) raises TypeError.
    return isinstance(tp, type) and hasattr(tp, "__annotations__") and hasattr(tp, "__total__")

def check_type(obj: Any, expected_type: Any, path: str = "value") -> TypeCheckResult:
    """Main entry point for type checking."""
    if expected_type is Any:
        return TypeCheckResult(True, [])

    origin = get_origin(expected_type)
    args = get_args(expected_type)

    # Union / Optional (support both typing.Union and PEP 604 unions)
    if origin in (Union, types.UnionType):  # CHANGED
        errors: list[str] = []
        for arg in args:
            res = check_type(obj, arg, path)
            if res.success:
                return res
            errors.extend(res.errors)
        return TypeCheckResult(False, errors or [f"{path} did not match any type in {expected_type}"])

    # TypedDict (generic and non-generic) — use safe detector
    if origin and _is_typeddict_class(origin):  # CHANGED
        return _check_generic_typeddict(obj, origin, args, path)
    if _is_typeddict_class(expected_type):  # CHANGED
        return _check_typeddict(obj, expected_type, path)

    # Literal
    if origin is Literal:
        if obj in args:
            return TypeCheckResult(True, [])
        return TypeCheckResult(False, [f"{path} expected literal in {args} but got {obj!r}"])

    # None
    if expected_type is None:
        if obj is None:
            return TypeCheckResult(True, [])
        return TypeCheckResult(False, [f"{path} expected None but got {type(obj).__name__}"])

    # Primitives
    if expected_type in (int, float, str, bool):
        if isinstance(obj, expected_type):
            return TypeCheckResult(True, [])
        return TypeCheckResult(False, [f"{path} expected {expected_type.__name__} but got {type(obj).__name__}"])

    # Tuple
    if origin is tuple:
        if not isinstance(obj, tuple):
            return TypeCheckResult(False, [f"{path} expected tuple but got {type(obj).__name__}"])
        if len(args) == 2 and args[1] is ...:
            elem_type = args[0]
            errors: list[str] = []
            for i, item in enumerate(obj):
                res = check_type(item, elem_type, f"{path}[{i}]")
                if not res.success:
                    errors.extend(res.errors)
            return TypeCheckResult(not errors, errors)
        if len(obj) != len(args):
            return TypeCheckResult(False, [f"{path} expected tuple of length {len(args)} but got {len(obj)}"])
        errors: list[str] = []
        for i, (item, typ) in enumerate(zip(obj, args)):
            res = check_type(item, typ, f"{path}[{i}]")
            if not res.success:
                errors.extend(res.errors)
        return TypeCheckResult(not errors, errors)

    # Sequence (list, etc.)
    if origin in (list, collections.abc.Sequence):
        if not isinstance(obj, collections.abc.Sequence):
            return TypeCheckResult(False, [f"{path} expected a sequence but got {type(obj).__name__}"])
        if isinstance(obj, (str, bytes)):
            return TypeCheckResult(False, [f"{path} expected a non-string sequence but got {type(obj).__name__}"])
        elem_type = args[0] if args else Any
        errors: list[str] = []
        for i, item in enumerate(obj):
            res = check_type(item, elem_type, f"{path}[{i}]")
            if not res.success:
                errors.extend(res.errors)
        return TypeCheckResult(not errors, errors)

    # Mapping (dict)
    if origin in (dict, collections.abc.Mapping):
        if not isinstance(obj, collections.abc.Mapping):
            return TypeCheckResult(False, [f"{path} expected a mapping but got {type(obj).__name__}"])
        key_type, val_type = args if args else (Any, Any)
        errors: list[str] = []
        for k, v in obj.items():
            res_key = check_type(k, key_type, f"{path} key {repr(k)}")
            res_val = check_type(v, val_type, f"{path}[{repr(k)}]")
            if not res_key.success:
                errors.extend(res_key.errors)
            if not res_val.success:
                errors.extend(res_val.errors)
        return TypeCheckResult(not errors, errors)

    # Fallback for regular classes; avoid TypeError on typing aliases / subscripted generics
    try:
        if isinstance(obj, expected_type):
            return TypeCheckResult(True, [])
    except TypeError:
        pass
    return TypeCheckResult(False, [f"{path} expected {expected_type} but got {type(obj).__name__}"])

def _check_typeddict(obj: Any, expected_type: type, path: str) -> TypeCheckResult:
    if not isinstance(obj, dict):
        return TypeCheckResult(False, [f"{path} expected dict for TypedDict but got {type(obj).__name__}"])

    # RESOLVED annotations instead of raw __annotations__
    annotations = _resolved_typedict_hints(expected_type)
    total = getattr(expected_type, "__total__", True)
    required_keys = getattr(expected_type, "__required_keys__", set())

    errors: list[str] = []
    for key, typ in annotations.items():
        eff_type = _strip_readonly(typ)
        if key not in obj:
            if total or key in required_keys:
                errors.append(f"{path} missing required key '{key}'")
        else:
            res = check_type(obj[key], eff_type, f"{path}['{key}']")
            if not res.success:
                errors.extend(res.errors)

    for key in obj:
        if key not in annotations:
            errors.append(f"{path} has unexpected key '{key}'")

    return TypeCheckResult(not errors, errors)

def _check_generic_typeddict(obj: Any, origin: type, args: tuple, path: str) -> TypeCheckResult:
    if not isinstance(obj, dict):
        return TypeCheckResult(False, [f"{path} expected dict for generic TypedDict but got {type(obj).__name__}"])

    type_vars = getattr(origin, "__parameters__", ())
    type_map = dict(zip(type_vars, args))

    # RESOLVED annotations here too
    annotations = _resolved_typedict_hints(origin)
    total = getattr(origin, "__total__", True)
    required_keys = getattr(origin, "__required_keys__", set())

    errors: list[str] = []
    for key, typ in annotations.items():
        base = _strip_readonly(typ)
        eff_type = _substitute_typevars(base, type_map)

        if key not in obj:
            if total or key in required_keys:
                errors.append(f"{path} missing required key '{key}'")
        else:
            res = check_type(obj[key], eff_type, f"{path}['{key}']")
            if not res.success:
                errors.extend(res.errors)

    for key in obj:
        if key not in annotations:
            errors.append(f"{path} has unexpected key '{key}']")

    return TypeCheckResult(not errors, errors)
