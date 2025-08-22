import collections.abc
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
    try:
        if isinstance(tp, type):
            return tp.__name__
    except Exception:
        pass
    return getattr(tp, "__qualname__", None) or str(tp)


def _parse_union_string(s: str, globalns, localns):
    # Convert "A | B | C" -> typing.Union[A, B, C]
    parts = [p.strip() for p in s.split("|")]
    resolved_parts = []
    for p in parts:
        try:
            # First try eval in the module context
            resolved_parts.append(eval(p, globalns, localns))
        except Exception:
            # fallback to Any
            resolved_parts.append(Any)
    return typing.Union[tuple(resolved_parts)]


def _is_typeddict_class(tp: Any) -> bool:
    return isinstance(tp, type) and hasattr(tp, "__annotations__") and hasattr(tp, "__total__")


def _strip_readonly(tp: Any) -> Any:
    origin = get_origin(tp)
    if origin is ReadOnly:
        args = get_args(tp)
        return args[0] if args else Any
    return tp


def _substitute_typevars(tp: Any, type_map: dict[TypeVar, Any]) -> Any:
    from typing import TypeVar as _TypeVar

    if isinstance(tp, _TypeVar):
        return type_map.get(tp, tp)

    origin = get_origin(tp)
    if origin is None:
        return tp

    args = get_args(tp)
    if not args:
        return tp

    new_args = tuple(_substitute_typevars(a, type_map) for a in args)
    try:
        return origin[new_args]
    except Exception:
        if len(new_args) == 1:
            return new_args[0]
        return tp


def _resolved_typedict_hints(
    td_cls: type, type_map: dict[TypeVar, Any] | None = None
) -> dict[str, Any]:
    try:
        mod = sys.modules.get(td_cls.__module__)
        globalns = vars(mod) if mod else {}
        localns = dict(vars(td_cls))
        hints = get_type_hints(td_cls, globalns=globalns, localns=localns, include_extras=True)
    except Exception:
        hints = getattr(td_cls, "__annotations__", {}).copy()

    if type_map:
        for k, v in list(hints.items()):
            hints[k] = _substitute_typevars(v, type_map)

    return hints


# ---------- forward reference aware resolver ----------
from typing import Any, ForwardRef, Literal, TypeVar


def _resolve_type(
    tp: Any,
    type_map: dict[TypeVar, Any] | None = None,
    globalns=None,
    localns=None,
    _seen: set | None = None,
) -> Any:
    if _seen is None:
        _seen = set()
    tp_id = id(tp)
    if tp_id in _seen:
        return Any
    _seen.add(tp_id)

    # Strip ReadOnly
    tp = _strip_readonly(tp)

    # Substitute TypeVar
    from typing import TypeVar as _TypeVar

    if isinstance(tp, _TypeVar):
        resolved = type_map.get(tp, tp) if type_map else tp
        if isinstance(resolved, _TypeVar) and resolved is tp:
            return tp  # <-- keep literal TypeVar until check
        return _resolve_type(resolved, type_map, globalns, localns, _seen)

    # Handle string-based unions safely
    if isinstance(tp, str) and " | " in tp:
        parts = [p.strip() for p in tp.split("|")]
        resolved_parts = tuple(
            _resolve_type(
                eval(p, globalns or {}, localns or {}), type_map, globalns, localns, _seen
            )
            for p in parts
        )
        return typing.Union[resolved_parts]

    # Evaluate ForwardRef
    if isinstance(tp, (ForwardRef, str)):
        try:
            ref = tp if isinstance(tp, ForwardRef) else ForwardRef(tp)
            tp = ref._evaluate(globalns or {}, localns or {}, set())
        except Exception:
            return tp  # <-- keep unresolved string/ForwardRef as-is

    # Recurse into Literal
    origin = get_origin(tp)
    args = get_args(tp)
    if origin is Literal:
        new_args = tuple(_resolve_type(a, type_map, globalns, localns, _seen) for a in args)
        return Literal.__getitem__(new_args)

    # Recurse into other generics
    if origin and args:
        new_args = tuple(_resolve_type(a, type_map, globalns, localns, _seen) for a in args)
        try:
            return origin[new_args]
        except Exception:
            if len(new_args) == 1:
                return new_args[0]
            return tp

    return tp


def _find_generic_typedict_base(cls: type) -> tuple[type | None, tuple[Any, ...] | None]:
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

    # Union
    if origin is typing.Union or isinstance(expected_type, types.UnionType):
        errors: list[str] = []
        union_args = args or (get_args(expected_type) if args else ())
        for arg in union_args:
            res = check_type(obj, arg, path)
            if res.success:
                return TypeCheckResult(True, [])
            errors.extend(res.errors)
        return TypeCheckResult(
            False, errors or [f"{path} did not match any type in {expected_type}"]
        )

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
        return TypeCheckResult(
            False, [f"{path} expected {_type_name(expected_type)} but got {type(obj).__name__}"]
        )

    # Generic TypedDict
    if (
        origin
        and isinstance(origin, type)
        and hasattr(origin, "__annotations__")
        and hasattr(origin, "__total__")
    ):
        return _check_generic_typeddict(obj, origin, args, path)

    # Non-generic TypedDict
    if _is_typeddict_class(expected_type):
        base_origin, base_args = _find_generic_typedict_base(expected_type)
        if base_origin is not None:
            type_vars = getattr(base_origin, "__parameters__", ())
            type_map = dict(zip(type_vars, base_args, strict=False))
            return _check_generic_typeddict(obj, base_origin, base_args, path, type_map=type_map)
        return _check_typeddict(obj, expected_type, path)

    # tuple[...] handled separately
    if origin is tuple:
        if not isinstance(obj, tuple):
            return TypeCheckResult(False, [f"{path} expected tuple but got {type(obj).__name__}"])
        targs = args
        errors: list[str] = []

        # Variadic tuple like tuple[int, ...]
        if len(targs) == 2 and targs[1] is Ellipsis:
            elem_t = targs[0]
            for i, item in enumerate(obj):
                res = check_type(item, elem_t, f"{path}[{i}]")
                if not res.success:
                    errors.extend(res.errors)
            return TypeCheckResult(not errors, errors)

        # Fixed-length tuple like tuple[int, str, None]
        if len(obj) != len(targs):
            return TypeCheckResult(
                False, [f"{path} expected tuple of length {len(targs)} but got {len(obj)}"]
            )
        for i, (item, tp) in enumerate(zip(obj, targs, strict=False)):
            expected = type(None) if tp is None else tp
            res = check_type(item, expected, f"{path}[{i}]")
            if not res.success:
                errors.extend(res.errors)
        return TypeCheckResult(not errors, errors)

    # Sequence / list
    if origin in (typing.Sequence, collections.abc.Sequence, list):
        if not isinstance(obj, typing.Sequence) or isinstance(obj, (str, bytes)):
            return TypeCheckResult(
                False, [f"{path} expected sequence but got {type(obj).__name__}"]
            )
        elem_type = args[0] if args else Any
        errors: list[str] = []
        for i, item in enumerate(obj):
            res = check_type(item, elem_type, f"{path}[{i}]")
            if not res.success:
                errors.extend(res.errors)
        return TypeCheckResult(not errors, errors)

    # Mapping / dict[K, V]
    if origin in (dict, typing.Mapping) or expected_type in (dict, typing.Mapping):
        if not isinstance(obj, dict) and not isinstance(obj, typing.Mapping):
            return TypeCheckResult(
                False, [f"{path} expected dict/mapping but got {type(obj).__name__}"]
            )
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

    # Fallback
    try:
        if isinstance(obj, expected_type):
            return TypeCheckResult(True, [])
        tn = _type_name(expected_type)
        return TypeCheckResult(False, [f"{path} expected {tn} but got {type(obj).__name__}"])
    except TypeError:
        return TypeCheckResult(False, [f"{path} cannot be checked against {expected_type}"])


# ---------- TypedDict handling ----------
def _check_typeddict(obj: Any, td_cls: type, path: str) -> TypeCheckResult:
    if not isinstance(obj, dict):
        return TypeCheckResult(
            False, [f"{path} expected dict for TypedDict but got {type(obj).__name__}"]
        )

    globalns = getattr(sys.modules.get(td_cls.__module__), "__dict__", {})
    localns = dict(vars(td_cls))
    annotations = _resolved_typedict_hints(td_cls)
    total = getattr(td_cls, "__total__", True)
    required_keys = getattr(td_cls, "__required_keys__", set())
    errors: list[str] = []

    for key, typ in annotations.items():
        eff = _resolve_type(typ, globalns=globalns, localns=localns)
        eff = _strip_readonly(eff)  # <-- strip ReadOnly here
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
        return TypeCheckResult(
            False, [f"{path} expected dict for generic TypedDict but got {type(obj).__name__}"]
        )

    if type_map is None:
        tvars = getattr(origin, "__parameters__", ())
        if len(tvars) != len(args):
            return TypeCheckResult(False, [f"{path} type parameter count mismatch"])
        type_map = dict(zip(tvars, args, strict=False))

    globalns = getattr(sys.modules.get(origin.__module__), "__dict__", {})
    localns = dict(vars(origin))
    annotations = _resolved_typedict_hints(origin, type_map)
    total = getattr(origin, "__total__", True)
    required_keys = getattr(origin, "__required_keys__", set())
    errors: list[str] = []

    for key, typ in annotations.items():
        eff = _resolve_type(typ, type_map, globalns=globalns, localns=localns)
        eff = _strip_readonly(eff)  # <-- strip ReadOnly here
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
