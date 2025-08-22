import sys
import types
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import (
    Any,
    ForwardRef,
    Literal,
    NotRequired,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

from typing_extensions import ReadOnly


@dataclass(frozen=True)
class TypeCheckResult:
    """
    Result of a type-checking operation.
    """
    success: bool
    errors: list[str]


@dataclass(frozen=True)
class UnresolvableType:
    """A placeholder for types that could not be resolved."""
    type_name: str


# ---------- helpers ----------
def _type_name(tp: Any) -> str:
    """Get a readable name for a type hint."""
    try:
        if isinstance(tp, type):
            return tp.__name__
    except Exception:
        pass
    return getattr(tp, "__qualname__", None) or str(tp)


def _is_typeddict_class(tp: object) -> bool:
    """
    Check if a type is a TypedDict class.
    """
    return isinstance(tp, type) and hasattr(tp, "__annotations__") and hasattr(tp, "__total__")


def _strip_readonly(tp: Any) -> Any:
    """
    Unpack an inner type contained in a ReadOnly declaration.
    """
    origin = get_origin(tp)
    if origin in (ReadOnly, NotRequired):
        args = get_args(tp)
        return args[0] if args else Any
    return tp


def _substitute_typevars(tp: Any, type_map: dict[TypeVar, Any]) -> Any:
    """
    Given a type and a mapping of typevars to types, substitute the typevars in the type.

    This function will recurse into nested types.

    Parameters
    ----------
    tp : Any
        The type to substitute.
    type_map : dict[TypeVar, Any]
        A mapping of typevars to types.

    Returns
    -------
    Any
        The substituted type.
    """
    if isinstance(tp, TypeVar):
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
    """
    Attempt to resolve the type hints for a typeddict.

    Parameters
    ----------
    td_cls : type
        The typeddict class.
    type_map : dict[TypeVar, Any], optional
        A mapping of typevars to types.

    Returns
    -------
    dict[str, Any]
        The resolved type hints.
    """
    try:
        # We have to resolve type hints defined in other modules
        # relative to the module-local namespace
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

def _find_generic_typeddict_base(cls: type) -> tuple[type | None, tuple[Any, ...] | None]:
    """
    Find the base class of a generic TypedDict class.

    This is necessary because the `__origin__` of a TypedDict is always `dict`
    and the `__args__` is always `(, )`. The actual base class is stored in
    `__orig_bases__`.

    Returns a tuple of `(base, args)` where `base` is the base class and `args`
    is a tuple of arguments to the base class (i.e. the key and value types of
    the TypedDict).

    Returns `(None, None)` if no base class is found.
    """
    for base in getattr(cls, "__orig_bases__", ()):
        origin = get_origin(base)
        if origin is None:
            continue
        if isinstance(origin, type) and hasattr(origin, "__annotations__"):
            return origin, get_args(base)
    return None, None

def _resolve_type(
    tp: Any,
    type_map: Mapping[TypeVar, Any] | None = None,
    globalns: Mapping[str, Any] | None=None,
    localns: Mapping[str, Any] | None=None,
    _seen: set[Any] | None = None,
) -> Any:
    """
    Resolve type hints and ForwardRef.
    """
    if _seen is None:
        _seen = set()
    tp_id = id(tp)
    if tp_id in _seen:
        return Any
    _seen.add(tp_id)

    # Strip ReadOnly
    tp = _strip_readonly(tp)

    # Substitute TypeVar
    if isinstance(tp, TypeVar):
        resolved = type_map.get(tp, tp) if type_map else tp
        if isinstance(resolved, TypeVar) and resolved is tp:
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
            # If resolution fails, return a dedicated unresolvable object.
            return UnresolvableType(str(tp))

    # Recurse into Literal
    origin = get_origin(tp)
    args = get_args(tp)
    if origin is Literal:
        # Pass literal arguments through as-is, they are values, not types to resolve.
        return Literal.__getitem__(args)

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


def check_type(obj: Any, expected_type: Any, path: str = "value") -> TypeCheckResult:
    """
    Check if `obj` is of type `expected_type`.
    """
    origin = get_origin(expected_type)

    if isinstance(expected_type, UnresolvableType):
        # Handle the custom unresolvable type placeholder
        return TypeCheckResult(False, [f"{path} has an unresolvable type: {expected_type.type_name}"])

    if expected_type is Any:
        return TypeCheckResult(True, [])

    if origin is typing.Union or isinstance(expected_type, types.UnionType):
        return check_union(obj, expected_type, path)

    if origin is typing.Literal:
        return check_literal(obj, expected_type, path)

    if expected_type is None or expected_type is type(None):
        return check_none(obj, path)

    # Check for TypedDict (now unified)
    if (origin and _is_typeddict_class(origin)) or _is_typeddict_class(expected_type):
        return _check_typeddict_unified(obj, expected_type, path)

    if origin is tuple:
        return check_tuple(obj, expected_type, path)

    if origin in (Sequence, list):
        return check_sequence_or_list(obj, expected_type, path)

    if origin in (dict, typing.Mapping) or expected_type in (dict, typing.Mapping):
        return check_mapping(obj, expected_type, path)

    if expected_type in (int, float, str, bool):
        return check_primitive(obj, expected_type, path)

    # Fallback
    try:
        if isinstance(obj, expected_type):
            return TypeCheckResult(True, [])
        tn = _type_name(expected_type)
        return TypeCheckResult(False, [f"{path} expected {tn} but got {type(obj).__name__}"])
    except TypeError:
        return TypeCheckResult(False, [f"{path} cannot be checked against {expected_type}"])


# ---------- Unified TypedDict Check Function ----------
def _check_typeddict_unified(
    obj: Any,
    td_type: Any,
    path: str,
) -> TypeCheckResult:
    """
    Check if an object matches a TypedDict, handling both generic
    and non-generic cases.

    This function determines if the provided TypedDict is a generic
    with parameters (e.g., MyTD[str]) or a regular class, and then
    performs a unified validation check.
    """
    if not isinstance(obj, dict):
        return TypeCheckResult(
            False, [f"{path} expected dict for TypedDict but got {type(obj).__name__}"]
        )

    # --- Unified logic for handling generic vs. non-generic TypedDicts ---
    origin = get_origin(td_type)
    
    if origin and _is_typeddict_class(origin):
        # Case: Generic TypedDict like MyTD[str]
        td_cls = origin
        args = get_args(td_type)
        tvars = getattr(td_cls, "__parameters__", ())
        if len(tvars) != len(args):
            return TypeCheckResult(False, [f"{path} type parameter count mismatch"])
        type_map = dict(zip(tvars, args, strict=False))
        globalns = getattr(sys.modules.get(td_cls.__module__), "__dict__", {})
        localns = dict(vars(td_cls))

    elif _is_typeddict_class(td_type):
        # Case: Non-generic TypedDict like MyTD
        td_cls = td_type
        # If it's a non-generic TypedDict, check if it inherits from a generic one
        base_origin, base_args = _find_generic_typeddict_base(td_cls)
        if base_origin is not None:
            tvars = getattr(base_origin, "__parameters__", ())
            if len(tvars) != len(base_args):
                return TypeCheckResult(False, [f"{path} type parameter count mismatch in generic base"])
            type_map = dict(zip(tvars, base_args, strict=False))
            # Get the correct global and local namespaces from the base class
            globalns = getattr(sys.modules.get(base_origin.__module__), "__dict__", {})
            localns = dict(vars(base_origin))
        else:
            type_map = None
            globalns = getattr(sys.modules.get(td_cls.__module__), "__dict__", {})
            localns = dict(vars(td_cls))

    else:
        # Fallback if it's not a TypedDict type at all
        return TypeCheckResult(False, [f"{path} expected a TypedDict but got {td_type!r}"])

    # --- Core validation logic (now unified) ---
    annotations = _resolved_typedict_hints(td_cls, type_map)
    total = getattr(td_cls, "__total__", True)
    required_keys = getattr(td_cls, "__required_keys__", set())
    errors: list[str] = []

    for key, typ in annotations.items():
        # The _resolve_type call is now universal for both cases
        eff = _resolve_type(typ, type_map, globalns=globalns, localns=localns)
        
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


def check_mapping(
    obj: Any, expected_type: Any, path: str
) -> TypeCheckResult:
    """
    Check if an object is assignable to a mapping type.
    """
    if not isinstance(obj, Mapping):
        return TypeCheckResult(
            False, [f"{path} expected Mapping but got {type(obj).__name__}"]
        )
    args = get_args(expected_type)
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
    return TypeCheckResult(len(errors) == 0, errors)

def check_sequence_or_list(
    obj: Any, expected_type: Any, path: str
) -> TypeCheckResult:
    """
    Check if an object is assignable to a sequence or list type.
    """
    args = get_args(expected_type)
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
    return TypeCheckResult(len(errors) == 0, errors)


def check_union(obj: Any, expected_type: Any, path: str) -> TypeCheckResult:
    """
    Check if an object is assignable to a union type.
    """
    args = get_args(expected_type)
    errors: list[str] = []
    for arg in args:
        res = check_type(obj, arg, path)
        if res.success:
            return TypeCheckResult(True, [])
        errors.extend(res.errors)
    return TypeCheckResult(
        False, errors or [f"{path} did not match any type in {expected_type}"])

def check_tuple(obj: Any, expected_type: Any, path: str) -> TypeCheckResult:
    """
    Check if an object is assignable to a tuple type.
    """
    if not isinstance(obj, tuple):
        return TypeCheckResult(False, [f"{path} expected tuple but got {type(obj).__name__}"])
    args = get_args(expected_type)
    targs = args
    errors: list[str] = []

    # Variadic tuple like tuple[int, ...]
    if len(targs) == 2 and targs[1] is Ellipsis:
        elem_t = targs[0]
        for i, item in enumerate(obj):
            res = check_type(item, elem_t, f"{path}[{i}]")
            if not res.success:
                errors.extend(res.errors)
        return TypeCheckResult(len(errors) == 0, errors)

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
    return TypeCheckResult(len(errors) == 0, errors)

def check_literal(obj: object, expected_type: Any, path: str) -> TypeCheckResult:
    """
    Check if an object is assignable to a literal type.
    """
    allowed = get_args(expected_type)
    if obj in allowed:
        return TypeCheckResult(True, [])
    msg = f"{path} expected literal in {allowed} but got {obj!r}"
    return TypeCheckResult(False, [msg])

def check_none(obj: object, path: str) -> TypeCheckResult:
    """
    Check if an object is None.
    """
    if obj is None:
        return TypeCheckResult(True, [])
    msg = f"{path} expected None but got {obj!r}"
    return TypeCheckResult(False, [msg])

def check_primitive(obj: object, expected_type: type, path: str) -> TypeCheckResult:
    """
    Check if an object is a primitive type, i.e. a type where isinstance(obj, type) will work.
    """
    if isinstance(obj, expected_type):
        return TypeCheckResult(True, [])
    msg = f"{path} expected an instance of {expected_type} but got {obj!r} with type {type(obj)}"
    return TypeCheckResult(
        False, [msg]
    )
