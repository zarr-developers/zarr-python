import collections
import collections.abc
import sys
import types
import typing
from dataclasses import dataclass
from typing import (
    Any,
    ForwardRef,
    Literal,
    NotRequired,
    TypeGuard,
    TypeVar,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from typing_extensions import ReadOnly, evaluate_forward_ref


class TypeResolutionError(Exception): ...


@dataclass(frozen=True)
class TypeCheckResult:
    """
    Result of a type-checking operation.
    """

    success: bool
    errors: list[str]


# ---------- helpers ----------
def _type_name(tp: Any) -> str:
    """Get a readable name for a type hint."""
    if isinstance(tp, type):
        return tp.__name__
    return str(tp)


def _is_typeddict_class(tp: object) -> bool:
    """
    Check if a type is a TypedDict class.
    """
    return isinstance(tp, type) and hasattr(tp, "__annotations__") and hasattr(tp, "__total__")


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
    type_map: dict[TypeVar, Any] | None = None,
    globalns: dict[str, Any] | None = None,
    localns: dict[str, Any] | None = None,
    _seen: set[Any] | None = None,
) -> Any:
    """
    Resolve type hints and ForwardRef. Maintains a cache of resolved types to avoid infinite recursion.
    """
    if _seen is None:
        _seen = set()

    # Use a more robust tracking mechanism
    type_repr = repr(tp)
    if type_repr in _seen:
        # Return Any for recursive types to break the cycle
        return Any

    _seen.add(type_repr)

    try:
        return _resolve_type_impl(tp, type_map, globalns, localns, _seen)
    finally:
        _seen.discard(type_repr)


def _resolve_type_impl(
    tp: Any,
    type_map: dict[TypeVar, Any] | None,
    globalns: dict[str, Any] | None,
    localns: dict[str, Any] | None,
    _seen: set[str],
) -> Any:
    """
    Internal implementation of type resolution.
    """
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
        return typing.Union[resolved_parts]  # noqa: UP007

    # Evaluate ForwardRef
    if isinstance(tp, (ForwardRef, str)):
        ref = tp if isinstance(tp, ForwardRef) else ForwardRef(tp)
        # Use frozenset to avoid issues with mutable default arguments
        tp = evaluate_forward_ref(ref, globals=globalns, locals=localns)

    # Recurse into Literal
    origin = get_origin(tp)
    args = get_args(tp)
    if origin is Literal:
        # Pass literal arguments through as-is, they are values, not types to resolve.
        return Literal.__getitem__(args)

    # Handle types.UnionType (Python 3.10+ union syntax like str | int)
    if isinstance(tp, types.UnionType):
        # Don't try to reconstruct UnionType, convert to typing.Union
        resolved_args = tuple(_resolve_type(a, type_map, globalns, localns, _seen) for a in args)
        return typing.Union[resolved_args]  # noqa: UP007

    # Recurse into other generics
    if origin and args:
        new_args = tuple(_resolve_type(a, type_map, globalns, localns, _seen) for a in args)
        # Special handling for single-argument generics like NotRequired, ReadOnly
        if len(new_args) == 1:
            return origin[new_args[0]]  # Pass single argument, not tuple
        else:
            return origin[new_args]  # Pass tuple for multi-argument generics

    return tp


def check_type(obj: Any, expected_type: Any, path: str = "value") -> TypeCheckResult:
    """
    Check if `obj` is of type `expected_type`.
    """
    origin = get_origin(expected_type)

    if origin in (NotRequired, ReadOnly):
        args = get_args(expected_type)
        inner_type = args[0] if args else Any
        return check_type(obj, inner_type, path)

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
        return check_typeddict(obj, expected_type, path)

    if origin is tuple:
        return check_tuple(obj, expected_type, path)

    if origin in (collections.abc.Sequence, list):
        return check_sequence_or_list(obj, expected_type, path)

    if origin in (dict, typing.Mapping, collections.abc.Mapping) or expected_type in (
        dict,
        typing.Mapping,
        collections.abc.Mapping,
    ):
        return check_mapping(obj, expected_type, path)

    if expected_type is int:
        return check_int(obj, path)

    if expected_type in (float, str, bool):
        return check_primitive(obj, expected_type, path)

    # Fallback
    try:
        if isinstance(obj, expected_type):
            return TypeCheckResult(True, [])
        tn = _type_name(expected_type)
        return TypeCheckResult(False, [f"{path} expected {tn} but got {type(obj).__name__}"])
    except TypeError:
        return TypeCheckResult(False, [f"{path} cannot be checked against {expected_type}"])


T = TypeVar("T")


def ensure_type(obj: object, expected_type: type[T], path: str = "value") -> T:
    """
    Check if obj is assignable to expected type. If so, return obj. Otherwise a TypeError is raised.
    """
    if check_type(obj, expected_type, path).success:
        return cast(T, obj)
    raise TypeError(
        f"Expected an instance of {expected_type} but got {obj!r} with type {type(obj)}"
    )


def guard_type(obj: object, expected_type: type[T], path: str = "value") -> TypeGuard[T]:
    """
    A type guard function that checks if obj is assignable to expected type.
    """
    return check_type(obj, expected_type, path).success


def check_typeddict(
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

    # --- Now get the metadata in a single, unified step ---
    td_cls, type_map, globalns, localns = _get_typeddict_metadata(td_type)

    if td_cls is None:
        # Fallback if it's not a TypedDict type at all
        return TypeCheckResult(False, [f"{path} expected a TypedDict but got {td_type!r}"])

    if type_map is not None and len(getattr(td_cls, "__parameters__", ())) != len(
        get_args(td_type)
    ):
        return TypeCheckResult(False, [f"{path} type parameter count mismatch"])

    if type_map is None and len(get_args(td_type)) > 0:
        base_origin, base_args = _find_generic_typeddict_base(td_cls)
        if (
            base_origin is not None
            and base_args is not None
            and len(getattr(base_origin, "__parameters__", ())) != len(base_args)
        ):
            return TypeCheckResult(False, [f"{path} type parameter count mismatch in generic base"])

    # --- Now call the shared validation logic ---
    errors = _validate_typeddict_fields(obj, td_cls, type_map, globalns, localns, path)

    return TypeCheckResult(not errors, errors)


def check_mapping(obj: Any, expected_type: Any, path: str) -> TypeCheckResult:
    """
    Check if an object is assignable to a mapping type.
    """
    if not isinstance(obj, collections.abc.Mapping):
        return TypeCheckResult(
            False, [f"{path} expected  collections.abc.Mapping but got {type(obj).__name__}"]
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


def check_sequence_or_list(obj: Any, expected_type: Any, path: str) -> TypeCheckResult:
    """
    Check if an object is assignable to a sequence or list type.
    """
    args = get_args(expected_type)
    if not isinstance(obj, typing.Sequence | collections.abc.Sequence) or isinstance(
        obj, (str, bytes)
    ):
        return TypeCheckResult(False, [f"{path} expected sequence but got {type(obj).__name__}"])
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
    return TypeCheckResult(False, errors or [f"{path} did not match any type in {expected_type}"])


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
    return TypeCheckResult(False, [msg])


def check_int(obj: object, path: str) -> TypeCheckResult:
    """
    Check if an object is an int.
    """
    if isinstance(obj, int) and not isinstance(obj, bool):  # bool is a subclass of int
        return TypeCheckResult(True, [])
    msg = f"{path} expected int but got {obj!r} with type {type(obj)}"
    return TypeCheckResult(False, [msg])


def _get_typeddict_metadata(
    td_type: Any,
) -> tuple[
    type | None,
    dict[TypeVar, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    """
    Extracts the TypedDict class, type variable map, and namespaces.
    """
    origin = get_origin(td_type)

    if origin and _is_typeddict_class(origin):
        td_cls = origin
        args = get_args(td_type)
        tvars = getattr(td_cls, "__parameters__", ())
        type_map = dict(zip(tvars, args, strict=False))

        # Enhanced namespace resolution - include calling frame locals
        mod = sys.modules.get(td_cls.__module__)
        globalns = vars(mod) if mod else {}
        localns = dict(vars(td_cls))

        return td_cls, type_map, globalns, localns

    elif _is_typeddict_class(td_type):
        td_cls = td_type
        base_origin, base_args = _find_generic_typeddict_base(td_cls)
        if base_origin is not None:
            tvars = getattr(base_origin, "__parameters__", ())
            type_map = dict(zip(tvars, base_args, strict=False))  # type: ignore[arg-type]

            mod = sys.modules.get(base_origin.__module__)
            globalns = vars(mod) if mod else {}
            localns = dict(vars(base_origin))
        else:
            type_map = None
            mod = sys.modules.get(td_cls.__module__)
            globalns = vars(mod) if mod else {}
            localns = dict(vars(td_cls))

        return td_cls, type_map, globalns, localns

    return None, None, None, None


def _validate_typeddict_fields(
    obj: Any,
    td_cls: type,
    type_map: dict[TypeVar, Any] | None,
    globalns: dict[str, Any] | None,
    localns: dict[str, Any] | None,
    path: str,
) -> list[str]:
    """
    Validates the fields of a dictionary against a TypedDict's annotations.
    """
    annotations = get_type_hints(td_cls, globalns=globalns, localns=localns, include_extras=True)
    errors: list[str] = []
    is_total_false = getattr(td_cls, "__total__", True) is False
    for key, typ in annotations.items():
        # Check if the key is not present in the object
        if key not in obj:
            # If total=False, all fields are optional unless explicitly Required
            if is_total_false:
                continue

            # Check the chain of parametrized types for a NotRequired.
            # We only need to look at the first parameter.
            is_optional = False
            if get_origin(typ) == NotRequired:
                is_optional = True
            else:
                sub_args = get_args(typ)
                while len(sub_args) > 0:
                    if get_origin(sub_args[0]) == NotRequired:
                        is_optional = True
                        break
                    sub_args = get_args(sub_args[0])

            if not is_optional:
                errors.append(f"{path} missing required key '{key}'")
            continue

        #  we have to further resolve this type because get_type_hints does not resolve
        # generic aliases
        resolved_typ = _resolve_type(typ, type_map, globalns=globalns, localns=localns)
        res = check_type(obj[key], resolved_typ, f"{path}['{key}']")
        if not res.success:
            errors.extend(res.errors)

    # We allow extra keys of any type right now
    # when PEP 728 is done, then we can refine this and do a type check on the keys
    # errors.extend([f"{path} has unexpected key '{key}'" for key in obj if key not in annotations])

    return errors
