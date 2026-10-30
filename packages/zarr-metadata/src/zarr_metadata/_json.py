"""JSON values and the problems found in them. This is the lowest layer of the reader.

`refine_json` is the single walk that both normalizes and checks a value.
`refine_user_data` is the same walk over user data, where a non-finite
number is kept as a float.
Every layer (the checker, the definitions and the model) reports with
`ValidationProblem` and `MetadataValidationError`.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, TypeAlias, TypeGuard, TypeVar, cast, get_args, overload

from zarr_metadata._common import JSONValue
from zarr_metadata._sentinel import UNSET

ProblemKind = Literal["missing_key", "invalid_type", "invalid_value", "invalid_json", "unknown_key"]
"""Machine-readable classification of a `ValidationProblem`.

- `missing_key`: a required key (document key or store key) is absent.
- `invalid_type`: a value has the wrong structural type (e.g. a string where
  a mapping is required, a non-JSON-serializable object).
- `invalid_value`: a value has an acceptable type but an invalid content
  (e.g. `zarr_format: 2` in a v3 document, `order: "Q"`).
- `invalid_json`: bytes that do not decode as JSON.
- `unknown_key`: a key that an object's type does not declare, where the
  type is closed (as a closed TypedDict is). Zarr configurations rarely
  state whether they are closed (zarr-developers/zarr-specs#270 has been
  open since 2023), and many readers reject such a key. It has its own
  kind so that a caller who tolerates it can distinguish it from a wrong
  value, and so that it never hides other problems in the same object.
"""


PROBLEM_KINDS: Final[tuple[ProblemKind, ...]] = get_args(ProblemKind)
"""The valid problem kinds. `ValidationProblem` checks its `kind` against these."""


JSON_DEPTH: Final = 256
"""The maximum number of nesting levels a reader walks.

A value nested deeper is a problem at the first level past the limit.
This means no document, however deep, makes a reader exceed the
interpreter's recursion limit: the package refines every value before
reading it, and refining stops at this depth. Every reader, writer and
comparison uses one stack frame per level, and `copy.deepcopy` (and
`pickle` before Python 3.12) uses two. A document at the limit therefore
uses about half of the interpreter's default limit of a thousand frames,
and the rest is left for the caller.
"""

_PAST_THE_LEVELS: Final = f"nested deeper than the {JSON_DEPTH} levels a reader walks"
"""The message for a container nested past `JSON_DEPTH`."""


class _Ctx(Mapping[str, JSONValue]):
    """The type of a problem's `ctx`: a private copy, with arrays as tuples, checked to be JSON when created and never modified after.

    It is a separate type so that a problem built from another problem's
    `ctx` (as `replace` builds one) can tell the value was already checked
    and skip the check. A mapping of any other type is checked and copied.
    """

    __slots__ = ("_held",)

    def __init__(self, held: dict[str, JSONValue]) -> None:
        self._held = held

    def __getitem__(self, key: str) -> JSONValue:
        return self._held[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._held)

    def __len__(self) -> int:
        return len(self._held)

    def __repr__(self) -> str:
        return repr(self._held)

    def __reduce__(self) -> tuple[type[_Ctx], tuple[dict[str, JSONValue]]]:
        return _Ctx, (self._held,)


_NO_CTX: Final[Mapping[str, JSONValue]] = _Ctx({})


def _no_ctx() -> Mapping[str, JSONValue]:
    """The empty `ctx`, for a problem that states no more than the expected type."""
    return _NO_CTX


@dataclass(frozen=True, slots=True)
class ValidationProblem:
    """A single problem found in a value: its location, what is wrong, the kind of problem, and the data used in the message.

    `loc` is the path from the root of the checked value to the bad value,
    e.g. `("codecs", 0, "name")` in a document. An empty `loc` refers to
    the root.
    `kind` classifies the failure for programmatic dispatch; `message`
    is the human-readable description.

    `input` and `ctx` hold the data in the message, in the same way that
    pydantic's `ErrorDetails` and zod's issues do. `input` is the JSON
    found at `loc`, e.g. `12` for a gzip `level` of 12. It is `UNSET` when
    nothing is there (zod does the same for a missing key; pydantic gives
    the object that lacks the key), or when the value there is not JSON,
    in which case the message shows it. `input` is the object the caller
    passed in, not a copy, as in pydantic: if the caller later changes its
    document, `input` shows the change. `ctx` holds what was expected,
    when that is more than a type:

    - `gt`, `ge`, `lt` and `le`: the bounds from the value's type or from
      a rule, with the names pydantic uses. For example `{"ge": 0, "le": 9}`
      for a gzip `level`, whose type is `Annotated[int, Interval(ge=0, le=9)]`.
    - `expected`: the values of a closed set, like zod's `values`, in the
      order the message lists them. For example the values of a `Literal`,
      of `node_type`, or of `zarr_format`.

    Neither field takes part in equality or the repr: two problems are
    equal when they have the same location, message and kind. Every
    function that returns or raises problems fills `input` from the value
    its caller passed, so a rule only needs to give the location.
    """

    loc: tuple[str | int, ...]
    message: str
    kind: ProblemKind
    input: JSONValue | UNSET = dataclasses.field(
        default=UNSET, kw_only=True, compare=False, repr=False
    )
    ctx: Mapping[str, JSONValue] = dataclasses.field(
        default_factory=_no_ctx, kw_only=True, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        # Enforce the annotations at runtime. A rule written without a type
        # checker (an extension's rule may be) then fails where it builds
        # the problem, instead of reporting a problem at an invalid location.
        loc = cast("object", self.loc)
        if not isinstance(loc, tuple) or not all(
            isinstance(part, str) or (isinstance(part, int) and not isinstance(part, bool))
            for part in cast("tuple[object, ...]", loc)
        ):
            msg = f"a ValidationProblem's loc is a tuple of keys and indices, got {loc!r}"
            raise TypeError(msg)
        message = cast("object", self.message)
        if not isinstance(message, str):
            msg = f"a ValidationProblem's message is a string, got {message!r}"
            raise TypeError(msg)
        kind = cast("object", self.kind)
        if not isinstance(kind, str) or kind not in PROBLEM_KINDS:
            msg = f"a ValidationProblem's kind is one of {PROBLEM_KINDS!r}, got {kind!r}"
            raise TypeError(msg)
        ctx = cast("object", self.ctx)
        if isinstance(ctx, _Ctx):
            # Another problem's `ctx`, already checked when that problem was
            # created. `replace` passes it to the copy.
            return
        if (
            not isinstance(ctx, Mapping)
            or not all(isinstance(key, str) for key in cast("Mapping[object, object]", ctx))
            or not is_json(dict(cast("Mapping[str, object]", ctx)))
        ):
            msg = f"a ValidationProblem's ctx is an object of JSON values, got {ctx!r}"
            raise TypeError(msg)
        # Store a view of a private deep copy, with arrays as tuples. Then
        # neither this view nor the caller's original mapping can modify a
        # raised error, which is a finished report.
        held = cast(
            "dict[str, JSONValue]",
            copied(cast("JSONValue", arrays_to_tuples(dict(cast("Mapping[str, object]", ctx))))),
        )
        object.__setattr__(self, "ctx", _Ctx(held))

    def __str__(self) -> str:
        location = ".".join(str(part) for part in self.loc) if self.loc else "<root>"
        return f"{location}: {self.message}"

    def __reduce__(
        self,
    ) -> tuple[Callable[..., ValidationProblem], tuple[object, ...]]:
        # Pickle and copy by calling the constructor again. The view that
        # holds `ctx` does not pickle, but the dict it wraps does.
        return (_problem, (self.loc, self.message, self.kind, self.input, dict(self.ctx)))


Problems: TypeAlias = tuple[ValidationProblem, ...]
"""Every problem a reader found, in the order it found them."""


def _problem(
    loc: tuple[str | int, ...],
    message: str,
    kind: ProblemKind,
    found: JSONValue | UNSET,
    ctx: Mapping[str, JSONValue],
) -> ValidationProblem:
    """Rebuild a problem from the values `ValidationProblem.__reduce__` returns."""
    return ValidationProblem(loc, message, kind, input=found, ctx=ctx)


class MetadataValidationError(ValueError):
    """Raised when a value fails validation, by the entry points that raise instead of returning problems.

    Carries every problem found (not just the first) in `.problems`, as an
    immutable tuple: a raised error is a finished report, and a caller
    inspecting it must not be able to edit the record.
    """

    problems: tuple[ValidationProblem, ...]

    def __init__(self, problems: Sequence[ValidationProblem]) -> None:
        self.problems = tuple(problems)
        for entry in cast("tuple[object, ...]", self.problems):
            # Enforce the annotation at runtime. A caller that is not
            # type-checked and passes something else fails here, instead of
            # later, at the code that reads a `loc` from it.
            if not isinstance(entry, ValidationProblem):
                msg = (
                    "MetadataValidationError takes ValidationProblem values, "
                    f"got {type(entry).__name__}"
                )
                raise TypeError(msg)
        super().__init__("\n".join(str(problem) for problem in self.problems))

    def __reduce__(
        self,
    ) -> tuple[
        type[MetadataValidationError], tuple[tuple[ValidationProblem, ...]], dict[str, object]
    ]:
        # By default an exception pickles and copies by calling its class
        # with its `args`, which here hold the message. This one is rebuilt
        # from its problems instead, and the rest of its state (including
        # its notes) is restored after.
        return (type(self), (self.problems,), self.__dict__)


def prefixed(
    loc_head: str | int, problems: Sequence[ValidationProblem]
) -> tuple[ValidationProblem, ...]:
    """Prepend `loc_head` to the `loc` of every problem (for nested validators)."""
    return tuple(dataclasses.replace(p, loc=(loc_head, *p.loc)) for p in problems)


def value_at(value: object, loc: tuple[str | int, ...]) -> object:
    """The value inside `value` at `loc`, where each key names an object member and each index an array element; `UNSET` if nothing is there."""
    for part in loc:
        if isinstance(part, str) and is_object(value):
            members = value
            if part not in members:
                return UNSET
            value = members[part]
        elif isinstance(part, int) and is_array(value) and 0 <= part < len(value):
            value = value[part]
        else:
            return UNSET
    return value


def with_input(
    problems: Sequence[ValidationProblem], value: object, loc: tuple[str | int, ...] = ()
) -> tuple[ValidationProblem, ...]:
    """Set each problem's `input` to the JSON in `value` at the problem's `loc`. `value` itself sits at `loc`.

    Every function that checks a value calls this on the problems it found.
    This way a rule only gives the location of a problem, and the problem
    carries the value found there. The function the caller called is the
    last to apply it, so a problem's `input` comes from the value the
    caller passed in, not from a copy made by an inner reader. If a
    problem's `loc` points to nothing in `value` (a missing key), or to
    something that is not JSON, the problem keeps its current `input`:
    `UNSET`, unless an inner reader found JSON there. This ensures no
    problem holds a value that might not pickle or copy.
    """
    if len(problems) == 0:
        return ()
    filled: list[ValidationProblem] = []
    for found in problems:
        if found.loc[: len(loc)] == loc:
            there = value_at(value, found.loc[len(loc) :])
            if there is not found.input and _holdable(there):
                found = dataclasses.replace(found, input=cast("JSONValue", there))
        filled.append(found)
    return tuple(filled)


def _holdable(value: object) -> bool:
    """Whether a problem can hold `value` as its input: JSON, nested no deeper than `JSON_DEPTH`.

    A value the validator did not walk (a member it only reports) may be
    nested deeper than that. Such a value is not stored, because
    `is_canonical_json` rejects it.
    """
    return is_canonical_json(value, finite=False)


def not_an_object(value: object) -> tuple[ValidationProblem, ...]:
    """The single problem, at the root, for a document that is not an object."""
    return with_input((ValidationProblem((), "expected an object", "invalid_type"),), value)


def validate_json(value: object, loc: tuple[str | int, ...] = ()) -> tuple[ValidationProblem, ...]:
    """Return every reason that `value`, located at `loc`, is not JSON, each at its own location: a non-finite float, a non-string key, a value of no JSON type, or nesting deeper than `JSON_DEPTH`, counted from the document root above `loc`."""
    return with_input(refine_json(value, loc)[1], value, loc)


def is_object(value: object) -> TypeGuard[Mapping[object, object]]:
    """Whether `value` is a JSON object as Python represents one: a mapping with any keys.

    The same test as `isinstance(value, Mapping)`, but it narrows the type
    to a mapping of `object`. The bare check leaves the key and value types
    unknown to a type checker.
    """
    return type(value) is dict or isinstance(value, Mapping)


def refined_object(value: object) -> dict[str, JSONValue]:
    """Refine `value`, an object that a read found no problems in, as user data. The result is the document a model holds.

    Raises `TypeError` if `value` is not an object. This cannot happen
    after a read that found no problems.
    """
    refined, _ = refine_user_data(value)
    if not isinstance(refined, dict):
        msg = f"expected an object a read found nothing wrong with, got {shown(value)}"
        raise TypeError(msg)
    return refined


def object_at(document: Mapping[str, JSONValue], key: str) -> dict[str, JSONValue]:
    """The object at `key` in the refined `document`, where a read found one; `TypeError` if the value is not an object, which that read rules out."""
    value = document[key]
    if not isinstance(value, dict):
        msg = f"expected an object at {key!r}, got {shown(value)}"
        raise TypeError(msg)
    return value


def is_json_object(value: object) -> TypeGuard[Mapping[str, object]]:
    """Whether `value` is a JSON object with string keys, as JSON requires: a mapping whose keys are all strings."""
    return isinstance(value, Mapping) and all(
        isinstance(key, str) for key in cast("Mapping[object, object]", value)
    )


def is_list_or_tuple(value: object) -> TypeGuard[list[object] | tuple[object, ...]]:
    """Whether `value` is a list or a tuple: the two containers canonical JSON uses for an array."""
    return isinstance(value, (list, tuple))


def is_tuple(value: object) -> TypeGuard[tuple[object, ...]]:
    """Whether `value` is a tuple, narrowed to a tuple of `object`."""
    return isinstance(value, tuple)


def is_array(value: object) -> TypeGuard[Sequence[object]]:
    """Whether `value` is a JSON array as Python represents one: a sequence that is not text or bytes."""
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def refine_json(
    value: object, loc: tuple[str | int, ...] = ()
) -> tuple[JSONValue | None, tuple[ValidationProblem, ...]]:
    """`value` as JSON with arrays as tuples, or None with every reason it is not JSON.

    A single walk both normalizes and checks: a mapping becomes a `dict`
    with string keys, a sequence becomes a tuple, and a float must be
    finite. For a value that is not JSON, the result is None, with the
    problems located at the leaves that are not JSON. JSON's `null` is
    also None, but with no problems, so `value` is JSON exactly when there
    are no problems.
    """
    return _refine(value, loc, finite=True)


def refine_user_data(
    value: object, loc: tuple[str | int, ...] = ()
) -> tuple[JSONValue | None, tuple[ValidationProblem, ...]]:
    """Refine user data the same way `refine_json` refines JSON, but keep a non-finite number as a float.

    A node's attributes are user data. The spec only requires each to be a
    JSON value, and does not interpret any of them. zarr-python writes them
    with the defaults of Python's `json` module, so an attribute can hold
    `NaN`, `Infinity` or `-Infinity` (for example xarray's `_FillValue` or
    a CF `missing_value`), which RFC 8259 does not allow. Wherever the spec
    interprets a value, it writes those numbers as strings. So a validator
    uses `refine_json` for values the spec interprets, and this function
    only for user data.
    """
    return _refine(value, loc, finite=False)


_Refined = tuple[JSONValue | None, tuple[ValidationProblem, ...]]


def nested_past_the_levels(value: object, loc: tuple[str | int, ...]) -> ValidationProblem | None:
    """The problem for `value` if it is a container (an object or an array) at `loc` deeper than `JSON_DEPTH`; None for a scalar, or for a container within the limit.

    `_refine` applies this rule to each container it reaches, through
    `_past_the_levels`, which handles a value known to be a container. A
    reader also calls this for each container it walks without refining
    (a document's members, or the consolidated metadata it descends into).
    This bounds a chain of documents in the same way as any other nesting,
    and checks every container at its location, with the same result as
    `refine_json` of the whole document.
    """
    if len(loc) < JSON_DEPTH or isinstance(value, (str, int, float, bool)) or value is None:
        return None
    if not isinstance(value, (Mapping, Sequence)) or isinstance(value, (bytes, bytearray)):
        return None
    return _past_the_levels(loc)


def _past_the_levels(loc: tuple[str | int, ...]) -> ValidationProblem:
    """The problem for a container at `loc` deeper than `JSON_DEPTH`: what `nested_past_the_levels` returns for a value known to be a container."""
    return ValidationProblem(loc, _PAST_THE_LEVELS, "invalid_value")


def within(
    problems: Sequence[ValidationProblem], at: tuple[str | int, ...]
) -> tuple[ValidationProblem, ...]:
    """Make the locations of `problems`, found in a value at `at`, relative to that value. This is the reverse of `prefixed`. It is for a reader that counts nesting levels from the document it was given, but reports locations within the document it reads.

    A problem that is not below `at` raises `TypeError`. Otherwise a reader
    that located a problem from the wrong root would report it in the
    wrong place.
    """
    if len(problems) == 0:
        return ()
    for problem in problems:
        if problem.loc[: len(at)] != at:
            msg = f"a problem at {problem.loc!r} does not sit below {at!r}"
            raise TypeError(msg)
    return tuple(dataclasses.replace(p, loc=p.loc[len(at) :]) for p in problems)


def _refine(value: object, loc: tuple[str | int, ...], *, finite: bool) -> _Refined:
    """`refine_json`, but a non-finite number counts as JSON unless `finite` is true."""
    # Check the exact types `json.loads` builds first, by type alone. Nearly
    # every value is one of these, so only the remaining values reach the
    # subclass checks below (a `StrEnum` member, an `IntEnum` member, a
    # mapping proxy).
    if type(value) is str or type(value) is bool or value is None:
        return value, ()
    if type(value) is int:
        if _writable(value):
            return value, ()
        return None, (ValidationProblem(loc, _too_many_digits(value), "invalid_value"),)
    if type(value) is float:
        if not finite or math.isfinite(value):
            return value, ()
        return None, (_not_finite(value, loc),)
    cls = type(value)
    if cls is dict:
        mapping = True
    elif cls is list or cls is tuple:
        mapping = False
    else:
        if isinstance(value, float):
            if not finite or math.isfinite(value):
                return float(value), ()
            return None, (_not_finite(value, loc),)
        if isinstance(value, int) and not isinstance(value, bool) and not _writable(value):
            return None, (ValidationProblem(loc, _too_many_digits(value), "invalid_value"),)
        if isinstance(value, bool):
            return value, ()
        if isinstance(value, str):
            # A subclass (such as a `StrEnum` member) becomes the string JSON writes for it.
            return str.__str__(value), ()
        if isinstance(value, int):
            return int(value), ()
        mapping = is_object(value)
        if not mapping and not is_array(value):
            return None, (
                ValidationProblem(
                    loc,
                    f"not a JSON-serializable value: {shown_by_python(value)}",
                    "invalid_type",
                ),
            )
    # The value is a container. Check its depth at its location with the
    # rule from `nested_past_the_levels`; only containers reach this point.
    if len(loc) >= JSON_DEPTH:
        return None, (_past_the_levels(loc),)
    if mapping:
        # Walk the members here rather than in a separate function, so that
        # each level of nesting costs one frame, `JSON_DEPTH` at most.
        members: dict[str, JSONValue] = {}
        found_in_members: list[ValidationProblem] = []
        for key, item in cast("Mapping[object, object]", value).items():
            if not isinstance(key, str):
                found_in_members.append(
                    ValidationProblem(
                        loc, f"non-string key {shown_key(key)} in JSON object", "invalid_type"
                    )
                )
                continue
            member, found = _refine(item, (*loc, key), finite=finite)
            found_in_members.extend(found)
            if len(found) == 0:
                members[key] = member
        return (members if len(found_in_members) == 0 else None), tuple(found_in_members)
    entries: list[JSONValue] = []
    found_in_entries: list[ValidationProblem] = []
    for index, item in enumerate(cast("Sequence[object]", value)):
        entry, found = _refine(item, (*loc, index), finite=finite)
        found_in_entries.extend(found)
        if len(found) == 0:
            entries.append(entry)
    return (tuple(entries) if len(found_in_entries) == 0 else None), tuple(found_in_entries)


def _not_finite(value: float, loc: tuple[str | int, ...]) -> ValidationProblem:
    """The problem for a non-finite number at `loc`, since JSON has no such numbers."""
    return ValidationProblem(loc, f"non-finite float {value!r} is not JSON", "invalid_value")


def _too_many_digits(value: int) -> str:
    """The message for an integer the interpreter will not convert to text, stating its size."""
    return (
        f"an integer of {value.bit_length()} bits has more digits than JSON text holds "
        "here, as sys.get_int_max_str_digits bounds it"
    )


def _writable(value: int) -> bool:
    """Whether the interpreter can convert `value` to text, as `json.dumps` and `repr` do. `sys.set_int_max_str_digits` limits this, and an integer past the limit could never be written back."""
    if value.bit_length() <= 64:
        return True
    try:
        str(value)
    except ValueError:
        return False
    return True


def json_text(value: JSONValue) -> str:
    """`value` as the JSON text `json.dumps` writes for it, with object keys sorted. `==` compares this text for a JSON value the package does not interpret.

    This is not Python's `==` on the value, which treats `true` as equal
    to `1`, `-0.0` as equal to `0.0`, and `NaN` as unequal to everything.
    It compares what a document contains: two values written the same way
    are the same value to every reader.
    """
    return _TEXT_ENCODER.encode(value)


def _as_object(value: object) -> dict[object, object]:
    """Convert a mapping that `json.dumps` does not handle (such as a read-only view) to a dict, so it is written as an object. Anything else raises the `TypeError` that `json.dumps` raises."""
    if is_object(value):
        return dict(value)
    msg = f"{value!r} is not JSON"
    raise TypeError(msg)


_TEXT_ENCODER: Final = json.JSONEncoder(sort_keys=True, ensure_ascii=False, default=_as_object)
"""The encoder `json_text` uses. `json.dumps` builds the same encoder on every call; this one is built once."""


def shown(value: object) -> str:
    """`value` as a problem's message shows it: as JSON (`null`, `[1, 2]`), or as its repr when it is not JSON. A value the interpreter will not write (an integer with too many digits, or a value nested too deep) is described instead."""
    if isinstance(value, int) and not isinstance(value, bool) and not _writable(value):
        return f"an integer of {value.bit_length()} bits"
    refined, problems = _refine(value, (), finite=False)
    if any(problem.message == _PAST_THE_LEVELS for problem in problems):
        # Nested deeper than `JSON_DEPTH`. Say so instead of using the repr,
        # which overflows at a depth set by the interpreter and platform.
        return "a value nested too deep to show"
    if len(problems) != 0:
        # Not JSON.
        return shown_by_python(value)
    try:
        return json.dumps(refined, ensure_ascii=False)
    except ValueError:
        # An integer with more digits than the interpreter converts to text.
        # If it is the value itself, show its size. If a container holds it,
        # Python cannot write the container either.
        if isinstance(value, int):
            return f"an integer of {value.bit_length()} bits"
        return shown_by_python(value)


def shown_key(key: object) -> str:
    """A non-string key as a message shows it: as Python shows it, since JSON has no such key to write. An integer with more digits than the interpreter writes is shown by its size."""
    if isinstance(key, int) and not isinstance(key, bool):
        try:
            return repr(key)
        except ValueError:
            return f"an integer of {key.bit_length()} bits"
    return shown_by_python(key)


def shown_by_python(value: object) -> str:
    """`value` as Python shows it (its repr), for a non-JSON value a reader finds. If the interpreter cannot write it (because it holds an integer with too many digits), the result says so."""
    try:
        return repr(value)
    except ValueError:
        return f"a value of type {type(value).__name__} the interpreter will not write"


V = TypeVar("V")


def said(faults: Sequence[str]) -> str:
    """Join `faults`, each a reason, into one clause: `a`, `a and b`, `a, b and c`."""
    if len(faults) <= 1:
        return "".join(faults)
    return f"{', '.join(faults[:-1])} and {faults[-1]}"


def choices(allowed: Sequence[object]) -> str:
    """A closed set of values as a message names it: `"C"` alone, or `one of ["C", "F"]`."""
    values = [shown(value) for value in listed(allowed)]
    return values[0] if len(values) == 1 else f"one of [{', '.join(values)}]"


def listed(allowed: Sequence[V]) -> tuple[V, ...]:
    """A closed set of values without duplicates, in the order a message lists them: sorted by their JSON text."""
    written = {shown(value): value for value in allowed}
    return tuple(written[json_text] for json_text in sorted(written))


def outside_of(
    loc: tuple[str | int, ...], value: object, allowed: Sequence[object]
) -> ValidationProblem:
    """The problem for `value`, found at `loc`, when it is not in the closed set `allowed`.

    The message lists the set. The kind tells whether the value has the
    wrong JSON type, or the right type and the wrong value. The `ctx`
    holds the set under `expected`.
    """
    message = f"expected {choices(allowed)}, got {shown(value)}"
    expected = cast("tuple[JSONValue, ...]", listed(allowed))
    return ValidationProblem(loc, message, refused_kind(value, allowed), ctx={"expected": expected})


def json_type(value: object) -> str:
    """The JSON type of `value`, as a message names it: "a string", "a number", "null"."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, (int, float)):
        return "a number"
    if isinstance(value, str):
        return "a string"
    if isinstance(value, Mapping):
        return "an object"
    if isinstance(value, (list, tuple)):
        return "an array"
    return "a value"


def refused_kind(value: object, allowed: Sequence[object]) -> ProblemKind:
    """The kind of problem for `value` outside a closed set: `invalid_type` when no member of the set has its type, else `invalid_value`.

    An integer is a type of its own here, as everywhere a reader expects
    one: `2.0` for `zarr_format` is the wrong type, as `4.0` in `shape` is.
    """
    same_type = _set_type(value) in {_set_type(entry) for entry in allowed}
    return "invalid_value" if same_type else "invalid_type"


def _set_type(value: object) -> str:
    """The type `refused_kind` compares: the JSON type, with an integer told apart from any other number."""
    if isinstance(value, int) and not isinstance(value, bool):
        return "an integer"
    return json_type(value)


def is_canonical_json(value: object, *, finite: bool = True) -> TypeGuard[JSONValue]:
    """Whether `value` already uses the concrete containers in `JSONValue`, nested no deeper than `JSON_DEPTH`.

    A non-finite number counts only when `finite` is false, as in user data
    such as attributes: the document's reader decides where such a number
    is allowed. The walk uses one frame per level of
    nesting, `JSON_DEPTH` at most, as `refine_json` does. A reader does not
    accept a container deeper than that as JSON, so this function does not
    either, and the walk stops there.
    """
    return _is_canonical(value, 0, finite=finite)


def _is_canonical(value: object, depth: int, *, finite: bool) -> bool:
    """`is_canonical_json` of `value`, which is `depth` levels deep."""
    if isinstance(value, float):
        return not finite or math.isfinite(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return True
    if depth >= JSON_DEPTH:
        return False
    if is_list_or_tuple(value):
        for item in value:  # noqa: SIM110 - a loop, not a generator, is one frame per level
            if not _is_canonical(item, depth + 1, finite=finite):
                return False
        return True
    if is_object(value):
        for key, item in value.items():
            if not isinstance(key, str) or not _is_canonical(item, depth + 1, finite=finite):
                return False
        return True
    return False


def is_json(value: object) -> TypeGuard[JSONValue]:
    """Whether `value` is a canonical JSON structure (recursively)."""
    return is_canonical_json(value)


def parse_json(value: object) -> JSONValue:
    """Return a canonical `JSONValue`, or raise `MetadataValidationError`."""
    refined, problems = refine_json(value)
    if len(problems) != 0:
        raise MetadataValidationError(with_input(problems, value))
    return refined


@overload
def frozen(value: Mapping[str, JSONValue]) -> Mapping[str, JSONValue]: ...
@overload
def frozen(value: JSONValue) -> JSONValue: ...
def frozen(value: JSONValue) -> JSONValue:
    """`value` as a read-only view at every level: each object a mapping proxy, each array a tuple, so the result cannot be changed in place.

    The result shares the scalars with `value`, and copies only the
    containers the view needs. A model uses this to expose its document.
    """
    cls = type(value)
    if cls in _SCALARS:
        return value
    if cls is dict or isinstance(value, Mapping):
        members = cast("Mapping[str, JSONValue]", value)
        return cast(
            "JSONValue",
            MappingProxyType(
                {
                    key: item if type(item) in _SCALARS else frozen(item)
                    for key, item in members.items()
                }
            ),
        )
    if cls is tuple or cls is list or isinstance(value, (tuple, list)):
        return tuple(
            item if type(item) in _SCALARS else frozen(item)
            for item in cast("Sequence[JSONValue]", value)
        )
    return value


_SCALARS: Final[frozenset[type]] = frozenset({str, int, float, bool, type(None)})
"""The Python types of JSON scalars. A read-only view shares these values without copying."""


def copied(value: JSONValue) -> JSONValue:
    """A deep copy of `value` that shares no containers with it: each object a new `dict`, each array a new array of the same type.

    The copy uses one frame for each level of nesting, as `refine_json`
    does, so any refined value can be copied, however deep it is.
    """
    if isinstance(value, Mapping):
        members: dict[str, JSONValue] = {}
        for key, item in value.items():
            members[key] = copied(item)
        return members
    if isinstance(value, (tuple, list)):
        # Use `map`, not a comprehension. Before Python 3.12 a comprehension
        # uses its own frame; this keeps it to one frame per level.
        entries = list(map(copied, value))
        return entries if isinstance(value, list) else tuple(entries)
    return value


def arrays_to_tuples(obj: object) -> object:
    """Recursively convert array-like values to tuples and mappings to dicts.

    A tuple or dict whose contents need no conversion is returned as is, not copied.
    """
    if is_array(obj):
        sequence = obj
        # Use `map` and loops, not comprehensions. Before Python 3.12 a
        # comprehension uses its own frame; this keeps it to one frame per
        # level, as in `copied`.
        converted_sequence = tuple(map(arrays_to_tuples, sequence))
        if isinstance(obj, tuple) and all(
            converted is original
            for converted, original in zip(converted_sequence, sequence, strict=True)
        ):
            return sequence
        return converted_sequence
    if is_object(obj):
        mapping = obj
        converted: dict[object, object] = {}
        for key, value in mapping.items():
            converted[key] = arrays_to_tuples(value)
        if isinstance(obj, dict) and all(converted[key] is value for key, value in mapping.items()):
            return mapping
        return converted
    return obj


__all__ = [
    "MetadataValidationError",
    "ProblemKind",
    "Problems",
    "ValidationProblem",
    "arrays_to_tuples",
    "choices",
    "copied",
    "frozen",
    "is_canonical_json",
    "is_json",
    "json_type",
    "listed",
    "nested_past_the_levels",
    "not_an_object",
    "outside_of",
    "parse_json",
    "prefixed",
    "refine_json",
    "refine_user_data",
    "refused_kind",
    "said",
    "shown",
    "shown_key",
    "validate_json",
    "value_at",
    "with_input",
    "within",
]
