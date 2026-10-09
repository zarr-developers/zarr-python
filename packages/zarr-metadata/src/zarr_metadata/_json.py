"""JSON values, and the problems found with them: where reading starts, below every layer.

`refine_json` is the one walk that normalizes and judges a value, and
`refine_user_data` the same walk over user data, where a non-finite number
is the float it is.
`ValidationProblem` and `MetadataValidationError` are what every
layer reports with: the checker, the definitions and the model alike.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, TypeGuard, cast, get_args, overload

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
- `unknown_key`: a key an object's type does not declare, where the type
  says it is closed, as a closed TypedDict does. Whether a Zarr
  configuration is closed is rarely said (zarr-developers/zarr-specs#270
  has been open since 2023), and many readers refuse such a key. It gets
  a kind of its own so that a caller who tolerates it can tell it from a
  wrong value, and so that it never masks the other findings about the
  same object.
"""


JSON_DEPTH: Final = 256
"""How many levels of nesting a reader walks.

A value nested deeper is a problem at the level past the last, so no
document, however deep, takes a reader past what the interpreter allows:
every value the package reads is refined first, and refining stops here.
Every reader, writer and comparison takes one frame for each level, and
`copy.deepcopy`, and `pickle` before Python 3.12, two: so a document at
the cap takes about half of the interpreter's default limit, a thousand
frames, and the rest is the caller's.
"""

_PAST_THE_LEVELS: Final = f"nested deeper than the {JSON_DEPTH} levels a reader walks"
"""The message of the problem a container past the levels a reader walks is."""


class _Ctx(Mapping[str, JSONValue]):
    """What a problem holds as its `ctx`: a copy of its own, arrays as tuples, checked to be JSON when it was made, which nothing edits after.

    Its own type, so a problem built of another's `ctx` -- as `replace`
    builds one -- knows it was checked, and skips the check; a mapping of
    any other type is checked and copied.
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
    """What a problem that says nothing more than its type was expected holds as its `ctx`: nothing."""
    return _NO_CTX


@dataclass(frozen=True, slots=True)
class ValidationProblem:
    """A single problem found in a value: where it is, what is wrong, what kind of wrong, and the data the message is made of.

    `loc` is the path from the root of what was judged to the offending
    value, e.g. `("codecs", 0, "name")` in a document, and an empty `loc`
    refers to that root.
    `kind` classifies the failure mode for programmatic dispatch; `message`
    is the human-readable description.

    `input` and `ctx` are what the message says, as data, as pydantic's
    `ErrorDetails` and zod's issues carry theirs. `input` is the JSON found
    at `loc` -- `12`, for a gzip `level` of 12 -- and `UNSET` where nothing
    is there, as zod has it for a key that is missing (pydantic gives the
    object missing it), or where what is there is not JSON, which the
    message shows. It is the object the caller handed in, as pydantic's
    is, not a copy: a caller that changes its document afterwards changes
    what `input` shows. `ctx` is what was expected, where that is more
    than a type:

    - `gt`, `ge`, `lt` and `le`: the bounds the value's type carries, as
      pydantic names them -- `{"ge": 0, "le": 9}` for a gzip `level`,
      whose type is `Annotated[int, Interval(ge=0, le=9)]` -- or a rule
      says.
    - `expected`: the values of a closed set, as zod's `values` holds
      them, in the order the message lists them -- a `Literal`'s,
      `node_type`'s, `zarr_format`'s.

    Neither takes part in equality or the repr: a problem is the same
    problem when it is found at the same place and says the same thing.
    Every function that returns or raises problems fills `input` from the
    value its caller handed it, so a rule says only where a problem is.
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
        # The runtime half of the annotations: a rule written without a type
        # checker, as an extension's may be, fails where it builds a problem
        # rather than reporting one at a location that is not one.
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
        if not isinstance(kind, str) or kind not in get_args(ProblemKind):
            msg = f"a ValidationProblem's kind is one of {get_args(ProblemKind)!r}, got {kind!r}"
            raise TypeError(msg)
        ctx = cast("object", self.ctx)
        if isinstance(ctx, _Ctx):
            # A problem's own, checked when it was made: what `replace`
            # hands a copy.
            return
        if (
            not isinstance(ctx, Mapping)
            or not all(isinstance(key, str) for key in cast("Mapping[object, object]", ctx))
            or not is_json(dict(cast("Mapping[str, object]", ctx)))
        ):
            msg = f"a ValidationProblem's ctx is an object of JSON values, got {ctx!r}"
            raise TypeError(msg)
        # Held as a view of a copy of its own at every level, arrays as
        # tuples, so a raised error, a finished report, cannot be edited
        # through it, nor through what was handed in.
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
        # Pickled and copied as its constructor called again: the view
        # `ctx` is held as does not pickle, and the dict it views does.
        return (_problem, (self.loc, self.message, self.kind, self.input, dict(self.ctx)))


def _problem(
    loc: tuple[str | int, ...],
    message: str,
    kind: ProblemKind,
    found: JSONValue | UNSET,
    ctx: Mapping[str, JSONValue],
) -> ValidationProblem:
    """A problem built again from what `ValidationProblem.__reduce__` gives."""
    return ValidationProblem(loc, message, kind, input=found, ctx=ctx)


class MetadataValidationError(ValueError):
    """Raised when a value fails validation, by the entry points that raise rather than report.

    Carries every problem found (not just the first) in `.problems`, as an
    immutable tuple: a raised error is a finished report, and a caller
    inspecting it must not be able to edit the record.
    """

    problems: tuple[ValidationProblem, ...]

    def __init__(self, problems: Sequence[ValidationProblem]) -> None:
        self.problems = tuple(problems)
        for entry in cast("tuple[object, ...]", self.problems):
            # The runtime half of the annotation: a caller that is not
            # type-checked, and hands over anything else, fails here
            # rather than far away, where a `loc` is read off it.
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
        # An exception pickles and copies as its class called with its
        # `args`, which here are the message; it is built from its problems,
        # and the rest of its state -- its notes among it -- follows.
        return (type(self), (self.problems,), self.__dict__)


def prefixed(
    loc_head: str | int, problems: Sequence[ValidationProblem]
) -> tuple[ValidationProblem, ...]:
    """Prepend `loc_head` to the `loc` of every problem (for nested validators)."""
    return tuple(dataclasses.replace(p, loc=(loc_head, *p.loc)) for p in problems)


def value_at(value: object, loc: tuple[str | int, ...]) -> object:
    """What `value` holds at `loc`, each key naming a member of an object and each index an element of an array; `UNSET` where it holds nothing."""
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
    """`problems`, found in `value`, which sits at `loc`: each given, as its `input`, the JSON `value` holds at its own `loc`.

    What every function that judges a value does with what it found, so a
    rule says only where a problem is, and the problem carries what is
    there. The function a caller called is the last to do it, so a
    problem's `input` is what the value the caller handed in holds, not a
    copy some reader inside it made. A problem whose `loc` names nothing
    in `value` -- a key that is missing -- or names what is not JSON
    keeps what it holds, `UNSET` unless a reader inside found JSON there:
    no problem holds what might not pickle, or copy.
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
    """Whether a problem can hold `value` as its input: JSON, nested no deeper than a reader walks.

    What a validator did not walk -- a member it only reports -- may be
    deeper than that; such a value is held as nothing, as
    `is_canonical_json` says.
    """
    return is_canonical_json(value, finite=False)


def not_an_object(value: object) -> tuple[ValidationProblem, ...]:
    """What is wrong with a document that is not an object: the one problem, at its root."""
    return with_input((ValidationProblem((), "expected an object", "invalid_type"),), value)


def validate_json(value: object, loc: tuple[str | int, ...] = ()) -> tuple[ValidationProblem, ...]:
    """Return every reason `value`, which sits at `loc`, is not JSON, each where it sits: a float that is not finite, a key that is not a string, a value of no JSON type, a level of nesting past `JSON_DEPTH`, counted from the document's root, which `loc` is below."""
    return with_input(refine_json(value, loc)[1], value, loc)


def is_object(value: object) -> TypeGuard[Mapping[object, object]]:
    """Whether `value` is a JSON object as Python holds one: a mapping, of any keys.

    What `isinstance(value, Mapping)` says, narrowed to a mapping of
    `object`, which the bare check leaves unknown to a type checker.
    """
    return isinstance(value, Mapping)


def refined_object(value: object) -> dict[str, JSONValue]:
    """`value`, an object a read found nothing wrong with, refined as user data: the document a model holds.

    `TypeError` when it is not an object, which a read that found no
    problem rules out.
    """
    refined, _ = refine_user_data(value)
    if not isinstance(refined, dict):
        msg = f"expected an object a read found nothing wrong with, got {shown(value)}"
        raise TypeError(msg)
    return refined


def object_at(document: Mapping[str, JSONValue], key: str) -> dict[str, JSONValue]:
    """The object `document`, refined, holds at `key`, which a read found there; `TypeError` when something else is, which that read rules out."""
    value = document[key]
    if not isinstance(value, dict):
        msg = f"expected an object at {key!r}, got {shown(value)}"
        raise TypeError(msg)
    return value


def is_json_object(value: object) -> TypeGuard[Mapping[str, object]]:
    """Whether `value` is a JSON object with the keys JSON gives one: a mapping whose keys are all strings."""
    return isinstance(value, Mapping) and all(
        isinstance(key, str) for key in cast("Mapping[object, object]", value)
    )


def is_list_or_tuple(value: object) -> TypeGuard[list[object] | tuple[object, ...]]:
    """Whether `value` is a list or a tuple: the two containers canonical JSON holds an array in."""
    return isinstance(value, (list, tuple))


def is_tuple(value: object) -> TypeGuard[tuple[object, ...]]:
    """Whether `value` is a tuple, narrowed to a tuple of `object`."""
    return isinstance(value, tuple)


def is_array(value: object) -> TypeGuard[Sequence[object]]:
    """Whether `value` is a JSON array as Python holds one: a sequence that is not text or bytes."""
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def refine_json(
    value: object, loc: tuple[str | int, ...] = ()
) -> tuple[JSONValue | None, tuple[ValidationProblem, ...]]:
    """`value` as JSON with arrays as tuples, or None with every reason it is not JSON.

    One walk normalizes and judges: a mapping becomes a `dict` with
    string keys, a sequence a tuple, and a float must be finite. A value
    that is not JSON is None, with the problems located at the leaves
    that are not. JSON's `null` is None too, with no problem, so whether
    `value` is JSON is whether there are problems.
    """
    return _refine(value, loc, finite=True)


def refine_user_data(
    value: object, loc: tuple[str | int, ...] = ()
) -> tuple[JSONValue | None, tuple[ValidationProblem, ...]]:
    """User data refined as `refine_json` refines JSON, a non-finite number being the float it is.

    A node's attributes are user data. The spec asks only that each be a
    JSON value and interprets none, and zarr-python writes them with the
    defaults of Python's `json` module, so an attribute can hold `NaN`,
    `Infinity` or `-Infinity` -- xarray's `_FillValue`, a CF
    `missing_value` -- which RFC 8259 lacks. Wherever the spec interprets
    a value it spells those numbers as strings, so a validator walks what
    it interprets with `refine_json`, and only user data with this.
    """
    return _refine(value, loc, finite=False)


_Refined = tuple[JSONValue | None, tuple[ValidationProblem, ...]]


def nested_past_the_levels(value: object, loc: tuple[str | int, ...]) -> ValidationProblem | None:
    """The problem `value` is when it is a container -- an object or an array -- at `loc`, past the levels a reader walks, `JSON_DEPTH` of them; None for a scalar, or within them.

    `_refine` asks it of every value it reaches, and a reader of each
    container it walks without refining -- a document's members, the
    consolidated metadata it descends into -- so a chain of documents is
    bounded as any other nesting is, and every container is judged where
    it sits, as `refine_json` of the whole document would judge it.
    """
    if len(loc) < JSON_DEPTH or isinstance(value, (str, int, float, bool)) or value is None:
        return None
    if not isinstance(value, (Mapping, Sequence)) or isinstance(value, (bytes, bytearray)):
        return None
    return ValidationProblem(loc, _PAST_THE_LEVELS, "invalid_value")


def within(
    problems: Sequence[ValidationProblem], at: tuple[str | int, ...]
) -> tuple[ValidationProblem, ...]:
    """`problems`, found in a value that sits at `at`, located from that value: the reverse of `prefixed`, for a reader that counts the levels it walks from the document handed in, but reports where a problem sits in the one it reads.

    A problem not below `at` is a `TypeError`: a reader that located one
    from the wrong root would otherwise report it in the wrong place.
    """
    for problem in problems:
        if problem.loc[: len(at)] != at:
            msg = f"a problem at {problem.loc!r} does not sit below {at!r}"
            raise TypeError(msg)
    return tuple(dataclasses.replace(p, loc=p.loc[len(at) :]) for p in problems)


def _refine(value: object, loc: tuple[str | int, ...], *, finite: bool) -> _Refined:
    """`refine_json`, a non-finite number being JSON unless `finite`."""
    if isinstance(value, float):
        if not finite or math.isfinite(value):
            return value, ()
        return None, (
            ValidationProblem(loc, f"non-finite float {value!r} is not JSON", "invalid_value"),
        )
    if isinstance(value, (str, int, bool)) or value is None:
        return value, ()
    if (past := nested_past_the_levels(value, loc)) is not None:
        return None, (past,)
    if is_object(value):
        # Walked here rather than through `_refine_members`, so that each
        # level of nesting costs one frame, `JSON_DEPTH` of them at most.
        members: dict[str, JSONValue] = {}
        found_in_members: list[ValidationProblem] = []
        for key, item in value.items():
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
    if is_array(value):
        entries: list[JSONValue] = []
        found_in_entries: list[ValidationProblem] = []
        for index, item in enumerate(value):
            entry, found = _refine(item, (*loc, index), finite=finite)
            found_in_entries.extend(found)
            if len(found) == 0:
                entries.append(entry)
        return (tuple(entries) if len(found_in_entries) == 0 else None), tuple(found_in_entries)
    return None, (
        ValidationProblem(
            loc, f"not a JSON-serializable value: {shown_by_python(value)}", "invalid_type"
        ),
    )


def json_text(value: JSONValue) -> str:
    """`value` as the JSON text `json.dumps` writes for it, an object's keys sorted: what `==` compares of a JSON value the package does not interpret.

    Not Python's `==` on the value, which takes `true` for `1`, `-0.0` for
    `0.0` and `NaN` for no value at all, but what a document writes: two
    values written alike are one value to every reader.
    """
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=_as_object)


def _as_object(value: object) -> dict[object, object]:
    """A read-only view of an object, as `json.dumps` is handed one, written as the object; anything else is the `TypeError` `json.dumps` raises."""
    if is_object(value):
        return dict(value)
    msg = f"{value!r} is not JSON"
    raise TypeError(msg)


def shown(value: object) -> str:
    """`value` as a problem's message shows it: as the JSON a document writes, `null` and `[1, 2]`, or by its repr when it is not JSON; what the interpreter will not write, an integer of too many digits or a value nested too deep, by saying so."""
    refined, problems = _refine(value, (), finite=False)
    if any(problem.message == _PAST_THE_LEVELS for problem in problems):
        # Nested past the levels a reader walks: said so, not left to the
        # repr, which overflows at a depth the interpreter and platform set.
        return "a value nested too deep to show"
    if len(problems) != 0:
        # Not JSON.
        return shown_by_python(value)
    try:
        return json.dumps(refined, ensure_ascii=False)
    except ValueError:
        # An integer of more digits than the interpreter converts to text:
        # the value itself, by its size, or one a container holds, which
        # is then what Python will not write either.
        if isinstance(value, int):
            return f"an integer of {value.bit_length()} bits"
        return shown_by_python(value)


def shown_key(key: object) -> str:
    """A key that is not a string, as a message shows it: as Python shows it, since no JSON object holds such a key to write; an integer of more digits than the interpreter writes, by its size."""
    if isinstance(key, int) and not isinstance(key, bool):
        try:
            return repr(key)
        except ValueError:
            return f"an integer of {key.bit_length()} bits"
    return shown_by_python(key)


def shown_by_python(value: object) -> str:
    """`value` as Python shows it, for what is not JSON a reader walks; what the interpreter will not write -- holding an integer of more digits than it writes -- said so."""
    try:
        return repr(value)
    except ValueError:
        return f"a value of type {type(value).__name__} the interpreter will not write"


def choices(allowed: Sequence[object]) -> str:
    """A closed set of values as a message names it: `"C"` alone, or `one of ["C", "F"]`."""
    values = [shown(value) for value in listed(allowed)]
    return values[0] if len(values) == 1 else f"one of [{', '.join(values)}]"


def listed(allowed: Sequence[object]) -> tuple[object, ...]:
    """A closed set of values, each once, in the order a message lists them: by the JSON each is written as."""
    written = {shown(value): value for value in allowed}
    return tuple(written[json_text] for json_text in sorted(written))


def outside_of(
    loc: tuple[str | int, ...], value: object, allowed: Sequence[object]
) -> ValidationProblem:
    """The problem with `value`, found at `loc`, outside the closed set `allowed`.

    Its message lists the set, its kind says whether the value is of the
    wrong JSON type or of the right one with the wrong value, and its
    `ctx` holds the set, as `expected`.
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
    """What is wrong with `value`, outside a closed set: its type, when none of the set is of its JSON type, else its value."""
    same_type = json_type(value) in {json_type(entry) for entry in allowed}
    return "invalid_value" if same_type else "invalid_type"


def is_canonical_json(value: object, *, finite: bool = True) -> TypeGuard[JSONValue]:
    """Whether `value` already uses the concrete containers in `JSONValue`, nested no deeper than a reader walks.

    A non-finite number counts only when `finite` is false, as a document's
    guard passes it: where one may be is the document's validator's to say.
    One frame per level of nesting, `JSON_DEPTH` of them at most, as
    `refine_json` takes: a container past them is no JSON a reader walks,
    so it is none to this either, and the walk stops there.
    """
    return _is_canonical(value, 0, finite=finite)


def _is_canonical(value: object, depth: int, *, finite: bool) -> bool:
    """`is_canonical_json` of `value`, which sits `depth` levels down."""
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
    """`value` as a read-only view at every level: each object a mapping proxy, each array a tuple, so nothing handed out can be changed in place.

    Shares the scalars with `value`, and copies nothing else than the
    containers a view needs. What a model shows of its document.
    """
    if isinstance(value, Mapping):
        return cast(
            "JSONValue", MappingProxyType({key: frozen(item) for key, item in value.items()})
        )
    if isinstance(value, (tuple, list)):
        return tuple(frozen(item) for item in value)
    return value


def copied(value: JSONValue) -> JSONValue:
    """`value` in containers of its own, sharing nothing with it: each object a new `dict`, each array a new one of its type.

    One frame for each level of nesting, as `refine_json` reads, so a
    value refined is copied however deep it is.
    """
    if isinstance(value, Mapping):
        members: dict[str, JSONValue] = {}
        for key, item in value.items():
            members[key] = copied(item)
        return members
    if isinstance(value, (tuple, list)):
        # A loop, not a comprehension, which is a frame of its own before
        # Python 3.12: one frame for each level.
        entries = list(map(copied, value))
        return entries if isinstance(value, list) else tuple(entries)
    return value


def arrays_to_tuples(obj: object) -> object:
    """Recursively materialize mappings and convert array-like values to tuples."""
    if is_array(obj):
        sequence = obj
        # Loops, not comprehensions, which are a frame of their own before
        # Python 3.12: one frame for each level, as `copied` takes.
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
    "shown",
    "shown_key",
    "validate_json",
    "value_at",
    "with_input",
    "within",
]
