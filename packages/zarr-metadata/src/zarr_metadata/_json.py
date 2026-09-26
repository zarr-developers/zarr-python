"""JSON values, and the problems found with them: where reading starts, below every layer.

`refine_json` is the one walk that normalizes and judges a value, and
`refine_user_data` the same walk over user data, where a non-finite number
is the float it is.
`ValidationProblem` and `MetadataValidationError` are what every
layer reports with: the checker, the definitions and the model alike.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeGuard, cast, get_args

from zarr_metadata._common import JSONValue

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


@dataclass(frozen=True, slots=True)
class ValidationProblem:
    """A single problem found in a value: where it is, what is wrong, and what kind of wrong.

    `loc` is the path from the root of what was judged to the offending
    value, e.g. `("codecs", 0, "name")` in a document, and an empty `loc`
    refers to that root.
    `kind` classifies the failure mode for programmatic dispatch; `message`
    is the human-readable description.
    """

    loc: tuple[str | int, ...]
    message: str
    kind: ProblemKind

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

    def __str__(self) -> str:
        location = ".".join(str(part) for part in self.loc) if self.loc else "<root>"
        return f"{location}: {self.message}"


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
    return tuple(ValidationProblem((loc_head, *p.loc), p.message, p.kind) for p in problems)


def validate_json(value: object) -> tuple[ValidationProblem, ...]:
    """Return every reason `value` is not JSON-serializable (recursively): `refine_json`'s problems."""
    return refine_json(value)[1]


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
    if isinstance(value, Mapping):
        # Walked here rather than through `_refine_members`, so that each
        # level of nesting costs one frame, as deep as the interpreter goes.
        members: dict[str, JSONValue] = {}
        found_in_members: list[ValidationProblem] = []
        for key, item in cast("Mapping[object, object]", value).items():
            if not isinstance(key, str):
                found_in_members.append(
                    ValidationProblem(loc, f"non-string key {key!r} in JSON object", "invalid_type")
                )
                continue
            member, found = _refine(item, (*loc, key), finite=finite)
            found_in_members.extend(found)
            if len(found) == 0:
                members[key] = member
        return (members if len(found_in_members) == 0 else None), tuple(found_in_members)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        entries: list[JSONValue] = []
        found_in_entries: list[ValidationProblem] = []
        for index, item in enumerate(cast("Sequence[object]", value)):
            entry, found = _refine(item, (*loc, index), finite=finite)
            found_in_entries.extend(found)
            if len(found) == 0:
                entries.append(entry)
        return (tuple(entries) if len(found_in_entries) == 0 else None), tuple(found_in_entries)
    return None, (
        ValidationProblem(loc, f"not a JSON-serializable value: {value!r}", "invalid_type"),
    )


def is_canonical_json(value: object, *, finite: bool = True) -> TypeGuard[JSONValue]:
    """Whether `value` already uses the concrete containers in `JSONValue`.

    A non-finite number counts only when `finite` is false, as a document's
    guard passes it: where one may be is the document's validator's to say.
    """
    if isinstance(value, float):
        return not finite or math.isfinite(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return True
    if isinstance(value, (list, tuple)):
        sequence = cast("list[object] | tuple[object, ...]", value)
        return all(is_canonical_json(item, finite=finite) for item in sequence)
    if isinstance(value, dict):
        mapping = cast("dict[object, object]", value)
        return all(
            isinstance(key, str) and is_canonical_json(item, finite=finite)
            for key, item in mapping.items()
        )
    return False


def is_json(value: object) -> TypeGuard[JSONValue]:
    """Whether `value` is a canonical JSON structure (recursively)."""
    return is_canonical_json(value)


def parse_json(value: object) -> JSONValue:
    """Return a canonical `JSONValue`, or raise `MetadataValidationError`."""
    refined, problems = refine_json(value)
    if len(problems) != 0:
        raise MetadataValidationError(problems)
    return refined


def arrays_to_tuples(obj: object) -> object:
    """Recursively materialize mappings and convert array-like values to tuples."""
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        sequence = cast("Sequence[object]", obj)
        converted_sequence = tuple(arrays_to_tuples(item) for item in sequence)
        if isinstance(obj, tuple) and all(
            converted is original
            for converted, original in zip(converted_sequence, sequence, strict=True)
        ):
            return sequence
        return converted_sequence
    if isinstance(obj, Mapping):
        mapping = cast("Mapping[object, object]", obj)
        converted: dict[object, object] = {
            key: arrays_to_tuples(value) for key, value in mapping.items()
        }
        if isinstance(obj, dict) and all(converted[key] is value for key, value in mapping.items()):
            return mapping
        return converted
    return obj


__all__ = [
    "MetadataValidationError",
    "ProblemKind",
    "ValidationProblem",
    "arrays_to_tuples",
    "is_canonical_json",
    "is_json",
    "parse_json",
    "prefixed",
    "refine_json",
    "refine_user_data",
    "validate_json",
]
