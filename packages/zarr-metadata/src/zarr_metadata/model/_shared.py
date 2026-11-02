"""Helpers that the v2 and v3 readers share: checking the keys of a document, refining its members and attributes, and reading and writing store JSON."""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING, Any, Protocol, TypeGuard, TypeVar, cast

from zarr_metadata._json import (
    MetadataValidationError,
    ValidationProblem,
    is_array,
    is_object,
    outside_of,
    refine_json,
    refine_user_data,
    shown_key,
    with_input,
    within,
)
from zarr_metadata._typed_json import typeddict_keys
from zarr_metadata.definition._definition import RefusedField, UnclaimedField, is_field
from zarr_metadata.definition._scope import Conflict, claim_key
from zarr_metadata.v3._definition import ExtraFieldDefinition

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from zarr_metadata._common import JSONValue
    from zarr_metadata._typed_json import Loc
    from zarr_metadata.definition._definition import ResolvedField


def missing_keys(
    required: frozenset[str], doc: Mapping[object, object]
) -> tuple[ValidationProblem, ...]:
    """One `missing_key` problem per required key absent from `doc`."""
    return tuple(
        ValidationProblem((key,), "missing required key", "missing_key")
        for key in sorted(required - doc.keys())
    )


def unexpected_keys(
    allowed: frozenset[str], doc: Mapping[object, object]
) -> tuple[ValidationProblem, ...]:
    """One problem per member outside a closed document's declared shape. The kind is `unknown_key`, as a closed TypedDict's checker reports it, so a caller who tolerates a member another writer added can tell it apart from a wrong value."""
    problems: list[ValidationProblem] = []
    for key in doc:
        if not isinstance(key, str):
            problems.append(
                ValidationProblem((), f"non-string document key {shown_key(key)}", "invalid_type")
            )
        elif key not in allowed:
            problems.append(ValidationProblem((key,), f"unexpected key {key!r}", "unknown_key"))
    return tuple(problems)


def check_literal(
    doc: Mapping[object, object], key: str, expected: object
) -> tuple[ValidationProblem, ...]:
    """One problem if `doc[key]` is present but not equal to `expected`. The problem is about the type when the value is not of `expected`'s JSON type, and about the value otherwise. The value is compared after refinement, so a `StrEnum` member counts as its string."""
    if key not in doc:
        return ()
    value, found = refine_json(doc[key], (key,))
    if len(found) != 0 or type(value) is not type(expected) or value != expected:
        return (outside_of((key,), doc[key], (expected,)),)
    return ()


def other_members(
    doc: Mapping[object, object],
    standard_keys: frozenset[str],
    *,
    additional_reserved_keys: frozenset[str] = frozenset(),
    at: Loc = (),
) -> tuple[dict[str, JSONValue], tuple[ValidationProblem, ...]]:
    """Each member outside `standard_keys`, refined as user data, and every problem found in them. A member that is not JSON is left out of the result.

    The spec does not interpret such a member, so like attributes, and
    like a v3 member that nothing claims, it may hold `NaN`, `Infinity`
    and `-Infinity`.
    """
    members: dict[str, JSONValue] = {}
    problems: list[ValidationProblem] = []
    reserved_keys = standard_keys | additional_reserved_keys
    for key, value in doc.items():
        if not isinstance(key, str):
            problems.append(
                ValidationProblem((), f"non-string top-level key {shown_key(key)}", "invalid_type")
            )
            continue
        if key in reserved_keys:
            continue
        refined, found = refine_user_data(value, (*at, key))
        problems.extend(within(found, at))
        if len(found) == 0:
            members[key] = refined
    return members, tuple(problems)


def member_changes(
    members: Mapping[str, object], update: type, spec_members: frozenset[str]
) -> dict[str, object]:
    """The document members that `members`, given to a model's `update` or `create_default`, set: each member that `update` (the model's update TypedDict) declares, and each member in its `extra_fields`. A value of `UNSET` removes its member.

    Raises `TypeError` for a member that `update` does not declare, and
    for a member of `extra_fields` that the spec defines (one of
    `spec_members`) or that `update` takes by name. So a misspelled
    member is never written as a member the spec does not define, which a
    reader that must understand every such member would refuse. Also
    raises `TypeError` for a field given where a member's JSON goes.
    """
    declared = typeddict_keys(update).members
    unknown = [key for key in members if key not in declared]
    if len(unknown) != 0:
        names = ", ".join(map(repr, unknown))
        msg = (
            f"unknown member {names}: give each member the spec defines by its name, and any "
            "other member in extra_fields"
            if "extra_fields" in declared
            else f"unknown member {names}"
        )
        raise TypeError(msg)
    changes = {key: value for key, value in members.items() if key != "extra_fields"}
    for key, value in changes.items():
        _refuse_fields(key, value)
    extra = members.get("extra_fields", {})
    if not is_object(extra):
        msg = f"extra_fields maps members to their values, got {extra!r}"
        raise TypeError(msg)
    for key, value in extra.items():
        if not isinstance(key, str):
            msg = (
                f"extra_fields maps each member's key, a string, to its value, got the key {key!r}"
            )
            raise TypeError(msg)
        # `extra_fields` itself is the mapping, so a member of that name is
        # given in it like any other member the spec does not define.
        if key in declared and key != "extra_fields":
            msg = f"{key!r} is a member that update takes by name; give it by name, not in extra_fields"
            raise TypeError(msg)
        if key in spec_members:
            msg = f"{key!r} is a member the spec defines, which extra_fields does not hold"
            raise TypeError(msg)
        _refuse_fields(key, value)
        changes[key] = value
    return changes


def unclaimed_fields_of(
    fields: Iterable[tuple[Loc, ResolvedField[Any]]], waived: Iterable[str]
) -> dict[Loc, UnclaimedField]:
    """Each of `fields` (a reading's fields, at every depth) that nothing in scope claims, by its location, except one inside an extra member whose key is in `waived`: a member that says `"must_understand": false`, which a reader may ignore whole, whether or not a definition claims it."""
    ignored = frozenset(waived)
    return {
        loc: field
        for loc, field in fields
        if isinstance(field, UnclaimedField) and not (len(loc) != 0 and loc[0] in ignored)
    }


def _refuse_fields(key: str, value: object) -> None:
    """Raise `TypeError` when `value`, given as the member at `key`, is a field or holds fields, rather than the JSON a document writes, with the JSON to give instead."""
    for item in value if is_array(value) else (value,):
        if not is_field(item):
            continue
        if isinstance(item, RefusedField):
            advice = "a refused field has no JSON to write; give the member as JSON"
        elif issubclass(item.read_as, ExtraFieldDefinition):
            # An extra field is read as the member alone, `{key: value}`.
            advice = f"give the member's value, field.to_json()[{item.name!r}]"
        else:
            advice = "give the field as JSON, field.to_json()"
        msg = f"{key}: a member is given as a document writes it, got a field; {advice}"
        raise TypeError(msg)


def _is_int_sequence(value: object) -> TypeGuard[Sequence[int]]:
    """Whether `value` is a JSON array of integers.

    JSON booleans decode to `bool`, which is an `int` subclass in Python but
    is not an integer in a metadata document, so booleans are excluded.
    This is a `TypeGuard`, not a `TypeIs`, because it returns False for
    `bytes`, which is a `Sequence[int]`.
    """
    return is_array(value) and all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    )


def dimension_lengths(
    doc: Mapping[object, object] | Mapping[str, object], key: str
) -> tuple[tuple[int, ...] | None, tuple[ValidationProblem, ...]]:
    """The dimension lengths `doc` holds at `key` (`shape`, `chunks`), and every problem with them.

    Dimension lengths are non-negative integers. The lengths are None when
    `doc` has nothing at `key`, or when the value at `key` has a problem.
    """
    if key not in doc:
        return None, ()
    value, found = refine_json(doc[key], (key,))
    if len(found) != 0:
        # An integer too large to write as JSON text, or a value nested
        # too deep.
        return None, found
    if not _is_int_sequence(value):
        return None, (ValidationProblem((key,), "expected an array of integers", "invalid_type"),)
    if any(item < 0 for item in value):
        return None, (ValidationProblem((key,), "expected non-negative integers", "invalid_value"),)
    return tuple(value), ()


def validate_attributes(value: object) -> tuple[ValidationProblem, ...]:
    """Validate an `attributes` value: a mapping with string keys.

    Returns a problem at `("attributes",)` if it is not, else `()`. Shared by the
    v2 and v3 readers. The loc is relative to the document, not to the
    value, since this function is only called with a document's `attributes`
    value.
    """
    return attributes_of(value)[1]


def attributes_of(
    value: object, at: Loc = ()
) -> tuple[dict[str, JSONValue] | None, tuple[ValidationProblem, ...]]:
    """An `attributes` value refined as user data, and every problem `validate_attributes` finds. The value is None when it is not an object with string keys, or holds a value that is not JSON.

    `at` is the location, within the input document, of the document
    that holds the attributes. Nesting depth is counted from the root of
    the input document. The problems are located relative to the
    document that holds the attributes.
    """
    if not is_object(value) or not all(isinstance(k, str) for k in value):
        return None, (
            ValidationProblem(
                ("attributes",), "expected an object with string keys", "invalid_type"
            ),
        )
    attributes: dict[str, JSONValue] = {}
    problems: list[ValidationProblem] = []
    for key, item in value.items():
        if not isinstance(key, str):
            continue
        refined, found = refine_user_data(item, (*at, "attributes", key))
        problems.extend(found)
        attributes[key] = refined
    return (attributes if len(problems) == 0 else None), within(problems, at)


class _HoldsReading(Protocol):
    """A model: an object that holds the reading it was built from."""

    @property
    def reading(self) -> object: ...


def reading_of(model: _HoldsReading) -> object:
    """The reading that `model` holds. Unpickling a reading that held a model rebuilds it with this function."""
    return model.reading


StoreKey = TypeVar("StoreKey", bound=str)
"""The key type of a mapping from store keys to bytes: `str`, or the literal keys of one document."""


def load_store_json(mapping: Mapping[StoreKey, bytes], key: str) -> object:
    """Decode the JSON document stored at `key` in `mapping`.

    Returns `object`, not `Any`: what a store holds is unknown until a
    reader says otherwise, and `Any` would let unchecked values flow
    into typed positions silently. Read the result with a reader.

    Decoding uses Python's `json` module, so `NaN`, `Infinity` and
    `-Infinity` are read as the corresponding floats, which matches how
    zarr-python writes attributes. The document's reader decides where
    such values are allowed. Every ingestion failure here
    surfaces as `MetadataValidationError`: a missing store key is a
    `missing_key` problem, a value that is not `bytes` an `invalid_type`
    problem, and undecodable bytes an `invalid_json` problem, rather than
    leaking `KeyError`, `TypeError` or `json.JSONDecodeError` to
    callers.
    """
    # Look up by a `str` key, whatever narrower key type the mapping
    # declares. A key the mapping does not have is simply absent.
    stored = cast("Mapping[str, bytes]", mapping)
    if key not in stored:
        raise MetadataValidationError(
            [ValidationProblem((key,), "missing store key", "missing_key")]
        )
    # Check the annotation at runtime: `json.loads` also decodes a `str`,
    # and raises `TypeError` for most other types.
    raw = cast("object", stored[key])
    if not isinstance(raw, bytes):
        refused = ValidationProblem(
            (key,), f"expected bytes, got {type(raw).__name__}", "invalid_type"
        )
        raise MetadataValidationError(with_input((refused,), stored))
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise MetadataValidationError(
            [ValidationProblem((key,), f"invalid JSON: {exc}", "invalid_json")]
        ) from exc


def dump_store_json(value: object, *, indent: int | str | None = None) -> bytes:
    """Encode a document that its reader has accepted as JSON bytes.

    A non-finite number is written as Python's `json` module writes it
    (`NaN`, `Infinity`, `-Infinity`), which matches how zarr-python
    writes attributes. The reader keeps such numbers out of every
    other place in the document.
    """
    return json.dumps(value, indent=indent, allow_nan=True).encode("utf-8")


def located_conflicts(
    fields: Iterable[tuple[Loc, ResolvedField[Any]]], conflicts: Sequence[Conflict]
) -> tuple[Conflict, ...]:
    """Locate each of `conflicts` (found against a reading's claims) in the document, as a problem is located. A conflict appears once for each place in `fields` where its name occurs."""
    located: list[Conflict] = []
    placed = list(fields)
    for conflict in conflicts:
        places = [(loc, field) for loc, field in placed if claim_key(field) == conflict.key]
        if len(places) == 0:
            located.append(conflict)
        located.extend(
            dataclasses.replace(conflict, loc=loc, written=field.name) for loc, field in places
        )
    return tuple(located)
