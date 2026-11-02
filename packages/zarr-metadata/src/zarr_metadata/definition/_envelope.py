"""The default field envelope: how a field is written as a name or as an object with `name`, `configuration` and `must_understand`, and which names are valid extension names.

This is the envelope Zarr v3 uses. `Definition` uses it by default, and a
kind of another format (the v2 kinds) overrides each method that uses it.
"""

import functools
import re
from typing import Final

from zarr_metadata._json import ValidationProblem, is_object, shown, shown_key

ENVELOPE_KEYS: frozenset[str] = frozenset({"name", "configuration", "must_understand"})
"""The keys a metadata field's envelope declares. Any other key is an unknown key."""


_EXTENSION_NAME: Final = re.compile(r"[a-z][a-z0-9_.-]+")
"""A registered extension name: the spec's `^[a-z][a-z0-9-_.]+$`, with the hyphen moved last so no regex engine reads it as a range."""

_URI: Final = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+")
"""A URI, to the extent this module checks names. It matches an RFC 3986 scheme, a colon, and one or more characters from the URI unreserved and reserved sets plus the `%` of percent-encoding (https://www.rfc-editor.org/rfc/rfc3986#section-2). Earlier versions of the spec required names of this form.

The characters are listed explicitly instead of written as `\\S`.
Python's `re` and ECMA-262 (the regex dialect of a JSON Schema
`pattern`) disagree on what `\\S` matches: `\\x1c`-`\\x1f` and `\\x85` are
whitespace in one, and `\\ufeff` is whitespace in the other. With `\\S`, a
schema validator could accept a name that this reader rejects, or the
reverse.
"""

EXTENSION_NAME_SCHEMA_PATTERN: Final = rf"^({_EXTENSION_NAME.pattern}|{_URI.pattern})(?![\s\S])"
r"""The names `well_named` accepts, as a JSON Schema `pattern`. It matches either pattern against the whole string. It uses `(?![\s\S])` instead of `$`, because `$` also matches before a final newline. `_RAW_BYTES_SCHEMA_PATTERN` does the same."""


@functools.lru_cache(maxsize=4096)
def well_named(name: str) -> bool:
    """Whether `name` is a valid extension name under the spec. Results are cached per name, because a document uses few names and a reader checks each one many times.

    A registered name "MUST start with one lower case letter a-z and then
    be followed by only lower case letters a-z, numerals 0-9, underscores,
    dots and dashes", regex `^[a-z][a-z0-9-_.]+$`
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1603-L1606);
    a URI, which earlier versions of the spec required, is "still
    permitted"
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1614-L1618).
    """
    return _EXTENSION_NAME.fullmatch(name) is not None or _URI.fullmatch(name) is not None


def name_problem(name: str, at: tuple[str | int, ...]) -> ValidationProblem | None:
    """The problem to report at `at` when `name` is not a valid extension name under the spec; None when it is valid, as decided by `well_named`."""
    if well_named(name):
        return None
    message = (
        "expected an extension name -- lower-case letters, digits, '-', '_' and '.', "
        f"starting with a letter -- or a URI, got {shown(name)}"
    )
    return ValidationProblem(at, message, "invalid_value")


def envelope_problems(
    value: object, *, allow_must_understand_false: bool
) -> tuple[ValidationProblem, ...]:
    """Every reason `value` is not a valid v3 metadata field envelope. The contents of the configuration are not checked.

    The envelope is the part around the configuration: a `name` that is a
    valid extension name under the spec, a `configuration` that is an
    object with string keys, a boolean `must_understand`, and no other
    keys. `resolve` calls this on a field it has already narrowed to JSON,
    so the configuration is walked only once.
    """
    if isinstance(value, str):
        bad = name_problem(value, ())
        return () if bad is None else (bad,)
    if not is_object(value):
        return (
            ValidationProblem(
                (),
                f"expected a metadata field (string or extension object), got {shown(value)}",
                "invalid_type",
            ),
        )
    field = value
    problems: list[ValidationProblem] = []
    for key in field:
        if not isinstance(key, str):
            problems.append(
                ValidationProblem(
                    (), f"non-string metadata field key {shown_key(key)}", "invalid_type"
                )
            )
        elif key not in ENVELOPE_KEYS:
            problems.append(ValidationProblem((key,), f"unexpected key {key!r}", "unknown_key"))
    if "name" not in field:
        problems.append(ValidationProblem(("name",), "missing required key", "missing_key"))
    elif not isinstance(field["name"], str):
        problems.append(ValidationProblem(("name",), "expected a string name", "invalid_type"))
    elif (bad := name_problem(field["name"], ("name",))) is not None:
        problems.append(bad)
    if "configuration" in field:
        configuration = field["configuration"]
        if not is_object(configuration):
            problems.append(
                ValidationProblem(("configuration",), "expected an object", "invalid_type")
            )
        elif not all(isinstance(k, str) for k in configuration):
            problems.append(
                ValidationProblem(("configuration",), "expected string keys", "invalid_type")
            )
    if "must_understand" in field:
        must_understand = field["must_understand"]
        if not isinstance(must_understand, bool):
            problems.append(
                ValidationProblem(("must_understand",), "expected a boolean", "invalid_type")
            )
        elif not allow_must_understand_false and not must_understand:
            problems.append(
                ValidationProblem(
                    ("must_understand",),
                    "false is not supported at this extension point",
                    "invalid_value",
                )
            )
    return tuple(problems)


__all__ = [
    "ENVELOPE_KEYS",
    "EXTENSION_NAME_SCHEMA_PATTERN",
    "envelope_problems",
    "name_problem",
    "well_named",
]
