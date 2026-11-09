"""`JSONWriter`, the one object that turns a document into the JSON bytes a store holds."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Final, cast, final

from typing_extensions import TypedDict


class JSONWriterSchema(TypedDict, total=False, closed=True):
    """The settings of a `JSONWriter` as a mapping, each by name and every one optional.

    A setting left out takes its default. `to_key_value` accepts this mapping
    wherever it accepts a `JSONWriter`, so `model.to_key_value({"indent": 2})`
    writes what `model.to_key_value(JSONWriter(indent=2))` writes.
    """

    indent: int | str | None
    """`JSONWriter.indent`."""
    sort_keys: bool
    """`JSONWriter.sort_keys`."""
    ensure_ascii: bool
    """`JSONWriter.ensure_ascii`."""
    trailing_newline: bool
    """`JSONWriter.trailing_newline`."""


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class JSONWriter:
    """How a document is written as JSON bytes. `encode` does the writing.

    The defaults are those of `json.dumps`: everything on one line, keys
    in document order, non-ASCII characters escaped, and no trailing
    newline. zarr-python indents its metadata by its `json_indent`
    setting, which is 2 unless configured otherwise, so it writes what
    `JSONWriter(indent=2)` writes.

    A setting that would change the bytes silently is refused when the
    writer is built: `json.dumps` reads `True` as an indent of 1 and a
    negative indent as 0, and writes a `str` indent verbatim, so one that
    is not whitespace would write bytes no JSON reader parses.

    The class is final: `encode` is the only way the models write JSON, so
    a subclass cannot change how a model is written behind its settings.
    """

    indent: int | str | None = None
    """`None` to write one line, an `int` to nest by that many spaces, or a `str` of JSON whitespace (spaces, tabs, line feeds and carriage returns) to nest by that string."""
    sort_keys: bool = False
    """Write the keys of every object in sorted order instead of document order."""
    ensure_ascii: bool = True
    """Escape every non-ASCII character (`"\\u00b5"`). Off, the bytes are UTF-8 text."""
    trailing_newline: bool = False
    """End the bytes with `"\\n"`, for stores whose files are read by line-oriented tools."""

    def __post_init__(self) -> None:
        indent = cast("object", self.indent)
        if isinstance(indent, bool) or not isinstance(indent, int | str | None):
            msg = f"indent must be an int, a str or None, got {type(indent).__name__}"
            raise TypeError(msg)
        if isinstance(indent, int) and indent < 0:
            msg = f"indent must not be negative, got {indent}"
            raise ValueError(msg)
        if isinstance(indent, str) and indent.strip(" \t\n\r") != "":
            msg = f"indent must hold only JSON whitespace, got {indent!r}"
            raise ValueError(msg)
        for name in ("sort_keys", "ensure_ascii", "trailing_newline"):
            flag = cast("object", getattr(self, name))
            if not isinstance(flag, bool):
                msg = f"{name} must be a bool, got {type(flag).__name__}"
                raise TypeError(msg)

    @classmethod
    def from_dict(cls, data: JSONWriterLike) -> JSONWriter:
        """The writer that `data` describes: `data` itself when it is a `JSONWriter`, else one built from its settings.

        A key that is not a setting raises `TypeError`, and a value the
        setting does not allow raises as the constructor does.
        """
        if isinstance(data, JSONWriter):
            return data
        given = cast("object", data)
        if not isinstance(given, Mapping):
            msg = f"expected a JSONWriter or a mapping of its settings, got {type(given).__name__}"
            raise TypeError(msg)
        settings = cast("Mapping[object, object]", given)
        unknown = sorted(repr(key) for key in settings if key not in _SETTINGS)
        if unknown:
            msg = f"unknown JSONWriter setting(s) {', '.join(unknown)}; the settings are {', '.join(_SETTINGS)}"
            raise TypeError(msg)
        return cls(**cast("JSONWriterSchema", settings))

    def to_dict(self) -> JSONWriterSchema:
        """Every setting of this writer by name, so that `JSONWriter.from_dict(writer.to_dict()) == writer`."""
        return {
            "indent": self.indent,
            "sort_keys": self.sort_keys,
            "ensure_ascii": self.ensure_ascii,
            "trailing_newline": self.trailing_newline,
        }

    def encode(self, value: object) -> bytes:
        """`value`, a document its reader accepted, as UTF-8 JSON bytes spelled by this writer's settings.

        The separators are those of Python's `json` module: `", "` between
        items and `": "` after a key, with the item separator becoming
        `","` when `indent` is set. A non-finite number is written as a
        bare token (`NaN`, `Infinity`, `-Infinity`), as zarr-python writes
        attributes; the readers keep such numbers out of every other place
        in a document. With `ensure_ascii` off, a lone surrogate (a string
        a reader accepts from an escape such as `"\\ud800"`, but which
        UTF-8 cannot encode) is still written as that escape.
        """
        text = json.dumps(
            value,
            indent=self.indent,
            sort_keys=self.sort_keys,
            ensure_ascii=self.ensure_ascii,
            allow_nan=True,
        )
        if self.trailing_newline:
            text += "\n"
        # `json.dumps` writes non-ASCII characters only inside strings, and the
        # only ones UTF-8 cannot encode are lone surrogates, so the replacement
        # writes each as the JSON escape `ensure_ascii` would have written.
        return text.encode("utf-8", "backslashreplace")


JSONWriterLike = JSONWriter | JSONWriterSchema
"""What `to_key_value` accepts as its writer: a `JSONWriter`, or a `JSONWriterSchema` mapping of the settings to change."""

_SETTINGS: Final = tuple(each.name for each in fields(JSONWriter))
"""The name of each setting, in declaration order."""

DEFAULT_JSON_WRITER: Final = JSONWriter()
"""The writer `to_key_value` uses when given none: the spelling of `json.dumps` with its own defaults."""


__all__ = ["DEFAULT_JSON_WRITER", "JSONWriter", "JSONWriterLike", "JSONWriterSchema"]
